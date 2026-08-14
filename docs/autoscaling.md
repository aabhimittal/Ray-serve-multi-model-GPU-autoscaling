# Latency-based GPU autoscaling

## The problem with queue-only scaling

Ray Serve's built-in autoscaler keeps a target number of **ongoing requests per
replica**. Little's Law (`L = λ · W`) says that if you hold queue length `L`
constant, latency `W` is stable **only while service time is constant**. On GPUs
service time is *not* constant — it moves with prompt length, batch size, cache
warmth, and contention from co-tenant models on the same card. When service time
rises, holding `target_ongoing_requests` keeps queues short but lets **tail
latency drift past your SLO**.

## The supervisory controller

`LatencyAutoscaler` measures the real thing — p95 end-to-end latency — and
adjusts the deployment's `min_replicas` floor to defend a per-model SLO.

### The policy (`decide`)

`decide(state, cfg)` is a pure function evaluated per model each control tick.
Guard rails apply in order:

1. **Min samples.** If fewer than `min_samples` requests were observed in the
   window, **hold** — don't react to noise from a near-idle endpoint.
2. **Cooldown.** If the last scaling action was less than `cooldown_s` ago,
   **hold** — prevents thrashing.
3. **Scale out.** If `p95 ≥ slo × scale_up_ratio`, raise the floor by
   `scale_step` (capped at `max_replicas`).
4. **Scale in.** If `p95 ≤ slo × scale_down_ratio`, lower the floor by
   `scale_step` (never below 1).
5. Otherwise **hold** — p95 is inside the healthy band.

```
                 scale in            hold            scale out
  p95:  0 ─────────────────┼──────────────────┼──────────────────▶ slo
                     slo×scale_down       slo×scale_up
                        (0.4·slo)            (0.9·slo)
```

The **dead band** between `scale_down_ratio` and `scale_up_ratio` is what keeps
the system from oscillating: latency has to genuinely recover before capacity is
released.

### Applying decisions

- `LoggingApplier` — records the decision, mutates nothing. Default for tests,
  CI, and `--dry-run`. Safe to run anywhere.
- `ServeRestApplier` — GETs the live Serve config from the dashboard agent,
  patches the target deployment's `autoscaling_config.min_replicas`, and PUTs it
  back. Serve applies the change in place without restarting healthy replicas.

## Tuning guide

| Symptom | Lever |
|---------|-------|
| Replicas flap up/down | widen the dead band (lower `scale_down_ratio` / raise `scale_up_ratio`), raise `cooldown_s`, raise `downscale_delay_s` |
| Scales out too late (SLO breaches before capacity arrives) | lower `scale_up_ratio` (e.g. 0.8), lower `upscale_delay_s`, shorten `control_interval_s` |
| Holds GPUs too long after load drops | raise `scale_down_ratio`, lower `downscale_delay_s` |
| Reacts to noise on a quiet endpoint | raise `min_samples`, lengthen the `LatencyWindow` |
| Cost too high | lower `max_replicas`, raise `target_ongoing_requests` (accept a touch more latency) |

## Interaction with the built-in autoscaler

The two controllers are **complementary**, not competing:

- The built-in autoscaler owns fine-grained, fast, queue-driven scaling between
  the current floor and `max_replicas`.
- The latency controller owns the **floor** — a slower, SLO-aware lower bound.

Because the latency controller only ever *raises or lowers `min_replicas`*, it
can never drive replicas below what queue-based scaling wants; it can only
guarantee *more* capacity when latency demands it. If you prefer a single
controller, disable the built-in one by pinning `min_replicas == max_replicas`
per model and letting the latency loop own the replica count — but the layered
default is more robust.

## Observability

- `GET /metrics/latency` — rolling p50/p95/p99/max/rps per model + the SLO.
- Ray dashboard (`:8265`) → Serve tab — replica counts, ongoing requests, and
  the built-in autoscaler's decisions.
- `rsa-serve autoscale --dry-run` — prints the latency controller's decisions
  and reasons each tick without changing the cluster.

---

## Beyond a single model

Everything above scales one deployment from its own p95. On a shared cluster
that is not enough: GPUs are finite, the sum of locally-correct decisions is
routinely infeasible, and replicas take 30-90s to become capacity.

[`docs/fleet-control.md`](fleet-control.md) covers the fleet controller, which
adds cold-start-aware prediction, queue-vs-compute decomposition,
priority-class GPU arbitration and a budget ceiling — plus the guards
(staleness, flap damping, breach confirmation, emergency bypass) that make a
control loop safe to leave running unattended.
