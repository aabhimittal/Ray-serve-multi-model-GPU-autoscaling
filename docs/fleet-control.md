# Fleet control: predictive, decomposed, arbitrated

The per-model latency autoscaler ([`docs/autoscaling.md`](autoscaling.md)) scales
one deployment from its own p95. It is the right default on an uncontended
cluster. This document covers the **fleet controller**, which closes the loop
over every model at once and adds the four things a shared GPU cluster needs.

```
read  ─▶ condition ─▶ plan ─▶ arbitrate ─▶ apply
         signals.py   planner.py  arbiter.py   controller.py
```

Every stage is a pure function of its input. That is not stylistic: it is what
lets the whole controller be driven through clock jumps, frozen metrics, GPU
exhaustion and budget starvation deterministically in milliseconds
(`tests/test_industrial_edge_cases.py`), instead of hoping those paths behave in
production.

---

## 1. Predictive, cold-start-aware sizing

A GPU replica is not capacity when you ask for it. Weights load, a CUDA context
initialises, and 30–90s later it serves its first request. A reactive controller
that orders replicas *when p95 crosses the SLO* is therefore guaranteed to breach
the SLO for the entire cold-start window — it is structurally always late.

The planner instead estimates the latency **trend** (least-squares slope, in ms
per second) and projects forward across the model's own `cold_start_s`:

```
projected_p95 = smoothed_p95 + trend_ms_per_s × cold_start_s
```

and sizes capacity for `projected_p95`. Concretely: 700ms observed, climbing
10ms/s, with a 60s cold start, projects to 1300ms — so a 1000ms SLO is already
lost unless replicas are ordered *now*.

Sizing itself is a **ratio controller** rather than ±1 steps:

```
desired = ceil(serving_replicas × observed / target)
```

A fleet 4× over its SLO gets 4× the capacity in one decision instead of crawling
there over four control intervals while burning error budget. `max_scale_out_step`
bounds the blast radius of a single decision.

> The ratio is taken against **serving** replicas, not all replicas. Capacity
> that is still cold-starting has not influenced the measurement yet; folding it
> in would re-order the same replicas every tick and overshoot the whole
> cluster. That is the classic autoscaler thundering herd, and
> `test_cold_start_storm_does_not_re_order_the_same_capacity` pins the fix.

## 2. Queue vs. compute decomposition

"p95 is 4000ms" is not actionable, because two different faults produce it and
they have opposite remedies. The gateway measures both parts of every request —
time waiting for a batch slot, and time inside the forward pass — so the planner
can tell them apart:

| Bound | Symptom | Correct response |
|-------|---------|------------------|
| **queue** | waiting dominates (`queue_ms / total > 0.3`) | scale out — replicas directly cut p95 |
| **compute** | forward pass dominates | scale-out capped at +1; report the real cause |

Adding replicas to a compute-bound model buys throughput but barely moves
per-request tail latency: you spend GPUs on a problem replicas cannot fix. When
the planner sees this it caps the step and says so:

```
compute-bound (queue 3% of 4000ms); p95 4000ms >= 900ms -> capped scale-out to 3
```

That line is the difference between a 20-minute investigation and an 8-GPU bill.

## 3. Cross-model GPU arbitration

Per-model planning asks "how much does *this* model want?". On a shared cluster
the sum of locally-correct answers routinely exceeds what exists — and Ray's
response is to leave the surplus replicas `PENDING` indefinitely. *Which* model
ends up starved is then decided by scheduling accident. A nightly batch
embedding job can quietly displace the model behind a customer-facing product.

The arbiter makes that allocation an explicit policy decision:

1. **Floors first, by priority.** `min_replicas` is an SLO commitment; those are
   honoured in priority order before anyone gets discretionary capacity.
2. **Surplus by strict priority class, urgency within class.** Priority
   dominates, so priority inversion is structurally impossible — a lower-priority
   model can never preempt a higher one however far past its SLO it is. Urgency
   (how far past the SLO a model already is) orders models *inside* a class.
3. **Everything reported.** Unmet demand, starvation and utilisation are surfaced
   in `TickResult.notes`, not silently absorbed.

```yaml
cluster:
  total_gpus: 8.0            # what the cluster can actually schedule
  gpu_headroom_fraction: 0.1 # never pack to 100%
models:
  - name: sentiment
    priority: 300            # customer-facing: served first
  - name: embedding
    priority: 100            # batch-ish: first to yield
```

Setting `total_gpus: 0` disables GPU arbitration entirely (CPU dev boxes and CI),
rather than starving the fleet because capacity was never declared.

## 4. Budget governance

GPU spend is a hard operational constraint, and an autoscaler with no cost model
will happily scale into a five-figure surprise. `max_hourly_budget_usd` is a
ceiling the arbiter enforces by trimming from the least important end:

1. discretionary capacity (above floors) is cut first, lowest priority first;
2. only if the cap still cannot be met does it dip **below floors** — an explicit
   SLO sacrifice, reported as starvation rather than done quietly.

## 5. Robustness: what the controller refuses to do

These are the guards that decide whether a control loop is safe to leave running
unattended. Each one exists because the alternative is an outage.

| Guard | Failure it prevents |
|-------|--------------------|
| **Staleness detection** | A frozen exporter serves its last payload forever. Counts stop advancing while values look plausible — the most dangerous failure mode for a closed loop, because it looks like health. |
| **`finite_or_none` ingestion** | A NaN latency coerced to `0.0` reads as *perfectly healthy* and can trigger scale-in during an outage. Bad samples are dropped, never zeroed. |
| **Breach confirmation** | Smoothing alone cannot absorb an arbitrarily large one-off spike (20% of a 40× outlier already clears the trigger). A breach must persist `scale_out_confirm_ticks` before capacity is ordered. |
| **Flap damping** | Every oscillation cycle pays a cold start and evicts a warm CUDA context. Repeated direction changes widen the dead band so only a decisive move acts. |
| **Clock-regression handling** | NTP steps and container suspend/resume move clocks backwards; an unguarded regression inverts the trend fit. |
| **In-flight drain protection** | Scaling in below `ceil(inflight / max_ongoing_requests)` strands requests already accepted. |
| **Emergency bypass** | With zero healthy replicas there is no traffic, so no samples — every ordinary guard says "hold" and the model stays down forever. Outage recovery bypasses every damper, including the cooldown. |
| **Reader failure → hold** | Holding is always safer than acting on data that could not be read. |

## 6. Operating it

```bash
# Explain one decision for every model and exit. Reads live metrics,
# applies nothing -- the fastest answer to "why did it (not) scale?"
rsa-serve --config config/models.yaml plan

model            now  want  grant     bound  reason
------------------------------------------------------------------
sentiment          2     4      4     queue  p95 361ms >= 135ms (x2.67) -> scale out to 4; trend +4.2ms/s
summarization      1     1      1   compute  compute-bound (queue 8% of 900ms); capped scale-out to 2
embedding          3     2      2     queue  p95 41ms <= 48ms -> scale in to 2

GPU 1.50/2.00 (75%)   cost $3.75/h of $40.00/h
```

```bash
rsa-serve fleet --dry-run     # run the loop, log decisions, change nothing
rsa-serve fleet               # run it for real
```

Tuning follows the same levers as the per-model controller
([`docs/autoscaling.md`](autoscaling.md#tuning-guide)), plus:

| Symptom | Lever |
|---------|-------|
| Reacts too slowly to genuine ramps | lower `scale_out_confirm_ticks`, raise `ewma_alpha`, shorten `control_interval_s` |
| Orders GPUs for spikes | raise `scale_out_confirm_ticks`, lower `ewma_alpha` |
| Predictive scaling overshoots | shorten `trend_window_s`, lower `cold_start_s` to the measured value |
| Wrong model starved under contention | fix `priority` — it is a policy choice, not a tuning knob |
| Spend too high | lower `max_hourly_budget_usd`; the arbiter trims from the least important end |
