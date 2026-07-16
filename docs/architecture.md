# Architecture

This document explains the moving parts and the request/control paths.

## Component diagram

```
                            ┌──────────────────────────────────────────────┐
                            │              Ray cluster / node(s)             │
                            │                                                │
  HTTP client ──▶ :8000 ──▶ │  Gateway deployment (FastAPI, @serve.ingress)  │
                            │   • CPU-only, autoscaled on request volume     │
                            │   • holds a DeploymentHandle per model         │
                            │        │              │              │         │
                            │        ▼              ▼              ▼         │
                            │  sentiment      summarization    embedding     │
                            │  (GPU 0.25)      (GPU 0.5)       (GPU 0.25)    │
                            │  min..max         min..max        min..max     │
                            │  @serve.batch     @serve.batch    @serve.batch │
                            │  LatencyWindow    LatencyWindow   LatencyWindow│
                            └──────────────────────────────────────────────┘
                                         ▲                    │
             GET /metrics/latency (p95)  │                    │  Serve REST API
                                         │                    ▼  PUT min_replicas
                            ┌──────────────────────────────────────────────┐
                            │        LatencyAutoscaler control loop          │
                            │   read p95 ─▶ decide() ─▶ apply floor          │
                            └──────────────────────────────────────────────┘
```

## Request path (data plane)

1. Client `POST /predict/{model}` hits the **Gateway** (FastAPI over
   `@serve.ingress`).
2. The Gateway looks up the model's `DeploymentHandle` and calls
   `handle.remote(text)`.
3. The model deployment's `__call__` submits the input to `@serve.batch`, which
   coalesces concurrent inputs into one padded batch and runs the backend's
   `predict()` (a GPU forward pass under `transformers`, a timed sleep under
   `simulated`).
4. The measured end-to-end time is recorded into that replica's
   `LatencyWindow`, and the response (with `latency_ms`) is returned up the
   chain.

## Control path (autoscaling)

Two independent controllers act on the same replica pools:

### Built-in (Ray Serve)
Each model deployment declares an `autoscaling_config`. Serve samples ongoing
requests per replica every `metrics_interval_s` and scales toward
`target_ongoing_requests`, bounded by `min_replicas`/`max_replicas`, with
asymmetric `upscale_delay_s` (fast) and `downscale_delay_s` (slow). This is
queue-driven.

### Custom (LatencyAutoscaler)
Every `control_interval_s` it:
1. **reads** each model's rolling p95 from `GET /metrics/latency`;
2. **decides** via the pure `decide()` policy whether p95 is pushing the SLO;
3. **applies** a new `min_replicas` floor through the Serve REST API.

Raising the floor forces Serve to add replicas immediately (defending the SLO);
lowering it lets Serve's downscale logic reclaim GPUs when latency is healthy.
The two controllers compose: the floor is a latency-driven *lower bound*, and
queue-based scaling still operates above it.

## Why the layers are separate

- **Backend vs. deployment.** Backends are plain classes with no Ray imports, so
  the inference logic is unit-testable and swappable (simulated ↔ transformers)
  without a cluster.
- **Policy vs. loop.** `decide()` is a pure function of observed state; the
  control loop only wires a *reader* to an *applier*. That makes the scaling
  policy exhaustively testable and the side-effecting parts injectable.
- **Gateway vs. models.** Clients depend on one stable endpoint; each model's
  GPU pool scales on its own SLO without the client knowing.

## Fractional GPUs

`ray_actor_options={"num_gpus": 0.25}` tells Ray to reserve a quarter of a GPU
per replica. Ray's scheduler tracks fractional GPU capacity per node and packs
replicas accordingly, so several small models (or several replicas of one) share
a single physical card. Set `num_gpus: 1.0` (or higher, with placement groups)
for models that need a whole device.
