# Ray Serve — Multi-Model Endpoints with GPU Autoscaling by Request Latency

Serve **multiple ML models** behind a single HTTP gateway on
[Ray Serve](https://docs.ray.io/en/latest/serve/index.html), give **each model
its own pool of (fractional) GPU replicas**, and **autoscale those GPUs from the
request latency they are actually serving** — not just from queue depth.

```
                         ┌─────────────────────────────────────────────┐
                         │            Ray Serve application             │
   client ──HTTP──▶  Gateway (FastAPI, CPU)                             │
                         │      ├─▶ sentiment      [GPU 0.25 ×N]  autoscaled
                         │      ├─▶ summarization  [GPU 0.5  ×N]  autoscaled
                         │      └─▶ embedding      [GPU 0.25 ×N]  autoscaled
                         └─────────────────────────────────────────────┘
                                        ▲
                    p95 latency │        │ set min_replicas floor (REST)
                                ▼        │
                       Latency Autoscaler (supervisory control loop)
```

Two layers of autoscaling work together:

| Layer | Mechanism | Reacts to | Where |
|-------|-----------|-----------|-------|
| **Built-in** | Ray Serve `autoscaling_config` (`target_ongoing_requests`) | queue depth per replica | `deployments/model_deployment.py` |
| **Custom (this project)** | Supervisory loop that adjusts each deployment's `min_replicas` floor | **measured p95 latency vs. per-model SLO** | `autoscaling/latency_autoscaler.py` |

> **Runs anywhere.** With no GPU (or without `torch`/`transformers`), every model
> transparently falls back to a deterministic **simulated backend** with a
> configurable compute time, so the full system — routing, batching, metrics and
> autoscaling — can be exercised end-to-end on a laptop or in CI. Flip
> `backend: transformers` on a CUDA box to load real HuggingFace models.

---

## Table of contents

1. [Why latency-based autoscaling](#1-why-latency-based-autoscaling)
2. [Architecture](#2-architecture)
3. [Quickstart (CPU / simulated)](#3-quickstart-cpu--simulated)
4. [Step-by-step: how it's built](#4-step-by-step-how-its-built)
5. [Running with real GPU models](#5-running-with-real-gpu-models)
6. [Watching autoscaling react to load](#6-watching-autoscaling-react-to-load)
7. [Configuration reference](#7-configuration-reference)
8. [Production deployment](#8-production-deployment)
9. [Testing & development](#9-testing--development)
10. [Project layout](#10-project-layout)
11. [FAQ / troubleshooting](#11-faq--troubleshooting)

---

## 1. Why latency-based autoscaling

Ray Serve's built-in autoscaler holds a **target number of in-flight requests
per replica**. By [Little's Law](https://en.wikipedia.org/wiki/Little%27s_law)
(`L = λ × W`), holding queue length constant keeps latency stable **as long as
per-request service time doesn't change**. But on GPUs it does change — a longer
prompt, a cold cache, a co-tenant hogging the card, or a bigger batch all raise
per-request time. When that happens, queue-based scaling can dutifully hold the
target ongoing-requests and still blow your latency SLO.

This project adds a thin **supervisory controller** that measures the true
end-to-end **p95 latency** each model is serving and raises/lowers that model's
`min_replicas` floor to defend the SLO:

- p95 climbing toward the SLO → **raise the floor** (add GPU replicas now).
- p95 comfortably below the SLO → **lower the floor** (release GPUs).

It *cooperates with* the built-in autoscaler rather than replacing it: the floor
guarantees enough capacity to meet the SLO, while Serve's queue logic still
handles fine-grained scaling above the floor. See
[`docs/autoscaling.md`](docs/autoscaling.md) for the control theory and tuning.

## 2. Architecture

- **Gateway** (`deployments/ingress.py`) — a CPU-only FastAPI deployment that is
  the single public endpoint. It holds a `DeploymentHandle` to every model and
  routes/​fans-out requests. Autoscaled on request volume.
- **Model deployments** (`deployments/model_deployment.py`) — one reusable body,
  bound per model with its own **fractional-GPU** `ray_actor_options`,
  `autoscaling_config`, and `@serve.batch` dynamic batching. Each replica keeps a
  rolling `LatencyWindow`.
- **Backends** (`models/base.py`) — `SimulatedBackend` (CPU/CI) and
  `TransformersBackend` (real GPU via HuggingFace). Chosen by `build_backend`
  with graceful fallback.
- **Latency autoscaler** (`autoscaling/latency_autoscaler.py`) — a pure
  `decide()` policy plus a `LatencyAutoscaler` control loop that reads p95 from
  the gateway and applies replica-floor changes via the Serve REST API.
- **Settings** (`settings.py`) — one typed config surface (env + YAML) shared by
  everything.

Full write-up: [`docs/architecture.md`](docs/architecture.md).

## 3. Quickstart (CPU / simulated)

```bash
# 1. Install (core deps only — no GPU/torch needed)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # exposes the `rsa-serve` CLI

# 2. Inspect the resolved config
rsa-serve --config config/models.yaml config

# 3. Deploy the app + latency autoscaler locally (blocks)
python scripts/run_local.py
#    …or via the CLI:
#    rsa-serve --config config/models.yaml serve --blocking
```

In another terminal:

```bash
# Single prediction
curl -s -XPOST localhost:8000/predict/sentiment \
  -H 'content-type: application/json' -d '{"text":"I love how easy this is!"}' | jq

# Batch
curl -s -XPOST localhost:8000/batch/embedding \
  -H 'content-type: application/json' -d '{"inputs":["a","b","c"]}' | jq

# What's deployed + rolling latency
curl -s localhost:8000/models | jq
curl -s localhost:8000/metrics/latency | jq
```

## 4. Step-by-step: how it's built

Each step maps to a file so you can read the code alongside the explanation.

1. **Typed configuration** (`settings.py`). One `Settings` object (env vars
   prefixed `RSA_`, overlaid on `config/models.yaml`) defines every model's task,
   GPU fraction, autoscaling bounds and **latency SLO**. Everything downstream
   reads from here, so there's a single source of truth.

2. **Latency accounting** (`autoscaling/metrics.py`). A dependency-free,
   thread-safe `LatencyWindow` records each request's service time and computes
   rolling p50/p95/p99 over a sliding horizon. It's cheap enough to live inside
   every replica and is the signal the custom autoscaler consumes.

3. **Model backends** (`models/base.py`). The `ModelBackend` interface hides
   *how* inference happens. `SimulatedBackend` sleeps for a configurable compute
   time and returns task-shaped payloads (so CI has real latency to scale on);
   `TransformersBackend` runs HuggingFace pipelines on CUDA. `build_backend`
   picks one and **falls back gracefully** if GPU deps are absent.

4. **GPU model deployment** (`deployments/model_deployment.py`). A single
   `ModelDeployment` body bound per model via `serve.deployment(...)` with:
   - `ray_actor_options={"num_gpus": <fraction>}` → **fractional-GPU placement**
     (pack several small models on one card);
   - `autoscaling_config` with `target_ongoing_requests`, `min/max_replicas`,
     and asymmetric up/down delays → **built-in autoscaling**;
   - `@serve.batch` → **dynamic batching** of concurrent requests into one GPU
     forward pass;
   - a per-replica `LatencyWindow` surfaced via `latency_stats()`.

5. **HTTP gateway** (`deployments/ingress.py`). A FastAPI app wrapped with
   `@serve.ingress`, holding a handle per model. It exposes `/predict/{model}`,
   `/batch/{model}`, `/models`, `/metrics/latency`, `/healthz`, and autoscales on
   request volume — so clients see one stable endpoint while each model's GPUs
   scale independently.

6. **Application assembly** (`app.py`). `build_app()` binds every model
   deployment from the registry and injects their handles into the `Gateway`.
   The module-level `app` object is what `serve.run` / a Serve config
   `import_path` consumes.

7. **Latency-based autoscaler** (`autoscaling/latency_autoscaler.py`). The pure
   `decide()` function maps `(p95, slo, current floor, cooldown, samples)` to a
   scaling decision with guard rails (min-samples, cooldown, max cap, never below
   1). The `LatencyAutoscaler` loop reads p95 from the gateway and applies the
   new `min_replicas` floor via `ServeRestApplier` (the Serve REST API) — or
   `LoggingApplier` for dry-runs.

8. **CLI & scripts** (`cli.py`, `scripts/`). `rsa-serve serve` deploys the app
   and runs the controller in a background thread; `rsa-serve autoscale` runs the
   controller standalone (e.g. as its own pod). `scripts/load_test.py` ramps load
   so you can watch p95 rise and replicas scale out.

## 5. Running with real GPU models

On a CUDA machine:

```bash
pip install -r requirements.txt -r requirements-ml.txt
export RSA_BACKEND=transformers
rsa-serve --config config/models.yaml serve --blocking
```

Each replica calls `torch.cuda.is_available()` and loads its HuggingFace model
onto the GPU fraction Ray assigned it. Tune `num_gpus` per model in
`config/models.yaml`: `0.25` packs four small models per card; use `1.0`+ for
large models. Ray's scheduler enforces the fractions and places replicas on
nodes with free GPU capacity.

## 6. Watching autoscaling react to load

```bash
# Terminal 1 — deploy
rsa-serve --config config/models.yaml serve --blocking

# Terminal 2 — ramp concurrency up then down and watch p95 + replicas
python scripts/load_test.py --model sentiment --ramp 5,20,60,20,5 --stage-seconds 30
```

As concurrency rises, `/metrics/latency` shows p95 climbing toward the SLO; the
built-in autoscaler adds replicas on queue depth, and when p95 crosses
`slo × scale_up_ratio` the latency controller raises the `min_replicas` floor.
`ray status` and the Ray dashboard (`http://localhost:8265`) show GPU replicas
appearing and draining. See [`docs/autoscaling.md`](docs/autoscaling.md) for how
to read the signals.

## 7. Configuration reference

All knobs live in [`config/models.yaml`](config/models.yaml) (see
[`settings.py`](src/ray_serve_autoscale/settings.py) for the schema). Common
environment overrides:

| Env var | Effect |
|---------|--------|
| `RSA_CONFIG_FILE` | path to the YAML registry |
| `RSA_BACKEND` | `simulated` (default) or `transformers` |
| `RSA_HTTP_PORT` | gateway port (default 8000) |
| `RSA_AUTOSCALER_ENABLED` | `true`/`false` to toggle the latency controller |

Per-model: `num_gpus`, `num_cpus`, `min_replicas`, `max_replicas`,
`target_ongoing_requests`, `max_ongoing_requests`, `latency_slo_ms`,
`simulated_latency_s`. Autoscaler: `control_interval_s`, `scale_up_ratio`,
`scale_down_ratio`, `scale_step`, `cooldown_s`, `min_samples`.

## 8. Production deployment

Use the declarative Serve config — the reconciled, restartable path:

```bash
ray start --head                         # or a KubeRay RayService
serve deploy config/serve_config.yaml    # deploy the app
serve status                             # watch it converge
rsa-serve autoscale --base-url http://<gateway>:8000 \
                    --dashboard-url http://<head>:8265   # run the controller
```

On Kubernetes, run the app as a KubeRay `RayService` and the latency controller
as a small sidecar/Deployment that talks to the Serve REST API. Details and a
manifest sketch in [`docs/deployment.md`](docs/deployment.md).

## 9. Testing & development

```bash
pip install -r requirements-dev.txt
pip install -e .

make test          # pure-logic unit tests (no cluster needed)
make test-int      # end-to-end test on a local Ray Serve instance
make lint          # ruff
make check         # lint + type-check + tests
```

The unit suite (`tests/test_metrics.py`, `test_autoscaler.py`, `test_models.py`,
`test_settings.py`) needs no GPU or cluster. `tests/test_integration_serve.py`
spins up a real local Serve instance and is skipped automatically if `ray`
isn't installed.

## 10. Project layout

```
config/
  models.yaml            # model registry + autoscaler config
  serve_config.yaml      # declarative Ray Serve deploy config
src/ray_serve_autoscale/
  settings.py            # typed config (env + YAML)
  app.py                 # build_app(): assembles the graph
  cli.py                 # `rsa-serve` entrypoint
  models/base.py         # backends: simulated + transformers
  deployments/
    model_deployment.py  # GPU deployment: placement + autoscale + batch
    ingress.py           # FastAPI gateway / router
  autoscaling/
    metrics.py           # sliding LatencyWindow + percentiles
    latency_autoscaler.py# decide() policy + control loop + appliers
scripts/
  run_local.py           # deploy + smoke test locally
  load_test.py           # async load generator
tests/                   # unit + integration
docs/                    # architecture, autoscaling, deployment
```

## 11. FAQ / troubleshooting

- **No GPU / `torch` not installed?** Nothing to do — the simulated backend runs
  automatically. Set `RSA_BACKEND=transformers` only where CUDA + the `ml`
  extras are present.
- **`FieldDescriptor object has no attribute 'label'` on `serve.run`.** A
  `protobuf` 5.x vs. Ray incompatibility. Pin `protobuf>=3.20,<5.0`.
- **Autoscaler never scales.** It needs `min_samples` requests per window and
  respects `cooldown_s`; drive real load with `scripts/load_test.py`. Use
  `rsa-serve autoscale --dry-run` to see decisions without mutating the cluster.
- **Replicas flap.** Widen the gap between `scale_up_ratio` and
  `scale_down_ratio`, raise `cooldown_s`, or increase `downscale_delay_s`.

---

Licensed under the [MIT License](LICENSE).
