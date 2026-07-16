# Deployment

## Local (single machine)

```bash
pip install -r requirements.txt && pip install -e .
rsa-serve --config config/models.yaml serve --blocking
```

`run_local.py` / `rsa-serve serve` start a local Ray, deploy the app, and run
the latency controller in a background thread.

## Declarative Serve config (recommended)

The reconciled, restartable path uses `config/serve_config.yaml`:

```bash
ray start --head                          # start (or attach to) a cluster
serve deploy config/serve_config.yaml     # deploy / update the application
serve status                              # watch it converge
serve shutdown                            # tear down
```

`num_replicas` is intentionally omitted per model so `autoscaling_config`
governs the count. Run the latency controller as a separate process so it can be
restarted/scaled independently of the app:

```bash
rsa-serve autoscale \
  --base-url http://<gateway-host>:8000 \
  --dashboard-url http://<ray-head>:8265
```

## Kubernetes (KubeRay)

Run the Serve app as a KubeRay **`RayService`** (it reconciles
`serveConfigV2` — paste the contents of `config/serve_config.yaml`), and run the
latency controller as a small sidecar/Deployment that reaches the Serve REST API
on the head node's dashboard agent (`:8265`).

Sketch:

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: multi-model-gateway
spec:
  serveConfigV2: |
    # paste config/serve_config.yaml here
  rayClusterConfig:
    headGroupSpec: { ... }
    workerGroupSpecs:
      - groupName: gpu-workers
        rayStartParams: { num-gpus: "1" }
        template:
          spec:
            containers:
              - name: ray-worker
                resources:
                  limits:
                    nvidia.com/gpu: 1
---
apiVersion: apps/v1
kind: Deployment
metadata: { name: latency-autoscaler }
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: autoscaler
          image: your-registry/ray-serve-autoscale:latest
          command: ["rsa-serve", "autoscale",
                    "--base-url", "http://multi-model-gateway-serve-svc:8000",
                    "--dashboard-url", "http://multi-model-gateway-head-svc:8265"]
```

### GPU node pools & fractional GPUs

- Request whole GPUs on worker pods (`nvidia.com/gpu: 1`); Ray then subdivides
  each physical GPU according to each deployment's `num_gpus` fraction.
- Size `max_replicas` against the total GPU fractions available so autoscaling
  can't request more GPU than the cluster has. Combine with the cluster
  autoscaler / KubeRay worker autoscaling to add GPU nodes under sustained load.

## Environment variables

| Var | Purpose |
|-----|---------|
| `RSA_CONFIG_FILE` | path to `models.yaml` |
| `RSA_BACKEND` | `simulated` or `transformers` |
| `RSA_RAY_ADDRESS` | attach to an existing Ray cluster |
| `RSA_HTTP_PORT` | gateway port |
| `RSA_AUTOSCALER_ENABLED` | toggle the latency controller |

## Health & readiness

- `GET /healthz` on the gateway for liveness/readiness probes.
- Each model deployment sets `health_check_period_s`; Serve replaces unhealthy
  replicas automatically.
- `graceful_shutdown_timeout_s` lets in-flight requests drain during scale-in so
  autoscaling never drops traffic.
