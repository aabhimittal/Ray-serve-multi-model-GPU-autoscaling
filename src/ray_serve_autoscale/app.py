"""Application assembly.

``build_app`` constructs the whole Serve application graph:

    client ──> Gateway (CPU ingress, FastAPI)
                  ├──> sentiment      (GPU deployment, autoscaled)
                  ├──> summarization  (GPU deployment, autoscaled)
                  └──> embedding      (GPU deployment, autoscaled)

It returns the bound ``Gateway`` deployment, which is what ``serve.run`` (or a
``serve.yaml`` ``import_path``) expects. The GPU deployments are created from a
single reusable body (deployments/model_deployment.py) with per-model GPU and
autoscaling options pulled from settings.
"""

from __future__ import annotations

from ray.serve.handle import DeploymentHandle

from ray_serve_autoscale.deployments.ingress import Gateway
from ray_serve_autoscale.deployments.model_deployment import make_deployment
from ray_serve_autoscale.settings import Settings, get_settings


def build_app(settings: Settings | None = None):
    """Build and return the bound ingress deployment for ``serve.run``."""
    settings = settings or get_settings()

    model_handles: dict[str, DeploymentHandle] = {}
    model_meta: dict[str, dict] = {}
    for model in settings.models:
        model_handles[model.name] = make_deployment(model, settings.backend)
        model_meta[model.name] = {
            "task": model.task,
            "hf_model_id": model.hf_model_id,
            "num_gpus": model.num_gpus,
            "min_replicas": model.min_replicas,
            "max_replicas": model.max_replicas,
            "target_ongoing_requests": model.target_ongoing_requests,
            "latency_slo_ms": model.latency_slo_ms,
        }

    return Gateway.bind(model_handles, model_meta)


# Module-level default app so a Serve config file can reference
# ``ray_serve_autoscale.app:app`` as its ``import_path``.
app = build_app()
