"""Generic GPU-backed model deployment.

One of these is instantiated per model in the registry. It combines three
concerns that every GPU serving deployment needs:

1. **GPU placement** -- ``ray_actor_options={"num_gpus": ...}`` reserves a
   (possibly fractional) GPU per replica. Ray's scheduler guarantees the actor
   lands on a node with capacity and pins the fraction.
2. **Ray Serve built-in autoscaling** -- ``autoscaling_config`` scales replicas
   between ``min_replicas`` and ``max_replicas`` to hold roughly
   ``target_ongoing_requests`` in flight per replica. Because queue depth and
   latency are linked by Little's Law, this already provides a first line of
   latency control.
3. **Dynamic request batching** -- ``@serve.batch`` coalesces concurrent calls
   into one padded GPU forward pass, which is where accelerators earn their
   keep.

On top of Serve's own metrics we keep a :class:`LatencyWindow` per replica so
the *custom* latency autoscaler (autoscaling/latency_autoscaler.py) can read
true end-to-end p95 and override the built-in targets when an SLO is at risk.
"""

from __future__ import annotations

import time
from typing import Any

from ray import serve

from ray_serve_autoscale.autoscaling.metrics import LatencyWindow
from ray_serve_autoscale.models.base import build_backend
from ray_serve_autoscale.settings import ModelConfig


class ModelDeployment:
    """Reusable deployment body. Bound to concrete options in ``app.build_app``.

    We deliberately do *not* decorate this class with ``@serve.deployment`` at
    definition time. Instead ``build_app`` calls ``serve.deployment(...)`` with
    per-model ``autoscaling_config`` and ``ray_actor_options`` so a single code
    path serves every model with model-specific GPU + scaling settings.
    """

    def __init__(self, config: ModelConfig, backend_kind: str) -> None:
        self.config = config
        self._backend = build_backend(config, backend_kind)
        self._latency = LatencyWindow(window_s=30.0)
        self._backend.load()
        self._started = time.time()

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.02)
    async def _infer_batch(self, inputs: list[str]) -> list[Any]:
        """Batched inference. Serve fills the batch from concurrent requests."""
        return self._backend.predict(inputs)

    async def __call__(self, text: str) -> dict:
        """Handle a single logical request; timing feeds the latency window."""
        start = time.perf_counter()
        result = await self._infer_batch(text)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._latency.record(elapsed_ms)
        return {
            "model": self.config.name,
            "task": self.config.task,
            "device": self._backend.device,
            "latency_ms": round(elapsed_ms, 2),
            "result": result,
        }

    async def latency_stats(self) -> dict:
        """Expose the rolling latency snapshot to the custom autoscaler."""
        snap = self._latency.snapshot()
        return {
            "model": self.config.name,
            "slo_ms": self.config.latency_slo_ms,
            **snap.as_dict(),
        }

    async def health(self) -> dict:
        return {
            "model": self.config.name,
            "device": self._backend.device,
            "uptime_s": round(time.time() - self._started, 1),
        }


def make_deployment(config: ModelConfig, backend_kind: str):
    """Return a bound Serve deployment for ``config``.

    The returned object is ``ModelDeployment`` wrapped with Serve options:
    fractional-GPU actor placement plus built-in latency-oriented autoscaling.
    """
    autoscaling_config = {
        "min_replicas": config.min_replicas,
        "max_replicas": config.max_replicas,
        "target_ongoing_requests": config.target_ongoing_requests,
        # Scale-up quickly under load, scale-down slowly to avoid thrashing GPUs.
        "upscale_delay_s": 10.0,
        "downscale_delay_s": 120.0,
    }
    # The simulated backend does no GPU compute, so it must NOT reserve GPU
    # resources -- otherwise replicas are unschedulable on CPU-only / CI nodes
    # and the deployment never becomes healthy. Real GPU placement kicks in only
    # for the transformers backend.
    num_gpus = config.num_gpus if backend_kind == "transformers" else 0.0
    deployment = serve.deployment(
        ModelDeployment,
        name=config.name,
        autoscaling_config=autoscaling_config,
        max_ongoing_requests=config.max_ongoing_requests,
        ray_actor_options={"num_gpus": num_gpus, "num_cpus": config.num_cpus},
        # Graceful, health-checked replicas so autoscaling never drops requests.
        health_check_period_s=10.0,
        graceful_shutdown_timeout_s=20.0,
    )
    return deployment.bind(config, backend_kind)
