"""Generic GPU-backed model deployment.

One of these is instantiated per model in the registry. It combines the
concerns every GPU serving deployment needs:

1. **GPU placement** -- ``ray_actor_options={"num_gpus": ...}`` reserves a
   (possibly fractional) GPU per replica. Ray's scheduler guarantees the actor
   lands on a node with capacity and pins the fraction.
2. **Ray Serve built-in autoscaling** -- ``autoscaling_config`` scales replicas
   between ``min_replicas`` and ``max_replicas`` to hold roughly
   ``target_ongoing_requests`` in flight per replica.
3. **Dynamic request batching** -- ``@serve.batch`` coalesces concurrent calls
   into one padded GPU forward pass, which is where accelerators earn their
   keep.
4. **Latency decomposition** -- every response carries how much of its service
   time was *queueing* (waiting for a batch slot) versus *compute* (the actual
   forward pass). That split is what lets the planner tell "we need more
   replicas" apart from "the model itself got slower", which are the same
   symptom with opposite remedies. See autoscaling/planner.py.
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
        self._queue = LatencyWindow(window_s=30.0)
        self._compute = LatencyWindow(window_s=30.0)
        self._inflight = 0
        self._errors = 0
        self._backend.load()
        self._started = time.time()

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.02)
    async def _infer_batch(self, inputs: list[str]) -> list[Any]:
        """Batched inference. Serve fills the batch from concurrent requests.

        The measured compute time is returned alongside every result so the
        caller can subtract it from end-to-end latency and recover the
        queueing component -- the batch is the only place that boundary is
        actually observable.
        """
        started = time.perf_counter()
        results = self._backend.predict(inputs)
        compute_ms = (time.perf_counter() - started) * 1000.0
        return [(result, compute_ms) for result in results]

    async def __call__(self, text: str) -> dict:
        """Handle a single logical request; timing feeds the latency windows."""
        start = time.perf_counter()
        self._inflight += 1
        try:
            result, compute_ms = await self._infer_batch(text)
        except Exception:
            self._errors += 1
            raise
        finally:
            self._inflight -= 1

        total_ms = (time.perf_counter() - start) * 1000.0
        # Everything that was not the forward pass was waiting: batch-window
        # delay plus contention for the GPU.
        queue_ms = max(total_ms - compute_ms, 0.0)
        self._latency.record(total_ms)
        self._queue.record(queue_ms)
        self._compute.record(compute_ms)

        return {
            "model": self.config.name,
            "task": self.config.task,
            "device": self._backend.device,
            "latency_ms": round(total_ms, 2),
            "queue_ms": round(queue_ms, 2),
            "compute_ms": round(compute_ms, 2),
            "result": result,
        }

    async def stream(self, text: str):
        """Token-by-token streaming, bypassing the batch path.

        Batching optimises throughput for complete responses; streaming
        optimises time-to-first-token. They pull in opposite directions, so a
        streaming request runs unbatched rather than waiting for a batch
        window it would only be held back by.
        """
        start = time.perf_counter()
        self._inflight += 1
        try:
            for chunk in self._backend.predict_stream(text):
                yield chunk
        finally:
            self._inflight -= 1
            self._latency.record((time.perf_counter() - start) * 1000.0)

    async def latency_stats(self) -> dict:
        """Per-replica rolling latency snapshot, decomposed."""
        total = self._latency.snapshot()
        queue = self._queue.snapshot()
        compute = self._compute.snapshot()
        return {
            "model": self.config.name,
            "slo_ms": self.config.latency_slo_ms,
            "queue_ms": round(queue.p95_ms, 2),
            "compute_ms": round(compute.p95_ms, 2),
            "inflight_requests": self._inflight,
            "errors": self._errors,
            **total.as_dict(),
        }

    async def health(self) -> dict:
        return {
            "model": self.config.name,
            "device": self._backend.device,
            "uptime_s": round(time.time() - self._started, 1),
            "inflight_requests": self._inflight,
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
