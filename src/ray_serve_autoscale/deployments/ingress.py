"""HTTP ingress: a FastAPI app fronting every model deployment.

The ingress is itself a lightweight Serve deployment (CPU-only, autoscaled on
request volume). It owns a handle to each GPU model deployment and routes
requests to them, so clients see a single stable endpoint while each model
scales its GPU replicas independently underneath.

Routes
------
POST /predict/{model}      run inference on one model
POST /batch/{model}        run inference on a list of inputs
GET  /models               list available models + their config
GET  /metrics/latency      rolling latency snapshot per model (feeds dashboards
                           and the custom autoscaler)
GET  /healthz              liveness/readiness
"""

# NOTE: intentionally NOT using `from __future__ import annotations` here.
# FastAPI introspects these route methods at registration time (under
# @serve.ingress); stringized annotations would make it misread the Pydantic
# body parameter as a query parameter. Real annotation objects keep routing
# correct. The `dict[...]`/`list[...]` generics below are runtime-safe on 3.9+.

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ray import serve
from ray.serve.handle import DeploymentHandle

api = FastAPI(title="Ray Serve Multi-Model Gateway", version="0.1.0")


class PredictRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    inputs: list[str]


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_ongoing_requests": 32,
    },
    ray_actor_options={"num_cpus": 1},
)
@serve.ingress(api)
class Gateway:
    """Router deployment. ``model_handles`` is injected by ``build_app``."""

    def __init__(self, model_handles: dict[str, DeploymentHandle], model_meta: dict[str, dict]):
        self._handles = model_handles
        self._meta = model_meta

    def _handle(self, model: str) -> DeploymentHandle:
        handle = self._handles.get(model)
        if handle is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown model '{model}'; available: {sorted(self._handles)}",
            )
        return handle

    @api.get("/models")
    async def list_models(self) -> dict:
        return {"models": self._meta}

    @api.get("/healthz")
    async def healthz(self) -> dict:
        return {"status": "ok", "models": sorted(self._handles)}

    @api.post("/predict/{model}")
    async def predict(self, model: str, req: PredictRequest) -> dict:
        handle = self._handle(model)
        return await handle.remote(req.text)

    @api.post("/batch/{model}")
    async def batch(self, model: str, req: BatchRequest) -> dict:
        handle = self._handle(model)
        # Fire concurrently; Serve's @serve.batch coalesces them downstream.
        results = [handle.remote(text) for text in req.inputs]
        gathered: list[Any] = [await r for r in results]
        return {"model": model, "count": len(gathered), "results": gathered}

    @api.get("/metrics/latency")
    async def latency(self) -> dict:
        stats = {}
        for name, handle in self._handles.items():
            stats[name] = await handle.latency_stats.remote()
        return {"latency": stats}
