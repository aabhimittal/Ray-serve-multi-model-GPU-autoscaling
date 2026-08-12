"""HTTP ingress: a FastAPI app fronting every model deployment.

The ingress is itself a lightweight Serve deployment (CPU-only, autoscaled on
request volume). It owns a handle to each GPU model deployment and routes
requests to them, so clients see a single stable endpoint while each model
scales its GPU replicas independently underneath.

It is also the natural **observation point** for the control loop. Individual
replicas only see their own traffic, and a handle call reaches exactly one of
them -- so per-replica stats are a biased sample of the fleet. The gateway sees
*every* request to *every* replica, which makes its rolling windows the honest
fleet-wide signal the autoscaler consumes.

Routes
------
POST /predict/{model}      run inference on one model
POST /batch/{model}        run inference on a list of inputs
POST /stream/{model}       server-sent events, token by token
GET  /models               list available models + their config
GET  /metrics/latency      decomposed fleet latency + replica state (control feed)
GET  /metrics              Prometheus exposition
GET  /admission            load-shedding and circuit-breaker state
GET  /healthz              liveness/readiness
"""

# NOTE: intentionally NOT using `from __future__ import annotations` here.
# FastAPI introspects these route methods at registration time (under
# @serve.ingress); stringized annotations would make it misread the Pydantic
# body parameter as a query parameter. Real annotation objects keep routing
# correct. The `dict[...]`/`list[...]` generics below are runtime-safe on 3.9+.

import json
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ray import serve
from ray.serve.handle import DeploymentHandle

from ray_serve_autoscale.autoscaling.metrics import LatencyWindow
from ray_serve_autoscale.observability.prometheus import render_prometheus
from ray_serve_autoscale.serving.admission import AdmissionController
from ray_serve_autoscale.settings import AdmissionConfig

api = FastAPI(title="Ray Serve Multi-Model Gateway", version="0.2.0")

# How long replica-state read from the Serve controller stays cached. Querying
# it per request would put control-plane latency on the data path.
_REPLICA_CACHE_TTL_S = 2.0


class PredictRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    inputs: list[str]


class _ModelStats:
    """Fleet-wide rolling windows for one model, as seen by the gateway."""

    def __init__(self) -> None:
        self.total = LatencyWindow(window_s=30.0)
        self.queue = LatencyWindow(window_s=30.0)
        self.compute = LatencyWindow(window_s=30.0)
        self.inflight = 0
        self.errors = 0
        self.requests = 0

    def record(self, total_ms: float, queue_ms: float, compute_ms: float) -> None:
        self.total.record(total_ms)
        self.queue.record(queue_ms)
        self.compute.record(compute_ms)
        self.requests += 1


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

    def __init__(
        self,
        model_handles: dict[str, DeploymentHandle],
        model_meta: dict[str, dict],
        admission_config: Optional[dict] = None,
    ):
        self._handles = model_handles
        self._meta = model_meta
        self._stats = {name: _ModelStats() for name in model_handles}

        cfg = AdmissionConfig(**(admission_config or {}))
        self._admission_cfg = cfg
        self._admission = {
            name: AdmissionController(
                name,
                slo_ms=meta.get("latency_slo_ms", 0.0),
                cfg=cfg,
                shed_at_slo_fraction=meta.get("shed_at_slo_fraction", 1.5),
            )
            for name, meta in model_meta.items()
        }
        self._replica_cache: dict = {}
        self._replica_cache_at = 0.0

    # -- helpers ----------------------------------------------------------

    def _handle(self, model: str) -> DeploymentHandle:
        handle = self._handles.get(model)
        if handle is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown model '{model}'; available: {sorted(self._handles)}",
            )
        return handle

    def _replica_state(self) -> dict:
        """Ready / starting replica counts per deployment, from Serve itself.

        This is what makes cold-start accounting real: ``STARTING`` replicas
        are capacity already on the way, and the planner must not re-order
        them every tick. Cached, and degraded gracefully -- the controller has
        a documented fallback when replica state is unavailable.
        """
        now = time.time()
        if now - self._replica_cache_at < _REPLICA_CACHE_TTL_S and self._replica_cache:
            return self._replica_cache

        state: dict = {}
        try:
            status = serve.status()
            for app in status.applications.values():
                for dep_name, dep in app.deployments.items():
                    counts = dict(getattr(dep, "replica_states", {}) or {})
                    state[dep_name] = {
                        "ready_replicas": int(counts.get("RUNNING", 0)),
                        "pending_replicas": int(
                            counts.get("STARTING", 0) + counts.get("PENDING_ALLOCATION", 0)
                        ),
                        "unhealthy_replicas": int(counts.get("UNHEALTHY", 0)),
                    }
        except Exception:
            # Serve status is unavailable in some contexts/versions; the
            # controller falls back to its configured floor.
            state = {}

        self._replica_cache = state
        self._replica_cache_at = now
        return state

    async def _dispatch(self, model: str, text: str) -> dict:
        """Admission-check, dispatch, and record one request."""
        stats = self._stats[model]
        controller = self._admission[model]
        replicas = self._replica_state().get(model, {})

        decision = controller.check(
            queue_depth=stats.inflight,
            replicas=max(replicas.get("ready_replicas", 0) or 1, 1),
            service_time_ms=stats.compute.snapshot().p50_ms or 1.0,
            max_concurrent_per_replica=self._meta[model].get("target_ongoing_requests", 1),
        )
        if not decision.admitted:
            # Fail fast and honestly: a caller that knows to retry in 2s is far
            # better served than one left holding a doomed request.
            raise HTTPException(
                status_code=503,
                detail=decision.reason,
                headers={"Retry-After": str(max(int(decision.retry_after_s), 1))},
            )

        stats.inflight += 1
        start = time.perf_counter()
        try:
            result = await self._handle(model).remote(text)
        except Exception:
            stats.errors += 1
            controller.record_result(False)
            raise
        finally:
            stats.inflight -= 1

        total_ms = (time.perf_counter() - start) * 1000.0
        stats.record(
            total_ms,
            float(result.get("queue_ms", 0.0)),
            float(result.get("compute_ms", 0.0)),
        )
        controller.record_result(True)
        return result

    # -- routes -----------------------------------------------------------

    @api.get("/models")
    async def list_models(self) -> dict:
        return {"models": self._meta}

    @api.get("/healthz")
    async def healthz(self) -> dict:
        return {"status": "ok", "models": sorted(self._handles)}

    @api.post("/predict/{model}")
    async def predict(self, model: str, req: PredictRequest) -> dict:
        self._handle(model)
        return await self._dispatch(model, req.text)

    @api.post("/batch/{model}")
    async def batch(self, model: str, req: BatchRequest) -> dict:
        handle = self._handle(model)
        # Fire concurrently; Serve's @serve.batch coalesces them downstream.
        responses = [handle.remote(text) for text in req.inputs]
        gathered: list[Any] = [await r for r in responses]
        stats = self._stats[model]
        for item in gathered:
            stats.record(
                float(item.get("latency_ms", 0.0)),
                float(item.get("queue_ms", 0.0)),
                float(item.get("compute_ms", 0.0)),
            )
        return {"model": model, "count": len(gathered), "results": gathered}

    @api.post("/stream/{model}")
    async def stream(self, model: str, req: PredictRequest):
        """Server-sent events, so callers see tokens instead of a spinner."""
        handle = self._handle(model)
        if not self._meta[model].get("enable_streaming", True):
            raise HTTPException(status_code=400, detail=f"streaming disabled for '{model}'")

        async def events():
            stats = self._stats[model]
            stats.inflight += 1
            start = time.perf_counter()
            try:
                generator = handle.options(stream=True).stream.remote(req.text)
                async for chunk in generator:
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                elapsed = (time.perf_counter() - start) * 1000.0
                stats.record(elapsed, 0.0, elapsed)
                yield f"data: {json.dumps({'done': True, 'latency_ms': round(elapsed, 2)})}\n\n"
            except Exception as exc:  # surface errors inside the stream
                stats.errors += 1
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            finally:
                stats.inflight -= 1

        return StreamingResponse(events(), media_type="text/event-stream")

    @api.get("/metrics/latency")
    async def latency(self) -> dict:
        """Decomposed, fleet-wide control feed consumed by the autoscaler."""
        replica_state = self._replica_state()
        out: dict = {}
        for name, stats in self._stats.items():
            total = stats.total.snapshot()
            meta = self._meta[name]
            entry = {
                "model": name,
                "slo_ms": meta.get("latency_slo_ms", 0.0),
                "queue_ms": round(stats.queue.snapshot().p95_ms, 2),
                "compute_ms": round(stats.compute.snapshot().p95_ms, 2),
                "inflight_requests": stats.inflight,
                "errors": stats.errors,
                **total.as_dict(),
            }
            entry.update(replica_state.get(name, {}))
            out[name] = entry
        return {"latency": out}

    @api.get("/admission")
    async def admission(self) -> dict:
        return {"admission": {name: ac.stats() for name, ac in self._admission.items()}}

    @api.get("/metrics")
    async def prometheus(self) -> Response:
        """Prometheus exposition for dashboards and alerting."""
        replica_state = self._replica_state()
        payload = render_prometheus(
            stats={
                name: {
                    "snapshot": s.total.snapshot(),
                    "queue_p95_ms": s.queue.snapshot().p95_ms,
                    "compute_p95_ms": s.compute.snapshot().p95_ms,
                    "inflight": s.inflight,
                    "errors": s.errors,
                    "requests": s.requests,
                }
                for name, s in self._stats.items()
            },
            meta=self._meta,
            replica_state=replica_state,
            admission={name: ac.stats() for name, ac in self._admission.items()},
        )
        return Response(content=payload, media_type="text/plain; version=0.0.4")
