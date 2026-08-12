"""Prometheus exposition for the serving fleet.

Written by hand rather than through ``prometheus_client`` for one reason: the
gateway is a Serve deployment that may run several replicas, and a process-wide
client registry would make each replica export a *different* slice of the same
counters. Rendering directly from the gateway's own rolling windows keeps the
exposition a pure function of observed state -- which also makes it testable
without scraping anything.

The series here are chosen to answer the questions an operator actually asks
during an incident:

* Is the SLO being met?                 ``..._latency_p95_ms`` vs ``..._slo_ms``
* Why is it slow -- queue or compute?   ``..._queue_p95_ms`` / ``..._compute_p95_ms``
* Is capacity arriving?                 ``..._replicas_ready`` / ``..._replicas_pending``
* Are we shedding, and why?             ``..._shed_total`` / ``..._breaker_state``
* What is this costing?                 ``..._gpus_allocated`` / ``..._hourly_cost_usd``
"""

from __future__ import annotations

from typing import Optional

_PREFIX = "rsa"

# Breaker states as a numeric series, since Prometheus stores floats.
_BREAKER_STATES = {"closed": 0.0, "half_open": 1.0, "open": 2.0}


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


class _Doc:
    """Accumulates metric families in exposition-format order."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def family(self, name: str, kind: str, help_text: str) -> None:
        self._lines.append(f"# HELP {name} {help_text}")
        self._lines.append(f"# TYPE {name} {kind}")

    def sample(self, name: str, labels: dict, value: float) -> None:
        rendered = ",".join(f'{k}="{_escape(str(v))}"' for k, v in sorted(labels.items()))
        suffix = f"{{{rendered}}}" if rendered else ""
        self._lines.append(f"{name}{suffix} {float(value):g}")

    def render(self) -> str:
        return "\n".join(self._lines) + "\n"


def render_prometheus(
    stats: dict,
    meta: dict,
    replica_state: Optional[dict] = None,
    admission: Optional[dict] = None,
    allocation: Optional[dict] = None,
) -> str:
    """Render the fleet's current state as Prometheus exposition text."""
    replica_state = replica_state or {}
    admission = admission or {}
    doc = _Doc()

    def emit(metric: str, kind: str, help_text: str, values: dict) -> None:
        if not values:
            return
        name = f"{_PREFIX}_{metric}"
        doc.family(name, kind, help_text)
        for model, value in sorted(values.items()):
            doc.sample(name, {"model": model}, value)

    emit(
        "requests_total",
        "counter",
        "Requests observed by the gateway.",
        {m: s.get("requests", 0) for m, s in stats.items()},
    )
    emit(
        "errors_total",
        "counter",
        "Requests that failed downstream.",
        {m: s.get("errors", 0) for m, s in stats.items()},
    )
    emit(
        "inflight_requests",
        "gauge",
        "Requests dispatched but not yet returned.",
        {m: s.get("inflight", 0) for m, s in stats.items()},
    )
    emit(
        "latency_p50_ms",
        "gauge",
        "Median end-to-end latency over the rolling window.",
        {m: s["snapshot"].p50_ms for m, s in stats.items() if "snapshot" in s},
    )
    emit(
        "latency_p95_ms",
        "gauge",
        "p95 end-to-end latency over the rolling window.",
        {m: s["snapshot"].p95_ms for m, s in stats.items() if "snapshot" in s},
    )
    emit(
        "latency_p99_ms",
        "gauge",
        "p99 end-to-end latency over the rolling window.",
        {m: s["snapshot"].p99_ms for m, s in stats.items() if "snapshot" in s},
    )
    emit(
        "queue_p95_ms",
        "gauge",
        "p95 time spent waiting for a replica (scale-out fixes this).",
        {m: s.get("queue_p95_ms", 0.0) for m, s in stats.items()},
    )
    emit(
        "compute_p95_ms",
        "gauge",
        "p95 time spent in the model forward pass (scale-out does NOT fix this).",
        {m: s.get("compute_p95_ms", 0.0) for m, s in stats.items()},
    )
    emit(
        "requests_per_second",
        "gauge",
        "Throughput over the rolling window.",
        {m: s["snapshot"].rps for m, s in stats.items() if "snapshot" in s},
    )
    emit(
        "slo_ms",
        "gauge",
        "Configured latency SLO.",
        {m: d.get("latency_slo_ms", 0.0) for m, d in meta.items()},
    )
    # Ratio > 1 means the SLO is being missed: the single most useful alerting
    # series, because it is already normalised across models.
    emit(
        "slo_attainment_ratio",
        "gauge",
        "p95 latency divided by the SLO; >1 means the SLO is being missed.",
        {
            m: (s["snapshot"].p95_ms / meta[m]["latency_slo_ms"])
            for m, s in stats.items()
            if "snapshot" in s and meta.get(m, {}).get("latency_slo_ms", 0) > 0
        },
    )
    emit(
        "replicas_ready",
        "gauge",
        "Replicas currently serving.",
        {m: v.get("ready_replicas", 0) for m, v in replica_state.items()},
    )
    emit(
        "replicas_pending",
        "gauge",
        "Replicas cold-starting (capacity already on the way).",
        {m: v.get("pending_replicas", 0) for m, v in replica_state.items()},
    )
    emit(
        "gpus_allocated",
        "gauge",
        "GPU fractions currently held by this model.",
        {
            m: replica_state.get(m, {}).get("ready_replicas", 0) * d.get("num_gpus", 0.0)
            for m, d in meta.items()
        },
    )
    emit(
        "hourly_cost_usd",
        "gauge",
        "Current GPU spend rate for this model.",
        {
            m: (
                replica_state.get(m, {}).get("ready_replicas", 0)
                * d.get("num_gpus", 0.0)
                * d.get("cost_per_gpu_hour_usd", 0.0)
            )
            for m, d in meta.items()
        },
    )
    emit(
        "admitted_total",
        "counter",
        "Requests admitted by admission control.",
        {m: a.get("admitted", 0) for m, a in admission.items()},
    )
    emit(
        "shed_total",
        "counter",
        "Requests shed to protect the SLO of admitted traffic.",
        {m: a.get("shed", 0) for m, a in admission.items()},
    )
    emit(
        "breaker_state",
        "gauge",
        "Circuit breaker state (0=closed, 1=half_open, 2=open).",
        {
            m: _BREAKER_STATES.get(a.get("breaker_state", "closed"), 0.0)
            for m, a in admission.items()
        },
    )

    if allocation:
        emit(
            "replicas_desired",
            "gauge",
            "Replicas the autoscaler wants before arbitration.",
            {m: v.get("desired_replicas", 0) for m, v in allocation.items()},
        )
        emit(
            "replicas_granted",
            "gauge",
            "Replicas granted after cluster arbitration.",
            {m: v.get("granted_replicas", 0) for m, v in allocation.items()},
        )
        emit(
            "replicas_unmet",
            "gauge",
            "Demand arbitration could not satisfy (GPU or budget contention).",
            {m: v.get("unmet_replicas", 0) for m, v in allocation.items()},
        )

    return doc.render()
