"""Tests for Prometheus exposition rendering."""

from ray_serve_autoscale.autoscaling.metrics import LatencyWindow
from ray_serve_autoscale.observability.prometheus import render_prometheus


def _stats(p95=250.0, queue=200.0, compute=50.0):
    window = LatencyWindow(window_s=100.0)
    for _ in range(20):
        window.record(p95, now=0.0)
    return {
        "sentiment": {
            "snapshot": window.snapshot(now=0.0),
            "queue_p95_ms": queue,
            "compute_p95_ms": compute,
            "inflight": 3,
            "errors": 1,
            "requests": 100,
        }
    }


_META = {
    "sentiment": {
        "latency_slo_ms": 500.0,
        "num_gpus": 0.5,
        "cost_per_gpu_hour_usd": 2.0,
    }
}


def _parse(payload: str) -> dict:
    out = {}
    for line in payload.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        key, _, value = line.rpartition(" ")
        out[key] = float(value)
    return out


def test_exposition_is_wellformed():
    payload = render_prometheus(_stats(), _META)
    assert payload.endswith("\n")
    for line in payload.splitlines():
        assert line.startswith("#") or " " in line


def test_core_series_are_present():
    series = _parse(render_prometheus(_stats(), _META))
    assert series['rsa_latency_p95_ms{model="sentiment"}'] == 250.0
    assert series['rsa_queue_p95_ms{model="sentiment"}'] == 200.0
    assert series['rsa_compute_p95_ms{model="sentiment"}'] == 50.0
    assert series['rsa_inflight_requests{model="sentiment"}'] == 3.0
    assert series['rsa_errors_total{model="sentiment"}'] == 1.0


def test_slo_attainment_ratio_normalizes_across_models():
    # The single most useful alerting series: >1 means the SLO is missed,
    # regardless of each model's absolute latency budget.
    series = _parse(render_prometheus(_stats(p95=250.0), _META))
    assert series['rsa_slo_attainment_ratio{model="sentiment"}'] == 0.5

    series = _parse(render_prometheus(_stats(p95=1000.0), _META))
    assert series['rsa_slo_attainment_ratio{model="sentiment"}'] == 2.0


def test_zero_slo_is_omitted_not_divided_by():
    payload = render_prometheus(_stats(), {"sentiment": {"latency_slo_ms": 0.0}})
    assert "slo_attainment_ratio" not in payload


def test_replica_and_cost_series():
    replica_state = {"sentiment": {"ready_replicas": 4, "pending_replicas": 2}}
    series = _parse(render_prometheus(_stats(), _META, replica_state=replica_state))
    assert series['rsa_replicas_ready{model="sentiment"}'] == 4.0
    assert series['rsa_replicas_pending{model="sentiment"}'] == 2.0
    assert series['rsa_gpus_allocated{model="sentiment"}'] == 2.0
    assert series['rsa_hourly_cost_usd{model="sentiment"}'] == 4.0


def test_admission_series_encode_breaker_state():
    admission = {"sentiment": {"admitted": 90, "shed": 10, "breaker_state": "open"}}
    series = _parse(render_prometheus(_stats(), _META, admission=admission))
    assert series['rsa_shed_total{model="sentiment"}'] == 10.0
    assert series['rsa_breaker_state{model="sentiment"}'] == 2.0

    admission["sentiment"]["breaker_state"] = "half_open"
    series = _parse(render_prometheus(_stats(), _META, admission=admission))
    assert series['rsa_breaker_state{model="sentiment"}'] == 1.0


def test_allocation_series_are_optional():
    payload = render_prometheus(_stats(), _META)
    assert "replicas_granted" not in payload

    allocation = {"sentiment": {"desired_replicas": 8, "granted_replicas": 3, "unmet_replicas": 5}}
    series = _parse(render_prometheus(_stats(), _META, allocation=allocation))
    assert series['rsa_replicas_granted{model="sentiment"}'] == 3.0
    assert series['rsa_replicas_unmet{model="sentiment"}'] == 5.0


def test_empty_fleet_renders_without_crashing():
    assert render_prometheus({}, {}) == "\n"


def test_label_values_are_escaped():
    stats = {'we"ird\\name': {"inflight": 1, "errors": 0, "requests": 1}}
    payload = render_prometheus(stats, {})
    assert r"we\"ird\\name" in payload
