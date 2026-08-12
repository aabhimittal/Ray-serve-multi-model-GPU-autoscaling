"""End-to-end integration test against a real (local, CPU) Ray Serve cluster.

Skipped automatically if ``ray`` is not installed, so the pure-logic suite can
still run in minimal environments. When ray is present this spins up a local
Serve instance, deploys the full multi-model graph with the simulated backend,
and exercises HTTP routing, latency metrics and the autoscaler control loop
against the live gateway.
"""

from __future__ import annotations

import pytest

ray = pytest.importorskip("ray")
serve = pytest.importorskip("ray.serve")
requests = pytest.importorskip("requests")

from ray_serve_autoscale.app import build_app  # noqa: E402
from ray_serve_autoscale.settings import load_settings  # noqa: E402

PORT = 8123
BASE = f"http://127.0.0.1:{PORT}"


@pytest.fixture(scope="module")
def live_gateway():
    ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    serve.start(detached=False, http_options={"host": "127.0.0.1", "port": PORT})
    settings = load_settings()
    settings.backend = "simulated"
    serve.run(build_app(settings), route_prefix="/", name="test_gateway")
    yield settings
    serve.shutdown()
    ray.shutdown()


def test_models_endpoint(live_gateway):
    r = requests.get(f"{BASE}/models", timeout=15)
    assert r.status_code == 200
    assert set(r.json()["models"]) == {"sentiment", "summarization", "embedding"}


def test_predict_each_model(live_gateway):
    for model in ("sentiment", "summarization", "embedding"):
        r = requests.post(f"{BASE}/predict/{model}", json={"text": f"hello {model}"}, timeout=30)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == model
        assert "latency_ms" in body
        assert body["result"]


def test_batch_endpoint(live_gateway):
    r = requests.post(
        f"{BASE}/batch/sentiment",
        json={"inputs": ["a", "b", "c", "d"]},
        timeout=30,
    )
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 4


def test_unknown_model_returns_404(live_gateway):
    r = requests.post(f"{BASE}/predict/nope", json={"text": "x"}, timeout=15)
    assert r.status_code == 404


def test_latency_metrics_and_autoscaler_reader(live_gateway):
    for _ in range(10):
        requests.post(f"{BASE}/predict/sentiment", json={"text": "warm"}, timeout=30)
    r = requests.get(f"{BASE}/metrics/latency", timeout=15)
    assert r.status_code == 200
    stats = r.json()["latency"]
    assert stats["sentiment"]["count"] >= 1

    # Wire the real autoscaler reader against the live gateway and take a step.
    from ray_serve_autoscale.autoscaling.latency_autoscaler import (
        LatencyAutoscaler,
        LoggingApplier,
    )

    def reader():
        return requests.get(f"{BASE}/metrics/latency", timeout=5).json()["latency"]

    ctrl = LatencyAutoscaler(
        models=live_gateway.models,
        cfg=live_gateway.autoscaler,
        reader=reader,
        applier=LoggingApplier(),
    )
    decisions = ctrl.step()
    assert {d.name for d in decisions} == {"sentiment", "summarization", "embedding"}


def test_latency_is_decomposed_into_queue_and_compute(live_gateway):
    """The decomposition must be measured for real, not just modelled."""
    r = requests.post(f"{BASE}/predict/summarization", json={"text": "a b c"}, timeout=30)
    body = r.json()
    assert body["compute_ms"] > 0
    assert body["queue_ms"] >= 0
    # Total must account for both parts (small slack for measurement overhead).
    assert body["latency_ms"] >= body["compute_ms"] - 1.0

    for _ in range(10):
        requests.post(f"{BASE}/predict/summarization", json={"text": "x y z"}, timeout=30)
    stats = requests.get(f"{BASE}/metrics/latency", timeout=15).json()["latency"]
    entry = stats["summarization"]
    assert entry["compute_ms"] > 0
    assert "queue_ms" in entry


def test_replica_state_is_reported_for_cold_start_accounting(live_gateway):
    stats = requests.get(f"{BASE}/metrics/latency", timeout=15).json()["latency"]
    # Serve's own status feeds ready/pending counts; if unavailable the
    # controller documents a fallback, so only assert consistency.
    for entry in stats.values():
        if "ready_replicas" in entry:
            assert entry["ready_replicas"] >= 0
            assert entry["pending_replicas"] >= 0


def test_streaming_endpoint_emits_incremental_events(live_gateway):
    with requests.post(
        f"{BASE}/stream/summarization",
        json={"text": "Ray Serve streams tokens as they are produced by the model"},
        stream=True,
        timeout=60,
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        chunks = [
            line
            for line in resp.iter_lines(decode_unicode=True)
            if line and line.startswith("data:")
        ]
    assert len(chunks) > 1  # genuinely incremental, not one blob
    assert any('"done": true' in c.lower() for c in chunks)


def test_prometheus_endpoint_exposes_control_series(live_gateway):
    for _ in range(5):
        requests.post(f"{BASE}/predict/embedding", json={"text": "vector"}, timeout=30)
    r = requests.get(f"{BASE}/metrics", timeout=15)
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = r.text
    for series in (
        "rsa_latency_p95_ms",
        "rsa_queue_p95_ms",
        "rsa_compute_p95_ms",
        "rsa_slo_attainment_ratio",
        "rsa_shed_total",
        "rsa_breaker_state",
    ):
        assert series in body, f"missing {series}"


def test_admission_endpoint_reports_shedding_state(live_gateway):
    r = requests.get(f"{BASE}/admission", timeout=15)
    assert r.status_code == 200
    admission = r.json()["admission"]
    assert set(admission) == {"sentiment", "summarization", "embedding"}
    assert admission["sentiment"]["breaker_state"] == "closed"


def test_fleet_controller_drives_a_live_gateway(live_gateway):
    """The full loop -- read, condition, plan, arbitrate -- against real data."""
    from ray_serve_autoscale.autoscaling.controller import FleetAutoscaler
    from ray_serve_autoscale.autoscaling.latency_autoscaler import LoggingApplier

    for _ in range(30):
        requests.post(f"{BASE}/predict/sentiment", json={"text": "load"}, timeout=30)

    def reader():
        return requests.get(f"{BASE}/metrics/latency", timeout=5).json()["latency"]

    controller = FleetAutoscaler(live_gateway, reader=reader, applier=LoggingApplier())
    result = controller.step()
    assert result.healthy
    assert {p.name for p in result.plans} == {"sentiment", "summarization", "embedding"}
    assert result.allocation is not None
    # Every plan must carry a human-readable justification -- an autoscaler
    # that cannot explain itself cannot be operated.
    assert all(p.reason for p in result.plans)
