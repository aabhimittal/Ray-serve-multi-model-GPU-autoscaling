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
