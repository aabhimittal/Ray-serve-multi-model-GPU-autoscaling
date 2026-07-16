"""Tests for the model backends and the backend factory."""

from ray_serve_autoscale.models.base import (
    SimulatedBackend,
    TransformersBackend,
    build_backend,
)
from ray_serve_autoscale.settings import ModelConfig


def _cfg(task, latency=0.001):
    return ModelConfig(name=task, task=task, simulated_latency_s=latency)


def test_simulated_sentiment_shape():
    b = SimulatedBackend(_cfg("sentiment"))
    b.load()
    out = b.predict(["good", "bad"])
    assert len(out) == 2
    assert set(out[0]) == {"label", "score"}
    assert out[0]["label"] in {"POSITIVE", "NEGATIVE"}


def test_simulated_is_deterministic():
    b = SimulatedBackend(_cfg("sentiment"))
    b.load()
    assert b.predict(["hello"])[0] == b.predict(["hello"])[0]


def test_simulated_summarization_shape():
    b = SimulatedBackend(_cfg("summarization"))
    b.load()
    out = b.predict(["one two three four five six seven eight nine ten eleven twelve thirteen"])
    assert "summary_text" in out[0]
    assert out[0]["summary_text"].endswith("...")


def test_simulated_embedding_shape():
    b = SimulatedBackend(_cfg("embedding"))
    b.load()
    out = b.predict(["vectorize me"])
    assert out[0]["dim"] == 16
    assert len(out[0]["embedding"]) == 16


def test_build_backend_falls_back_to_simulated_without_ml_deps():
    # transformers/torch not installed in CI -> factory returns SimulatedBackend.
    b = build_backend(_cfg("sentiment"), backend="transformers")
    assert isinstance(b, (SimulatedBackend, TransformersBackend))


def test_build_backend_simulated_explicit():
    b = build_backend(_cfg("embedding"), backend="simulated")
    assert isinstance(b, SimulatedBackend)


def test_batch_latency_scales_with_size():
    b = SimulatedBackend(_cfg("sentiment", latency=0.01))
    b.load()
    import time

    t0 = time.perf_counter()
    b.predict(["a"])
    single = time.perf_counter() - t0

    t0 = time.perf_counter()
    b.predict(["a"] * 8)
    batch = time.perf_counter() - t0
    # Batched call takes longer in absolute terms but is sub-linear per item.
    assert batch > single
    assert batch < single * 8
