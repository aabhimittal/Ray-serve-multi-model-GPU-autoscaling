"""Tests for the sliding latency window and percentile math."""

from ray_serve_autoscale.autoscaling.metrics import LatencyWindow, _percentile


def test_percentile_nearest_rank():
    values = [float(v) for v in range(1, 101)]  # 1..100
    assert _percentile(values, 50) == 50
    assert _percentile(values, 95) == 95
    assert _percentile(values, 99) == 99
    assert _percentile(values, 100) == 100
    assert _percentile(values, 0) == 1


def test_percentile_empty():
    assert _percentile([], 95) == 0.0


def test_window_empty_snapshot():
    w = LatencyWindow(window_s=10.0)
    snap = w.snapshot(now=0.0)
    assert snap.count == 0
    assert snap.p95_ms == 0.0


def test_window_records_and_aggregates():
    w = LatencyWindow(window_s=100.0)
    for i in range(1, 101):
        w.record(float(i), now=0.0)
    snap = w.snapshot(now=0.0)
    assert snap.count == 100
    assert snap.p50_ms == 50
    assert snap.p95_ms == 95
    assert snap.max_ms == 100


def test_window_evicts_old_samples():
    w = LatencyWindow(window_s=10.0)
    # Old samples at t=0, fresh samples at t=100 -> only fresh survive.
    for _ in range(5):
        w.record(1000.0, now=0.0)
    for _ in range(3):
        w.record(50.0, now=100.0)
    snap = w.snapshot(now=100.0)
    assert snap.count == 3
    assert snap.max_ms == 50.0


def test_snapshot_as_dict_rounds():
    w = LatencyWindow(window_s=100.0)
    w.record(123.456789, now=0.0)
    d = w.snapshot(now=0.0).as_dict()
    assert set(d) == {"count", "p50_ms", "p95_ms", "p99_ms", "max_ms", "rps"}
    assert d["max_ms"] == 123.46
