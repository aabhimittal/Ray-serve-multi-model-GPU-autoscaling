"""Tests for signal conditioning: sanitization, EWMA, trend, staleness, flap."""

import math

from ray_serve_autoscale.autoscaling.signals import (
    Ewma,
    FlapDamper,
    SignalTracker,
    StalenessDetector,
    TrendEstimator,
    finite_or_none,
    sanitize,
)
from ray_serve_autoscale.settings import SignalConfig

# --- sanitize / finite_or_none ------------------------------------------------


def test_sanitize_rejects_non_finite():
    assert sanitize(float("nan")) == 0.0
    assert sanitize(float("inf")) == 0.0
    assert sanitize(float("-inf")) == 0.0


def test_sanitize_rejects_negative_by_default():
    assert sanitize(-5.0) == 0.0
    assert sanitize(-5.0, allow_negative=True) == -5.0


def test_sanitize_handles_garbage_types():
    assert sanitize("not a number") == 0.0
    assert sanitize(None) == 0.0
    assert sanitize({}) == 0.0
    assert sanitize("12.5") == 12.5


def test_finite_or_none_drops_rather_than_zeroes():
    # The critical distinction: a bad latency must NOT read as 0.0 (healthy).
    assert finite_or_none(float("nan")) is None
    assert finite_or_none(float("inf")) is None
    assert finite_or_none(-1.0) is None
    assert finite_or_none("junk") is None
    assert finite_or_none(42.0) == 42.0


# --- EWMA ---------------------------------------------------------------------


def test_ewma_first_sample_is_unbiased():
    # Without bias correction the first sample would read 40, not 100 --
    # low enough for a controller to miss the start of a ramp.
    e = Ewma(0.4)
    assert e.update(100.0) == 100.0


def test_ewma_converges_to_steady_state():
    e = Ewma(0.5)
    for _ in range(50):
        e.update(200.0)
    assert abs(e.value - 200.0) < 1e-6


def test_ewma_smooths_a_single_outlier():
    e = Ewma(0.3)
    for _ in range(20):
        e.update(100.0)
    e.update(10_000.0)  # one pathological spike
    # Smoothed value must stay far below the spike, or one bad sample
    # would order GPUs the fleet does not need.
    assert e.value < 3200.0


def test_ewma_ignores_garbage_values():
    e = Ewma(0.5)
    e.update(100.0)
    before = e.value
    e.update(float("nan"))
    e.update(float("-inf"))
    e.update("junk")
    assert e.value == before


def test_ewma_clamps_pathological_alpha():
    assert Ewma(0.0).update(50.0) == 50.0
    assert Ewma(99.0).update(50.0) == 50.0
    assert Ewma(-1.0).update(50.0) == 50.0


def test_ewma_uninitialized_reports_zero_and_flag():
    e = Ewma(0.5)
    assert e.value == 0.0
    assert not e.initialized
    e.update(1.0)
    assert e.initialized


# --- Trend --------------------------------------------------------------------


def test_trend_fits_a_linear_ramp():
    t = TrendEstimator(window_s=1000)
    for i in range(10):
        t.add(float(i), 10.0 * i)  # +10 ms per second
    assert abs(t.slope_ms_per_s() - 10.0) < 1e-6


def test_trend_needs_three_points():
    t = TrendEstimator()
    t.add(0.0, 1.0)
    t.add(1.0, 2.0)
    assert t.slope_ms_per_s() == 0.0


def test_trend_handles_identical_timestamps():
    # Zero variance in x would divide by zero in the regression.
    t = TrendEstimator()
    for v in (1.0, 2.0, 3.0, 4.0):
        t.add(5.0, v)
    assert t.slope_ms_per_s() == 0.0


def test_trend_resets_when_clock_goes_backwards():
    # Container suspend/resume and NTP steps really do move clocks backwards.
    t = TrendEstimator(window_s=1000)
    for i in range(5):
        t.add(float(i), 10.0 * i)
    t.add(0.5, 999.0)  # timestamp regression
    assert t.slope_ms_per_s() == 0.0  # history cleared, not inverted


def test_trend_evicts_outside_window():
    t = TrendEstimator(window_s=10.0)
    for i in range(5):
        t.add(float(i), 100.0)
    for i in range(100, 105):
        t.add(float(i), 500.0)
    # Only the recent flat cluster remains -> slope ~0, not a huge fake ramp.
    assert abs(t.slope_ms_per_s()) < 1e-6


def test_trend_projection_looks_ahead():
    t = TrendEstimator(window_s=1000)
    for i in range(10):
        t.add(float(i), 10.0 * i)
    # 100ms now, climbing 10ms/s, 30s cold start -> 400ms when replicas land.
    assert abs(t.project(100.0, 30.0) - 400.0) < 1e-6


def test_trend_projection_never_goes_negative():
    t = TrendEstimator(window_s=1000)
    for i in range(10):
        t.add(float(i), 1000.0 - 100.0 * i)  # steeply falling
    assert t.project(10.0, 60.0) == 0.0


def test_trend_ignores_invalid_values():
    t = TrendEstimator(window_s=1000)
    for i in range(10):
        t.add(float(i), 10.0 * i)
    slope = t.slope_ms_per_s()
    t.add(11.0, float("nan"))
    assert abs(t.slope_ms_per_s() - slope) < 1e-9


# --- Staleness ----------------------------------------------------------------


def test_staleness_flags_a_frozen_feed():
    d = StalenessDetector(staleness_ticks=3)
    assert not d.observe(10)
    assert not d.observe(10)  # repeats=1
    assert not d.observe(10)  # repeats=2
    assert d.observe(10)  # repeats=3 -> stale
    assert d.is_stale


def test_staleness_resets_when_counter_advances():
    d = StalenessDetector(staleness_ticks=2)
    d.observe(5)
    d.observe(5)
    d.observe(5)
    assert d.is_stale
    d.observe(6)
    assert not d.is_stale


def test_staleness_ignores_pathological_limit():
    d = StalenessDetector(staleness_ticks=0)  # clamped to 1
    d.observe(1)
    assert d.observe(1)


# --- Flap damping -------------------------------------------------------------


def test_flap_detects_oscillation():
    f = FlapDamper(window=6, threshold=3)
    for direction in (1, -1, 1, -1):
        f.record(direction)
    assert f.reversals == 3
    assert f.is_flapping


def test_flap_ignores_steady_direction():
    f = FlapDamper(window=6, threshold=3)
    for _ in range(6):
        f.record(1)
    assert f.reversals == 0
    assert not f.is_flapping


def test_flap_ignores_holds():
    f = FlapDamper(window=6, threshold=1)
    f.record(0)
    f.record(0)
    assert f.reversals == 0
    assert not f.is_flapping


# --- SignalTracker integration ------------------------------------------------


def test_tracker_produces_trustworthy_signal():
    tracker = SignalTracker("m", SignalConfig())
    sig = None
    for i in range(5):
        sig = tracker.update(
            p95_ms=100.0, sample_count=10 * (i + 1), timestamp=float(i), horizon_s=30.0
        )
    assert sig.trustworthy
    assert sig.smoothed_p95_ms > 0


def test_tracker_marks_frozen_feed_untrustworthy():
    tracker = SignalTracker("m", SignalConfig(staleness_ticks=2))
    for i in range(6):
        sig = tracker.update(p95_ms=100.0, sample_count=42, timestamp=float(i), horizon_s=30.0)
    assert sig.is_stale
    assert not sig.trustworthy


def test_tracker_all_garbage_feed_has_no_estimate():
    # Never saw a usable value -> must not present 0.0 as a healthy latency.
    tracker = SignalTracker("m", SignalConfig())
    for i in range(4):
        sig = tracker.update(
            p95_ms=float("nan"), sample_count=10 * (i + 1), timestamp=float(i), horizon_s=30.0
        )
    assert not sig.has_estimate
    assert not sig.trustworthy


def test_tracker_projects_a_rising_ramp():
    tracker = SignalTracker("m", SignalConfig(ewma_alpha=1.0))
    for i in range(20):
        sig = tracker.update(
            p95_ms=10.0 * i, sample_count=100 * (i + 1), timestamp=float(i), horizon_s=10.0
        )
    assert sig.trend_ms_per_s > 0
    assert sig.projected_p95_ms > sig.smoothed_p95_ms


def test_tracker_deadband_widens_only_when_flapping():
    cfg = SignalConfig(flap_window=6, flap_threshold=3, flap_deadband_widening=2.0)
    tracker = SignalTracker("m", cfg)
    assert tracker.deadband_multiplier() == 1.0
    for direction in (1, -1, 1, -1):
        tracker.record_action(direction)
    assert tracker.deadband_multiplier() == 2.0


def test_tracker_survives_infinite_and_negative_inputs():
    tracker = SignalTracker("m", SignalConfig())
    tracker.update(p95_ms=100.0, sample_count=50, timestamp=0.0, horizon_s=30.0)
    sig = tracker.update(p95_ms=float("inf"), sample_count=60, timestamp=1.0, horizon_s=30.0)
    assert math.isfinite(sig.smoothed_p95_ms)
    assert math.isfinite(sig.projected_p95_ms)
    sig = tracker.update(p95_ms=-42.0, sample_count=70, timestamp=2.0, horizon_s=30.0)
    assert math.isfinite(sig.smoothed_p95_ms)
