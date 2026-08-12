"""Tests for load shedding and circuit breaking on the request path."""

import math

import pytest

from ray_serve_autoscale.serving.admission import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    AdmissionController,
    CircuitBreaker,
    estimate_wait_ms,
)
from ray_serve_autoscale.settings import AdmissionConfig


class FakeClock:
    """Deterministic clock so breaker timing is testable without sleeping."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def cfg():
    return AdmissionConfig()


# --- wait estimation ----------------------------------------------------------


def test_empty_queue_means_no_wait():
    assert estimate_wait_ms(0, 4, 50.0) == 0.0


def test_no_replicas_means_unbounded_wait():
    # Must be inf, not a ZeroDivisionError and not a deceptively small number.
    assert math.isinf(estimate_wait_ms(10, 0, 50.0))


def test_wait_scales_with_queue_and_replicas():
    assert estimate_wait_ms(100, 2, 50.0) == pytest.approx(2500.0)
    assert estimate_wait_ms(100, 4, 50.0) == pytest.approx(1250.0)


def test_batching_concurrency_reduces_estimated_wait():
    assert estimate_wait_ms(100, 2, 50.0, max_concurrent_per_replica=4) == pytest.approx(625.0)


def test_wait_estimation_sanitizes_garbage():
    assert estimate_wait_ms(-5, 2, 50.0) == 0.0
    assert estimate_wait_ms(10, 2, float("nan")) == 0.0
    assert estimate_wait_ms(10, 2, 50.0, max_concurrent_per_replica=0) == pytest.approx(250.0)


# --- load shedding ------------------------------------------------------------


def test_admits_when_the_queue_is_short(cfg):
    ac = AdmissionController("m", 200.0, cfg, clock=FakeClock())
    decision = ac.check(queue_depth=2, replicas=2, service_time_ms=50.0)
    assert decision.admitted
    assert decision.status_code == 200


def test_sheds_when_projected_wait_blows_the_slo(cfg):
    # 50 queued, 2 replicas, 50ms each -> 1250ms wait against a 300ms budget.
    # Accepting this request would burn GPU time on an answer nobody waits for.
    ac = AdmissionController("m", 200.0, cfg, shed_at_slo_fraction=1.5, clock=FakeClock())
    decision = ac.check(queue_depth=50, replicas=2, service_time_ms=50.0)
    assert not decision.admitted
    assert decision.status_code == 503
    assert "projected wait" in decision.reason
    assert decision.retry_after_s > 0


def test_sheds_when_no_replicas_are_serving(cfg):
    ac = AdmissionController("m", 200.0, cfg, clock=FakeClock())
    decision = ac.check(queue_depth=5, replicas=0, service_time_ms=50.0)
    assert not decision.admitted
    # Retry-After must stay finite even though the wait estimate is infinite.
    assert math.isfinite(decision.retry_after_s)


def test_queue_depth_cap_is_enforced_independently(cfg):
    # An unbounded queue is a memory leak as well as a latency problem, so it
    # is capped even when per-request service time looks trivial.
    ac = AdmissionController("m", 10_000.0, cfg, clock=FakeClock())
    decision = ac.check(queue_depth=1000, replicas=1, service_time_ms=0.001)
    assert not decision.admitted
    assert "queue depth" in decision.reason


def test_disabled_admission_always_admits():
    ac = AdmissionController("m", 200.0, AdmissionConfig(enabled=False), clock=FakeClock())
    decision = ac.check(queue_depth=10_000, replicas=0, service_time_ms=5000.0)
    assert decision.admitted


def test_zero_slo_disables_wait_shedding(cfg):
    # No SLO means no basis to call a request doomed; only the hard cap applies.
    ac = AdmissionController("m", 0.0, cfg, clock=FakeClock())
    assert ac.check(queue_depth=5, replicas=2, service_time_ms=5000.0).admitted


# --- circuit breaker ----------------------------------------------------------


def test_breaker_opens_on_sustained_error_rate(cfg):
    clock = FakeClock()
    breaker = CircuitBreaker(cfg, clock)
    for _ in range(cfg.breaker_min_requests):
        breaker.record(False)
    assert breaker.state == OPEN
    assert not breaker.allow()


def test_breaker_ignores_a_short_burst_of_failures(cfg):
    clock = FakeClock()
    breaker = CircuitBreaker(cfg, clock)
    for _ in range(5):  # below breaker_min_requests
        breaker.record(False)
    assert breaker.state == CLOSED
    assert breaker.allow()


def test_breaker_stays_closed_below_the_error_threshold(cfg):
    clock = FakeClock()
    breaker = CircuitBreaker(cfg, clock)
    for i in range(40):
        breaker.record(i % 4 == 0)  # 75% failures... opens
    assert breaker.state == OPEN

    ok_breaker = CircuitBreaker(cfg, FakeClock())
    for i in range(40):
        ok_breaker.record(i % 4 != 0)  # 25% failures -> below 50% threshold
    assert ok_breaker.state == CLOSED


def test_breaker_half_opens_after_cooldown(cfg):
    clock = FakeClock()
    breaker = CircuitBreaker(cfg, clock)
    for _ in range(cfg.breaker_min_requests):
        breaker.record(False)
    assert breaker.state == OPEN

    clock.advance(cfg.breaker_open_s)
    assert breaker.state == HALF_OPEN


def test_half_open_admits_only_a_bounded_number_of_probes(cfg):
    clock = FakeClock()
    breaker = CircuitBreaker(cfg, clock)
    for _ in range(cfg.breaker_min_requests):
        breaker.record(False)
    clock.advance(cfg.breaker_open_s)

    admitted = sum(1 for _ in range(20) if breaker.allow())
    assert admitted == cfg.breaker_half_open_probes


def test_successful_probe_closes_the_breaker(cfg):
    clock = FakeClock()
    breaker = CircuitBreaker(cfg, clock)
    for _ in range(cfg.breaker_min_requests):
        breaker.record(False)
    clock.advance(cfg.breaker_open_s)
    breaker.allow()
    breaker.record(True)
    assert breaker.state == CLOSED
    assert breaker.allow()


def test_failed_probe_reopens_the_breaker(cfg):
    clock = FakeClock()
    breaker = CircuitBreaker(cfg, clock)
    for _ in range(cfg.breaker_min_requests):
        breaker.record(False)
    clock.advance(cfg.breaker_open_s)
    breaker.allow()
    breaker.record(False)
    assert breaker.state == OPEN
    assert not breaker.allow()


def test_recovery_does_not_immediately_retrip_on_stale_history(cfg):
    # After a successful probe the old failures must not instantly reopen the
    # breaker -- otherwise a recovered model can never get traffic back.
    clock = FakeClock()
    breaker = CircuitBreaker(cfg, clock)
    for _ in range(cfg.breaker_min_requests):
        breaker.record(False)
    clock.advance(cfg.breaker_open_s)
    breaker.allow()
    breaker.record(True)
    for _ in range(5):
        breaker.record(True)
    assert breaker.state == CLOSED


# --- controller integration ---------------------------------------------------


def test_open_breaker_sheds_at_the_controller(cfg):
    clock = FakeClock()
    ac = AdmissionController("m", 200.0, cfg, clock=clock)
    for _ in range(cfg.breaker_min_requests):
        ac.record_result(False)
    decision = ac.check(queue_depth=0, replicas=4, service_time_ms=10.0)
    assert not decision.admitted
    assert "circuit breaker" in decision.reason


def test_stats_track_admit_and_shed_counts(cfg):
    ac = AdmissionController("m", 200.0, cfg, clock=FakeClock())
    ac.check(queue_depth=0, replicas=4, service_time_ms=10.0)
    ac.check(queue_depth=500, replicas=1, service_time_ms=100.0)
    stats = ac.stats()
    assert stats["admitted"] == 1
    assert stats["shed"] == 1
    assert stats["shed_rate"] == pytest.approx(0.5)
    assert stats["breaker_state"] == CLOSED
