"""Tests for predictive, cold-start aware, latency-decomposed capacity planning."""

import pytest

from ray_serve_autoscale.autoscaling.planner import (
    BOUND_COMPUTE,
    BOUND_QUEUE,
    CapacityState,
    LatencyBreakdown,
    plan_model,
)
from ray_serve_autoscale.autoscaling.signals import ConditionedSignal
from ray_serve_autoscale.settings import AutoscalerConfig, ModelConfig


@pytest.fixture
def cfg():
    return AutoscalerConfig()


def _model(**kw):
    base = dict(
        name="m",
        task="sentiment",
        latency_slo_ms=200.0,
        min_replicas=1,
        max_replicas=10,
        num_gpus=0.5,
        max_ongoing_requests=16,
        cold_start_s=30.0,
    )
    base.update(kw)
    return ModelConfig(**base)


def _signal(p95, *, count=100, trend=0.0, stale=False, flapping=False, estimate=True):
    return ConditionedSignal(
        name="m",
        raw_p95_ms=p95,
        smoothed_p95_ms=p95,
        projected_p95_ms=p95,
        trend_ms_per_s=trend,
        sample_count=count,
        is_stale=stale,
        is_flapping=flapping,
        has_estimate=estimate,
    )


def _queue_bound(total=360.0):
    return LatencyBreakdown.build(total, total * 0.85, total * 0.15)


def _compute_bound(total=360.0):
    return LatencyBreakdown.build(total, total * 0.1, total * 0.9)


# --- ratio sizing -------------------------------------------------------------


def test_scale_out_sizes_by_ratio_not_single_steps(cfg):
    # 360ms against a 180ms target is 2x over -> 2x the replicas in ONE
    # decision, instead of crawling +1 per tick while burning error budget.
    plan = plan_model(
        _model(), _signal(360.0), CapacityState(ready_replicas=2), cfg, _queue_bound()
    )
    assert plan.desired_replicas == 4
    assert plan.bound == BOUND_QUEUE
    assert plan.direction == 1


def test_scale_out_is_capped_per_step(cfg):
    # A 20x breach must not order the entire cluster in one tick.
    plan = plan_model(
        _model(max_replicas=100),
        _signal(4000.0),
        CapacityState(ready_replicas=2),
        cfg,
        _queue_bound(4000.0),
    )
    assert plan.desired_replicas == 2 + cfg.max_scale_out_step


def test_scale_out_respects_max_replicas(cfg):
    plan = plan_model(
        _model(max_replicas=3),
        _signal(4000.0),
        CapacityState(ready_replicas=1),
        cfg,
        _queue_bound(),
    )
    assert plan.desired_replicas == 3


def test_scale_in_is_gradual(cfg):
    plan = plan_model(
        _model(), _signal(40.0), CapacityState(ready_replicas=4), cfg, _queue_bound(40.0)
    )
    # Ratio wants 1, but scale-in is capped at one replica per step.
    assert plan.desired_replicas == 3
    assert plan.direction == -1


def test_hold_inside_dead_band(cfg):
    plan = plan_model(
        _model(), _signal(100.0), CapacityState(ready_replicas=2), cfg, _queue_bound(100.0)
    )
    assert plan.direction == 0
    assert "hold" in plan.reason


# --- compute vs queue decomposition -------------------------------------------


def test_compute_bound_breach_caps_scale_out(cfg):
    # Latency is compute, not queueing: replicas cannot fix it, so we add at
    # most one for throughput and say why instead of buying 4 GPUs.
    plan = plan_model(
        _model(), _signal(360.0), CapacityState(ready_replicas=2), cfg, _compute_bound()
    )
    assert plan.bound == BOUND_COMPUTE
    assert plan.desired_replicas == 3
    assert "compute-bound" in plan.reason


def test_queue_bound_breach_scales_freely(cfg):
    plan = plan_model(
        _model(), _signal(360.0), CapacityState(ready_replicas=2), cfg, _queue_bound()
    )
    assert plan.bound == BOUND_QUEUE
    assert plan.desired_replicas == 4


def test_breakdown_reconciles_missing_total():
    bd = LatencyBreakdown.build(0, 80.0, 20.0)
    assert bd.total_ms == 100.0
    assert bd.queue_share == 0.8


def test_breakdown_without_parts_is_unknown(cfg):
    bd = LatencyBreakdown.build(360.0, 0, 0)
    assert not bd.known
    # Unknown decomposition must not silently behave as compute-bound.
    plan = plan_model(_model(), _signal(360.0), CapacityState(ready_replicas=2), cfg, bd)
    assert plan.desired_replicas == 4


def test_breakdown_clamps_impossible_shares():
    bd = LatencyBreakdown.build(100.0, 500.0, 10.0)  # queue > total
    assert bd.queue_share == 1.0


# --- cold-start / pending-replica accounting ----------------------------------


def test_pending_replicas_prevent_double_ordering(cfg):
    # 1 serving replica measured 2x over target -> needs 2 total. Three are
    # already warming, so the correct order is *none*, not another four.
    plan = plan_model(
        _model(),
        _signal(360.0),
        CapacityState(ready_replicas=1, pending_replicas=3),
        cfg,
        _queue_bound(),
    )
    assert plan.desired_replicas == 4  # == already-effective capacity
    assert plan.direction == 0
    assert "already warming" in plan.reason


def test_no_scale_in_while_replicas_are_warming(cfg):
    plan = plan_model(
        _model(),
        _signal(40.0),
        CapacityState(ready_replicas=2, pending_replicas=2),
        cfg,
        _queue_bound(40.0),
    )
    assert plan.direction == 0
    assert "still warming" in plan.reason


def test_unhealthy_replicas_are_not_capacity(cfg):
    # 4 replicas exist but 3 are wedged: the ratio must be taken against the
    # 1 that is actually serving, or we badly under-provision.
    plan = plan_model(
        _model(),
        _signal(360.0),
        CapacityState(ready_replicas=4, unhealthy_replicas=3),
        cfg,
        _queue_bound(),
    )
    assert plan.desired_replicas == 2


# --- guards -------------------------------------------------------------------


def test_total_replica_loss_restores_floor_immediately(cfg):
    # No replicas -> no traffic -> no samples. Every sample/dead-band guard
    # would say "hold", which would leave the model down forever.
    plan = plan_model(
        _model(min_replicas=2),
        _signal(0.0, count=0),
        CapacityState(ready_replicas=0),
        cfg,
        _queue_bound(0.0),
    )
    assert plan.desired_replicas == 2
    assert "no healthy replicas" in plan.reason


def test_all_replicas_unhealthy_is_also_an_outage(cfg):
    plan = plan_model(
        _model(min_replicas=2),
        _signal(0.0, count=0),
        CapacityState(ready_replicas=3, unhealthy_replicas=3),
        cfg,
        _queue_bound(0.0),
    )
    assert plan.desired_replicas == 2
    assert "no healthy replicas" in plan.reason


def test_stale_signal_holds(cfg):
    plan = plan_model(
        _model(), _signal(4000.0, stale=True), CapacityState(ready_replicas=2), cfg, _queue_bound()
    )
    assert plan.direction == 0
    assert "stale" in plan.reason


def test_missing_estimate_holds(cfg):
    plan = plan_model(
        _model(),
        _signal(4000.0, estimate=False),
        CapacityState(ready_replicas=2),
        cfg,
        _queue_bound(),
    )
    assert plan.direction == 0


def test_insufficient_samples_holds(cfg):
    plan = plan_model(
        _model(), _signal(4000.0, count=3), CapacityState(ready_replicas=2), cfg, _queue_bound()
    )
    assert plan.direction == 0
    assert "samples" in plan.reason


def test_zero_slo_is_not_scalable(cfg):
    plan = plan_model(
        _model(latency_slo_ms=0.0), _signal(500.0), CapacityState(ready_replicas=2), cfg
    )
    assert plan.direction == 0
    assert "no latency SLO" in plan.reason


def test_negative_slo_is_rejected(cfg):
    plan = plan_model(
        _model(latency_slo_ms=-100.0), _signal(500.0), CapacityState(ready_replicas=2), cfg
    )
    assert plan.direction == 0


def test_never_scales_below_floor(cfg):
    plan = plan_model(
        _model(
            min_replicas=3), _signal(1.0), CapacityState(ready_replicas=3), cfg, _queue_bound(1.0
        )
    )
    assert plan.desired_replicas == 3


def test_inflight_requests_block_over_aggressive_drain(cfg):
    # 64 in-flight at 16 per replica needs 4 replicas; scale-in must not
    # strand them mid-request.
    plan = plan_model(
        _model(),
        _signal(20.0),
        CapacityState(ready_replicas=4, inflight_requests=64),
        cfg,
        _queue_bound(20.0),
    )
    assert plan.desired_replicas == 4


# --- flap damping -------------------------------------------------------------


def test_widened_deadband_suppresses_marginal_scale_out(cfg):
    signal = _signal(185.0)  # just over the normal 180ms trigger
    normal = plan_model(_model(), signal, CapacityState(ready_replicas=2), cfg, _queue_bound(185.0))
    assert normal.direction == 1

    damped = plan_model(
        _model(),
        signal,
        CapacityState(ready_replicas=2),
        cfg,
        _queue_bound(185.0),
        deadband_multiplier=2.0,
    )
    assert damped.direction == 0


def test_widened_deadband_still_allows_decisive_breach(cfg):
    damped = plan_model(
        _model(),
        _signal(900.0),
        CapacityState(ready_replicas=2),
        cfg,
        _queue_bound(900.0),
        deadband_multiplier=2.0,
    )
    assert damped.direction == 1


# --- predictive vs reactive ---------------------------------------------------


def test_predictive_mode_uses_the_projection():
    cfg = AutoscalerConfig(predictive=True)
    # Present latency is healthy, but the projection over the cold-start
    # horizon already breaches -- order capacity now so it lands in time.
    signal = ConditionedSignal(
        name="m",
        raw_p95_ms=100.0,
        smoothed_p95_ms=100.0,
        projected_p95_ms=400.0,
        trend_ms_per_s=10.0,
        sample_count=100,
        is_stale=False,
        is_flapping=False,
    )
    plan = plan_model(_model(), signal, CapacityState(ready_replicas=2), cfg, _queue_bound(400.0))
    assert plan.direction == 1
    assert "trend" in plan.reason


def test_reactive_mode_ignores_the_projection():
    cfg = AutoscalerConfig(predictive=False)
    signal = ConditionedSignal(
        name="m",
        raw_p95_ms=100.0,
        smoothed_p95_ms=100.0,
        projected_p95_ms=400.0,
        trend_ms_per_s=10.0,
        sample_count=100,
        is_stale=False,
        is_flapping=False,
    )
    plan = plan_model(_model(), signal, CapacityState(ready_replicas=2), cfg, _queue_bound(100.0))
    assert plan.direction == 0


def test_plan_reports_gpu_demand(cfg):
    plan = plan_model(
        _model(num_gpus=0.25), _signal(360.0), CapacityState(ready_replicas=2), cfg, _queue_bound()
    )
    assert plan.gpu_demand == pytest.approx(4 * 0.25)
