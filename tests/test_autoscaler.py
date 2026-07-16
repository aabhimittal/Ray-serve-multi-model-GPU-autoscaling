"""Tests for the latency-based autoscaling policy and control loop."""

import pytest

from ray_serve_autoscale.autoscaling.latency_autoscaler import (
    DeploymentState,
    LatencyAutoscaler,
    LoggingApplier,
    decide,
)
from ray_serve_autoscale.settings import AutoscalerConfig, ModelConfig


@pytest.fixture
def cfg():
    return AutoscalerConfig(
        enabled=True,
        control_interval_s=15.0,
        scale_up_ratio=0.9,
        scale_down_ratio=0.4,
        scale_step=1,
        cooldown_s=60.0,
        min_samples=20,
    )


def _state(**kw):
    base = dict(
        name="m",
        p95_ms=100.0,
        slo_ms=200.0,
        sample_count=100,
        current_min_replicas=2,
        max_replicas=6,
        seconds_since_last_action=999.0,
    )
    base.update(kw)
    return DeploymentState(**base)


def test_scale_out_when_p95_near_slo(cfg):
    # p95 190 >= 200*0.9=180 -> scale out.
    d = decide(_state(p95_ms=190.0), cfg)
    assert d.changed
    assert d.target_min_replicas == 3


def test_scale_in_when_p95_low(cfg):
    # p95 50 <= 200*0.4=80 -> scale in.
    d = decide(_state(p95_ms=50.0, current_min_replicas=3), cfg)
    assert d.changed
    assert d.target_min_replicas == 2


def test_hold_inside_healthy_band(cfg):
    # 80 < p95 100 < 180 -> hold.
    d = decide(_state(p95_ms=100.0), cfg)
    assert not d.changed
    assert d.reason == "hold"


def test_respects_max_replicas(cfg):
    d = decide(_state(p95_ms=500.0, current_min_replicas=6, max_replicas=6), cfg)
    assert not d.changed
    assert "max_replicas" in d.reason


def test_never_scales_below_one(cfg):
    d = decide(_state(p95_ms=1.0, current_min_replicas=1), cfg)
    assert not d.changed
    assert d.target_min_replicas == 1


def test_insufficient_samples_holds(cfg):
    d = decide(_state(p95_ms=500.0, sample_count=5), cfg)
    assert not d.changed
    assert "insufficient samples" in d.reason


def test_cooldown_blocks_action(cfg):
    d = decide(_state(p95_ms=500.0, seconds_since_last_action=10.0), cfg)
    assert not d.changed
    assert "cooldown" in d.reason


def test_controller_step_applies_and_updates_state(cfg):
    models = [
        ModelConfig(
            name="m", task="sentiment", min_replicas=1, max_replicas=4, latency_slo_ms=200.0
        )
    ]
    clock = {"t": 0.0}
    # p95 above SLO threshold -> should scale out.
    reader = lambda: {"m": {"p95_ms": 190.0, "slo_ms": 200.0, "count": 100}}  # noqa: E731
    applier = LoggingApplier()
    ctrl = LatencyAutoscaler(models, cfg, reader=reader, applier=applier, clock=lambda: clock["t"])

    decisions = ctrl.step()
    assert decisions[0].target_min_replicas == 2
    assert applier.applied["m"] == 2

    # Immediately after, cooldown should prevent another scale-out.
    clock["t"] = 10.0
    decisions = ctrl.step()
    assert not decisions[0].changed
    assert "cooldown" in decisions[0].reason

    # After cooldown expires, it can scale out again.
    clock["t"] = 100.0
    decisions = ctrl.step()
    assert decisions[0].target_min_replicas == 3


def test_controller_handles_missing_model_snapshot(cfg):
    models = [ModelConfig(name="m", task="sentiment", latency_slo_ms=200.0)]
    ctrl = LatencyAutoscaler(
        models, cfg, reader=lambda: {}, applier=LoggingApplier(), clock=lambda: 0.0
    )
    decisions = ctrl.step()
    assert not decisions[0].changed  # zero samples -> hold
