"""Industrial edge cases, driven end-to-end through the fleet controller.

These are the scenarios that break autoscalers in production. Each one is a
failure mode that has taken real GPU services down: metrics feeds that freeze,
clocks that jump, replicas that all die at once, cold-start storms that
over-provision a whole cluster, oscillation that pays a cold start every
minute, and contention that silently starves whichever model happened to lose
the scheduling race.

The controller is deliberately built from pure functions plus injected
clock/reader/applier, so every one of these can be reproduced deterministically
in milliseconds with no cluster, no GPU and no sleeping.
"""

import copy

import pytest

from ray_serve_autoscale.autoscaling.controller import FleetAutoscaler
from ray_serve_autoscale.settings import (
    AutoscalerConfig,
    ClusterConfig,
    ModelConfig,
    Settings,
    SignalConfig,
)


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class Harness:
    """Drives a FleetAutoscaler with a scripted metrics feed and fake clock."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.clock = FakeClock()
        self.feed: dict = {}
        self.applied: list = []
        self.apply_ok = True
        self.reader_error: BaseException | None = None
        # Replica counts the "cluster" is actually running, updated on apply
        # so the feed reflects the controller's own decisions.
        self.replicas = {m.name: m.min_replicas for m in settings.models}
        self.controller = FleetAutoscaler(
            settings, reader=self._read, applier=self._apply, clock=self.clock
        )

    def _read(self) -> dict:
        if self.reader_error is not None:
            raise self.reader_error
        return copy.deepcopy(self.feed)

    def _apply(self, name: str, target: int) -> bool:
        if not self.apply_ok:
            return False
        self.applied.append((name, target))
        self.replicas[name] = target
        return True

    def observe(self, name: str, p95, *, count=1000, queue=None, compute=None, **kw) -> None:
        """Publish one model's observation into the feed."""
        obs = {
            "p95_ms": p95,
            "count": count,
            "ready_replicas": kw.pop("ready_replicas", self.replicas.get(name, 1)),
        }
        if queue is not None or compute is not None:
            obs["queue_ms"] = queue if queue is not None else 0.0
            obs["compute_ms"] = compute if compute is not None else 0.0
        obs.update(kw)
        self.feed[name] = obs

    def tick(self, advance: float = 90.0):
        self.clock.advance(advance)
        return self.controller.step()


def _model(name="m", **kw) -> ModelConfig:
    base = dict(
        name=name,
        task="sentiment",
        latency_slo_ms=1000.0,
        min_replicas=1,
        max_replicas=20,
        num_gpus=0.5,
        max_ongoing_requests=16,
        cold_start_s=60.0,
        priority=100,
        cost_per_gpu_hour_usd=2.0,
    )
    base.update(kw)
    return ModelConfig(**base)


def _settings(models=None, **kw) -> Settings:
    return Settings(
        models=models or [_model()],
        # alpha=1.0 makes smoothing a no-op so scenarios are exactly
        # reproducible; smoothing itself is covered in test_signals.py.
        signals=kw.pop("signals", SignalConfig(ewma_alpha=1.0)),
        autoscaler=kw.pop("autoscaler", AutoscalerConfig()),
        cluster=kw.pop("cluster", ClusterConfig()),
        **kw,
    )


# =============================================================================
# 1. Metrics-feed pathologies
# =============================================================================


def test_frozen_metrics_feed_freezes_the_controller_not_the_fleet():
    """A wedged exporter keeps serving its last payload forever.

    Counts stop advancing while the values still look plausible -- the most
    dangerous failure mode for a closed loop, because it looks like health.
    """
    h = Harness(_settings(signals=SignalConfig(ewma_alpha=1.0, staleness_ticks=2)))
    for i in range(3):  # healthy baseline, counter advancing
        h.observe("m", 100.0, count=1000 * (i + 1), queue=90.0, compute=10.0)
        h.tick()
    baseline = len(h.applied)

    for _ in range(6):  # exporter wedges at a screaming value
        h.observe("m", 5000.0, count=99_999)
        result = h.tick()

    # The first frozen sample is indistinguishable from a real one, so at most
    # one action may escape before staleness is detected -- then it must stop.
    assert len(h.applied) - baseline <= 1
    assert any("stale" in n for n in result.notes)


def test_reader_failure_holds_the_entire_fleet():
    h = Harness(_settings())
    h.reader_error = ConnectionError("metrics endpoint unreachable")
    result = h.tick()
    assert not result.healthy
    assert h.applied == []
    assert any("metrics read failed" in n for n in result.notes)


def test_nan_and_infinite_latencies_never_move_the_fleet():
    """Garbage must not read as 0ms -- that would look perfectly healthy."""
    h = Harness(_settings())
    for bad in (float("nan"), float("inf"), -1.0, "corrupted", None):
        h.observe("m", bad, count=1000)
        h.tick()
    assert h.applied == []


def test_missing_model_in_the_feed_holds_only_that_model():
    h = Harness(_settings(models=[_model("a"), _model("b")]))
    h.observe("a", 5000.0, count=1000, queue=4800.0, compute=200.0)
    # "b" is simply absent from the payload.
    result = h.tick()
    assert result.granted("a") > 1
    assert result.plan_for("b").direction == 0


def test_partial_observation_fields_are_tolerated():
    # An exporter that reports only p95 and count (no replica/queue detail)
    # must still drive the loop rather than crash it.
    h = Harness(_settings())
    h.feed["m"] = {"p95_ms": 5000.0, "count": 1000}
    result = h.tick()
    assert result.healthy
    assert result.granted("m") > 1


# =============================================================================
# 2. Clock pathologies
# =============================================================================


def test_clock_running_backwards_does_not_corrupt_the_controller():
    """NTP steps and container suspend/resume really do move clocks back."""
    h = Harness(_settings())
    for i in range(4):
        h.observe("m", 200.0 + 10 * i, count=1000 * (i + 1))
        h.tick()
    h.clock.advance(-500.0)  # time travel
    h.observe("m", 300.0, count=9000)
    result = h.tick(advance=0.0)
    assert result.healthy  # no crash, no negative-trend nonsense
    assert all(t >= 1 for _, t in h.applied)


def test_enormous_clock_jump_forward_is_survivable():
    h = Harness(_settings())
    h.observe("m", 100.0, count=1000)
    h.tick()
    h.observe("m", 100.0, count=2000)
    result = h.tick(advance=86_400.0)  # a day
    assert result.healthy


# =============================================================================
# 3. Capacity / outage pathologies
# =============================================================================


def test_total_replica_loss_restores_capacity_immediately():
    """No replicas -> no traffic -> no samples: every ordinary guard says
    "hold", which would leave the model down forever."""
    h = Harness(_settings(models=[_model(min_replicas=2)]))
    h.observe("m", 0.0, count=0, ready_replicas=0)
    result = h.tick()
    plan = result.plan_for("m")
    assert "no healthy replicas" in plan.reason
    assert plan.desired_replicas >= 2


def test_outage_recovery_bypasses_the_cooldown():
    """Capacity returning after a starvation cut must not wait out a timer."""
    cluster = ClusterConfig(total_gpus=0.4)  # cannot fit even one 0.5-GPU replica
    h = Harness(_settings(cluster=cluster, autoscaler=AutoscalerConfig(cooldown_s=600.0)))
    h.observe("m", 100.0, count=1000)
    h.tick()
    assert h.controller.current_floors()["m"] == 0  # starved to zero

    # GPUs come back, but the model is now dead and reports nothing.
    h.settings.cluster.total_gpus = 8.0
    h.observe("m", 0.0, count=0, ready_replicas=0)
    result = h.tick(advance=5.0)  # far inside the 600s cooldown
    assert result.applied.get("m") == 1
    assert "m" not in result.skipped


def test_unhealthy_replicas_are_not_counted_as_capacity():
    h = Harness(_settings())
    h.replicas["m"] = 4
    h.observe(
        "m", 2000.0, count=1000, ready_replicas=4, unhealthy_replicas=3, queue=1900.0, compute=100.0
    )
    result = h.tick()
    # Ratio taken against the 1 serving replica, not the 4 nominal ones.
    assert result.plan_for("m").desired_replicas >= 2


def test_cold_start_storm_does_not_re_order_the_same_capacity():
    """The classic thundering herd: latency stays high while replicas warm,
    so a naive controller orders the same capacity again every single tick
    and overshoots the entire cluster."""
    h = Harness(_settings())
    h.replicas["m"] = 1
    for _ in range(5):
        h.observe(
            "m",
            4000.0,
            count=1000,
            ready_replicas=1,
            pending_replicas=7,  # already warming
            queue=3800.0,
            compute=200.0,
        )
        h.tick()
    # 1 serving replica measured 4x over target needs ~4 total; 8 are already
    # effective, so no further orders are justified.
    assert all(target <= 8 for _, target in h.applied)


def test_in_flight_requests_are_not_stranded_by_scale_in():
    h = Harness(_settings())
    h.replicas["m"] = 4
    h.observe("m", 10.0, count=1000, ready_replicas=4, inflight_requests=64, queue=5.0, compute=5.0)
    result = h.tick()
    # 64 in flight at 16 per replica needs 4 replicas -- draining below that
    # would kill requests already accepted.
    assert result.plan_for("m").desired_replicas == 4


# =============================================================================
# 4. Load-shape pathologies
# =============================================================================


def test_predictive_control_acts_before_the_slo_is_breached():
    """The whole point of cold-start-aware scaling.

    Present latency is comfortably inside the SLO, but the ramp guarantees a
    breach by the time replicas finish booting. Reactive control cannot win
    this race; predictive control orders capacity that lands in time.
    """
    autoscaler = AutoscalerConfig(predictive=True)
    h = Harness(_settings(autoscaler=autoscaler))
    for i, p95 in enumerate((100.0, 300.0, 500.0, 700.0)):
        h.observe("m", p95, count=1000 * (i + 1), queue=p95 * 0.9, compute=p95 * 0.1)
        result = h.tick(advance=20.0)
    plan = result.plan_for("m")
    # 700ms observed is under the 900ms trigger, but +10ms/s over a 60s cold
    # start projects to 1300ms.
    assert plan.direction == 1
    assert "trend" in plan.reason


def test_reactive_control_misses_the_same_ramp():
    h = Harness(_settings(autoscaler=AutoscalerConfig(predictive=False)))
    for i, p95 in enumerate((100.0, 300.0, 500.0, 700.0)):
        h.observe("m", p95, count=1000 * (i + 1), queue=p95 * 0.9, compute=p95 * 0.1)
        result = h.tick(advance=20.0)
    assert result.plan_for("m").direction == 0


def test_single_transient_spike_never_orders_gpus():
    """One GC pause or cold cache must not order GPUs.

    Smoothing alone cannot carry this: 20% of a 40x outlier already clears the
    trigger. Breach *confirmation* is what makes the guarantee hold -- a spike
    lasting one tick can never reach the confirmation threshold.
    """
    settings = _settings(
        signals=SignalConfig(ewma_alpha=0.2),
        autoscaler=AutoscalerConfig(predictive=False),
    )
    h = Harness(settings)
    for i in range(10):
        h.observe("m", 100.0, count=1000 * (i + 1), queue=90.0, compute=10.0)
        h.tick()
    h.observe("m", 4000.0, count=12_000, queue=3900.0, compute=100.0)  # single spike
    result = h.tick()
    h.observe("m", 100.0, count=13_000, queue=90.0, compute=10.0)  # recovered
    h.tick()

    assert h.applied == []
    assert "confirmation" in result.skipped.get("m", "")


def test_sustained_pressure_does_scale_out():
    settings = _settings(
        signals=SignalConfig(ewma_alpha=0.2),
        autoscaler=AutoscalerConfig(predictive=False),
    )
    h = Harness(settings)
    for i in range(10):
        h.observe("m", 100.0, count=1000 * (i + 1), queue=90.0, compute=10.0)
        h.tick()
    for i in range(8):
        h.observe("m", 4000.0, count=20_000 + 1000 * i, queue=3900.0, compute=100.0)
        h.tick()
    # Confirmed pressure reaches the cluster and keeps growing capacity until
    # the model's own ceiling stops it.
    assert h.applied
    assert h.controller.current_floors()["m"] > 1
    assert all(target <= 20 for _, target in h.applied)


def test_breach_confirmation_requires_consecutive_ticks():
    cfg = AutoscalerConfig(predictive=False, cooldown_s=0.0, scale_out_confirm_ticks=3)
    h = Harness(_settings(autoscaler=cfg))
    for i in range(2):
        h.observe("m", 5000.0, count=1000 * (i + 1), queue=4900.0, compute=100.0)
        h.tick()
    assert h.applied == []  # 2 of 3 confirmations

    h.observe("m", 5000.0, count=9000, queue=4900.0, compute=100.0)
    h.tick()
    assert h.applied  # third consecutive breach releases the order


def test_intermittent_breaches_reset_the_confirmation_counter():
    cfg = AutoscalerConfig(predictive=False, cooldown_s=0.0, scale_out_confirm_ticks=2)
    h = Harness(_settings(autoscaler=cfg))
    for i in range(8):  # alternating breach / healthy, never two in a row
        p95 = 5000.0 if i % 2 == 0 else 50.0
        h.observe("m", p95, count=1000 * (i + 1), queue=p95 * 0.98, compute=p95 * 0.02)
        h.tick()
    assert all(target <= 1 for _, target in h.applied)


def test_compute_bound_latency_does_not_buy_useless_gpus():
    """Latency is the model itself, not queueing. Replicas cannot fix it, and
    buying four of them wastes GPUs while the real cause goes unreported."""
    h = Harness(_settings())
    h.replicas["m"] = 2
    h.observe("m", 4000.0, count=1000, ready_replicas=2, queue=100.0, compute=3900.0)
    result = h.tick()
    plan = result.plan_for("m")
    assert plan.bound == "compute"
    assert plan.desired_replicas == 3  # capped at +1, not 2 x 4.4
    assert "compute-bound" in plan.reason


def test_idle_model_never_scales_to_zero():
    h = Harness(_settings(models=[_model(min_replicas=1)]))
    for _i in range(8):
        h.observe("m", 1.0, count=0)  # completely idle
        h.tick()
    assert all(target >= 1 for _, target in h.applied)
    assert h.controller.current_floors()["m"] >= 1


def test_burst_then_silence_does_not_thrash():
    h = Harness(
        _settings(autoscaler=AutoscalerConfig(cooldown_s=300.0, scale_out_confirm_ticks=1))
    )
    h.observe("m", 5000.0, count=5000, queue=4900.0, compute=100.0)
    h.tick()
    scaled_to = h.controller.current_floors()["m"]
    assert scaled_to > 1

    # Traffic vanishes instantly; the cooldown must hold capacity briefly
    # rather than dumping every warm replica the moment the burst ends.
    h.observe("m", 5.0, count=6000, ready_replicas=scaled_to)
    result = h.tick(advance=30.0)
    assert "m" in result.skipped
    assert h.controller.current_floors()["m"] == scaled_to


def test_oscillation_engages_flap_damping():
    """Every flap cycle pays a cold start and evicts a warm CUDA context."""
    h = Harness(
        _settings(autoscaler=AutoscalerConfig(cooldown_s=0.0, scale_out_confirm_ticks=1))
    )
    result = None
    for i in range(8):
        p95 = 5000.0 if i % 2 == 0 else 10.0
        h.observe("m", p95, count=1000 * (i + 1), queue=p95 * 0.98, compute=p95 * 0.02)
        result = h.tick()
    assert any("flap damping" in n for n in result.notes)


# =============================================================================
# 5. Contention: finite GPUs and finite money
# =============================================================================


def test_gpu_exhaustion_preserves_priority_ordering():
    """When every model is breaching and the cluster cannot serve them all,
    *which* model gets starved must be a policy decision, not a scheduling
    accident."""
    models = [
        _model("interactive", priority=300, num_gpus=0.5),
        _model("internal", priority=200, num_gpus=0.5),
        _model("batch", priority=100, num_gpus=0.5),
    ]
    h = Harness(_settings(models=models, cluster=ClusterConfig(total_gpus=2.0)))
    for name in ("interactive", "internal", "batch"):
        h.observe(name, 8000.0, count=5000, queue=7900.0, compute=100.0)
    result = h.tick()

    assert result.granted("interactive") >= result.granted("internal")
    assert result.granted("internal") >= result.granted("batch")
    assert result.allocation.allocated_gpus <= 2.0 + 1e-9
    assert result.allocation.contended


def test_high_priority_model_is_never_preempted_by_a_more_urgent_batch_job():
    models = [
        _model("interactive", priority=300, latency_slo_ms=1000.0),
        _model("batch", priority=100, latency_slo_ms=100.0),
    ]
    h = Harness(_settings(models=models, cluster=ClusterConfig(total_gpus=1.5)))
    h.observe("interactive", 1100.0, count=5000, queue=1000.0, compute=100.0)
    h.observe("batch", 9000.0, count=5000, queue=8900.0, compute=100.0)  # 90x over SLO
    result = h.tick()
    assert result.granted("interactive") >= result.granted("batch")


def test_budget_ceiling_caps_total_spend():
    models = [_model("a", priority=200), _model("b", priority=100)]
    h = Harness(
        _settings(
            models=models,
            cluster=ClusterConfig(total_gpus=64.0, max_hourly_budget_usd=4.0),
        )
    )
    for name in ("a", "b"):
        h.observe(name, 9000.0, count=5000, queue=8900.0, compute=100.0)
    result = h.tick()
    assert result.allocation.hourly_cost_usd <= 4.0 + 1e-6
    assert not result.allocation.over_budget
    assert result.granted("a") >= result.granted("b")


def test_cluster_without_declared_gpus_still_scales():
    # CI and CPU dev boxes must not be starved into a no-op fleet.
    h = Harness(_settings(cluster=ClusterConfig(total_gpus=0.0)))
    h.observe("m", 5000.0, count=5000, queue=4900.0, compute=100.0)
    result = h.tick()
    assert result.granted("m") > 1


def test_starvation_is_surfaced_not_silent():
    models = [_model("winner", priority=300), _model("loser", priority=100)]
    h = Harness(_settings(models=models, cluster=ClusterConfig(total_gpus=0.5)))
    for name in ("winner", "loser"):
        h.observe(name, 5000.0, count=5000, queue=4900.0, compute=100.0)
    result = h.tick()
    assert "loser" in result.allocation.starved
    assert any("STARVED" in n for n in result.notes)


# =============================================================================
# 6. Control-plane pathologies
# =============================================================================


def test_failed_apply_does_not_corrupt_controller_state():
    """If the Serve REST call fails, the controller must not believe it
    succeeded -- or it will never retry the change."""
    h = Harness(_settings(autoscaler=AutoscalerConfig(scale_out_confirm_ticks=1)))
    h.apply_ok = False
    h.observe("m", 5000.0, count=5000, queue=4900.0, compute=100.0)
    result = h.tick()
    assert result.skipped.get("m") == "applier rejected the change"
    assert h.controller.current_floors()["m"] == 1  # unchanged

    h.apply_ok = True
    h.observe("m", 5000.0, count=6000, queue=4900.0, compute=100.0)
    result = h.tick()
    assert result.applied.get("m", 0) > 1  # retried and succeeded


def test_cooldown_prevents_rapid_successive_changes():
    h = Harness(
        _settings(autoscaler=AutoscalerConfig(cooldown_s=300.0, scale_out_confirm_ticks=1))
    )
    h.observe("m", 5000.0, count=5000, queue=4900.0, compute=100.0)
    h.tick()
    h.observe("m", 9000.0, count=6000, queue=8900.0, compute=100.0)
    result = h.tick(advance=10.0)
    assert "cooldown" in result.skipped.get("m", "")


def test_no_redundant_applies_when_target_is_unchanged():
    h = Harness(_settings(autoscaler=AutoscalerConfig(cooldown_s=0.0)))
    for i in range(5):
        h.observe("m", 100.0, count=1000 * (i + 1), queue=90.0, compute=10.0)
        h.tick()
    assert h.applied == []  # steady state writes nothing


def test_extreme_values_do_not_overflow_or_hang():
    h = Harness(_settings(models=[_model(max_replicas=1000)]))
    h.observe("m", 1e12, count=10**9, queue=9.9e11, compute=1e10)
    result = h.tick()
    plan = result.plan_for("m")
    assert plan.desired_replicas <= 1000
    assert plan.desired_replicas == 1 + AutoscalerConfig().max_scale_out_step


def test_many_models_scale_independently():
    models = [_model(f"m{i}", priority=100 + i) for i in range(12)]
    h = Harness(_settings(models=models, cluster=ClusterConfig(total_gpus=64.0)))
    for i, m in enumerate(models):
        p95 = 5000.0 if i % 2 == 0 else 50.0
        h.observe(m.name, p95, count=5000, queue=p95 * 0.98, compute=p95 * 0.02)
    result = h.tick()
    for i, m in enumerate(models):
        expected = 1 if i % 2 == 0 else 0
        assert result.plan_for(m.name).direction == expected


@pytest.mark.parametrize("bad_cluster", [-1.0, 0.0, 1e9])
def test_pathological_cluster_sizes_are_survivable(bad_cluster):
    h = Harness(_settings(cluster=ClusterConfig(total_gpus=bad_cluster)))
    h.observe("m", 5000.0, count=5000, queue=4900.0, compute=100.0)
    result = h.tick()
    assert result.healthy
    assert result.granted("m") >= 1
