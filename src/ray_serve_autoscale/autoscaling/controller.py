"""Fleet controller: the closed loop over every model at once.

``LatencyAutoscaler`` (latency_autoscaler.py) scales one deployment at a time
from its own p95. That is the right starting point, and it is what the original
implementation shipped. It has one structural blind spot: each model decides in
isolation, so on a cluster where GPUs are finite the sum of individually-correct
decisions is routinely infeasible.

``FleetAutoscaler`` closes the loop over the whole fleet:

    read  -> condition (EWMA / trend / staleness / flap)      signals.py
          -> plan      (predictive, cold-start aware, decomposed)  planner.py
          -> arbitrate (priority classes, budget ceiling)      arbiter.py
          -> apply     (cooldowns, emergency bypass)           here

Each stage is a pure function of its input, so the whole controller can be
driven deterministically through pathological scenarios in tests -- clock jumps,
frozen metrics, GPU exhaustion, budget starvation -- without a Ray cluster.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ray_serve_autoscale.autoscaling.arbiter import Allocation, arbitrate, required_gpus
from ray_serve_autoscale.autoscaling.planner import (
    CapacityState,
    LatencyBreakdown,
    ModelPlan,
    plan_model,
)
from ray_serve_autoscale.autoscaling.signals import SignalTracker, sanitize
from ray_serve_autoscale.settings import Settings

logger = logging.getLogger("ray_serve_autoscale.fleet")


@dataclass
class TickResult:
    """Everything one control iteration decided, for logs, tests and metrics."""

    plans: list[ModelPlan] = field(default_factory=list)
    allocation: Optional[Allocation] = None
    applied: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    healthy: bool = True

    def plan_for(self, name: str) -> Optional[ModelPlan]:
        return next((p for p in self.plans if p.name == name), None)

    def granted(self, name: str) -> Optional[int]:
        if self.allocation is None:
            return None
        grant = self.allocation.grants.get(name)
        return grant.granted_replicas if grant else None


class FleetAutoscaler:
    """Cross-model, SLO-driven, budget-aware GPU autoscaling controller."""

    def __init__(
        self,
        settings: Settings,
        reader: Callable[[], dict],
        applier: Optional[Callable[[str, int], bool]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._reader = reader
        self._clock = clock
        self._models = {m.name: m for m in settings.models}
        self._trackers = {
            name: SignalTracker(name, settings.signals) for name in self._models
        }
        # Replica floor currently in force per model (what we last applied).
        self._current = {name: m.min_replicas for name, m in self._models.items()}
        self._last_action_at = {name: -1e9 for name in self._models}
        # Consecutive ticks each model has wanted to scale out (see
        # AutoscalerConfig.scale_out_confirm_ticks).
        self._breach_ticks = {name: 0 for name in self._models}

        if applier is None:
            from ray_serve_autoscale.autoscaling.latency_autoscaler import LoggingApplier

            applier = LoggingApplier()
        self._applier = applier

    # -- observation parsing ------------------------------------------------

    @staticmethod
    def _capacity_from(obs: dict, fallback_replicas: int) -> CapacityState:
        """Read fleet state from an observation, tolerating missing fields.

        Older exporters (and the simple gateway endpoint) do not report replica
        counts. Falling back to the last applied floor keeps the ratio
        controller sane instead of dividing by a phantom zero.
        """
        ready = obs.get("ready_replicas")
        return CapacityState(
            ready_replicas=int(sanitize(ready, default=fallback_replicas))
            if ready is not None
            else fallback_replicas,
            pending_replicas=int(sanitize(obs.get("pending_replicas", 0))),
            unhealthy_replicas=int(sanitize(obs.get("unhealthy_replicas", 0))),
            inflight_requests=int(sanitize(obs.get("inflight_requests", 0))),
            configured_replicas=fallback_replicas,
        )

    # -- the loop -----------------------------------------------------------

    def step(self) -> TickResult:
        """Run one control iteration."""
        now = self._clock()
        result = TickResult()

        try:
            observations = self._reader() or {}
        except Exception as exc:
            # A broken metrics feed must never move the fleet. Holding is
            # always safer than acting on data we could not read.
            logger.error("metrics read failed, holding fleet: %s", exc)
            result.healthy = False
            result.notes.append(f"metrics read failed: {exc}")
            return result

        cfg = self._settings.autoscaler
        plans: list[ModelPlan] = []

        for name, model in self._models.items():
            obs = observations.get(name) or {}
            tracker = self._trackers[name]
            signal = tracker.update(
                p95_ms=obs.get("p95_ms", 0.0),
                sample_count=obs.get("count", 0),
                timestamp=now,
                horizon_s=model.cold_start_s,
            )
            capacity = self._capacity_from(obs, self._current[name])
            breakdown = LatencyBreakdown.build(
                obs.get("p95_ms", 0.0),
                obs.get("queue_ms", 0.0),
                obs.get("compute_ms", 0.0),
            )
            plan = plan_model(
                model,
                signal,
                capacity,
                cfg,
                breakdown,
                deadband_multiplier=tracker.deadband_multiplier(),
            )
            plans.append(plan)
            # Track how long this model has *wanted* more capacity. A single
            # tick of pressure is a spike; several in a row is a trend.
            if plan.direction > 0:
                self._breach_ticks[name] += 1
            else:
                self._breach_ticks[name] = 0
            if signal.is_stale:
                result.notes.append(f"{name}: metrics feed stale, holding")
            if signal.is_flapping:
                result.notes.append(f"{name}: flap damping engaged (dead band widened)")

        result.plans = plans

        allocation = arbitrate(plans, self._settings.cluster)
        result.allocation = allocation
        result.notes.extend(allocation.notes)

        demand = required_gpus(plans)
        if allocation.usable_gpus > 0 and demand > allocation.usable_gpus:
            result.notes.append(
                f"fleet demand {demand:.2f} GPU exceeds usable {allocation.usable_gpus:.2f} GPU"
            )

        # -- apply -----------------------------------------------------------
        for plan in plans:
            name = plan.name
            grant = allocation.grants.get(name)
            if grant is None:
                continue
            target = grant.granted_replicas
            current = self._current[name]
            if target == current:
                continue

            # An outage (zero healthy replicas) bypasses every damper: waiting
            # out a timer while nothing is serving would be indefensible.
            emergency = "no healthy replicas" in plan.reason

            confirm = max(cfg.scale_out_confirm_ticks, 1)
            if not emergency and target > current and self._breach_ticks[name] < confirm:
                result.skipped[name] = (
                    f"awaiting breach confirmation ({self._breach_ticks[name]}/{confirm})"
                )
                continue

            since = now - self._last_action_at[name]
            if not emergency and since < cfg.cooldown_s:
                result.skipped[name] = f"cooldown ({since:.0f}s < {cfg.cooldown_s:.0f}s)"
                continue

            if self._applier(name, target):
                direction = 1 if target > current else -1
                self._trackers[name].record_action(direction)
                self._current[name] = target
                self._last_action_at[name] = now
                result.applied[name] = target
                logger.info("%s: %d -> %d (%s)", name, current, target, grant.reason)
            else:
                result.skipped[name] = "applier rejected the change"

        return result

    def current_floors(self) -> dict[str, int]:
        """Replica floors currently in force (what the controller last set)."""
        return dict(self._current)

    def run_forever(self, stop: Optional[Callable[[], bool]] = None) -> None:  # pragma: no cover
        interval = self._settings.autoscaler.control_interval_s
        logger.info(
            "fleet autoscaler started (interval=%.0fs, models=%s)",
            interval,
            list(self._models),
        )
        while not (stop and stop()):
            try:
                self.step()
            except Exception as exc:
                logger.exception("fleet autoscaler tick failed: %s", exc)
            time.sleep(interval)
