"""Latency-driven supervisory autoscaler for GPU model deployments.

Why a *custom* autoscaler on top of Ray Serve's built-in one?
------------------------------------------------------------
Ray Serve already autoscales each deployment toward a target number of ongoing
requests per replica. That controls latency *indirectly* (queue depth -> wait
time, by Little's Law) but it has no notion of your actual latency SLO. If a
model gets slower per-request (a bigger prompt, a cold cache, GPU contention),
queue-based scaling can hold the target ongoing-requests and still blow the SLO.

This controller closes that gap. It periodically reads the true end-to-end p95
latency each deployment is serving and adjusts the deployment's **autoscaling
floor** (``min_replicas``):

* p95 climbing toward the SLO  -> raise the floor (add GPU replicas now).
* p95 comfortably under the SLO -> lower the floor (let Serve scale back in and
  release GPUs).

It cooperates with -- rather than fights -- the built-in autoscaler: the floor
guarantees enough capacity to meet the SLO, while Serve's queue-based logic
still handles fine-grained scaling above that floor.

Design
------
``decide()`` is a *pure function* of the observed state, so all the policy is
unit-tested without a cluster. ``LatencyAutoscaler`` wires a *reader* (where
latency snapshots come from) to an *applier* (how replica floors are changed),
either of which is injectable for testing and for different environments.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from ray_serve_autoscale.settings import AutoscalerConfig, ModelConfig

logger = logging.getLogger("ray_serve_autoscale.autoscaler")


@dataclass(frozen=True)
class DeploymentState:
    """Everything the policy needs to reason about one deployment."""

    name: str
    p95_ms: float
    slo_ms: float
    sample_count: int
    current_min_replicas: int
    max_replicas: int
    seconds_since_last_action: float


@dataclass(frozen=True)
class ScalingDecision:
    """Result of the policy for one deployment."""

    name: str
    current_min_replicas: int
    target_min_replicas: int
    reason: str

    @property
    def changed(self) -> bool:
        return self.target_min_replicas != self.current_min_replicas


def decide(state: DeploymentState, cfg: AutoscalerConfig) -> ScalingDecision:
    """Pure policy: map an observed deployment state to a replica-floor target.

    Guard rails, in order:
      * too few samples this window -> hold (don't react to noise).
      * still in cooldown after the last action -> hold.
      * p95 >= slo * scale_up_ratio -> add ``scale_step`` replicas (capped).
      * p95 <= slo * scale_down_ratio -> remove ``scale_step`` replicas (>=1).
      * otherwise -> hold (inside the healthy band).
    """
    current = state.current_min_replicas
    hold = ScalingDecision(state.name, current, current, "hold")

    if state.sample_count < cfg.min_samples:
        return ScalingDecision(
            state.name,
            state.current_min_replicas,
            state.current_min_replicas,
            f"insufficient samples ({state.sample_count} < {cfg.min_samples})",
        )

    if state.seconds_since_last_action < cfg.cooldown_s:
        return ScalingDecision(
            state.name,
            state.current_min_replicas,
            state.current_min_replicas,
            f"cooldown ({state.seconds_since_last_action:.0f}s < {cfg.cooldown_s:.0f}s)",
        )

    upper = state.slo_ms * cfg.scale_up_ratio
    lower = state.slo_ms * cfg.scale_down_ratio

    if state.p95_ms >= upper:
        target = min(state.current_min_replicas + cfg.scale_step, state.max_replicas)
        if target == state.current_min_replicas:
            return ScalingDecision(state.name, current, target, "at max_replicas")
        return ScalingDecision(
            state.name,
            state.current_min_replicas,
            target,
            f"p95 {state.p95_ms:.0f}ms >= {upper:.0f}ms -> scale out to {target}",
        )

    if state.p95_ms <= lower:
        target = max(state.current_min_replicas - cfg.scale_step, 1)
        if target == state.current_min_replicas:
            return ScalingDecision(state.name, state.current_min_replicas, target, "at floor (1)")
        return ScalingDecision(
            state.name,
            state.current_min_replicas,
            target,
            f"p95 {state.p95_ms:.0f}ms <= {lower:.0f}ms -> scale in to {target}",
        )

    return hold


# ---- Reader / Applier plumbing ------------------------------------------------


class LatencyReader(Protocol):
    """Supplies the current latency snapshot for every model."""

    def __call__(self) -> dict[str, dict]:  # {model_name: latency_stats_dict}
        ...


class ReplicaApplier(Protocol):
    """Applies a new replica floor to a deployment. Returns success."""

    def __call__(self, model: str, target_min_replicas: int) -> bool:
        ...


class LoggingApplier:
    """Safe default applier: records the decision without mutating the cluster.

    Useful in tests, dry-runs and CI. Swap for :class:`ServeRestApplier` in a
    live cluster to actually move replica floors.
    """

    def __init__(self) -> None:
        self.applied: dict[str, int] = {}

    def __call__(self, model: str, target_min_replicas: int) -> bool:
        self.applied[model] = target_min_replicas
        logger.info("[dry-run] would set %s min_replicas -> %d", model, target_min_replicas)
        return True


class ServeRestApplier:
    """Applies replica floors via the Ray Serve REST API on the dashboard agent.

    It GETs the current declarative config, patches the target deployment's
    ``autoscaling_config.min_replicas``, and PUTs it back. Serve performs a
    lightweight in-place update -- it does not restart healthy replicas.
    """

    def __init__(self, dashboard_url: str = "http://127.0.0.1:8265", timeout_s: float = 10.0):
        self._url = dashboard_url.rstrip("/") + "/api/serve/applications/"
        self._timeout = timeout_s

    def __call__(self, model: str, target_min_replicas: int) -> bool:  # pragma: no cover
        import requests

        try:
            resp = requests.get(self._url, timeout=self._timeout)
            resp.raise_for_status()
            config = resp.json()
            patched = False
            for app in config.get("applications", []):
                for dep in app.get("deployments", []):
                    if dep.get("name") == model:
                        dep.setdefault("autoscaling_config", {})
                        dep["autoscaling_config"]["min_replicas"] = target_min_replicas
                        patched = True
            if not patched:
                logger.warning("deployment %s not found in Serve config", model)
                return False
            put = requests.put(self._url, json=config, timeout=self._timeout)
            put.raise_for_status()
            return True
        except Exception as exc:
            logger.error("failed to apply min_replicas for %s: %s", model, exc)
            return False


# ---- The controller loop ------------------------------------------------------


class LatencyAutoscaler:
    """Supervisory control loop tying a reader to an applier via ``decide``."""

    def __init__(
        self,
        models: list[ModelConfig],
        cfg: AutoscalerConfig,
        reader: LatencyReader,
        applier: ReplicaApplier | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = cfg
        self._reader = reader
        self._applier = applier or LoggingApplier()
        self._clock = clock
        self._slo = {m.name: m.latency_slo_ms for m in models}
        self._max = {m.name: m.max_replicas for m in models}
        self._min_replicas = {m.name: m.min_replicas for m in models}
        self._last_action_at = {m.name: -1e9 for m in models}

    def step(self) -> list[ScalingDecision]:
        """One control iteration: read, decide, apply. Returns the decisions."""
        now = self._clock()
        snapshots = self._reader()
        decisions: list[ScalingDecision] = []
        for name, slo_ms in self._slo.items():
            snap = snapshots.get(name, {})
            state = DeploymentState(
                name=name,
                p95_ms=float(snap.get("p95_ms", 0.0)),
                slo_ms=float(snap.get("slo_ms", slo_ms)),
                sample_count=int(snap.get("count", 0)),
                current_min_replicas=self._min_replicas[name],
                max_replicas=self._max[name],
                seconds_since_last_action=now - self._last_action_at[name],
            )
            decision = decide(state, self._cfg)
            decisions.append(decision)
            if decision.changed and self._applier(name, decision.target_min_replicas):
                self._min_replicas[name] = decision.target_min_replicas
                self._last_action_at[name] = now
                logger.info("scaled %s: %s", name, decision.reason)
        return decisions

    def run_forever(self, stop: Callable[[], bool] | None = None) -> None:  # pragma: no cover
        """Blocking control loop. ``stop`` lets an embedder request shutdown."""
        logger.info(
            "latency autoscaler started (interval=%.0fs, models=%s)",
            self._cfg.control_interval_s,
            list(self._slo),
        )
        while not (stop and stop()):
            try:
                self.step()
            except Exception as exc:
                logger.exception("autoscaler step failed: %s", exc)
            time.sleep(self._cfg.control_interval_s)
