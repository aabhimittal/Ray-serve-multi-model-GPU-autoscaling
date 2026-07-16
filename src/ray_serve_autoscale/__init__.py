"""Multi-model serving on Ray Serve with latency-driven GPU autoscaling.

Public surface (imported lazily so pure-logic modules -- settings, metrics,
the autoscaling policy -- don't require ``ray`` to be installed/imported):

    build_app        -- construct the Serve application (needs ray[serve])
    Settings         -- runtime configuration loaded from env / YAML
    get_settings     -- cached settings accessor
    LatencyAutoscaler-- custom controller that scales replicas from p95 latency
"""

from __future__ import annotations

import typing

from ray_serve_autoscale.autoscaling.latency_autoscaler import LatencyAutoscaler
from ray_serve_autoscale.settings import Settings, get_settings

__all__ = ["build_app", "Settings", "get_settings", "LatencyAutoscaler"]

__version__ = "0.1.0"

if typing.TYPE_CHECKING:
    from ray_serve_autoscale.app import build_app


def __getattr__(name: str):
    # Lazily import the Serve-dependent app builder only when actually used.
    if name == "build_app":
        from ray_serve_autoscale.app import build_app

        return build_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
