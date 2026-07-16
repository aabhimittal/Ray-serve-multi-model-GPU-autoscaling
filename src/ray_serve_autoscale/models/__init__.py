"""Model backends served behind Ray Serve deployments."""

from ray_serve_autoscale.models.base import ModelBackend, build_backend

__all__ = ["ModelBackend", "build_backend"]
