"""Model backend abstraction.

A *backend* is the thing that turns a request into a prediction. It is
deliberately decoupled from Ray Serve so it can be unit-tested without a Ray
cluster and swapped between a real GPU model (``transformers``) and a
deterministic ``simulated`` implementation used on CPU-only / CI machines.

The Serve deployment (see ``deployments/model_deployment.py``) owns one backend
instance per replica and is responsible for GPU placement, batching and
latency accounting.
"""

from __future__ import annotations

import abc
import hashlib
import json
import time
from collections.abc import Iterator
from typing import Any

from ray_serve_autoscale.settings import ModelConfig


class ModelBackend(abc.ABC):
    """Interface implemented by every model backend."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def load(self) -> None:
        """Load weights onto the device. Called once per replica at startup."""

    @abc.abstractmethod
    def predict(self, inputs: list[str]) -> list[Any]:
        """Run inference over a batch of inputs and return one result each."""

    def predict_stream(self, text: str) -> Iterator[str]:
        """Yield the response incrementally.

        The default implementation runs the ordinary batched path and chunks
        its output, so every backend gets a working streaming endpoint. A
        backend able to emit real incremental tokens should override this --
        the win of streaming is time-to-first-token, which chunking after the
        fact does not deliver.
        """
        result = self.predict([text])[0]
        text_out = result if isinstance(result, str) else json.dumps(result)
        for i in range(0, len(text_out), 32):
            yield text_out[i : i + 32]

    @property
    def device(self) -> str:
        return getattr(self, "_device", "cpu")


class SimulatedBackend(ModelBackend):
    """Deterministic, dependency-free stand-in for a real model.

    It sleeps for the configured compute time (so latency-based autoscaling has
    something real to react to) and returns a stable, task-shaped payload
    derived from a hash of the input. This lets the *entire* system -- routing,
    batching, metrics, autoscaling -- be exercised end-to-end without a GPU.
    """

    def load(self) -> None:
        self._device = "cpu(simulated)"

    def predict(self, inputs: list[str]) -> list[Any]:
        # Simulate GPU compute time; scales mildly with batch size to mimic
        # the sub-linear speedup of real batched inference on an accelerator.
        batch = max(len(inputs), 1)
        compute_s = self.config.simulated_latency_s * (1.0 + 0.15 * (batch - 1))
        time.sleep(compute_s)
        return [self._fake_result(text) for text in inputs]

    def predict_stream(self, text: str) -> Iterator[str]:
        """Emit real incremental chunks, paced like token generation."""
        result = self._fake_result(text)
        if self.config.task == "summarization":
            pieces = result["summary_text"].split()
        else:
            pieces = json.dumps(result).split()
        # Per-token pacing, derived from the configured compute time so a
        # streaming client sees realistic inter-token gaps.
        per_token = self.config.simulated_latency_s / max(len(pieces), 1)
        for piece in pieces:
            time.sleep(per_token)
            yield piece + " "

    def _fake_result(self, text: str) -> Any:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        task = self.config.task
        if task == "sentiment":
            label = "POSITIVE" if seed % 2 == 0 else "NEGATIVE"
            score = 0.5 + (seed % 1000) / 2000.0
            return {"label": label, "score": round(score, 4)}
        if task == "summarization":
            words = text.split()
            summary = " ".join(words[:12]) + ("..." if len(words) > 12 else "")
            return {"summary_text": summary or "(empty)"}
        if task == "embedding":
            # Deterministic pseudo-embedding, unit-length-ish, 16 dims.
            vec = [((seed >> (i * 2)) & 0xFF) / 255.0 for i in range(16)]
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            return {"embedding": [round(v / norm, 6) for v in vec], "dim": 16}
        return {"raw": digest[:16]}


class TransformersBackend(ModelBackend):
    """Real GPU-backed backend using HuggingFace ``transformers``.

    Only imported/instantiated when ``backend=transformers`` and the ML extras
    are installed. Placement onto the GPU is handled by Ray Serve via
    ``ray_actor_options={"num_gpus": ...}`` -- inside the replica CUDA device 0
    is the fraction Ray assigned to this actor.
    """

    def load(self) -> None:
        import torch  # noqa: F401  (import here so CPU/CI paths never need it)

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        task = self.config.task
        model_id = self.config.hf_model_id

        if task == "embedding":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_id, device=self._device)
            self._kind = "sentence-transformer"
            return

        from transformers import pipeline

        hf_task = {"sentiment": "sentiment-analysis", "summarization": "summarization"}[task]
        device_index = 0 if self._device == "cuda" else -1
        self._pipe = pipeline(hf_task, model=model_id, device=device_index)
        self._kind = "pipeline"

    def predict(self, inputs: list[str]) -> list[Any]:
        if self._kind == "sentence-transformer":
            vectors = self._model.encode(inputs, convert_to_numpy=True)
            return [{"embedding": v.tolist(), "dim": int(v.shape[0])} for v in vectors]
        outputs = self._pipe(inputs)
        # transformers returns a dict for single input, list for batches.
        if isinstance(outputs, dict):
            outputs = [outputs]
        return list(outputs)


def build_backend(config: ModelConfig, backend: str) -> ModelBackend:
    """Factory that returns the appropriate backend, with graceful fallback.

    If ``transformers`` is requested but the ML dependencies are missing, we
    fall back to the simulated backend rather than crashing the replica -- the
    system stays up and the log line makes the downgrade obvious.
    """
    if backend == "transformers":
        try:
            import torch  # noqa: F401

            return TransformersBackend(config)
        except Exception:  # pragma: no cover - depends on optional deps
            import logging

            logging.getLogger(__name__).warning(
                "transformers backend requested but ML deps unavailable; "
                "falling back to simulated backend for model '%s'",
                config.name,
            )
    return SimulatedBackend(config)
