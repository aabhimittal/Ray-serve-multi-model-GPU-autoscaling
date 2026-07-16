#!/usr/bin/env python3
"""Convenience launcher for local development.

Starts a local Ray instance, deploys the multi-model app, sends a couple of
smoke requests so you can confirm it works, and (unless --no-autoscaler) starts
the latency autoscaler in dry-run mode so you can watch its decisions without a
GPU cluster.

    python scripts/run_local.py                 # deploy + smoke test, blocking
    python scripts/run_local.py --no-block      # deploy, smoke test, exit
"""

from __future__ import annotations

import argparse
import time

import ray
import requests
from ray import serve

from ray_serve_autoscale.app import build_app
from ray_serve_autoscale.settings import load_settings


def smoke_test(base_url: str) -> None:
    print("\n--- smoke test ---")
    print("GET /models:", requests.get(f"{base_url}/models", timeout=10).status_code)
    for model, text in [
        ("sentiment", "I really love how easy this was to deploy!"),
        ("summarization", "Ray Serve lets you compose multiple models behind one endpoint."),
        ("embedding", "vectorize me"),
    ]:
        r = requests.post(f"{base_url}/predict/{model}", json={"text": text}, timeout=30)
        body = r.json()
        print(f"POST /predict/{model}: {r.status_code} latency={body.get('latency_ms')}ms")
    print("GET /metrics/latency:", requests.get(f"{base_url}/metrics/latency", timeout=10).json())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--no-block", action="store_true")
    p.add_argument("--no-autoscaler", action="store_true")
    args = p.parse_args()

    settings = load_settings(args.config)
    ray.init(ignore_reinit_error=True)
    serve.start(http_options={"host": settings.http_host, "port": settings.http_port})
    serve.run(build_app(settings), route_prefix=settings.route_prefix, name="multi_model_gateway")

    base_url = f"http://127.0.0.1:{settings.http_port}"
    print(f"Deployed. Gateway at {base_url}")
    time.sleep(1.0)
    smoke_test(base_url)

    if not args.no_autoscaler and settings.autoscaler.enabled:
        from ray_serve_autoscale.autoscaling.latency_autoscaler import (
            LatencyAutoscaler,
            LoggingApplier,
        )

        def reader():
            return requests.get(f"{base_url}/metrics/latency", timeout=5).json().get("latency", {})

        controller = LatencyAutoscaler(
            models=settings.models,
            cfg=settings.autoscaler,
            reader=reader,
            applier=LoggingApplier(),
        )
        print("\n--- autoscaler (dry-run) decisions ---")
        for _ in range(3):
            for d in controller.step():
                print(f"  {d.name}: {d.reason}")
            time.sleep(2)

    if not args.no_block:
        print("\nBlocking. Ctrl-C to shut down. "
              f"Try: python scripts/load_test.py --model sentiment --concurrency 40")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    serve.shutdown()


if __name__ == "__main__":
    main()
