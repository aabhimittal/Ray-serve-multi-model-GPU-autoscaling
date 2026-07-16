#!/usr/bin/env python3
"""Async load generator to drive latency and watch autoscaling react.

Ramps concurrent request load against one or more models, then prints the
gateway's rolling latency snapshot so you can watch p95 climb toward the SLO
and replicas scale out in response.

Examples
--------
    # Steady 50 concurrent clients against the sentiment model for 60s:
    python scripts/load_test.py --model sentiment --concurrency 50 --duration 60

    # Ramp load in stages to observe scale-out then scale-in:
    python scripts/load_test.py --model summarization --ramp 5,20,50,10 --stage-seconds 30
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time

import aiohttp

SAMPLE_TEXTS = [
    "This product exceeded my expectations and I love it.",
    "Terrible experience, would not recommend to anyone.",
    "The service was okay, nothing special but not bad either.",
    "Ray Serve makes scaling GPU models across a cluster straightforward.",
    "The quarterly report highlighted strong growth in cloud revenue and margins.",
]


async def _one_request(session: aiohttp.ClientSession, url: str, payload: dict) -> float:
    start = time.perf_counter()
    async with session.post(url, json=payload) as resp:
        await resp.read()
    return (time.perf_counter() - start) * 1000.0


async def _client_worker(session, url, stop_at, latencies, errors):
    while time.perf_counter() < stop_at:
        payload = {"text": random.choice(SAMPLE_TEXTS)}
        try:
            latencies.append(await _one_request(session, url, payload))
        except Exception:
            errors[0] += 1


async def _run_stage(base_url, model, concurrency, seconds):
    url = f"{base_url.rstrip('/')}/predict/{model}"
    latencies: list[float] = []
    errors = [0]
    stop_at = time.perf_counter() + seconds
    async with aiohttp.ClientSession() as session:
        workers = [
            asyncio.create_task(_client_worker(session, url, stop_at, latencies, errors))
            for _ in range(concurrency)
        ]
        await asyncio.gather(*workers)
    return latencies, errors[0]


def _report(model, concurrency, seconds, latencies, errors):
    n = len(latencies)
    if n == 0:
        print(f"[{model}] c={concurrency:<4} no successful requests ({errors} errors)")
        return
    latencies.sort()
    p50 = latencies[int(0.50 * (n - 1))]
    p95 = latencies[int(0.95 * (n - 1))]
    p99 = latencies[int(0.99 * (n - 1))]
    rps = n / seconds
    print(
        f"[{model}] c={concurrency:<4} reqs={n:<6} rps={rps:7.1f} "
        f"p50={p50:7.1f}ms p95={p95:7.1f}ms p99={p99:7.1f}ms "
        f"mean={statistics.mean(latencies):7.1f}ms errors={errors}"
    )


async def _poll_gateway(base_url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url.rstrip('/')}/metrics/latency", timeout=5) as r:
                data = await r.json()
        print("  gateway latency snapshot:")
        for name, s in data.get("latency", {}).items():
            print(
                f"    {name:<14} count={s['count']:<5} p95={s['p95_ms']:>7.1f}ms "
                f"slo={s['slo_ms']:>6.0f}ms rps={s['rps']:.1f}"
            )
    except Exception as exc:
        print(f"  (could not read gateway metrics: {exc})")


async def main_async(args):
    stages = (
        [int(x) for x in args.ramp.split(",")] if args.ramp else [args.concurrency]
    )
    seconds = args.stage_seconds if args.ramp else args.duration
    print(f"Load testing model='{args.model}' at {args.base_url}")
    for concurrency in stages:
        latencies, errors = await _run_stage(args.base_url, args.model, concurrency, seconds)
        _report(args.model, concurrency, seconds, latencies, errors)
        await _poll_gateway(args.base_url)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="sentiment")
    p.add_argument("--concurrency", type=int, default=25)
    p.add_argument("--duration", type=int, default=30, help="seconds (single stage)")
    p.add_argument("--ramp", help="comma-separated concurrency stages, e.g. 5,20,50,10")
    p.add_argument("--stage-seconds", type=int, default=30, help="seconds per ramp stage")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
