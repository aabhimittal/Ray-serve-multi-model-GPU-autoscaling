#!/usr/bin/env python3
"""Find each model's throughput/latency knee and recommend autoscaling config.

`target_ongoing_requests` is the single most consequential autoscaling knob,
and it is almost always guessed. It sets how deep each replica's queue is
allowed to get, which by Little's Law fixes the latency/throughput trade-off:
too low wastes GPUs, too high blows the SLO no matter how many replicas the
autoscaler adds.

This harness measures the curve instead of guessing it. It sweeps concurrency
against a live gateway, records the p95 latency and throughput at each level,
and reports:

* the **saturation knee** -- where added concurrency stops buying throughput
  and only buys queueing;
* the highest concurrency that still meets the SLO;
* a recommended ``target_ongoing_requests`` derived from that point.

Run it once per model per GPU type; paste the recommendations into
``config/models.yaml``.

    python scripts/gpu_benchmark.py --model sentiment --slo-ms 150
    python scripts/gpu_benchmark.py --all --output results/benchmark.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp

SAMPLE_TEXT = (
    "Ray Serve makes it straightforward to compose multiple models behind one "
    "endpoint and scale each of them independently on GPU."
)


async def _worker(session, url, stop_at, latencies, errors):
    while time.perf_counter() < stop_at:
        start = time.perf_counter()
        try:
            async with session.post(url, json={"text": SAMPLE_TEXT}) as resp:
                await resp.read()
                if resp.status >= 400:
                    errors[0] += 1
                    continue
            latencies.append((time.perf_counter() - start) * 1000.0)
        except Exception:
            errors[0] += 1


async def _measure(base_url: str, model: str, concurrency: int, seconds: float) -> dict:
    url = f"{base_url.rstrip('/')}/predict/{model}"
    latencies: list = []
    errors = [0]
    connector = aiohttp.TCPConnector(limit=max(concurrency * 2, 16))
    stop_at = time.perf_counter() + seconds
    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(
            *[_worker(session, url, stop_at, latencies, errors) for _ in range(concurrency)]
        )

    if not latencies:
        return {"concurrency": concurrency, "requests": 0, "errors": errors[0]}

    latencies.sort()
    n = len(latencies)
    return {
        "concurrency": concurrency,
        "requests": n,
        "errors": errors[0],
        "rps": round(n / seconds, 2),
        "p50_ms": round(latencies[int(0.50 * (n - 1))], 2),
        "p95_ms": round(latencies[int(0.95 * (n - 1))], 2),
        "p99_ms": round(latencies[int(0.99 * (n - 1))], 2),
        "mean_ms": round(statistics.mean(latencies), 2),
    }


def _analyze(points: list, slo_ms: float) -> dict:
    """Locate the saturation knee and derive an autoscaling recommendation."""
    usable = [p for p in points if p.get("requests")]
    if not usable:
        return {"error": "no successful requests"}

    peak_rps = max(p["rps"] for p in usable)
    # The knee is the lowest concurrency reaching ~95% of peak throughput.
    # Past it, extra concurrency converts directly into queueing delay.
    knee = min(
        (p for p in usable if p["rps"] >= 0.95 * peak_rps),
        key=lambda p: p["concurrency"],
    )
    within_slo = [p for p in usable if slo_ms <= 0 or p["p95_ms"] <= slo_ms]
    best_slo = max(within_slo, key=lambda p: p["concurrency"]) if within_slo else None

    # Respect both limits: never queue deeper than throughput rewards, and
    # never deeper than the SLO tolerates.
    if best_slo is None:
        recommended = max(knee["concurrency"] // 2, 1)
        note = (
            f"SLO {slo_ms:.0f}ms not met at any concurrency "
            f"(best p95 {min(p['p95_ms'] for p in usable):.0f}ms) -- "
            "the model is too slow for this SLO on this hardware; "
            "use a smaller model, a faster GPU, or relax the SLO."
        )
    else:
        recommended = max(min(knee["concurrency"], best_slo["concurrency"]), 1)
        note = "knee and SLO limit agree" if knee["concurrency"] <= best_slo["concurrency"] else (
            "SLO binds before saturation: latency, not throughput, is the limit"
        )

    return {
        "peak_rps": peak_rps,
        "knee_concurrency": knee["concurrency"],
        "knee_p95_ms": knee["p95_ms"],
        "max_concurrency_within_slo": best_slo["concurrency"] if best_slo else None,
        "recommended_target_ongoing_requests": recommended,
        "recommended_max_ongoing_requests": recommended * 2,
        "note": note,
    }


async def _run_model(base_url, model, levels, seconds, slo_ms) -> dict:
    print(f"\n=== {model} (SLO {slo_ms:.0f}ms) ===")
    print(f"{'conc':>6}{'rps':>10}{'p50':>10}{'p95':>10}{'p99':>10}{'err':>6}")
    points = []
    for concurrency in levels:
        point = await _measure(base_url, model, concurrency, seconds)
        points.append(point)
        if point.get("requests"):
            flag = "  <-- over SLO" if slo_ms > 0 and point["p95_ms"] > slo_ms else ""
            print(
                f"{concurrency:>6}{point['rps']:>10.1f}{point['p50_ms']:>10.1f}"
                f"{point['p95_ms']:>10.1f}{point['p99_ms']:>10.1f}{point['errors']:>6}{flag}"
            )
        else:
            print(f"{concurrency:>6}{'-':>10}{'-':>10}{'-':>10}{'-':>10}{point['errors']:>6}")
        # Let queues drain so the next level starts from a clean state.
        await asyncio.sleep(2.0)

    analysis = _analyze(points, slo_ms)
    print(f"\n  peak throughput      : {analysis.get('peak_rps', 0):.1f} rps")
    print(f"  saturation knee      : concurrency {analysis.get('knee_concurrency')}")
    print(f"  max concurrency @SLO : {analysis.get('max_concurrency_within_slo')}")
    print(f"  -> target_ongoing_requests: {analysis.get('recommended_target_ongoing_requests')}")
    print(f"  -> max_ongoing_requests   : {analysis.get('recommended_max_ongoing_requests')}")
    print(f"  note: {analysis.get('note')}")
    return {"model": model, "slo_ms": slo_ms, "points": points, "analysis": analysis}


async def main_async(args) -> None:
    levels = [int(x) for x in args.levels.split(",")]

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{args.base_url.rstrip('/')}/models") as resp:
            catalog = (await resp.json())["models"]

    targets = list(catalog) if args.all else [args.model]
    results = []
    for model in targets:
        slo = args.slo_ms if args.slo_ms > 0 else catalog.get(model, {}).get("latency_slo_ms", 0.0)
        results.append(await _run_model(args.base_url, model, levels, args.seconds, slo))

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="sentiment")
    p.add_argument("--all", action="store_true", help="benchmark every deployed model")
    p.add_argument("--levels", default="1,2,4,8,16,32,64", help="concurrency sweep")
    p.add_argument("--seconds", type=float, default=15.0, help="seconds per level")
    p.add_argument("--slo-ms", type=float, default=0.0, help="override the model's SLO")
    p.add_argument("--output", help="write full results as JSON")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
