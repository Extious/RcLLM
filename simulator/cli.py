#!/usr/bin/env python3
"""Command-line entry point for the RcLLM+ calibrated cluster simulator."""

import argparse
from collections import Counter
import json

import numpy as np

from rcllm.config import MODEL_SPECS
from simulator.cluster import ClusterSimulator, RequestTraceRow
from simulator.placement import build_metis_or_hash_placement


def make_synthetic_trace(num_rows: int = 256, catalog_size: int = 2000, seed: int = 42):
    """Generate a heavy-tailed Zipfian recommendation trace for quick smoke runs."""
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, catalog_size + 1, dtype=np.float64)
    probs = ranks ** (-1.15)
    probs /= probs.sum()
    popularity = Counter()
    rows = []
    for _ in range(num_rows):
        chosen = rng.choice(catalog_size, size=20, replace=False, p=probs)
        items = tuple((f"item_{int(idx)}", int(rng.integers(140, 220))) for idx in chosen)
        for item_id, _ in items:
            popularity[item_id] += 1
        history_tokens = int(rng.integers(900, 1500))
        prefix_tokens = 128
        seq_len = prefix_tokens + history_tokens + sum(cnt for _, cnt in items) + 96
        rows.append(RequestTraceRow(seq_len=seq_len, prefix_tokens=prefix_tokens, history_tokens=history_tokens, items=items))
    return rows, dict(popularity)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RcLLM+ calibrated cluster simulation.")
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), default="qwen3")
    parser.add_argument("--replicas", type=int, default=40)
    parser.add_argument("--qps", type=float, default=40.0)
    parser.add_argument("--route", choices=("round-robin", "affinity", "hit-only", "load-only"), default="affinity")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows, popularity = make_synthetic_trace(seed=args.seed)
    placement = build_metis_or_hash_placement(popularity, replicas=args.replicas)
    sim = ClusterSimulator(args.model, rows, placement)

    summary = {}
    for method in ("Full", "Prefix-Cache", "CacheBlend", "RcLLM", "RcLLM+"):
        res = sim.replay(
            method,
            args.qps,
            route=args.route,
            num_requests=args.requests,
            warmup=args.warmup,
            seed=args.seed,
        )
        summary[method] = {
            "mean_ttft_ms": round(res.mean_ttft_s * 1000.0, 2),
            "p50_ttft_ms": round(res.p50_ttft_s * 1000.0, 2),
            "p99_ttft_ms": round(res.p99_ttft_s * 1000.0, 2),
            "mean_hit_ratio": round(res.mean_hit_ratio, 4),
        }
    print(json.dumps({"model": args.model, "replicas": args.replicas, "qps": args.qps, "route": args.route, "results": summary}, indent=2))


if __name__ == "__main__":
    main()
