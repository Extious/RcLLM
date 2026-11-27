#!/usr/bin/env python3
"""Recommendation ranking metrics and accuracy evaluation helper for RcLLM+."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Sequence

import numpy as np


def ranking_metrics(scores: Sequence[float], target_index: int) -> Dict[str, float]:
    """Compute HR@1/5/10, NDCG@5/10/20, and MRR for a 20-candidate recommendation list."""
    arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-arr, kind="stable")
    rank = int(np.flatnonzero(order == target_index)[0]) + 1
    return {
        "target_rank": float(rank),
        "hr@1": 1.0 if rank <= 1 else 0.0,
        "hr@5": 1.0 if rank <= 5 else 0.0,
        "hr@10": 1.0 if rank <= 10 else 0.0,
        "ndcg@5": (1.0 / math.log2(rank + 1)) if rank <= 5 else 0.0,
        "ndcg@10": (1.0 / math.log2(rank + 1)) if rank <= 10 else 0.0,
        "ndcg@20": (1.0 / math.log2(rank + 1)) if rank <= 20 else 0.0,
        "mrr": 1.0 / rank,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate candidate score JSONL receipts.")
    parser.add_argument("receipts", type=Path, help="JSONL file with fields `scores` and `target`.")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.receipts.read_text().splitlines() if line.strip()]
    metrics = [ranking_metrics(r["scores"], int(r["target"])) for r in rows]
    summary = {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]}
    print(json.dumps({"count": len(metrics), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
