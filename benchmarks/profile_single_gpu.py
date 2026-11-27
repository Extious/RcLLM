#!/usr/bin/env python3
"""Single-GPU 6-stage TTFT profiler for RcLLM+ with optional layer-wise KV prefetch overlap."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rcllm.adapter import load_decoder
from rcllm.cache import pin_kv_cache
from rcllm.config import MODEL_SPECS, PromptSpec, RegionSpan
from rcllm.engine import RcLLMEngine


def load_prompt(path: Path) -> PromptSpec:
    raw = json.loads(path.read_text())
    regions = [
        RegionSpan(role=r["role"], identity=str(r.get("identity", "")), start=int(r["start"]), end=int(r["end"]))
        for r in raw["reuse_regions"]
    ]
    return PromptSpec(
        input_ids=raw["input_ids"],
        reuse_regions=regions,
        alias_token_ids=raw["alias_token_ids"],
        dataset=raw.get("dataset", "amazon"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile single-GPU TTFT across prefill methods.")
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), default="qwen3")
    parser.add_argument("--model-path", required=True, help="Local path or Hugging Face model ID.")
    parser.add_argument("--prompt-json", type=Path, required=True, help="PromptSpec JSON file.")
    parser.add_argument("--method", choices=("full", "rcllm_plus", "rcllm_conf", "cacheblend", "epic", "reuse"), default="rcllm_plus")
    parser.add_argument("--rate", type=int, default=None, help="Recomputation percentage (defaults to model spec).")
    parser.add_argument("--cutoff", type=int, default=None, help="Depth cutoff layer L_c (defaults to model spec).")
    parser.add_argument("--overlap", action="store_true", help="Enable LayerKVPrefetch on a dedicated CUDA stream.")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model]
    adapter = load_decoder(args.model_path)
    engine = RcLLMEngine(adapter)
    prompt = load_prompt(args.prompt_json)

    device_cache = adapter.encode_reuse_regions(prompt)
    host_cache = pin_kv_cache(device_cache)
    del device_cache
    torch.cuda.empty_cache()

    stream = torch.cuda.Stream(adapter.device) if args.overlap else None
    rate = args.rate if args.rate is not None else spec.iso_recompute_rate
    cutoff = args.cutoff if args.cutoff is not None else spec.default_cutoff

    # Warmup
    cache_arg = host_cache if args.overlap else [(k.to(adapter.device), v.to(adapter.device)) for k, v in host_cache]
    engine.prefill(prompt, cache_arg, method=args.method, recompute_rate_pct=rate, cutoff_layer=cutoff, overlap_stream=stream)

    totals = []
    stage_records = []
    for _ in range(args.repeats):
        cache_arg = host_cache if args.overlap else [(k.to(adapter.device), v.to(adapter.device)) for k, v in host_cache]
        res = engine.prefill(
            prompt,
            cache_arg,
            method=args.method,
            recompute_rate_pct=rate,
            cutoff_layer=cutoff,
            overlap_stream=stream,
            profile_stages=True,
        )
        totals.append(res.total_ms)
        stage_records.append(res.stage_ms)

    median_stages = {k: round(float(np.median([r[k] for r in stage_records])), 3) for k in stage_records[0]}
    print(json.dumps({
        "model": args.model,
        "method": args.method,
        "rate": rate,
        "cutoff": cutoff,
        "overlap": args.overlap,
        "median_ttft_ms": round(float(np.median(totals)), 3),
        "median_stages_ms": median_stages,
    }, indent=2))


if __name__ == "__main__":
    main()
