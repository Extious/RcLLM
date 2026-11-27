# RcLLM+: Beyond-Prefix KV Caching with Depth-Bounded Selective Recomputation for Generative Recommendation

Official open-source implementation of **RcLLM+** (*IEEE Transactions on Parallel and Distributed Systems*, extending our ICDCS'26 paper *RcLLM*).

## Overview

Large Language Models (LLMs) turn recommendation into a generative ranking task, but long personalized prompts create a severe prefill bottleneck with minimal prefix sharing across requests.

**RcLLM+** combines two complementary layers:
1. **Beyond-Prefix Block KV Caching (ICDCS'26 RcLLM)**: Decomposes prompts into reusable semantic segments (user review histories and candidate items), stores pre-RoPE KV caches per block, replicates semantic histories and hot items (`0.1%` hot set), shards cold item caches via METIS co-occurrence partitioning, and routes requests with cache-affinity scheduling.
2. **Rank-Normalized Selection & Depth-Bounded Selective Recomputation (TPDS RcLLM+)**:
   - After dense layers `0..1`, ranks cached tokens by equal-weight average-rank normalization of layer-0 causal received attention and layer-1 K/V L2 deviations (implemented with 3 fused Triton CUDA kernels).
   - Stops refreshing cached tokens after depth cutoff layer $L_c$, spending fewer FLOPs in deep layers to refresh more tokens in shallow layers.
   - Overlaps pinned-host KV transfers with early-layer computation via layer-ordered `LayerKVPrefetch` on a dedicated CUDA stream.

## Repository Layout

```text
RcLLM/
├── rcllm/                     # Core RcLLM+ inference & selective recomputation engine (no transformers fork needed)
│   ├── config.py              # ModelSpec (Qwen3-8B, Mistral-7B-v0.3, GLM-4-9B) & PromptSpec
│   ├── adapter.py             # Model-agnostic HF decoder adapter (QK-Norm, GLM-4 partial RoPE & post-norms)
│   ├── cache.py               # Pre-RoPE block KV caching & LayerKVPrefetch CUDA stream overlap
│   ├── selector.py            # Average-rank selector (NumPy reference + 3-kernel fused Triton GPU kernels)
│   └── engine.py              # Depth-bounded selective recomputation engine (RcLLM+, RcLLM, CacheBlend, EPIC, Full)
├── simulator/                 # Final TPDS calibrated multi-replica cluster simulator
│   ├── csrc/fcfs.cpp          # C++17 nonpreemptive FCFS & affinity-routing event replay kernel
│   ├── kernel.py              # Auto-compiler and ctypes binding for fcfs.cpp
│   ├── placement.py           # METIS/hash item sharding & hot-item replication
│   ├── calibration.py         # 6-stage NNLS latency model & analytical decoder FLOP estimator
│   ├── cluster.py             # Multi-replica TTFT CDF replay & P99 SLO capacity search
│   └── cli.py                 # CLI driver (`rcllm-sim`)
├── benchmarks/                # Evaluation and profiling entry points
│   ├── profile_single_gpu.py  # Single-GPU 6-stage TTFT profiler with optional KV prefetch overlap
│   └── eval_accuracy.py       # Recommendation ranking metrics (NDCG@5/10/20, HR@1/5/10, MRR)
└── tests/                     # CPU-runnable unit tests for selector, engine, and C++17 simulator
```

## Installation

```bash
# Core package + C++17 cluster simulator (requires g++ with C++17)
pip install -e .

# With GPU inference & Triton selector dependencies
pip install -e ".[gpu,dev]"
```

## Quick Start

### 1. Run the Calibrated Cluster Simulator (CPU, C++17)

```bash
rcllm-sim --model qwen3 --replicas 40 --qps 40 --route affinity
```

Or programmatically:

```python
from simulator import ClusterSimulator, build_metis_or_hash_placement
from simulator.cli import make_synthetic_trace

rows, popularity = make_synthetic_trace(num_rows=256)
placement = build_metis_or_hash_placement(popularity, replicas=40, hot_fraction=0.001)
sim = ClusterSimulator("qwen3", rows, placement)
result = sim.replay("RcLLM+", qps=40.0, route="affinity")
print(f"P50={result.p50_ttft_s*1000:.1f} ms, P99={result.p99_ttft_s*1000:.1f} ms")
```

### 2. Single-GPU Depth-Bounded Selective Recomputation

```python
import torch
from rcllm.adapter import load_decoder
from rcllm.cache import pin_kv_cache
from rcllm.engine import RcLLMEngine

adapter = load_decoder("Qwen/Qwen3-8B")
engine = RcLLMEngine(adapter)
# `prompt` is a PromptSpec with input_ids, reuse_regions, and alias_token_ids
host_kv = pin_kv_cache(adapter.encode_reuse_regions(prompt))
stream = torch.cuda.Stream(adapter.device)

res = engine.prefill(
    prompt,
    host_kv,
    method="rcllm_plus",
    recompute_rate_pct=7,   # near-iso-accuracy setting for Qwen3-8B
    cutoff_layer=25,
    overlap_stream=stream,
    profile_stages=True,
)
print(res.total_ms, res.stage_ms)
```

### 3. Run Tests

```bash
pytest -v
```

## Evaluated Models & Checkpoints

| Model | Hugging Face Repo | Revision | Layers | Default $L_c$ |
| --- | --- | --- | ---: | ---: |
| Qwen3-8B | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | 36 | 25 |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | 32 | 20 |
| GLM-4-9B-0414 | `zai-org/GLM-4-9B-0414` | `645b8482494e31b6b752272bf7f7f273ef0f3caf` | 40 | 25 |

## Branches

- `main`: Complete **RcLLM+** open-source system (inference engine, Triton selector, layer-wise KV prefetch, and final TPDS C++17 cluster simulator).
- `old`: Archived preliminary prototype (2025-05).

## Citation

If you find this repository useful, please cite our papers:

```bibtex
@inproceedings{zhao2026rcllm,
  title     = {RcLLM: Enabling Low-Latency LLM-based Generative Recommendation via Beyond-Prefix KV Caching},
  author    = {Zhao, Zhan and Wang, Yuxin and Zheng, Long and Zhou, Amelie Chi},
  booktitle = {Proceedings of the IEEE International Conference on Distributed Computing Systems (ICDCS)},
  pages     = {642--652},
  year      = {2026},
  doi       = {10.1109/2575-8411.2026.00067}
}
```
