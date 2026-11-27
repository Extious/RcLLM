"""Depth-bounded selective recomputation engine for RcLLM+ and baselines.

Implements:
  - `rcllm_plus`: Rank-normalized selector (layer-0 received attention + layer-1 K/V
    L2 deviation) after layer 1, with depth-bounded cutoff at layer `L_c`.
  - `rcllm_conf`: ICDCS'26 conference selector (layer-0 attention + review KV drift)
    reconstructed on shared exact-text caches.
  - `cacheblend`: Layer-1 KV L2 deviation selector without depth cutoff.
  - `epic`: Fixed leading-token (LegoLink-16) recomputation per cached chunk.
  - `full`: Dense full prefill baseline.
  - `reuse`: Protected-only zero-recomputation lower bound.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from rcllm.adapter import DecoderAdapter
from rcllm.cache import LayerKVPrefetch
from rcllm.config import CONFERENCE_DEFAULTS, PromptSpec
from rcllm.selector import HAS_TRITON, received_attention_triton, select_tokens_gpu, select_tokens_numpy


@dataclass
class PrefillResult:
    """Outputs and diagnostics from a single prefill execution."""

    candidate_scores: np.ndarray
    active_counts_per_layer: List[int]
    selected_indices: np.ndarray
    total_ms: float
    stage_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def token_layers(self) -> int:
        return sum(self.active_counts_per_layer)


def received_attention_torch(q: torch.Tensor, keys: torch.Tensor, num_kv_heads: int, chunk: int = 64) -> torch.Tensor:
    """Reference PyTorch causal received attention mass per token (FP32 softmax)."""
    _, num_q_heads, seq_len, head_dim = q.shape
    rep = num_q_heads // num_kv_heads
    expanded = keys.repeat_interleave(rep, dim=1)
    mass = torch.zeros(seq_len, device=q.device, dtype=torch.float32)
    axis = torch.arange(seq_len, device=q.device)
    scale = 1.0 / math.sqrt(head_dim)
    for start in range(0, seq_len, chunk):
        end = min(seq_len, start + chunk)
        pos = axis[start:end]
        logits = (q[:, :, start:end] @ expanded.transpose(-1, -2)).float() * scale
        allowed = axis[None, :] <= pos[:, None]
        mass += torch.softmax(logits.masked_fill(~allowed, -torch.inf), dim=-1).sum((0, 1, 2))
    return mass


def kv_l2_deviations(
    proj_k: torch.Tensor,
    proj_v: torch.Tensor,
    cached_kv: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token L2 deviation between freshly projected K/V and cached pre-RoPE K/V."""
    ck, cv = cached_kv
    dk = (proj_k.float() - ck.float()).square().sum((0, 1, 3)).sqrt()
    dv = (proj_v.float() - cv.float()).square().sum((0, 1, 3)).sqrt()
    return dk, dv


class RcLLMEngine:
    """Unified prefill engine supporting RcLLM+ and comparison baselines."""

    def __init__(self, adapter: DecoderAdapter):
        self.adapter = adapter

    @torch.inference_mode()
    def prefill(
        self,
        prompt: PromptSpec,
        cache=None,
        *,
        method: str = "rcllm_plus",
        recompute_rate_pct: int = 22,
        cutoff_layer: Optional[int] = None,
        epic_prefix_tokens: int = 16,
        use_triton: bool = True,
        overlap_stream=None,
        profile_stages: bool = False,
    ) -> PrefillResult:
        """Execute prefill and return candidate scores plus per-layer active token counts."""
        ad = self.adapter
        num_layers = ad.num_layers
        seq_len = prompt.seq_len
        is_cuda = ad.device.type == "cuda"

        if is_cuda:
            torch.cuda.synchronize(ad.device)
        t_start = t_last = time.perf_counter()
        stages: Dict[str, float] = {}

        def mark(name: str) -> None:
            nonlocal t_last
            if profile_stages:
                if is_cuda:
                    torch.cuda.synchronize(ad.device)
                now = time.perf_counter()
                stages[name] = (now - t_last) * 1000.0
                t_last = now

        # 1. Prepare KV cache (optionally overlapped via LayerKVPrefetch on a CUDA stream).
        if method == "full":
            active_cache = None
        elif overlap_stream is not None and is_cuda:
            first_needed = 1 if method in ("rcllm_plus", "cacheblend") else 0
            active_cache = LayerKVPrefetch(cache, ad.device, overlap_stream, first_needed=first_needed)
        else:
            active_cache = cache
        mark("kv_load")

        all_ids = torch.arange(seq_len, device=ad.device, dtype=torch.long)
        input_tensor = torch.tensor([list(prompt.input_ids)], device=ad.device, dtype=torch.long)
        hidden = ad.embed(input_tensor)
        rotary = ad.rotary(hidden)
        active_counts: List[int] = []

        eligible_list, protected_list = prompt.eligible_and_protected()
        eligible_np = np.asarray(eligible_list, dtype=np.int64)
        protected_np = np.asarray(protected_list, dtype=np.int64)

        # 2. Early dense layers (layer 0 for conference/epic, layers 0..1 for rcllm_plus/cacheblend).
        layer0_qk = None
        layer0_drift = None
        layer1_kv = None

        if method == "full":
            hidden = ad.run_segment(0, num_layers - 1, hidden, all_ids, None, rotary)
            active_counts = [seq_len] * num_layers
            mark("layers_0_1")
            mark("selection")
            mark("middle")
            mark("tail")
            scores = ad.score_candidates(hidden, prompt.alias_token_ids).cpu().numpy()
            mark("output")
            if is_cuda:
                torch.cuda.synchronize(ad.device)
            return PrefillResult(scores, active_counts, np.arange(seq_len, dtype=np.int64), (time.perf_counter() - t_start) * 1000.0, stages)

        if method == "reuse":
            selected_np = protected_np
            sel_t = torch.as_tensor(selected_np, device=ad.device, dtype=torch.long)
            hidden = hidden.index_select(1, sel_t)
            mark("layers_0_1")
            mark("selection")
            hidden = ad.run_segment(0, num_layers - 1, hidden, sel_t, active_cache, rotary)
            active_counts = [int(sel_t.numel())] * num_layers
            mark("middle")
            mark("tail")
            scores = ad.score_candidates(hidden, prompt.alias_token_ids).cpu().numpy()
            mark("output")
            if is_cuda:
                torch.cuda.synchronize(ad.device)
            return PrefillResult(scores, active_counts, selected_np, (time.perf_counter() - t_start) * 1000.0, stages)

        # Layer 0 dense forward (always dense for rcllm_plus, rcllm_conf, cacheblend, epic).
        z0 = ad.project_layer(0, hidden, all_ids, active_cache, rotary)
        if method in ("rcllm_plus", "rcllm_conf"):
            layer0_qk = (z0["q"], z0["keys"])
        if method == "rcllm_conf":
            ck0, cv0 = active_cache[0]
            layer0_drift = (z0["k"].float() - ck0.float()).abs().sum((0, 1, 3)) + (z0["v"].float() - cv0.float()).abs().sum((0, 1, 3))
        hidden = ad.finish_layer(0, hidden, all_ids, z0, rotary)
        active_counts.append(seq_len)
        del z0

        # Layer 1 dense forward for two-layer methods (rcllm_plus, cacheblend).
        start_sparse = 2 if method in ("rcllm_plus", "cacheblend") else 1
        if start_sparse == 2:
            z1 = ad.project_layer(1, hidden, all_ids, active_cache, rotary)
            layer1_kv = (z1["k"], z1["v"])
            hidden = ad.finish_layer(1, hidden, all_ids, z1, rotary)
            active_counts.append(seq_len)
            del z1
        mark("layers_0_1")

        # 3. Token selection.
        if method == "rcllm_plus":
            if use_triton and HAS_TRITON and is_cuda and hidden.dtype == torch.bfloat16:
                attn_mass = received_attention_triton(*layer0_qk)
                elig_t = torch.as_tensor(eligible_np, device=ad.device, dtype=torch.long)
                prot_t = torch.as_tensor(protected_np, device=ad.device, dtype=torch.long)
                sel_t = select_tokens_gpu(attn_mass, layer1_kv, active_cache[1], elig_t, prot_t, recompute_rate_pct)
                selected_np = sel_t.cpu().numpy()
            else:
                attn_mass = received_attention_torch(layer0_qk[0], layer0_qk[1], ad.num_kv_heads)
                dk, dv = kv_l2_deviations(layer1_kv[0], layer1_kv[1], active_cache[1])
                selected_np, _ = select_tokens_numpy(
                    attn_mass.cpu().numpy(),
                    dk.cpu().numpy(),
                    dv.cpu().numpy(),
                    eligible_np,
                    protected_np,
                    recompute_rate_pct,
                )
                sel_t = torch.as_tensor(selected_np, device=ad.device, dtype=torch.long)
        elif method == "cacheblend":
            dk, dv = kv_l2_deviations(layer1_kv[0], layer1_kv[1], active_cache[1])
            score = (dk.square() + dv.square()).sqrt().cpu().numpy()[eligible_np]
            keep = min(len(eligible_np), (recompute_rate_pct * len(eligible_np) + 99) // 100)
            chosen = eligible_np[np.argsort(-score, kind="stable")[:keep]]
            selected_np = np.sort(np.concatenate([protected_np, chosen]))
            sel_t = torch.as_tensor(selected_np, device=ad.device, dtype=torch.long)
        elif method == "epic":
            keep_set = set(int(x) for x in protected_np)
            for region in prompt.reuse_regions:
                if region.start > 0:
                    keep_set.update(range(region.start, min(region.start + epic_prefix_tokens, region.end)))
            selected_np = np.asarray(sorted(keep_set), dtype=np.int64)
            sel_t = torch.as_tensor(selected_np, device=ad.device, dtype=torch.long)
        elif method == "rcllm_conf":
            attn_mass = received_attention_torch(layer0_qk[0], layer0_qk[1], ad.num_kv_heads)
            roles = prompt.role_indices()
            r_rev, r_item, lam = CONFERENCE_DEFAULTS.get(prompt.dataset, (30, 30, 0.3))
            chosen_parts = [protected_np]
            for role, rate in (("review", r_rev), ("item", r_item)):
                pos = np.asarray(roles[role], dtype=np.int64)
                if len(pos) == 0:
                    continue
                pos_t = torch.as_tensor(pos, device=ad.device, dtype=torch.long)
                s_val = attn_mass.index_select(0, pos_t)
                if role == "review":
                    s_val = (1.0 - lam) * s_val + lam * layer0_drift.index_select(0, pos_t)
                keep = min(len(pos), (rate * len(pos) + 99) // 100)
                top_idx = torch.argsort(s_val, descending=True, stable=True)[:keep].cpu().numpy()
                chosen_parts.append(pos[top_idx])
            selected_np = np.sort(np.concatenate(chosen_parts))
            sel_t = torch.as_tensor(selected_np, device=ad.device, dtype=torch.long)
        else:
            raise ValueError(f"Unsupported method: {method}")
        mark("selection")

        # 4. Middle selective layers (start_sparse .. stop_layer - 1).
        stop_layer = cutoff_layer if (method == "rcllm_plus" and cutoff_layer is not None) else num_layers
        stop_layer = max(start_sparse, min(num_layers, stop_layer))
        hidden = hidden.index_select(1, sel_t)
        if start_sparse < stop_layer:
            hidden = ad.run_segment(start_sparse, stop_layer - 1, hidden, sel_t, active_cache, rotary)
            active_counts.extend([int(sel_t.numel())] * (stop_layer - start_sparse))
        mark("middle")

        # 5. Deep tail layers (stop_layer .. num_layers - 1): protected tokens only.
        if stop_layer < num_layers:
            prot_t = torch.as_tensor(protected_np, device=ad.device, dtype=torch.long)
            gather_idx = torch.searchsorted(sel_t, prot_t)
            hidden = hidden.index_select(1, gather_idx)
            hidden = ad.run_segment(stop_layer, num_layers - 1, hidden, prot_t, active_cache, rotary)
            active_counts.extend([int(prot_t.numel())] * (num_layers - stop_layer))
        mark("tail")

        scores = ad.score_candidates(hidden, prompt.alias_token_ids).cpu().numpy()
        mark("output")
        if is_cuda:
            torch.cuda.synchronize(ad.device)
        return PrefillResult(
            candidate_scores=scores,
            active_counts_per_layer=active_counts,
            selected_indices=selected_np,
            total_ms=(time.perf_counter() - t_start) * 1000.0,
            stage_ms=stages,
        )
