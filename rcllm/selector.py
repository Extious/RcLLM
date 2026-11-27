"""Rank-normalized token selector for RcLLM+ (NumPy reference and fused Triton GPU kernels).

The selector combines three signals with equal average-rank weight after layer 1:
  1. Layer-0 causal received attention mass (reusing forward layer-0 Q/K);
  2. Layer-1 Key L2 deviation from cached pre-RoPE K;
  3. Layer-1 Value L2 deviation from cached V.
"""

import math
from typing import Dict, Sequence, Tuple

import numpy as np


def rank_norm_numpy(values: Sequence[float]) -> np.ndarray:
    """Average-rank normalization to [0, 1] in FP64 with exact tie averaging."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if not np.isfinite(v).all():
        raise ValueError("Selector scores must be finite")
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    order = np.argsort(v, kind="stable")
    sv = v[order]
    starts = np.r_[0, np.flatnonzero(sv[1:] != sv[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    ranked = np.empty(n, dtype=np.float64)
    for a, b in zip(starts, ends):
        ranked[order[a:b]] = (a + b - 1) / 2.0 / (n - 1)
    return ranked


def select_tokens_numpy(
    attention: np.ndarray,
    k_dev: np.ndarray,
    v_dev: np.ndarray,
    eligible: Sequence[int],
    protected: Sequence[int],
    recompute_rate_pct: int,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Select tokens to recompute from `eligible` at integer percentage `recompute_rate_pct`.

    Returns:
        selected: sorted 1D int64 array of all active token positions (protected U chosen).
        scores: 1D float64 array of combined rank-normalized scores aligned with `eligible`.
    """
    elig = np.asarray(eligible, dtype=np.int64)
    prot = np.asarray(protected, dtype=np.int64)
    w_attn, w_k, w_v = weights
    if len(elig) == 0:
        return np.sort(prot), np.zeros(0, dtype=np.float64)
    score = (
        w_attn * rank_norm_numpy(np.asarray(attention)[elig])
        + w_k * rank_norm_numpy(np.asarray(k_dev)[elig])
        + w_v * rank_norm_numpy(np.asarray(v_dev)[elig])
    )
    keep = min(len(elig), (recompute_rate_pct * len(elig) + 99) // 100)
    order = np.argsort(-score, kind="stable")
    chosen = elig[order[:keep]]
    selected = np.sort(np.concatenate([prot, chosen]))
    return selected, score


def select_by_role_numpy(
    attention: np.ndarray,
    k_dev: np.ndarray,
    v_dev: np.ndarray,
    roles: Dict[str, Sequence[int]],
    protected: Sequence[int],
    drop_fraction: float,
    global_budget: bool = True,
) -> Tuple[np.ndarray, Dict[str, int], Dict[int, float]]:
    """Reference selector supporting both global (RcLLM+) and per-role budgets."""
    if not (0.0 < drop_fraction < 1.0):
        raise ValueError(f"drop_fraction must be in (0, 1), got {drop_fraction}")
    eligible = np.array(sorted(set().union(*map(set, roles.values()))), dtype=np.int64)
    scopes = [eligible] if global_budget else [np.asarray(roles[r], dtype=np.int64) for r in ("review", "item")]
    chosen = set(int(x) for x in protected)
    scores: Dict[int, float] = {}
    for ids in scopes:
        if len(ids) == 0:
            continue
        score = (
            rank_norm_numpy(np.asarray(attention)[ids])
            + rank_norm_numpy(np.asarray(k_dev)[ids])
            + rank_norm_numpy(np.asarray(v_dev)[ids])
        )
        keep = len(ids) - int(np.floor(drop_fraction * len(ids)))
        order = np.argsort(-score, kind="stable")
        chosen.update(ids[order[:keep]].tolist())
        scores.update(zip(ids.tolist(), score.tolist()))
    selected = np.array(sorted(chosen), dtype=np.int64)
    kept = {r: len(set(ids) & chosen) for r, ids in roles.items()}
    return selected, kept, scores


try:
    import torch
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _row_norm(
        Q, K, Norm, T,
        SQH: tl.constexpr, SQT: tl.constexpr, SQD: tl.constexpr,
        SKH: tl.constexpr, SKT: tl.constexpr, SKD: tl.constexpr,
        G: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
        M: tl.constexpr, N: tl.constexpr,
    ):
        qtile, head = tl.program_id(0), tl.program_id(1)
        row = qtile * M + tl.arange(0, M)
        dim = tl.arange(0, D)
        col = tl.arange(0, N)
        q = tl.load(Q + head * SQH + row[:, None] * SQT + dim[None, :] * SQD, row[:, None] < T, 0)
        key = K + (head // G) * SKH + dim[:, None] * SKD
        maximum = tl.full((M,), float("-inf"), tl.float32)
        total = tl.full((M,), 0, tl.float32)
        for start in range(0, tl.minimum(T, (qtile + 1) * M), N):
            positions = start + col
            k = tl.load(key + positions[None, :] * SKT, positions[None, :] < T, 0)
            score = tl.dot(q, k).to(tl.bfloat16).to(tl.float32) * SCALE
            score = tl.where((positions[None, :] <= row[:, None]) & (positions[None, :] < T), score, float("-inf"))
            new_max = tl.maximum(maximum, tl.max(score, 1))
            total = total * tl.exp(maximum - new_max) + tl.sum(tl.exp(score - new_max[:, None]), 1)
            maximum = new_max
        tl.store(Norm + (head * T + row) * 2, maximum, row < T)
        tl.store(Norm + (head * T + row) * 2 + 1, 1.0 / total, row < T)

    @triton.jit
    def _column_mass(
        Q, K, Norm, Partial, T,
        SQH: tl.constexpr, SQT: tl.constexpr, SQD: tl.constexpr,
        SKH: tl.constexpr, SKT: tl.constexpr, SKD: tl.constexpr,
        G: tl.constexpr, H: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
        M: tl.constexpr, N: tl.constexpr, SPAN: tl.constexpr,
    ):
        ktile, head, group = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        col = ktile * N + tl.arange(0, N)
        dim = tl.arange(0, D)
        k = tl.load(K + (head // G) * SKH + col[:, None] * SKT + dim[None, :] * SKD, col[:, None] < T, 0)
        query = Q + head * SQH + dim[:, None] * SQD
        acc = tl.full((N, M), 0, tl.float32)
        for start in range(tl.maximum(group * SPAN, ktile * N), tl.minimum((group + 1) * SPAN, T), M):
            row = start + tl.arange(0, M)
            q = tl.load(query + row[None, :] * SQT, row[None, :] < T, 0)
            maximum = tl.load(Norm + (head * T + row) * 2, row < T, 0)
            inv_total = tl.load(Norm + (head * T + row) * 2 + 1, row < T, 0)
            score = tl.dot(k, q).to(tl.bfloat16).to(tl.float32) * SCALE
            delta = tl.where((row[None, :] >= col[:, None]) & (row[None, :] < T), score - maximum[None, :], float("-inf"))
            acc += tl.exp(delta) * inv_total[None, :]
        tl.store(Partial + (group * H + head) * T + col, tl.sum(acc, 1), col < T)

    @triton.jit(do_not_specialize=["N"])
    def _kv_deviation(
        K, V, CK, CV, E, Out, N,
        KH: tl.constexpr, KT: tl.constexpr, KD: tl.constexpr,
        VH: tl.constexpr, VT: tl.constexpr, VD: tl.constexpr,
        CKH: tl.constexpr, CKT: tl.constexpr, CKD: tl.constexpr,
        CVH: tl.constexpr, CVT: tl.constexpr, CVD: tl.constexpr,
        D: tl.constexpr, WIDTH: tl.constexpr, TOK: tl.constexpr, FEAT: tl.constexpr,
    ):
        ix = tl.program_id(0) * TOK + tl.arange(0, TOK)
        pos = tl.load(E + ix, ix < N, 0)[:, None]
        offsets = tl.arange(0, FEAT)
        h, d = offsets // D, offsets % D
        valid = (ix[:, None] < N) & (offsets[None, :] < WIDTH)
        k = tl.load(K + pos * KT + h[None, :] * KH + d[None, :] * KD, valid, 0).to(tl.float32)
        v = tl.load(V + pos * VT + h[None, :] * VH + d[None, :] * VD, valid, 0).to(tl.float32)
        ck = tl.load(CK + pos * CKT + h[None, :] * CKH + d[None, :] * CKD, valid, 0).to(tl.float32)
        cv = tl.load(CV + pos * CVT + h[None, :] * CVH + d[None, :] * CVD, valid, 0).to(tl.float32)
        dk, dv = k - ck, v - cv
        tl.store(Out + ix, tl.sqrt_rn(tl.sum(dk * dk, 1)), ix < N)
        tl.store(Out + N + ix, tl.sqrt_rn(tl.sum(dv * dv, 1)), ix < N)


def received_attention_triton(q, keys):
    """Fused Triton kernel computing layer-0 causal received attention mass per token."""
    if not HAS_TRITON:
        raise RuntimeError("Triton and PyTorch CUDA are required for received_attention_triton")
    _, heads, tokens, dim = q.shape
    group_size = heads // keys.shape[1]
    scale = 1.0 / math.sqrt(dim)
    normals = torch.empty((heads, tokens, 2), device=q.device, dtype=torch.float32)
    span = 512
    groups = triton.cdiv(tokens, span)
    partial = torch.empty((groups, heads, tokens), device=q.device, dtype=torch.float32)
    strides = (*q.stride()[1:], *keys.stride()[1:])
    _row_norm[(triton.cdiv(tokens, 32), heads)](
        q, keys, normals, tokens, *strides, group_size, dim, scale, 32, 128, num_warps=4
    )
    _column_mass[(triton.cdiv(tokens, 64), heads, groups)](
        q, keys, normals, partial, tokens, *strides, group_size, heads, dim, scale, 32, 64, span, num_warps=4
    )
    return partial.sum((0, 1))


def fused_kv_deviation_triton(current, cached, eligible):
    """Fused Triton kernel computing per-token L2 deviations for layer-1 K and V."""
    if not HAS_TRITON:
        raise RuntimeError("Triton and PyTorch CUDA are required for fused_kv_deviation_triton")
    k, v = current
    ck, cv = cached
    n = eligible.numel()
    size = k.shape[1] * k.shape[-1]
    out = torch.empty((2, n), dtype=torch.float32, device=k.device)
    _kv_deviation[(triton.cdiv(n, 8),)](
        k, v, ck, cv, eligible, out, n,
        *k.stride()[1:], *v.stride()[1:], *ck.stride()[1:], *cv.stride()[1:],
        k.shape[-1], size, 8, triton.next_power_of_2(size),
    )
    return out.unbind(0)


def rank_norm_torch(values):
    """GPU average-rank normalization matching FP64 tie handling of `rank_norm_numpy`."""
    import torch as _torch

    n = values.numel()
    if n <= 1:
        return _torch.zeros_like(values, dtype=_torch.float64)
    order = _torch.argsort(values, stable=True)
    sorted_values = values.index_select(0, order)
    positions = _torch.arange(n, device=values.device)
    change = sorted_values[1:] != sorted_values[:-1]
    edge = _torch.ones(1, device=values.device, dtype=_torch.bool)
    first = _torch.cummax(_torch.where(_torch.cat((edge, change)), positions, 0), 0).values
    last = _torch.cummin(_torch.where(_torch.cat((change, edge)), positions, n - 1).flip(0), 0).values.flip(0)
    divisor = _torch.full((), n - 1, dtype=_torch.float64, device=values.device)
    ranks = ((first + last).to(_torch.float64) * 0.5) / divisor
    return _torch.empty_like(ranks).scatter_(0, order, ranks)


def select_tokens_gpu(attention, current_kv, cached_kv, eligible, protected, recompute_rate_pct: int):
    """Fused GPU selector returning sorted active token indices."""
    import torch as _torch

    if eligible.numel() == 0:
        return protected.sort().values
    dk, dv = fused_kv_deviation_triton(current_kv, cached_kv, eligible)
    score = rank_norm_torch(attention.index_select(0, eligible)) + rank_norm_torch(dk) + rank_norm_torch(dv)
    order = _torch.argsort(score, descending=True, stable=True)
    keep = (recompute_rate_pct * eligible.numel() + 99) // 100
    return _torch.cat((protected, eligible.index_select(0, order[:keep]))).sort().values
