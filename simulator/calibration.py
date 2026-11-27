"""Six-stage NNLS latency calibration and analytical decoder FLOP cost model."""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.optimize import nnls

from rcllm.config import MODEL_SPECS, ModelSpec


@dataclass(frozen=True)
class StageCoefficients:
    """Non-negative least-squares coefficients (ms) for prefill stage modeling."""

    kv_load: Tuple[float, float] = (0.25, 0.0045)  # (intercept_ms, ms_per_cached_token)
    early_layers: Tuple[float, float, float] = (0.50, 0.006, 2.5e-7)  # (c0, c_tok, c_pairs)
    selection: Tuple[float, float] = (1.20, 0.0005)  # (c0, c_eligible)
    middle_per_layer: Tuple[float, float, float] = (0.04, 0.0024, 1.0e-7)  # per middle layer
    tail_per_layer: Tuple[float, float, float] = (0.04, 0.0024, 1.0e-7)  # per deep tail layer
    output_head_ms: float = 0.25
    overlap_discount: float = 0.85  # Fraction of kv_load hidden by LayerKVPrefetch


def analytical_decoder_flops(
    spec: ModelSpec,
    seq_len: int,
    active_per_layer: Sequence[int],
) -> Tuple[float, float]:
    """Compute analytical decoder prefill FLOPs and ratio relative to Full prefill."""
    hidden = spec.num_attention_heads * spec.head_dim
    linear_per_q = 2 * hidden * ((2 * spec.num_attention_heads + 2 * spec.num_kv_heads) * spec.head_dim + 3 * spec.mlp_intermediate_size)
    attn_per_pair = 4 * spec.head_dim * spec.num_attention_heads
    full_pairs = seq_len * (seq_len + 1) // 2
    full_flops = spec.num_layers * (linear_per_q * seq_len + attn_per_pair * full_pairs)
    actual_flops = 0.0
    for active in active_per_layer:
        # Expected causal attention pairs for `active` tokens uniformly spanning [0, seq_len).
        est_pairs = active * (seq_len + 1) / 2.0 if active < seq_len else full_pairs
        actual_flops += linear_per_q * active + attn_per_pair * est_pairs
    return actual_flops, (actual_flops / full_flops if full_flops > 0 else 1.0)


def fit_stage_model(features: np.ndarray, observed_ms: np.ndarray) -> np.ndarray:
    """Fit non-negative linear stage timing coefficients with NNLS."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(observed_ms, dtype=np.float64)
    coef, _ = nnls(x, y)
    return coef


def predict_service_ms(
    model_key: str,
    method: str,
    seq_len: int,
    prefix_tokens: int,
    cached_tokens: int,
    recompute_rate_pct: int | None = None,
    cutoff_layer: int | None = None,
    coefs: StageCoefficients = StageCoefficients(),
    use_overlap: bool = True,
) -> float:
    """Predict single-request GPU TTFT (ms) from the calibrated 6-stage model."""
    spec = MODEL_SPECS[model_key]
    depth = spec.num_layers
    full_pairs = seq_len * (seq_len + 1) / 2.0

    def _layer_ms(c: Tuple[float, float, float], active: float, pairs: float) -> float:
        return c[0] + c[1] * active + c[2] * pairs

    if method == "Full":
        early = _layer_ms(coefs.early_layers, seq_len, full_pairs)
        rest = (depth - 2) * _layer_ms(coefs.middle_per_layer, seq_len, full_pairs)
        return early + rest + coefs.output_head_ms

    if method == "Prefix-Cache":
        active = max(1, seq_len - prefix_tokens)
        pairs = max(1.0, full_pairs - prefix_tokens * (prefix_tokens + 1) / 2.0)
        return depth * _layer_ms(coefs.middle_per_layer, active, pairs) + coefs.output_head_ms

    eligible = min(seq_len - 1, max(0, cached_tokens))
    protected = seq_len - eligible
    kv_ms = coefs.kv_load[0] + coefs.kv_load[1] * eligible
    if use_overlap and method in ("RcLLM+", "RcLLM"):
        kv_ms *= max(0.0, 1.0 - coefs.overlap_discount)

    early = _layer_ms(coefs.early_layers, seq_len, full_pairs)
    sel_ms = coefs.selection[0] + coefs.selection[1] * eligible

    if method == "RcLLM+":
        rate = recompute_rate_pct if recompute_rate_pct is not None else spec.iso_recompute_rate
        cut = cutoff_layer if cutoff_layer is not None else spec.default_cutoff
        mid_layers = max(0, cut - 2)
        tail_layers = max(0, depth - cut)
        mid_active = protected + (rate * eligible + 99) // 100
        mid_pairs = mid_active * (seq_len + 1) / 2.0
        tail_pairs = protected * (seq_len + 1) / 2.0
        return (
            kv_ms
            + early
            + sel_ms
            + mid_layers * _layer_ms(coefs.middle_per_layer, mid_active, mid_pairs)
            + tail_layers * _layer_ms(coefs.tail_per_layer, protected, tail_pairs)
            + coefs.output_head_ms
        )

    if method == "CacheBlend":
        rate = recompute_rate_pct if recompute_rate_pct is not None else 15
        mid_active = protected + (rate * eligible + 99) // 100
        mid_pairs = mid_active * (seq_len + 1) / 2.0
        return kv_ms + early + sel_ms + (depth - 2) * _layer_ms(coefs.middle_per_layer, mid_active, mid_pairs) + coefs.output_head_ms

    if method == "RcLLM":
        rate = recompute_rate_pct if recompute_rate_pct is not None else 15
        mid_active = protected + (rate * eligible + 99) // 100
        mid_pairs = mid_active * (seq_len + 1) / 2.0
        return kv_ms + 0.5 * early + sel_ms + (depth - 1) * _layer_ms(coefs.middle_per_layer, mid_active, mid_pairs) + coefs.output_head_ms

    raise ValueError(f"Unsupported method '{method}'")
