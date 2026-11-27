"""Model-agnostic Hugging Face decoder adapter for RcLLM and RcLLM+.

Supports Qwen3-8B (with QK-Norm), Mistral-7B-Instruct-v0.3, and GLM-4-9B-0414
(with interleaved partial RoPE and post-attention / post-MLP layer norms)
without modifying upstream `transformers` source code.
"""

import inspect
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from rcllm.config import PromptSpec


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def apply_standard_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


def apply_interleaved_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """GLM-4 interleaved partial RoPE (rotates the first `cos.shape[-1]` dims)."""
    c = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, -1).unsqueeze(1)
    s = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, -1).unsqueeze(1)
    d = c.shape[-1]
    z = x[..., :d]
    rotated = torch.stack((-z[..., 1::2], z[..., ::2]), dim=-1).flatten(-2)
    return torch.cat((z * c + rotated * s, x[..., d:]), dim=-1)


class AttentionAdapter:
    """Wraps a decoder layer's self-attention Q/K/V projections and optional QK-Norm."""

    def __init__(self, attn, num_q_heads: int, num_kv_heads: int, head_dim: int):
        self.attn = attn
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_norm = getattr(attn, "q_norm", None) or (lambda x: x)
        self.k_norm = getattr(attn, "k_norm", None) or (lambda x: x)

    def qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = x.shape[1]
        q = self.q_norm(self.attn.q_proj(x).view(1, n, self.num_q_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.attn.k_proj(x).view(1, n, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.attn.v_proj(x).view(1, n, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def k_out_module(self):
        return getattr(self.attn, "k_norm", None) or self.attn.k_proj


class DecoderAdapter:
    """Uniform pre-RoPE block caching and selective-token layer execution."""

    def __init__(self, model, base=None, config=None):
        self.official = model
        self.base = base if base is not None else model.model
        self.config = config if config is not None else model.config
        self.layers = self.base.layers
        self.kind = getattr(self.config, "model_type", "generic")
        self.num_q_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = getattr(self.config, "head_dim", None) or (self.config.hidden_size // self.num_q_heads)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.adapters = [
            AttentionAdapter(layer.self_attn, self.num_q_heads, self.num_kv_heads, self.head_dim)
            for layer in self.layers
        ]

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.base.embed_tokens(input_ids)

    def rotary(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(hidden.shape[1], device=self.device)[None]
        return self.base.rotary_emb(hidden, pos)

    def rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if self.kind == "glm4":
            return apply_interleaved_partial_rope(x, cos, sin)
        return apply_standard_rope(x, cos, sin)

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.official.lm_head(hidden)

    def score_candidates(self, hidden: torch.Tensor, alias_token_ids: Sequence[int]) -> torch.Tensor:
        logits = self.lm_head(self.base.norm(hidden[:, -1, :])).float()
        return logits[0, list(alias_token_ids)]

    @torch.inference_mode()
    def encode_reuse_regions(self, prompt: PromptSpec) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Independently encode each reusable prompt segment and store pre-RoPE K and V."""
        seq_len = prompt.seq_len
        cache = [
            (
                torch.zeros(1, self.num_kv_heads, seq_len, self.head_dim, device=self.device, dtype=self.dtype),
                torch.zeros(1, self.num_kv_heads, seq_len, self.head_dim, device=self.device, dtype=self.dtype),
            )
            for _ in range(self.num_layers)
        ]
        for region in prompt.reuse_regions:
            left, right = region.start, region.end
            if right <= left:
                continue
            handles = []
            for idx, layer in enumerate(self.layers):
                kmod = self.adapters[idx].k_out_module()
                if kmod is self.adapters[idx].attn.k_proj:
                    def kh(_m, _a, out, i=idx, n=right - left, l=left, r=right):
                        cache[i][0][:, :, l:r].copy_(out.view(1, n, self.num_kv_heads, self.head_dim).transpose(1, 2))
                else:
                    def kh(_m, _a, out, i=idx, l=left, r=right):
                        cache[i][0][:, :, l:r].copy_(out.transpose(1, 2))

                def vh(_m, _a, out, i=idx, n=right - left, l=left, r=right):
                    cache[i][1][:, :, l:r].copy_(out.view(1, n, self.num_kv_heads, self.head_dim).transpose(1, 2))

                handles.append(kmod.register_forward_hook(kh))
                handles.append(layer.self_attn.v_proj.register_forward_hook(vh))
            ids = torch.tensor([list(prompt.input_ids[left:right])], device=self.device)
            try:
                self.base(input_ids=ids, use_cache=False, return_dict=True)
            finally:
                for h in handles:
                    h.remove()
        return cache

    def project_layer(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        indices: torch.Tensor,
        cache,
        rotary: Tuple[torch.Tensor, torch.Tensor],
    ) -> dict:
        """Compute Q/K/V for active `indices` and scatter into cached KV."""
        layer = self.layers[layer_idx]
        q, k, v = self.adapters[layer_idx].qkv(layer.input_layernorm(hidden))
        cos, sin = rotary
        c, s = cos.index_select(1, indices), sin.index_select(1, indices)
        q = self.rotate(q, c, s)
        if indices.numel() == cos.shape[1]:
            raw_k, vals = k, v
        else:
            cached_k, cached_v = cache[layer_idx]
            raw_k = cached_k.clone()
            vals = cached_v.clone()
            raw_k[:, :, indices] = k
            vals[:, :, indices] = v
        return {
            "q": q,
            "k": k,
            "v": v,
            "raw_k": raw_k,
            "values": vals,
            "keys": self.rotate(raw_k, cos, sin),
        }

    def finish_layer(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        indices: torch.Tensor,
        proj: dict,
        rotary: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Run SDPA attention, O-projection, residual, and MLP for layer `layer_idx`."""
        seq_len = rotary[0].shape[1]
        complete = indices.numel() == seq_len
        mask = None if complete else (torch.arange(seq_len, device=self.device)[None, :] <= indices[:, None])[None, None]
        rep = self.num_q_heads // self.num_kv_heads
        keys = proj["keys"].repeat_interleave(rep, dim=1) if not complete else proj["keys"]
        values = proj["values"].repeat_interleave(rep, dim=1) if not complete else proj["values"]
        use_native_gqa = complete and hasattr(F, "scaled_dot_product_attention") and self._supports_enable_gqa()
        if complete and not use_native_gqa:
            keys = proj["keys"].repeat_interleave(rep, dim=1)
            values = proj["values"].repeat_interleave(rep, dim=1)
        sdpa_kwargs = {"attn_mask": mask, "is_causal": complete, "dropout_p": 0.0}
        if use_native_gqa:
            sdpa_kwargs["enable_gqa"] = True
        out = F.scaled_dot_product_attention(proj["q"], keys, values, **sdpa_kwargs)
        out = out.transpose(1, 2).reshape(1, indices.numel(), -1)
        layer = self.layers[layer_idx]
        out = layer.self_attn.o_proj(out)
        if self.kind == "glm4" and hasattr(layer, "post_self_attn_layernorm"):
            out = layer.post_self_attn_layernorm(out)
        hidden = hidden + out
        mlp_out = layer.mlp(layer.post_attention_layernorm(hidden))
        if self.kind == "glm4" and hasattr(layer, "post_mlp_layernorm"):
            mlp_out = layer.post_mlp_layernorm(mlp_out)
        return hidden + mlp_out

    def run_segment(
        self,
        first: int,
        last: int,
        hidden: torch.Tensor,
        indices: torch.Tensor,
        cache,
        rotary: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        for layer_idx in range(first, last + 1):
            proj = self.project_layer(layer_idx, hidden, indices, cache, rotary)
            hidden = self.finish_layer(layer_idx, hidden, indices, proj, rotary)
        return hidden

    @staticmethod
    def _supports_enable_gqa() -> bool:
        try:
            ver = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
            return ver >= (2, 5)
        except Exception:
            return False


def load_decoder(model_name_or_path: str, dtype=torch.bfloat16, device: str = "cuda", local_files_only: bool = False) -> DecoderAdapter:
    """Load a Hugging Face causal LM and wrap it in a `DecoderAdapter`."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    if getattr(config, "sliding_window", None):
        raise ValueError("Windowed attention models are not supported by the dense SDPA adapter")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=local_files_only,
    ).eval().to(device)
    return DecoderAdapter(model, model.model, model.config)
