"""CPU smoke tests for the RcLLM+ depth-bounded selective recomputation engine."""

from types import SimpleNamespace
import unittest

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    from rcllm.adapter import DecoderAdapter
    from rcllm.config import PromptSpec, RegionSpan
    from rcllm.engine import RcLLMEngine

    class _DummyRotary(nn.Module):
        def __init__(self, head_dim: int):
            super().__init__()
            self.head_dim = head_dim

        def forward(self, hidden: torch.Tensor, position_ids: torch.Tensor):
            t = position_ids.shape[1]
            cos = torch.ones((1, t, self.head_dim), device=hidden.device, dtype=hidden.dtype)
            sin = torch.zeros((1, t, self.head_dim), device=hidden.device, dtype=hidden.dtype)
            return cos, sin

    class _DummyAttn(nn.Module):
        def __init__(self, hidden_size: int, num_q: int, num_kv: int):
            super().__init__()
            head_dim = hidden_size // num_q
            self.q_proj = nn.Linear(hidden_size, num_q * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, num_kv * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, num_kv * head_dim, bias=False)
            self.o_proj = nn.Linear(num_q * head_dim, hidden_size, bias=False)

    class _DummyLayer(nn.Module):
        def __init__(self, hidden_size: int, num_q: int, num_kv: int):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(hidden_size)
            self.post_attention_layernorm = nn.LayerNorm(hidden_size)
            self.self_attn = _DummyAttn(hidden_size, num_q, num_kv)
            self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 2), nn.SiLU(), nn.Linear(hidden_size * 2, hidden_size))

    class _DummyBase(nn.Module):
        def __init__(self, vocab: int, hidden_size: int, num_layers: int, num_q: int, num_kv: int):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, hidden_size)
            self.rotary_emb = _DummyRotary(hidden_size // num_q)
            self.layers = nn.ModuleList([_DummyLayer(hidden_size, num_q, num_kv) for _ in range(num_layers)])
            self.norm = nn.LayerNorm(hidden_size)

    class _DummyCausalLM(nn.Module):
        def __init__(self, vocab: int = 64, hidden_size: int = 32, num_layers: int = 6, num_q: int = 4, num_kv: int = 2):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="qwen3",
                hidden_size=hidden_size,
                num_attention_heads=num_q,
                num_key_value_heads=num_kv,
                head_dim=hidden_size // num_q,
                num_hidden_layers=num_layers,
            )
            self.model = _DummyBase(vocab, hidden_size, num_layers, num_q, num_kv)
            self.lm_head = nn.Linear(hidden_size, vocab, bias=False)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestEngineCPU(unittest.TestCase):
    def test_depth_bounded_active_counts_and_methods(self):
        torch.manual_seed(0)
        model = _DummyCausalLM(num_layers=6).eval()
        adapter = DecoderAdapter(model)
        engine = RcLLMEngine(adapter)

        prompt = PromptSpec(
            input_ids=list(range(24)),
            reuse_regions=[
                RegionSpan(role="review", identity="u0", start=2, end=10),
                RegionSpan(role="item", identity="i0", start=10, end=20),
            ],
            alias_token_ids=[1, 2, 3, 4],
            dataset="amazon",
        )
        # Build synthetic zero-deviation cache from full forward projections.
        cache = [
            (torch.randn(1, 2, 24, 8) * 0.05, torch.randn(1, 2, 24, 8) * 0.05)
            for _ in range(6)
        ]

        res_full = engine.prefill(prompt, None, method="full")
        self.assertEqual(res_full.active_counts_per_layer, [24] * 6)
        self.assertEqual(len(res_full.candidate_scores), 4)

        res_rc = engine.prefill(prompt, cache, method="rcllm_plus", recompute_rate_pct=25, cutoff_layer=4, use_triton=False)
        # Layers 0..1 dense (24), layers 2..3 selective (6 protected + ceil(0.25*18)=5 -> 11), layers 4..5 protected only (6).
        self.assertEqual(res_rc.active_counts_per_layer, [24, 24, 11, 11, 6, 6])
        self.assertEqual(len(res_rc.candidate_scores), 4)

        for method in ("cacheblend", "epic", "rcllm_conf", "reuse"):
            res = engine.prefill(prompt, cache, method=method, recompute_rate_pct=25, use_triton=False)
            self.assertEqual(len(res.active_counts_per_layer), 6)
            self.assertEqual(len(res.candidate_scores), 4)


if __name__ == "__main__":
    unittest.main()
