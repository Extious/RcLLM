"""Model specifications and prompt structures for RcLLM and RcLLM+."""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class ModelSpec:
    """Architecture and TPDS default operating parameters for an evaluated LLM."""

    name: str
    hf_repo: str
    hf_revision: str
    num_layers: int
    default_cutoff: int
    iso_recompute_rate: int
    budget_recompute_rate: int
    kv_bytes_per_token_layer: int
    mlp_intermediate_size: int
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128

    @property
    def kv_bytes_per_token(self) -> int:
        return self.num_layers * self.kv_bytes_per_token_layer


MODEL_SPECS: Dict[str, ModelSpec] = {
    "qwen3": ModelSpec(
        name="Qwen3-8B",
        hf_repo="Qwen/Qwen3-8B",
        hf_revision="b968826d9c46dd6066d109eabc6255188de91218",
        num_layers=36,
        default_cutoff=25,
        iso_recompute_rate=7,
        budget_recompute_rate=22,
        kv_bytes_per_token_layer=4096,
        mlp_intermediate_size=12288,
    ),
    "mistral7": ModelSpec(
        name="Mistral-7B-Instruct-v0.3",
        hf_repo="mistralai/Mistral-7B-Instruct-v0.3",
        hf_revision="c170c708c41dac9275d15a8fff4eca08d52bab71",
        num_layers=32,
        default_cutoff=20,
        iso_recompute_rate=7,
        budget_recompute_rate=25,
        kv_bytes_per_token_layer=4096,
        mlp_intermediate_size=14336,
    ),
    "glm4": ModelSpec(
        name="GLM-4-9B-0414",
        hf_repo="zai-org/GLM-4-9B-0414",
        hf_revision="645b8482494e31b6b752272bf7f7f273ef0f3caf",
        num_layers=40,
        default_cutoff=25,
        iso_recompute_rate=9,
        budget_recompute_rate=25,
        kv_bytes_per_token_layer=2048,
        mlp_intermediate_size=13696,
        num_attention_heads=32,
        num_kv_heads=2,
        head_dim=128,
    ),
}

# ICDCS'26 RcLLM conference selector defaults: (review_rate_pct, item_rate_pct) and review drift lambda.
CONFERENCE_DEFAULTS: Dict[str, Tuple[int, int, float]] = {
    "amazon": (30, 30, 0.3),
    "goodreads": (20, 30, 0.5),
    "yelp": (50, 40, 0.4),
}


@dataclass(frozen=True)
class RegionSpan:
    """A contiguous reusable prompt segment (user history review or candidate item)."""

    role: str  # "review" or "item"
    identity: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


@dataclass(frozen=True)
class PromptSpec:
    """Decomposed generative recommendation prompt with reusable spans and candidate tokens."""

    input_ids: Sequence[int]
    reuse_regions: Sequence[RegionSpan]
    alias_token_ids: Sequence[int]
    dataset: str = "amazon"

    @property
    def seq_len(self) -> int:
        return len(self.input_ids)

    def role_indices(self) -> Dict[str, List[int]]:
        roles: Dict[str, set] = {"review": set(), "item": set()}
        for region in self.reuse_regions:
            if region.role in roles:
                roles[region.role].update(range(region.start, region.end))
        return {role: sorted(indices) for role, indices in roles.items()}

    def eligible_and_protected(self) -> Tuple[List[int], List[int]]:
        roles = self.role_indices()
        eligible = sorted(set(roles["review"]) | set(roles["item"]))
        eligible_set = set(eligible)
        protected = [i for i in range(self.seq_len) if i not in eligible_set]
        return eligible, protected
