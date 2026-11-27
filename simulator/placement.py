"""Item KV sharding, hot-set replication, and archived METIS layout loader."""

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import pickle
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

ARCHIVED_HOT_FRACTION = 0.0001
DEFAULT_HOT_FRACTION = 0.001


@dataclass(frozen=True)
class ItemPlacement:
    """Per-replica resident item sets and hot-set metadata."""

    replicas: int
    shards: Tuple[FrozenSet[str], ...]
    hot_items: FrozenSet[str]

    def hit_ratio(self, replica: int, candidate_items: Sequence[str]) -> float:
        if not candidate_items:
            return 0.0
        shard = self.shards[replica]
        return sum(1 for item in candidate_items if item in shard) / len(candidate_items)

    def replica_item_counts(self) -> List[int]:
        return [len(s) for s in self.shards]


def build_metis_or_hash_placement(
    item_popularity: Dict[str, int],
    replicas: int,
    hot_fraction: float = DEFAULT_HOT_FRACTION,
    owner_map: Dict[str, int] | None = None,
) -> ItemPlacement:
    """Build a sharded placement with full replication of the top `hot_fraction` items.

    When `owner_map` (e.g., from offline METIS co-occurrence partitioning) is supplied,
    each non-hot item is assigned to its METIS owner; otherwise deterministic SHA-256
    hashing shards non-hot items across replicas.
    """
    if replicas < 1:
        raise ValueError("replicas must be >= 1")
    ranked = sorted(item_popularity, key=lambda item: (-item_popularity[item], item))
    hot_count = max(1, round(hot_fraction * len(ranked))) if ranked else 0
    hot_set = frozenset(ranked[:hot_count])
    buckets: List[Set[str]] = [set(hot_set) for _ in range(replicas)]
    for item in ranked[hot_count:]:
        if owner_map is not None and item in owner_map:
            owner = int(owner_map[item]) % replicas
        else:
            digest = hashlib.sha256(item.encode("utf-8")).digest()
            owner = int.from_bytes(digest[:8], "big") % replicas
        buckets[owner].add(item)
    return ItemPlacement(replicas=replicas, shards=tuple(frozenset(b) for b in buckets), hot_items=hot_set)


def load_archived_layout(
    placement_dir: Path,
    dataset: str,
    replicas: int,
    hot_fraction: float = DEFAULT_HOT_FRACTION,
) -> ItemPlacement:
    """Load an archived ICDCS/TPDS METIS placement directory and expand hot replication."""
    placement_dir = Path(placement_dir)
    with gzip.open(placement_dir / "item_popularity.pkl.gz", "rb") as stream:
        popularity = pickle.load(stream)
    ranked = sorted(popularity, key=lambda item: (-popularity[item], item))
    hot_count = max(1, round(hot_fraction * len(ranked)))
    archived_count = max(1, round(ARCHIVED_HOT_FRACTION * len(ranked)))
    archived_hot = frozenset(ranked[:archived_count])
    expanded_hot = frozenset(ranked[:hot_count])

    manifest = json.loads((placement_dir / "placement_summary.json").read_text())
    if (manifest["k"], manifest["dataset"]) != (replicas, dataset):
        raise ValueError(f"Manifest mismatch in {placement_dir}: expected ({replicas}, {dataset})")

    shards: List[FrozenSet[str]] = []
    for r in range(replicas):
        with (placement_dir / f"replica_{r}_items.txt").open() as stream:
            catalog = {line.rstrip("\n") for line in stream}
        if not archived_hot <= catalog:
            raise ValueError(f"Replica {r} missing archived hot items")
        catalog.update(expanded_hot)
        shards.append(frozenset(catalog))
    return ItemPlacement(replicas=replicas, shards=tuple(shards), hot_items=expanded_hot)
