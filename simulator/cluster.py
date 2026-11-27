"""Multi-replica cluster replay, TTFT CDF, and P99 SLO capacity search."""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from rcllm.config import MODEL_SPECS
from simulator.calibration import StageCoefficients, predict_service_ms
from simulator.kernel import run_replay
from simulator.placement import ItemPlacement

KV_POOL_BYTES = 4 * (1 << 30)  # 4 GiB active KV pool per replica


@dataclass(frozen=True)
class RequestTraceRow:
    """Single recommendation request entry in a cluster trace."""

    seq_len: int
    prefix_tokens: int
    history_tokens: int
    items: Tuple[Tuple[str, int], ...]  # (item_id, token_count)


@dataclass(frozen=True)
class ClusterReplayResult:
    """Summary latency and throughput metrics from a cluster replay."""

    qps: float
    mean_ttft_s: float
    p50_ttft_s: float
    p95_ttft_s: float
    p99_ttft_s: float
    mean_hit_ratio: float
    ttft_samples_s: np.ndarray


class ClusterSimulator:
    """Calibrated multi-replica cluster simulator driven by the C++17 FCFS kernel."""

    def __init__(
        self,
        model_key: str,
        rows: Sequence[RequestTraceRow],
        placement: ItemPlacement,
        coefs: StageCoefficients = StageCoefficients(),
        use_overlap: bool = True,
    ):
        self.model_key = model_key
        self.spec = MODEL_SPECS[model_key]
        self.rows = list(rows)
        self.placement = placement
        self.coefs = coefs
        self.use_overlap = use_overlap
        self.kv_tokens = KV_POOL_BYTES / self.spec.kv_bytes_per_token

    def build_service_tables(
        self,
        method: str,
        *,
        recompute_rate_pct: int | None = None,
        cutoff_layer: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Precompute per-request, per-replica service times (s) and item hit ratios."""
        m = len(self.rows)
        k = self.placement.replicas
        tokens = np.empty(m, dtype=np.int64)
        service = np.empty((m, k), dtype=np.float64)
        hit = np.zeros((m, k), dtype=np.float64)

        for i, row in enumerate(self.rows):
            tokens[i] = row.seq_len
            if method in ("Full", "Prefix-Cache"):
                svc_s = predict_service_ms(
                    self.model_key,
                    method,
                    row.seq_len,
                    row.prefix_tokens,
                    0,
                    coefs=self.coefs,
                    use_overlap=self.use_overlap,
                ) / 1000.0
                service[i, :] = svc_s
                continue

            item_ids = [item for item, _ in row.items]
            for r in range(k):
                shard = self.placement.shards[r]
                local_item_tokens = sum(cnt for item, cnt in row.items if item in shard)
                hit[i, r] = self.placement.hit_ratio(r, item_ids)
                cached_tokens = row.history_tokens + local_item_tokens
                service[i, r] = predict_service_ms(
                    self.model_key,
                    method,
                    row.seq_len,
                    row.prefix_tokens,
                    cached_tokens,
                    recompute_rate_pct=recompute_rate_pct,
                    cutoff_layer=cutoff_layer,
                    coefs=self.coefs,
                    use_overlap=self.use_overlap,
                ) / 1000.0
        return tokens, service, hit

    def replay(
        self,
        method: str,
        qps: float,
        *,
        route: str = "affinity",
        num_requests: int = 2000,
        warmup: int = 200,
        seed: int = 42,
        recompute_rate_pct: int | None = None,
        cutoff_layer: int | None = None,
    ) -> ClusterReplayResult:
        """Replay a Poisson arrival stream at `qps` and report post-warmup TTFT metrics."""
        if qps <= 0:
            raise ValueError("qps must be positive")
        tokens, service, hit = self.build_service_tables(
            method,
            recompute_rate_pct=recompute_rate_pct,
            cutoff_layer=cutoff_layer,
        )
        rng = np.random.default_rng(seed)
        req_idx = rng.integers(0, len(self.rows), size=num_requests, dtype=np.int64)
        inter_arrivals = rng.exponential(1.0 / qps, size=num_requests)
        arrivals = np.cumsum(inter_arrivals)
        effective_route = "round-robin" if method in ("Full", "Prefix-Cache") else route
        start, finish, assigned = run_replay(
            req_idx,
            arrivals,
            tokens,
            service,
            hit,
            route=effective_route,
            kv_tokens=self.kv_tokens,
        )
        ttft = finish[warmup:] - arrivals[warmup:]
        post_req = req_idx[warmup:]
        post_rep = assigned[warmup:]
        mean_hit = float(np.mean(hit[post_req, post_rep])) if len(post_req) else 0.0
        return ClusterReplayResult(
            qps=qps,
            mean_ttft_s=float(np.mean(ttft)),
            p50_ttft_s=float(np.percentile(ttft, 50)),
            p95_ttft_s=float(np.percentile(ttft, 95)),
            p99_ttft_s=float(np.percentile(ttft, 99)),
            mean_hit_ratio=mean_hit,
            ttft_samples_s=ttft,
        )

    def find_capacity_qps(
        self,
        method: str,
        *,
        slo_p99_s: float = 1.0,
        qps_grid: Sequence[float] = tuple(range(5, 205, 5)),
        route: str = "affinity",
        num_requests: int = 2000,
        warmup: int = 200,
        seed: int = 42,
    ) -> float:
        """Return the highest monotone-safe QPS whose P99 TTFT stays <= `slo_p99_s`."""
        best = 0.0
        for qps in qps_grid:
            res = self.replay(method, qps, route=route, num_requests=num_requests, warmup=warmup, seed=seed)
            if res.p99_ttft_s <= slo_p99_s:
                best = float(qps)
            else:
                break
        return best
