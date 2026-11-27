"""Unit tests for the C++17 cluster simulator, placement, and calibration."""

import unittest

import numpy as np

from rcllm.config import MODEL_SPECS
from simulator import (
    ClusterSimulator,
    analytical_decoder_flops,
    build_metis_or_hash_placement,
    predict_service_ms,
    run_replay,
)
from simulator.cli import make_synthetic_trace


class TestSimulator(unittest.TestCase):
    def test_cpp_kernel_single_replica_fcfs(self):
        requests = np.array([0, 0, 0], dtype=np.int64)
        arrivals = np.array([0.0, 0.1, 0.5], dtype=np.float64)
        tokens = np.array([4096], dtype=np.int64)
        service = np.array([[0.25]], dtype=np.float64)
        hit = np.array([[1.0]], dtype=np.float64)
        start, finish, assigned = run_replay(requests, arrivals, tokens, service, hit, route="affinity")
        np.testing.assert_allclose(start, [0.0, 0.25, 0.5])
        np.testing.assert_allclose(finish, [0.25, 0.50, 0.75])
        np.testing.assert_array_equal(assigned, [0, 0, 0])

    def test_placement_hot_replication(self):
        popularity = {f"item_{i}": 1000 - i for i in range(100)}
        placement = build_metis_or_hash_placement(popularity, replicas=4, hot_fraction=0.05)
        self.assertEqual(len(placement.hot_items), 5)
        for shard in placement.shards:
            self.assertTrue(placement.hot_items.issubset(shard))

    def test_calibrated_service_ordering_and_flops(self):
        full_ms = predict_service_ms("qwen3", "Full", 5000, 128, 0)
        cb_ms = predict_service_ms("qwen3", "CacheBlend", 5000, 128, 4200)
        rc_ms = predict_service_ms("qwen3", "RcLLM+", 5000, 128, 4200)
        self.assertLess(rc_ms, cb_ms)
        self.assertLess(cb_ms, full_ms)

        spec = MODEL_SPECS["qwen3"]
        active = [5000, 5000] + [1200] * 23 + [400] * 11
        _, ratio = analytical_decoder_flops(spec, 5000, active)
        self.assertTrue(0.15 < ratio < 0.45)

    def test_cluster_replay_all_routes(self):
        rows, popularity = make_synthetic_trace(num_rows=64, catalog_size=300, seed=7)
        placement = build_metis_or_hash_placement(popularity, replicas=8, hot_fraction=0.01)
        sim = ClusterSimulator("qwen3", rows, placement)
        res_full = sim.replay("Full", qps=10.0, num_requests=300, warmup=50, seed=7)
        res_rc = sim.replay("RcLLM+", qps=10.0, route="affinity", num_requests=300, warmup=50, seed=7)
        self.assertLess(res_rc.p99_ttft_s, res_full.p99_ttft_s)
        self.assertGreater(res_rc.mean_hit_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
