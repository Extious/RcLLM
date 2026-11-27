"""Calibrated C++17 cluster simulator for RcLLM and RcLLM+ (TPDS final edition)."""

from simulator.calibration import StageCoefficients, analytical_decoder_flops, fit_stage_model, predict_service_ms
from simulator.cluster import ClusterReplayResult, ClusterSimulator, RequestTraceRow
from simulator.kernel import ROUTE_IDS, compile_kernel, run_replay
from simulator.placement import ItemPlacement, build_metis_or_hash_placement, load_archived_layout

__all__ = [
    "ClusterReplayResult",
    "ClusterSimulator",
    "ItemPlacement",
    "ROUTE_IDS",
    "RequestTraceRow",
    "StageCoefficients",
    "analytical_decoder_flops",
    "build_metis_or_hash_placement",
    "compile_kernel",
    "fit_stage_model",
    "load_archived_layout",
    "predict_service_ms",
    "run_replay",
]
