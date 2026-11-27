"""Build and invoke the C++17 FCFS & affinity-routing event replay kernel."""

import ctypes
from pathlib import Path
import subprocess
import tempfile
from typing import Dict, Tuple

import numpy as np

ROUTE_IDS: Dict[str, int] = {
    "round-robin": 0,
    "affinity": 1,
    "hit-only": 2,
    "load-only": 3,
}

_KERNEL_CACHE: Dict[str, ctypes.CDLL] = {}
_CSRC = Path(__file__).resolve().parent / "csrc" / "fcfs.cpp"


def compile_kernel(compiler: str = "g++") -> ctypes.CDLL:
    """Compile `simulator/csrc/fcfs.cpp` into a shared library and cache the handle."""
    if compiler in _KERNEL_CACHE:
        return _KERNEL_CACHE[compiler]
    build_dir = Path(tempfile.mkdtemp(prefix="rcllm_sim_"))
    so_path = build_dir / "libfcfs.so"
    subprocess.run(
        [compiler, "-O3", "-std=c++17", "-shared", "-fPIC", str(_CSRC), "-o", str(so_path)],
        check=True,
    )
    lib = ctypes.CDLL(str(so_path))
    i64p = ctypes.POINTER(ctypes.c_int64)
    f64p = ctypes.POINTER(ctypes.c_double)
    lib.replay.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        i64p,
        f64p,
        i64p,
        f64p,
        f64p,
        f64p,
        f64p,
        i64p,
    ]
    lib.replay.restype = None
    _KERNEL_CACHE[compiler] = lib
    return lib


def run_replay(
    requests: np.ndarray,
    arrivals: np.ndarray,
    tokens: np.ndarray,
    service: np.ndarray,
    hit: np.ndarray,
    *,
    route: str = "affinity",
    kv_tokens: float = 262144.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute the C++17 nonpreemptive event replay kernel.

    Args:
        requests: int64 array of shape (N,) indexing into `tokens`, `service`, `hit`.
        arrivals: float64 array of shape (N,) arrival timestamps in seconds.
        tokens: int64 array of shape (M,) prompt lengths.
        service: float64 array of shape (M, K) per-replica service times in seconds.
        hit: float64 array of shape (M, K) per-replica item cache hit ratios in [0, 1].
        route: routing policy in {"round-robin", "affinity", "hit-only", "load-only"}.
        kv_tokens: per-replica KV capacity in tokens used for occupancy normalization.

    Returns:
        start, finish, assigned: arrays of shape (N,).
    """
    if route not in ROUTE_IDS:
        raise ValueError(f"Unknown route '{route}'; expected one of {tuple(ROUTE_IDS)}")
    req = np.ascontiguousarray(requests, dtype=np.int64)
    arr = np.ascontiguousarray(arrivals, dtype=np.float64)
    tok = np.ascontiguousarray(tokens, dtype=np.int64)
    svc = np.ascontiguousarray(service, dtype=np.float64)
    ht = np.ascontiguousarray(hit, dtype=np.float64)
    n = len(req)
    replicas = svc.shape[1]
    start = np.empty(n, dtype=np.float64)
    finish = np.empty(n, dtype=np.float64)
    assigned = np.empty(n, dtype=np.int64)
    lib = compile_kernel()
    lib.replay(
        n,
        replicas,
        ROUTE_IDS[route],
        float(kv_tokens),
        req.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        tok.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        svc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ht.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        start.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        finish.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        assigned.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    return start, finish, assigned
