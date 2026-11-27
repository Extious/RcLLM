"""Non-contiguous pre-RoPE KV block caching and layer-wise CUDA stream prefetch."""

from typing import List, Optional, Sequence, Tuple

try:
    import torch
except ImportError:
    torch = None


class LayerKVPrefetch:
    """Layer-ordered pinned-host KV transfer on a dedicated CUDA stream.

    Layer `first_needed` (layer 1 for RcLLM+, layer 0 for RcLLM/CacheBlend) is
    scheduled first so the compute stream can overlap H2D transfers with dense
    early-layer computation.
    """

    def __init__(self, host_kv: Sequence[Tuple["torch.Tensor", "torch.Tensor"]], device, transfer_stream, first_needed: int = 1):
        if torch is None:
            raise RuntimeError("PyTorch is required for LayerKVPrefetch")
        self.layers: List[Optional[Tuple["torch.Tensor", "torch.Tensor"]]] = [None] * len(host_kv)
        self.ready: List[Optional["torch.cuda.Event"]] = [None] * len(host_kv)
        self.compute_stream = torch.cuda.current_stream(device)
        order = (first_needed, *range(first_needed), *range(first_needed + 1, len(host_kv)))
        with torch.cuda.stream(transfer_stream):
            for idx in order:
                self.layers[idx] = tuple(t.to(device, non_blocking=True) for t in host_kv[idx])
                ev = torch.cuda.Event()
                ev.record(transfer_stream)
                self.ready[idx] = ev

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        event = self.ready[layer]
        if event is not None:
            self.compute_stream.wait_event(event)
            self.ready[layer] = None
        return self.layers[layer]


def pin_kv_cache(device_kv: Sequence[Tuple["torch.Tensor", "torch.Tensor"]]) -> List[Tuple["torch.Tensor", "torch.Tensor"]]:
    """Copy a list of per-layer (K, V) tensors into pinned host memory."""
    if torch is None:
        raise RuntimeError("PyTorch is required for pin_kv_cache")
    return [
        (k.detach().to("cpu", non_blocking=False).pin_memory(), v.detach().to("cpu", non_blocking=False).pin_memory())
        for k, v in device_kv
    ]
