"""Shared runtime helpers for flagos_track1 operators.

Centralizes:
- Triton import + USE_TRITON gate (works on CPU fallback too).
- The autotune config grid we reuse across pointwise kernels.
- A capability probe so unsupported GPUs (e.g. Kaggle's Tesla P100, sm_60)
  fall back to torch cleanly.
"""

from __future__ import annotations

import os
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False
    triton = None
    tl = None

CUDA = torch.cuda.is_available()


def _probe_cuda_supported() -> bool:
    if not CUDA:
        return False
    try:
        torch.zeros(1, device="cuda")
        cap = torch.cuda.get_device_capability(0)
        return cap[0] >= 7
    except Exception:
        return False


CUDA_SUPPORTED = _probe_cuda_supported()
USE_TRITON = HAS_TRITON and CUDA_SUPPORTED and os.environ.get("FLAGOS_FORCE_TORCH", "") != "1"

_SUPPORTED_TRITON_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def use_triton_for(x: torch.Tensor) -> bool:
    return USE_TRITON and x.is_cuda and x.dtype in _SUPPORTED_TRITON_DTYPES


def pointwise_autotune_configs():
    """24-config grid covering BLOCK_SIZE x num_warps x num_stages."""
    if not HAS_TRITON:
        return []
    return [
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in (1024, 2048, 4096, 8192)
        for nw in (4, 8)
        for ns in (2, 3)
    ]


def prepare(x: torch.Tensor, out: torch.Tensor | None):
    """Promote integer inputs to fp32 (matches PyTorch), make contiguous, allocate out."""
    if not x.is_floating_point():
        x = x.to(torch.float32)
    if not x.is_contiguous():
        x = x.contiguous()
    if out is None:
        out = torch.empty_like(x)
    return x, out


def launch_unary(kernel, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    if n == 0:
        return out
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    kernel[grid](x, out, n)
    return out
