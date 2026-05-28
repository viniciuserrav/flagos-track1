"""Triton replacement for ``torch.log10``.

Matches ``torch.log10`` within dtype-appropriate tolerances (see tests).
fp16 / bf16 inputs are promoted to fp32 inside the kernel and cast back on
store. fp32 stays in fp32. fp64 falls back to ``torch.log10`` (Triton lacks
fp64 transcendentals on T4-class hardware). Integer inputs are promoted to
fp32 first, matching PyTorch.
"""

from __future__ import annotations

import math
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
USE_TRITON = HAS_TRITON and CUDA and os.environ.get("FLAGOS_FORCE_TORCH", "") != "1"

RECIP_LN10 = 1.0 / math.log(10.0)  # 0.4342944819032518


def _autotune_configs():
    if not HAS_TRITON:
        return []
    return [
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in (1024, 2048, 4096, 8192)
        for nw in (4, 8)
        for ns in (2, 3)
    ]


if HAS_TRITON:

    @triton.autotune(configs=_autotune_configs(), key=["n_elements"])
    @triton.jit
    def _log10_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=1.0)
        y = (tl.log(x.to(tl.float32)) * 0.4342944819032518).to(x.dtype)
        tl.store(y_ptr + offs, y, mask=mask)


def log10(input: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in replacement for ``torch.log10``.

    Args:
        input: any floating-point or integer tensor. Integer inputs are promoted to
            fp32 (matching PyTorch). fp64 routes to ``torch.log10``.
        out: optional output tensor of the same dtype/shape as ``input`` after
            promotion. If provided, written in place and returned.
    """
    if not input.is_floating_point():
        input = input.to(torch.float32)
    if not (USE_TRITON and input.is_cuda and input.dtype in (torch.float16, torch.bfloat16, torch.float32)):
        return torch.log10(input, out=out) if out is not None else torch.log10(input)
    if not input.is_contiguous():
        input = input.contiguous()
    if out is None:
        out = torch.empty_like(input)
    n = input.numel()
    if n == 0:
        return out
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _log10_kernel[grid](input, out, n)
    return out


def log10_(input: torch.Tensor) -> torch.Tensor:
    """In-place ``log10`` — writes ``log10(input)`` back into ``input``."""
    return log10(input, out=input)
