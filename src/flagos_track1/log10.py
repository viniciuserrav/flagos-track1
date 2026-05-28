"""Triton replacement for ``torch.log10``.

Matches ``torch.log10`` within dtype-appropriate tolerances (see tests).
fp16 / bf16 inputs are promoted to fp32 inside the kernel and cast back on
store. fp32 stays in fp32. fp64 falls back to ``torch.log10`` (Triton lacks
fp64 transcendentals on T4-class hardware). Integer inputs are promoted to
fp32 first, matching PyTorch.
"""

from __future__ import annotations

import math

import torch

from ._runtime import (
    HAS_TRITON,
    launch_unary,
    pointwise_autotune_configs,
    prepare,
    triton,
    tl,
    use_triton_for,
)

RECIP_LN10 = 1.0 / math.log(10.0)  # 0.4342944819032518


if HAS_TRITON:

    @triton.autotune(configs=pointwise_autotune_configs(), key=["n_elements"])
    @triton.jit
    def _log10_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=1.0)
        y = (tl.log(x.to(tl.float32)) * 0.4342944819032518).to(x.dtype)
        tl.store(y_ptr + offs, y, mask=mask)


def log10(input: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in replacement for ``torch.log10``."""
    if not use_triton_for(input):
        return torch.log10(input, out=out) if out is not None else torch.log10(input)
    x, out = prepare(input, out)
    return launch_unary(_log10_kernel, x, out)


def log10_(input: torch.Tensor) -> torch.Tensor:
    """In-place ``log10`` — writes ``log10(input)`` back into ``input``."""
    return log10(input, out=input)
