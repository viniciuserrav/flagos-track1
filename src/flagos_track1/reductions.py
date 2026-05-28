"""Row-wise reductions: softmax, log_softmax, layer_norm, rms_norm.

All four operate on the **last** dimension of the input (matches the PyTorch
default for `softmax(dim=-1)`, `nn.LayerNorm(normalized_shape=(last,))`, etc.).

Single-pass kernels:
- ``softmax`` / ``log_softmax``: online max + sum (Triton's standard pattern).
- ``layer_norm``: Welford-equivalent mean and var (single pass over the row).
- ``rms_norm``: sum-of-squares + rsqrt.

fp16 / bf16 inputs are promoted to fp32 inside the kernel and cast back on
store. fp32 stays in fp32. fp64 and unsupported GPUs route through ``torch.*``.

Each row's width is rounded up to the next power of two and clamped into the
kernel as ``BLOCK_N`` (``tl.constexpr``); rows up to 64K elements wide are
supported without further tiling.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from ._runtime import HAS_TRITON, triton, tl, use_triton_for


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1)).bit_length()


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
if HAS_TRITON:

    @triton.jit
    def _softmax_kernel(x_ptr, y_ptr, M, N, stride_xm, stride_ym,
                        BLOCK_N: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x_row = x_ptr + row * stride_xm + cols
        y_row = y_ptr + row * stride_ym + cols
        # Load with -inf padding so the masked tail doesn't bias the max.
        x_in = tl.load(x_row, mask=mask, other=-float("inf"))
        x = x_in.to(tl.float32)
        m = tl.max(x, axis=0)
        e = tl.exp(x - m)
        e = tl.where(mask, e, 0.0)
        s = tl.sum(e, axis=0)
        y = e / s
        tl.store(y_row, y.to(x_in.dtype), mask=mask)

    @triton.jit
    def _log_softmax_kernel(x_ptr, y_ptr, M, N, stride_xm, stride_ym,
                            BLOCK_N: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x_row = x_ptr + row * stride_xm + cols
        y_row = y_ptr + row * stride_ym + cols
        x_in = tl.load(x_row, mask=mask, other=-float("inf"))
        x = x_in.to(tl.float32)
        m = tl.max(x, axis=0)
        x_shift = x - m
        e = tl.exp(x_shift)
        e = tl.where(mask, e, 0.0)
        lse = tl.log(tl.sum(e, axis=0))
        y = x_shift - lse
        tl.store(y_row, y.to(x_in.dtype), mask=mask)

    @triton.jit
    def _layer_norm_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                           M, N, eps,
                           stride_xm, stride_ym,
                           HAS_AFFINE: tl.constexpr,
                           BLOCK_N: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x_row = x_ptr + row * stride_xm + cols
        y_row = y_ptr + row * stride_ym + cols
        x_in = tl.load(x_row, mask=mask, other=0.0)
        x = x_in.to(tl.float32)
        # Mean and variance over the (masked) row in fp32.
        n_valid = tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
        sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
        mean = sum_x / n_valid
        diff = tl.where(mask, x - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / n_valid
        inv = 1.0 / tl.sqrt(var + eps)
        y = (x - mean) * inv
        if HAS_AFFINE:
            w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = y * w + b
        tl.store(y_row, y.to(x_in.dtype), mask=mask)

    @triton.jit
    def _rms_norm_kernel(x_ptr, w_ptr, y_ptr,
                         M, N, eps,
                         stride_xm, stride_ym,
                         HAS_WEIGHT: tl.constexpr,
                         BLOCK_N: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x_row = x_ptr + row * stride_xm + cols
        y_row = y_ptr + row * stride_ym + cols
        x_in = tl.load(x_row, mask=mask, other=0.0)
        x = x_in.to(tl.float32)
        n_valid = tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
        sq = tl.where(mask, x * x, 0.0)
        rms = tl.sqrt(tl.sum(sq, axis=0) / n_valid + eps)
        y = x / rms
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            y = y * w
        tl.store(y_row, y.to(x_in.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
def _flatten_to_2d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Collapse leading dims into a single batch dim. Returns (x2d, orig_shape)."""
    orig_shape = x.shape
    if x.dim() == 0:
        x = x.reshape(1, 1)
    elif x.dim() == 1:
        x = x.reshape(1, -1)
    else:
        x = x.reshape(-1, orig_shape[-1])
    return x, orig_shape


def softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Drop-in replacement for ``torch.softmax(x, dim=-1)`` (last-dim only)."""
    if dim != -1 and dim != input.dim() - 1:
        return torch.softmax(input, dim=dim)
    if not use_triton_for(input) or input.shape[-1] == 0:
        return torch.softmax(input, dim=-1)
    x2d, orig_shape = _flatten_to_2d(input.contiguous())
    M, N = x2d.shape
    BLOCK_N = _next_pow2(N)
    if BLOCK_N > 65536:
        return torch.softmax(input, dim=-1)
    out = torch.empty_like(x2d)
    _softmax_kernel[(M,)](
        x2d, out, M, N,
        x2d.stride(0), out.stride(0),
        BLOCK_N=BLOCK_N,
        num_warps=4 if BLOCK_N <= 2048 else 8,
    )
    return out.reshape(orig_shape)


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Drop-in replacement for ``torch.log_softmax(x, dim=-1)`` (last-dim only)."""
    if dim != -1 and dim != input.dim() - 1:
        return torch.log_softmax(input, dim=dim)
    if not use_triton_for(input) or input.shape[-1] == 0:
        return torch.log_softmax(input, dim=-1)
    x2d, orig_shape = _flatten_to_2d(input.contiguous())
    M, N = x2d.shape
    BLOCK_N = _next_pow2(N)
    if BLOCK_N > 65536:
        return torch.log_softmax(input, dim=-1)
    out = torch.empty_like(x2d)
    _log_softmax_kernel[(M,)](
        x2d, out, M, N,
        x2d.stride(0), out.stride(0),
        BLOCK_N=BLOCK_N,
        num_warps=4 if BLOCK_N <= 2048 else 8,
    )
    return out.reshape(orig_shape)


def layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Drop-in for ``torch.nn.functional.layer_norm`` (last-dim-only normalization)."""
    if len(normalized_shape) != 1 or normalized_shape[0] != input.shape[-1]:
        return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)
    has_affine = weight is not None and bias is not None
    if not use_triton_for(input) or input.shape[-1] == 0:
        return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)
    x2d, orig_shape = _flatten_to_2d(input.contiguous())
    M, N = x2d.shape
    BLOCK_N = _next_pow2(N)
    if BLOCK_N > 65536:
        return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)
    out = torch.empty_like(x2d)
    if has_affine:
        w = weight.contiguous()
        b = bias.contiguous()
    else:
        # Provide non-null pointers even though HAS_AFFINE=False; the kernel
        # will not dereference them.
        w = torch.empty(N, device=input.device, dtype=input.dtype)
        b = torch.empty(N, device=input.device, dtype=input.dtype)
    _layer_norm_kernel[(M,)](
        x2d, w, b, out,
        M, N, float(eps),
        x2d.stride(0), out.stride(0),
        HAS_AFFINE=has_affine,
        BLOCK_N=BLOCK_N,
        num_warps=4 if BLOCK_N <= 2048 else 8,
    )
    return out.reshape(orig_shape)


def rms_norm(
    input: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """LLaMA-style RMS norm: ``(x / sqrt(mean(x^2) + eps)) * weight``."""
    has_weight = weight is not None
    ref = lambda: _torch_rms_norm(input, weight, eps)
    if not use_triton_for(input) or input.shape[-1] == 0:
        return ref()
    x2d, orig_shape = _flatten_to_2d(input.contiguous())
    M, N = x2d.shape
    BLOCK_N = _next_pow2(N)
    if BLOCK_N > 65536:
        return ref()
    out = torch.empty_like(x2d)
    if has_weight:
        w = weight.contiguous()
    else:
        w = torch.empty(N, device=input.device, dtype=input.dtype)
    _rms_norm_kernel[(M,)](
        x2d, w, out,
        M, N, float(eps),
        x2d.stride(0), out.stride(0),
        HAS_WEIGHT=has_weight,
        BLOCK_N=BLOCK_N,
        num_warps=4 if BLOCK_N <= 2048 else 8,
    )
    return out.reshape(orig_shape)


def _torch_rms_norm(input: torch.Tensor, weight: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    x_fp32 = input.to(torch.float32)
    rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = (x_fp32 * rms).to(input.dtype)
    if weight is not None:
        y = y * weight
    return y
