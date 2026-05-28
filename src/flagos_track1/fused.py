"""Fused operators that combine two common transformer ops in a single kernel.

Currently:
- ``fused_residual_layer_norm`` — ``layer_norm(x + residual)``. Reads ``x`` and
  ``residual`` once each, computes the affine and norm in one pass, writes once.
  Saves the materialization of the intermediate sum and a kernel launch.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from ._runtime import HAS_TRITON, triton, tl, use_triton_for


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1)).bit_length()


if HAS_TRITON:

    @triton.jit
    def _fused_residual_ln_kernel(
        x_ptr, r_ptr, w_ptr, b_ptr, y_ptr,
        M, N, eps,
        stride_xm, stride_rm, stride_ym,
        HAS_AFFINE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + row * stride_rm + cols, mask=mask, other=0.0).to(tl.float32)
        s = x + r

        n_valid = tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
        sum_s = tl.sum(tl.where(mask, s, 0.0), axis=0)
        mean = sum_s / n_valid
        diff = tl.where(mask, s - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / n_valid
        inv = 1.0 / tl.sqrt(var + eps)
        y = (s - mean) * inv

        if HAS_AFFINE:
            w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = y * w + b

        # Cast to the *input* dtype on store (we read x's dtype implicitly above).
        # Triton infers store dtype from the output pointer.
        tl.store(y_ptr + row * stride_ym + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def fused_residual_layer_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """``layer_norm(input + residual)`` fused into a single kernel.

    Equivalent to::

        torch.nn.functional.layer_norm(input + residual, normalized_shape, weight, bias, eps)

    Args:
        input, residual: same shape, same dtype, broadcast not supported here.
        normalized_shape: tuple/list whose only element is ``input.shape[-1]``.
        weight, bias: optional affine.
        eps: numerical stability constant.
    """
    if input.shape != residual.shape:
        return torch.nn.functional.layer_norm(input + residual, normalized_shape, weight, bias, eps)
    if len(normalized_shape) != 1 or normalized_shape[0] != input.shape[-1]:
        return torch.nn.functional.layer_norm(input + residual, normalized_shape, weight, bias, eps)
    if not use_triton_for(input) or input.shape[-1] == 0:
        return torch.nn.functional.layer_norm(input + residual, normalized_shape, weight, bias, eps)

    has_affine = weight is not None and bias is not None
    input = input.contiguous()
    residual = residual.contiguous()

    orig_shape = input.shape
    x2d = input.reshape(-1, orig_shape[-1])
    r2d = residual.reshape(-1, orig_shape[-1])
    M, N = x2d.shape
    BLOCK_N = _next_pow2(N)
    if BLOCK_N > 65536:
        return torch.nn.functional.layer_norm(input + residual, normalized_shape, weight, bias, eps)
    out = torch.empty_like(x2d)

    if has_affine:
        w = weight.contiguous()
        b = bias.contiguous()
    else:
        w = torch.empty(N, device=input.device, dtype=input.dtype)
        b = torch.empty(N, device=input.device, dtype=input.dtype)

    _fused_residual_ln_kernel[(M,)](
        x2d, r2d, w, b, out,
        M, N, float(eps),
        x2d.stride(0), r2d.stride(0), out.stride(0),
        HAS_AFFINE=has_affine,
        BLOCK_N=BLOCK_N,
        num_warps=4 if BLOCK_N <= 2048 else 8,
    )
    return out.reshape(orig_shape)
