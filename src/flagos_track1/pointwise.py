"""Element-wise (pointwise) operators implemented in Triton.

Operators in this module:
    abs, exp, log, log1p, sigmoid, relu, tanh, gelu, silu, leaky_relu

Each ``foo`` is a drop-in replacement for ``torch.foo`` (or the matching
``torch.nn.functional`` entry where applicable). fp16/bf16 inputs are
promoted to fp32 inside the kernel and cast back on store. fp32 stays in
fp32. fp64 and unsupported GPUs route through ``torch.*``.

The ``out=`` keyword is supported (in-place via ``out=x``).
"""

from __future__ import annotations

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

if HAS_TRITON:
    _CONFIGS = pointwise_autotune_configs()

    @triton.autotune(configs=_CONFIGS, key=["n_elements"])
    @triton.jit
    def _abs_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        tl.store(y_ptr + offs, tl.abs(x), mask=mask)

    @triton.autotune(configs=_CONFIGS, key=["n_elements"])
    @triton.jit
    def _exp_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x_in = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.exp(x_in.to(tl.float32)).to(x_in.dtype)
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.autotune(configs=_CONFIGS, key=["n_elements"])
    @triton.jit
    def _log_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x_in = tl.load(x_ptr + offs, mask=mask, other=1.0)
        y = tl.log(x_in.to(tl.float32)).to(x_in.dtype)
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.autotune(configs=_CONFIGS, key=["n_elements"])
    @triton.jit
    def _log1p_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x_in = tl.load(x_ptr + offs, mask=mask, other=0.0)
        # log1p(x) = log(1 + x); compute in fp32 for numerical safety on fp16/bf16.
        y = tl.log(1.0 + x_in.to(tl.float32)).to(x_in.dtype)
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.autotune(configs=_CONFIGS, key=["n_elements"])
    @triton.jit
    def _sigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x_in = tl.load(x_ptr + offs, mask=mask, other=0.0)
        x = x_in.to(tl.float32)
        y = 1.0 / (1.0 + tl.exp(-x))
        tl.store(y_ptr + offs, y.to(x_in.dtype), mask=mask)

    @triton.autotune(configs=_CONFIGS, key=["n_elements"])
    @triton.jit
    def _relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)

    @triton.autotune(configs=_CONFIGS, key=["n_elements"])
    @triton.jit
    def _tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x_in = tl.load(x_ptr + offs, mask=mask, other=0.0)
        x = x_in.to(tl.float32)
        e = tl.exp(2.0 * x)
        y = (e - 1.0) / (e + 1.0)
        tl.store(y_ptr + offs, y.to(x_in.dtype), mask=mask)

    @triton.autotune(configs=_CONFIGS, key=["n_elements"])
    @triton.jit
    def _gelu_tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x_in = tl.load(x_ptr + offs, mask=mask, other=0.0)
        x = x_in.to(tl.float32)
        # tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        eu = tl.exp(2.0 * u)
        t = (eu - 1.0) / (eu + 1.0)
        y = 0.5 * x * (1.0 + t)
        tl.store(y_ptr + offs, y.to(x_in.dtype), mask=mask)

    @triton.autotune(configs=_CONFIGS, key=["n_elements"])
    @triton.jit
    def _silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x_in = tl.load(x_ptr + offs, mask=mask, other=0.0)
        x = x_in.to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-x))
        tl.store(y_ptr + offs, (x * sig).to(x_in.dtype), mask=mask)

    @triton.autotune(configs=_CONFIGS, key=["n_elements"])
    @triton.jit
    def _leaky_relu_kernel(x_ptr, y_ptr, n_elements, NEG_SLOPE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.where(x >= 0, x, NEG_SLOPE * x.to(tl.float32)).to(x.dtype)
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.autotune(configs=_CONFIGS, key=["n_elements"])
    @triton.jit
    def _rsqrt_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x_in = tl.load(x_ptr + offs, mask=mask, other=1.0)
        y = (1.0 / tl.sqrt(x_in.to(tl.float32))).to(x_in.dtype)
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.autotune(configs=_CONFIGS, key=["n_elements"])
    @triton.jit
    def _softplus_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        # softplus(x) = log(1 + exp(x)), with the standard overflow guard
        # softplus(x) = max(x, 0) + log(1 + exp(-|x|)).
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x_in = tl.load(x_ptr + offs, mask=mask, other=0.0)
        x = x_in.to(tl.float32)
        ax = tl.abs(x)
        y = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-ax))
        tl.store(y_ptr + offs, y.to(x_in.dtype), mask=mask)

    @triton.autotune(configs=_CONFIGS, key=["n_elements"])
    @triton.jit
    def _mish_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        # mish(x) = x * tanh(softplus(x)); softplus computed with the overflow guard above.
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x_in = tl.load(x_ptr + offs, mask=mask, other=0.0)
        x = x_in.to(tl.float32)
        ax = tl.abs(x)
        sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-ax))
        e = tl.exp(2.0 * sp)
        t = (e - 1.0) / (e + 1.0)
        y = x * t
        tl.store(y_ptr + offs, y.to(x_in.dtype), mask=mask)


def _fallback_unary(torch_fn, x: torch.Tensor, out: torch.Tensor | None):
    if out is None:
        return torch_fn(x)
    return torch_fn(x, out=out)


def abs(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in replacement for ``torch.abs`` (integer dtypes promoted to fp32)."""
    if not use_triton_for(x):
        return _fallback_unary(torch.abs, x, out)
    x, out = prepare(x, out)
    return launch_unary(_abs_kernel, x, out)


def exp(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in replacement for ``torch.exp``."""
    if not use_triton_for(x):
        return _fallback_unary(torch.exp, x, out)
    x, out = prepare(x, out)
    return launch_unary(_exp_kernel, x, out)


def log(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in replacement for ``torch.log``."""
    if not use_triton_for(x):
        return _fallback_unary(torch.log, x, out)
    x, out = prepare(x, out)
    return launch_unary(_log_kernel, x, out)


def log1p(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in replacement for ``torch.log1p`` — log(1 + x), numerically stable near 0."""
    if not use_triton_for(x):
        return _fallback_unary(torch.log1p, x, out)
    x, out = prepare(x, out)
    return launch_unary(_log1p_kernel, x, out)


def sigmoid(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in replacement for ``torch.sigmoid``."""
    if not use_triton_for(x):
        return _fallback_unary(torch.sigmoid, x, out)
    x, out = prepare(x, out)
    return launch_unary(_sigmoid_kernel, x, out)


def relu(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in replacement for ``torch.relu``."""
    if not use_triton_for(x):
        return _fallback_unary(torch.relu, x, out)
    x, out = prepare(x, out)
    return launch_unary(_relu_kernel, x, out)


def tanh(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in replacement for ``torch.tanh``."""
    if not use_triton_for(x):
        return _fallback_unary(torch.tanh, x, out)
    x, out = prepare(x, out)
    return launch_unary(_tanh_kernel, x, out)


def gelu(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in for ``torch.nn.functional.gelu(approximate='tanh')``."""
    if not use_triton_for(x):
        ref = torch.nn.functional.gelu(x, approximate="tanh")
        if out is not None:
            out.copy_(ref)
            return out
        return ref
    x, out = prepare(x, out)
    return launch_unary(_gelu_tanh_kernel, x, out)


def silu(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in for ``torch.nn.functional.silu`` (x * sigmoid(x))."""
    if not use_triton_for(x):
        ref = torch.nn.functional.silu(x)
        if out is not None:
            out.copy_(ref)
            return out
        return ref
    x, out = prepare(x, out)
    return launch_unary(_silu_kernel, x, out)


def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in for ``torch.nn.functional.leaky_relu``."""
    if not use_triton_for(x):
        ref = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
        if out is not None:
            out.copy_(ref)
            return out
        return ref
    x, out = prepare(x, out)
    n = x.numel()
    if n == 0:
        return out
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _leaky_relu_kernel[grid](x, out, n, float(negative_slope))
    return out


def rsqrt(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in for ``torch.rsqrt`` (reciprocal square root)."""
    if not use_triton_for(x):
        return _fallback_unary(torch.rsqrt, x, out)
    x, out = prepare(x, out)
    return launch_unary(_rsqrt_kernel, x, out)


def softplus(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in for ``torch.nn.functional.softplus`` (with the standard overflow guard)."""
    if not use_triton_for(x):
        ref = torch.nn.functional.softplus(x)
        if out is not None:
            out.copy_(ref)
            return out
        return ref
    x, out = prepare(x, out)
    return launch_unary(_softplus_kernel, x, out)


def mish(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in for ``torch.nn.functional.mish`` — ``x * tanh(softplus(x))``."""
    if not use_triton_for(x):
        ref = torch.nn.functional.mish(x)
        if out is not None:
            out.copy_(ref)
            return out
        return ref
    x, out = prepare(x, out)
    return launch_unary(_mish_kernel, x, out)
