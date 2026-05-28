"""Correctness tests for pointwise operators vs the torch reference."""

from __future__ import annotations

import math

import pytest
import torch

import flagos_track1 as fg

DEVICE = "cuda" if torch.cuda.is_available() and fg.CUDA_SUPPORTED else "cpu"

TOL = {
    torch.float16:  (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1.6e-2),
    torch.float32:  (1e-5, 1.3e-6),
}

def _rand_in(shape, dtype, *, low: float, high: float) -> torch.Tensor:
    """Uniform on [low, high)."""
    return torch.rand(shape, device=DEVICE, dtype=dtype) * (high - low) + low


# (name, our_fn, torch_ref_fn, low, high) — input range chosen per op's valid domain.
CASES = [
    ("abs",        fg.abs,        torch.abs,                                                 -2.0, 2.0),
    ("exp",        fg.exp,        torch.exp,                                                 -2.0, 2.0),
    ("log",        fg.log,        torch.log,                                                  0.1, 2.0),
    ("log1p",      fg.log1p,      torch.log1p,                                               -0.5, 2.0),
    ("sigmoid",    fg.sigmoid,    torch.sigmoid,                                             -4.0, 4.0),
    ("relu",       fg.relu,       torch.relu,                                                -2.0, 2.0),
    ("tanh",       fg.tanh,       torch.tanh,                                                -3.0, 3.0),
    ("gelu",       fg.gelu,       lambda x: torch.nn.functional.gelu(x, approximate="tanh"), -3.0, 3.0),
    ("silu",       fg.silu,       torch.nn.functional.silu,                                  -3.0, 3.0),
    ("leaky_relu", fg.leaky_relu, torch.nn.functional.leaky_relu,                            -2.0, 2.0),
]


@pytest.mark.parametrize("name,our_fn,ref_fn,low,high", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("dtype", list(TOL.keys()), ids=lambda d: str(d).replace("torch.", ""))
def test_random(name, our_fn, ref_fn, low, high, dtype):
    rtol, atol = TOL[dtype]
    torch.manual_seed(42)
    x = _rand_in((1024, 1024), dtype, low=low, high=high)
    torch.testing.assert_close(our_fn(x), ref_fn(x), rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("name,our_fn,ref_fn,low,high", CASES, ids=[c[0] for c in CASES])
def test_edge_values(name, our_fn, ref_fn, low, high):
    base = [0.0, 1.0, -1.0, 1e-30, 1e30, math.inf, -math.inf, math.nan]
    x = torch.tensor(base, device=DEVICE, dtype=torch.float32)
    if name == "log":
        x = x.abs()  # PyTorch defines log(negative) as NaN; both paths agree but skip to keep test focused
    torch.testing.assert_close(our_fn(x), ref_fn(x), rtol=1e-5, atol=1.3e-6, equal_nan=True)


@pytest.mark.parametrize("name,our_fn,ref_fn,low,high", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("shape", [(1,), (33,), (3, 17, 5), (128, 256)])
def test_multi_shape(name, our_fn, ref_fn, low, high, shape):
    x = _rand_in(shape, torch.float32, low=low, high=high)
    torch.testing.assert_close(our_fn(x), ref_fn(x), rtol=1e-5, atol=1.3e-6, equal_nan=True)


@pytest.mark.parametrize("name,our_fn,ref_fn,low,high", CASES, ids=[c[0] for c in CASES])
def test_empty_tensor(name, our_fn, ref_fn, low, high):
    x = torch.empty(0, device=DEVICE, dtype=torch.float32)
    assert our_fn(x).numel() == 0


@pytest.mark.parametrize("name,our_fn,ref_fn,low,high", CASES, ids=[c[0] for c in CASES])
def test_non_contiguous(name, our_fn, ref_fn, low, high):
    base = _rand_in((64, 128), torch.float32, low=low, high=high)
    x = base.t()  # non-contiguous view
    torch.testing.assert_close(our_fn(x), ref_fn(x), rtol=1e-5, atol=1.3e-6, equal_nan=True)


def test_leaky_relu_negative_slope():
    """Custom slope argument must propagate through to the kernel."""
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=DEVICE, dtype=torch.float32)
    for slope in (0.0, 0.01, 0.1, 0.2):
        ours = fg.leaky_relu(x, negative_slope=slope)
        ref = torch.nn.functional.leaky_relu(x, negative_slope=slope)
        torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1.3e-6)
