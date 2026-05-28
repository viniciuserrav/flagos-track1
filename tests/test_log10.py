"""Correctness tests for ``flagos_track1.log10`` vs ``torch.log10``.

Skips Triton paths automatically when no CUDA device is available — the same
tests then exercise the torch fallback path.
"""

import math

import pytest
import torch

from flagos_track1 import log10, log10_

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOL = {
    torch.float16:  (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1.6e-2),
    torch.float32:  (1e-5, 1.3e-6),
    torch.float64:  (1e-7, 1e-7),
}


@pytest.mark.parametrize("dtype", list(TOL.keys()))
def test_random_tensor(dtype):
    rtol, atol = TOL[dtype]
    torch.manual_seed(0)
    x = torch.rand(1024, 1024, device=DEVICE, dtype=dtype) + 0.1
    torch.testing.assert_close(log10(x), torch.log10(x), rtol=rtol, atol=atol, equal_nan=True)


def test_edge_values():
    x = torch.tensor(
        [0.0, -1.0, 1.0, 10.0, 1e-30, 1e30, math.inf, -math.inf, math.nan],
        device=DEVICE, dtype=torch.float32,
    )
    torch.testing.assert_close(log10(x), torch.log10(x), rtol=1e-5, atol=1.3e-6, equal_nan=True)


@pytest.mark.parametrize("shape", [(1,), (1, 1), (33,), (3, 17, 5), (128, 256), (1024, 1024)])
def test_multi_shape(shape):
    x = torch.rand(shape, device=DEVICE, dtype=torch.float32) + 0.1
    torch.testing.assert_close(log10(x), torch.log10(x), rtol=1e-5, atol=1.3e-6)


def test_out_kwarg_aliases():
    x = torch.rand(257, device=DEVICE, dtype=torch.float32) + 0.1
    out = torch.empty_like(x)
    ret = log10(x, out=out)
    assert ret.data_ptr() == out.data_ptr(), "out= path must alias to the provided tensor"
    torch.testing.assert_close(out, torch.log10(x), rtol=1e-5, atol=1.3e-6)


def test_in_place():
    z = torch.rand(257, device=DEVICE, dtype=torch.float32) + 0.1
    ref = torch.log10(z)
    log10_(z)
    torch.testing.assert_close(z, ref, rtol=1e-5, atol=1.3e-6)


def test_integer_promotion():
    ints = torch.arange(1, 11, device=DEVICE)
    torch.testing.assert_close(log10(ints), torch.log10(ints.float()), rtol=1e-5, atol=1.3e-6)


def test_empty_tensor():
    x = torch.empty(0, device=DEVICE, dtype=torch.float32)
    assert log10(x).numel() == 0


def test_non_contiguous():
    base = torch.rand(64, 128, device=DEVICE, dtype=torch.float32) + 0.1
    x = base.t()  # non-contiguous view
    torch.testing.assert_close(log10(x), torch.log10(x), rtol=1e-5, atol=1.3e-6)
