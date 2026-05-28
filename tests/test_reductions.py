"""Correctness tests for row-wise reductions vs torch.* references."""

from __future__ import annotations

import pytest
import torch

import flagos_track1 as fg

DEVICE = "cuda" if torch.cuda.is_available() and fg.CUDA_SUPPORTED else "cpu"

TOL = {
    torch.float16:  (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1.6e-2),
    torch.float32:  (1e-5, 1.3e-6),
}

SHAPES = [(1, 17), (4, 128), (16, 768), (32, 4096), (3, 5, 13)]


# ---------------------------------------------------------------------------
# softmax / log_softmax
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("dtype", list(TOL.keys()), ids=lambda d: str(d).replace("torch.", ""))
def test_softmax_matches_torch(shape, dtype):
    rtol, atol = TOL[dtype]
    torch.manual_seed(0)
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    torch.testing.assert_close(fg.softmax(x), torch.softmax(x, dim=-1), rtol=rtol, atol=atol)


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("dtype", list(TOL.keys()), ids=lambda d: str(d).replace("torch.", ""))
def test_log_softmax_matches_torch(shape, dtype):
    rtol, atol = TOL[dtype]
    torch.manual_seed(0)
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    torch.testing.assert_close(fg.log_softmax(x), torch.log_softmax(x, dim=-1), rtol=rtol, atol=atol)


def test_softmax_large_values_no_overflow():
    """The online-max trick must protect against overflow in fp32."""
    x = torch.tensor([[1e4, 1e4 + 1, 1e4 + 2]], device=DEVICE, dtype=torch.float32)
    out = fg.softmax(x)
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(1, device=DEVICE), rtol=1e-5, atol=1e-6)


def test_softmax_non_last_dim_falls_back():
    """Asking for a non-last dim should pass through to torch.softmax."""
    x = torch.randn(3, 5, 7, device=DEVICE)
    torch.testing.assert_close(fg.softmax(x, dim=1), torch.softmax(x, dim=1))


# ---------------------------------------------------------------------------
# layer_norm
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(4, 128), (16, 768), (32, 4096), (3, 5, 768)], ids=str)
@pytest.mark.parametrize("dtype", list(TOL.keys()), ids=lambda d: str(d).replace("torch.", ""))
def test_layer_norm_matches_torch(shape, dtype):
    rtol, atol = TOL[dtype]
    torch.manual_seed(0)
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    N = shape[-1]
    w = torch.randn(N, device=DEVICE, dtype=dtype) * 0.5 + 1.0
    b = torch.randn(N, device=DEVICE, dtype=dtype) * 0.1
    ours = fg.layer_norm(x, (N,), w, b)
    ref = torch.nn.functional.layer_norm(x, (N,), w, b)
    torch.testing.assert_close(ours, ref, rtol=rtol, atol=atol)


def test_layer_norm_no_affine():
    x = torch.randn(8, 64, device=DEVICE, dtype=torch.float32)
    ours = fg.layer_norm(x, (64,))
    ref = torch.nn.functional.layer_norm(x, (64,))
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1.3e-6)


def test_layer_norm_eps():
    """Custom epsilon must propagate."""
    x = torch.randn(4, 32, device=DEVICE, dtype=torch.float32)
    for eps in (1e-5, 1e-3, 0.1):
        ours = fg.layer_norm(x, (32,), eps=eps)
        ref = torch.nn.functional.layer_norm(x, (32,), eps=eps)
        torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1.3e-6)


# ---------------------------------------------------------------------------
# rms_norm
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(4, 128), (16, 768), (32, 4096), (3, 5, 768)], ids=str)
@pytest.mark.parametrize("dtype", list(TOL.keys()), ids=lambda d: str(d).replace("torch.", ""))
def test_rms_norm_matches_reference(shape, dtype):
    rtol, atol = TOL[dtype]
    torch.manual_seed(0)
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    N = shape[-1]
    w = torch.randn(N, device=DEVICE, dtype=dtype) * 0.5 + 1.0
    ours = fg.rms_norm(x, w, eps=1e-6)
    # Reference is the torch fallback path (same formula).
    x_fp32 = x.to(torch.float32)
    rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    ref = (x_fp32 * rms).to(dtype) * w
    torch.testing.assert_close(ours, ref, rtol=rtol, atol=atol)


def test_rms_norm_no_weight():
    x = torch.randn(8, 64, device=DEVICE, dtype=torch.float32)
    ours = fg.rms_norm(x, eps=1e-6)
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    ref = x * rms
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1.3e-6)
