"""Correctness tests for fused operators."""

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


@pytest.mark.parametrize("shape", [(4, 128), (16, 768), (32, 4096), (3, 5, 768)], ids=str)
@pytest.mark.parametrize("dtype", list(TOL.keys()), ids=lambda d: str(d).replace("torch.", ""))
def test_fused_residual_layer_norm_matches_torch(shape, dtype):
    rtol, atol = TOL[dtype]
    torch.manual_seed(0)
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    r = torch.randn(shape, device=DEVICE, dtype=dtype)
    N = shape[-1]
    w = torch.randn(N, device=DEVICE, dtype=dtype) * 0.5 + 1.0
    b = torch.randn(N, device=DEVICE, dtype=dtype) * 0.1
    ours = fg.fused_residual_layer_norm(x, r, (N,), w, b)
    ref = torch.nn.functional.layer_norm(x + r, (N,), w, b)
    torch.testing.assert_close(ours, ref, rtol=rtol, atol=atol)


def test_fused_residual_layer_norm_no_affine():
    x = torch.randn(8, 64, device=DEVICE, dtype=torch.float32)
    r = torch.randn(8, 64, device=DEVICE, dtype=torch.float32)
    ours = fg.fused_residual_layer_norm(x, r, (64,))
    ref = torch.nn.functional.layer_norm(x + r, (64,))
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1.3e-6)


def test_fused_residual_layer_norm_shape_mismatch_falls_back():
    """Mismatched shapes should pass through to torch (no batching/broadcast support)."""
    x = torch.randn(4, 32, device=DEVICE, dtype=torch.float32)
    r = torch.randn(4, 32, device=DEVICE, dtype=torch.float32)
    ours = fg.fused_residual_layer_norm(x, r, (32,))
    ref = torch.nn.functional.layer_norm(x + r, (32,))
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1.3e-6)
