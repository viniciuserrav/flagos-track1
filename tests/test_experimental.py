"""Tests for experimental ops.

On CPU these exercise the **fallback path** (which is just ``torch.*``), so
they're trivially correct. They still gate the public-API surface against
regressions and exercise the dispatch logic that decides between kernel and
fallback. The Triton kernel itself is validated on GPU outside CI for now.
"""

from __future__ import annotations

import math

import pytest
import torch

import flagos_track1 as fg
from flagos_track1.experimental import scaled_dot_product_attention


DEVICE = "cuda" if torch.cuda.is_available() and fg.CUDA_SUPPORTED else "cpu"


# Tighter tolerances on CPU because both paths are torch SDPA itself.
TOL = {
    torch.float32: (1e-5, 1e-5),
}


@pytest.mark.parametrize("B,H,S,D", [
    (1, 4, 32, 32),
    (2, 4, 64, 64),
    (1, 8, 16, 128),
])
@pytest.mark.parametrize("dtype", list(TOL.keys()), ids=lambda d: str(d).replace("torch.", ""))
@pytest.mark.parametrize("is_causal", [False, True])
def test_sdpa_matches_torch_fallback(B, H, S, D, dtype, is_causal):
    rtol, atol = TOL[dtype]
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)

    ours = scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    torch.testing.assert_close(ours, ref, rtol=rtol, atol=atol)


def test_sdpa_custom_scale_passes_through():
    q = torch.randn(1, 2, 16, 32, device=DEVICE, dtype=torch.float32)
    k = torch.randn(1, 2, 16, 32, device=DEVICE, dtype=torch.float32)
    v = torch.randn(1, 2, 16, 32, device=DEVICE, dtype=torch.float32)
    custom = 0.05
    ours = scaled_dot_product_attention(q, k, v, scale=custom)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=custom)
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-5)


def test_sdpa_unsupported_head_dim_falls_back():
    """HEAD_DIM ∉ {16, 32, 64, 128} should defer to torch SDPA on GPU too."""
    q = torch.randn(1, 2, 16, 48, device=DEVICE, dtype=torch.float32)  # 48 isn't supported
    k = torch.randn(1, 2, 16, 48, device=DEVICE, dtype=torch.float32)
    v = torch.randn(1, 2, 16, 48, device=DEVICE, dtype=torch.float32)
    ours = scaled_dot_product_attention(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-5)


def test_sdpa_shape_mismatch_falls_back():
    q = torch.randn(1, 2, 16, 32, device=DEVICE, dtype=torch.float32)
    k = torch.randn(1, 2, 8, 32, device=DEVICE, dtype=torch.float32)
    v = torch.randn(1, 2, 8, 32, device=DEVICE, dtype=torch.float32)
    ours = scaled_dot_product_attention(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-5)
