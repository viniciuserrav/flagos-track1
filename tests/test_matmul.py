"""Correctness tests for matmul vs ``torch.matmul``."""

from __future__ import annotations

import pytest
import torch

import flagos_track1 as fg

DEVICE = "cuda" if torch.cuda.is_available() and fg.CUDA_SUPPORTED else "cpu"

TOL = {
    torch.float16:  (1e-3, 1e-2),   # bigger atol — accumulated rounding over K
    torch.bfloat16: (1e-2, 5e-2),
    torch.float32:  (1e-4, 1e-4),
}

SHAPES = [
    (64, 32, 64),
    (128, 128, 128),
    (256, 256, 64),
    (1, 768, 768),   # row-vector case
    (33, 64, 17),    # non-power-of-two
]


@pytest.mark.parametrize("M,K,N", SHAPES, ids=lambda s: str(s))
@pytest.mark.parametrize("dtype", list(TOL.keys()), ids=lambda d: str(d).replace("torch.", ""))
def test_matmul_matches_torch(M, K, N, dtype):
    rtol, atol = TOL[dtype]
    torch.manual_seed(0)
    a = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.1
    b = torch.randn(K, N, device=DEVICE, dtype=dtype) * 0.1
    ours = fg.matmul(a, b)
    ref = torch.matmul(a, b)
    torch.testing.assert_close(ours, ref, rtol=rtol, atol=atol)


def test_matmul_non_2d_falls_back():
    """3-D batched matmul should pass through to torch."""
    a = torch.randn(4, 32, 64, device=DEVICE)
    b = torch.randn(4, 64, 16, device=DEVICE)
    torch.testing.assert_close(fg.matmul(a, b), torch.matmul(a, b))


def test_matmul_dtype_mismatch_falls_back():
    """Mismatched-dtype matmul should defer to torch (which raises — our wrapper
    must not silently produce wrong results, only either run the kernel on
    same-dtype tensors or pass through to torch's normal behavior)."""
    a = torch.randn(8, 8, device=DEVICE, dtype=torch.float32)
    b = torch.randn(8, 8, device=DEVICE, dtype=torch.float16)
    with pytest.raises(RuntimeError):
        fg.matmul(a, b)


def test_matmul_non_contiguous():
    a_base = torch.randn(64, 32, device=DEVICE, dtype=torch.float32)
    b_base = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
    a = a_base.t().contiguous().t()  # stride/permutation churn
    torch.testing.assert_close(fg.matmul(a, b_base), torch.matmul(a, b_base), rtol=1e-4, atol=1e-4)
