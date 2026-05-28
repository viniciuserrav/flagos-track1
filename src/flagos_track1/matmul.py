"""Triton block-tile matmul.

Drop-in replacement for ``A @ B`` (``torch.matmul`` on 2D inputs) with:
- fp32 accumulator inside the K-loop,
- autotune over BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages,
- output dtype follows the input dtype (matches PyTorch).

3-D and broadcasting inputs fall back to ``torch.matmul`` — keeps the API clean
while the kernel only handles the well-tested 2-D case.
"""

from __future__ import annotations

import torch

from ._runtime import HAS_TRITON, triton, tl, use_triton_for


def _matmul_autotune_configs():
    if not HAS_TRITON:
        return []
    return [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
            num_warps=nw, num_stages=ns,
        )
        for bm, bn, bk in (
            (64, 64, 32),
            (128, 64, 32),
            (64, 128, 32),
            (128, 128, 32),
            (128, 128, 64),
        )
        for nw in (4, 8)
        for ns in (2, 3)
    ]


if HAS_TRITON:

    @triton.autotune(configs=_matmul_autotune_configs(), key=["M", "N", "K"])
    @triton.jit
    def _matmul_kernel(
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        # Swizzle program ids in groups of GROUP_M rows for better L2 reuse.
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Drop-in for 2-D ``torch.matmul`` (``a @ b``)."""
    if a.dim() != 2 or b.dim() != 2 or a.shape[1] != b.shape[0]:
        return torch.matmul(a, b)
    if not use_triton_for(a) or not use_triton_for(b):
        return torch.matmul(a, b)
    if a.dtype != b.dtype:
        return torch.matmul(a, b)
    a = a.contiguous() if not a.is_contiguous() else a
    b = b.contiguous() if not b.is_contiguous() else b
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
