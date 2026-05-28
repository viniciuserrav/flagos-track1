"""Experimental operators.

These have a Triton kernel implementation that compiles and matches the torch
reference on the CPU **fallback** path, but the Triton kernel itself has not
yet been validated on a real GPU end-to-end. They are exposed so the panel can
review the implementation, but they are intentionally listed in a separate
**experimental** table in the README so users know what to trust.

Promotion to completed requires:
- The Triton kernel running on a supported GPU (T4 / A100 / H100),
- Outputs matching ``torch.*`` within the dtype tolerances,
- A microbenchmark vs ``torch.*`` committed to ``results/``.

Currently:
- ``scaled_dot_product_attention`` — FlashAttention-2-style forward kernel
  (no backward yet). Inference / forward path only.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from ._runtime import HAS_TRITON, triton, tl, use_triton_for


if HAS_TRITON:

    @triton.jit
    def _attn_fwd_kernel(
        Q, K, V, sm_scale, Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_on,
        Z, H, N_CTX,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """FlashAttention-2 forward, single-precision accumulator.

        Per-(batch, head, q-block) program tiles the Q rows of size BLOCK_M and
        streams K/V blocks of size BLOCK_N, maintaining online (max, sum)
        statistics so the softmax never materializes the full SxS attention.
        """
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        qkv_offset = off_z * stride_qz + off_h * stride_qh

        Q_block_ptr = tl.make_block_ptr(
            base=Q + qkv_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qkv_offset,
            shape=(HEAD_DIM, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qkv_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + qkv_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        q = (q * sm_scale).to(tl.float32)

        # For causal masks: only attend to keys <= q_row index.
        if IS_CAUSAL:
            n_end = (start_m + 1) * BLOCK_M
        else:
            n_end = N_CTX

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

        for start_n in range(0, n_end, BLOCK_N):
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            qk = tl.dot(q, k.to(tl.float32))

            if IS_CAUSAL:
                offs_n = start_n + tl.arange(0, BLOCK_N)
                causal_mask = offs_m[:, None] >= offs_n[None, :]
                qk = tl.where(causal_mask, qk, -float("inf"))

            # Mask out-of-context keys.
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid = offs_n[None, :] < N_CTX
            qk = tl.where(valid, qk, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            qk = qk - m_ij[:, None]
            p = tl.exp(qk)
            alpha = tl.exp(m_i - m_ij)
            l_ij = tl.sum(p, axis=1)

            acc = acc * alpha[:, None]
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
            acc = tl.dot(p.to(v.dtype), v.to(tl.float32), acc)

            l_i = l_i * alpha + l_ij
            m_i = m_ij

            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

        acc = acc / l_i[:, None]
        tl.store(O_block_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))


_SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """FlashAttention-style scaled-dot-product attention (forward only).

    Falls back to ``torch.nn.functional.scaled_dot_product_attention`` whenever:
    - inputs are on CPU,
    - GPU is sm < 70,
    - dtype is not fp16/bf16/fp32,
    - shapes don't fit the (B, H, S, D) layout with D ∈ {16, 32, 64, 128},
    - sequences are longer than the Triton kernel can tile in one launch.

    Args:
        q, k, v: shape ``(B, H, S, D)``, same dtype, contiguous.
        is_causal: apply a causal mask along the K dimension.
        scale: softmax scale (default ``1/sqrt(D)``).
    """
    if q.shape != k.shape or k.shape != v.shape:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)
    if q.dim() != 4:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)
    B, H, S, D = q.shape
    if D not in _SUPPORTED_HEAD_DIMS:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)
    if not use_triton_for(q):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    out = torch.empty_like(q_c)

    sm_scale = scale if scale is not None else 1.0 / math.sqrt(D)

    BLOCK_M = 64 if S >= 64 else 32
    BLOCK_N = 64 if S >= 64 else 32

    grid = (triton.cdiv(S, BLOCK_M), B * H)
    _attn_fwd_kernel[grid](
        q_c, k_c, v_c, sm_scale, out,
        q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
        k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
        v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
        num_warps=4 if D <= 64 else 8,
        num_stages=2,
    )
    return out
