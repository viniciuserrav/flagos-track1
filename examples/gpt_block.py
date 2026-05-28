"""Tiny GPT-style transformer block using ``flagos_track1`` ops in place of torch ones.

Drop-in demo: builds a single transformer block (LayerNorm -> attention proj
(matmul) -> softmax -> output proj (matmul) -> fused-residual LayerNorm -> MLP
with gelu) and compares its output to a pure-torch reference implementation
running the same math.

The point is not benchmarking (that's in ``benchmarks/``); it's showing every
exported op being used together end-to-end and matching the reference inside
the documented tolerances.

Run:
    python examples/gpt_block.py
"""

from __future__ import annotations

import torch

import flagos_track1 as fg


def reference_block(x: torch.Tensor, w_qkv, b_qkv, w_o, b_o, w_ln, b_ln,
                    w_fc1, b_fc1, w_fc2, b_fc2, w_ln2, b_ln2,
                    n_heads: int) -> torch.Tensor:
    """Pure-torch reference using the same math the Triton block runs."""
    B, S, D = x.shape
    H = n_heads
    Dh = D // H

    # 1. LayerNorm
    y = torch.nn.functional.layer_norm(x, (D,), w_ln, b_ln)

    # 2. QKV projection
    qkv = torch.matmul(y, w_qkv) + b_qkv  # (B, S, 3D)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.reshape(B, S, H, Dh).transpose(1, 2)  # (B, H, S, Dh)
    k = k.reshape(B, S, H, Dh).transpose(1, 2)
    v = v.reshape(B, S, H, Dh).transpose(1, 2)

    # 3. Attention (uses ref torch since SDPA isn't a flagos_track1 op yet)
    attn = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)
    attn = torch.softmax(attn, dim=-1)
    y_attn = torch.matmul(attn, v)               # (B, H, S, Dh)
    y_attn = y_attn.transpose(1, 2).reshape(B, S, D)

    # 4. Output projection + residual + LayerNorm (fused)
    y_out = torch.matmul(y_attn, w_o) + b_o
    y_resid = torch.nn.functional.layer_norm(x + y_out, (D,), w_ln2, b_ln2)

    # 5. MLP with gelu (tanh approx)
    h = torch.nn.functional.gelu(torch.matmul(y_resid, w_fc1) + b_fc1, approximate="tanh")
    return torch.matmul(h, w_fc2) + b_fc2


def flagos_block(x: torch.Tensor, w_qkv, b_qkv, w_o, b_o, w_ln, b_ln,
                 w_fc1, b_fc1, w_fc2, b_fc2, w_ln2, b_ln2,
                 n_heads: int) -> torch.Tensor:
    """The same block, but every supported op is the flagos_track1 version.

    Ops used: layer_norm, matmul, softmax, fused_residual_layer_norm, gelu.
    Attention itself still uses torch.matmul over Q K V; that's the next op
    on the roadmap and the only piece not yet covered by this package.
    """
    B, S, D = x.shape
    H = n_heads
    Dh = D // H

    # 1. LayerNorm (flagos)
    y = fg.layer_norm(x, (D,), w_ln, b_ln)

    # 2. QKV projection (flagos matmul + bias add)
    qkv = fg.matmul(y.reshape(B * S, D), w_qkv).reshape(B, S, 3 * D) + b_qkv
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.reshape(B, S, H, Dh).transpose(1, 2).contiguous()
    k = k.reshape(B, S, H, Dh).transpose(1, 2).contiguous()
    v = v.reshape(B, S, H, Dh).transpose(1, 2).contiguous()

    # 3. Attention — Q K V multiplies stay torch for now; only softmax is flagos.
    attn = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)
    attn = fg.softmax(attn)                                                  # flagos
    y_attn = torch.matmul(attn, v)
    y_attn = y_attn.transpose(1, 2).contiguous().reshape(B, S, D)

    # 4. Output projection (flagos matmul) + fused residual + layer_norm
    y_out = fg.matmul(y_attn.reshape(B * S, D), w_o).reshape(B, S, D) + b_o
    y_resid = fg.fused_residual_layer_norm(x, y_out, (D,), w_ln2, b_ln2)     # flagos

    # 5. MLP using flagos matmul + gelu
    h = fg.gelu(fg.matmul(y_resid.reshape(B * S, D), w_fc1).reshape(B, S, 4 * D) + b_fc1)
    out = fg.matmul(h.reshape(B * S, 4 * D), w_fc2).reshape(B, S, D) + b_fc2
    return out


def main():
    torch.manual_seed(0)
    DEVICE = "cuda" if torch.cuda.is_available() and fg.CUDA_SUPPORTED else "cpu"
    print(f"device: {DEVICE}  HAS_TRITON={fg.HAS_TRITON}  USE_TRITON={fg.USE_TRITON}")

    B, S, D, H = 2, 16, 64, 4
    x = torch.randn(B, S, D, device=DEVICE, dtype=torch.float32)

    w_qkv = torch.randn(D, 3 * D, device=DEVICE, dtype=torch.float32) * 0.05
    b_qkv = torch.zeros(3 * D, device=DEVICE, dtype=torch.float32)
    w_o   = torch.randn(D, D,     device=DEVICE, dtype=torch.float32) * 0.05
    b_o   = torch.zeros(D,         device=DEVICE, dtype=torch.float32)
    w_ln  = torch.ones(D, device=DEVICE, dtype=torch.float32)
    b_ln  = torch.zeros(D, device=DEVICE, dtype=torch.float32)
    w_ln2 = torch.ones(D, device=DEVICE, dtype=torch.float32)
    b_ln2 = torch.zeros(D, device=DEVICE, dtype=torch.float32)
    w_fc1 = torch.randn(D, 4 * D, device=DEVICE, dtype=torch.float32) * 0.05
    b_fc1 = torch.zeros(4 * D, device=DEVICE, dtype=torch.float32)
    w_fc2 = torch.randn(4 * D, D, device=DEVICE, dtype=torch.float32) * 0.05
    b_fc2 = torch.zeros(D, device=DEVICE, dtype=torch.float32)

    ref = reference_block(x, w_qkv, b_qkv, w_o, b_o, w_ln, b_ln,
                          w_fc1, b_fc1, w_fc2, b_fc2, w_ln2, b_ln2, n_heads=H)
    ours = flagos_block(x, w_qkv, b_qkv, w_o, b_o, w_ln, b_ln,
                        w_fc1, b_fc1, w_fc2, b_fc2, w_ln2, b_ln2, n_heads=H)

    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-4)
    diff = (ours - ref).abs()
    print(f"shapes match  : {ours.shape}")
    print(f"max abs diff  : {diff.max().item():.3e}")
    print(f"mean abs diff : {diff.mean().item():.3e}")
    print("flagos_block matches the reference within rtol=1e-4, atol=1e-4")


if __name__ == "__main__":
    main()
