"""Smoke test for the end-to-end example.

Imports the GPT-block example and runs it to confirm that the full
flagos_track1 stack (layer_norm, matmul, softmax, fused_residual_layer_norm,
gelu) composes cleanly and produces the same output as a pure-torch reference.
This gates regressions where a single op's API drifts and silently breaks the
composition path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

import flagos_track1 as fg


def _load_example():
    path = Path(__file__).resolve().parents[1] / "examples" / "gpt_block.py"
    spec = importlib.util.spec_from_file_location("gpt_block_example", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpt_block_example"] = module
    spec.loader.exec_module(module)
    return module


def test_gpt_block_matches_reference():
    mod = _load_example()
    device = "cuda" if torch.cuda.is_available() and fg.CUDA_SUPPORTED else "cpu"
    torch.manual_seed(123)

    B, S, D, H = 2, 8, 32, 2
    x = torch.randn(B, S, D, device=device, dtype=torch.float32)

    def _zero(*shape):
        return torch.zeros(*shape, device=device, dtype=torch.float32)

    def _rand(*shape, scale=0.05):
        return torch.randn(*shape, device=device, dtype=torch.float32) * scale

    args = (
        x,
        _rand(D, 3 * D),
        _zero(3 * D),
        _rand(D, D),
        _zero(D),
        torch.ones(D, device=device),
        _zero(D),
        _rand(D, 4 * D),
        _zero(4 * D),
        _rand(4 * D, D),
        _zero(D),
        torch.ones(D, device=device),
        _zero(D),
    )

    ref = mod.reference_block(*args, n_heads=H)
    ours = mod.flagos_block(*args, n_heads=H)
    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-4)
