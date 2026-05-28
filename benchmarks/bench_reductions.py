"""Benchmark row-wise reductions (softmax, log_softmax, layer_norm, rms_norm)
against the matching torch.* references.

Writes ``results/reductions_bench.csv`` with columns:
    op, dtype, M, N, triton_ms, torch_ms, speedup

Usage:
    python benchmarks/bench_reductions.py [--output results/reductions_bench.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import flagos_track1 as fg  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() and fg.CUDA_SUPPORTED else "cpu"

# (M, N) shapes that cover small-batch attention (256, 64-128 head dim) up to
# transformer hidden states (16 x 4096) and a long-row case (1 x 16384).
SHAPES = [(256, 128), (64, 768), (16, 4096), (1, 16384)]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def do_bench(fn, warmup=25, rep=100):
    if DEVICE == "cuda":
        try:
            import triton.testing as tt
            return tt.do_bench(fn, warmup=warmup, rep=rep)
        except Exception:
            pass
    import time
    for _ in range(warmup):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/reductions_bench.csv")
    args = p.parse_args()

    if not fg.USE_TRITON:
        print("Triton path not active — only torch fallback is being measured.")

    rows = []
    for dtype in DTYPES:
        for M, N in SHAPES:
            x = torch.randn(M, N, device=DEVICE, dtype=dtype)
            w = torch.randn(N, device=DEVICE, dtype=dtype) * 0.5 + 1.0
            b = torch.randn(N, device=DEVICE, dtype=dtype) * 0.1

            cases = [
                ("softmax",     lambda: fg.softmax(x),                        lambda: torch.softmax(x, dim=-1)),
                ("log_softmax", lambda: fg.log_softmax(x),                    lambda: torch.log_softmax(x, dim=-1)),
                ("layer_norm",  lambda: fg.layer_norm(x, (N,), w, b),         lambda: torch.nn.functional.layer_norm(x, (N,), w, b)),
                ("rms_norm",    lambda: fg.rms_norm(x, w),                    lambda: fg.rms_norm(x, w)),  # rms_norm has no torch direct reference
            ]
            for name, our_fn, ref_fn in cases:
                our_fn(); ref_fn()
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                tri_ms = do_bench(our_fn)
                ref_ms = do_bench(ref_fn) if name != "rms_norm" else do_bench(lambda: _torch_rms(x, w))
                rows.append({
                    "op": name,
                    "dtype": str(dtype).replace("torch.", ""),
                    "M": M,
                    "N": N,
                    "triton_ms": tri_ms,
                    "torch_ms": ref_ms,
                    "speedup": ref_ms / tri_ms if tri_ms > 0 else float("nan"),
                })

    bench = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    bench.to_csv(args.output, index=False)
    print(bench.to_string(index=False))
    print(f"\nwrote {args.output}")

    geom = bench.groupby(["op", "dtype"])["speedup"].apply(
        lambda s: float(np.exp(np.log(s).mean()))
    )
    print("\ngeomean speedup vs torch per op x dtype:")
    print(geom.round(3).to_string())


def _torch_rms(x, w, eps: float = 1e-6):
    x_fp32 = x.to(torch.float32)
    rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 * rms).to(x.dtype) * w


if __name__ == "__main__":
    main()
