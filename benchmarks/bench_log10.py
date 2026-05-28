"""Benchmark ``flagos_track1.log10`` vs ``torch.log10`` across element counts.

Writes ``results/log10_bench.csv`` with columns:
    elements, dtype, triton_ms, torch_ms, speedup, bandwidth_gbs

Usage:
    python benchmarks/bench_log10.py [--output results/log10_bench.csv]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flagos_track1 import log10, USE_TRITON  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIZES = [1 << k for k in (10, 13, 16, 18, 20, 22, 24, 26)]
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
    p.add_argument("--output", default="results/log10_bench.csv")
    args = p.parse_args()

    if not USE_TRITON:
        print("Triton path not active (CPU run or Triton unavailable). "
              "The benchmark only measures the torch fallback path.")

    rows = []
    for dtype in DTYPES:
        for n in SIZES:
            x = (torch.rand(n, device=DEVICE, dtype=dtype) + 0.1)
            log10(x); torch.log10(x)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            tri_ms = do_bench(lambda: log10(x))
            ref_ms = do_bench(lambda: torch.log10(x))
            bytes_per_elem = x.element_size() * 2
            rows.append({
                "elements": n,
                "dtype": str(dtype).replace("torch.", ""),
                "triton_ms": tri_ms,
                "torch_ms": ref_ms,
                "speedup": ref_ms / tri_ms if tri_ms > 0 else float("nan"),
                "bandwidth_gbs": (bytes_per_elem * n) / (tri_ms * 1e-3) / 1e9 if tri_ms > 0 else float("nan"),
            })

    bench = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    bench.to_csv(args.output, index=False)
    print(bench.to_string(index=False))
    print(f"\nwrote {args.output}")

    grouped = bench.groupby("dtype")["speedup"].apply(lambda s: float(np.exp(np.log(s).mean())))
    print("\ngeomean speedup vs torch.log10 per dtype:")
    print(grouped.round(3).to_string())


if __name__ == "__main__":
    main()
