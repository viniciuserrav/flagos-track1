"""Benchmark every pointwise op in ``flagos_track1.pointwise`` vs its torch reference.

Writes ``results/pointwise_bench.csv`` with columns:
    op, dtype, elements, triton_ms, torch_ms, speedup, bandwidth_gbs

Usage:
    python benchmarks/bench_pointwise.py [--output results/pointwise_bench.csv]
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

SIZES = [1 << k for k in (16, 20, 22, 24)]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]

OPS = [
    ("abs",        fg.abs,        torch.abs,                                                 -2.0, 2.0),
    ("exp",        fg.exp,        torch.exp,                                                 -2.0, 2.0),
    ("log",        fg.log,        torch.log,                                                  0.1, 2.0),
    ("log1p",      fg.log1p,      torch.log1p,                                               -0.5, 2.0),
    ("sigmoid",    fg.sigmoid,    torch.sigmoid,                                             -4.0, 4.0),
    ("relu",       fg.relu,       torch.relu,                                                -2.0, 2.0),
    ("tanh",       fg.tanh,       torch.tanh,                                                -3.0, 3.0),
    ("gelu",       fg.gelu,       lambda x: torch.nn.functional.gelu(x, approximate="tanh"), -3.0, 3.0),
    ("silu",       fg.silu,       torch.nn.functional.silu,                                  -3.0, 3.0),
    ("leaky_relu", fg.leaky_relu, torch.nn.functional.leaky_relu,                            -2.0, 2.0),
]


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
    p.add_argument("--output", default="results/pointwise_bench.csv")
    args = p.parse_args()

    if not fg.USE_TRITON:
        print("Triton path not active (CPU run, Triton unavailable, or unsupported GPU). "
              "Only torch fallback path is being measured.")

    rows = []
    for name, our_fn, ref_fn, low, high in OPS:
        for dtype in DTYPES:
            for n in SIZES:
                x = torch.rand(n, device=DEVICE, dtype=dtype) * (high - low) + low
                our_fn(x); ref_fn(x)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                tri_ms = do_bench(lambda: our_fn(x))
                ref_ms = do_bench(lambda: ref_fn(x))
                bytes_per_elem = x.element_size() * 2
                rows.append({
                    "op": name,
                    "dtype": str(dtype).replace("torch.", ""),
                    "elements": n,
                    "triton_ms": tri_ms,
                    "torch_ms": ref_ms,
                    "speedup": ref_ms / tri_ms if tri_ms > 0 else float("nan"),
                    "bandwidth_gbs": (bytes_per_elem * n) / (tri_ms * 1e-3) / 1e9 if tri_ms > 0 else float("nan"),
                })

    bench = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    bench.to_csv(args.output, index=False)

    print(bench.to_string(index=False, max_rows=60))
    print(f"\nwrote {args.output}")

    geom = bench.groupby(["op", "dtype"])["speedup"].apply(
        lambda s: float(np.exp(np.log(s).mean()))
    )
    print("\ngeomean speedup vs torch per op×dtype:")
    print(geom.round(3).to_string())


if __name__ == "__main__":
    main()
