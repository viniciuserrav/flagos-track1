"""Plot benchmark CSVs as a single figure per file.

Reads any of:
  results/log10_bench.csv
  results/pointwise_bench.csv
  results/reductions_bench.csv

Writes a matching PNG next to each.

Usage:
    python scripts/plot_bench.py [--dir results/]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _plot(df: pd.DataFrame, out_path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    x_col = "elements" if "elements" in df.columns else "N"
    if "op" in df.columns:
        for (op, dtype), sub in df.groupby(["op", "dtype"]):
            sub = sub.sort_values(x_col)
            ax.plot(sub[x_col], sub["speedup"], marker="o", label=f"{op} | {dtype}")
    else:
        for dtype, sub in df.groupby("dtype"):
            sub = sub.sort_values(x_col)
            ax.plot(sub[x_col], sub["speedup"], marker="o", label=str(dtype))
    ax.set_xscale("log", base=2)
    ax.set_xlabel(x_col)
    ax.set_ylabel("speedup vs torch (>1 is faster)")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results")
    args = p.parse_args()
    root = Path(args.dir)
    if not root.exists():
        raise SystemExit(f"no such directory: {root}")
    candidates = {
        "log10_bench.csv": "log10 — Triton vs torch.log10",
        "pointwise_bench.csv": "Pointwise ops — Triton vs torch",
        "reductions_bench.csv": "Reductions — Triton vs torch",
    }
    for fname, title in candidates.items():
        path = root / fname
        if not path.exists():
            print(f"skipping {path} (not found)")
            continue
        df = pd.read_csv(path)
        _plot(df, path.with_suffix(".png"), title)


if __name__ == "__main__":
    main()
