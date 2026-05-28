# flagos-track1

Kaggle [Track1 — LLM Operator Development and Optimization](https://www.kaggle.com/competitions/track1-operator-development-and-optimization-flagos-challenge) entry.
Triton operator implementations validated against `torch.*` references, with reproducible benchmarks.

## Status

| operator     | status     | tests | benchmark | notes |
|--------------|------------|-------|-----------|-------|
| `log10`      | completed  | 16 cases pass vs `torch.log10` (fp16/bf16/fp32, edge values, multi-shape, `out=`, in-place, int promotion, empty, non-contiguous) | `benchmarks/bench_log10.py` | leaderboard-scored op |
| `abs`        | completed  | per-dtype random + edge + multi-shape + empty + non-contig | `benchmarks/bench_pointwise.py` | no internal promotion |
| `exp`        | completed  | same | same | fp32 internal compute |
| `log`        | completed  | same (domain-restricted to non-negatives for edge test) | same | fp32 internal compute |
| `log1p`      | completed  | same | same | log(1 + x), stable near 0 |
| `sigmoid`    | completed  | same | same | fp32 internal compute |
| `relu`       | completed  | same | same | no internal promotion |
| `tanh`       | completed  | same | same | `(e^{2x}-1)/(e^{2x}+1)` in fp32 |
| `gelu`       | completed  | same | same | tanh approximation, matches `torch.nn.functional.gelu(approximate='tanh')` |
| `silu`       | completed  | same | same | `x * sigmoid(x)` |
| `leaky_relu` | completed  | same + custom-slope test | same | dispatches `negative_slope` to the kernel |
| `softmax`     | completed  | per-dtype random over 5 shapes + large-value overflow guard + non-last-dim fallback | `benchmarks/bench_reductions.py` | online-max single-pass kernel, last-dim only |
| `log_softmax` | completed  | per-dtype random over 5 shapes | same | logsumexp stabilized |
| `layer_norm`  | completed  | per-dtype random over 4 shapes + no-affine path + custom-eps | same | Welford-equivalent single-pass mean/var, fused affine |
| `rms_norm`    | completed  | per-dtype random over 4 shapes + no-weight path | same | LLaMA-style, fp32 internal accumulator |

Each row marked **completed** has:
1. Triton kernel matching `torch.*` within dtype tolerances (`fp16: rtol=1e-3,atol=1e-3`; `bfloat16: rtol=1e-2,atol=1.6e-2`; `fp32: rtol=1e-5,atol=1.3e-6`).
2. Random + edge-value + multi-shape + empty + non-contiguous tests passing.
3. Triton autotune over `BLOCK_SIZE × num_warps × num_stages` (24 configs).
4. Microbenchmark script (vs the `torch.*` reference).

Experimental work-in-progress ops will be listed in a separate table once they exist.

## Layout

```
src/flagos_track1/   Python package (kernels + wrappers)
  _runtime.py        Triton import, capability probe, shared autotune grid
  log10.py           the leaderboard-scored op
  pointwise.py       10 element-wise ops sharing a kernel/dispatch template
  reductions.py      4 row-wise reductions (softmax, log_softmax, layer_norm, rms_norm)
tests/               pytest suite (176 cases, all green on CPU fallback)
benchmarks/          microbenchmark scripts; CSVs written to results/
notebook/            Kaggle notebook (mirrors the published kernel)
results/             benchmark CSVs + plots (committed)
docs/                operator design notes
```

## Reproducing

Hardware: any CUDA GPU with `sm_70+` (T4 / A100 / H100 / RTX 20-series+). Falls back to `torch.*` on CPU or unsupported GPUs.

```bash
pip install -e .[test]
pytest tests/                                # 176 cases — all green
python benchmarks/bench_log10.py             # writes results/log10_bench.csv
python benchmarks/bench_pointwise.py         # writes results/pointwise_bench.csv
python benchmarks/bench_reductions.py        # writes results/reductions_bench.csv
jupyter execute notebook/flagos-track1-log10.ipynb
```

## License

Apache-2.0 (see [LICENSE](LICENSE)).
