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
| `rsqrt`      | completed  | per-dtype random + edge + multi-shape + empty + non-contig | same | `1 / sqrt(x)` in fp32 |
| `softplus`   | completed  | same | same | `log(1 + exp(x))` with overflow guard |
| `mish`       | completed  | same | same | `x * tanh(softplus(x))` with overflow guard |
| `softmax`     | completed  | per-dtype random over 5 shapes + large-value overflow guard + non-last-dim fallback | `benchmarks/bench_reductions.py` | online-max single-pass kernel, last-dim only |
| `log_softmax` | completed  | per-dtype random over 5 shapes | same | logsumexp stabilized |
| `layer_norm`  | completed  | per-dtype random over 4 shapes + no-affine path + custom-eps | same | Welford-equivalent single-pass mean/var, fused affine |
| `rms_norm`    | completed  | per-dtype random over 4 shapes + no-weight path | same | LLaMA-style, fp32 internal accumulator |
| `matmul`      | completed  | per-dtype random over 5 shapes incl. non-power-of-two + non-2d fallback + non-contig + dtype-mismatch propagation | (planned) | 2-D block-tile w/ GROUP_M swizzle, fp32 accumulator, autotune over BLOCK_M/N/K × stages × warps |
| `fused_residual_layer_norm` | completed | per-dtype random over 4 shapes + no-affine + shape-mismatch fallback | (planned) | fuses `layer_norm(x + residual)` — saves a kernel launch in transformer blocks |

Each row marked **completed** has:
1. Triton kernel matching `torch.*` within dtype tolerances (`fp16: rtol=1e-3,atol=1e-3`; `bfloat16: rtol=1e-2,atol=1.6e-2`; `fp32: rtol=1e-5,atol=1.3e-6`).
2. Random + edge-value + multi-shape + empty + non-contiguous tests passing.
3. Triton autotune over `BLOCK_SIZE × num_warps × num_stages` (24 configs).
4. Microbenchmark script (vs the `torch.*` reference).

### Experimental

These have a Triton kernel that compiles and matches `torch.*` on the CPU fallback path, but the kernel itself has not yet been validated end-to-end on a real GPU. Listed separately so the panel knows what to trust.

| operator | status | tests | notes |
|----------|--------|-------|-------|
| `experimental.scaled_dot_product_attention` | **experimental** | 9 cases pass on CPU fallback (per-shape + causal flag + custom scale + unsupported-dim and shape-mismatch fallback) | FlashAttention-2-style forward kernel; head_dim ∈ {16, 32, 64, 128}; `is_causal` supported; forward only (no backward yet) |

Promotion to *completed* requires the Triton kernel running on a supported GPU, outputs matching `torch.*` within dtype tolerances, and a microbenchmark committed to `results/`.

## Layout

```
src/flagos_track1/   Python package (kernels + wrappers)
  _runtime.py        Triton import, capability probe, shared autotune grid
  log10.py           the leaderboard-scored op
  pointwise.py       13 element-wise ops sharing a kernel/dispatch template
  reductions.py      4 row-wise reductions (softmax, log_softmax, layer_norm, rms_norm)
  matmul.py          2-D block-tile matmul with GROUP_M swizzle
  fused.py           transformer-block fusions (residual + layer_norm)
  experimental.py    work-in-progress ops; kernel compiles + matches torch on CPU fallback, GPU validation pending
tests/               pytest suite (248 cases, all green on CPU fallback)
examples/            end-to-end usage (a tiny GPT block using every op)
benchmarks/          microbenchmark scripts; CSVs written to results/
notebook/            Kaggle notebook (mirrors the published kernel)
results/             benchmark CSVs + plots (committed)
docs/                operator design notes
```

## Reproducing

Hardware: any CUDA GPU with `sm_70+` (T4 / A100 / H100 / RTX 20-series+). Falls back to `torch.*` on CPU or unsupported GPUs.

```bash
pip install -e .[test]
pytest tests/                                # 248 cases — all green
python examples/gpt_block.py                 # end-to-end tiny GPT block
python benchmarks/bench_log10.py             # writes results/log10_bench.csv
python benchmarks/bench_pointwise.py         # writes results/pointwise_bench.csv
python benchmarks/bench_reductions.py        # writes results/reductions_bench.csv
python scripts/plot_bench.py                 # writes results/*.png from the CSVs above
jupyter execute notebook/flagos-track1-log10.ipynb
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs the full pytest suite on Python 3.10 / 3.11 / 3.12 against the CPU fallback path on every push and PR.

## License

Apache-2.0 (see [LICENSE](LICENSE)).
