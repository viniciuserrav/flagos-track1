# Operator design notes — `flagos-track1`

Per-operator design choices, tolerances, edge cases, and references.

## Common pattern (used by every pointwise op + `log10`)

```
# Element-wise template (Triton)
@triton.autotune(configs=24-config-grid, key=['n_elements'])
@triton.jit
def _kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=<safe_value>)
    # compute in fp32 if the op needs transcendentals; pass-through otherwise
    y = compute(x)
    tl.store(y_ptr + offs, y, mask=mask)
```

Wrappers always:
1. Promote integer inputs to fp32 (matches PyTorch).
2. Make the tensor contiguous if necessary (Triton requires it).
3. Allocate `out` if not provided; if provided, write into it and return it (aliasing-preserving).
4. Short-circuit to `torch.*` on CPU, on unsupported GPUs (sm < 70), and for fp64.

## Numerics — common tolerance table

| dtype     | rtol   | atol     | notes |
|-----------|--------|----------|-------|
| float16   | 1e-3   | 1e-3     | promoted to fp32 internally where needed |
| bfloat16  | 1e-2   | 1.6e-2   | promoted to fp32 internally where needed |
| float32   | 1e-5   | 1.3e-6   | native |
| float64   | 1e-7   | 1e-7     | routed to `torch.*` |

## Per-op notes

### `log10`
- Identity: `log10(x) = ln(x) * 0.4342944819032518` (Triton has no native `tl.log10`).
- Test set: random per dtype, edge values (`0, ±1, ±inf, nan, 1e±30`), multi-shape, `out=` aliasing, in-place `log10_`, integer promotion, empty tensor, non-contiguous view.

### `abs`
- Trivial: `tl.abs(x)`; no internal promotion (works directly in input dtype).

### `exp`
- `tl.exp(x.to(fp32)).to(orig_dtype)`. Overflow on large positives produces `inf`, matching PyTorch.

### `log`
- `tl.log(x.to(fp32)).to(orig_dtype)`. log(0) → −inf, log(neg) → nan, both match PyTorch.

### `log1p`
- `tl.log(1 + x.to(fp32)).to(orig_dtype)`. Approaches 0 cleanly for small x because addition happens in fp32.

### `sigmoid`
- `1 / (1 + exp(-x))`, computed in fp32. We don't use the `sign+abs` split trick — the fp32 path is already stable across the test ranges.

### `relu`
- `tl.maximum(x, 0)`. No internal promotion needed.

### `tanh`
- `(e^{2x} - 1) / (e^{2x} + 1)`, computed in fp32. Equivalent to PyTorch's `torch.tanh` within tolerance.

### `gelu`
- Tanh approximation: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`. Constants:
  - `sqrt(2/π) ≈ 0.7978845608028654`
- Matches `torch.nn.functional.gelu(approximate="tanh")`.

### `silu`
- `x * sigmoid(x)`, computed in fp32.

### `leaky_relu`
- `x if x >= 0 else negative_slope * x`. Slope is a `tl.constexpr` so each unique value gets its own kernel binary (cheap — leaky_relu is rarely called with many different slopes).

## Row-wise reductions (last dim)

All four reductions operate on the **last** dimension of a tensor of arbitrary rank. The input is flattened to `(M, N)` where `M = prod(shape[:-1])` and `N = shape[-1]`. Each row is handled by one program; `BLOCK_N = next_pow2(N)` clamped to ≤ 65 536. Wider rows fall back to the `torch.*` reference rather than tiling (kept simple intentionally; tiled variants come in a later pass if benchmarks warrant it).

### `softmax`
- Standard Triton single-pass online-max + sum kernel.
- Masked tail loaded as `-inf` so the max ignores padding.
- Tested for large-magnitude inputs to confirm the online-max trick prevents overflow in fp32.
- Non-last-dim requests are forwarded to `torch.softmax(x, dim=dim)`.

### `log_softmax`
- Same structure as softmax, returning `x_shift - log(sum(exp(x_shift)))`.

### `layer_norm`
- Single pass mean + variance in fp32 (Welford-equivalent for fixed row length).
- Optional affine: `weight` and `bias` consumed in fp32, output cast back to input dtype.
- `eps` flows in as a runtime float.

### `rms_norm`
- `x / sqrt(mean(x^2) + eps)`, optionally multiplied by a per-channel `weight`.
- Reference comparison uses an in-tree fp32 formulation (no canonical `torch.rms_norm` in older PyTorch wheels).
- Default `eps=1e-6` matches LLaMA / Mistral conventions; override per call.

## `matmul`

Standard Triton tutorial layout, lightly cleaned up:
- `BLOCK_M × BLOCK_N × BLOCK_K` tile with `GROUP_M=8` row-grouping swizzle to improve L2 reuse on grids that exceed the SM count.
- Autotune key is `(M, N, K)`; configs cover the 64–128 tile family with `num_warps ∈ {4, 8}` and `num_stages ∈ {2, 3}` (20 configs total).
- K-loop accumulates in `tl.float32` regardless of input dtype; output casts back to the input dtype on store.
- Non-2D inputs, dtype mismatches, and broadcasting fall through to `torch.matmul` rather than emulating PyTorch's full promotion logic — keeps the kernel scope tight.

### Tolerances
| dtype     | rtol | atol |
|-----------|------|------|
| float16   | 1e-3 | 1e-2 |
| bfloat16  | 1e-2 | 5e-2 |
| float32   | 1e-4 | 1e-4 |

Larger atol than the element-wise ops because matmul accumulates `K` products and rounding compounds.

## `fused_residual_layer_norm`

Replaces the common transformer pattern:
```
y = layer_norm(x + residual)
```
with a single kernel that reads `x` and `residual` once each, adds in fp32, computes the layer-norm mean/var/affine, and writes the output once. Saves one tensor materialization and one kernel launch per transformer block.

Same numerical contract as `layer_norm`. Shape mismatches and `normalized_shape != (last_dim,)` fall back to `torch.nn.functional.layer_norm(input + residual, …)`.
