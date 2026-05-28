# Operator design notes — `flagos-track1`

This file tracks per-operator design choices, tolerances, edge cases, and
references. Each completed operator has an entry; experimental work-in-progress
operators get added once they exist.

---

## `log10`

**Reference:** [`torch.log10`](https://docs.pytorch.org/docs/stable/generated/torch.log10.html)

**Identity used by the kernel:**
```
log10(x) = ln(x) * 0.4342944819032518   # 1 / ln(10)
```

**Why this identity:** Triton exposes `tl.log` (natural log) but no native
`tl.log10`. Multiplying by `1/ln(10)` lets us reuse the well-optimized natural
log without sacrificing precision (the constant is representable exactly enough
in fp32 to stay within fp32 tolerance).

### Numerics

| dtype | rtol | atol | notes |
|-------|------|------|-------|
| float16 | 1e-3 | 1e-3 | promoted to fp32 internally, cast back on store |
| bfloat16 | 1e-2 | 1.6e-2 | promoted to fp32 internally, cast back on store |
| float32 | 1e-5 | 1.3e-6 | native |
| float64 | 1e-7 | 1e-7 | falls back to `torch.log10` (Triton lacks fp64 transcendentals on T4-class HW) |

### Edge cases tested

- `0.0` → `-inf` (matches PyTorch)
- negative finite → `nan` (matches PyTorch)
- `inf` → `inf`
- `-inf` → `nan`
- `nan` → `nan`
- subnormals (1e-30) and very large (1e30) — both within rtol/atol

### Performance

- Autotuned over `BLOCK_SIZE ∈ {1024, 2048, 4096, 8192}`, `num_warps ∈ {4, 8}`, `num_stages ∈ {2, 3}` → 24 configs, picked per shape.
- Single masked load + masked store; the `tl.log` call dominates.
- Effective bandwidth (`bytes = 2 * n * sizeof(dtype)`) reported in `results/log10_bench.csv` once the bench runs on T4.

### Memory contract

- Input is made contiguous via `.contiguous()` if needed (Triton requires it).
- `out=` keyword: result is written into the provided tensor and that same
  tensor is returned (aliasing-preserving — tested in `tests/test_log10.py::test_out_kwarg_aliases`).
- In-place `log10_(x)` calls `log10(x, out=x)`.
