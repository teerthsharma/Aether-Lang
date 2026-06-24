## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-05-01 - Avoid High-Level Tensor Ops in Scalar Reductions
**Learning:** High-level `Tensor` operations like `sub()` and `mul()` trigger intermediate heap allocations for shape and stride metadata. When computing scalar reductions (like MSE, distances, or loss functions), using these operations introduces severe memory overhead inside hot loops. Attempting to use `.min()` length truncation as a safeguard is an anti-pattern as it masks shape mismatch errors.
**Action:** For scalar reductions, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) to eliminate intermediate allocations and safely compute the result.

## 2026-06-20 - Fast Spatial Neighborhood Checks
**Learning:** In hot paths doing spatial scanning (like SparseAttentionGraph), calling `libm::sqrt` for distance thresholds is a massive bottleneck. Moreover, using a squared distance optimization `d^2 < r^2` must be robust to NaN values.
**Action:** Always optimize spatial neighborhood checks using squared distances (`eps_sq`) inside an inline loop. Explicitly check for invalid thresholds before the loop (e.g. `!(epsilon > 0.0)`), and use the `!(sum < eps_sq)` pattern for early exits to handle NaNs safely while matching standard Euclidean semantics without the `sqrt` overhead.

## 2026-06-24 - Zero-Copy Backpropagation via Option::take()
**Learning:** In `aether-core`, `Tensor` structs contain heap-allocated metadata (like shape and strides). Operations like `last_z.as_ref().unwrap().clone()` in `DenseLayer::backward` trigger redundant heap allocations and slow down the training loop, even when the data itself is reference-counted.
**Action:** When a cached tensor is only needed once during the backward pass (e.g., `last_z` and `last_input`), use `Option::take()` instead of `as_ref().clone()`. This transfers ownership out of the cache without cloning the metadata, saving memory allocations during backpropagation.
