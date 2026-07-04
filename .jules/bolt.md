## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-05-01 - Avoid High-Level Tensor Ops in Scalar Reductions
**Learning:** High-level `Tensor` operations like `sub()` and `mul()` trigger intermediate heap allocations for shape and stride metadata. When computing scalar reductions (like MSE, distances, or loss functions), using these operations introduces severe memory overhead inside hot loops. Attempting to use `.min()` length truncation as a safeguard is an anti-pattern as it masks shape mismatch errors.
**Action:** For scalar reductions, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) to eliminate intermediate allocations and safely compute the result.

## 2026-06-20 - Fast Spatial Neighborhood Checks
**Learning:** In hot paths doing spatial scanning (like SparseAttentionGraph), calling `libm::sqrt` for distance thresholds is a massive bottleneck. Moreover, using a squared distance optimization `d^2 < r^2` must be robust to NaN values.
**Action:** Always optimize spatial neighborhood checks using squared distances (`eps_sq`) inside an inline loop. Explicitly check for invalid thresholds before the loop (e.g. `!(epsilon > 0.0)`), and use the `!(sum < eps_sq)` pattern for early exits to handle NaNs safely while matching standard Euclidean semantics without the `sqrt` overhead.

## 2026-06-25 - Replace Selection Sort with O(N) Select for KNN
**Learning:** For K-Nearest Neighbors on small, fixed-size arrays (like those bounded by `MAX_POINTS`), manually implementing `O(K * N)` selection sort is unnecessary and slow when Rust's standard library provides `select_nth_unstable_by`. It offers an `O(N)` average-time complexity using introselect. We must also remember to guard it with `k_min > 0` to prevent underflow from `k_min - 1`, and use `partial_cmp().unwrap_or(core::cmp::Ordering::Equal)` safely for floats.
**Action:** When picking top-K elements without requiring full sorted order, always favor `select_nth_unstable_by` over manual sorting or full sorting methods like `.sort_by()`.
