## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-05-01 - Avoid High-Level Tensor Ops in Scalar Reductions
**Learning:** High-level `Tensor` operations like `sub()` and `mul()` trigger intermediate heap allocations for shape and stride metadata. When computing scalar reductions (like MSE, distances, or loss functions), using these operations introduces severe memory overhead inside hot loops. Attempting to use `.min()` length truncation as a safeguard is an anti-pattern as it masks shape mismatch errors.
**Action:** For scalar reductions, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) to eliminate intermediate allocations and safely compute the result.

## 2026-06-20 - Fast Spatial Neighborhood Checks
**Learning:** In hot paths doing spatial scanning (like SparseAttentionGraph), calling `libm::sqrt` for distance thresholds is a massive bottleneck. Moreover, using a squared distance optimization `d^2 < r^2` must be robust to NaN values.
**Action:** Always optimize spatial neighborhood checks using squared distances (`eps_sq`) inside an inline loop. Explicitly check for invalid thresholds before the loop (e.g. `!(epsilon > 0.0)`), and use the `!(sum < eps_sq)` pattern for early exits to handle NaNs safely while matching standard Euclidean semantics without the `sqrt` overhead.
## 2026-06-21 - Optimization in DBSCAN and Auto-K Selection
**Learning:** In hot loops such as `auto_k_selection` and `DBSCAN::region_query`, avoiding `libm::sqrt` by using squared distances (`d^2 < r^2`) significantly reduces mathematical overhead and speeds up the algorithms. Also, explicitly rejecting negative or NaN thresholds before the loop (e.g., `!(epsilon > 0.0)`) and using early exits within the loop formatted as `!(sum < eps_sq)` safely handles `NaN` values without altering behavioral logic.
**Action:** Whenever iterating over distance computations that compare against a threshold, prefer comparing the sum of squared differences directly against the squared threshold instead of taking the square root. Handle edge cases like invalid distances properly.
