## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-05-01 - Avoid High-Level Tensor Ops in Scalar Reductions
**Learning:** High-level `Tensor` operations like `sub()` and `mul()` trigger intermediate heap allocations for shape and stride metadata. When computing scalar reductions (like MSE, distances, or loss functions), using these operations introduces severe memory overhead inside hot loops. Attempting to use `.min()` length truncation as a safeguard is an anti-pattern as it masks shape mismatch errors.
**Action:** For scalar reductions, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) to eliminate intermediate allocations and safely compute the result.

## 2026-05-30 - Optimize Euclidean Distance in Hot Spatial Scans
**Learning:** `libm::sqrt` is relatively slow, especially when computing exact distances repeatedly within hot inner loops, such as `SparseAttentionGraph::add_point` which performs O(N) spatial scans using `ManifoldPoint::is_neighbor`.
**Action:** Replace `is_neighbor` calculations with inline, squared distance comparisons (`d^2 < r^2`). Explicitly reject non-positive or `NaN` thresholds (e.g., `!(epsilon > 0.0)`), and utilize early loop exits formatted as `!(sum < eps_sq)` to safely manage `NaN` coordinates without altering behavior, effectively avoiding the `sqrt` bottleneck.
