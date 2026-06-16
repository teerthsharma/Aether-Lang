## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-05-01 - Avoid High-Level Tensor Ops in Scalar Reductions
**Learning:** High-level `Tensor` operations like `sub()` and `mul()` trigger intermediate heap allocations for shape and stride metadata. When computing scalar reductions (like MSE, distances, or loss functions), using these operations introduces severe memory overhead inside hot loops. Attempting to use `.min()` length truncation as a safeguard is an anti-pattern as it masks shape mismatch errors.
**Action:** For scalar reductions, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) to eliminate intermediate allocations and safely compute the result.

## 2026-06-16 - Optimize Hot Distance Checks with Squared Distance
**Learning:** In `SparseAttentionGraph::add_point`, the `is_neighbor` check is called inside a hot loop (O(N) scans). Using standard Euclidean distance calculation involves a costly `libm::sqrt` operation. For pure distance comparisons `d < epsilon`, this can be optimized by comparing squared values `d^2 < epsilon^2`, completely avoiding the square root calculation and allowing for early loop termination.
**Action:** In pure distance comparisons against a threshold, use squared distance and an early exit condition `!(sum < eps_sq)`. Always ensure negative or NaN thresholds are rejected upfront `!(epsilon > 0.0)`.
