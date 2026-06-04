## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-05-01 - Avoid High-Level Tensor Ops in Scalar Reductions
**Learning:** High-level `Tensor` operations like `sub()` and `mul()` trigger intermediate heap allocations for shape and stride metadata. When computing scalar reductions (like MSE, distances, or loss functions), using these operations introduces severe memory overhead inside hot loops. Attempting to use `.min()` length truncation as a safeguard is an anti-pattern as it masks shape mismatch errors.
**Action:** For scalar reductions, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) to eliminate intermediate allocations and safely compute the result.

## 2026-06-04 - Avoid libm::sqrt in Distance Comparisons
**Learning:** In `aether-core::manifold::SparseAttentionGraph::add_point`, the `is_neighbor` function is a critical hot path used to build the spatial proximity graph in $O(N)$ comparisons for each added point. The previous implementation computed the exact Euclidean distance involving an expensive `libm::sqrt` call. Since we only need to compare against a threshold (`epsilon`), computing the squared distance (`d^2 < r^2`) is sufficient and eliminates the costly square root operation.
**Action:** Always prefer squared distance comparisons over `sqrt` in hot spatial query loops. Use early exits (`!(sum < eps_sq)`) inside the summation loop to further short-circuit the computation, and handle `NaN` cleanly by comparing using `!` logic rather than explicitly matching.
