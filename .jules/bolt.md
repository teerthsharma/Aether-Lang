## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-05-01 - Avoid High-Level Tensor Ops in Scalar Reductions
**Learning:** High-level `Tensor` operations like `sub()` and `mul()` trigger intermediate heap allocations for shape and stride metadata. When computing scalar reductions (like MSE, distances, or loss functions), using these operations introduces severe memory overhead inside hot loops. Attempting to use `.min()` length truncation as a safeguard is an anti-pattern as it masks shape mismatch errors.
**Action:** For scalar reductions, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) to eliminate intermediate allocations and safely compute the result.

## 2026-05-01 - Avoid sqrt in Hot Distance Thresholding Loops
**Learning:** In operations that scan distances for thresholding (like `SparseAttentionGraph::add_point` bounding the neighborhood radius `epsilon`), calculating the true Euclidean distance with `libm::sqrt` is a significant bottleneck.
**Action:** Replace `distance < epsilon` checks with an inline squared distance loop `sum(d^2) < epsilon^2`. Include an early exit condition `!(sum < eps_sq)` inside the loop to short-circuit the calculation, and safely handle edge cases like `!(epsilon > 0.0)` at the beginning.
