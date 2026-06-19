## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-05-01 - Avoid High-Level Tensor Ops in Scalar Reductions
**Learning:** High-level `Tensor` operations like `sub()` and `mul()` trigger intermediate heap allocations for shape and stride metadata. When computing scalar reductions (like MSE, distances, or loss functions), using these operations introduces severe memory overhead inside hot loops. Attempting to use `.min()` length truncation as a safeguard is an anti-pattern as it masks shape mismatch errors.
**Action:** For scalar reductions, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) to eliminate intermediate allocations and safely compute the result.

## 2026-06-25 - Avoid `libm::sqrt` in Spatial Scans
**Learning:** In hot loops like `SparseAttentionGraph` spatial scans, utilizing `libm::sqrt` for distance calculations creates a significant performance overhead. Comparing exact distances is often unnecessary when boolean adjacency is the goal.
**Action:** When determining neighborhoods or filtering distances, always optimize by calculating the squared distance (`d^2`) and comparing against the squared threshold (`r^2`). Combine this with an early loop exit (`!(sum < eps_sq)`) and pre-loop validation of parameters to safely handle cases like `NaN`.
