## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

<<<<<<< HEAD
## 2026-05-01 - Avoid High-Level Tensor Ops in Scalar Reductions
**Learning:** High-level `Tensor` operations like `sub()` and `mul()` trigger intermediate heap allocations for shape and stride metadata. When computing scalar reductions (like MSE, distances, or loss functions), using these operations introduces severe memory overhead inside hot loops. Attempting to use `.min()` length truncation as a safeguard is an anti-pattern as it masks shape mismatch errors.
**Action:** For scalar reductions, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) to eliminate intermediate allocations and safely compute the result.
=======
## 2026-05-18 - Single-Pass Scalar Reductions in Linear Algebra
**Learning:** High-level tensor operations like `a.sub(b)` and `.mul()` generate costly intermediate `Tensor` heap allocations containing shape and strides metadata. When computing scalar reductions (like loss functions or distance metrics) over tensors, this overhead is unnecessary and significantly impacts performance in hot paths.
**Action:** Always avoid intermediate `Tensor` allocations for scalar reductions (e.g., `mse`, `mae`, `binary_cross_entropy`, `hinge_loss`, `euclidean_distance`, `manhattan_distance`, `chebyshev_distance`, `rbf_kernel`). Instead, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`).
>>>>>>> 5c3b036 (Perf: single pass scalar reductions)
