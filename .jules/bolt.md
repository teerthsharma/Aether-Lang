## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-04-21 - Avoid Tensor Heap Allocations in Scalar Reductions
**Learning:** High-level tensor operations like `a.sub(b)` and `.mul()` in scalar reductions (e.g., `mse`, `euclidean_distance`) trigger costly intermediate `Tensor` heap allocations for shape and strides vectors.
**Action:** Use `assert_eq!(a.shape, b.shape)` and perform single-pass iterations directly over the underlying borrowed data arrays (`a.data.borrow()`) to compute scalar reductions without allocations. Do not use `.min()` length truncation as it masks shape mismatch errors.
