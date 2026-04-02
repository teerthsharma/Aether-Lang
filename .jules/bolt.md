## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-04-02 - Optimize Scalar Reductions to Prevent Intermediate Allocations
**Learning:** In `aether-core::ml::linalg`, high-level tensor operations like `a.sub(b)` and `.mul()` trigger costly intermediate `Tensor` heap allocations when computing scalar reductions (e.g., `mse`, `euclidean_distance`). Furthermore, using `.min()` length truncation can mask critical shape mismatch errors.
**Action:** Avoid high-level tensor operations for scalar reductions. Instead, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`).
