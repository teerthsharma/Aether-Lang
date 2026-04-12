## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2024-04-12 - Optimize scalar reductions in ml::linalg
**Learning:** High-level tensor operations like `a.sub(b)` and `.mul()` when computing scalar reductions (e.g., `mse`, `euclidean_distance`) trigger costly intermediate `Tensor` heap allocations. `.min()` length truncation masks shape mismatch errors.
**Action:** Assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`).
