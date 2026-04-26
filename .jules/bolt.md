## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-04-26 - Single Pass Array Iteration for Tensor Reductions
**Learning:** In `aether-core::ml::linalg`, high-level tensor operations like `a.sub(b)` and `diff.mul(&diff)` used for scalar reductions (`mse`, `euclidean_distance`, etc.) trigger costly intermediate `Tensor` heap allocations (for the result's shape and strides metadata). Additionally, using `.min()` for length truncation silently masks shape mismatch errors.
**Action:** When computing scalar reductions between tensors, avoid intermediate tensor allocations by asserting shape equality (`assert_eq!(a.shape, b.shape)`) and performing a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`).
