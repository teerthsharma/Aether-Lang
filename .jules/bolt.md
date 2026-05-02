## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-05-02 - Avoid High-Level Tensor Math in Metrics
**Learning:** In `aether-core::ml::linalg`, operations like `a.sub(b)` and `a.mul(b)` are convenient but create entirely new `Tensor` objects, which internally trigger heap allocations for metadata vectors (`shape`, `strides`). Calling these inside tight loops or metrics (like `mse`, `euclidean_distance`) creates severe memory bottlenecks due to allocation thrashing.
**Action:** When computing scalar reductions or metrics over tensors, always assert shape equality first (`assert_eq!(a.shape, b.shape)`), then perform a single-pass loop directly over the borrowed data (`a.data.borrow()`).
