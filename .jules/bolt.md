## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-04-07 - Avoid Tensor Allocations in Scalar Reductions
**Learning:** High-level Tensor operations like `a.sub(b)` and `.mul()` generate intermediate heap allocations for metadata (`shape`, `strides`) even when producing scalar values (e.g. distances or MSE).
**Action:** For scalar reductions, assert shape equality and directly iterate over the underlying data arrays (`a.data.borrow()`) to compute the value in a single pass without extra allocations.
