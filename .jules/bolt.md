## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-04-22 - Avoid Tensor Heap Allocations in Scalar Reductions
**Learning:** In `aether-core::ml::linalg`, using high-level tensor operations like `a.sub(b)` and `.mul()` to compute scalar reductions (e.g., in `mse`, `euclidean_distance`) triggers costly intermediate `Tensor` heap allocations for metadata vectors (`shape` and `strides`). Length truncation with `.min()` also un-safely masks shape mismatch errors.
**Action:** Always assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) when computing scalar reductions, avoiding intermediate tensor creation entirely.
