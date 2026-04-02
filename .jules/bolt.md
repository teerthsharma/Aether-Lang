## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-04-10 - Avoid Tensor Allocation in Scalar Reductions
**Learning:** In `aether-core::ml::linalg`, operations like `a.sub(b)` allocate entirely new dynamic Tensors on the heap, and scalar reductions like `mse`, `euclidean_distance`, `manhattan_distance`, and `chebyshev_distance` were discarding these intermediates immediately.
**Action:** Replace intermediate Tensors (`sub`, `mul`) with direct single-pass iterations over `data.borrow()`, with `assert_eq!(a.shape, b.shape)` to guarantee shapes match without high-level validation overhead.
