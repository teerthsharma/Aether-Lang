## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-04-05 - Avoid Intermediate Tensor Allocations in Scalar Reductions
**Learning:** Computing scalar metrics like MSE or Euclidean distance using high-level Tensor operations (e.g., `a.sub(b)`, `.mul()`) creates costly intermediate heap allocations for Tensor metadata and data buffers. Furthermore, using manual `.push()` loops for tensor initialization has high memory management overhead compared to iterator-based `.collect()`.
**Action:** When implementing scalar reductions across Tensors, assert shape equality and perform a single-pass iteration directly over the underlying borrowed data arrays. During tensor initialization, use `.collect()` and `Tensor::from_vec()` instead of manual `.push()` loops.
