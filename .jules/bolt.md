## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-04-02 - Single Pass Scalar Reductions for Tensors
**Learning:** High-level tensor operations like `.sub()` and `.mul()` internally allocate new `Tensor` instances, including allocating `Vec` metadata for shapes and strides on the heap. When computing simple scalar reductions (like MSE, Euclidean distance, or MAE), these intermediate allocations cause significant performance bottlenecks and memory overhead in hot paths. Additionally, truncating iterations using `.min()` can mask shape mismatch errors.
**Action:** When computing scalar reductions between two tensors, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) instead of using intermediate tensor operations.
