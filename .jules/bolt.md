## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-06-25 - Avoid Intermediate Tensor Allocations for Scalar Reductions
**Learning:** Using high-level tensor operations like `a.sub(b)` or `a.mul(b)` before calling a scalar reduction (like `.sum()`) triggers intermediate `Tensor` heap allocations for metadata (`shape` and `strides`). When computing distances or metrics (e.g., Euclidean distance, MSE), this overhead becomes significant, particularly in inner loops.
**Action:** When computing a single scalar reduction across tensors, ensure shape equality with `assert_eq!(a.shape, b.shape)`, then borrow the underlying data arrays (`a.data.borrow()`) and perform the calculation in a single manual pass to completely avoid intermediate allocations.
