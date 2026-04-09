## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2025-04-06 - Avoid intermediate Tensor allocations in scalar reductions
**Learning:** In AETHER's `Tensor` implementation, high-level operations like `.sub()` and `.mul()` trigger costly intermediate heap allocations for shape and stride metadata, especially noticeable in hot-path scalar reductions (e.g., loss functions and distance metrics). Furthermore, using `.min()` for length truncation during iterations can mask critical shape mismatch errors.
**Action:** Assert shape equality explicitly (`assert_eq!(a.shape, b.shape)`) and iterate directly over the underlying borrowed data arrays (`a.data.borrow()`) in a single pass for all scalar reductions and gradient computations.
