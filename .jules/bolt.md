## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-03-21 - Avoiding libm::sqrt in Spatial Scans
**Learning:** In highly recursive or looping spatial operations like `SparseAttentionGraph::add_point`, the overhead of `libm::sqrt` can quickly accumulate. By manually unrolling the distance loop with an early exit condition `!(sum < eps_sq)` and handling NaN/negative epsilons up front, you avoid floating-point exceptions and reduce time complexity.
**Action:** Use inline squared distance loops combined with NaN-safe early exits instead of generic `distance` functions inside hot spatial search loops.
