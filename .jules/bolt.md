## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-03-22 - Squared Distance Loop Early Exit
**Learning:** In hot loops such as `SparseAttentionGraph::add_point`, replacing Euclidean distance (`libm::sqrt`) with a manual loop calculating squared distance enables a significant performance optimization: early exit when the accumulated squared distance exceeds the threshold squared. However, explicit checks for negative or NaN thresholds (`!(epsilon > 0.0)`) and safely structured loop conditions (`!(sum < eps_sq)`) are required to prevent regressions when processing edge case data (like NaN coordinates).
**Action:** Always prefer squared distance comparisons with early exit loops over Euclidean distance for spatial scans. Ensure NaN values are handled correctly by using the negations of `<` or `>` operators.
