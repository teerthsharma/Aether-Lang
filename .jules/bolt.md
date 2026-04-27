## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.
## 2026-04-27 - Avoid libm::sqrt in hot spatial scans
**Learning:** The `ManifoldPoint::is_neighbor` method was computing the full Euclidean distance (with `libm::sqrt`) to check for epsilon neighborhoods, which is slow in tight loops like `SparseAttentionGraph::add_point`.
**Action:** Optimize distance threshold checks by comparing squared distances and adding early exit conditions.
