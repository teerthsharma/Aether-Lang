## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-04-04 - Squared Distance Optimization for Spatial Queries
**Learning:** `ManifoldPoint::distance` relies on `libm::sqrt`, which is a costly operation. Using `is_neighbor` with `libm::sqrt` inside a hot `O(N)` loop (like `SparseAttentionGraph::add_point`) introduces significant performance overhead, as it computes the precise distance when only checking if a point is within a threshold.
**Action:** When filtering points based on proximity, avoid exact distance calculations by comparing the squared components against the squared threshold (`d^2 < r^2`). Include early exit conditions and gracefully reject invalid (negative) threshold checks before the loop.
