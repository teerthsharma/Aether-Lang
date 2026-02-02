## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-05-21 - Squared Distance Optimization in Manifold Geometry
**Learning:** `ManifoldPoint::is_neighbor` used `sqrt` for distance checks, causing significant overhead (40%) in `SparseAttentionGraph` construction.
**Action:** Always prefer squared distance comparisons (`dist_sq < epsilon^2`) for geometric predicates to avoid expensive `sqrt` calls.
