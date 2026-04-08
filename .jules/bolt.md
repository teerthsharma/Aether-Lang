## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2025-03-15 - [Avoid `sqrt` in Distance Calculations for Spatial Scans]
**Learning:** In hot O(N) spatial scans (like `SparseAttentionGraph::add_point`), computing exact distances using `libm::sqrt` creates significant performance overhead, especially in a `no_std` context. We can achieve the same topological filtering logic (`distance < epsilon`) by substituting it with squared distance comparisons (`squared_distance < epsilon^2`), which eliminates the `sqrt` calls entirely.
**Action:** Always prefer squared distance comparisons coupled with early exits inside accumulation loops when filtering points by distance, rather than computing the exact Euclidean distance.
