## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.
## 2026-02-04 - Squared Distance Optimization in Manifold Point
**Learning:** Replacing `sqrt` with squared distance comparison in `is_neighbor` checks yielded a ~5x speedup (28ms -> 5.7ms).
**Action:** When performing distance checks against a threshold, always compare `distance_squared < threshold_squared` to avoid expensive square root operations, but ensure to check `threshold > 0` to handle negative values correctly.
