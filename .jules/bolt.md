## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-02-09 - Avoid unnecessary `sqrt` in geometric queries
**Learning:** `ManifoldPoint::is_neighbor` was calculating full Euclidean distance (`sqrt`) just to compare against a threshold. Since `sqrt` is monotonic for non-negative numbers, comparing squared distance against squared threshold is equivalent and avoids the expensive square root operation.
**Action:** When implementing distance-based queries (e.g., k-NN, range search), always provide a `distance_squared` method and use it for threshold comparisons. Be careful with negative thresholds, as squaring makes them positive.
