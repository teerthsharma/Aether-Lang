## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-04-02 - Squared Distance in Sparse Spatial Scans
**Learning:** `SparseAttentionGraph::add_point` performs hot O(N) spatial scans using `ManifoldPoint::is_neighbor`. Relying on `distance()` calculates a full `libm::sqrt()`, which creates significant overhead during point insertion.
**Action:** Prefer inline squared distance comparisons (`d^2 < r^2`) over `sqrt` in hot spatial loops. Explicitly reject negative or NaN thresholds before the loop (`!(epsilon > 0.0)`), and use early exits within the loop (`!(sum < eps_sq)`) to safely handle `NaN` coordinates while skipping unnecessary calculations.
