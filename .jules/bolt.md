## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-04-08 - Avoid `libm::sqrt` in Hot Path Distance Calculations
**Learning:** In `SparseAttentionGraph::add_point`, the spatial scan loops through existing points sequentially. This creates a hot O(N) critical path where calculating distances using `libm::sqrt` (as done by `is_neighbor` calling `distance`) introduces significant overhead.
**Action:** When performing neighborhood checks, prefer squared distance comparisons (`d^2 < r^2`) over Euclidean distance to eliminate `libm::sqrt`. Implement this with inline loops to allow for early exits if the accumulated squared distance exceeds `epsilon^2`, which provides further performance gains, especially in higher dimensions or with sparse data.
