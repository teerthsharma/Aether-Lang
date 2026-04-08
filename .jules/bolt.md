## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-03-30 - Avoid `libm::sqrt` overhead in `is_neighbor`
**Learning:** The `SparseAttentionGraph::add_point` performs O(N) spatial scans, where exact Euclidean distance `libm::sqrt` overhead is unnecessary for neighborhood checks.
**Action:** Use squared distance comparisons (`d^2 < r^2`) with an early exit condition inside loops to skip `libm::sqrt` calculations and improve performance significantly.
