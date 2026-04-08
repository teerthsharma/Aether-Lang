## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-04-02 - Squared Distance in Sparse Attention Graph
**Learning:** The `is_neighbor` function in `ManifoldPoint` was computing `sqrt` to determine if two points are within an epsilon neighborhood, which was called inside the hot loop `add_point` of `SparseAttentionGraph`.
**Action:** Always prefer computing squared distance instead of `sqrt` for threshold comparisons (`d^2 < r^2`). Added a check to reject negative or NaN thresholds before the loop (using `!(epsilon > 0.0)`) and an early return within the loop (`!(sum < eps_sq)`) to handle `NaN` safely and boost performance.
