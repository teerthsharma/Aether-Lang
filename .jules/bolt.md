## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-01-29 - Avoid libm::sqrt in Spatial Attention Scans
**Learning:** `ManifoldPoint::is_neighbor` relies on `distance()` which calls `libm::sqrt`. In hot paths like `SparseAttentionGraph::add_point` performing O(N) spatial scans, the overhead of calculating the square root for every neighbor check is significant.
**Action:** Replace `sqrt(sum) < epsilon` with `sum < epsilon * epsilon` to avoid `libm::sqrt`. Reject `epsilon <= 0.0` or `NaN` upfront via `!(epsilon > 0.0)`, and use early exit `!(sum < eps_sq)` in the coordinate loop to handle `NaN` safely.
