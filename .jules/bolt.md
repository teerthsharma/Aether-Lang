## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-04-19 - Sparse Attention Graph distance calculations
**Learning:** Checking neighbors in the `SparseAttentionGraph::add_point` hot loop uses `is_neighbor` extensively. Calling `self.distance(other) < epsilon` performs a full calculation including `libm::sqrt`. For simple threshold checking, this calculation introduces unnecessary processing overhead.
**Action:** When computing distances just to compare against a threshold `epsilon`, calculate the squared distance against `epsilon * epsilon` to skip `libm::sqrt` overhead, reject non-positive thresholds (`!(epsilon > 0.0)`), and apply early exit within the summing loop (`!(sum < eps_sq)`) for maximum efficiency.
