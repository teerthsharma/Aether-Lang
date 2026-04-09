## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-04-09 - Spatial Neighborhood Checks (Avoiding `sqrt`)
**Learning:** In hot path distance calculations like `SparseAttentionGraph::add_point`, calling `libm::sqrt` for every distance comparison is a significant performance bottleneck.
**Action:** When comparing distances against a threshold `epsilon`, compute the squared distance and compare it against `epsilon * epsilon`. Include an early exit loop (`!(sum < eps_sq)`) to avoid summing all dimensions if the threshold is already exceeded. Reject invalid thresholds upfront (`!(epsilon > 0.0)`).
