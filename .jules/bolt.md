## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-03-28 - Squared Distance Early Exits for Spatial Scans
**Learning:** In hot spatial scanning paths like `SparseAttentionGraph::add_point`, comparing raw distances using `libm::sqrt` incurs significant overhead. Using squared distances and checking against the squared threshold iteratively inside the coordinate loop enables safe early exits.
**Action:** Optimize geometric distance checks by squaring thresholds upfront (`r^2 > d^2`), exiting early when sums exceed `eps_sq`, and checking for negative/NaN thresholds explicitly (`!(epsilon > 0.0)`) prior to processing.
