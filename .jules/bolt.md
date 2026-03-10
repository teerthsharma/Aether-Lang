## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-03-10 - Spatial Scan Optimization (Manifold Point Distance)
**Learning:** In hot spatial scans like `SparseAttentionGraph::add_point`, calculating distances using `libm::sqrt` causes significant overhead due to the large number of comparisons needed to establish ε-neighborhoods.
**Action:** When comparing distances against a threshold, prefer comparing squared distances Wrt squared thresholds (`sum < epsilon * epsilon`) with early exits inside the loop to avoid the `sqrt` function call completely.
