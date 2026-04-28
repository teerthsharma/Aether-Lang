## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-04-28 - Squared Distance Comparison in Spatial Bounds Checking
**Learning:** Checking whether points are within a certain spatial threshold (e.g., `epsilon` in sparse attention) often defaults to computing Euclidean distance via `sqrt`. In hot paths like `ManifoldPoint::is_neighbor`, `libm::sqrt` overhead becomes a bottleneck. Comparing squared distances is significantly faster and allows for an early loop exit. Using `!(sum < eps_sq)` safely handles NaN coordinates compared to standard logic.
**Action:** When validating spatial thresholds, square the threshold and compare it incrementally inside the loop to avoid `sqrt` and allow early exits, using `!(x < y)` for `NaN` safety.
