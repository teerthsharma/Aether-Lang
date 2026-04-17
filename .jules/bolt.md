## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-04-16 - Squared Distance Comparison in Spatial Scans
**Learning:** Hot-path spatial scans like `ManifoldPoint::is_neighbor` were calculating full Euclidean distances using `libm::sqrt`. In performance-critical tight loops (O(N) scans per point added), the floating-point square root operation adds significant overhead.
**Action:** When only checking if a distance is within a threshold, compare the accumulated squared distance (`d^2`) against the squared threshold (`r^2`). This avoids the `sqrt` entirely. Use `!(epsilon > 0.0)` for safe early exits and `!(sum < eps_sq)` within the loop to correctly handle potential `NaN` coordinate values without altering the logic.
