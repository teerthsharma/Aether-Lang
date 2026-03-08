## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-03-08 - Fast Path Distance Checks
**Learning:** `ManifoldPoint::is_neighbor` was computing full Euclidean distance using `libm::sqrt` which is exceptionally slow in a hot path. Since it only needs to check if the distance is `< epsilon`, it can instead compare squared distances against `epsilon * epsilon`.
**Action:** When determining if two points are within a radius, avoid square root operations by comparing squared distances. Include an early exit (`sum_sq >= eps_sq`) to abort the calculation early.
