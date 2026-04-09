## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-03-31 - Fast Distance Computation with Early Exits
**Learning:** In hot loops checking spatial neighborhoods (e.g. `is_neighbor`), exact Euclidean distance computation (`libm::sqrt`) is a significant bottleneck. Furthermore, accumulating squared differences can be short-circuited if the running sum exceeds the squared threshold (`eps_sq`).
**Action:** When comparing distances against a threshold in high-dimensional space or hot O(N) scans, compare squared distances instead of exact ones (`d^2 < r^2`) to avoid `libm::sqrt`. Introduce an early exit condition inside the loop `!(sum < eps_sq)` to safely bypass unneeded computations while safely handling `NaN` values. Also, reject negative or `NaN` thresholds outright using `!(epsilon > 0.0)`.
