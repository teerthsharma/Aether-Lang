## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-03-19 - Optimize `ManifoldPoint::is_neighbor` with Squared Distance and Early Exit
**Learning:** `ManifoldPoint::is_neighbor` in `aether-core` previously called `distance()`, which uses `libm::sqrt`. For high-frequency spatial scans like in `SparseAttentionGraph::add_point`, calculating the square root is an unnecessary overhead when checking if a distance is within a threshold `epsilon`.
**Action:** Replace `distance() < epsilon` with an inline squared distance loop. Safely reject negative/NaN epsilons upfront (`!(epsilon > 0.0)`), and use an early exit condition `!(sum < eps_sq)` inside the loop to securely handle NaNs while avoiding `sqrt`.
