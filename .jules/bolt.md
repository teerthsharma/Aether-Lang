## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-02-12 - Squared Distance Optimization in Manifold Neighborhood Checks
**Learning:** The `is_neighbor` function in `ManifoldPoint` was using `distance` which computes `sqrt`. This is O(N^2) in `SparseAttentionGraph` construction. The Rust compiler optimizes away constant loops extremely aggressively, making micro-benchmarks misleading without dynamic inputs.
**Action:** When optimizing distance checks (r < ε), always use squared distance (r^2 < ε^2) to avoid sqrt. Ensure benchmark loops mutate data or use black_box to prevent const folding.
