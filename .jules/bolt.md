## 2026-03-12 - Early Exit Squared Distance Comparison in Tight Loops
**Learning:** In performance-critical hot paths like `SparseAttentionGraph::add_point` which scan large portions of a dataset, computing the Euclidean distance with `libm::sqrt` repeatedly introduces a significant and avoidable bottleneck.
**Action:** Always prefer squared distance comparisons (`d^2 < r^2`) over full Euclidean distance in tight loops. Implement explicit rejection of invalid thresholds (e.g., negative/NaN `epsilon`) and add early exit conditions within the loop body to break out as soon as the accumulated squared distance exceeds the threshold.

## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.
