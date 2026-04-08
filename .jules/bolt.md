## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-01-30 - O(N) Sliding Window Validation
**Learning:** Checking heuristics against a sliding window in `verify_sliding_window` natively leads to an O(N*W) algorithm, doing duplicate verification work as the window translates. However, topological primitives (clusters/gaps and loops) are easily tracked incrementally at the boundaries.
**Action:** Replace `is_shape_valid(window)` loops that re-scan `W` bytes with single-pass logic that maintains state and updates based only on entering/leaving edges, reducing theoretical complexity to O(N).
