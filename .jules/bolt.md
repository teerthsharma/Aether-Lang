## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-03-18 - Optimize Distance Calculation in Spatial Scans
**Learning:** In operations like SparseAttentionGraph's point addition where spatial scans are hot O(N) paths, calculating the exact distance using `libm::sqrt` is a significant bottleneck. Comparing squared distance is mathematically equivalent and avoids the heavy function call. Additionally, an early exit within the dimension loop provides fast rejection.
**Action:** When evaluating geometric conditions like `distance < threshold`, prefer `squared_distance < threshold_sq`. Always validate the threshold beforehand (e.g. `!(threshold > 0.0)`) and use early-exit constructs that safely handle `NaN` like `!(sum < threshold_sq)` for maximum efficiency and correctness.
