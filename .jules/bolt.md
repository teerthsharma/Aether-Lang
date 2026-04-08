## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-03-27 - Optimize hot spatial scan by avoiding libm::sqrt
**Learning:** `libm::sqrt` overhead in a no_std environment is a common performance bottleneck in spatial scans. Replacing it with squared distance comparison significantly improves the hot path performance. Furthermore, adding an early exit condition with `!(sum < eps_sq)` is critical for ensuring safe `NaN` handling without changing existing behavior or losing efficiency.
**Action:** Always optimize loops doing spatial scanning to use squared distances with safe `NaN` early exits.
