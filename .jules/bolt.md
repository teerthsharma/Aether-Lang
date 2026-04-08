## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2023-10-24 - Avoiding `libm::sqrt` in Hot Spatial Scans
**Learning:** In highly iterated spatial loops like `SparseAttentionGraph::add_point`, checking geometric distance using `distance(other) < epsilon` introduces a major bottleneck due to the underlying `libm::sqrt` call. Using squared distance calculations allows bypassing this entirely.
**Action:** When performing neighborhood or radius checks (`d < r`) in hot paths, compare squared distances (`d^2 < r^2`) and introduce an early exit if the accumulated squared distance becomes `>=` squared threshold. Furthermore, always explicitly handle `NaN` inputs using inverse comparison `!(sum < eps_sq)` instead of `sum >= eps_sq` when defining bounds logic to be `NaN` resilient without breaking early exits.
