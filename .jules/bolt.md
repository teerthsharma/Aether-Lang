## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-05-01 - Avoid High-Level Tensor Ops in Scalar Reductions
**Learning:** High-level `Tensor` operations like `sub()` and `mul()` trigger intermediate heap allocations for shape and stride metadata. When computing scalar reductions (like MSE, distances, or loss functions), using these operations introduces severe memory overhead inside hot loops. Attempting to use `.min()` length truncation as a safeguard is an anti-pattern as it masks shape mismatch errors.
**Action:** For scalar reductions, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) to eliminate intermediate allocations and safely compute the result.

## 2026-05-28 - O(N) Nearest Neighbor Selection
**Learning:** `KNNClassifier` uses a manual partial selection sort with $O(K \cdot N)$ complexity to find the $K$ nearest neighbors. This is highly inefficient in hot loops, especially when finding neighbors is a frequent operation. Finding $K$ nearest neighbors does not strictly require full sorting of the elements, only partitioning.
**Action:** Replace manual partial sorts with `select_nth_unstable_by` which provides $O(N)$ average time complexity. Use `partial_cmp().unwrap_or(core::cmp::Ordering::Equal)` to safely compare `f64` distances in the presence of `NaN` values without breaking the total ordering contract.
