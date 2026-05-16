## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-05-01 - Avoid High-Level Tensor Ops in Scalar Reductions
**Learning:** High-level `Tensor` operations like `sub()` and `mul()` trigger intermediate heap allocations for shape and stride metadata. When computing scalar reductions (like MSE, distances, or loss functions), using these operations introduces severe memory overhead inside hot loops. Attempting to use `.min()` length truncation as a safeguard is an anti-pattern as it masks shape mismatch errors.
**Action:** For scalar reductions, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) to eliminate intermediate allocations and safely compute the result.

## 2026-05-16 - O(N) k-nearest neighbors using select_nth_unstable_by
**Learning:** `KNNClassifier::predict` previously used an O(K * N) partial selection sort (bubble sort style) to find the k-nearest neighbors. For larger values of K and N, this is slow. The standard library provides `slice::select_nth_unstable_by` which finds the k-th smallest element in O(N) average time, effectively partitioning the slice so the smallest k elements are at the beginning.
**Action:** Always prefer `select_nth_unstable_by` over manual sorting when you only need to find the top K items without fully sorting the entire collection or even fully sorting the top K items.
