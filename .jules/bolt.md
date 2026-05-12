## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-05-01 - Avoid High-Level Tensor Ops in Scalar Reductions
**Learning:** High-level `Tensor` operations like `sub()` and `mul()` trigger intermediate heap allocations for shape and stride metadata. When computing scalar reductions (like MSE, distances, or loss functions), using these operations introduces severe memory overhead inside hot loops. Attempting to use `.min()` length truncation as a safeguard is an anti-pattern as it masks shape mismatch errors.
**Action:** For scalar reductions, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) to eliminate intermediate allocations and safely compute the result.

## 2026-05-12 - O(N) Sliding Window Optimization for Topological Metrics
**Learning:** The `verify_sliding_window` function in `aether-core::topology` was recomputing `betti_0` and `betti_1` for every window of size `W`, leading to an $O(N \cdot W)$ complexity where $N$ is the binary size. Given the metrics' reliance on local byte patterns (`is_gap` and `is_loop_pattern`), they can be updated incrementally at the edges of the sliding window.
**Action:** When evaluating topological signatures (like Betti numbers) over sliding windows, implement an incremental update step tracking the leaving and entering edges to achieve $O(N)$ performance, drastically accelerating continuous stream verification (like ELF loader checks). Add a fallback for extremely small windows (e.g. `< 4`) to prevent edge case out-of-bounds array access.
