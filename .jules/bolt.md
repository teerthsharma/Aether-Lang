## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-05-01 - Avoid High-Level Tensor Ops in Scalar Reductions
**Learning:** High-level `Tensor` operations like `sub()` and `mul()` trigger intermediate heap allocations for shape and stride metadata. When computing scalar reductions (like MSE, distances, or loss functions), using these operations introduces severe memory overhead inside hot loops. Attempting to use `.min()` length truncation as a safeguard is an anti-pattern as it masks shape mismatch errors.
**Action:** For scalar reductions, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) to eliminate intermediate allocations and safely compute the result.

## 2026-05-02 - Avoid Redundant Allocations in Tensor Initialization
**Learning:** `Tensor::new()` takes a slice `&[f64]` and calls `.to_vec()`, causing a redundant heap allocation. Furthermore, using manual `for` loops to `push` into a `Vec<f64>` introduces bounds checking and loop overhead.
**Action:** When creating new Tensors from raw data loops, always use iterator chains (e.g., `data_a.iter().zip(data_b.iter()).map(...).collect()`) and pass the resulting vector directly to `Tensor::from_vec()`. This elides bounds checks, optimizes vector building, and completely eliminates the double allocation.
