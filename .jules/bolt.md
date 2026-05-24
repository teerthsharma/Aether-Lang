## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-05-01 - Avoid High-Level Tensor Ops in Scalar Reductions
**Learning:** High-level `Tensor` operations like `sub()` and `mul()` trigger intermediate heap allocations for shape and stride metadata. When computing scalar reductions (like MSE, distances, or loss functions), using these operations introduces severe memory overhead inside hot loops. Attempting to use `.min()` length truncation as a safeguard is an anti-pattern as it masks shape mismatch errors.
**Action:** For scalar reductions, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) to eliminate intermediate allocations and safely compute the result.

## 2026-10-25 - Avoid Tensor Cloning in Neural Network Forward/Backward Passes
**Learning:** In the `aether-core` library, the `Tensor` struct contains heap-allocated metadata (`shape` and `strides` vectors). Cloning a `Tensor` triggers new heap allocations for this metadata even though the actual data is reference-counted. In hot paths like neural network forward and backward passes, this leads to significant memory overhead.
**Action:** Always prefer borrowing `Tensor` instances directly as references (e.g., using `as_ref()` for cached `last_z` and `last_input` in backprop, or passing the initial `input` as a `&Tensor` reference directly to the first layer's forward method) to eliminate redundant metadata cloning and `Rc` increments.
