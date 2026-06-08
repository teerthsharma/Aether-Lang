## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-05-01 - Avoid High-Level Tensor Ops in Scalar Reductions
**Learning:** High-level `Tensor` operations like `sub()` and `mul()` trigger intermediate heap allocations for shape and stride metadata. When computing scalar reductions (like MSE, distances, or loss functions), using these operations introduces severe memory overhead inside hot loops. Attempting to use `.min()` length truncation as a safeguard is an anti-pattern as it masks shape mismatch errors.
**Action:** For scalar reductions, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) to eliminate intermediate allocations and safely compute the result.

## 2026-06-08 - Eliminate Redundant Clones in Autograd Backward Pass
**Learning:** In the reverse-mode `backward` pass of `aether-core::ml::autograd`, fetching the target gradient via `.clone()` and then passing references to `accumulate_grad` forces unnecessary allocations of `Tensor` metadata (shape/strides arrays). Even though the underlying data uses `Rc<RefCell>`, cloning the `Tensor` struct itself triggers heap allocations.
**Action:** Use `Option::take()` to acquire ownership of the gradient during backpropagation, and change `accumulate_grad` to take the gradient `Tensor` by value. Re-inserting the gradient back into the `grads` vector preserves state while bypassing the borrow checker cleanly.
