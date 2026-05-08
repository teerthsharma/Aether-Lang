## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-05-01 - Avoid High-Level Tensor Ops in Scalar Reductions
**Learning:** High-level `Tensor` operations like `sub()` and `mul()` trigger intermediate heap allocations for shape and stride metadata. When computing scalar reductions (like MSE, distances, or loss functions), using these operations introduces severe memory overhead inside hot loops. Attempting to use `.min()` length truncation as a safeguard is an anti-pattern as it masks shape mismatch errors.
**Action:** For scalar reductions, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays (`a.data.borrow()`) to eliminate intermediate allocations and safely compute the result.

## 2026-05-09 - Avoid Tensor Clones in Autograd Backpropagation
**Learning:** During reverse-mode backpropagation in `aether-core::ml::autograd`, fetching the output gradient using `clone()` triggers a heap allocation for the `Tensor` shape and strides vectors, even though the data storage is shared. This causes massive memory overhead across all operations (`Add`, `Mul`, `MatMul`, `ReLU`).
**Action:** Use `Option::take()` to acquire ownership of the gradient from the `grads` vector, avoiding borrow checker issues. Pass the tensor by value to the `accumulate_grad` helper (moving newly allocated tensors directly into the vector without cloning), and then put the original output gradient back into the tape using `grads[out.index] = Some(grad)`.
