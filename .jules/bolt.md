## 2026-07-06 - Tensor metadata cloning in autograd
**Learning:** In reverse-mode autograd passes, cloning `Option<Tensor>` or passing references triggers unnecessary heap allocations for tensor metadata (shape/strides), even though the underlying data is reference-counted.
**Action:** Use `Option::take()` to acquire ownership of gradients during the backward pass, and pass tensors by value to `accumulate_grad` to eliminate metadata clones.
## 2026-07-14 - Optimizing Tensor allocations in linear algebra
**Learning:** High-level tensor operations like `.sub()` and `.map()` during gradient calculations trigger costly intermediate heap allocations for both data and metadata.
**Action:** Use single-pass iterators (`.iter().zip().map().collect()`) directly over the borrowed data arrays and consume the resulting vector with `Tensor::from_vec()` to avoid redundant O(N) slice allocations.
## 2026-07-28 - Neural Network Tensor Metadata Cloning
**Learning:** In neural network forward and backward passes (e.g., `MLP::forward`, `DenseLayer::backward`), using `.clone()` on `Option<Tensor>` caches or `Tensor` references causes costly intermediate heap allocations for shape and stride metadata even when data buffers are shared via `Rc`.
**Action:** Use `Option::take()` to acquire ownership of cached tensors in backprop, and extract the first layer in the forward pass to consume the input reference directly without cloning.
