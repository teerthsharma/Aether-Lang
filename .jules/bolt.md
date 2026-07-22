## 2026-07-06 - Tensor metadata cloning in autograd
**Learning:** In reverse-mode autograd passes, cloning `Option<Tensor>` or passing references triggers unnecessary heap allocations for tensor metadata (shape/strides), even though the underlying data is reference-counted.
**Action:** Use `Option::take()` to acquire ownership of gradients during the backward pass, and pass tensors by value to `accumulate_grad` to eliminate metadata clones.
## 2026-07-14 - Optimizing Tensor allocations in linear algebra
**Learning:** High-level tensor operations like `.sub()` and `.map()` during gradient calculations trigger costly intermediate heap allocations for both data and metadata.
**Action:** Use single-pass iterators (`.iter().zip().map().collect()`) directly over the borrowed data arrays and consume the resulting vector with `Tensor::from_vec()` to avoid redundant O(N) slice allocations.

## 2026-07-16 - Optimizing MLP Forward Pass
**Learning:** In a multi-layer perceptron, cloning the input tensor to initialize the forward loop creates unnecessary heap allocations and Rc increments. The first layer naturally consumes the input without mutating it.
**Action:** Use an iterator over the layers to pass the input reference directly to the first layer's forward pass, eliminating the initial clone.
