## 2026-07-06 - Tensor metadata cloning in autograd
**Learning:** In reverse-mode autograd passes, cloning `Option<Tensor>` or passing references triggers unnecessary heap allocations for tensor metadata (shape/strides), even though the underlying data is reference-counted.
**Action:** Use `Option::take()` to acquire ownership of gradients during the backward pass, and pass tensors by value to `accumulate_grad` to eliminate metadata clones.
## 2026-07-14 - Optimizing Tensor allocations in linear algebra
**Learning:** High-level tensor operations like `.sub()` and `.map()` during gradient calculations trigger costly intermediate heap allocations for both data and metadata.
**Action:** Use single-pass iterators (`.iter().zip().map().collect()`) directly over the borrowed data arrays and consume the resulting vector with `Tensor::from_vec()` to avoid redundant O(N) slice allocations.

## 2026-07-28 - Avoid tensor cloning in Neural Networks
**Learning:** In MLP forward passes and DenseLayer backward passes, unnecessary tensor cloning (metadata and Rc bumps) can be avoided by passing inputs by reference to the first layer and using Option::take() to acquire ownership of cached tensors.
**Action:** Use Option::take() on last_z and last_input in DenseLayer backward pass, and avoid cloning the input tensor before passing it to the first layer in MLP forward pass.
