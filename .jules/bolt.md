## 2026-07-06 - Tensor metadata cloning in autograd
**Learning:** In reverse-mode autograd passes, cloning `Option<Tensor>` or passing references triggers unnecessary heap allocations for tensor metadata (shape/strides), even though the underlying data is reference-counted.
**Action:** Use `Option::take()` to acquire ownership of gradients during the backward pass, and pass tensors by value to `accumulate_grad` to eliminate metadata clones.
## 2026-07-14 - Optimizing Tensor allocations in linear algebra
**Learning:** High-level tensor operations like `.sub()` and `.map()` during gradient calculations trigger costly intermediate heap allocations for both data and metadata.
**Action:** Use single-pass iterators (`.iter().zip().map().collect()`) directly over the borrowed data arrays and consume the resulting vector with `Tensor::from_vec()` to avoid redundant O(N) slice allocations.
## 2026-07-18 - Eliminating Tensor metadata cloning in neural network backpropagation
**Learning:** During the reverse-mode backward pass of the MLP, DenseLayer unnecessarily clones the cached last_z and last_input Tensors by reference. This triggers costly heap allocations for the shape and strides vectors inside the Tensor struct.
**Action:** Use Option::take() to acquire ownership of these cached tensors during the backward pass. This avoids metadata clones and borrow checker restrictions on subsequent operations.
