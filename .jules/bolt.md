## 2026-07-06 - Tensor metadata cloning in autograd
**Learning:** In reverse-mode autograd passes, cloning `Option<Tensor>` or passing references triggers unnecessary heap allocations for tensor metadata (shape/strides), even though the underlying data is reference-counted.
**Action:** Use `Option::take()` to acquire ownership of gradients during the backward pass, and pass tensors by value to `accumulate_grad` to eliminate metadata clones.
## 2026-07-13 - Intermediate Tensor allocations in derivative computations
**Learning:** High-level `Tensor` operations like `sub()`, `scale()`, and `map()` in derivative computations (e.g., `LossConfig::derivative`) trigger costly intermediate heap allocations for the data vector and shape/stride metadata.
**Action:** Use iterator chains (`.iter().zip().map().collect()`) directly on the underlying borrowed data arrays, and consume the resulting vector using `Tensor::from_vec()` to avoid redundant intermediate tensor creations.
