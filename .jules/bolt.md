## 2026-07-06 - Tensor metadata cloning in autograd
**Learning:** In reverse-mode autograd passes, cloning `Option<Tensor>` or passing references triggers unnecessary heap allocations for tensor metadata (shape/strides), even though the underlying data is reference-counted.
**Action:** Use `Option::take()` to acquire ownership of gradients during the backward pass, and pass tensors by value to `accumulate_grad` to eliminate metadata clones.
## 2026-07-14 - Optimizing Tensor allocations in linear algebra
**Learning:** High-level tensor operations like `.sub()` and `.map()` during gradient calculations trigger costly intermediate heap allocations for both data and metadata.
**Action:** Use single-pass iterators (`.iter().zip().map().collect()`) directly over the borrowed data arrays and consume the resulting vector with `Tensor::from_vec()` to avoid redundant O(N) slice allocations.
## 2026-07-25 - Tensor metadata cloning in MLP forward passes
**Learning:** In `aether-core::ml::neural`, cloning the `input` tensor in `MLP::forward` before passing its reference to the first layer's `forward` method triggers an unnecessary heap allocation for tensor metadata and an `Rc` increment.
**Action:** Extract the first layer using `self.layers.iter_mut()` to pass the initial `input` as a `&Tensor` reference directly, as subsequent layers naturally consume the output of the previous layer.
## 2026-10-09 - Iterators over manual loops in Linear Algebra Reductions
**Learning:** High-level scalar reductions in `aether-core::ml::linalg` (like `mse`, `mae`, `binary_cross_entropy`, and distance functions) can trigger costly manual bounds checks and inhibit auto-vectorization when written as manual index-based `for i in 0..n` loops.
**Action:** Replace manual loops with single-pass iterator chains (`.iter().zip().map().sum()` or `.fold()`) directly over the borrowed data arrays. This elides bounds checks and allows LLVM to auto-vectorize more effectively, dramatically speeding up scalar reductions during training loops. Do not use `.min()` for length, just `.zip()`.
