## 2026-07-06 - Tensor metadata cloning in autograd
**Learning:** In reverse-mode autograd passes, cloning `Option<Tensor>` or passing references triggers unnecessary heap allocations for tensor metadata (shape/strides), even though the underlying data is reference-counted.
**Action:** Use `Option::take()` to acquire ownership of gradients during the backward pass, and pass tensors by value to `accumulate_grad` to eliminate metadata clones.
## 2026-07-14 - Optimizing Tensor allocations in linear algebra
**Learning:** High-level tensor operations like `.sub()` and `.map()` during gradient calculations trigger costly intermediate heap allocations for both data and metadata.
**Action:** Use single-pass iterators (`.iter().zip().map().collect()`) directly over the borrowed data arrays and consume the resulting vector with `Tensor::from_vec()` to avoid redundant O(N) slice allocations.
## 2026-07-25 - Tensor metadata cloning in MLP forward passes
**Learning:** In `aether-core::ml::neural`, cloning the `input` tensor in `MLP::forward` before passing its reference to the first layer's `forward` method triggers an unnecessary heap allocation for tensor metadata and an `Rc` increment.
**Action:** Extract the first layer using `self.layers.iter_mut()` to pass the initial `input` as a `&Tensor` reference directly, as subsequent layers naturally consume the output of the previous layer.
## 2026-07-28 - Scalar Reductions Bounds Checks and Auto-Vectorization
**Learning:** Manual `for i in 0..n` index-based loops in `linalg.rs` fail to elide bounds checks for tensor data slices, hindering LLVM's auto-vectorization during scalar reductions like `mse` and distance functions.
**Action:** Perform a single-pass iteration directly over the underlying borrowed data arrays using functional iterator chains (e.g., `.iter().zip().map().sum()`). This elides bounds checks and allows auto-vectorization. For `max` operations, use `.fold()` instead of `.max()` to handle `f64`.
