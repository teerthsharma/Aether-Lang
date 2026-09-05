## 2026-07-06 - Tensor metadata cloning in autograd
**Learning:** In reverse-mode autograd passes, cloning `Option<Tensor>` or passing references triggers unnecessary heap allocations for tensor metadata (shape/strides), even though the underlying data is reference-counted.
**Action:** Use `Option::take()` to acquire ownership of gradients during the backward pass, and pass tensors by value to `accumulate_grad` to eliminate metadata clones.
## 2026-07-14 - Optimizing Tensor allocations in linear algebra
**Learning:** High-level tensor operations like `.sub()` and `.map()` during gradient calculations trigger costly intermediate heap allocations for both data and metadata.
**Action:** Use single-pass iterators (`.iter().zip().map().collect()`) directly over the borrowed data arrays and consume the resulting vector with `Tensor::from_vec()` to avoid redundant O(N) slice allocations.
## 2026-07-25 - Tensor metadata cloning in MLP forward passes
**Learning:** In `aether-core::ml::neural`, cloning the `input` tensor in `MLP::forward` before passing its reference to the first layer's `forward` method triggers an unnecessary heap allocation for tensor metadata and an `Rc` increment.
**Action:** Extract the first layer using `self.layers.iter_mut()` to pass the initial `input` as a `&Tensor` reference directly, as subsequent layers naturally consume the output of the previous layer.
## 2026-09-05 - Optimize ml/linalg.rs scalar reductions
**Learning:** In `aether-core::ml::linalg`, avoid high-level tensor operations like `a.sub(b)` and `.mul()` when computing scalar reductions (e.g., `mse`, `mae`, etc.) as they trigger costly intermediate heap allocations. Instead, assert shape equality (`assert_eq!(a.shape, b.shape)`) and perform a single-pass iteration directly over the underlying borrowed data arrays using functional iterator chains (e.g., `.iter().zip().map().sum()`). This elides bounds checks and allows LLVM to auto-vectorize more effectively compared to manual `for i in 0..n` index-based loops.
**Action:** Use `.iter().zip().map().sum()` for scalar reductions to avoid intermediate heap allocations and bounds checks. Ensure explicit `return` statements within these blocks are converted to implicit tail expressions to avoid `needless_return` clippy warnings.
