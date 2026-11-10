## 2026-07-06 - Tensor metadata cloning in autograd
**Learning:** In reverse-mode autograd passes, cloning `Option<Tensor>` or passing references triggers unnecessary heap allocations for tensor metadata (shape/strides), even though the underlying data is reference-counted.
**Action:** Use `Option::take()` to acquire ownership of gradients during the backward pass, and pass tensors by value to `accumulate_grad` to eliminate metadata clones.
## 2026-07-14 - Optimizing Tensor allocations in linear algebra
**Learning:** High-level tensor operations like `.sub()` and `.map()` during gradient calculations trigger costly intermediate heap allocations for both data and metadata.
**Action:** Use single-pass iterators (`.iter().zip().map().collect()`) directly over the borrowed data arrays and consume the resulting vector with `Tensor::from_vec()` to avoid redundant O(N) slice allocations.
## 2026-07-25 - Tensor metadata cloning in MLP forward passes
**Learning:** In `aether-core::ml::neural`, cloning the `input` tensor in `MLP::forward` before passing its reference to the first layer's `forward` method triggers an unnecessary heap allocation for tensor metadata and an `Rc` increment.
**Action:** Extract the first layer using `self.layers.iter_mut()` to pass the initial `input` as a `&Tensor` reference directly, as subsequent layers naturally consume the output of the previous layer.
## 2026-11-05 - Iterator Vectorization over Manual Loops in Tensor Math
**Learning:** In `aether-core::ml::linalg`, explicitly indexing into raw tensor data borrows inside a `for` loop (e.g., `for i in 0..n { diff = true_data[i] - pred_data[i]; sum += diff * diff; }`) disables loop unrolling and LLVM auto-vectorization due to bounds checking overhead at each index.
**Action:** Replace manual indexing with single-pass iterator chains (`.iter().zip().map().sum()`) directly on the underlying tensor data to elide bounds checks and dramatically speed up scalar reduction operations like MSE, MAE, distance functions, and kernels. When needing a max value (Chebyshev distance), use `.fold()` instead of `.max()` because `f64` does not implement `Ord`.
## 2026-11-10 - Iterator Vectorization over Manual Loops in State Module Math
**Learning:** In `aether-core::state` (`SystemState`), explicitly indexing into the state vector inside a `for` loop (`for i in 0..D { sum += abs(self.vector[i] - other.vector[i]); }`) prevents efficient auto-vectorization due to bounds checking overhead at each index.
**Action:** Replace manual indexing with single-pass iterator chains (e.g., `.iter().zip().map().sum()`) directly on the underlying tensor data to elide bounds checks and dramatically speed up scalar reduction operations like deviation, max_deviation, manhattan_deviation, and magnitude. When computing `max_deviation` with `f64`, use `.fold()` instead of `.max()` because `f64` does not implement `Ord`.
