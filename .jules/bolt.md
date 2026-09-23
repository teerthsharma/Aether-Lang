## 2026-07-06 - Tensor metadata cloning in autograd
**Learning:** In reverse-mode autograd passes, cloning `Option<Tensor>` or passing references triggers unnecessary heap allocations for tensor metadata (shape/strides), even though the underlying data is reference-counted.
**Action:** Use `Option::take()` to acquire ownership of gradients during the backward pass, and pass tensors by value to `accumulate_grad` to eliminate metadata clones.
## 2026-07-14 - Optimizing Tensor allocations in linear algebra
**Learning:** High-level tensor operations like `.sub()` and `.map()` during gradient calculations trigger costly intermediate heap allocations for both data and metadata.
**Action:** Use single-pass iterators (`.iter().zip().map().collect()`) directly over the borrowed data arrays and consume the resulting vector with `Tensor::from_vec()` to avoid redundant O(N) slice allocations.
## 2026-07-25 - Tensor metadata cloning in MLP forward passes
**Learning:** In `aether-core::ml::neural`, cloning the `input` tensor in `MLP::forward` before passing its reference to the first layer's `forward` method triggers an unnecessary heap allocation for tensor metadata and an `Rc` increment.
**Action:** Extract the first layer using `self.layers.iter_mut()` to pass the initial `input` as a `&Tensor` reference directly, as subsequent layers naturally consume the output of the previous layer.
## 2026-09-23 - Optimizing Agglomerative Clustering Distance Calculation
**Learning:** The previous implementation of agglomerative clustering calculated exact distances `sqrt(squared_distance)` during the linkage computation (e.g. `Linkage::Single`, `Linkage::Complete`). The square root operation is computationally expensive and unnecessary during the search for the minimum distance because distance is monotonically increasing with squared distance. However, some algorithms like `DBSCANResult` and `auto_k_selection` already used `squared_distance` internally.
**Action:** Replace `distance` with `squared_distance` when comparing point distances during clustering. Only compute `sqrt(min_dist_sq)` when storing the final `min_dist` in the merge history list, to preserve external API expectations.
