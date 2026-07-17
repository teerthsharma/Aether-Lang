## 2026-07-06 - Tensor metadata cloning in autograd
**Learning:** In reverse-mode autograd passes, cloning `Option<Tensor>` or passing references triggers unnecessary heap allocations for tensor metadata (shape/strides), even though the underlying data is reference-counted.
**Action:** Use `Option::take()` to acquire ownership of gradients during the backward pass, and pass tensors by value to `accumulate_grad` to eliminate metadata clones.
## 2026-07-14 - Optimizing Tensor allocations in linear algebra
**Learning:** High-level tensor operations like `.sub()` and `.map()` during gradient calculations trigger costly intermediate heap allocations for both data and metadata.
**Action:** Use single-pass iterators (`.iter().zip().map().collect()`) directly over the borrowed data arrays and consume the resulting vector with `Tensor::from_vec()` to avoid redundant O(N) slice allocations.
## 2026-07-17 - [Eliminated Tensor Clones in Neural Network Forward/Backward Passes]
**Learning:** In Aether's Tensor engine, standard cloning of Tensors creates costly heap allocations for structural metadata (`shape`, `strides`) and increments Rc ref counts. This overhead compounds heavily during iterative deep learning passes, specifically within neural network layer logic where tensors are often short-lived or naturally consumed.
**Action:** Replace structural `.clone()` calls during backpropagation (`last_z.as_ref().unwrap().clone()`) with `.take()` to assume ownership of cached states. Furthermore, restructure iterator consumption for layer pipelines (e.g. `MLP::forward`) to pass initial references directly to the first layer rather than initializing the iterative state with a clone.
