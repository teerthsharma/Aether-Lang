## 2026-07-05 - Optimize Autograd Backpropagation
**Learning:** In reverse-mode backpropagation, cloning `Tensor` metadata for intermediate gradients causes redundant heap allocations. Using `Option::take()` to acquire ownership of gradients during backprop and re-insertion avoids borrow checker issues and unnecessary cloning.
**Action:** Pass `Tensor` by value in `accumulate_grad` to utilize empty `None` slots efficiently.
