## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-04-14 - Autograd Backpropagation Performance
**Learning:** During the reverse-mode backward pass (`autograd.rs`), the `Tensor::clone()` method was being called on `grads[out.index]`. Because `Tensor` in `aether-core` stores its shape and strides as dynamically allocated `Vec`s on the heap (inside the struct, even if the data itself is `Rc<RefCell<Vec<f64>>>`), cloning the `Tensor` triggers costly heap allocations in the hot loop.
**Action:** By utilizing `Option::take()` on `grads[out.index]`, the code bypasses the borrow checker restrictions, allowing mutable accumulation to the gradient buffer without cloning the shape and stride metadata, and effectively eliminating O(N) array copies during the backward pass.
