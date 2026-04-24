## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-02-05 - Direct Data Iteration for Scalar Tensor Reductions
**Learning:** High-level tensor operations like `.sub()` or `.mul()` on `Tensor` structures (backed by `Rc<RefCell<Vec<f64>>>` and heap-allocated shape/strides metadata) create expensive intermediate allocations when used for simple scalar reductions like distance calculations (`mse`, `euclidean_distance`, etc.).
**Action:** When calculating scalar results from multiple tensors (e.g., loss functions or distances), avoid `.sub()` and `.mul()`. Instead, assert shape equality and directly iterate over the borrowed underlying data vectors (`a.data.borrow()`) in a single pass to eliminate unnecessary intermediate heap allocations and garbage collection overhead.
