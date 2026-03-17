## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-02-09 - [Autograd Tensor Clone Optimization]
**Learning:** In hot paths (like autograd backward pass), `Option::take()` coupled with re-insertion can completely bypass borrow checker constraints over vectors, avoiding expensive heap allocations like `Option::clone()` which clones the inner tensor metadata.
**Action:** Use `.take()` and re-insert when temporarily needing ownership from an array or vector element in loops.
