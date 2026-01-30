## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-01-30 - Vectorization of Sparse Data Structures
**Learning:** Iterating over sparse data using bitmasks prevents SIMD vectorization and introduces branch mispredictions. By ensuring "empty" slots contain a neutral identity value (0.0), we can replace branchy loops with unconditional, vectorized loops.
**Action:** Replace masked loops with invariant-based unconditional loops where possible.
