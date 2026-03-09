## 2026-01-29 - Single Pass Variance Calculation in Manifold Heap
**Learning:** The `ChebyshevGuard::calculate` function in `ManifoldHeap` was performing two passes over the memory blocks to calculate mean and variance separately. This is a common pattern when following the mathematical definition directly. However, in a performance-critical "metabolism" loop (GC), this doubles the memory access overhead.
**Action:** Always check for opportunities to compute statistics (mean, variance) in a single pass using Welford's algorithm or accumulated sums, especially when iterating over large data structures.

## 2026-03-09 - Avoid Formatting Unrelated Files in PRs
**Learning:** `cargo fmt` blindly formats the whole workspace if run in the root, which can create a massive diff that pollutes the PR with formatting noise. Code reviews will deduct points for unrelated changes.
**Action:** When working on a specific optimization, either only format the specific file using `rustfmt <file>` or `cargo fmt --package <crate>`, or better yet, run `cargo fmt` but then isolate the logic changes in git so the formatting changes on unrelated files are not staged/committed.
