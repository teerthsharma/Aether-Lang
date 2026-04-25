
## 2026-01-29 - Avoiding `sqrt` in Spatial Scans
**Learning:** In `SparseAttentionGraph::add_point`, the `is_neighbor` function is called heavily during point addition, turning into a hot O(N) spatial scan path. Using `libm::sqrt` for distance calculation incurs significant overhead.
**Action:** Replace direct distance comparisons involving `sqrt` with squared distance calculations (`d^2 < r^2`) combined with early exit loops (`!(sum < eps_sq)`). Always carefully handle edge cases like non-positive epsilon and NaN safely using `!` operators.
