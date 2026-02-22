## 2024-05-22 - Topological Optimization Mismatch
**Learning:** `compute_betti_0` in `aether-core` implements a "gap clustering" algorithm (counting contiguous sequences of large gaps) rather than standard Betti-0 (connected components). This distinction is crucial for optimization; standard Betti-0 update logic would be incorrect here.
**Action:** Always verify the specific algorithm implementation rather than relying on mathematical names, especially when "approximation" or "heuristic" is mentioned.
