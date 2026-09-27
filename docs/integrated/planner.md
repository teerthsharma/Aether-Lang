# Runtime Planner

Module: `aether_core::planner`, in `crates/aether-core/src/planner.rs`. It contains three algorithms, each ported from a merged upstream change. Each decides something about a computation before it runs:

- where its tensors live in one arena;
- which ordering edges its schedule needs;
- which of its constraint groups can be solved independently.

Each algorithm is exact for what it states. The offset planner is greedy, and says so, but it never returns an unsound plan. Evidence: 15 tests in `tests/planner.rs`.

## Offset planning: google/XNNPACK#10801

**Problem.** Tensors \(i = 1, \ldots, n\) have byte sizes \(w_i\) and inclusive lifetimes \([f_i, \ell_i]\) over node indices. Assign each tensor an offset \(o_i\) in one arena such that

\[
[f_i, \ell_i] \cap [f_j, \ell_j] \ne \varnothing \;\Longrightarrow\; [o_i,\ o_i + w_i) \cap [o_j,\ o_j + w_j) = \varnothing,
\]

and keep the arena \(\max_i (o_i + w_i)\) small. Minimising the arena exactly is dynamic storage allocation (Garey and Johnson, problem SR2), which is NP-complete, so the planner is greedy.

**Algorithm.** Tensors are placed in decreasing size. Each one collects the byte ranges of already-placed tensors whose lifetimes intersect its own, then sorts and coalesces them into disjoint blocks. It takes the smallest free interval that fits (best fit). The leading interval \([0, \text{first block})\) is a candidate and wins ties. If nothing fits, the tensor is appended after the last block.

**Invariant and lower bound.** After every placement, no two placed tensors with intersecting lifetimes share a byte. Let \(\operatorname{peak} = \max_t \sum_{i : t \in [f_i, \ell_i]} w_i\) be the largest total size of tensors live at one node. That is the maximum-weight clique of the lifetime interval graph. The tensors live at the peak node pairwise intersect in lifetime, so they are byte-disjoint inside the arena:

\[
\text{arena} \;\ge\; \operatorname{peak}.
\]

**Complexity.** \(O(n^2 \log n)\) time in the worst case, and \(O(n)\) memory. `peak_live_bytes` is \(O(n \log n)\) by a sweep over lifetime endpoints.

**Upstream fix.** The gap search considered the intervals between live blocks and the space after the last block, but never \([0, \text{first.start})\). A single live block also took a fast path that always appended. A tensor could therefore grow the arena while a leading interval large enough for it was free for its whole lifetime. The change made the leading interval a best-fit candidate. The pull request reports FP32 MobileNet V1 peak allocation falling from 23.862980 MiB to 22.331730 MiB (6.42%) on an Intel Core i7-14700HX.

## Transitive reduction: tensorflow/tensorflow#124410

**Problem.** Given a DAG, find the fewest edges with the same reachability. For a finite DAG this transitive reduction is unique. It is exactly the set of edges \(u \to v\) for which no other path leads from \(u\) to \(v\):

\[
(u \to v) \in \operatorname{TR}(G) \iff v \notin \bigcup_{w \in N^+(u),\ w \ne v} \operatorname{reach}(w),
\]

where \(N^+(u)\) is the set of successors of \(u\) and \(\operatorname{reach}(w)\) is the set of nodes reachable from \(w\), including \(w\) itself.

**Algorithm.** Reachability is closed as a dense bit matrix in one pass over the nodes in reverse topological order, so a successor's row is final before it is folded into its predecessor's. Nodes must be numbered topologically, with \(u < v\) for every edge, so descending index is a reverse topological order.

**Invariant.** Every edge is decided independently against the complete closure. The result is therefore a function of the edge set alone, not of the order in which edges arrive.

**Complexity.** \(O(E \log E + (n + E)\lceil n/64\rceil)\) time for \(n\) nodes and \(E\) edges. Memory is \(n^2/8\) bytes for the matrix plus \(O(n + E)\) for adjacency.

**Upstream fix.** TensorFlow's collective ordering copied a destination's reachable set when an edge was created and never propagated later additions back. A redundant edge whose alternate path had length three or more therefore survived the prune. The prune's output also depended on the iteration order of a pointer-keyed hash set. The change closed reachability exactly and decided each edge independently. Its scratch comparison, computed from container layout rather than measured, is 5,387,410 bytes against 131,072 bytes at 1,024 collectives.

## Island discovery: google-deepmind/mujoco#3396 and mujoco_warp#1541

**Problem.** Given nodes and incidences between them, partition the nodes that appear in at least one incidence into connected components, or "islands". A constraint solver can then treat the islands independently. A node in no incidence belongs to no island.

**Algorithm.** A disjoint-set forest in a single parent array. Union links the larger root under the smaller, so every root is the minimum member of its set. Find compresses the path it walks. One ascending pass then numbers the roots in order and gives every other node its root's number.

**Invariant.** \(\text{parent}[x] \le x\) for every active node, and each root is its set's minimum. Labels are therefore canonical. Island \(k\) is the one whose smallest member is the \(k\)-th smallest among islands, whatever the order and orientation of the incidences.

**Complexity.** One machine word of scratch per node, independent of the number of incidences. Linking is by minimum index, not by rank or size, so the inverse-Ackermann bound does not apply. Path compression alone gives \(O(\log n)\) amortised per operation (Tarjan and van Leeuwen, 1984).

**Upstream fix.** MuJoCo built an \(\text{ntree} \times \text{ntree}\) byte adjacency matrix and an \(\text{ntree} \times \text{ntree}\) integer column array before flood fill. The change replaced both with this forest. For the generated singleton family of mujoco#3388, peak `mj_island` stack use went from \(5\,\text{ntree}^2 + 36\,\text{ntree} + 32\) bytes to \(16\,\text{ntree} + 32\) bytes: 84,033,568 to 65,568 bytes at 4,096 trees. mujoco_warp#1541 carried the same forest and canonical labelling to the GPU, hooking roots with atomic compare-and-swap. This module is the sequential form.

`aether_core::persistence` computes \(H_0\) by \(\mathbb{Z}_2\) column reduction and has no union-find to share. `attention` and `scheduled` each keep a private path-compressing `find`, equivalent to the one in `islands`.

## What is exact and what is not

| Algorithm | Exact | Not claimed |
| --- | --- | --- |
| `plan_offsets` | Soundness: tensors whose lifetimes intersect never share a byte | Optimality. The problem is NP-complete and the planner is greedy |
| `transitive_reduction` | The unique minimal edge set with the same reachability, independent of edge order | Input outside the topological numbering is refused by panic, not reduced |
| `islands` | Connected components with canonical labels | The inverse-Ackermann bound; \(O(\log n)\) amortised applies instead |

On the 500 seeded instances of the soundness test, the port measured arena over peak live bytes at a mean of 1.0283 and a worst case of 1.2826 (port report). No test asserts these two ratios.

## Refusals

The module panics rather than returning an error, and each panic marks a precondition whose violation would make the result silently wrong:

- `plan_offsets`: a tensor with `first_use > last_use`.
- `transitive_reduction`: an edge with \(u \ge v\), or \(v \ge\) `node_count`. A back edge would make the single closure pass wrong.
- `islands`: an incidence naming a node at or beyond `node_count`.

A zero-size tensor is not planned and gets offset 0. An incidence \((t, t)\) activates \(t\) on its own. MuJoCo passes −1 for a static endpoint and substitutes the other endpoint, which amounts to the same thing.

## Rust API

```rust
pub struct TensorLifetime { pub size: usize, pub first_use: usize, pub last_use: usize }
pub struct MemoryPlan { pub offsets: Vec<usize>, pub arena_size: usize }

pub fn plan_offsets(tensors: &[TensorLifetime]) -> MemoryPlan;
pub fn peak_live_bytes(tensors: &[TensorLifetime]) -> usize;
pub fn transitive_reduction(node_count: usize, edges: &[(usize, usize)]) -> Vec<(usize, usize)>;
pub fn islands(node_count: usize, incidences: &[(usize, usize)]) -> (Vec<Option<usize>>, usize);
```

XNNPACK orders by size with `qsort`, which leaves equal sizes in an unspecified order. Here equal sizes keep input order, so the plan is deterministic.

## Test evidence

`tests/planner.rs` holds 15 `#[test]` functions. Each ported algorithm is checked against a brute-force oracle written in the test file, not against itself. It is also checked against the exact cases its upstream change added.

```bash
cargo test -p aether-core --test planner
```

| Test | Pins |
| --- | --- |
| `tensors_alive_at_a_common_node_never_share_a_byte` | Soundness on 500 seeded instances |
| `the_arena_is_never_smaller_than_the_peak_of_live_bytes` | `peak_live_bytes` equals a brute-force peak, and the arena is at least that peak, over 500 seeds |
| `peak_live_bytes_treats_lifetimes_as_inclusive` | Tensors sharing node 1 are both live there. Consecutive lifetimes are not |
| `a_free_leading_interval_is_reused_instead_of_appending` | XNNPACK's `ReusesLeadingGap`: offsets `[0, 100, 0]`, arena 180, which equals the live peak. Without the reuse the arena was 240 |
| `a_strictly_smaller_internal_gap_beats_the_leading_gap` | XNNPACK's `PrefersSmallerInternalGapOverLeadingGap`: offsets `[0, 100, 190, 270, 190]`, arena 340 |
| `the_leading_gap_wins_an_equal_fit` | XNNPACK's `PrefersLeadingGapOnEqualFit`: offsets `[0, 110, 210, 270, 320, 0]`, arena 365 |
| `the_reduction_has_the_same_reachability_as_the_original` | Against Warshall's closure on 300 DAGs of up to 24 nodes |
| `deleting_any_surviving_edge_changes_reachability` | Minimality on 300 DAGs |
| `a_bypass_implied_by_a_path_of_length_three_is_removed_in_every_edge_order` | TensorFlow's `TransitiveReductionIsExact`: \(0 \to 3\) is implied by \(0 \to 1 \to 2 \to 3\) and is removed in all 24 edge orders |
| `an_edge_against_the_topological_numbering_is_rejected` | Panics with "topological" |
| `island_labels_equal_connected_components_from_a_traversal` | Against depth-first search on 1,000 seeded graphs of up to 60 nodes |
| `island_labels_ignore_edge_order_and_orientation` | 500 seeds |
| `islands_are_numbered_by_their_smallest_member` | A hand case, then 500 seeds: each new label is exactly the next integer |
| `a_self_incidence_activates_a_singleton_and_an_untouched_node_stays_inactive` | \((t, t)\) activates \(t\). Untouched nodes are `None` |
| `a_long_chain_is_resolved_without_recursion` | A parent chain of depth \(n - 1\) at \(n = 200{,}000\) resolves to one island without stack overflow |

## Provenance

| Algorithm | Upstream change | Source files |
| --- | --- | --- |
| `plan_offsets` | [google/XNNPACK#10801](https://github.com/google/XNNPACK/pull/10801) (merged) | `src/memory-planner.c` (`xnn_plan_value_allocation_tracker`, `find_value_alloc_offset`), `test/subgraph/memory-planner.cc` |
| `transitive_reduction` | [tensorflow/tensorflow#124410](https://github.com/tensorflow/tensorflow/pull/124410) (merged) | `tensorflow/core/graph/collective_order.cc` (`CreateControlDependencies`) |
| `islands` | [google-deepmind/mujoco#3396](https://github.com/google-deepmind/mujoco/pull/3396) (merged) and [google-deepmind/mujoco_warp#1541](https://github.com/google-deepmind/mujoco_warp/pull/1541) (merged, GPU form) | `src/engine/engine_island.c` (`mj_dsuMerge`, `mj_dsuRoot`, `mj_dsuAssign`) |

See [Upstream Contributions](upstream.md).
