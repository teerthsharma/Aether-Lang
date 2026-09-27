# Upstream Contributions

Several integrated modules port algorithms that the author first contributed to other projects. This page lists those upstream pull requests, the algorithmic idea of each, and the `aether-core` module that carries it, if any. States, merge dates and line counts come from `gh pr view`, queried on 2026-09-27: 9 merged, 1 open, 2 closed unmerged.

## Merged

| Pull request | Merged | +/− | Algorithmic idea | In Aether |
| --- | --- | --- | --- | --- |
| [triton-lang/kernels#22](https://github.com/triton-lang/kernels/pull/22) | 2026-07-28 | +804/−1 | Topology-derived sparse attention: a CSR block schedule from sink blocks, a local window and top-k \(H_0\)-persistence salient blocks | `aether_core::scheduled` ([Status Matrix](../reference/status.md)), ported before this integration |
| [google/XNNPACK#10801](https://github.com/google/XNNPACK/pull/10801) | 2026-07-23 | +131/−8 | Best-fit gap search in the memory planner also considers the space before the first live block | [`planner::plan_offsets`](planner.md#offset-planning-googlexnnpack10801) |
| [google-deepmind/mujoco#3396](https://github.com/google-deepmind/mujoco/pull/3396) | 2026-07-20 | +645/−93 | A disjoint-set forest over constraint–tree incidences replaces the \(\text{ntree}^2\) adjacency matrix in island discovery | [`planner::islands`](planner.md#island-discovery-google-deepmindmujoco3396-and-mujoco_warp1541) |
| [google-deepmind/mujoco#3450](https://github.com/google-deepmind/mujoco/pull/3450) | 2026-08-03 | +16/−12 | Convex-hull graph construction inverts `vert_globalid` once, replacing \(3V^2 - 6V\) probes with \(8V - 12\) | Documented only |
| [tensorflow/tensorflow#124410](https://github.com/tensorflow/tensorflow/pull/124410) | 2026-08-05 | +362/−26 | Exact transitive reduction of collective control edges, decided per edge against the full closure | [`planner::transitive_reduction`](planner.md#transitive-reduction-tensorflowtensorflow124410) |
| [google/highway#3244](https://github.com/google/highway/pull/3244) | 2026-08-05 | +173/−5 | PHast builder: only keys with overlapping slice windows can collide, so collision and scan tests are pruned by slice structure | Documented only |
| [NVIDIA/NeMo-Relay#481](https://github.com/NVIDIA/NeMo-Relay/pull/481) | 2026-08-10 | +1370/−86 | A stable prompt scaffold, chosen by strict-majority descent of a prefix tree, is reused across adaptive calls | Documented only |
| [dsx-ai-factory/topograph#432](https://github.com/dsx-ai-factory/topograph/pull/432) | 2026-08-19 | +145/−96 | Gates the main ClusterRole rules by engine and provider in a Helm chart | Not mathematics |
| [google-deepmind/mujoco_warp#1541](https://github.com/google-deepmind/mujoco_warp/pull/1541) | 2026-08-25 | +615/−1184 | Linear-memory GPU disjoint-set island discovery, hooking roots by atomic compare-and-swap, with MuJoCo's canonical labels | [`planner::islands`](planner.md#island-discovery-google-deepmindmujoco3396-and-mujoco_warp1541) (sequential form) |

## Open

| Pull request | State | +/− | Algorithmic idea | In Aether |
| --- | --- | --- | --- | --- |
| [vllm-project/vllm#47942](https://github.com/vllm-project/vllm/pull/47942) | Open | +583/−0 | Segment witnesses: a covering constraint on the sparse MLA indexer top-k, with one witness per context segment the learned row misses | [`kvwitness`](kvwitness.md), from commit `a41354cb34` |

## Closed without merging

| Pull request | State | +/− | Title | In Aether |
| --- | --- | --- | --- | --- |
| [openxla/xla#46539](https://github.com/openxla/xla/pull/46539) | Closed | +5/−1 | "fix(gpu): make reduction group order deterministic" | Not carried |
| [facebook/pyrefly#4180](https://github.com/facebook/pyrefly/pull/4180) | Closed | +70/−2 | "test: capture capped recheck propagation panic" | Not carried |

## Reading this table

A merged pull request records that a maintainer accepted the change. It is not evidence for any claim on this site. The evidence for an Aether module is its own test suite, run by `cargo test --workspace --exclude aether-kernel` in CI. The figures quoted from pull request bodies on the module pages are the pull requests' own measurements, on the hardware those pull requests name, and are attributed to them there. "Documented only" means the idea is recorded here and no Aether module implements it.
