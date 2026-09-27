//! Runtime planning over tensor lifetimes and dependency graphs.
//!
//! Three algorithms ported from merged upstream changes, each deciding something
//! about a computation before it runs: where its tensors live in one arena, which
//! ordering edges its schedule needs, and which of its constraint groups can be
//! solved independently. Each is exact for what it states; the offset planner is
//! greedy, and says so, but never returns an unsound plan.
//!
//! # Offset planning — google/XNNPACK#10801
//!
//! **Problem.** Given tensors with byte sizes and inclusive lifetimes
//! `[first_use, last_use]` over node indices, assign each a byte offset in one
//! arena such that two tensors live at a common node occupy disjoint byte ranges,
//! and keep the arena small. Minimising it exactly is dynamic storage allocation
//! (Garey and Johnson, problem SR2), which is NP-complete, so the planner is
//! greedy.
//!
//! **Algorithm.** Tensors are placed in decreasing size. Each collects the byte
//! ranges of already-placed tensors whose lifetimes intersect its own, sorts and
//! coalesces them into disjoint blocks, and takes the smallest free interval that
//! fits (best fit). The leading interval `[0, first block)` is a candidate and
//! wins ties; if nothing fits, the tensor is appended after the last block.
//!
//! **Invariant.** After every placement, no two placed tensors with intersecting
//! lifetimes share a byte. Consequently the arena is at least
//! [`peak_live_bytes`]: the tensors live at the peak node pairwise intersect in
//! lifetime, so they are byte-disjoint inside the arena.
//!
//! **Complexity.** `O(n^2 log n)` time for `n` tensors in the worst case, since
//! each placement scans every earlier one and sorts up to `n` blocks; `O(n)`
//! memory.
//!
//! **Upstream fix.** The gap search considered the intervals between live blocks
//! and the space after the last, never `[0, first.start)`, and a single live
//! block took a fast path that always appended. A tensor could therefore grow the
//! arena while a leading interval large enough for it was free for its whole
//! lifetime. The change made the leading interval a best-fit candidate. The PR
//! reports, on an Intel Core i7-14700HX, FP32 MobileNet V1 peak allocation
//! falling from 23.862980 MiB to 22.331730 MiB (6.42%).
//!
//! # Transitive reduction — tensorflow/tensorflow#124410
//!
//! **Problem.** Given a DAG, find the fewest edges with the same reachability.
//! For a finite DAG this transitive reduction is unique: it is exactly the set of
//! edges `u -> v` for which no other path leads from `u` to `v`.
//!
//! **Algorithm.** Reachability is closed as a dense bit matrix in one pass over
//! the nodes in reverse topological order, so a successor's row is final before
//! it is folded into its predecessor's. Edge `u -> v` then survives iff `v` lies
//! outside the union of the rows of `u`'s successors.
//!
//! **Invariant.** Every edge is decided independently against the complete
//! closure, so the result is a function of the edge set alone and not of the
//! order in which edges arrive.
//!
//! **Complexity.** `O(E log E + (n + E) * ceil(n / 64))` time for `n` nodes and
//! `E` edges; `n^2 / 8` bytes for the matrix plus `O(n + E)` for adjacency.
//!
//! **Upstream fix.** TensorFlow's collective ordering copied a destination's
//! reachable set when an edge was created and never propagated later additions
//! back, so a redundant edge whose alternate path had length three or more
//! survived the prune; the prune's output also depended on the iteration order of
//! a pointer-keyed hash set. The change closed reachability exactly and decided
//! each edge independently. Its scratch comparison, computed from container
//! layout rather than measured: 5,387,410 bytes to 131,072 bytes at 1,024
//! collectives.
//!
//! # Island discovery — google-deepmind/mujoco#3396, google-deepmind/mujoco_warp#1541
//!
//! **Problem.** Given nodes and incidences between them, partition the nodes that
//! appear in at least one incidence into connected components ("islands"), which
//! a constraint solver can then treat independently. A node in no incidence
//! belongs to no island.
//!
//! **Algorithm.** A disjoint-set forest in a single parent array. Union links the
//! larger root under the smaller, so every root is the minimum member of its set;
//! find compresses the path it walks. One ascending pass then numbers the roots
//! in order and gives every other node its root's number.
//!
//! **Invariant.** `parent[x] <= x` for every active node, and each root is its
//! set's minimum. Labels are therefore canonical: island `k` is the one whose
//! smallest member is the `k`-th smallest among islands, whatever the order and
//! orientation of the incidences.
//!
//! **Complexity.** One machine word of scratch per node, independent of the
//! number of incidences. Linking is by minimum index, not by rank or size, so the
//! inverse-Ackermann bound does not apply; path compression alone gives
//! `O(log n)` amortized per operation (Tarjan and van Leeuwen, 1984).
//!
//! **Upstream fix.** MuJoCo built an `ntree x ntree` byte adjacency matrix and an
//! `ntree x ntree` integer column array before flood fill. The change replaced
//! both with this forest; for the generated singleton family of mujoco#3388, peak
//! `mj_island` stack use went from `5*ntree^2 + 36*ntree + 32` to
//! `16*ntree + 32` bytes, which is 84,033,568 to 65,568 bytes at 4,096 trees.
//! mujoco_warp#1541 carried the same forest and canonical labelling to the GPU,
//! hooking roots with atomic compare-and-swap; this is the sequential form.
//!
//! `persistence` computes H0 by Z2 column reduction and has no union-find to
//! share. `attention` and `scheduled` each keep a private path-compressing `find`
//! equivalent to [`islands`]' own.

#![warn(missing_docs)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

// ═══════════════════════════════════════════════════════════════════════════════
// Offset planning
// ═══════════════════════════════════════════════════════════════════════════════

/// A tensor's size and the inclusive range of nodes over which it must stay
/// resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorLifetime {
    /// Bytes the tensor occupies. A zero-size tensor is not planned.
    pub size: usize,
    /// Index of the first node that reads or writes the tensor.
    pub first_use: usize,
    /// Index of the last node that reads or writes the tensor, inclusive.
    pub last_use: usize,
}

/// Byte offsets into one arena, produced by [`plan_offsets`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPlan {
    /// `offsets[i]` is where tensor `i` starts. Zero-size tensors get 0.
    pub offsets: Vec<usize>,
    /// The furthest byte any tensor reaches: the arena size the plan needs.
    pub arena_size: usize,
}

/// Assign every tensor a byte offset such that tensors live at a common node
/// never overlap, reusing freed leading intervals.
///
/// Ported from google/XNNPACK#10801, `src/memory-planner.c`
/// (`xnn_plan_value_allocation_tracker` and `find_value_alloc_offset`). XNNPACK
/// orders by size with `qsort`, which leaves equal sizes in an unspecified
/// order; here they keep input order, so the plan is deterministic.
///
/// # Panics
///
/// If any tensor has `first_use > last_use`.
pub fn plan_offsets(tensors: &[TensorLifetime]) -> MemoryPlan {
    for (i, t) in tensors.iter().enumerate() {
        assert!(
            t.first_use <= t.last_use,
            "tensor {i} ends at node {} before it starts at node {}",
            t.last_use,
            t.first_use
        );
    }

    let mut order: Vec<usize> = (0..tensors.len())
        .filter(|&i| tensors[i].size != 0)
        .collect();
    order.sort_by(|&a, &b| tensors[b].size.cmp(&tensors[a].size));

    let mut offsets = vec![0usize; tensors.len()];
    let mut live: Vec<(usize, usize)> = Vec::with_capacity(order.len());
    let mut arena_size = 0;
    for (placed, &current) in order.iter().enumerate() {
        let t = tensors[current];
        live.clear();
        for &earlier in &order[..placed] {
            let e = tensors[earlier];
            if t.first_use <= e.last_use && e.first_use <= t.last_use {
                live.push((offsets[earlier], offsets[earlier] + e.size));
            }
        }
        let offset = best_fit_offset(&mut live, t.size);
        offsets[current] = offset;
        arena_size = arena_size.max(offset + t.size);
    }
    MemoryPlan {
        offsets,
        arena_size,
    }
}

/// Smallest free interval of at least `size` bytes among `live` blocks
/// `[start, end)`, the leading interval winning ties; else append after the last.
fn best_fit_offset(live: &mut [(usize, usize)], size: usize) -> usize {
    if live.is_empty() {
        return 0;
    }
    live.sort_unstable_by_key(|block| block.0);

    // Coalesce overlapping or touching blocks in place.
    let mut count = 1;
    for i in 1..live.len() {
        let end = live[count - 1].1;
        if live[i].0 > end {
            live[count] = live[i];
            count += 1;
        } else if live[i].1 > end {
            live[count - 1].1 = live[i].1;
        }
    }
    let blocks = &live[..count];

    let mut best_gap = usize::MAX;
    let mut offset = blocks[count - 1].1;
    if blocks[0].0 >= size {
        best_gap = blocks[0].0;
        offset = 0;
    }
    for pair in blocks.windows(2) {
        let gap = pair[1].0 - pair[0].1;
        if gap >= size && gap < best_gap {
            best_gap = gap;
            offset = pair[0].1;
        }
    }
    offset
}

/// The largest total size of tensors live at any single node: a lower bound on
/// the arena of every sound plan, [`plan_offsets`]' included.
///
/// Not part of google/XNNPACK#10801; it is the maximum-weight clique of the
/// lifetime interval graph, provided so a plan can be reported against it.
/// `O(n log n)` by a sweep over lifetime endpoints.
pub fn peak_live_bytes(tensors: &[TensorLifetime]) -> usize {
    // At a shared node, acquisitions (0) sort before releases (1): lifetimes
    // are inclusive, so a tensor released at node k is still live at k.
    let mut events: Vec<(usize, u8, usize)> = Vec::with_capacity(2 * tensors.len());
    for t in tensors {
        events.push((t.first_use, 0, t.size));
        events.push((t.last_use, 1, t.size));
    }
    events.sort_unstable();

    let (mut live, mut peak) = (0usize, 0usize);
    for (_, kind, size) in events {
        if kind == 0 {
            live += size;
            peak = peak.max(live);
        } else {
            live -= size;
        }
    }
    peak
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transitive reduction
// ═══════════════════════════════════════════════════════════════════════════════

/// The unique minimal edge set with the same reachability as the DAG `edges`
/// on nodes `0..node_count`, sorted, with repeated edges collapsed.
///
/// Nodes must be numbered topologically: every edge `(u, v)` has `u < v`, so
/// descending index is a reverse topological order and one closure pass
/// suffices. TensorFlow's collectives meet the same condition with instance
/// keys that decrease along an edge, and close in ascending key order.
///
/// Ported from tensorflow/tensorflow#124410,
/// `tensorflow/core/graph/collective_order.cc` (`CreateControlDependencies`).
///
/// # Panics
///
/// If an edge has `u >= v` or `v >= node_count`.
pub fn transitive_reduction(node_count: usize, edges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut successors = vec![Vec::new(); node_count];
    for &(u, v) in edges {
        assert!(
            u < v && v < node_count,
            "edge ({u}, {v}) breaks the topological numbering: every edge must run \
             from a lower index to a higher one below {node_count}"
        );
        successors[u].push(v);
    }
    for list in &mut successors {
        list.sort_unstable();
        list.dedup();
    }

    // Bit (a, b) set iff a path a -> ... -> b exists.
    let words = node_count.div_ceil(64);
    let mut reaches = vec![0u64; node_count * words];
    for u in (0..node_count).rev() {
        for &v in &successors[u] {
            let (head, tail) = reaches.split_at_mut(v * words);
            let row = &mut head[u * words..(u + 1) * words];
            row[v / 64] |= 1 << (v % 64);
            for (word, &reached) in row.iter_mut().zip(&tail[..words]) {
                *word |= reached;
            }
        }
    }

    // Everything u's successors reach. A successor inside that union is reached
    // through another successor, so the direct edge to it is implied. The graph
    // is acyclic, so no row contains its own node.
    let mut kept = Vec::new();
    let mut combined = vec![0u64; words];
    for (u, list) in successors.iter().enumerate() {
        combined.fill(0);
        for &v in list {
            for (word, &reached) in combined
                .iter_mut()
                .zip(&reaches[v * words..(v + 1) * words])
            {
                *word |= reached;
            }
        }
        kept.extend(
            list.iter()
                .filter(|&&v| combined[v / 64] >> (v % 64) & 1 == 0)
                .map(|&v| (u, v)),
        );
    }
    kept
}

// ═══════════════════════════════════════════════════════════════════════════════
// Island discovery
// ═══════════════════════════════════════════════════════════════════════════════

/// A node no incidence has touched. MuJoCo stores -1.
const INACTIVE: usize = usize::MAX;

/// Connected components of the nodes named by `incidences`, as
/// `(labels, island_count)`.
///
/// `labels[x]` is `None` for a node in no incidence, otherwise its island, and
/// islands are numbered in ascending order of their smallest member. An
/// incidence `(t, t)` activates `t` on its own: MuJoCo passes -1 for a static
/// endpoint and substitutes the other endpoint, which is this.
///
/// Ported from google-deepmind/mujoco#3396, `src/engine/engine_island.c`
/// (`mj_dsuMerge`, `mj_dsuRoot`, `mj_dsuAssign`); google-deepmind/mujoco_warp#1541
/// computes the same labels on the GPU.
///
/// # Panics
///
/// If an incidence names a node at or beyond `node_count`.
pub fn islands(node_count: usize, incidences: &[(usize, usize)]) -> (Vec<Option<usize>>, usize) {
    let mut parent = vec![INACTIVE; node_count];
    for &(a, b) in incidences {
        assert!(
            a < node_count && b < node_count,
            "incidence ({a}, {b}) names a node outside 0..{node_count}"
        );
        if parent[a] == INACTIVE {
            parent[a] = a;
        }
        if parent[b] == INACTIVE {
            parent[b] = b;
        }
        if parent[a] == parent[b] {
            continue;
        }
        let (root_a, root_b) = (root(&mut parent, a), root(&mut parent, b));
        // The larger root goes under the smaller, keeping every root its set's
        // minimum.
        if root_a < root_b {
            parent[root_b] = root_a;
        } else if root_b < root_a {
            parent[root_a] = root_b;
        }
    }

    // Ascending order visits a root before anything under it, and since
    // parent[x] <= x, a node's parent has already been pointed at its root.
    let mut labels = vec![None; node_count];
    let mut count = 0;
    for node in 0..node_count {
        if parent[node] == INACTIVE {
            continue;
        }
        if parent[node] == node {
            labels[node] = Some(count);
            count += 1;
        } else {
            parent[node] = parent[parent[node]];
            labels[node] = labels[parent[node]];
        }
    }
    (labels, count)
}

/// Root of an active node, compressing the path to it. Iterative: linking by
/// minimum index can build chains as long as the node count.
fn root(parent: &mut [usize], node: usize) -> usize {
    let mut root = node;
    while parent[root] != root {
        root = parent[root];
    }
    let mut current = node;
    while parent[current] != root {
        let next = parent[current];
        parent[current] = root;
        current = next;
    }
    root
}
