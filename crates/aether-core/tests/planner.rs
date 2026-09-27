//! Invariant tests for `aether_core::planner`.
//!
//! Each ported algorithm is checked against a brute-force oracle written here,
//! not against itself, and against the exact cases its upstream change added.
//! A planner that overlaps two live tensors, a reduction that drops a needed
//! edge, or a labelling that splits a component all return plausible output;
//! these properties are what such output violates.
//!
//! Properties asserted here:
//!   1. Plan soundness      — tensors alive at a common node never share a byte.
//!   2. Lower bound         — the arena is at least the peak of live bytes.
//!   3. Leading-gap reuse   — the three cases google/XNNPACK#10801 added.
//!   4. Reachability        — the reduction has the original transitive closure.
//!   5. Minimality          — deleting any surviving edge changes the closure.
//!   6. Regression          — tensorflow/tensorflow#124410's length-three bypass.
//!   7. Order independence  — every order of that edge set reduces identically.
//!   8. Components          — island labels equal an independent traversal.
//!   9. Canonical labels    — islands are numbered by their smallest member,
//!      whatever the order and orientation of the incidences.

use aether_core::planner::{
    islands, peak_live_bytes, plan_offsets, transitive_reduction, TensorLifetime,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Deterministic sampling
// ═══════════════════════════════════════════════════════════════════════════════

/// xorshift64*. Seeded per test so a failure is reproducible from the seed alone.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `0..n`.
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Fisher-Yates shuffle in place.
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Memory planning oracles
// ═══════════════════════════════════════════════════════════════════════════════

fn tensor(size: usize, first_use: usize, last_use: usize) -> TensorLifetime {
    TensorLifetime {
        size,
        first_use,
        last_use,
    }
}

/// Up to 40 tensors over 32 nodes, one in ten of size zero.
fn random_tensors(rng: &mut Rng) -> Vec<TensorLifetime> {
    let n = 1 + rng.below(40);
    (0..n)
        .map(|_| {
            let first = rng.below(24);
            let size = if rng.below(10) == 0 {
                0
            } else {
                1 + rng.below(1024)
            };
            tensor(size, first, first + rng.below(8))
        })
        .collect()
}

fn lifetimes_overlap(a: &TensorLifetime, b: &TensorLifetime) -> bool {
    a.first_use <= b.last_use && b.first_use <= a.last_use
}

/// The peak by definition: sum the live bytes at every node and take the max.
fn brute_force_peak(tensors: &[TensorLifetime]) -> usize {
    let horizon = tensors.iter().map(|t| t.last_use + 1).max().unwrap_or(0);
    (0..horizon)
        .map(|node| {
            tensors
                .iter()
                .filter(|t| t.first_use <= node && node <= t.last_use)
                .map(|t| t.size)
                .sum()
        })
        .max()
        .unwrap_or(0)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Memory planning: google/XNNPACK#10801
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn tensors_alive_at_a_common_node_never_share_a_byte() {
    for seed in 0..500 {
        let mut rng = Rng::new(seed);
        let tensors = random_tensors(&mut rng);
        let plan = plan_offsets(&tensors);
        assert_eq!(plan.offsets.len(), tensors.len(), "seed {seed}");

        let extent = tensors
            .iter()
            .zip(&plan.offsets)
            .map(|(t, &offset)| offset + t.size)
            .max()
            .unwrap_or(0);
        assert_eq!(
            plan.arena_size, extent,
            "seed {seed}: arena_size must be the furthest byte any tensor reaches"
        );

        for (i, a) in tensors.iter().enumerate() {
            for (j, b) in tensors.iter().enumerate().skip(i + 1) {
                if !lifetimes_overlap(a, b) {
                    continue;
                }
                let (oa, ob) = (plan.offsets[i], plan.offsets[j]);
                let share_a_byte = oa < ob + b.size && ob < oa + a.size;
                assert!(
                    !share_a_byte,
                    "seed {seed}: tensors {i} [{oa}, {}) and {j} [{ob}, {}) are both \
                     live at some node and overlap in memory",
                    oa + a.size,
                    ob + b.size
                );
            }
        }
    }
}

#[test]
fn the_arena_is_never_smaller_than_the_peak_of_live_bytes() {
    let (mut worst, mut total, mut counted) = (1.0f64, 0.0f64, 0usize);
    for seed in 0..500 {
        let mut rng = Rng::new(seed);
        let tensors = random_tensors(&mut rng);
        let peak = peak_live_bytes(&tensors);
        assert_eq!(peak, brute_force_peak(&tensors), "seed {seed}: peak");

        let arena = plan_offsets(&tensors).arena_size;
        assert!(
            arena >= peak,
            "seed {seed}: arena {arena} below peak {peak}"
        );
        if peak > 0 {
            let ratio = arena as f64 / peak as f64;
            worst = worst.max(ratio);
            total += ratio;
            counted += 1;
        }
    }
    println!(
        "  planned / lower bound over {counted} instances: mean {:.4}, worst {:.4}",
        total / counted as f64,
        worst
    );
}

#[test]
fn peak_live_bytes_treats_lifetimes_as_inclusive() {
    // Both are used at node 1, so both are live there.
    assert_eq!(peak_live_bytes(&[tensor(100, 0, 1), tensor(80, 1, 2)]), 180);
    // Consecutive but disjoint: never live together.
    assert_eq!(peak_live_bytes(&[tensor(100, 0, 0), tensor(80, 1, 1)]), 100);
    assert_eq!(peak_live_bytes(&[]), 0);
}

#[test]
fn a_free_leading_interval_is_reused_instead_of_appending() {
    // google/XNNPACK#10801, test/subgraph/memory-planner.cc, ReusesLeadingGap.
    let tensors = [tensor(100, 0, 1), tensor(80, 1, 2), tensor(60, 2, 3)];
    let plan = plan_offsets(&tensors);
    assert_eq!(plan.offsets, vec![0, 100, 0]);
    assert_eq!(plan.arena_size, 180);

    // Without leading-gap reuse, a lone live block took a fast path that always
    // appended: tensor 2 saw only tensor 1 at [100, 180) and went to 180, for an
    // arena of 240, although [0, 100) was free for tensor 2's whole lifetime.
    const WITHOUT_LEADING_GAP: usize = 240;
    assert!(plan.arena_size < WITHOUT_LEADING_GAP);
    // And the reuse is optimal here: 180 is the live peak at node 1.
    assert_eq!(plan.arena_size, peak_live_bytes(&tensors));
}

#[test]
fn a_strictly_smaller_internal_gap_beats_the_leading_gap() {
    // google/XNNPACK#10801, PrefersSmallerInternalGapOverLeadingGap. Tensor 4
    // sees free [0, 100) and [190, 270); best fit takes the 80-byte gap.
    let tensors = [
        tensor(100, 0, 0),
        tensor(90, 0, 2),
        tensor(80, 0, 0),
        tensor(70, 0, 2),
        tensor(60, 2, 3),
    ];
    let plan = plan_offsets(&tensors);
    assert_eq!(plan.offsets, vec![0, 100, 190, 270, 190]);
    assert_eq!(plan.arena_size, 340);
}

#[test]
fn the_leading_gap_wins_an_equal_fit() {
    // google/XNNPACK#10801, PrefersLeadingGapOnEqualFit. Tensor 5 sees two free
    // 110-byte intervals, [0, 110) and [210, 320); the tie goes to the first.
    let tensors = [
        tensor(110, 0, 0),
        tensor(100, 0, 2),
        tensor(60, 0, 0),
        tensor(50, 0, 0),
        tensor(45, 0, 2),
        tensor(40, 2, 3),
    ];
    let plan = plan_offsets(&tensors);
    assert_eq!(plan.offsets, vec![0, 110, 210, 270, 320, 0]);
    assert_eq!(plan.arena_size, 365);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transitive reduction: tensorflow/tensorflow#124410
// ═══════════════════════════════════════════════════════════════════════════════

/// A DAG on up to 24 nodes numbered topologically (`u < v`), in shuffled edge
/// order, sometimes with a repeated edge.
fn random_dag(rng: &mut Rng) -> (usize, Vec<(usize, usize)>) {
    let n = 1 + rng.below(24);
    let density = 1 + rng.below(5);
    let mut edges = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            if rng.below(10) < density {
                edges.push((u, v));
            }
        }
    }
    if !edges.is_empty() && rng.below(2) == 0 {
        let repeated = edges[rng.below(edges.len())];
        edges.push(repeated);
    }
    rng.shuffle(&mut edges);
    (n, edges)
}

/// Transitive closure by Warshall's algorithm on a dense boolean matrix.
fn closure(n: usize, edges: &[(usize, usize)]) -> Vec<bool> {
    let mut reach = vec![false; n * n];
    for &(u, v) in edges {
        reach[u * n + v] = true;
    }
    for k in 0..n {
        for i in 0..n {
            if reach[i * n + k] {
                for j in 0..n {
                    if reach[k * n + j] {
                        reach[i * n + j] = true;
                    }
                }
            }
        }
    }
    reach
}

#[test]
fn the_reduction_has_the_same_reachability_as_the_original() {
    for seed in 0..300 {
        let mut rng = Rng::new(seed);
        let (n, edges) = random_dag(&mut rng);
        let reduced = transitive_reduction(n, &edges);

        for edge in &reduced {
            assert!(
                edges.contains(edge),
                "seed {seed}: {edge:?} was not an input edge"
            );
        }
        assert!(
            reduced.windows(2).all(|w| w[0] < w[1]),
            "seed {seed}: output must be sorted with no repeats: {reduced:?}"
        );
        assert_eq!(
            closure(n, &reduced),
            closure(n, &edges),
            "seed {seed}: reachability changed"
        );
    }
}

#[test]
fn deleting_any_surviving_edge_changes_reachability() {
    for seed in 0..300 {
        let mut rng = Rng::new(seed);
        let (n, edges) = random_dag(&mut rng);
        let reduced = transitive_reduction(n, &edges);
        let full = closure(n, &reduced);

        for skip in 0..reduced.len() {
            let mut fewer = reduced.clone();
            let removed = fewer.remove(skip);
            assert_ne!(
                closure(n, &fewer),
                full,
                "seed {seed}: {removed:?} is implied by the other edges and should \
                 have been removed"
            );
        }
    }
}

#[test]
fn a_bypass_implied_by_a_path_of_length_three_is_removed_in_every_edge_order() {
    // tensorflow/tensorflow#124410, TransitiveReductionIsExact. The collectives
    // c4, c3, c2, c1 are nodes 0, 1, 2, 3: TensorFlow's instance keys decrease
    // along an edge where these indices increase. Edges in the order the old code
    // created them: 0 -> 3 is implied by 0 -> 1 -> 2 -> 3.
    //
    // The old code copied the destination's reachable set when an edge was
    // created and never propagated later additions back, so node 1's set stayed
    // {2} after 2 -> 3 arrived, the length-three path was invisible, and 0 -> 3
    // survived: four edges where three are correct.
    let created = [(0, 1), (0, 3), (1, 2), (2, 3)];
    let expected = vec![(0, 1), (1, 2), (2, 3)];
    assert_eq!(transitive_reduction(4, &created), expected);

    // The old prune also depended on hash-set iteration order. Every order of
    // the same edge set must produce the same reduction.
    let mut orders = 0;
    for code in 0usize..256 {
        let perm = [code & 3, (code >> 2) & 3, (code >> 4) & 3, (code >> 6) & 3];
        if perm.iter().fold(0, |mask, &i| mask | (1 << i)) != 0b1111 {
            continue;
        }
        let edges: Vec<_> = perm.iter().map(|&i| created[i]).collect();
        assert_eq!(transitive_reduction(4, &edges), expected, "order {edges:?}");
        orders += 1;
    }
    assert_eq!(orders, 24);
}

#[test]
#[should_panic(expected = "topological")]
fn an_edge_against_the_topological_numbering_is_rejected() {
    // A back edge would make the single closure pass silently wrong, so it is
    // refused rather than reduced.
    transitive_reduction(2, &[(1, 0)]);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Island discovery: google-deepmind/mujoco#3396, mujoco_warp#1541
// ═══════════════════════════════════════════════════════════════════════════════

/// Up to 60 nodes and as many incidences, including self-incidences and repeats.
fn random_graph(rng: &mut Rng) -> (usize, Vec<(usize, usize)>) {
    let n = 1 + rng.below(60);
    let m = rng.below(n + 1);
    let edges = (0..m).map(|_| (rng.below(n), rng.below(n))).collect();
    (n, edges)
}

/// Depth-first components over an adjacency list, numbered in the order their
/// smallest member is met; nodes on no edge are `None`.
fn traversal_components(n: usize, edges: &[(usize, usize)]) -> (Vec<Option<usize>>, usize) {
    let mut adjacent = vec![Vec::new(); n];
    for &(a, b) in edges {
        adjacent[a].push(b);
        adjacent[b].push(a);
    }
    let mut label = vec![None; n];
    let mut count = 0;
    for start in 0..n {
        if adjacent[start].is_empty() || label[start].is_some() {
            continue;
        }
        let mut stack = vec![start];
        label[start] = Some(count);
        while let Some(node) = stack.pop() {
            for &next in &adjacent[node] {
                if label[next].is_none() {
                    label[next] = Some(count);
                    stack.push(next);
                }
            }
        }
        count += 1;
    }
    (label, count)
}

#[test]
fn island_labels_equal_connected_components_from_a_traversal() {
    for seed in 0..1000 {
        let mut rng = Rng::new(seed);
        let (n, edges) = random_graph(&mut rng);
        assert_eq!(
            islands(n, &edges),
            traversal_components(n, &edges),
            "seed {seed}: {edges:?}"
        );
    }
}

#[test]
fn island_labels_ignore_edge_order_and_orientation() {
    for seed in 0..500 {
        let mut rng = Rng::new(seed);
        let (n, mut edges) = random_graph(&mut rng);
        let original = islands(n, &edges);

        rng.shuffle(&mut edges);
        for edge in edges.iter_mut() {
            if rng.below(2) == 0 {
                *edge = (edge.1, edge.0);
            }
        }
        assert_eq!(islands(n, &edges), original, "seed {seed}");
    }
}

#[test]
fn islands_are_numbered_by_their_smallest_member() {
    // {1, 3, 5, 6} has smallest member 1 and is island 0; {4} is island 1.
    let (labels, count) = islands(7, &[(5, 6), (3, 1), (4, 4), (6, 3)]);
    assert_eq!(
        labels,
        vec![None, Some(0), None, Some(0), Some(1), Some(0), Some(0)]
    );
    assert_eq!(count, 2);

    for seed in 0..500 {
        let mut rng = Rng::new(seed);
        let (n, edges) = random_graph(&mut rng);
        let (labels, count) = islands(n, &edges);
        // Scanning nodes upward, each new label must be exactly the next
        // integer: island k's smallest member precedes island k + 1's.
        let mut next = 0;
        for label in labels.iter().flatten() {
            assert!(*label <= next, "seed {seed}: label {label} before {next}");
            if *label == next {
                next += 1;
            }
        }
        assert_eq!(next, count, "seed {seed}");
    }
}

#[test]
fn a_self_incidence_activates_a_singleton_and_an_untouched_node_stays_inactive() {
    // MuJoCo passes -1 for a static endpoint and substitutes the other endpoint,
    // so a constraint against the world activates one tree on its own. Here that
    // is the self-incidence (t, t).
    let (labels, count) = islands(3, &[(2, 2)]);
    assert_eq!(labels, vec![None, None, Some(0)]);
    assert_eq!(count, 1);

    assert_eq!(islands(4, &[]), (vec![None; 4], 0));
    assert_eq!(islands(0, &[]), (Vec::new(), 0));
}

#[test]
fn a_long_chain_is_resolved_without_recursion() {
    // Linking the larger root under the smaller one, with edges arriving from
    // the top, builds a parent chain of depth n - 1. The closing edge forces a
    // root search along all of it; a recursive find would overflow the stack.
    let n = 200_000;
    let mut edges: Vec<(usize, usize)> = (1..n).rev().map(|i| (i, i - 1)).collect();
    edges.push((n - 1, 0));
    let (labels, count) = islands(n, &edges);
    assert_eq!(count, 1);
    assert!(labels.iter().all(|&label| label == Some(0)));
}
