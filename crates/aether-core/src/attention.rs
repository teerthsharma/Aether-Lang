//! Sparse attention with topology-derived key selection — CPU reference.
//!
//! This is a reference implementation, not a fast kernel. It exists so the
//! correctness contracts in `tests/attention_contracts.rs` have something to
//! bind to, and so the same-budget ablation can be run at all. There is no GPU
//! path in this workspace; a fast kernel needs one, and claiming a speedup
//! without it would be measuring nothing.
//!
//! What it does provide:
//!
//!   - `dense_attention`: ordinary scaled dot-product attention, the reference
//!     every sparse path is checked against.
//!   - `sparse_attention`: the same computation restricted to a boolean mask,
//!     with an explicit guard for rows that select nothing.
//!   - `Selector`: the key-selection rules, including the topological one and the
//!     three baselines an honest evaluation needs.
//!   - `attention_mass_recovered`: the diagnostic that says whether topology did
//!     the work or sparsity did.
//!
//! No backward pass. When one is added, a `gradcheck` against the dense path on
//! the same mask is the first test that must accompany it: a forward-correct
//! kernel with a wrong backward trains to a plausible worse optimum, which loss
//! curves alone will not reveal.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use libm::{exp, sqrt};

/// How a row picks the keys it may attend to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Selector {
    /// Every key. The control: sparse must equal dense here.
    Dense,

    /// A sliding window of the `window` most recent keys (including self).
    ///
    /// The locality-only baseline. If a topological selector cannot beat this, it
    /// is rediscovering adjacency, not topology.
    Local { window: usize },

    /// `budget` keys drawn uniformly without replacement from the legal set.
    ///
    /// The sparsity baseline. Any gain a selector shows over this is the gain
    /// attributable to the selection rule rather than to sparsity itself.
    Random { budget: usize, seed: u64 },

    /// The `budget` keys with the genuinely largest attention weights, computed
    /// densely.
    ///
    /// Unimplementable in production — it needs the scores it is meant to avoid
    /// computing — but decisive as a diagnostic upper bound.
    OracleTopK { budget: usize },

    /// The `budget` keys nearest the query in the shared embedding space, keeping
    /// only those within `radius_scale` times the row's median query-key
    /// distance.
    ///
    /// This is the mechanism under test: the claim is that geometric proximity in
    /// key space proxies attention mass.
    ///
    /// `radius_scale` is **relative**, not an absolute length. An absolute radius
    /// makes the selector silently dataset-dependent — the same failure the
    /// persistence scale-equivariance test exists to catch — and in practice it
    /// under-spends the budget: with an absolute radius of 0.6 against a median
    /// query-key distance of 2.4, the selector picked 1.0 keys per row while its
    /// same-budget baselines picked 5.5, and lost the ablation on budget rather
    /// than on mechanism. Use `f64::INFINITY` for pure budget-nearest.
    Topological { budget: usize, radius_scale: f64 },

    /// Route through a topological clustering of key *directions*, then rank the
    /// resulting candidate set by the exact dot product.
    ///
    /// The fix the ablation demanded. `Topological` conflated two jobs: finding
    /// candidates and scoring them. Euclidean proximity does both badly once key
    /// norms vary, because `‖q − k‖² = ‖q‖² + ‖k‖² − 2·q·k` — a large-norm key is
    /// far away *and* high-scoring. This variant splits them:
    ///
    ///   1. Single-linkage (H0) clustering of the unit-normalised keys builds the
    ///      candidate set. Normalising first makes the clustering invariant to
    ///      per-key rescaling, which is exactly what the old rule could not be.
    ///   2. The exact dot product ranks within the candidates, restoring the norm
    ///      sensitivity the geometry deliberately discarded.
    ///
    /// Cost: `clusters` centroid dot products to route, plus one per candidate to
    /// rank — not `seq`. That is what makes it a sparse method rather than a
    /// re-described dense one.
    TopologicalRouted { budget: usize, clusters: usize },

    /// Route when [`routing_plan`] says routing will pay, and fall back to a
    /// sliding window of the same budget when it will not.
    ///
    /// `TopologicalRouted` is a real sparsity win only when the keys have H0
    /// structure. On a cloud with no density gaps, single-linkage chains — 64
    /// uniform keys give components of size `[61, 1, 1, 1]` — so routing to the
    /// top cluster is routing to everything, at 0.999x the dense dot-product
    /// count. That is not a clustering bug; it is H0 correctly reporting that
    /// there is nothing to route on.
    ///
    /// A condition nobody checks at runtime is an assumption. This variant checks
    /// it, which is what makes routing safe to enable by default: the fallback
    /// guarantees the selector never inspects more scores than dense would.
    Adaptive { budget: usize, clusters: usize },
}

/// What routing will cost on a given key tensor, decided before any query runs.
///
/// The H0 clustering is computed once per key tensor and amortised over every
/// query, head, and layer that reuses it, so this is cheap relative to attention
/// itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutingPlan {
    /// Dot products per row the router will perform, as a fraction of dense.
    pub cost_ratio: f64,

    /// Share of keys in the largest H0 component. Near 1 means chaining: one
    /// giant component and singletons, so the candidate set is everything.
    pub largest_cluster_share: f64,

    /// Ratio of the first merge height above the cut to the last one below it,
    /// derived from the H0 barcode alone.
    ///
    /// A clear separation between clusters shows up as a large ratio; chaining
    /// shows up as a ratio near 1. This is the signal a runtime could cache and
    /// reuse without redoing the clustering.
    pub gap_ratio: f64,

    /// The `cost_ratio` below which routing is judged worthwhile.
    pub threshold: f64,

    /// Whether to route. `cost_ratio < threshold`.
    pub worth_routing: bool,
}

/// Cost ratio below which routing is judged to pay for itself.
///
/// Routing has to beat dense by enough to be worth the clustering and the extra
/// control flow; 0.6 is the measured point where the discount is unambiguous
/// (4 real key clusters give 0.449, uniform keys give 0.999). Nothing depends on
/// the exact value — `the_plan_declines_to_route_exactly_when_routing_would_not_pay`
/// asserts the decision against whatever this is.
pub const ROUTING_COST_THRESHOLD: f64 = 0.6;

/// Decide whether topological routing will pay on this key tensor.
pub fn routing_plan(
    k: &[f64],
    seq: usize,
    head_dim: usize,
    clusters: usize,
    budget: usize,
    causal: bool,
) -> RoutingPlan {
    let (assignment, merge_heights) = single_linkage_clusters(k, seq, head_dim, clusters, true);
    let cluster_count = assignment.iter().copied().max().map_or(0, |m| m + 1);

    let mut sizes = vec![0usize; cluster_count.max(1)];
    for &label in &assignment {
        sizes[label] += 1;
    }
    let largest = sizes.iter().copied().max().unwrap_or(0);

    let cost_ratio = selection_dot_cost(
        Selector::TopologicalRouted { budget, clusters },
        k,
        seq,
        head_dim,
        causal,
    ) / dense_dot_cost(seq, causal);

    // The cut sits between merge `seq - cluster_count - 1` and `seq - cluster_count`:
    // that many merges are taken, the rest are not. The ratio across it is how
    // sharply the data separates.
    let gap_ratio = {
        let taken = seq.saturating_sub(cluster_count);
        match (
            taken.checked_sub(1).and_then(|i| merge_heights.get(i)),
            merge_heights.get(taken),
        ) {
            (Some(&below), Some(&above)) if below > 0.0 => above / below,
            // No merge on one side of the cut, or a degenerate zero-height merge:
            // report 1.0, which reads as "no separation" and never triggers routing
            // on its own.
            _ => 1.0,
        }
    };

    RoutingPlan {
        cost_ratio,
        largest_cluster_share: largest as f64 / seq.max(1) as f64,
        gap_ratio,
        threshold: ROUTING_COST_THRESHOLD,
        worth_routing: cost_ratio < ROUTING_COST_THRESHOLD,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Attention
// ═══════════════════════════════════════════════════════════════════════════════

/// Scaled dot-product attention over a single head.
///
/// `q`, `k`, `v` are row-major `[seq, head_dim]`. Returns `[seq, head_dim]`.
///
/// Equivalent to `sparse_attention` with an all-true mask, and asserted to be so.
pub fn dense_attention(q: &[f64], k: &[f64], v: &[f64], seq: usize, head_dim: usize) -> Vec<f64> {
    sparse_attention(q, k, v, seq, head_dim, &vec![true; seq * seq])
}

/// Scaled dot-product attention restricted to `mask`, a row-major `[seq, seq]`
/// boolean where `mask[i * seq + j]` permits query `i` to attend to key `j`.
///
/// A row that selects no keys returns zeros rather than NaN. That case is
/// reachable: a topological selector on an isolated token — a point that is its
/// own connected component — selects nothing. Softmax over an empty set is
/// undefined, and the naive form (exp of all `-inf`) yields `0/0`.
pub fn sparse_attention(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    seq: usize,
    head_dim: usize,
    mask: &[bool],
) -> Vec<f64> {
    assert_eq!(q.len(), seq * head_dim, "q must be [seq, head_dim]");
    assert_eq!(k.len(), seq * head_dim, "k must be [seq, head_dim]");
    assert_eq!(v.len(), seq * head_dim, "v must be [seq, head_dim]");
    assert_eq!(mask.len(), seq * seq, "mask must be [seq, seq]");

    let scale = 1.0 / sqrt(head_dim as f64);
    let mut out = vec![0.0f64; seq * head_dim];
    let mut logits = vec![0.0f64; seq];

    for i in 0..seq {
        // Scores for the permitted keys, and their running maximum. Subtracting
        // the max before exp is what keeps large logits from overflowing to +inf.
        let mut max_logit = f64::NEG_INFINITY;
        let mut any = false;
        for j in 0..seq {
            if !mask[i * seq + j] {
                continue;
            }
            let mut dot = 0.0;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            let logit = dot * scale;
            logits[j] = logit;
            if logit > max_logit {
                max_logit = logit;
            }
            any = true;
        }

        // The guard. Zeros, not NaN, and documented as the defined value.
        if !any {
            continue;
        }

        let mut denominator = 0.0;
        for j in 0..seq {
            if mask[i * seq + j] {
                denominator += exp(logits[j] - max_logit);
            }
        }

        for j in 0..seq {
            if !mask[i * seq + j] {
                continue;
            }
            let weight = exp(logits[j] - max_logit) / denominator;
            for d in 0..head_dim {
                out[i * head_dim + d] += weight * v[j * head_dim + d];
            }
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════════════
// Selection
// ═══════════════════════════════════════════════════════════════════════════════

/// Build the `[seq, seq]` boolean mask a selector produces.
///
/// `causal` restricts every row `i` to keys `j <= i`. Selectors apply their rule
/// within the causal set, never outside it, so no budget is spent on keys that
/// would be masked away afterwards.
///
/// Every row selects at least one key: a row's own position is always legal under
/// causal masking, so the budgeted selectors always have something to fall back
/// on. The all-masked guard in `sparse_attention` therefore covers hand-built
/// masks, not selector output.
pub fn select_mask(
    selector: Selector,
    q: &[f64],
    k: &[f64],
    _v: &[f64],
    seq: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<bool> {
    let selector = resolve(selector, k, seq, head_dim, causal);
    let mut mask = vec![false; seq * seq];

    // Routing structure is built once over all keys, not per row: a key's cluster
    // does not depend on which query is looking at it. This is what keeps the
    // clustering cost amortised across the sequence rather than paid per row.
    let routing = match selector {
        Selector::TopologicalRouted { clusters, .. } => {
            let (assignment, _) = single_linkage_clusters(k, seq, head_dim, clusters, true);
            let cluster_count = assignment.iter().copied().max().map_or(0, |m| m + 1);

            // Centroids of the unit-normalised keys: the direction each cluster
            // represents. Routing compares queries against these, so the router
            // does `cluster_count` dot products instead of `seq`.
            let mut centroids = vec![0.0f64; cluster_count * head_dim];
            let mut members = vec![0usize; cluster_count];
            for (t, &label) in assignment.iter().enumerate() {
                let norm = sqrt(
                    (0..head_dim)
                        .map(|d| k[t * head_dim + d] * k[t * head_dim + d])
                        .sum::<f64>(),
                );
                let inverse = if norm > 0.0 { 1.0 / norm } else { 0.0 };
                for d in 0..head_dim {
                    centroids[label * head_dim + d] += k[t * head_dim + d] * inverse;
                }
                members[label] += 1;
            }
            for label in 0..cluster_count {
                let count = members[label].max(1) as f64;
                for d in 0..head_dim {
                    centroids[label * head_dim + d] /= count;
                }
            }
            Some((assignment, centroids, cluster_count))
        }
        _ => None,
    };

    for i in 0..seq {
        let legal_end = if causal { i + 1 } else { seq };
        let legal: Vec<usize> = (0..legal_end).collect();

        let chosen: Vec<usize> = match selector {
            Selector::Dense => legal.clone(),

            Selector::Local { window } => {
                let start = legal_end.saturating_sub(window.max(1));
                (start..legal_end).collect()
            }

            Selector::Random { budget, seed } => {
                // Deterministic per (seed, row): the same call must always produce
                // the same mask, or every A/B comparison is measuring noise.
                let mut pool = legal.clone();
                let mut state = splitmix(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let take = budget.min(pool.len());
                for slot in 0..take {
                    state = splitmix(state);
                    let pick = slot + (state as usize) % (pool.len() - slot);
                    pool.swap(slot, pick);
                }
                pool.truncate(take);
                pool
            }

            Selector::OracleTopK { budget } => {
                let mut scored: Vec<(usize, f64)> = legal
                    .iter()
                    .map(|&j| (j, dot(q, k, i, j, head_dim)))
                    .collect();
                // Descending by score, ties broken by index so the result is
                // deterministic rather than sort-stability-dependent.
                scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                scored.truncate(budget);
                scored.into_iter().map(|(j, _)| j).collect()
            }

            Selector::Topological {
                budget,
                radius_scale,
            } => {
                // Geometric proximity in key space, NOT the dot product: this rule
                // must not peek at the scores it claims to predict, or the
                // comparison against OracleTopK is circular.
                let mut scored: Vec<(usize, f64)> = legal
                    .iter()
                    .map(|&j| (j, key_distance(q, k, i, j, head_dim)))
                    .collect();
                scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));

                // The radius is a multiple of this row's median query-key
                // distance, which makes the whole rule scale-equivariant: scaling
                // q and k by c scales every distance and the median alike, so the
                // selected set is unchanged.
                let cutoff = if radius_scale.is_finite() {
                    radius_scale * scored[scored.len() / 2].1
                } else {
                    f64::INFINITY
                };

                let mut chosen: Vec<usize> = scored
                    .iter()
                    .filter(|&&(_, distance)| distance <= cutoff)
                    .take(budget)
                    .map(|&(j, _)| j)
                    .collect();

                // An isolated token has no neighbour inside the radius. Falling
                // back to self keeps the row well-defined without silently
                // widening the radius, which would make the parameter meaningless.
                if chosen.is_empty() {
                    chosen.push(if causal { i } else { i.min(seq - 1) });
                }
                chosen
            }

            // `resolve` collapses Adaptive before the loop, so it cannot appear
            // here. Panicking rather than silently picking a branch keeps the
            // "cost reported equals path taken" guarantee honest.
            Selector::Adaptive { .. } => unreachable!("Adaptive is resolved before selection"),

            Selector::TopologicalRouted { budget, .. } => {
                let (assignment, centroids, cluster_count) =
                    routing.as_ref().expect("routing built for this selector");

                // Step 1: rank clusters by query-to-centroid alignment. This is
                // the only place the query meets the topology, and it costs
                // `cluster_count` dot products.
                let mut ranked: Vec<(usize, f64)> = (0..*cluster_count)
                    .map(|label| {
                        let score: f64 = (0..head_dim)
                            .map(|d| q[i * head_dim + d] * centroids[label * head_dim + d])
                            .sum();
                        (label, score)
                    })
                    .collect();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

                // Step 2: pull legal members of the best clusters until there are
                // at least `budget` candidates to choose between.
                let mut candidates: Vec<usize> = Vec::new();
                for (label, _) in ranked {
                    if candidates.len() >= budget {
                        break;
                    }
                    candidates.extend(legal.iter().copied().filter(|&j| assignment[j] == label));
                }

                // Step 3: the exact dot product decides within the candidate set.
                // This is what the nearest-neighbour rule was missing: geometry
                // finds where to look, the score decides what to take.
                let mut scored: Vec<(usize, f64)> = candidates
                    .into_iter()
                    .map(|j| (j, dot(q, k, i, j, head_dim)))
                    .collect();
                scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                scored.truncate(budget);

                let mut chosen: Vec<usize> = scored.into_iter().map(|(j, _)| j).collect();
                if chosen.is_empty() {
                    chosen.push(if causal { i } else { i.min(seq - 1) });
                }
                chosen
            }
        };

        for j in chosen {
            mask[i * seq + j] = true;
        }
    }
    mask
}

/// Single-linkage clustering of `points` (row-major `[count, dim]`) cut so that
/// `clusters` components remain.
///
/// Returns `(assignment, merge_heights)`. `assignment[i]` is the component label
/// of point `i`, canonicalised so labels appear in ascending order of first
/// occurrence — otherwise the labels would depend on union-find internals and the
/// result would not be comparable across runs. `merge_heights` is every merge
/// distance in ascending order.
///
/// Single-linkage merge heights are exactly the finite H0 deaths of the
/// Vietoris-Rips persistence of the same cloud; `single_linkage_merge_heights_
/// equal_the_h0_persistence_deaths` asserts that against the persistence engine
/// rather than trusting two implementations of one theorem.
///
/// With `normalize`, points are projected to the unit sphere first, which makes
/// the clustering invariant to per-point rescaling. That is the property the
/// routed selector needs and the nearest-neighbour selector lacks.
///
/// ponytail: O(n^2) edge enumeration plus a sort. Fine at the sequence lengths
/// this reference kernel targets; a real router would use a neighbour graph.
pub fn single_linkage_clusters(
    points: &[f64],
    count: usize,
    dim: usize,
    clusters: usize,
    normalize: bool,
) -> (Vec<usize>, Vec<f64>) {
    assert_eq!(points.len(), count * dim, "points must be [count, dim]");

    let prepared: Vec<f64> = if normalize {
        let mut out = vec![0.0; count * dim];
        for t in 0..count {
            let norm = sqrt(
                (0..dim)
                    .map(|d| points[t * dim + d] * points[t * dim + d])
                    .sum::<f64>(),
            );
            // A zero vector has no direction; leave it at the origin rather than
            // inventing one, and let it cluster with whatever is nearest.
            let inverse = if norm > 0.0 { 1.0 / norm } else { 0.0 };
            for d in 0..dim {
                out[t * dim + d] = points[t * dim + d] * inverse;
            }
        }
        out
    } else {
        points.to_vec()
    };

    let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(count * count / 2);
    for i in 0..count {
        for j in (i + 1)..count {
            let mut sum = 0.0;
            for d in 0..dim {
                let delta = prepared[i * dim + d] - prepared[j * dim + d];
                sum += delta * delta;
            }
            edges.push((sqrt(sum), i, j));
        }
    }
    // Ties broken by index so the merge order, and therefore the cut, is
    // deterministic rather than sort-stability-dependent.
    edges.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

    let mut parent: Vec<usize> = (0..count).collect();
    let mut merge_heights = Vec::with_capacity(count.saturating_sub(1));
    let mut components = count;
    let target = clusters.clamp(1, count.max(1));

    for (height, i, j) in edges {
        let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
        if ri == rj {
            continue;
        }
        // Every merge is recorded, including those past the cut: the heights are
        // the full H0 barcode, and the cut is applied separately below.
        merge_heights.push(height);
        if components > target {
            parent[ri] = rj;
            components -= 1;
        }
    }

    // Canonical labels: first occurrence order.
    let mut label_of = vec![usize::MAX; count];
    let mut assignment = vec![0usize; count];
    let mut next = 0usize;
    for t in 0..count {
        let root = find(&mut parent, t);
        if label_of[root] == usize::MAX {
            label_of[root] = next;
            next += 1;
        }
        assignment[t] = label_of[root];
    }
    (assignment, merge_heights)
}

fn find(parent: &mut [usize], x: usize) -> usize {
    let mut root = x;
    while parent[root] != root {
        root = parent[root];
    }
    // Path compression, iterative: recursion here would blow the kernel stack on
    // a long chain, and single-linkage produces exactly those.
    let mut current = x;
    while parent[current] != root {
        let next = parent[current];
        parent[current] = root;
        current = next;
    }
    root
}

fn dot(q: &[f64], k: &[f64], i: usize, j: usize, head_dim: usize) -> f64 {
    let mut sum = 0.0;
    for d in 0..head_dim {
        sum += q[i * head_dim + d] * k[j * head_dim + d];
    }
    sum
}

/// Euclidean distance between query `i` and key `j` in the shared embedding
/// space. The topological selector's only view of the data.
fn key_distance(q: &[f64], k: &[f64], i: usize, j: usize, head_dim: usize) -> f64 {
    let mut sum = 0.0;
    for d in 0..head_dim {
        let delta = q[i * head_dim + d] - k[j * head_dim + d];
        sum += delta * delta;
    }
    sqrt(sum)
}

fn splitmix(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ═══════════════════════════════════════════════════════════════════════════════
// The ablation
// ═══════════════════════════════════════════════════════════════════════════════

/// Collapse [`Selector::Adaptive`] to the concrete rule it will actually run.
///
/// Everything downstream — mask construction, cost accounting, the ablation — goes
/// through this, so the adaptive selector cannot report one path's cost while
/// running another's.
fn resolve(selector: Selector, k: &[f64], seq: usize, head_dim: usize, causal: bool) -> Selector {
    match selector {
        Selector::Adaptive { budget, clusters } => {
            if routing_plan(k, seq, head_dim, clusters, budget, causal).worth_routing {
                Selector::TopologicalRouted { budget, clusters }
            } else {
                // Dense, not a cheap window.
                //
                // Measured: on unstructured keys a budget-6 sliding window lands at
                // placement +0.014 — indistinguishable from picking keys at random.
                // There is no cheap-and-good option when the keys have no structure,
                // because finding the top-k without computing the scores is exactly
                // what the structure was supposed to make possible.
                //
                // So the fallback preserves correctness rather than cost. The
                // guarantee this variant offers is "never worse than dense, in cost
                // or in quality", which is the only one that is safe to enable by
                // default. A caller that would rather trade quality for cost should
                // ask for `Local` explicitly.
                Selector::Dense
            }
        }
        other => other,
    }
}

/// Mean number of `head_dim`-length dot products a selector performs per row.
///
/// Quality without cost is not a result. `attention_mass_recovered` says how good
/// a selector's choice is; this says what the choice cost to make. A selector that
/// recovers 90% of attention mass while examining every key is dense attention
/// with extra steps, and only this function reveals that.
///
/// Dense attention costs `i + 1` dot products at row `i` under causal masking, so
/// compare against `dense_dot_cost`.
///
/// Selectors that inspect no scores (`Local`, `Random`) cost nothing to evaluate
/// and report 0; the number is about search, not about the attention arithmetic
/// that follows. `OracleTopK` costs the full dense count by construction — that is
/// precisely why it is a diagnostic and not an implementation.
pub fn selection_dot_cost(
    selector: Selector,
    k: &[f64],
    seq: usize,
    head_dim: usize,
    causal: bool,
) -> f64 {
    let total: usize = match resolve(selector, k, seq, head_dim, causal) {
        Selector::Adaptive { .. } => unreachable!("Adaptive is resolved before costing"),
        Selector::Local { .. } | Selector::Random { .. } => 0,
        Selector::Dense | Selector::OracleTopK { .. } | Selector::Topological { .. } => {
            (0..seq).map(|i| if causal { i + 1 } else { seq }).sum()
        }
        Selector::TopologicalRouted { budget, clusters } => {
            let (assignment, _) = single_linkage_clusters(k, seq, head_dim, clusters, true);
            let cluster_count = assignment.iter().copied().max().map_or(0, |m| m + 1);
            let mut sizes = vec![0usize; cluster_count];
            for &label in &assignment {
                sizes[label] += 1;
            }
            let mut labels: Vec<usize> = (0..cluster_count).collect();
            labels.sort_by_key(|&l| core::cmp::Reverse(sizes[l]));

            (0..seq)
                .map(|i| {
                    let legal_end = if causal { i + 1 } else { seq };
                    let mut candidates = 0usize;
                    for &label in &labels {
                        if candidates >= budget {
                            break;
                        }
                        candidates += (0..legal_end).filter(|&j| assignment[j] == label).count();
                    }
                    cluster_count + candidates
                })
                .sum()
        }
    };
    total as f64 / seq as f64
}

/// Mean dot products per row that dense attention performs. The denominator for
/// any sparsity claim.
pub fn dense_dot_cost(seq: usize, causal: bool) -> f64 {
    let total: usize = (0..seq).map(|i| if causal { i + 1 } else { seq }).sum();
    total as f64 / seq as f64
}

/// Fraction of true dense attention mass that a selector's chosen keys carry,
/// averaged over rows.
///
/// For each row, computes the full softmax over the legal keys and sums the
/// weights of the keys the selector picked. `Selector::OracleTopK` maximises this
/// by construction, so it is the upper bound at a given budget; `Random` is the
/// floor. Where a selector lands between them is the measurement that decides
/// whether the mechanism contributes anything:
///
///   - at random → the mechanism contributes nothing, and any downstream gain is
///     sparsity acting as a regulariser
///   - at oracle → the selector's summary is a good proxy for attention mass,
///     which is the actual scientific claim
///   - between → report the fraction, not the headline
pub fn attention_mass_recovered(
    selector: Selector,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    seq: usize,
    head_dim: usize,
    causal: bool,
) -> f64 {
    let mask = select_mask(selector, q, k, v, seq, head_dim, causal);
    let scale = 1.0 / sqrt(head_dim as f64);
    let mut total = 0.0;

    for i in 0..seq {
        let legal_end = if causal { i + 1 } else { seq };

        let mut max_logit = f64::NEG_INFINITY;
        let mut logits = vec![0.0f64; legal_end];
        for (j, logit) in logits.iter_mut().enumerate() {
            *logit = dot(q, k, i, j, head_dim) * scale;
            if *logit > max_logit {
                max_logit = *logit;
            }
        }

        let denominator: f64 = logits.iter().map(|&l| exp(l - max_logit)).sum();
        let recovered: f64 = (0..legal_end)
            .filter(|&j| mask[i * seq + j])
            .map(|j| exp(logits[j] - max_logit) / denominator)
            .sum();
        total += recovered;
    }
    total / seq as f64
}
