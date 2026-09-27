//! Invariant tests for `aether_core::orbit`.
//!
//! Each test is one proved statement turned into an executable assertion. A
//! partition bug does not crash: it returns a plausible block structure, and
//! every bound computed from it is then a plausible wrong number. These are the
//! properties a plausible-but-wrong partition or bound violates.
//!
//! Asserted here:
//!   1. Partition correctness  — `from_map` and `from_edges` match brute force.
//!   2. Relabelling invariance — orbits belong to the relation, not to names.
//!   3. Bound 1 (Thms 1, 1*)   — err ≥ n − m* ≥ n − m on every instance; attained.
//!   4. Bound 2 (Thm 2)        — no decoder beats 1/k on a k-orbit; attained.
//!   5. Bound 3 (Thm 2*)       — no decoder of the tuple beats m_join/n; attained.
//!   6. Bound 4 (Thms 6, 6*)   — precision floor holds against every truth; attained.
//!   7. Bound 5 (Thms 7, 8)    — recall floor holds exhaustively; attained; zero at shuffle.
//!   8. Degenerate inputs, and the refusals the precondition requires.

use aether_core::orbit::{
    admissible_collapsed_blocks, admissible_distinct, admissible_error_bound,
    admissible_precision_bound, certified_error_floor, certified_precision_bound,
    join_recovery_bound, orbit_error_bound, pooling_recovery_bound, recall_floor, Partition,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Deterministic sampling
// ═══════════════════════════════════════════════════════════════════════════════

/// xorshift64*. Seeded per instance so a failure is reproducible from the seed alone.
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

    /// Uniform in `0..k`, for `k ≥ 1`.
    fn below(&mut self, k: usize) -> usize {
        (self.next_u64() % k as u64) as usize
    }

    fn coin(&mut self) -> bool {
        self.next_u64() >> 63 == 1
    }

    /// Fisher-Yates permutation of `0..n`.
    fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.below(i + 1);
            p.swap(i, j);
        }
        p
    }

    /// `n` values uniform in `0..alphabet`.
    fn map(&mut self, n: usize, alphabet: usize) -> Vec<u32> {
        (0..n).map(|_| self.below(alphabet) as u32).collect()
    }
}

/// A hidden injective truth and an observed map on `n ≥ 1` elements.
///
/// `truth` is `n` distinct values drawn from `0..2n`. Each element answers
/// correctly on a coin flip and otherwise draws from a random alphabet inside
/// `0..2n`, so answers are a mix of correct, admissibly wrong, and inadmissible,
/// and orbits range from all singletons to one block.
struct Instance {
    f: Vec<u32>,
    truth: Vec<u32>,
    /// `G = R(E)` as a set: sorted, so the pairing is gone.
    gold: Vec<u32>,
}

fn instance(rng: &mut Rng, n: usize) -> Instance {
    let truth: Vec<u32> = rng.permutation(2 * n)[..n].iter().map(|&v| v as u32).collect();
    let alphabet = 1 + rng.below(2 * n);
    let f = truth
        .iter()
        .map(|&t| if rng.coin() { t } else { rng.below(alphabet) as u32 })
        .collect();
    let mut gold = truth.clone();
    gold.sort_unstable();
    Instance { f, truth, gold }
}

fn errors(f: &[u32], truth: &[u32]) -> usize {
    f.iter().zip(truth).filter(|(a, b)| a != b).count()
}

/// Every sequence of `len` values drawn from `0..k`; there are `k^len`.
fn tuples(k: usize, len: usize) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new()];
    for _ in 0..len {
        out = out
            .into_iter()
            .flat_map(|t: Vec<usize>| {
                (0..k).map(move |v| {
                    let mut t = t.clone();
                    t.push(v);
                    t
                })
            })
            .collect();
    }
    out
}

/// Every ordering of `items`: every injective truth with image `items`.
fn permutations(items: &[u32]) -> Vec<Vec<u32>> {
    if items.is_empty() {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for i in 0..items.len() {
        let mut rest = items.to_vec();
        let head = rest.remove(i);
        for mut tail in permutations(&rest) {
            tail.insert(0, head);
            out.push(tail);
        }
    }
    out
}

/// Representatives of a partition, one per orbit, ascending.
fn representatives(p: &Partition) -> Vec<usize> {
    (0..p.n()).filter(|&e| p.representative(e) == e).collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Partition correctness against brute force
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn from_map_matches_a_brute_force_fibre_computation() {
    // The orbit of e is {x : f(x) = f(e)}, computed here by an O(n²) scan with no
    // sorting. A violation means a run boundary or the stable-sort tie-break in
    // `from_map` is wrong, and every count downstream inherits it.
    for seed in 0..500u64 {
        let mut rng = Rng::new(seed);
        let n = rng.below(20);
        let alphabet = 1 + rng.below(n + 2);
        let f = rng.map(n, alphabet);
        let p = Partition::from_map(&f);

        assert_eq!(p.n(), n, "seed {seed}");
        let mut orbits = 0;
        let mut collapsed = 0;
        for (e, v) in f.iter().enumerate() {
            let fibre: Vec<usize> = (0..n).filter(|&x| f[x] == *v).collect();
            assert_eq!(p.representative(e), fibre[0], "seed {seed}: element {e}");
            assert_eq!(p.block_size(e), fibre.len(), "seed {seed}: element {e}");
            assert_eq!(p.is_flagged(e), fibre.len() > 1, "seed {seed}: element {e}");
            orbits += usize::from(fibre[0] == e);
            collapsed += usize::from(fibre[0] == e && fibre.len() > 1);
        }
        assert_eq!(p.m(), orbits, "seed {seed}");
        assert_eq!(p.collapsed_blocks(), collapsed, "seed {seed}");
        assert_eq!(p.flagged(), (0..n).filter(|&e| p.block_size(e) > 1).count());
        assert_eq!(
            p.largest(),
            (0..n).map(|e| p.block_size(e)).max().unwrap_or(0),
            "seed {seed}"
        );
    }
}

/// Components by min-label propagation to a fixed point. Independent of
/// union-find, and correct by inspection: labels only fall, only travel along
/// edges, and adjacent labels agree at the fixed point.
fn components_by_propagation(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut label: Vec<usize> = (0..n).collect();
    loop {
        let mut changed = false;
        for &(a, b) in edges {
            let low = label[a].min(label[b]);
            for v in [a, b] {
                if label[v] != low {
                    label[v] = low;
                    changed = true;
                }
            }
        }
        if !changed {
            return label;
        }
    }
}

#[test]
fn from_edges_matches_a_brute_force_closure_for_any_edge_order() {
    // H0 of a relation given pairwise. Canonical output means edge order,
    // endpoint order and repeated edges cannot change the partition.
    for seed in 0..500u64 {
        let mut rng = Rng::new(seed);
        let n = rng.below(14);
        let edges: Vec<(usize, usize)> = if n == 0 {
            Vec::new()
        } else {
            (0..rng.below(18))
                .map(|_| (rng.below(n), rng.below(n)))
                .collect()
        };
        let p = Partition::from_edges(n, &edges).expect("edges are in range");
        let label = components_by_propagation(n, &edges);

        for (e, &l) in label.iter().enumerate() {
            assert_eq!(p.representative(e), l, "seed {seed}: vertex {e}");
            assert_eq!(
                p.block_size(e),
                label.iter().filter(|&&x| x == l).count(),
                "seed {seed}: vertex {e}"
            );
        }

        let reordered: Vec<(usize, usize)> = edges
            .iter()
            .rev()
            .map(|&(a, b)| (b, a))
            .chain(edges.iter().copied())
            .collect();
        assert_eq!(Partition::from_edges(n, &reordered), Some(p), "seed {seed}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Relabelling invariance
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn orbits_are_invariant_under_relabelling_of_elements_and_values() {
    // Recoding values injectively must leave the partition identical; moving
    // element e to position π(e) must carry its orbit, and every score, with it.
    for seed in 0..300u64 {
        let mut rng = Rng::new(seed);
        let n = 1 + rng.below(20);
        let alphabet = 1 + rng.below(n);
        let f = rng.map(n, alphabet);
        let p = Partition::from_map(&f);

        let code = rng.permutation(alphabet);
        let recoded: Vec<u64> = f
            .iter()
            .map(|&v| 7919 * code[v as usize] as u64 + 3)
            .collect();
        assert_eq!(Partition::from_map(&recoded), p, "seed {seed}: value recoding");

        let pi = rng.permutation(n);
        let mut moved = vec![0u32; n];
        for (e, &v) in f.iter().enumerate() {
            moved[pi[e]] = v;
        }
        let q = Partition::from_map(&moved);
        assert_eq!(
            (q.m(), q.largest(), q.flagged(), q.collapsed_blocks()),
            (p.m(), p.largest(), p.flagged(), p.collapsed_blocks()),
            "seed {seed}: element relabelling changed a count"
        );
        for (e, &pe) in pi.iter().enumerate() {
            assert_eq!(q.block_size(pe), p.block_size(e), "seed {seed}");
            assert_eq!(q.collision(pe), p.collision(e), "seed {seed}");
            for (x, &px) in pi.iter().enumerate() {
                assert_eq!(
                    q.representative(pe) == q.representative(px),
                    p.representative(e) == p.representative(x),
                    "seed {seed}: elements {e}, {x}"
                );
            }
        }

        let edges: Vec<(usize, usize)> = (0..rng.below(2 * n))
            .map(|_| (rng.below(n), rng.below(n)))
            .collect();
        let moved_edges: Vec<(usize, usize)> = edges.iter().map(|&(a, b)| (pi[a], pi[b])).collect();
        let (g, h) = (
            Partition::from_edges(n, &edges).unwrap(),
            Partition::from_edges(n, &moved_edges).unwrap(),
        );
        for (e, &pe) in pi.iter().enumerate() {
            assert_eq!(h.block_size(pe), g.block_size(e), "seed {seed}: graph");
        }
        assert_eq!(h.m(), g.m(), "seed {seed}: graph");
    }
}

#[test]
fn collision_is_the_fraction_of_other_elements_sharing_the_value() {
    for seed in 0..300u64 {
        let mut rng = Rng::new(seed);
        let n = 2 + rng.below(18);
        let alphabet = 1 + rng.below(n);
        let f = rng.map(n, alphabet);
        let p = Partition::from_map(&f);
        for (e, v) in f.iter().enumerate() {
            let others = (0..n).filter(|&x| x != e && f[x] == *v).count();
            let score = p.collision(e).expect("n ≥ 2");
            assert_eq!(score, others as f64 / (n - 1) as f64, "seed {seed}: {e}");
            // Under an injective truth, any collision puts e in the certified set.
            assert_eq!(score > 0.0, p.is_flagged(e), "seed {seed}: {e}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Bound 1: err(f) ≥ n − m* ≥ n − m
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bound_1_never_exceeds_the_true_error_count() {
    // The certificate consults no truth; the check here does. A single instance
    // with the bound above the true count refutes the implementation.
    for seed in 0..2000u64 {
        let mut rng = Rng::new(seed);
        let n = 1 + rng.below(16);
        let x = instance(&mut rng, n);
        let p = Partition::from_map(&x.f);
        let m_star = admissible_distinct(&x.f, &x.gold).expect("|G| = n");
        let t1 = orbit_error_bound(n, p.m()).unwrap();
        let t1_star = admissible_error_bound(n, m_star).unwrap();
        let err = errors(&x.f, &x.truth);

        assert!(m_star <= p.m(), "seed {seed}: m* = {m_star} > m = {}", p.m());
        assert!(
            t1 <= t1_star && t1_star <= err,
            "seed {seed}: n − m = {t1}, n − m* = {t1_star}, true errors {err}"
        );
        let sum: usize = representatives(&p).iter().map(|&r| p.block_size(r) - 1).sum();
        assert_eq!(t1, sum, "seed {seed}: n − m is Σ (s − 1) over orbits");
        assert_eq!(
            certified_error_floor(n, p.m()),
            Some(t1 as f64 / n as f64),
            "seed {seed}"
        );
    }
}

#[test]
fn bound_1_is_attained_when_every_orbit_holds_one_correct_value() {
    // Tightness, from the proof: an orbit contributes exactly s − 1 when one
    // member is correct. With G, an orbit whose value is inadmissible contributes
    // all s, and n − m* is attained exactly.
    for seed in 0..500u64 {
        let mut rng = Rng::new(seed);
        let n = 1 + rng.below(16);
        let alphabet = 1 + rng.below(n);
        let f = rng.map(n, alphabet);
        let p = Partition::from_map(&f);
        let keep: Vec<bool> = (0..n).map(|_| rng.coin()).collect();

        // Fresh values lie outside f's alphabet, so each truth is injective.
        let fresh = |e: usize| 1000 + e as u32;
        let every: Vec<u32> = (0..n)
            .map(|e| if p.representative(e) == e { f[e] } else { fresh(e) })
            .collect();
        assert_eq!(
            errors(&f, &every),
            orbit_error_bound(n, p.m()).unwrap(),
            "seed {seed}: n − m not attained"
        );

        let some: Vec<u32> = (0..n)
            .map(|e| if p.representative(e) == e && keep[e] { f[e] } else { fresh(e) })
            .collect();
        let m_star = admissible_distinct(&f, &some).unwrap();
        assert_eq!(
            errors(&f, &some),
            admissible_error_bound(n, m_star).unwrap(),
            "seed {seed}: n − m* not attained"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Bound 2: Pr[h(f(e)) = e] ≤ 1/k on a k-orbit
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bound_2_no_decoder_beats_one_over_k_and_the_best_attains_it() {
    // Exhaustive over every map f and every decoder h on k ≤ 4 elements. h ∘ f
    // is constant on an orbit, so no h recovers more than one member of it.
    for k in 1..=4usize {
        for f in tuples(k, k) {
            let p = Partition::from_map(&f);
            let reps = representatives(&p);
            let mut best_total = 0;
            let mut best_in_orbit = vec![0usize; k];
            for h in tuples(k, k) {
                let hit: Vec<bool> = f.iter().enumerate().map(|(e, &v)| h[v] == e).collect();
                best_total = best_total.max(hit.iter().filter(|&&x| x).count());
                for &r in &reps {
                    let size = p.block_size(r);
                    let recovered = (0..k)
                        .filter(|&e| p.representative(e) == r && hit[e])
                        .count();
                    assert!(
                        recovered as f64 / size as f64 <= pooling_recovery_bound(size).unwrap(),
                        "f = {f:?}, h = {h:?}: {recovered} of an orbit of {size}"
                    );
                    best_in_orbit[r] = best_in_orbit[r].max(recovered);
                }
            }
            for &r in &reps {
                assert_eq!(best_in_orbit[r], 1, "f = {f:?}: 1/k is attained");
            }
            assert_eq!(best_total, p.m(), "f = {f:?}: one recovery per orbit");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. Bound 3: recovery from the value tuple ≤ m_join / n
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bound_3_no_decoder_of_the_tuple_beats_m_join_over_n() {
    for seed in 0..300u64 {
        let mut rng = Rng::new(seed);
        let n = 1 + rng.below(5);
        let maps: Vec<Vec<u32>> = (0..2 + rng.below(2))
            .map(|_| {
                let alphabet = 1 + rng.below(n);
                rng.map(n, alphabet)
            })
            .collect();
        let parts: Vec<Partition> = maps.iter().map(|f| Partition::from_map(f)).collect();
        let join = parts[1..]
            .iter()
            .fold(parts[0].clone(), |acc, p| acc.join(p).unwrap());

        // The join is the fibres of the tuple map, and refines every component.
        let tuples_of: Vec<Vec<u32>> = (0..n).map(|e| maps.iter().map(|f| f[e]).collect()).collect();
        assert_eq!(Partition::from_map(&tuples_of), join, "seed {seed}");
        let m_join = join.m();
        assert!(parts.iter().all(|p| m_join >= p.m()), "seed {seed}");

        // A decoder of the tuple is any function from join blocks to elements.
        let reps = representatives(&join);
        let block: Vec<usize> = (0..n)
            .map(|e| reps.binary_search(&join.representative(e)).unwrap())
            .collect();
        let ceiling = join_recovery_bound(n, m_join).unwrap();
        let mut best = 0;
        for h in tuples(n, m_join) {
            let recovered = block.iter().enumerate().filter(|&(e, &b)| h[b] == e).count();
            assert!(recovered as f64 / n as f64 <= ceiling, "seed {seed}: h = {h:?}");
            best = best.max(recovered);
        }
        assert_eq!(best, m_join, "seed {seed}: one recovery per join block");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. Bound 4: precision(S) ≥ (|S| − b_adm)/|S| ≥ (n − m)/|S|
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bound_4_precision_floor_holds_against_every_injective_truth_and_is_attained() {
    // For a fixed observation and G, enumerate every injective truth with image
    // G. The worst of them places one correct member in each admissible
    // collapsed orbit, and the floor must equal that worst case exactly.
    for seed in 0..300u64 {
        let mut rng = Rng::new(seed);
        let n = 2 + rng.below(5);
        let gold: Vec<u32> = rng.permutation(2 * n)[..n].iter().map(|&v| v as u32).collect();
        let alphabet = 1 + rng.below(2 * n);
        let f = rng.map(n, alphabet);
        let p = Partition::from_map(&f);
        let (s, b) = (p.flagged(), p.collapsed_blocks());
        let b_adm = admissible_collapsed_blocks(&f, &gold).expect("|G| = n");

        if s == 0 {
            assert_eq!(certified_precision_bound(n, p.m(), s), None, "seed {seed}");
            assert_eq!(admissible_precision_bound(s, b_adm), None, "seed {seed}");
            continue;
        }
        let t6 = certified_precision_bound(n, p.m(), s).unwrap();
        let t6_star = admissible_precision_bound(s, b_adm).unwrap();
        assert_eq!(n - p.m(), s - b, "seed {seed}: n − m = |S| − b");
        assert!(b_adm <= b && t6 >= 0.5 && t6_star >= t6, "seed {seed}");
        if b_adm == b {
            assert_eq!(t6_star, t6, "seed {seed}: no sharpening without an inadmissible orbit");
        }

        let mut worst = usize::MAX;
        for truth in permutations(&gold) {
            let wrong_in_s = (0..n)
                .filter(|&e| p.is_flagged(e) && f[e] != truth[e])
                .count();
            assert!(
                wrong_in_s as f64 / s as f64 >= t6_star,
                "seed {seed}: truth {truth:?} gives precision below the floor"
            );
            worst = worst.min(wrong_in_s);
        }
        assert_eq!(worst, s - b_adm, "seed {seed}: floor not attained");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. Bound 5: recall(S*) ≥ (n − m*)/n
// ═══════════════════════════════════════════════════════════════════════════════

/// `|S* ∩ wrong|` and `|wrong|`, where `S*` is `S` plus every inadmissible answer.
fn recall_counts(f: &[u32], truth: &[u32], gold: &[u32]) -> (usize, usize) {
    let p = Partition::from_map(f);
    let wrong: Vec<usize> = (0..f.len()).filter(|&e| f[e] != truth[e]).collect();
    let caught = wrong
        .iter()
        .filter(|&&e| p.is_flagged(e) || gold.binary_search(&f[e]).is_err())
        .count();
    (caught, wrong.len())
}

#[test]
fn bound_5_recall_floor_holds_exhaustively_and_is_attained() {
    // Every truth and every map over a vocabulary two larger than G, for
    // n = 2..=4, then seeded instances up to n = 16. The floor is compared by
    // f64 division on both sides, which is exact for equal rationals because
    // IEEE division rounds correctly.
    let mut attained = 0;
    for n in 2..=4usize {
        let gold: Vec<u32> = (0..n as u32).collect();
        for truth in permutations(&gold) {
            for f in tuples(n + 2, n) {
                let f: Vec<u32> = f.into_iter().map(|v| v as u32).collect();
                let (caught, wrong) = recall_counts(&f, &truth, &gold);
                if wrong == 0 {
                    continue;
                }
                let m_star = admissible_distinct(&f, &gold).unwrap();
                let floor = recall_floor(n, m_star).unwrap();
                let recall = caught as f64 / wrong as f64;
                assert!(recall >= floor, "f = {f:?}, truth = {truth:?}: {recall} < {floor}");
                assert_eq!(floor == 0.0, m_star == n, "f = {f:?}: zero exactly at m* = n");
                attained += usize::from(recall == floor);
            }
        }
    }
    assert!(attained > 0, "the floor is never attained, so it is not tight");

    for seed in 0..2000u64 {
        let mut rng = Rng::new(seed);
        let n = 1 + rng.below(16);
        let x = instance(&mut rng, n);
        let (caught, wrong) = recall_counts(&x.f, &x.truth, &x.gold);
        if wrong > 0 {
            let floor = recall_floor(n, admissible_distinct(&x.f, &x.gold).unwrap()).unwrap();
            assert!(caught as f64 / wrong as f64 >= floor, "seed {seed}");
        }
    }
}

#[test]
fn bound_5_is_zero_exactly_at_the_shuffle_witness() {
    // Theorem 7. f is a bijection onto G. Under R = f every answer is right;
    // under R = f ∘ σ, σ a cyclic shift, every answer is wrong. The observation
    // is the same object in both worlds, nothing is flagged, and recall is 0 —
    // so the floor must be 0 there, and is.
    let n = 8usize;
    let gold: Vec<u32> = (0..n as u32).collect();
    let f = gold.clone();
    let shifted: Vec<u32> = (0..n).map(|e| gold[(e + 1) % n]).collect();

    let p = Partition::from_map(&f);
    assert_eq!((p.m(), p.flagged()), (n, 0));
    let m_star = admissible_distinct(&f, &gold).unwrap();
    assert_eq!(m_star, n);
    assert_eq!(recall_floor(n, m_star), Some(0.0));
    assert_eq!(recall_counts(&f, &f, &gold), (0, 0), "R = f: nothing wrong");
    assert_eq!(recall_counts(&f, &shifted, &gold), (0, n), "R = f∘σ: all wrong, none caught");

    // One answer leaving G makes the floor positive, and it still holds.
    let mut off = f.clone();
    off[0] = 99;
    let m_star = admissible_distinct(&off, &gold).unwrap();
    let floor = recall_floor(n, m_star).unwrap();
    let (caught, wrong) = recall_counts(&off, &shifted, &gold);
    assert!(floor > 0.0 && caught as f64 / wrong as f64 >= floor);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 8. Degenerate inputs and refusals
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn a_gold_set_smaller_than_n_voids_the_certificate_and_is_refused() {
    // caustic's measured precondition failure: at the first-token encoding,
    // Asmara and Asuncion share token 1634 and Lusaka and Ljubljana share 444.
    // Every answer below is correct, yet n − m certifies two errors. The
    // partition cannot see this; G can, since |G| < n is the violation.
    let truth = [1634u32, 1634, 444, 444];
    let f = truth;
    let p = Partition::from_map(&f);
    assert_eq!(errors(&f, &truth), 0);
    assert_eq!(orbit_error_bound(4, p.m()), Some(2), "the false certificate");
    assert_eq!(admissible_distinct(&f, &truth), None);
    assert_eq!(admissible_collapsed_blocks(&f, &truth), None);
}

#[test]
fn empty_and_singleton_inputs_have_the_trivial_partition() {
    let empty = Partition::from_map::<u32>(&[]);
    assert_eq!(
        (empty.n(), empty.m(), empty.largest(), empty.flagged(), empty.collapsed_blocks()),
        (0, 0, 0, 0, 0)
    );
    assert_eq!(Partition::from_edges(0, &[]), Some(empty.clone()));
    assert_eq!(empty.join(&empty), Some(empty.clone()));
    assert_eq!(admissible_distinct::<u32>(&[], &[]), Some(0));
    assert_eq!(orbit_error_bound(0, 0), None, "no bound over nothing");

    let one = Partition::from_map(&[7u32]);
    assert_eq!(
        (one.n(), one.m(), one.largest(), one.flagged(), one.collapsed_blocks()),
        (1, 1, 1, 0, 0)
    );
    assert_eq!(Partition::from_edges(1, &[(0, 0)]), Some(one.clone()), "a self-loop");
    assert_eq!(one.collision(0), None, "no other element to collide with");
    assert_eq!(orbit_error_bound(1, 1), Some(0));
    assert_eq!(pooling_recovery_bound(one.largest()), Some(1.0));
    assert_eq!(join_recovery_bound(1, 1), Some(1.0));
}

#[test]
fn identity_and_constant_maps_sit_at_the_two_ends_of_every_bound() {
    let n = 20usize;

    // Identity: fully separated. The certificate is silent, and says so.
    let identity: Vec<u32> = (0..n as u32).collect();
    let p = Partition::from_map(&identity);
    assert_eq!((p.m(), p.largest(), p.flagged(), p.collapsed_blocks()), (n, 1, 0, 0));
    assert_eq!(orbit_error_bound(n, p.m()), Some(0));
    assert_eq!(certified_error_floor(n, p.m()), Some(0.0));
    assert_eq!(certified_precision_bound(n, p.m(), p.flagged()), None);
    assert_eq!(pooling_recovery_bound(p.largest()), Some(1.0));
    assert!((0..n).all(|e| p.collision(e) == Some(0.0)));

    // Constant: total collapse onto one value.
    let constant = vec![7u32; n];
    let c = Partition::from_map(&constant);
    assert_eq!((c.m(), c.largest(), c.flagged(), c.collapsed_blocks()), (1, n, n, 1));
    assert_eq!(orbit_error_bound(n, c.m()), Some(n - 1));
    assert_eq!(certified_precision_bound(n, c.m(), c.flagged()), Some(19.0 / 20.0));
    assert_eq!(pooling_recovery_bound(c.largest()), Some(1.0 / 20.0));
    assert!((0..n).all(|e| c.collision(e) == Some(1.0)));

    // With G disjoint from the shared value, no member can be correct: every
    // bound that reads G moves to its extreme.
    let gold: Vec<u32> = (100..100 + n as u32).collect();
    let m_star = admissible_distinct(&constant, &gold).unwrap();
    let b_adm = admissible_collapsed_blocks(&constant, &gold).unwrap();
    assert_eq!((m_star, b_adm), (0, 0));
    assert_eq!(admissible_error_bound(n, m_star), Some(n));
    assert_eq!(admissible_precision_bound(c.flagged(), b_adm), Some(1.0));
    assert_eq!(recall_floor(n, m_star), Some(1.0));
}

#[test]
fn every_bound_refuses_counts_no_partition_can_produce() {
    assert_eq!(orbit_error_bound(0, 1), None);
    assert_eq!(orbit_error_bound(5, 0), None);
    assert_eq!(orbit_error_bound(5, 6), None);
    assert_eq!(certified_error_floor(5, 0), None);
    assert_eq!(admissible_error_bound(0, 0), None);
    assert_eq!(admissible_error_bound(5, 6), None);
    assert_eq!(admissible_error_bound(5, 0), Some(5), "m* = 0 is total inadmissible collapse");
    assert_eq!(recall_floor(5, 6), None);
    assert_eq!(pooling_recovery_bound(0), None);
    assert_eq!(join_recovery_bound(4, 0), None);
    assert_eq!(join_recovery_bound(4, 5), None);
    assert_eq!(certified_precision_bound(12, 12, 0), None, "empty S has no precision");
    assert_eq!(certified_precision_bound(20, 15, 3), None, "5 errors cannot fit in |S| = 3");
    assert_eq!(certified_precision_bound(5, 1, 6), None, "|S| > n");
    assert_eq!(admissible_precision_bound(0, 0), None, "empty S has no precision");
    assert_eq!(admissible_precision_bound(4, 3), None, "3 orbits of ≥ 2 need |S| ≥ 6");
    assert_eq!(Partition::from_edges(3, &[(0, 3)]), None, "edge out of range");
    assert_eq!(
        Partition::from_map(&[1u8, 2]).join(&Partition::from_map(&[1u8, 2, 3])),
        None,
        "join over different element counts"
    );
}
