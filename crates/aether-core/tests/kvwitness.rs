//! Contracts for `aether_core::kvwitness`, the segment-witness policy of
//! vllm-project/vllm#47942 (commit a41354cb34).
//!
//! Written before the module. The first two and the rejection cases are the
//! source's own hand examples, so a divergence from the Python original fails
//! here before any property is consulted. The rest are the contracts the source
//! states — causal, duplicate-free, prefix-preserving, bounded — checked over
//! randomised rows built the way `_random_case` in the source test builds them.
//!
//! Two tests pin behaviour that is weaker than a first reading of the PR suggests:
//! the merged row is positional, so permuting the learned row can change the
//! output set, and occupancy is read from slots the merge then overwrites, so a
//! segment can end up uncovered. Both are the source's behaviour, not defects of
//! the port, and are asserted so that neither is later "fixed" by accident.

use std::collections::BTreeSet;

use aether_core::attention::{select_mask, Selector};
use aether_core::kvwitness::{
    apply_topology_witnesses, attention_mass_recall, locality_null, random_null,
    scatter_topology_witnesses, segment_coverage, segment_of, topology_witness_indices,
    WitnessError, MAX_TOPOLOGY_SEGMENTS,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Deterministic inputs
// ═══════════════════════════════════════════════════════════════════════════════

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

/// One indexer batch: `rows x topk` distinct in-range offsets per row, padded
/// with -1, as `_random_case` in the source test builds it. Duplicates in the
/// input would make the no-duplicate contract untestable.
struct Case {
    lens: Vec<i32>,
    learned: Vec<i32>,
    topk: usize,
    segments: usize,
}

fn random_case(rng: &mut Rng, segments: Option<usize>) -> Case {
    let rows = 1 + rng.below(5);
    let topk = 1 + rng.below(39);
    let segments = segments.unwrap_or_else(|| 1 + rng.below(MAX_TOPOLOGY_SEGMENTS));
    let context = 1 + rng.below(199);
    let lens: Vec<i32> = (0..rows).map(|_| rng.below(context + 1) as i32).collect();
    let mut learned = vec![-1; rows * topk];
    for (row, &len) in lens.iter().enumerate() {
        let mut pool: Vec<i32> = (0..len).collect();
        rng.shuffle(&mut pool);
        let n = pool.len().min(topk);
        learned[row * topk..row * topk + n].copy_from_slice(&pool[..n]);
    }
    Case {
        lens,
        learned,
        topk,
        segments,
    }
}

fn valid(row: &[i32]) -> Vec<i32> {
    row.iter().copied().filter(|&x| x >= 0).collect()
}

fn segments_hit(tokens: &[i32], len: usize, segments: usize) -> BTreeSet<usize> {
    tokens
        .iter()
        .map(|&x| segment_of(x as usize, len, segments))
        .collect()
}

fn all_segments(len: usize, segments: usize) -> BTreeSet<usize> {
    (0..segments.min(len)).collect()
}

fn assert_distinct(tokens: &[i32], what: &str) {
    let set: BTreeSet<i32> = tokens.iter().copied().collect();
    assert_eq!(
        set.len(),
        tokens.len(),
        "{what}: duplicate token in {tokens:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. The source's hand examples
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn a_witness_fills_only_the_segment_the_learned_row_misses() {
    // Context 16 in 4 segments starts them at 0, 4, 8, 12. The learned row
    // occupies segments 0, 1 and 2, so only segment 3 earns a witness.
    let witnesses = topology_witness_indices(&[0, 5, 9, -1], 4, &[16], 4).unwrap();
    assert_eq!(witnesses, vec![-1, -1, -1, 12]);
}

#[test]
fn the_merge_keeps_the_prefix_and_writes_witnesses_into_the_tail() {
    let merged = apply_topology_witnesses(&[0, 5, 9, -1], 4, &[16], 2, 4, 2).unwrap();
    assert_eq!(merged, vec![0, 5, 12, -1]);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Disabled, fallback and refusal contracts
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn a_disabled_policy_is_the_identity() {
    let mut rng = Rng::new(1);
    for _ in 0..100 {
        let case = random_case(&mut rng, None);

        // Zero segments is the off switch, and it touches nothing.
        let mut buffer = case.learned.clone();
        scatter_topology_witnesses(&mut buffer, case.topk, &case.lens, 0).unwrap();
        assert_eq!(buffer, case.learned);

        // A zero replacement budget, by either route, is the other passthrough.
        let s = case.segments;
        let kept_whole =
            apply_topology_witnesses(&case.learned, case.topk, &case.lens, case.topk, s, s);
        assert_eq!(kept_whole.unwrap(), case.learned);
        let no_budget = apply_topology_witnesses(&case.learned, case.topk, &case.lens, 0, s, 0);
        assert_eq!(no_budget.unwrap(), case.learned);
    }
}

#[test]
fn short_and_empty_rows_follow_the_source_fallback() {
    // L < S: S_eff = L, every segment is one token, and every token the learned
    // row does not hold is witnessed.
    let witnesses = topology_witness_indices(&[1, -1, -1], 3, &[3], 8).unwrap();
    assert_eq!(witnesses, vec![0, -1, 2, -1, -1, -1, -1, -1]);

    // A zero-length row stays all padding: a witness at 0 would point the kernel
    // at a token the row cannot attend to. A negative length is treated the same.
    for len in [0, -3] {
        let merged = apply_topology_witnesses(&[-1; 4], 4, &[len], 0, 4, 4).unwrap();
        assert_eq!(merged, vec![-1; 4], "context length {len}");
    }
}

#[test]
fn an_out_of_range_learned_index_is_clamped_rather_than_refused() {
    // 20 is not a legal offset in a row of 16. The source does not reject it: its
    // segment floor(20 * 4 / 16) = 5 is clamped to S - 1 = 3, so it marks the
    // last segment covered, and in the retained prefix it passes through.
    let witnesses = topology_witness_indices(&[20, -1], 2, &[16], 4).unwrap();
    assert_eq!(witnesses, vec![0, 4, 8, -1]);
    let merged = apply_topology_witnesses(&[20, -1, -1], 3, &[16], 1, 4, 2).unwrap();
    assert_eq!(merged, vec![20, 0, 4]);

    // With S_eff < S the clamped segment lies outside the partition and marks
    // nothing: L = 4, S = 8 clamps 20 to segment 7, beyond S_eff = 4.
    let witnesses = topology_witness_indices(&[20, -1], 2, &[4], 8).unwrap();
    assert_eq!(witnesses, vec![0, 1, 2, 3, -1, -1, -1, -1]);
}

#[test]
fn malformed_calls_are_refused() {
    let learned = [0; 8];
    let lens = [8, 8];

    for (segments, keep) in [(0, 2), (MAX_TOPOLOGY_SEGMENTS + 1, 2)] {
        assert_eq!(
            apply_topology_witnesses(&learned, 4, &lens, keep, segments, 2),
            Err(WitnessError::SegmentsOutOfRange {
                num_segments: segments
            })
        );
    }
    assert_eq!(
        apply_topology_witnesses(&learned, 4, &lens, 5, 4, 2),
        Err(WitnessError::LearnedKeepOutOfRange {
            learned_keep: 5,
            topk: 4
        })
    );
    assert_eq!(
        topology_witness_indices(&learned, 4, &lens, 0),
        Err(WitnessError::SegmentsOutOfRange { num_segments: 0 })
    );

    // Coordinate mismatch: row-relative offsets without their row's length have
    // no range to be read against, and are refused rather than guessed at.
    let mismatch = WitnessError::RowsMismatch {
        learned_len: 8,
        topk: 4,
        rows: 1,
    };
    assert_eq!(
        topology_witness_indices(&learned, 4, &[8], 4),
        Err(mismatch)
    );
    assert_eq!(
        apply_topology_witnesses(&learned, 4, &[8], 2, 4, 2),
        Err(mismatch)
    );
    let mut buffer = learned;
    assert_eq!(
        scatter_topology_witnesses(&mut buffer, 4, &[8], 4),
        Err(mismatch)
    );
    assert_eq!(
        buffer, learned,
        "a refused call must leave the buffer untouched"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. The construction
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn every_witness_lands_in_its_own_segment() {
    // sigma(ceil(s L / S_eff)) = s is what makes de-duplication free. Checked
    // against the module's own edges, over every S the merge accepts.
    for len in 1..=200usize {
        for segments in 1..=MAX_TOPOLOGY_SEGMENTS {
            let witnesses = topology_witness_indices(&[-1], 1, &[len as i32], segments).unwrap();
            let effective = segments.min(len);
            for (s, &edge) in witnesses.iter().enumerate() {
                if s < effective {
                    assert_eq!(
                        segment_of(edge as usize, len, segments),
                        s,
                        "L {len}, S {segments}"
                    );
                } else {
                    assert_eq!(
                        edge, -1,
                        "segment {s} does not exist at L {len}, S {segments}"
                    );
                }
            }
        }
    }
    // Rounding the edge down does not round-trip: L = 10, S = 4, s = 1 gives
    // floor(10 / 4) = 2, which lies in segment 0.
    assert_eq!(segment_of(2, 10, 4), 0);
}

#[test]
fn witnesses_are_causal_distinct_disjoint_and_complete() {
    for segments in [1, 3, 8, MAX_TOPOLOGY_SEGMENTS] {
        let mut rng = Rng::new(segments as u64);
        for _ in 0..50 {
            let case = random_case(&mut rng, Some(segments));
            let witnesses =
                topology_witness_indices(&case.learned, case.topk, &case.lens, segments).unwrap();
            assert_eq!(witnesses.len(), case.lens.len() * segments);

            for (row, &len) in case.lens.iter().enumerate() {
                let emitted = valid(&witnesses[row * segments..(row + 1) * segments]);
                let learned = valid(&case.learned[row * case.topk..(row + 1) * case.topk]);

                assert!(
                    emitted.iter().all(|&x| x < len),
                    "row {row}: {emitted:?} not < {len}"
                );
                assert_distinct(&emitted, "witnesses");
                assert!(
                    emitted.iter().all(|x| !learned.contains(x)),
                    "witness repeats a learned token"
                );

                if len > 0 {
                    let union: Vec<i32> = learned.iter().chain(&emitted).copied().collect();
                    assert_eq!(
                        segments_hit(&union, len as usize, segments),
                        all_segments(len as usize, segments)
                    );
                }
            }
        }
    }
}

#[test]
fn the_serving_row_is_causal_distinct_ordered_and_within_budget() {
    let mut rng = Rng::new(5);
    for _ in 0..200 {
        let case = random_case(&mut rng, None);
        let (topk, segments) = (case.topk, case.segments);
        let keep = topk.saturating_sub(segments);
        let budget = segments.min(topk - keep);

        let witnesses =
            topology_witness_indices(&case.learned, topk, &case.lens, segments).unwrap();
        let mut out = case.learned.clone();
        scatter_topology_witnesses(&mut out, topk, &case.lens, segments).unwrap();
        assert_eq!(out.len(), case.learned.len(), "row width changed");

        for (row, &len) in case.lens.iter().enumerate() {
            let before = &case.learned[row * topk..(row + 1) * topk];
            let after = &out[row * topk..(row + 1) * topk];

            // Retention: the prefix is bit-identical, so at least `keep` learned
            // slots survive whatever the witnesses do.
            assert_eq!(&after[..keep], &before[..keep]);

            // Budget and order: exactly min(#witnesses, budget) slots change, as
            // one run starting at `keep`, holding witnesses in ascending order.
            let written = valid(&witnesses[row * segments..(row + 1) * segments])
                .len()
                .min(budget);
            let changed: Vec<usize> = (0..topk).filter(|&i| after[i] != before[i]).collect();
            assert_eq!(changed, (keep..keep + written).collect::<Vec<_>>());
            assert!(after[keep..keep + written].windows(2).all(|w| w[0] < w[1]));

            let tokens = valid(after);
            assert!(
                tokens.iter().all(|&x| x < len),
                "non-causal index in {after:?}, L {len}"
            );
            assert_distinct(&tokens, "serving row");
        }
    }
}

#[test]
fn every_segment_is_covered_when_the_reserved_tail_is_padding() {
    // The serving tail is S slots and at most S witnesses exist, so when those
    // slots hold padding nothing is overwritten and the output covers every
    // segment of every row.
    let mut rng = Rng::new(9);
    for _ in 0..200 {
        let segments = 1 + rng.below(MAX_TOPOLOGY_SEGMENTS);
        let len = 1 + rng.below(300);
        let held = rng.below(len.min(24) + 1);
        let mut pool: Vec<i32> = (0..len as i32).collect();
        rng.shuffle(&mut pool);

        let mut row = pool[..held].to_vec();
        row.resize(held + segments, -1);
        scatter_topology_witnesses(&mut row, held + segments, &[len as i32], segments).unwrap();

        assert_eq!(
            segments_hit(&valid(&row), len, segments),
            all_segments(len, segments)
        );
        assert_eq!(segment_coverage(&row, len, segments), 1.0);
    }
}

#[test]
fn overwriting_a_sole_tail_occupant_uncovers_its_segment() {
    // Occupancy is read from the whole learned row, tail included, and the merge
    // then overwrites the tail. Token 12 alone covers segment 3 and sits in slot 4,
    // which the first witness takes, so segment 3 is lost from the output.
    let mut row = vec![0, 1, 2, 3, 12, -1, -1, -1];
    scatter_topology_witnesses(&mut row, 8, &[16], 4).unwrap();
    assert_eq!(row, vec![0, 1, 2, 3, 4, 8, -1, -1]);
    assert_eq!(segment_coverage(&row, 16, 4), 0.75);
}

#[test]
fn permuting_the_learned_row_leaves_the_witness_set_unchanged() {
    // Occupancy is a set function of the learned row, so the witnesses are too.
    // The merge is positional, so the output set is invariant under permutations
    // of the retained prefix, but not of the whole row.
    let mut rng = Rng::new(13);
    for _ in 0..100 {
        let case = random_case(&mut rng, None);
        let (topk, segments) = (case.topk, case.segments);
        let keep = topk.saturating_sub(segments);

        let mut whole = case.learned.clone();
        let mut prefix = case.learned.clone();
        for row in 0..case.lens.len() {
            rng.shuffle(&mut whole[row * topk..(row + 1) * topk]);
            rng.shuffle(&mut prefix[row * topk..row * topk + keep]);
        }

        assert_eq!(
            topology_witness_indices(&whole, topk, &case.lens, segments).unwrap(),
            topology_witness_indices(&case.learned, topk, &case.lens, segments).unwrap()
        );

        let mut original = case.learned.clone();
        scatter_topology_witnesses(&mut original, topk, &case.lens, segments).unwrap();
        scatter_topology_witnesses(&mut prefix, topk, &case.lens, segments).unwrap();
        for row in 0..case.lens.len() {
            let a: BTreeSet<i32> = valid(&original[row * topk..(row + 1) * topk])
                .into_iter()
                .collect();
            let b: BTreeSet<i32> = valid(&prefix[row * topk..(row + 1) * topk])
                .into_iter()
                .collect();
            assert_eq!(a, b);
        }
    }

    // Moving a learned token across the keep boundary changes which one the
    // merge retains: {0, 5, 12} before, {5, 9, 12} after, same witness.
    assert_eq!(
        apply_topology_witnesses(&[0, 5, 9, -1], 4, &[16], 2, 4, 2).unwrap(),
        vec![0, 5, 12, -1]
    );
    assert_eq!(
        apply_topology_witnesses(&[9, -1, 0, 5], 4, &[16], 2, 4, 2).unwrap(),
        vec![9, -1, 12, 5]
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Null models and metrics
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn the_nulls_spend_exactly_the_policy_budget() {
    let mut rng = Rng::new(17);
    for trial in 0..60 {
        let case = random_case(&mut rng, None);
        let mut out = case.learned.clone();
        scatter_topology_witnesses(&mut out, case.topk, &case.lens, case.segments).unwrap();

        for (row, &len) in case.lens.iter().enumerate() {
            let len = len as usize;
            let k = valid(&out[row * case.topk..(row + 1) * case.topk]).len();

            let random = random_null(len, k, trial);
            assert_eq!(random.len(), k, "random null spent a different budget");
            assert!(
                random.windows(2).all(|w| w[0] < w[1]),
                "random null not sorted and distinct"
            );
            assert!(random.iter().all(|&x| (x as usize) < len));
            assert_eq!(
                random,
                random_null(len, k, trial),
                "random null is not deterministic"
            );

            let local = locality_null(len, k);
            assert_eq!(local, ((len - k) as i32..len as i32).collect::<Vec<_>>());
        }
    }
}

#[test]
fn the_nulls_are_the_last_row_of_the_attention_selectors() {
    // Reuse, asserted: a second implementation of either null could drift from
    // the one the attention ablation measures.
    let (len, k) = (37usize, 9usize);
    for (selector, null) in [
        (
            Selector::Random { budget: k, seed: 3 },
            random_null(len, k, 3),
        ),
        (Selector::Local { window: k }, locality_null(len, k)),
    ] {
        let mask = select_mask(selector, &[], &[], &[], len, 0, true);
        let last: Vec<i32> = (0..len)
            .filter(|&j| mask[(len - 1) * len + j])
            .map(|j| j as i32)
            .collect();
        assert_eq!(null, last, "{selector:?}");
    }
}

#[test]
fn coverage_and_recall_match_a_hand_computed_example() {
    // L = 8, S = 4: segments {0,1} {2,3} {4,5} {6,7}. Dyadic weights, so every
    // sum below is exact.
    let attention = [0.5, 0.0, 0.25, 0.0, 0.0, 0.125, 0.0625, 0.0625];

    // Padding, a duplicate and an out-of-range index count for nothing.
    let noisy = [0, 5, 5, -1, 9];
    assert_eq!(segment_coverage(&noisy, 8, 4), 0.5);
    assert_eq!(attention_mass_recall(&noisy, &attention), 0.625);

    // Policy against the locality null at the same budget of four tokens.
    let mut policy = vec![0, 5, -1, -1, -1, -1];
    scatter_topology_witnesses(&mut policy, 6, &[8], 4).unwrap();
    assert_eq!(policy, vec![0, 5, 2, 6, -1, -1]);
    assert_eq!(segment_coverage(&policy, 8, 4), 1.0);
    assert_eq!(attention_mass_recall(&policy, &attention), 0.9375);

    let local = locality_null(8, 4);
    assert_eq!(local, vec![4, 5, 6, 7]);
    assert_eq!(segment_coverage(&local, 8, 4), 0.5);
    assert_eq!(attention_mass_recall(&local, &attention), 0.25);

    // A context with no segments has nothing left uncovered.
    assert_eq!(segment_coverage(&[], 0, 4), 1.0);
}
