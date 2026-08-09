//! Correctness contracts for `aether_core::attention`.
//!
//! Written before the module exists. Ordered by how much bug each catches per
//! line of test, which is why dense parity comes first: it catches transposed
//! strides, a wrong scale factor, an off-by-one in the key index, and softmax
//! over the wrong axis in a single assertion, before any topology is involved.
//!
//! What is deliberately NOT here: a gradient check. There is no backward pass in
//! this module, so there is nothing for `gradcheck` to disagree with. When a
//! backward is added, the gradient check is the first test that must come with
//! it — a forward-correct kernel with a wrong backward trains to a plausible
//! worse optimum, which is nearly undetectable from loss curves.

use aether_core::attention::{
    attention_mass_recovered, dense_attention, dense_dot_cost, routing_plan, select_mask,
    selection_dot_cost, single_linkage_clusters, sparse_attention, Selector,
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
    fn signed(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Row-major `[seq, head_dim]` matrices for q, k, v.
fn qkv(seq: usize, head_dim: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let n = seq * head_dim;
    let q = (0..n).map(|_| rng.signed()).collect();
    let k = (0..n).map(|_| rng.signed()).collect();
    let v = (0..n).map(|_| rng.signed()).collect();
    (q, k, v)
}

fn full_mask(seq: usize) -> Vec<bool> {
    vec![true; seq * seq]
}

fn causal_mask(seq: usize) -> Vec<bool> {
    let mut mask = vec![false; seq * seq];
    for i in 0..seq {
        for j in 0..=i {
            mask[i * seq + j] = true;
        }
    }
    mask
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Dense parity — the highest-value assertion in the file
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn a_full_mask_reproduces_dense_attention_exactly() {
    // If the selector returns every key, the sparse path must equal dense SDPA.
    // This is bitwise, not approximate: both paths do the same arithmetic in the
    // same order when nothing is masked out.
    for (seq, head_dim) in [(1usize, 4usize), (5, 3), (8, 8), (13, 7), (17, 6)] {
        let (q, k, v) = qkv(seq, head_dim, 42);
        let dense = dense_attention(&q, &k, &v, seq, head_dim);
        let sparse = sparse_attention(&q, &k, &v, seq, head_dim, &full_mask(seq));

        assert_eq!(dense.len(), seq * head_dim);
        for (i, (d, s)) in dense.iter().zip(sparse.iter()).enumerate() {
            assert_eq!(
                d, s,
                "seq {seq}, head_dim {head_dim}, element {i}: dense {d} vs sparse {s}"
            );
        }
    }
}

#[test]
fn attention_output_is_a_convex_combination_of_values() {
    // Softmax weights are non-negative and sum to one, so every output coordinate
    // lies within the range of that coordinate across all values. Catches a
    // missing normalisation and a wrong reduction axis.
    let (seq, head_dim) = (11usize, 5usize);
    let (q, k, v) = qkv(seq, head_dim, 7);
    let out = dense_attention(&q, &k, &v, seq, head_dim);

    for d in 0..head_dim {
        let column: Vec<f64> = (0..seq).map(|t| v[t * head_dim + d]).collect();
        let lo = column.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for i in 0..seq {
            let got = out[i * head_dim + d];
            assert!(
                got >= lo - 1e-12 && got <= hi + 1e-12,
                "row {i}, dim {d}: {got} outside the value range [{lo}, {hi}]"
            );
        }
    }
}

#[test]
fn a_uniform_query_averages_the_values() {
    // With all-zero queries every logit is zero, so softmax is uniform and the
    // output is the mean of v. A closed-form check on the scale factor and the
    // normalisation together.
    let (seq, head_dim) = (6usize, 4usize);
    let (_, k, v) = qkv(seq, head_dim, 3);
    let q = vec![0.0; seq * head_dim];

    let out = dense_attention(&q, &k, &v, seq, head_dim);
    for d in 0..head_dim {
        let mean: f64 = (0..seq).map(|t| v[t * head_dim + d]).sum::<f64>() / seq as f64;
        for i in 0..seq {
            assert!(
                (out[i * head_dim + d] - mean).abs() < 1e-12,
                "row {i}, dim {d}: {} != mean {mean}",
                out[i * head_dim + d]
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Mask fidelity
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn a_masked_key_contributes_exactly_nothing() {
    // Attending to approximately the right keys is a different algorithm from the
    // one described. Test it behaviourally: change a masked-out value row and the
    // output must be bitwise identical.
    let (seq, head_dim) = (9usize, 4usize);
    let (q, k, mut v) = qkv(seq, head_dim, 11);

    let mut mask = full_mask(seq);
    let (row, blocked) = (3usize, 6usize);
    mask[row * seq + blocked] = false;

    let before = sparse_attention(&q, &k, &v, seq, head_dim, &mask);

    // Perturb the value row that row 3 is forbidden to see.
    for d in 0..head_dim {
        v[blocked * head_dim + d] += 100.0;
    }
    let after = sparse_attention(&q, &k, &v, seq, head_dim, &mask);

    for d in 0..head_dim {
        assert_eq!(
            before[row * head_dim + d],
            after[row * head_dim + d],
            "row {row} dim {d} moved after changing a masked-out value"
        );
    }
    // And a row that IS allowed to see it must have moved, or the test is vacuous.
    assert!(
        (0..head_dim).any(|d| before[2 * head_dim + d] != after[2 * head_dim + d]),
        "row 2 should see key {blocked}; the perturbation had no effect anywhere"
    );
}

#[test]
fn the_realized_pattern_equals_the_requested_pattern() {
    // Density statistics are not fidelity. Compare element-wise.
    let (seq, head_dim) = (12usize, 4usize);
    let (q, k, v) = qkv(seq, head_dim, 13);

    for selector in [
        Selector::Dense,
        Selector::Local { window: 3 },
        Selector::Random { budget: 4, seed: 9 },
        Selector::OracleTopK { budget: 4 },
        Selector::Topological {
            budget: 4,
            radius_scale: 1.5,
        },
    ] {
        let mask = select_mask(selector, &q, &k, &v, seq, head_dim, false);
        assert_eq!(mask.len(), seq * seq, "{selector:?}: wrong mask length");

        // Every row must select at least one key, or softmax has nothing to
        // normalise. This is the contract that makes the NaN guard reachable
        // only through a hand-built mask, never through a selector.
        for i in 0..seq {
            let selected = (0..seq).filter(|&j| mask[i * seq + j]).count();
            assert!(
                selected > 0,
                "{selector:?}: row {i} selected no keys at all"
            );
        }
    }
}

#[test]
fn a_budgeted_selector_respects_its_budget() {
    let (seq, head_dim) = (16usize, 4usize);
    let (q, k, v) = qkv(seq, head_dim, 17);
    let budget = 5;

    for selector in [
        Selector::Random { budget, seed: 2 },
        Selector::OracleTopK { budget },
        Selector::Topological {
            budget,
            radius_scale: 1.5,
        },
    ] {
        let mask = select_mask(selector, &q, &k, &v, seq, head_dim, false);
        for i in 0..seq {
            let selected = (0..seq).filter(|&j| mask[i * seq + j]).count();
            assert!(
                selected <= budget,
                "{selector:?}: row {i} selected {selected} keys, budget {budget}"
            );
        }
    }
}

#[test]
fn the_topological_selector_is_scale_equivariant() {
    // Scaling q and k by c > 0 scales every query-key distance by c. A selector
    // whose radius is relative to the data must return the IDENTICAL mask; one
    // with an absolute radius silently selects a different set on every rescaled
    // dataset.
    //
    // This test exists because the first version of the selector used an absolute
    // radius. It did not fail any correctness contract — it failed the ablation,
    // by selecting 1.0 keys per row where its same-budget baselines selected 5.5,
    // and so lost on budget rather than on mechanism. Same bug class as the
    // hardcoded epsilon the persistence scale-equivariance test catches.
    let (seq, head_dim) = (20usize, 6usize);
    let (q, k, v) = qkv(seq, head_dim, 101);
    let selector = Selector::Topological {
        budget: 5,
        radius_scale: 1.2,
    };

    let base = select_mask(selector, &q, &k, &v, seq, head_dim, true);

    for factor in [0.01, 0.5, 2.0, 100.0] {
        let qs: Vec<f64> = q.iter().map(|x| x * factor).collect();
        let ks: Vec<f64> = k.iter().map(|x| x * factor).collect();
        let scaled = select_mask(selector, &qs, &ks, &v, seq, head_dim, true);
        assert_eq!(
            base, scaled,
            "factor {factor}: the selected key set changed under rescaling"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Causality
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn no_output_position_depends_on_a_later_position() {
    // Tested behaviourally rather than by inspecting the mask: perturb position j
    // and assert that every output at i < j is bitwise unchanged. This catches
    // leakage that mask inspection misses.
    let (seq, head_dim) = (10usize, 4usize);
    let (q, k, v) = qkv(seq, head_dim, 23);
    let mask = causal_mask(seq);

    let base = sparse_attention(&q, &k, &v, seq, head_dim, &mask);

    for j in 1..seq {
        let mut k2 = k.clone();
        let mut v2 = v.clone();
        for d in 0..head_dim {
            k2[j * head_dim + d] += 7.5;
            v2[j * head_dim + d] -= 3.25;
        }
        let perturbed = sparse_attention(&q, &k2, &v2, seq, head_dim, &mask);

        for i in 0..j {
            for d in 0..head_dim {
                assert_eq!(
                    base[i * head_dim + d],
                    perturbed[i * head_dim + d],
                    "position {j} leaked into output {i} (dim {d})"
                );
            }
        }
        // Position j itself must change, or the perturbation was inert.
        assert!(
            (0..head_dim).any(|d| base[j * head_dim + d] != perturbed[j * head_dim + d]),
            "perturbing position {j} did not change its own output"
        );
    }
}

#[test]
fn causal_selectors_never_select_a_future_key() {
    let (seq, head_dim) = (14usize, 4usize);
    let (q, k, v) = qkv(seq, head_dim, 29);

    for selector in [
        Selector::Dense,
        Selector::Local { window: 4 },
        Selector::Random { budget: 3, seed: 5 },
        Selector::OracleTopK { budget: 3 },
        Selector::Topological {
            budget: 3,
            radius_scale: 1.5,
        },
    ] {
        let mask = select_mask(selector, &q, &k, &v, seq, head_dim, true);
        for i in 0..seq {
            for j in (i + 1)..seq {
                assert!(
                    !mask[i * seq + j],
                    "{selector:?}: row {i} selected future key {j} under causal masking"
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. All-masked row — the input-dependent bug that escapes into production
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn an_all_masked_row_returns_zeros_rather_than_nan() {
    // softmax over all -inf produces NaN. Topological selectors hit this on
    // isolated tokens: a point that is its own connected component selects
    // nothing. The guard must return a defined value.
    let (seq, head_dim) = (7usize, 4usize);
    let (q, k, v) = qkv(seq, head_dim, 31);

    let mut mask = full_mask(seq);
    let empty_row = 4usize;
    for j in 0..seq {
        mask[empty_row * seq + j] = false;
    }

    let out = sparse_attention(&q, &k, &v, seq, head_dim, &mask);

    for d in 0..head_dim {
        let got = out[empty_row * head_dim + d];
        assert!(
            got.is_finite(),
            "row {empty_row} dim {d} is {got}, not finite"
        );
        assert_eq!(got, 0.0, "an all-masked row should read as zero, got {got}");
    }
    // Every other row must still be finite and unaffected by the empty row.
    assert!(
        out.iter().all(|x| x.is_finite()),
        "an all-masked row contaminated the rest of the output"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. Numerical safety
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn large_logits_do_not_overflow_the_softmax() {
    // Without subtracting the row max, exp() of a large logit is +inf and the
    // output is NaN. Scale the inputs until naive exp would certainly overflow.
    let (seq, head_dim) = (6usize, 4usize);
    let (q, k, v) = qkv(seq, head_dim, 37);
    let big_q: Vec<f64> = q.iter().map(|x| x * 400.0).collect();
    let big_k: Vec<f64> = k.iter().map(|x| x * 400.0).collect();

    let out = dense_attention(&big_q, &big_k, &v, seq, head_dim);
    assert!(
        out.iter().all(|x| x.is_finite()),
        "large logits produced non-finite output: {out:?}"
    );

    // With logits this extreme the softmax is effectively one-hot, so each output
    // row must coincide with a single value row.
    for i in 0..seq {
        let matches = (0..seq).any(|t| {
            (0..head_dim).all(|d| (out[i * head_dim + d] - v[t * head_dim + d]).abs() < 1e-6)
        });
        assert!(matches, "row {i} did not saturate onto a single value row");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. Determinism
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn repeated_runs_are_bitwise_identical() {
    // Non-determinism here invalidates every A/B benchmark downstream, because
    // part of the variance being measured is the kernel's own.
    let (seq, head_dim) = (13usize, 5usize);
    let (q, k, v) = qkv(seq, head_dim, 41);
    let mask = select_mask(
        Selector::Topological {
            budget: 4,
            radius_scale: 1.5,
        },
        &q,
        &k,
        &v,
        seq,
        head_dim,
        true,
    );

    let first = sparse_attention(&q, &k, &v, seq, head_dim, &mask);
    for run in 1..5 {
        let again = sparse_attention(&q, &k, &v, seq, head_dim, &mask);
        assert_eq!(first, again, "run {run} differed from run 0");
        let mask_again = select_mask(
            Selector::Topological {
                budget: 4,
                radius_scale: 1.5,
            },
            &q,
            &k,
            &v,
            seq,
            head_dim,
            true,
        );
        assert_eq!(
            mask, mask_again,
            "selector was non-deterministic on run {run}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. Shape edges — the tail tile is where index arithmetic fails
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn shapes_around_the_block_boundary_are_handled() {
    const BLOCK: usize = 8;
    for seq in [1usize, BLOCK - 1, BLOCK, BLOCK + 1, 2 * BLOCK + 1] {
        // head_dim deliberately includes a non-power-of-two.
        for head_dim in [1usize, 3, 8, 6] {
            let (q, k, v) = qkv(seq, head_dim, 43);
            let out = dense_attention(&q, &k, &v, seq, head_dim);
            assert_eq!(
                out.len(),
                seq * head_dim,
                "seq {seq}, head_dim {head_dim}: wrong output length"
            );
            assert!(
                out.iter().all(|x| x.is_finite()),
                "seq {seq}, head_dim {head_dim}: non-finite output"
            );

            let sparse = sparse_attention(&q, &k, &v, seq, head_dim, &causal_mask(seq));
            assert_eq!(sparse.len(), seq * head_dim);
            assert!(sparse.iter().all(|x| x.is_finite()));
        }
    }
}

#[test]
fn a_single_position_attends_to_itself() {
    let (q, k, v) = qkv(1, 4, 47);
    let out = dense_attention(&q, &k, &v, 1, 4);
    for d in 0..4 {
        assert!(
            (out[d] - v[d]).abs() < 1e-12,
            "seq 1 must return v exactly; dim {d} gave {} vs {}",
            out[d],
            v[d]
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 8. The ablation that decides whether topology did the work
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn oracle_top_k_upper_bounds_every_other_selector_at_the_same_budget() {
    // Oracle top-k picks the keys with the genuinely largest attention weights,
    // computed densely. It is unimplementable in production but decisive as a
    // diagnostic: no same-budget selector can recover more attention mass.
    //
    // If this fails, the metric is wrong, not the selector.
    let (seq, head_dim) = (24usize, 6usize);
    let budget = 5;

    for seed in [53u64, 59, 61] {
        let (q, k, v) = qkv(seq, head_dim, seed);
        let oracle = attention_mass_recovered(
            Selector::OracleTopK { budget },
            &q,
            &k,
            &v,
            seq,
            head_dim,
            true,
        );

        for selector in [
            Selector::Random { budget, seed: 3 },
            Selector::Local { window: budget },
            Selector::Topological {
                budget,
                radius_scale: 1.5,
            },
        ] {
            let got = attention_mass_recovered(selector, &q, &k, &v, seq, head_dim, true);
            assert!(
                got <= oracle + 1e-12,
                "seed {seed}: {selector:?} recovered {got:.4} > oracle {oracle:.4}"
            );
        }
    }
}

#[test]
fn the_topological_selector_is_placed_on_the_random_to_oracle_axis() {
    // The actual scientific question, stated as a test rather than a claim.
    //
    //   topological ~= random  -> the mechanism contributes nothing; any gain is
    //                             sparsity regularisation
    //   topological ~= oracle  -> the topological summary is a good proxy for
    //                             attention mass
    //
    // This test does not assert which. It asserts the comparison is FAIR — every
    // selector spends the same mean budget, so a selector cannot lose by simply
    // declining to select — and that the axis is well-formed, then prints where
    // the topological selector lands so the number is reported, not assumed.
    let (seq, head_dim) = (32usize, 8usize);
    let budget = 6;
    let mut placements = Vec::new();

    for seed in [67u64, 71, 73, 79] {
        let (q, k, v) = qkv(seq, head_dim, seed);

        let mean_keys = |s: Selector| {
            let mask = select_mask(s, &q, &k, &v, seq, head_dim, true);
            (0..seq)
                .map(|i| (0..seq).filter(|&j| mask[i * seq + j]).count())
                .sum::<usize>() as f64
                / seq as f64
        };
        let mass = |s: Selector| attention_mass_recovered(s, &q, &k, &v, seq, head_dim, true);

        let oracle_sel = Selector::OracleTopK { budget };
        let random_sel = Selector::Random { budget, seed: 1 };
        let topo_sel = Selector::Topological {
            budget,
            radius_scale: f64::INFINITY,
        };

        // Fairness precondition. Without it, "topological loses" can mean
        // "topological selected fewer keys", which is a measurement artifact.
        let reference = mean_keys(oracle_sel);
        for (name, selector) in [
            ("random", random_sel),
            ("topological", topo_sel),
            ("local", Selector::Local { window: budget }),
        ] {
            let keys = mean_keys(selector);
            assert!(
                (keys - reference).abs() < 1e-9,
                "seed {seed}: {name} spends {keys:.2} keys/row against the oracle's                  {reference:.2}; the budgets differ so the mass comparison is unfair"
            );
        }

        let (oracle, random, topological) = (mass(oracle_sel), mass(random_sel), mass(topo_sel));

        assert!(
            oracle > random + 1e-9,
            "seed {seed}: oracle {oracle:.4} does not beat random {random:.4};              the axis is degenerate and no placement is meaningful"
        );

        let placement = (topological - random) / (oracle - random);
        println!(
            "seed {seed}: random {random:.4}  topological {topological:.4}               oracle {oracle:.4}  placement {placement:+.3}"
        );
        placements.push(placement);
    }

    // A placement outside [-1, 2] means the metric or the selector is broken, not
    // that the mechanism is unusually good or bad.
    for (i, p) in placements.iter().enumerate() {
        assert!(
            (-1.0..=2.0).contains(p),
            "seed index {i}: placement {p} is outside any plausible range"
        );
    }
}

#[test]
fn the_topological_advantage_collapses_when_key_norms_vary() {
    // The finding that keeps the placement number honest.
    //
    // ||q - k||^2 = ||q||^2 + ||k||^2 - 2 q.k. When the key norms are all about
    // equal, ranking keys by Euclidean proximity IS ranking them by dot product,
    // so the topological selector cannot lose and its ~0.9 placement on uniform
    // random data is close to tautological rather than evidence.
    //
    // Vary the key norms and the two rankings decouple. Measured over 8 trials at
    // seq 32, head_dim 8, budget 6:
    //
    //   spread 0.0 -> placement +0.884
    //   spread 1.0 -> placement +0.533
    //   spread 2.0 -> placement +0.202
    //   spread 4.0 -> placement -0.109
    //   spread 8.0 -> placement -0.285
    //
    // At high spread the selector is WORSE than uniform random. Any claim that
    // this mechanism proxies attention mass is therefore conditional on roughly
    // homogeneous key norms, and must say so.
    let (seq, head_dim, budget) = (32usize, 8usize, 6usize);

    let placement_at = |spread: f64| {
        let mut total = 0.0;
        let trials = 8;
        for trial in 0..trials {
            let mut rng = Rng::new(trial * 2 + 67);
            let n = seq * head_dim;
            let q: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
            let v: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
            let mut k = vec![0.0; n];
            for t in 0..seq {
                let gain = 1.0 + spread * (rng.signed() * 0.5 + 0.5);
                for d in 0..head_dim {
                    k[t * head_dim + d] = rng.signed() * gain;
                }
            }
            let mass =
                |sel: Selector| attention_mass_recovered(sel, &q, &k, &v, seq, head_dim, true);
            let random = mass(Selector::Random { budget, seed: 1 });
            let oracle = mass(Selector::OracleTopK { budget });
            let topological = mass(Selector::Topological {
                budget,
                radius_scale: f64::INFINITY,
            });
            total += (topological - random) / (oracle - random);
        }
        total / trials as f64
    };

    let flat = placement_at(0.0);
    let spread = placement_at(8.0);

    assert!(
        flat > 0.7,
        "constant-norm keys should give a high placement (the near-tautological          regime); got {flat:.3}"
    );
    assert!(
        spread < flat - 0.5,
        "placement must degrade substantially as key norms spread: flat {flat:.3},          spread-8 {spread:.3}. If this stops holding, either the selector changed          or the metric is no longer measuring what it claims."
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9. The routed selector — the fix the ablation demanded
//
// The nearest-neighbour selector fails at high key-norm spread because Euclidean
// proximity stops tracking the dot product once norms vary. The routed selector
// separates the two jobs the old one conflated:
//
//   1. TOPOLOGY builds a candidate set cheaply — single-linkage (H0) clustering
//      of the DIRECTIONS of the keys, which is norm-invariant by construction.
//   2. The exact dot product ranks WITHIN that candidate set, which restores the
//      norm sensitivity the geometry threw away.
//
// The cost model that makes this a real sparse method: `clusters` centroid dots
// to route, plus |candidates| dots to rank, instead of `seq`.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn single_linkage_merge_heights_equal_the_h0_persistence_deaths() {
    // The clustering the router uses IS H0 persistence. Assert it against the
    // engine that milestones 1 and 2 tested, rather than trusting a second
    // implementation of the same mathematics.
    use aether_core::manifold::ManifoldPoint;
    use aether_core::persistence::{persistent_homology, PersistenceConfig};

    let mut rng = Rng::new(2027);
    let n = 24usize;
    let raw: Vec<[f64; 2]> = (0..n).map(|_| [rng.signed(), rng.signed()]).collect();
    let flat: Vec<f64> = raw.iter().flat_map(|p| p.iter().copied()).collect();

    // Merge heights the router derives, ascending. `normalize = false` so this
    // compares like with like against the raw point cloud.
    let heights = single_linkage_clusters(&flat, n, 2, 1, false).1;

    // H0 finite deaths from the persistence engine, ascending.
    let points: Vec<ManifoldPoint<2>> = raw.iter().map(|&c| ManifoldPoint::new(c)).collect();
    let diagram = persistent_homology(&points, PersistenceConfig::h0_only()).unwrap();
    let mut deaths: Vec<f64> = diagram
        .pairs
        .iter()
        .filter(|p| p.dimension == 0)
        .filter_map(|p| p.death)
        .collect();
    deaths.sort_by(f64::total_cmp);

    assert_eq!(
        heights.len(),
        deaths.len(),
        "{} merge heights vs {} H0 deaths",
        heights.len(),
        deaths.len()
    );
    for (i, (h, d)) in heights.iter().zip(deaths.iter()).enumerate() {
        assert!(
            (h - d).abs() < 1e-12,
            "merge {i}: single-linkage {h} vs H0 death {d}"
        );
    }
}

#[test]
fn routing_clusters_are_invariant_to_per_key_rescaling() {
    // The whole point of clustering directions: multiplying an individual key by
    // a positive scalar must not move it to a different cluster. This is exactly
    // what the nearest-neighbour selector could not do.
    let (seq, head_dim) = (24usize, 6usize);
    let (_, k, _) = qkv(seq, head_dim, 2029);

    let base = single_linkage_clusters(&k, seq, head_dim, 4, true).0;

    let mut rng = Rng::new(9091);
    let mut rescaled = k.clone();
    for t in 0..seq {
        let gain = 1.0 + 6.0 * (rng.signed() * 0.5 + 0.5); // in [1, 7]
        for d in 0..head_dim {
            rescaled[t * head_dim + d] *= gain;
        }
    }
    let after = single_linkage_clusters(&rescaled, seq, head_dim, 4, true).0;

    assert_eq!(
        base, after,
        "per-key rescaling changed the cluster assignment; the router is not norm-invariant and will inherit the failure it was built to fix"
    );
}

#[test]
fn the_routed_selector_obeys_every_contract_the_others_do() {
    let (seq, head_dim) = (20usize, 6usize);
    let (q, k, v) = qkv(seq, head_dim, 2031);
    let selector = Selector::TopologicalRouted {
        budget: 5,
        clusters: 4,
    };

    // Budget, non-empty rows, and causality.
    let mask = select_mask(selector, &q, &k, &v, seq, head_dim, true);
    for i in 0..seq {
        let selected: Vec<usize> = (0..seq).filter(|&j| mask[i * seq + j]).collect();
        assert!(!selected.is_empty(), "row {i} selected nothing");
        assert!(selected.len() <= 5, "row {i} selected {}", selected.len());
        assert!(
            selected.iter().all(|&j| j <= i),
            "row {i} selected a future key"
        );
    }

    // Determinism, selector and kernel.
    for run in 1..4 {
        assert_eq!(
            mask,
            select_mask(selector, &q, &k, &v, seq, head_dim, true),
            "selector non-deterministic on run {run}"
        );
    }
    let out = sparse_attention(&q, &k, &v, seq, head_dim, &mask);
    assert!(out.iter().all(|x| x.is_finite()));

    // Global scale equivariance.
    for factor in [0.02, 3.0, 50.0] {
        let qs: Vec<f64> = q.iter().map(|x| x * factor).collect();
        let ks: Vec<f64> = k.iter().map(|x| x * factor).collect();
        assert_eq!(
            mask,
            select_mask(selector, &qs, &ks, &v, seq, head_dim, true),
            "factor {factor}: routed mask changed under global rescaling"
        );
    }
}

#[test]
fn the_routed_selector_survives_the_key_norm_spread_that_broke_the_old_one() {
    // THE claim, stated so it can fail.
    //
    // The nearest-neighbour selector measured -0.109 at spread 4 and -0.285 at
    // spread 8: worse than picking keys at random. If routing through a
    // norm-invariant topological clustering and then ranking by exact dot product
    // is the right fix, the routed selector must stay clearly positive there.
    let (seq, head_dim, budget) = (32usize, 8usize, 6usize);

    let placements = |spread: f64| {
        let mut nearest_total = 0.0;
        let mut routed_total = 0.0;
        let trials = 8;
        for trial in 0..trials {
            let mut rng = Rng::new(trial * 2 + 67);
            let n = seq * head_dim;
            let q: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
            let v: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
            let mut k = vec![0.0; n];
            for t in 0..seq {
                let gain = 1.0 + spread * (rng.signed() * 0.5 + 0.5);
                for d in 0..head_dim {
                    k[t * head_dim + d] = rng.signed() * gain;
                }
            }
            let mass =
                |sel: Selector| attention_mass_recovered(sel, &q, &k, &v, seq, head_dim, true);
            let random = mass(Selector::Random { budget, seed: 1 });
            let oracle = mass(Selector::OracleTopK { budget });
            let span = oracle - random;

            nearest_total += (mass(Selector::Topological {
                budget,
                radius_scale: f64::INFINITY,
            }) - random)
                / span;
            routed_total += (mass(Selector::TopologicalRouted {
                budget,
                clusters: 6,
            }) - random)
                / span;
        }
        (nearest_total / trials as f64, routed_total / trials as f64)
    };

    for spread in [0.0f64, 2.0, 4.0, 8.0] {
        let (nearest, routed) = placements(spread);
        println!("spread {spread:>4.1}: nearest {nearest:+.3}   routed {routed:+.3}");

        assert!(
            routed > 0.3,
            "spread {spread}: routed placement {routed:+.3} is not clearly better than random. The candidate-set fix does not work as specified."
        );
        if spread >= 4.0 {
            assert!(
                routed > nearest + 0.3,
                "spread {spread}: routed {routed:+.3} must decisively beat the nearest-neighbour selector {nearest:+.3}, which is the whole reason this selector exists"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 10. Cost — the contract that was missing
//
// Every test above measures how GOOD a selector's choice is. None measured what
// the choice COST. That gap let the routed selector post a placement of +0.87 on
// uniform keys while examining 0.96-1.28x the dense dot-product count: dense
// attention with clustering overhead, reported as a win.
//
// A sparsity claim needs both numbers or it is not a claim.
// ═══════════════════════════════════════════════════════════════════════════════

/// `groups` tight direction-clusters on the sphere, with per-key norm spread.
fn clustered_keys(
    seq: usize,
    head_dim: usize,
    groups: usize,
    spread: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    let centers: Vec<Vec<f64>> = (0..groups)
        .map(|_| (0..head_dim).map(|_| rng.signed()).collect())
        .collect();
    let mut k = vec![0.0; seq * head_dim];
    for t in 0..seq {
        let center = &centers[t % groups];
        let gain = 1.0 + spread * (rng.signed() * 0.5 + 0.5);
        for d in 0..head_dim {
            k[t * head_dim + d] = (center[d] + 0.1 * rng.signed()) * gain;
        }
    }
    k
}

#[test]
fn routing_is_sparse_only_when_the_keys_have_h0_structure() {
    // The conditional result, stated so both halves can fail.
    //
    // Uniform keys have no density gaps, so H0 single-linkage chains: 64 keys give
    // components of size [61, 1, 1, 1]. Routing to the largest cluster is routing
    // to everything. That is H0 correctly reporting "no structure", not a
    // clustering bug — and it means the method cannot save work on such data.
    //
    // Give the keys real cluster structure and H0 recovers it balanced
    // ([16, 16, 16, 16]), and the router examines under half the keys.
    let (seq, head_dim, budget, clusters) = (64usize, 8usize, 8usize, 4usize);
    let dense = dense_dot_cost(seq, true);
    let selector = Selector::TopologicalRouted { budget, clusters };

    let mut rng = Rng::new(4001);
    let uniform: Vec<f64> = (0..seq * head_dim).map(|_| rng.signed()).collect();
    let uniform_cost = selection_dot_cost(selector, &uniform, seq, head_dim, true) / dense;
    assert!(
        uniform_cost > 0.85,
        "uniform keys have no H0 structure, so routing cannot be sparse; measured \
         {uniform_cost:.3} of dense. If this drops, the clustering found structure \
         that is not there."
    );

    let mut rng = Rng::new(4001);
    let structured = clustered_keys(seq, head_dim, clusters, 0.0, &mut rng);
    let structured_cost = selection_dot_cost(selector, &structured, seq, head_dim, true) / dense;
    assert!(
        structured_cost < 0.6,
        "with {clusters} genuine key clusters the router should examine well under \
         60% of the keys; measured {structured_cost:.3} of dense"
    );
}

#[test]
fn routing_buys_its_quality_at_a_real_discount_on_structured_keys() {
    // Both numbers together. Under half the dense dot products, and still close to
    // the oracle — including at the key-norm spread that sent the nearest-neighbour
    // selector negative.
    let (seq, head_dim, budget, clusters) = (64usize, 8usize, 8usize, 4usize);
    let dense = dense_dot_cost(seq, true);
    let selector = Selector::TopologicalRouted { budget, clusters };

    for spread in [0.0f64, 8.0] {
        let mut cost_total = 0.0;
        let mut place_total = 0.0;
        let trials = 6;

        for trial in 0..trials {
            let mut rng = Rng::new(trial * 2 + 67);
            let n = seq * head_dim;
            let q: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
            let v: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
            let k = clustered_keys(seq, head_dim, clusters, spread, &mut rng);

            cost_total += selection_dot_cost(selector, &k, seq, head_dim, true) / dense;

            let mass =
                |sel: Selector| attention_mass_recovered(sel, &q, &k, &v, seq, head_dim, true);
            let random = mass(Selector::Random { budget, seed: 1 });
            let oracle = mass(Selector::OracleTopK { budget });
            place_total += (mass(selector) - random) / (oracle - random);
        }

        let (cost, placement) = (cost_total / trials as f64, place_total / trials as f64);
        println!("spread {spread:>4.1}: cost {cost:.3} of dense   placement {placement:+.3}");

        assert!(
            cost < 0.6,
            "spread {spread}: cost {cost:.3} of dense is not a sparsity win"
        );
        assert!(
            placement > 0.8,
            "spread {spread}: placement {placement:+.3} does not justify the routing"
        );
    }
}

#[test]
fn the_oracle_is_priced_as_the_diagnostic_it_is() {
    // OracleTopK needs every score to pick its keys, so it costs exactly dense.
    // Asserting that here stops it ever being quoted as an implementable method.
    let (seq, head_dim) = (32usize, 8usize);
    let mut rng = Rng::new(4003);
    let k: Vec<f64> = (0..seq * head_dim).map(|_| rng.signed()).collect();

    let dense = dense_dot_cost(seq, true);
    assert_eq!(
        selection_dot_cost(Selector::OracleTopK { budget: 4 }, &k, seq, head_dim, true),
        dense,
        "oracle top-k must be priced at the full dense cost"
    );
    assert_eq!(
        selection_dot_cost(
            Selector::Random { budget: 4, seed: 1 },
            &k,
            seq,
            head_dim,
            true
        ),
        0.0,
        "random selection inspects no scores"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 11. Deciding whether to route, before paying for it
//
// Milestone 4 left a conditional result: routing is a real sparsity win exactly
// when the keys have H0 structure, and no win at all when they do not. A
// condition nobody checks at runtime is an assumption, and this one silently
// degrades to dense-with-overhead when it fails.
//
// `routing_plan` performs the H0 clustering once per key tensor — amortised over
// every query, head and layer that reuses it — and reports what routing will
// actually cost. `Selector::Adaptive` acts on that report.
// ═══════════════════════════════════════════════════════════════════════════════

/// Keys drawn from `groups` direction-clusters. `groups == seq` degenerates to
/// unstructured, since every key gets its own centre.
fn structured_keys(seq: usize, head_dim: usize, groups: usize, rng: &mut Rng) -> Vec<f64> {
    let centers: Vec<Vec<f64>> = (0..groups)
        .map(|_| (0..head_dim).map(|_| rng.signed()).collect())
        .collect();
    let mut k = vec![0.0; seq * head_dim];
    for t in 0..seq {
        let center = &centers[t % groups];
        for d in 0..head_dim {
            k[t * head_dim + d] = center[d] + 0.1 * rng.signed();
        }
    }
    k
}

#[test]
fn the_plan_predicts_the_cost_it_will_actually_incur() {
    // The plan is computed from the clustering; the cost is measured from the
    // selector. They must agree exactly, or the decision is made on a number that
    // has nothing to do with what happens.
    let (seq, head_dim, budget, clusters) = (48usize, 8usize, 6usize, 6usize);
    let dense = dense_dot_cost(seq, true);

    for groups in [1usize, 2, 4, 6, 12, 48] {
        let mut rng = Rng::new(5000 + groups as u64);
        let k = structured_keys(seq, head_dim, groups, &mut rng);

        let plan = routing_plan(&k, seq, head_dim, clusters, budget, true);
        let measured = selection_dot_cost(
            Selector::TopologicalRouted { budget, clusters },
            &k,
            seq,
            head_dim,
            true,
        ) / dense;

        assert!(
            (plan.cost_ratio - measured).abs() < 1e-12,
            "groups {groups}: plan predicted {:.4} of dense, selector cost {measured:.4}",
            plan.cost_ratio
        );
    }
}

#[test]
fn the_plan_declines_to_route_exactly_when_routing_would_not_pay() {
    // The classifier, with no hidden slack: `worth_routing` must agree with the
    // measured cost ratio against the same threshold, on both structured and
    // unstructured keys.
    let (seq, head_dim, budget, clusters) = (48usize, 8usize, 6usize, 6usize);
    let dense = dense_dot_cost(seq, true);

    let mut structured_seen = false;
    let mut unstructured_seen = false;

    for groups in [1usize, 2, 3, 4, 6, 8, 16, 24, 48] {
        for trial in 0..3u64 {
            let mut rng = Rng::new(6000 + groups as u64 * 17 + trial);
            let k = structured_keys(seq, head_dim, groups, &mut rng);

            let plan = routing_plan(&k, seq, head_dim, clusters, budget, true);
            let measured = selection_dot_cost(
                Selector::TopologicalRouted { budget, clusters },
                &k,
                seq,
                head_dim,
                true,
            ) / dense;

            assert_eq!(
                plan.worth_routing,
                measured < plan.threshold,
                "groups {groups} trial {trial}: plan says worth_routing = {}, but \
                 measured cost {measured:.4} against threshold {:.2}",
                plan.worth_routing,
                plan.threshold
            );

            if plan.worth_routing {
                structured_seen = true;
            } else {
                unstructured_seen = true;
            }
        }
    }

    // A classifier that always answers the same way is not a classifier.
    assert!(
        structured_seen && unstructured_seen,
        "the plan never changed its mind across the whole distribution family; \
         structured_seen = {structured_seen}, unstructured_seen = {unstructured_seen}"
    );
}

#[test]
fn the_h0_barcode_alone_separates_the_two_regimes() {
    // The cheaper signal. `largest_cluster_share` comes from the clustering, but
    // `gap_ratio` comes from the barcode alone — the ratio of the first merge
    // height above the cut to the last one below it. A clear separation between
    // clusters shows up as a large ratio; chaining shows up as a ratio near 1.
    //
    // This is the number that would let a runtime skip the clustering entirely on
    // a cached barcode.
    let (seq, head_dim, budget, clusters) = (48usize, 8usize, 6usize, 6usize);

    let mut structured_gaps = Vec::new();
    let mut chained_gaps = Vec::new();

    for trial in 0..6u64 {
        let mut rng = Rng::new(7000 + trial);
        let structured = structured_keys(seq, head_dim, clusters, &mut rng);
        structured_gaps
            .push(routing_plan(&structured, seq, head_dim, clusters, budget, true).gap_ratio);

        let mut rng = Rng::new(7100 + trial);
        let uniform: Vec<f64> = (0..seq * head_dim).map(|_| rng.signed()).collect();
        chained_gaps.push(routing_plan(&uniform, seq, head_dim, clusters, budget, true).gap_ratio);
    }

    let worst_structured = structured_gaps
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let worst_chained = chained_gaps.iter().copied().fold(0.0f64, f64::max);

    println!("gap ratio: structured min {worst_structured:.2}, chained max {worst_chained:.2}");
    assert!(
        worst_structured > worst_chained,
        "the H0 gap ratio does not separate the regimes: structured bottoms out at \
         {worst_structured:.2} while chained reaches {worst_chained:.2}. The barcode \
         is not a usable predictor and only the full clustering can decide."
    );
}

#[test]
fn the_adaptive_selector_never_costs_more_than_dense() {
    // The guarantee the plan buys. Routing that does not pay must fall back, so
    // the adaptive selector is never worse than the dense path it replaces —
    // which is the property that makes it safe to switch on by default.
    let (seq, head_dim, budget, clusters) = (48usize, 8usize, 6usize, 6usize);
    let dense = dense_dot_cost(seq, true);
    let adaptive = Selector::Adaptive { budget, clusters };

    for groups in [1usize, 2, 4, 6, 12, 24, 48] {
        for trial in 0..3u64 {
            let mut rng = Rng::new(8000 + groups as u64 * 13 + trial);
            let k = structured_keys(seq, head_dim, groups, &mut rng);

            let cost = selection_dot_cost(adaptive, &k, seq, head_dim, true) / dense;
            assert!(
                cost <= 1.0 + 1e-12,
                "groups {groups} trial {trial}: adaptive cost {cost:.4} of dense — \
                 the fallback did not fire"
            );

            // And it must still produce a legal mask.
            let q: Vec<f64> = (0..seq * head_dim).map(|_| rng.signed()).collect();
            let v: Vec<f64> = (0..seq * head_dim).map(|_| rng.signed()).collect();
            let mask = select_mask(adaptive, &q, &k, &v, seq, head_dim, true);
            for i in 0..seq {
                let selected: Vec<usize> = (0..seq).filter(|&j| mask[i * seq + j]).collect();
                assert!(
                    !selected.is_empty(),
                    "groups {groups}: row {i} selected nothing"
                );
                assert!(
                    selected.iter().all(|&j| j <= i),
                    "groups {groups}: row {i} selected a future key"
                );
            }
        }
    }
}

#[test]
fn the_adaptive_selector_keeps_the_quality_of_whichever_path_it_picks() {
    // Falling back must not cost quality.
    //
    // The first version of this test expected a cheap fallback to stay good, and
    // it failed: a budget-6 sliding window on unstructured keys placed at +0.014,
    // indistinguishable from random. That is not a bug in the fallback. When the
    // keys have no H0 structure there is no cheap-and-good option at all, because
    // finding the top-k without computing the scores is precisely what the
    // structure was supposed to make possible.
    //
    // So `Adaptive` falls back to dense, and the guarantee it offers is "never
    // worse than dense, in cost or in quality" — the only guarantee safe enough to
    // switch on by default.
    //
    // The two regimes are scored on DIFFERENT scales on purpose. Placement is
    // only meaningful for budget-limited selectors: dense recovers all the mass,
    // which sits above the budget-limited oracle and makes the ratio blow up
    // (+7.6 in one run). Reporting that as a placement would read as a 700% win
    // over an oracle it never competed with.
    let (seq, head_dim, budget, clusters) = (48usize, 8usize, 6usize, 6usize);
    let adaptive = Selector::Adaptive { budget, clusters };
    let trials = 4;

    // Structured keys: routing fires, so placement on the random-to-oracle axis
    // is the right scale.
    let mut placement_total = 0.0;
    for trial in 0..trials {
        let mut rng = Rng::new(9000 + trial);
        let k = structured_keys(seq, head_dim, clusters, &mut rng);
        let q: Vec<f64> = (0..seq * head_dim).map(|_| rng.signed()).collect();
        let v: Vec<f64> = (0..seq * head_dim).map(|_| rng.signed()).collect();

        let mass = |sel: Selector| attention_mass_recovered(sel, &q, &k, &v, seq, head_dim, true);
        let random = mass(Selector::Random { budget, seed: 1 });
        let oracle = mass(Selector::OracleTopK { budget });
        placement_total += (mass(adaptive) - random) / (oracle - random);

        assert!(
            routing_plan(&k, seq, head_dim, clusters, budget, true).worth_routing,
            "structured trial {trial}: the plan declined to route, so this is not              measuring the routed path"
        );
    }
    let placement = placement_total / trials as f64;
    println!("structured: routed, placement {placement:+.3}");
    assert!(
        placement > 0.8,
        "structured: adaptive placement {placement:+.3} does not justify routing"
    );

    // Unstructured keys: routing is declined, so the only meaningful claim is that
    // the fallback loses nothing against dense. Measured as recovered mass, which
    // must be exactly 1.
    for trial in 0..trials {
        let mut rng = Rng::new(9100 + trial);
        let k: Vec<f64> = (0..seq * head_dim).map(|_| rng.signed()).collect();
        let q: Vec<f64> = (0..seq * head_dim).map(|_| rng.signed()).collect();
        let v: Vec<f64> = (0..seq * head_dim).map(|_| rng.signed()).collect();

        assert!(
            !routing_plan(&k, seq, head_dim, clusters, budget, true).worth_routing,
            "unstructured trial {trial}: the plan chose to route on keys with no              H0 structure"
        );

        let recovered = attention_mass_recovered(adaptive, &q, &k, &v, seq, head_dim, true);
        assert!(
            (recovered - 1.0).abs() < 1e-9,
            "unstructured trial {trial}: the dense fallback recovered {recovered:.6}              of the attention mass, not all of it"
        );
    }
    println!("unstructured: declined to route, dense fallback recovers 1.000 of mass");
}
