//! Contracts for the same-budget baselines that decide whether the topological
//! selector does any work.
//!
//! The measurement these support is the one a reviewer asks for first: not how
//! much faster a sparse schedule runs, but how much of the true attention mass
//! it keeps relative to random selection and to the best possible selection at
//! the same cost. That number is only meaningful if the two brackets are what
//! they claim, so they are pinned here rather than assumed.
//!
//! The oracle carries a theorem. Recovered mass is additive over key blocks, so
//! choosing the largest per-block scores is optimal by construction — no subset
//! search is involved. That makes "no schedule beats the oracle at the same
//! budget" an assertion rather than an aspiration, and it is the single most
//! valuable test in the file: if it ever fails, the table the oracle ranks by is
//! not measuring the same thing the recovery measures, and every ablation number
//! computed from either is meaningless.

use aether_core::scheduled::{
    block_mass_recovered, dense_causal_block_schedule, oracle_block_schedule,
    random_block_schedule, schedule_budget, topology_block_schedule, BlockSchedule,
    TopologyScheduleConfig,
};

fn fill(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        })
        .collect()
}

struct Case {
    q: Vec<f64>,
    k: Vec<f64>,
    seq: usize,
    head_dim: usize,
    block_size: usize,
}

impl Case {
    fn new(seq: usize, head_dim: usize, block_size: usize, seed: u64) -> Self {
        Self {
            q: fill(seq * head_dim, seed),
            k: fill(seq * head_dim, seed + 1),
            seq,
            head_dim,
            block_size,
        }
    }

    fn recovered(&self, schedule: &BlockSchedule) -> f64 {
        block_mass_recovered(
            schedule,
            &self.q,
            &self.k,
            self.seq,
            self.head_dim,
            self.block_size,
        )
        .expect("valid shapes")
    }

    fn oracle(&self, budget: &[usize]) -> BlockSchedule {
        oracle_block_schedule(
            &self.q,
            &self.k,
            self.seq,
            self.head_dim,
            self.block_size,
            budget,
        )
        .expect("valid budget")
    }
}

/// The dense causal schedule loses nothing, by definition.
///
/// The calibration point for the whole scale. If this is not 1.0 then the mass
/// table does not sum to the softmax it claims to decompose, and every other
/// number here is measured against a broken unit.
#[test]
fn the_dense_schedule_recovers_all_of_the_mass() {
    let case = Case::new(64, 16, 8, 3);
    let dense = dense_causal_block_schedule(case.seq / case.block_size);
    let recovered = case.recovered(&dense);

    assert!(
        (recovered - 1.0).abs() < 1e-12,
        "the dense causal schedule recovered {recovered}, not 1.0; the mass \
         table does not decompose the softmax it is derived from"
    );
}

/// No schedule may beat the oracle at the same per-row budget.
///
/// The theorem this file exists to protect. Checked against many random
/// schedules rather than one, and against the topological schedule, because a
/// single comparison could pass by luck on a case where the oracle and the
/// challenger happen to agree.
///
/// A tolerance is needed only for floating-point summation order: the two
/// schedules add the same table entries in different sequences.
#[test]
fn no_schedule_recovers_more_mass_than_the_oracle() {
    for (seq, head_dim, block_size, seed) in [
        (64, 16, 8, 5),
        (32, 8, 4, 9),
        (96, 12, 8, 13),
        (16, 4, 2, 17),
    ] {
        let case = Case::new(seq, head_dim, block_size, seed);
        let num_blocks = seq / block_size;

        // A budget that leaves real choices: enough blocks to matter, never so
        // many that every candidate is forced and the comparison is vacuous.
        let budget: Vec<usize> = (0..num_blocks).map(|q| 1 + q / 2).collect();
        let oracle = case.oracle(&budget);
        let ceiling = case.recovered(&oracle);

        for trial in 0..24u64 {
            let challenger =
                random_block_schedule(&budget, seed * 100 + trial).expect("valid budget");
            let got = case.recovered(&challenger);
            assert!(
                got <= ceiling + 1e-12,
                "seq={seq} trial={trial}: a random schedule recovered {got}, \
                 above the oracle's {ceiling}, at an identical per-row budget. \
                 The oracle is not selecting by the quantity the recovery \
                 measures."
            );
        }

        let config = TopologyScheduleConfig {
            block_size,
            local_radius_blocks: 1,
            sink_blocks: 1,
            topk_topology_blocks: 1,
        };
        let topological =
            topology_block_schedule(&case.k, seq, head_dim, config).expect("valid config");
        let topo_budget = schedule_budget(&topological);
        let matched = case.oracle(&topo_budget);
        let topo_recovered = case.recovered(&topological);
        let matched_ceiling = case.recovered(&matched);

        assert!(
            topo_recovered <= matched_ceiling + 1e-12,
            "seq={seq}: the topological schedule recovered {topo_recovered}, \
             above the oracle's {matched_ceiling} at its own budget"
        );
    }
}

/// A baseline must spend exactly the budget it was given, per row.
///
/// The comparison is only about selection if the two schedules cost the same.
/// A baseline that quietly spent fewer blocks would flatter the selector it is
/// measured against, and one that spent more would be a different experiment.
#[test]
fn the_baselines_spend_exactly_the_budget_they_are_given() {
    let case = Case::new(64, 16, 8, 21);
    let num_blocks = case.seq / case.block_size;
    let budget: Vec<usize> = (0..num_blocks).map(|q| 1 + q / 3).collect();

    let random = random_block_schedule(&budget, 7).expect("valid budget");
    let oracle = case.oracle(&budget);

    for (q_block, &want) in budget.iter().enumerate() {
        // Early rows have fewer causal candidates than the budget asks for, and
        // both baselines clamp to what exists.
        let available = q_block + 1;
        let expected = want.min(available);

        assert_eq!(
            random.row(q_block).len(),
            expected,
            "random row {q_block} spent {} blocks, not {expected}",
            random.row(q_block).len()
        );
        assert_eq!(
            oracle.row(q_block).len(),
            expected,
            "oracle row {q_block} spent {} blocks, not {expected}",
            oracle.row(q_block).len()
        );
    }
}

/// The random baseline must be reproducible from its seed, and must actually
/// vary between seeds.
///
/// Both halves matter. A baseline that changed between runs could not be
/// compared against anything; a baseline that ignored its seed would report a
/// single arbitrary draw as though it were a distribution, which is how a
/// selector gets credited for beating one unlucky sample.
#[test]
fn the_random_baseline_is_reproducible_and_not_constant() {
    let num_blocks = 8;
    let budget: Vec<usize> = (0..num_blocks).map(|q| 1 + q / 2).collect();

    let a = random_block_schedule(&budget, 42).expect("valid budget");
    let b = random_block_schedule(&budget, 42).expect("valid budget");
    assert_eq!(a, b, "the same seed produced two different schedules");

    let different = (0..16u64)
        .map(|s| random_block_schedule(&budget, 1000 + s).expect("valid budget"))
        .any(|other| other != a);
    assert!(
        different,
        "sixteen seeds all produced the same schedule, so the baseline is not \
         sampling and reports one draw as a distribution"
    );
}

/// Every schedule a baseline produces must satisfy the kernel's invariants.
///
/// `BlockSchedule::from_rows` enforces causality, sortedness, uniqueness and
/// non-emptiness, so a baseline that violated one would fail to construct rather
/// than produce a schedule the kernel mishandles. This asserts the construction
/// is actually routed through that check for a wide range of budgets, including
/// the saturating case where the budget exceeds the causal candidates.
#[test]
fn baselines_produce_schedules_the_kernel_accepts() {
    let case = Case::new(32, 8, 4, 27);
    let num_blocks = case.seq / case.block_size;

    for (label, budget) in [
        ("minimal", vec![1usize; num_blocks]),
        ("saturating", (0..num_blocks).map(|q| q + 1).collect()),
        ("over-budget", vec![num_blocks + 5; num_blocks]),
        ("uneven", (0..num_blocks).map(|q| 1 + (q % 3)).collect()),
    ] {
        for seed in 0..8u64 {
            let random = random_block_schedule(&budget, seed)
                .unwrap_or_else(|e| panic!("{label} budget, seed {seed}: {e:?}"));
            let oracle = case.oracle(&budget);

            for schedule in [&random, &oracle] {
                for q_block in 0..num_blocks {
                    let row = schedule.row(q_block);
                    assert!(!row.is_empty(), "{label}: row {q_block} is empty");
                    assert!(
                        row.windows(2).all(|w| w[0] < w[1]),
                        "{label}: row {q_block} is not strictly sorted: {row:?}"
                    );
                    assert!(
                        row.iter().all(|&b| b <= q_block),
                        "{label}: row {q_block} scheduled a future block: {row:?}"
                    );
                }
            }
        }
    }
}

/// Recovered mass must lie in [0, 1] for any valid schedule.
///
/// It is a fraction of a softmax, so a value outside that range means the table
/// is not normalised — which would be invisible in a comparison between two
/// schedules that were both wrong by the same factor.
#[test]
fn recovered_mass_is_a_fraction() {
    let case = Case::new(48, 12, 4, 33);
    let num_blocks = case.seq / case.block_size;

    for seed in 0..20u64 {
        let budget: Vec<usize> = (0..num_blocks)
            .map(|q| 1 + (q + seed as usize) % 4)
            .collect();
        let schedule = random_block_schedule(&budget, seed).expect("valid budget");
        let recovered = case.recovered(&schedule);

        assert!(
            (0.0..=1.0 + 1e-12).contains(&recovered),
            "seed {seed}: recovered mass {recovered} is not a fraction"
        );
    }
}

/// Adding a block to a row can never reduce recovered mass.
///
/// Monotonicity in the budget. Every block carries non-negative mass, so a
/// superset schedule recovers at least as much. This catches a sign error or a
/// double subtraction in the table that the bounds check above would not: a
/// table with a negative entry can still produce totals inside [0, 1].
#[test]
fn a_larger_budget_never_recovers_less() {
    let case = Case::new(32, 8, 4, 39);
    let num_blocks = case.seq / case.block_size;

    for extra in 1..4usize {
        let small: Vec<usize> = (0..num_blocks).map(|q| 1 + q / 4).collect();
        let large: Vec<usize> = small.iter().map(|&b| b + extra).collect();

        let recovered_small = case.recovered(&case.oracle(&small));
        let recovered_large = case.recovered(&case.oracle(&large));

        assert!(
            recovered_large >= recovered_small - 1e-12,
            "extra={extra}: raising the budget lowered recovered mass, \
             {recovered_small} to {recovered_large}"
        );
    }
}
