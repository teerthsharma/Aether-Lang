//! Does recovered attention mass predict what a trained model can do?
//!
//! Run:
//!   cargo run -p aether-gpu --example recall_training --release
//!
//! `selector_ablation` measures how much true attention mass each schedule keeps
//! and finds the topological ranking below random. That is a statement about the
//! schedule against the attention it approximates, and it carries an assumption
//! nothing has checked: that losing mass costs a model something. A selector
//! could in principle drop most of the mass and still preserve whatever the task
//! needs, in which case the mass measurement would be technically true and
//! practically irrelevant.
//!
//! This closes that gap by training a model on each schedule's output and
//! comparing accuracy against recovered mass.
//!
//! # The task
//!
//! Associative recall. Each sample is a sequence whose final query is planted to
//! match the key at one earlier position, so attention at that final row should
//! retrieve the value stored there. The label is the sign of a component of that
//! value.
//!
//! The task is chosen so that the answer is *only* available through retrieval.
//! The planted position is content-addressed rather than fixed, so the label
//! cannot be read off position; the payload lives nowhere else in the sequence,
//! so a schedule that misses the block holding it leaves the head with features
//! that carry no information about the label, and accuracy falls to chance. That
//! floor is the control — without it a model could score well on a schedule that
//! retrieved nothing, and the comparison would measure the head rather than the
//! attention.
//!
//! # What is trained, and what is not
//!
//! The head is a small MLP trained on GPU with resident tensors and Adam. There
//! is no backward pass through attention: the schedule is fixed and its output is
//! a frozen feature. That is deliberate. Training through attention would let the
//! model compensate for a bad schedule by reshaping the queries, which is a
//! different and much larger experiment; freezing it isolates the question asked
//! here, which is how much task-relevant signal each schedule's output retains.

use aether_core::scheduled::{
    block_mass_recovered, dense_causal_block_schedule, inverted_topology_block_schedule,
    random_block_schedule, schedule_budget, topology_block_schedule, BlockSchedule,
    TopologyScheduleConfig,
};
use aether_gpu::{GpuContext, GpuTensor};

const SEQ: usize = 128;
const HEAD_DIM: usize = 16;
const BLOCK: usize = 8;
const SAMPLES: usize = 2400;
const HIDDEN: usize = 24;
const EPOCHS: usize = 400;
const FOLDS: usize = 5;
/// How strongly the final query is planted onto the target key. See `sample`.
const MATCH: f64 = 30.0;
/// Dense attention must clear this for any comparison below it to mean anything.
///
/// Dense sees every block, so it is an upper bound on what retrieval can offer
/// on this task. If it scores near chance the task is not learnable as
/// configured, and the differences between the sparse schedules are differences
/// between equally uninformative feature sets. Reporting them anyway would be
/// reporting noise with a table around it.
const DENSE_FLOOR: f32 = 0.80;

struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as usize % bound
    }
}

struct Sample {
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    label: f32,
}

/// One associative-recall sequence.
///
/// The final query is set to the key at `target`, plus a little noise so the
/// match is strong rather than exact — an exact copy would make the retrieval
/// degenerate in a way a real sequence never is.
fn sample(rng: &mut Lcg) -> Sample {
    let mut q: Vec<f64> = (0..SEQ * HEAD_DIM).map(|_| rng.next_f64()).collect();
    let k: Vec<f64> = (0..SEQ * HEAD_DIM).map(|_| rng.next_f64()).collect();
    let v: Vec<f64> = (0..SEQ * HEAD_DIM).map(|_| rng.next_f64()).collect();

    // Anywhere before the final block, so retrieval is never satisfied by the
    // local window alone. A target inside the last block would be found by every
    // schedule and the comparison would be vacuous.
    let target = rng.next_usize(SEQ - BLOCK);

    let last = SEQ - 1;
    for d in 0..HEAD_DIM {
        // The multiplier is load-bearing and was calibrated, not guessed. The
        // planted logit is `MATCH * |k|^2 / sqrt(head_dim)` while each of the
        // SEQ-1 distractors sits near `MATCH * |k|^2 / (4 sqrt(head_dim))`, so
        // the target's share of the softmax is set by the gap between those
        // against a count of 127. At MATCH = 6 the target draws about 5% of the
        // mass: the retrieval is real but drowned, dense attention learns almost
        // nothing, and every schedule scores near chance for the same reason.
        //
        // That is a broken ceiling rather than a result, which is what the dense
        // control below exists to catch.
        q[last * HEAD_DIM + d] = k[target * HEAD_DIM + d] * MATCH + rng.next_f64() * 0.2;
    }

    let label = if v[target * HEAD_DIM] > 0.0 { 1.0 } else { 0.0 };
    Sample { q, k, v, label }
}

fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

/// Attention output at the final row: the features the head sees.
fn features(ctx: &GpuContext, s: &Sample, schedule: &BlockSchedule) -> Vec<f32> {
    let out = ctx
        .scheduled_attention(
            &to_f32(&s.q),
            &to_f32(&s.k),
            &to_f32(&s.v),
            SEQ,
            HEAD_DIM,
            schedule,
            BLOCK,
        )
        .expect("dispatch");
    out[(SEQ - 1) * HEAD_DIM..].to_vec()
}

struct Mlp {
    w1: GpuTensor,
    b1: GpuTensor,
    w2: GpuTensor,
    b2: GpuTensor,
}

fn init(ctx: &GpuContext, rng: &mut Lcg) -> Mlp {
    let scale1 = (2.0 / HEAD_DIM as f64).sqrt();
    let scale2 = (2.0 / HIDDEN as f64).sqrt();
    let w1: Vec<f32> = (0..HEAD_DIM * HIDDEN)
        .map(|_| (rng.next_f64() * 2.0 * scale1) as f32)
        .collect();
    let w2: Vec<f32> = (0..HIDDEN)
        .map(|_| (rng.next_f64() * 2.0 * scale2) as f32)
        .collect();

    Mlp {
        w1: ctx.upload(&w1, HEAD_DIM, HIDDEN).expect("w1"),
        b1: ctx.upload(&vec![0.0; HIDDEN], 1, HIDDEN).expect("b1"),
        w2: ctx.upload(&w2, HIDDEN, 1).expect("w2"),
        b2: ctx.upload(&[0.0], 1, 1).expect("b2"),
    }
}

/// Train on GPU with resident tensors, then report accuracy on the held-out fold.
///
/// Every intermediate stays on the device between steps; only the final
/// predictions come back. Adam rather than SGD because the feature scales differ
/// sharply between schedules — a schedule that retrieves nothing produces
/// near-constant features, and a fixed learning rate tuned for one case would
/// confound the comparison with an optimisation artefact.
fn train_fold(
    ctx: &GpuContext,
    x_train: &[f32],
    y_train: &[f32],
    x_test: &[f32],
    y_test: &[f32],
    rng: &mut Lcg,
) -> Vec<bool> {
    let n = y_train.len();
    let mut mlp = init(ctx, rng);

    let x = ctx.upload(x_train, n, HEAD_DIM).expect("x");
    let y = ctx.upload(y_train, n, 1).expect("y");
    let xt = ctx.transpose_resident(&x).expect("xt");

    let mut s_w1 = ctx.adam_state(&mlp.w1).expect("state w1");
    let mut s_b1 = ctx.adam_state(&mlp.b1).expect("state b1");
    let mut s_w2 = ctx.adam_state(&mlp.w2).expect("state w2");
    let mut s_b2 = ctx.adam_state(&mlp.b2).expect("state b2");
    let lr = 0.02;

    for _ in 0..EPOCHS {
        let z1 = ctx.matmul_resident(&x, &mlp.w1).expect("z1");
        let z1 = ctx.add_bias_resident(&z1, &mlp.b1).expect("z1 bias");
        let a1 = ctx.relu_resident(&z1).expect("a1");
        let z2 = ctx.matmul_resident(&a1, &mlp.w2).expect("z2");
        let z2 = ctx.add_bias_resident(&z2, &mlp.b2).expect("z2 bias");

        // Fused sigmoid + BCE gradient: dz2 = sigmoid(z2) - y, averaged.
        let dz2 = ctx.sigmoid_bce_grad_resident(&z2, &y).expect("dz2");

        let a1t = ctx.transpose_resident(&a1).expect("a1t");
        let gw2 = ctx.matmul_resident(&a1t, &dz2).expect("gw2");
        let gb2 = ctx.column_sums_resident(&dz2).expect("gb2");

        let w2t = ctx.transpose_resident(&mlp.w2).expect("w2t");
        let da1 = ctx.matmul_resident(&dz2, &w2t).expect("da1");
        let dz1 = ctx.relu_backward_resident(&z1, &da1).expect("dz1");
        let gw1 = ctx.matmul_resident(&xt, &dz1).expect("gw1");
        let gb1 = ctx.column_sums_resident(&dz1).expect("gb1");

        mlp.w1 = ctx
            .adam_update_resident(&mlp.w1, &gw1, &mut s_w1, lr)
            .expect("w1");
        mlp.b1 = ctx
            .adam_update_resident(&mlp.b1, &gb1, &mut s_b1, lr)
            .expect("b1");
        mlp.w2 = ctx
            .adam_update_resident(&mlp.w2, &gw2, &mut s_w2, lr)
            .expect("w2");
        mlp.b2 = ctx
            .adam_update_resident(&mlp.b2, &gb2, &mut s_b2, lr)
            .expect("b2");
    }

    let m = y_test.len();
    let xe = ctx.upload(x_test, m, HEAD_DIM).expect("x test");
    let z1 = ctx.matmul_resident(&xe, &mlp.w1).expect("z1");
    let z1 = ctx.add_bias_resident(&z1, &mlp.b1).expect("z1 bias");
    let a1 = ctx.relu_resident(&z1).expect("a1");
    let z2 = ctx.matmul_resident(&a1, &mlp.w2).expect("z2");
    let z2 = ctx.add_bias_resident(&z2, &mlp.b2).expect("z2 bias");
    let p = ctx.sigmoid_resident(&z2).expect("p");
    let pred = ctx.read(&p).expect("read");

    pred.iter()
        .zip(y_test)
        .map(|(&p, &t)| (p >= 0.5) == (t >= 0.5))
        .collect()
}

/// Wilson score interval for a proportion.
///
/// Every sample receives exactly one out-of-fold prediction, from a model that
/// never trained on it, so pooled accuracy is a binomial over `n` independent
/// trials and an interval on it is legitimate. The fold range reported earlier is
/// not: five numbers bound a spread and say nothing about where the underlying
/// rate lies.
///
/// Wilson rather than the textbook normal approximation because the latter
/// misbehaves near 0 and 1 and can produce bounds outside [0, 1], which for a
/// chance-floor task is exactly the region several arms occupy.
fn wilson(correct: usize, n: usize) -> (f64, f64) {
    let z = 1.96;
    let p = correct as f64 / n as f64;
    let nf = n as f64;
    let denom = 1.0 + z * z / nf;
    let centre = (p + z * z / (2.0 * nf)) / denom;
    let half = z * (p * (1.0 - p) / nf + z * z / (4.0 * nf * nf)).sqrt() / denom;

    // Clamped because the result is a probability interval and the endpoints are
    // exact at the extremes: at `correct == n` the upper bound is exactly 1 in
    // real arithmetic, and f64 lands one ulp below at 0.9999999999999999. The
    // clamp reports the quantity rather than the rounding, and it is not hiding a
    // wide excursion — a formula error large enough to matter would still fail
    // the bracket assertion in the tests below.
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

/// McNemar's test on paired predictions.
///
/// The arms are evaluated on the same sequences, so comparing two independent
/// intervals throws away the pairing and loses power. What carries the signal is
/// the discordant pairs: samples one arm got right and the other did not.
///
/// This exists because of a specific failure. At 600 samples this experiment
/// reported the topological schedule at 58.5% against random at 52.2% — a
/// six-point gap favouring the mechanism under test — and only four times the
/// data revealed it as noise. A comparison that reports two point estimates and
/// leaves the reader to eyeball the gap will keep producing that outcome.
///
/// Returns `(b, c, chi-squared)` where `b` counts samples the first arm got right
/// and the second wrong. The continuity correction is Edwards's, which is the
/// conservative choice: it makes the test slightly less likely to call a
/// difference real, and this file has already been wrong in that direction once.
fn mcnemar(a: &[bool], b_arm: &[bool]) -> (usize, usize, f64) {
    let b = a.iter().zip(b_arm).filter(|(&x, &y)| x && !y).count();
    let c = a.iter().zip(b_arm).filter(|(&x, &y)| !x && y).count();

    if b + c == 0 {
        return (b, c, 0.0);
    }
    let diff = (b as f64 - c as f64).abs();
    let chi2 = if diff <= 1.0 {
        0.0
    } else {
        (diff - 1.0).powi(2) / (b + c) as f64
    };
    (b, c, chi2)
}

/// Complementary error function, Numerical Recipes 6.2.2.
///
/// Accurate to about 1.2e-7 relative, which is far finer than the threshold it
/// feeds. `std` has no `erfc` and pulling a dependency in for one call would be
/// heavier than the seven lines.
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 2.0 / (2.0 + z);
    let ty = 4.0 * t - 2.0;
    let coefficients = [
        -1.3026537197817094,
        6.419_697_923_564_902e-1,
        1.9476473204185836e-2,
        -9.561_514_786_808_631e-3,
        -9.46595344482036e-4,
        3.66839497852761e-4,
        4.2523324806907e-5,
        -2.0278578112534e-5,
        -1.624290004647e-6,
        1.303655835580e-6,
        1.5626441722e-8,
        -8.5238095915e-8,
    ];
    let mut d = 0.0;
    let mut dd = 0.0;
    for &c in coefficients.iter().rev().take(coefficients.len() - 1) {
        let tmp = d;
        d = ty * d - dd + c;
        dd = tmp;
    }
    let ans = t * (-z * z + 0.5 * (coefficients[0] + ty * d) - dd).exp();
    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

/// Chi-squared critical value at `alpha` with one degree of freedom.
///
/// For one degree of freedom the survival function is exactly
/// `erfc(sqrt(x / 2))`, so this inverts it by bisection rather than carrying a
/// table that would silently stop matching the number of comparisons.
fn chi2_critical(alpha: f64) -> f64 {
    let (mut lo, mut hi) = (0.0f64, 200.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if erfc((mid / 2.0).sqrt()) > alpha {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Family-wise error rate for the whole table of comparisons.
///
/// Not the per-comparison rate, and the difference is the reason this constant
/// exists. Ten pairs tested at 5% each produce about one spurious "better" per
/// two runs by construction, and the interesting comparison is always chosen
/// after seeing the table.
///
/// That is not hypothetical here. At 600 sequences this file reported
/// `topological better` over random with chi-squared 4.72, clearing the
/// uncorrected 3.841 threshold. The 2,400-sequence run puts the same pair at
/// 0.09 — the two arms are indistinguishable, and the earlier verdict was the
/// false positive a 5% test is entitled to produce. Bonferroni at 0.05/10 sets
/// the bar at 7.88 and declines it, which is the behaviour wanted from an
/// instrument whose job is to say when the data cannot decide.
const FAMILY_ALPHA: f64 = 0.05;

/// One out-of-fold verdict per sample, in sample order.
///
/// Returned per sample rather than as a fold average so the arms can be compared
/// pairwise on identical sequences. Aggregating to five fold accuracies first
/// discards which samples each arm got right, which is the whole content of a
/// paired test.
fn cross_validate(ctx: &GpuContext, x: &[f32], y: &[f32], rng: &mut Lcg) -> Vec<bool> {
    let n = y.len();
    let fold = n / FOLDS;
    let mut verdicts = vec![false; n];

    for f in 0..FOLDS {
        let lo = f * fold;
        let hi = if f == FOLDS - 1 { n } else { lo + fold };

        let mut x_train = Vec::new();
        let mut y_train = Vec::new();
        let mut x_test = Vec::new();
        let mut y_test = Vec::new();
        for i in 0..n {
            let (xs, ys) = if i >= lo && i < hi {
                (&mut x_test, &mut y_test)
            } else {
                (&mut x_train, &mut y_train)
            };
            xs.extend_from_slice(&x[i * HEAD_DIM..(i + 1) * HEAD_DIM]);
            ys.push(y[i]);
        }

        let fold_verdicts = train_fold(ctx, &x_train, &y_train, &x_test, &y_test, rng);
        verdicts[lo..hi].copy_from_slice(&fold_verdicts);
    }

    verdicts
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("No GPU adapter: {e}");
            std::process::exit(1);
        }
    };

    let config = TopologyScheduleConfig {
        block_size: BLOCK,
        local_radius_blocks: 1,
        sink_blocks: 1,
        topk_topology_blocks: 4,
    };

    let mut rng = Lcg(0x5EED);
    let samples: Vec<Sample> = (0..SAMPLES).map(|_| sample(&mut rng)).collect();

    let info = ctx.adapter_info();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Associative recall: does recovered mass predict trained accuracy?");
    println!("  adapter {}  |  {}", info.name, info.backend);
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
    println!("  seq {SEQ}, head_dim {HEAD_DIM}, block {BLOCK}, {SAMPLES} sequences");
    println!("  head: {HEAD_DIM} -> {HIDDEN} -> 1, Adam, {EPOCHS} epochs, {FOLDS}-fold CV");
    println!("  attention and training both on GPU; no gradient flows into the schedule");
    println!();
    println!(
        "  {:>12}  {:>8}  {:>7}  {:>9}  {:>15}",
        "schedule", "density", "mass", "accuracy", "95% interval"
    );
    println!(
        "  {:->12}  {:->8}  {:->7}  {:->9}  {:->15}",
        "", "", "", "", ""
    );

    let mut outcomes: Vec<(&str, Vec<bool>)> = Vec::new();

    // Each sample gets its own schedule, derived from its own keys. A schedule
    // shared across samples would be reading structure the sequences do not have
    // in common.
    let variants: [&str; 5] = ["dense", "topological", "inverted", "random", "oracle-free"];

    for variant in variants {
        let mut x = Vec::with_capacity(SAMPLES * HEAD_DIM);
        let mut y = Vec::with_capacity(SAMPLES);
        let mut mass_total = 0.0;
        let mut density_total = 0.0;

        for (i, s) in samples.iter().enumerate() {
            let topological =
                topology_block_schedule(&s.k, SEQ, HEAD_DIM, config).expect("valid config");
            let budget = schedule_budget(&topological);

            let schedule = match variant {
                "dense" => dense_causal_block_schedule(SEQ / BLOCK),
                "topological" => topological,
                "inverted" => inverted_topology_block_schedule(&s.k, SEQ, HEAD_DIM, config)
                    .expect("valid config"),
                "random" => random_block_schedule(&budget, 7000 + i as u64).expect("valid budget"),
                // Local window and sinks only: the schedule with the topology
                // removed entirely. It shares the shape of the others without
                // any content-derived selection, so it separates "the salience
                // helps" from "the fixed structure was doing the work".
                _ => topology_block_schedule(
                    &s.k,
                    SEQ,
                    HEAD_DIM,
                    TopologyScheduleConfig {
                        topk_topology_blocks: 0,
                        ..config
                    },
                )
                .expect("valid config"),
            };

            mass_total += block_mass_recovered(&schedule, &s.q, &s.k, SEQ, HEAD_DIM, BLOCK)
                .expect("valid shapes");
            let spent: usize = schedule_budget(&schedule).iter().sum();
            let dense: usize = schedule_budget(&dense_causal_block_schedule(SEQ / BLOCK))
                .iter()
                .sum();
            density_total += spent as f64 / dense as f64;

            x.extend_from_slice(&features(&ctx, s, &schedule));
            y.push(s.label);
        }

        let mut fold_rng = Lcg(0xC0FFEE);
        let verdicts = cross_validate(&ctx, &x, &y, &mut fold_rng);
        let correct = verdicts.iter().filter(|&&v| v).count();
        let mean = correct as f32 / verdicts.len() as f32;
        let (lo, hi) = wilson(correct, verdicts.len());

        println!(
            "  {variant:>12}  {:>7.1}%  {:>7.4}  {:>8.1}%  {:>6.1}% - {:>5.1}%",
            100.0 * density_total / SAMPLES as f64,
            mass_total / SAMPLES as f64,
            100.0 * mean,
            100.0 * lo,
            100.0 * hi
        );
        outcomes.push((variant, verdicts));

        if variant == "dense" && mean < DENSE_FLOOR {
            println!();
            eprintln!(
                "dense attention scored {:.1}%, below the {:.0}% this comparison \
                 needs.",
                100.0 * mean,
                100.0 * DENSE_FLOOR
            );
            eprintln!(
                "Dense sees every block, so it bounds what retrieval can offer here. \
                 Near chance means the task is not learnable as configured and the \
                 remaining rows would differ only by noise."
            );
            eprintln!(
                "Raise MATCH so the planted key wins more of the softmax, or shorten \
                 SEQ so it competes against fewer distractors. Refusing to print a \
                 comparison whose control failed."
            );
            std::process::exit(2);
        }
    }

    // Pairwise on identical sequences. Two overlapping intervals do not settle a
    // comparison — they discard the pairing, which is where the power is — and
    // two non-overlapping ones are a stronger claim than the data has to make.
    println!();
    println!("───────────────────────────────────────────────────────────────────────");
    let pairs = outcomes.len() * (outcomes.len() - 1) / 2;
    let per_comparison = FAMILY_ALPHA / pairs as f64;
    let critical = chi2_critical(per_comparison);

    println!("  Paired comparison on identical sequences (McNemar)");
    println!(
        "  {pairs} comparisons, Bonferroni: {:.1}% family-wise, {:.4} each, chi2 > {critical:.2}",
        100.0 * FAMILY_ALPHA,
        per_comparison
    );
    println!();
    println!(
        "  {:>13} vs {:<13}  {:>6}  {:>6}  {:>8}  {:>14}",
        "A", "B", "A>B", "B>A", "chi2", "verdict"
    );
    println!(
        "  {:->13}    {:->13}  {:->6}  {:->6}  {:->8}  {:->14}",
        "", "", "", "", "", ""
    );

    for i in 0..outcomes.len() {
        for j in (i + 1)..outcomes.len() {
            let (name_a, ref va) = outcomes[i];
            let (name_b, ref vb) = outcomes[j];
            let (b, c, chi2) = mcnemar(va, vb);

            let verdict = if chi2 > critical {
                if b > c {
                    format!("{name_a} better")
                } else {
                    format!("{name_b} better")
                }
            } else {
                "not resolved".to_string()
            };

            println!("  {name_a:>13} vs {name_b:<13}  {b:>6}  {c:>6}  {chi2:>8.2}  {verdict:>14}");
        }
    }

    println!();
    println!("  'not resolved' means the discordant counts are close enough that");
    println!("  this many sequences cannot separate the two arms. It is a statement");
    println!("  about the experiment's resolution, not a claim the arms are equal.");
    println!();
    println!("  This exists because an earlier run of this file, at 600 sequences,");
    println!("  reported topological 58.5% against random 52.2% and would have been");
    println!("  read as the mechanism winning. Four times the data moved random to");
    println!("  the top. Two point estimates and an eyeballed gap will keep");
    println!("  producing that; a paired test on the same data would not have.");

    println!();
    println!("───────────────────────────────────────────────────────────────────────");
    println!("  The label is the sign of a value component at a content-addressed");
    println!("  position, so it is reachable only by retrieving that position. A");
    println!("  schedule that misses the block holding it leaves the head with");
    println!("  features carrying no information about the label, and accuracy");
    println!("  falls to chance. 50% is therefore the floor, not a bad score.");
    println!();
    println!("  'oracle-free' keeps the local window and sinks with the salience");
    println!("  selection removed. It separates a gain from the topology from a");
    println!("  gain the fixed structure would have produced on its own.");
    println!("═══════════════════════════════════════════════════════════════════════");
}

/// The statistics decide the verdicts, so they are pinned rather than trusted.
///
/// `chi2_critical` inverts an approximation by bisection, and a wrong constant
/// here does not fail loudly — it silently moves the bar and changes every
/// "resolved" in the table.
///
/// Run with `cargo test -p aether-gpu --example recall_training`.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_critical_values_match_the_published_table() {
        // Chi-squared, one degree of freedom, from any standard table.
        for (alpha, expected) in [
            (0.05, 3.841),
            (0.01, 6.635),
            (0.005, 7.879),
            (0.001, 10.828),
        ] {
            let got = chi2_critical(alpha);
            assert!(
                (got - expected).abs() < 0.01,
                "chi2_critical({alpha}) gave {got:.4}, table says {expected}"
            );
        }
    }

    #[test]
    fn the_correction_raises_the_bar_above_the_uncorrected_test() {
        let uncorrected = chi2_critical(0.05);
        let corrected = chi2_critical(0.05 / 10.0);
        assert!(
            corrected > uncorrected,
            "correcting for ten comparisons lowered the threshold, {corrected} \
             against {uncorrected}"
        );

        // The observed statistic from the 600-sequence run that this correction
        // exists to decline. It cleared the uncorrected bar and must not clear
        // the corrected one.
        let observed = 4.72;
        assert!(
            observed > uncorrected && observed < corrected,
            "the 600-sequence topological-vs-random statistic of {observed} no \
             longer sits between {uncorrected} and {corrected}, so this test has \
             stopped guarding the case it was written for"
        );
    }

    #[test]
    fn mcnemar_ignores_agreement_and_counts_only_disagreement() {
        // Identical arms: no discordant pairs, nothing to resolve.
        let a = vec![true, false, true, false];
        assert_eq!(mcnemar(&a, &a), (0, 0, 0.0));

        // Agreement is irrelevant to the statistic. Adding samples both arms get
        // right must leave it unchanged, which is the property that makes the
        // test paired rather than a comparison of two accuracies.
        let x = vec![true, false, false];
        let y = vec![false, true, false];
        let (b1, c1, chi1) = mcnemar(&x, &y);

        let mut x2 = x.clone();
        let mut y2 = y.clone();
        for _ in 0..50 {
            x2.push(true);
            y2.push(true);
        }
        let (b2, c2, chi2) = mcnemar(&x2, &y2);

        assert_eq!(
            (b1, c1),
            (b2, c2),
            "agreement changed the discordant counts"
        );
        assert!(
            (chi1 - chi2).abs() < 1e-12,
            "fifty agreeing samples moved the statistic from {chi1} to {chi2}"
        );
    }

    #[test]
    fn the_wilson_interval_brackets_the_estimate_and_stays_in_range() {
        for (correct, n) in [
            (0usize, 100usize),
            (50, 100),
            (100, 100),
            (1, 5),
            (2400, 2400),
        ] {
            let p = correct as f64 / n as f64;
            let (lo, hi) = wilson(correct, n);

            // A rounding tolerance, not a fudge. At `correct == n` the exact
            // upper bound is 1, and f64 evaluates the expression to
            // 0.9999999999999999 — one ulp low, so a strict comparison here
            // asserts exact arithmetic rather than a property of the interval.
            // 1e-12 is roughly ten thousand times the observed 1.1e-16
            // discrepancy and still orders of magnitude below anything that
            // would matter: a wrong formula moves these bounds by 1e-3 or more.
            const ROUNDING: f64 = 1e-12;
            assert!(
                lo - ROUNDING <= p && p <= hi + ROUNDING,
                "{correct}/{n}: {p} outside [{lo}, {hi}]"
            );
            assert!(
                lo >= 0.0 && hi <= 1.0,
                "{correct}/{n}: interval [{lo}, {hi}] leaves [0, 1], which is the \
                 failure the normal approximation has and Wilson does not"
            );
        }

        // More data must not widen the interval at a fixed rate.
        let (lo_small, hi_small) = wilson(60, 100);
        let (lo_large, hi_large) = wilson(600, 1000);
        assert!(
            (hi_large - lo_large) < (hi_small - lo_small),
            "ten times the data did not narrow the interval"
        );
    }
}
