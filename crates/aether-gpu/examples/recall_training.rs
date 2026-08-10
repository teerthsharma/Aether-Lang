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
) -> f32 {
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

    let correct = pred
        .iter()
        .zip(y_test)
        .filter(|(&p, &t)| (p >= 0.5) == (t >= 0.5))
        .count();
    correct as f32 / m as f32
}

fn cross_validate(ctx: &GpuContext, x: &[f32], y: &[f32], rng: &mut Lcg) -> (f32, f32, f32) {
    let n = y.len();
    let fold = n / FOLDS;
    let mut scores = Vec::with_capacity(FOLDS);

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
        scores.push(train_fold(ctx, &x_train, &y_train, &x_test, &y_test, rng));
    }

    let mean = scores.iter().sum::<f32>() / FOLDS as f32;
    let lo = scores.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (mean, lo, hi)
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
        "schedule", "density", "mass", "accuracy", "fold range"
    );
    println!(
        "  {:->12}  {:->8}  {:->7}  {:->9}  {:->15}",
        "", "", "", "", ""
    );

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
        let (mean, lo, hi) = cross_validate(&ctx, &x, &y, &mut fold_rng);

        println!(
            "  {variant:>12}  {:>7.1}%  {:>7.4}  {:>8.1}%  {:>6.1}% - {:>5.1}%",
            100.0 * density_total / SAMPLES as f64,
            mass_total / SAMPLES as f64,
            100.0 * mean,
            100.0 * lo,
            100.0 * hi
        );

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
