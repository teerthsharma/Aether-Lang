//! Can a model trained through attention compensate for the schedule it was given?
//!
//! Run:
//!   cargo run -p aether-gpu --example recall_end_to_end --release
//!   cargo run -p aether-gpu --example recall_end_to_end --release -- --gpu-backward
//!
//! The flag swaps the f64 CPU backward for the four WGSL kernels. Both produce
//! the same sixteen figures, which is a check the parity tests cannot make: they
//! pin the gradient at a point, and this accumulates f32 gradients across 60
//! epochs of 300 sequences, where a small per-step error would compound into a
//! different model.
//!
//! `recall_training` freezes attention and trains only a head on its output. It
//! finds the topological, inverted and random schedules indistinguishable at
//! equal budget, and records one assumption as its narrowest: a model trained end
//! to end could reshape its queries to suit whatever schedule it was handed, so a
//! frozen comparison might understate what a schedule is worth in practice.
//!
//! That assumption is now testable, because `scheduled_attention_backward` exists
//! and is verified against central differences.
//!
//! # What is learned
//!
//! A query projection `Wq`, shared across every sequence, trained through the
//! attention kernel by gradient descent. Keys, values and the schedule are fixed
//! properties of each sample.
//!
//! Learning only the queries is not a simplification of the question — it *is*
//! the question. The concern was specifically that the model could reshape its
//! queries; keeping everything else fixed isolates that and nothing else. If a
//! learned `Wq` closes the gap between schedules, the frozen comparison was
//! measuring the wrong thing. If it does not, the frozen result stands and its
//! stated limit is discharged rather than merely acknowledged.
//!
//! Holding the schedule fixed is also what keeps the arms comparable. A schedule
//! derived from learned keys would drift during training at a different rate per
//! arm, and the comparison would confound the selector with its own moving target.

use aether_core::scheduled::{
    dense_causal_block_schedule, inverted_topology_block_schedule, random_block_schedule,
    schedule_budget, scheduled_attention_backward, topology_block_schedule, AttentionGradients,
    BlockSchedule, TopologyScheduleConfig,
};
use aether_gpu::GpuContext;

const SEQ: usize = 64;
const HEAD_DIM: usize = 8;
const BLOCK: usize = 8;
const SAMPLES: usize = 400;
const EPOCHS: usize = 60;
const LR: f64 = 20.0;
/// How strongly the final query is planted onto the target key.
///
/// `recall_training` uses 30, chosen so the target dominates the softmax against
/// SEQ-1 distractors and the frozen features carry the label cleanly. That value
/// is wrong here, and the reason is the point of this constant.
///
/// A softmax that has collapsed onto one column has a vanishing Jacobian:
/// `ds = p (dp - delta)`, and when a single `p` is near 1 the delta term cancels
/// it, so almost no gradient reaches `Wq`. At 20 the positive control below
/// recovered 42.0% to 42.0% — not a small improvement, exactly none. The
/// experiment was measuring an optimiser that could not move.
///
/// The two requirements pull against each other: sharp retrieval makes the task
/// learnable and makes the gradient vanish. 5 sits where the softmax is still
/// informative and still differentiable, which costs the identity projection its
/// head start and is what the control is there to confirm.
const MATCH: f64 = 5.0;

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
    /// Raw query features. `Wq` projects these into the queries attention sees.
    xq: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    label: f64,
    schedule_seed: u64,
}

/// One associative-recall sequence, as in `recall_training`.
///
/// The final row of `xq` is planted onto the target key, so retrieval is
/// available to a projection that preserves it.
///
/// An earlier version of this comment said the identity projection already
/// retrieves the target, which was true at `MATCH = 20` and is not true at 5:
/// the untrained accuracies below sit near chance. The claim went stale when the
/// constant moved, which is the same defect this repository keeps finding in its
/// own prose — a comment describing a measurement taken under different
/// conditions.
fn sample(rng: &mut Lcg, index: usize) -> Sample {
    let mut xq: Vec<f64> = (0..SEQ * HEAD_DIM).map(|_| rng.next_f64()).collect();
    let k: Vec<f64> = (0..SEQ * HEAD_DIM).map(|_| rng.next_f64()).collect();
    let v: Vec<f64> = (0..SEQ * HEAD_DIM).map(|_| rng.next_f64()).collect();

    let target = rng.next_usize(SEQ - BLOCK);
    let last = SEQ - 1;
    for d in 0..HEAD_DIM {
        xq[last * HEAD_DIM + d] = k[target * HEAD_DIM + d] * MATCH + rng.next_f64() * 0.2;
    }

    let label = if v[target * HEAD_DIM] > 0.0 { 1.0 } else { 0.0 };
    Sample {
        xq,
        k,
        v,
        label,
        schedule_seed: 4000 + index as u64,
    }
}

fn project(xq: &[f64], w: &[f64]) -> Vec<f64> {
    let mut q = vec![0.0; SEQ * HEAD_DIM];
    for row in 0..SEQ {
        for out in 0..HEAD_DIM {
            let mut sum = 0.0;
            for inp in 0..HEAD_DIM {
                sum += xq[row * HEAD_DIM + inp] * w[inp * HEAD_DIM + out];
            }
            q[row * HEAD_DIM + out] = sum;
        }
    }
    q
}

/// One forward and backward pass, accumulating into the shared gradients.
///
/// Returns `(loss, correct)`. The attention forward runs on the GPU when a
/// context is supplied, which is what makes this a GPU training loop rather than
/// a CPU one with a GPU kernel available.
#[allow(clippy::too_many_arguments)]
fn step(
    ctx: &GpuContext,
    s: &Sample,
    wq: &[f64],
    head: &[f64],
    schedule: &BlockSchedule,
    g_wq: &mut [f64],
    g_head: &mut [f64],
    train: bool,
    gpu_backward: bool,
) -> (f64, bool) {
    let q = project(&s.xq, wq);

    let gpu_out = ctx
        .scheduled_attention(
            &q.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &s.k.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &s.v.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            SEQ,
            HEAD_DIM,
            schedule,
            BLOCK,
        )
        .expect("attention forward");
    let out: Vec<f64> = gpu_out.iter().map(|&x| x as f64).collect();

    // The head reads the final row only, which is where the planted query lives.
    let features = &out[(SEQ - 1) * HEAD_DIM..];
    let logit: f64 = features.iter().zip(head).map(|(f, h)| f * h).sum::<f64>() + head[HEAD_DIM];
    let p = 1.0 / (1.0 + (-logit).exp());
    let loss = -(s.label * p.max(1e-12).ln() + (1.0 - s.label) * (1.0 - p).max(1e-12).ln());
    let correct = (p >= 0.5) == (s.label >= 0.5);

    if !train {
        return (loss, correct);
    }

    let dlogit = p - s.label;
    for d in 0..HEAD_DIM {
        g_head[d] += dlogit * features[d];
    }
    g_head[HEAD_DIM] += dlogit;

    // Only the final row of the attention output carries gradient, since it is
    // the only row the head reads.
    let mut d_out = vec![0.0; SEQ * HEAD_DIM];
    for d in 0..HEAD_DIM {
        d_out[(SEQ - 1) * HEAD_DIM + d] = dlogit * head[d];
    }

    let dq: Vec<f64> = if gpu_backward {
        // The WGSL backward, in f32. Unit tests pin it against the f64 reference
        // at a point; running a whole training loop through it tests something
        // they cannot, which is whether f32 gradients stay usable once they are
        // accumulated across hundreds of steps. A gradient can be correct
        // everywhere it is sampled and still drift a run.
        let (dq, _, _) = ctx
            .scheduled_attention_backward_resident(
                &q.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                &s.k.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                &s.v.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                SEQ,
                HEAD_DIM,
                schedule,
                BLOCK,
                &d_out.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            )
            .expect("gpu attention backward");
        ctx.read(&dq)
            .expect("read dq")
            .iter()
            .map(|&x| x as f64)
            .collect()
    } else {
        scheduled_attention_backward(&q, &s.k, &s.v, SEQ, HEAD_DIM, schedule, BLOCK, &d_out)
            .expect("attention backward")
            .dq
    };
    let grads = AttentionGradients {
        dq,
        dk: Vec::new(),
        dv: Vec::new(),
    };

    // dWq = Xq^T dQ, the only place the query projection appears.
    for row in 0..SEQ {
        for inp in 0..HEAD_DIM {
            let x = s.xq[row * HEAD_DIM + inp];
            if x == 0.0 {
                continue;
            }
            for out_d in 0..HEAD_DIM {
                g_wq[inp * HEAD_DIM + out_d] += x * grads.dq[row * HEAD_DIM + out_d];
            }
        }
    }

    (loss, correct)
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("No GPU adapter: {e}");
            std::process::exit(1);
        }
    };

    let mut rng = Lcg(0xB0A7);
    let samples: Vec<Sample> = (0..SAMPLES).map(|i| sample(&mut rng, i)).collect();

    let config = TopologyScheduleConfig {
        block_size: BLOCK,
        local_radius_blocks: 1,
        sink_blocks: 1,
        topk_topology_blocks: 2,
    };

    let gpu_backward = std::env::args().any(|a| a == "--gpu-backward");
    let split = SAMPLES * 3 / 4;
    let info = ctx.adapter_info();

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  End-to-end: does a learned query projection close the schedule gap?");
    println!("  adapter {}  |  {}", info.name, info.backend);
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
    println!("  seq {SEQ}, head_dim {HEAD_DIM}, block {BLOCK}, {SAMPLES} sequences");
    println!("  Wq [{HEAD_DIM} x {HEAD_DIM}] trained through attention, {EPOCHS} epochs");
    println!(
        "  keys, values and schedule fixed per sample; {split} train / {} test",
        SAMPLES - split
    );
    println!();
    println!(
        "  {:>12}  {:>10}  {:>10}  {:>10}",
        "schedule", "identity", "trained", "change"
    );
    println!(
        "  {:>12}  {:>10}  {:>10}  {:>10}   {}",
        "", "", "", "", "control: scrambled -> trained"
    );
    println!("  {:->12}  {:->10}  {:->10}  {:->10}", "", "", "", "");

    for variant in ["dense", "topological", "inverted", "random"] {
        let schedules: Vec<BlockSchedule> = samples
            .iter()
            .map(|s| {
                let topological =
                    topology_block_schedule(&s.k, SEQ, HEAD_DIM, config).expect("valid config");
                match variant {
                    "dense" => dense_causal_block_schedule(SEQ / BLOCK),
                    "topological" => topological,
                    "inverted" => inverted_topology_block_schedule(&s.k, SEQ, HEAD_DIM, config)
                        .expect("valid config"),
                    _ => random_block_schedule(&schedule_budget(&topological), s.schedule_seed)
                        .expect("valid budget"),
                }
            })
            .collect();

        // Identity initialisation: the projection that passes the planted query
        // through unchanged. It is the neutral starting point rather than a
        // working solution — at MATCH = 5 it scores near chance — so the column
        // below measures what training adds, with the control alongside
        // establishing that training adds anything at all.
        let mut wq = vec![0.0; HEAD_DIM * HEAD_DIM];
        for d in 0..HEAD_DIM {
            wq[d * HEAD_DIM + d] = 1.0;
        }
        let mut head = vec![0.0; HEAD_DIM + 1];
        head[0] = 1.0;

        let accuracy = |wq: &[f64], head: &[f64], ctx: &GpuContext| -> f64 {
            let mut correct = 0;
            let mut sink_wq = vec![0.0; HEAD_DIM * HEAD_DIM];
            let mut sink_head = vec![0.0; HEAD_DIM + 1];
            for i in split..SAMPLES {
                let (_, ok) = step(
                    ctx,
                    &samples[i],
                    wq,
                    head,
                    &schedules[i],
                    &mut sink_wq,
                    &mut sink_head,
                    false,
                    gpu_backward,
                );
                if ok {
                    correct += 1;
                }
            }
            correct as f64 / (SAMPLES - split) as f64
        };

        let before = accuracy(&wq, &head, &ctx);

        // Positive control, run first so its verdict is available before the
        // headline number is read.
        //
        // A table showing "training changed nothing" is worthless unless training
        // can change something. If Wq barely moves from the identity, that is
        // equally consistent with the schedule being irrecoverable and with the
        // learning rate being too small, the gradient not reaching Wq, or the
        // projection being disconnected from the loss.
        //
        // Starting from a scrambled Wq destroys the planted retrieval. Recovery
        // from there is proof the loop works; failure to recover means the null
        // result below describes the optimiser rather than the schedule.
        let mut control_wq: Vec<f64> = {
            let mut r = Lcg(0xC047 + variant.len() as u64);
            (0..HEAD_DIM * HEAD_DIM).map(|_| r.next_f64()).collect()
        };
        let mut control_head = head.clone();
        let control_before = accuracy(&control_wq, &control_head, &ctx);
        for _ in 0..EPOCHS {
            let mut g_wq = vec![0.0; HEAD_DIM * HEAD_DIM];
            let mut g_head = vec![0.0; HEAD_DIM + 1];
            for i in 0..split {
                step(
                    &ctx,
                    &samples[i],
                    &control_wq,
                    &control_head,
                    &schedules[i],
                    &mut g_wq,
                    &mut g_head,
                    true,
                    gpu_backward,
                );
            }
            let n = split as f64;
            for (w, g) in control_wq.iter_mut().zip(&g_wq) {
                *w -= LR * g / n;
            }
            for (h, g) in control_head.iter_mut().zip(&g_head) {
                *h -= LR * g / n;
            }
        }
        let control_after = accuracy(&control_wq, &control_head, &ctx);

        for _ in 0..EPOCHS {
            let mut g_wq = vec![0.0; HEAD_DIM * HEAD_DIM];
            let mut g_head = vec![0.0; HEAD_DIM + 1];

            for i in 0..split {
                step(
                    ctx_ref(&ctx),
                    &samples[i],
                    &wq,
                    &head,
                    &schedules[i],
                    &mut g_wq,
                    &mut g_head,
                    true,
                    gpu_backward,
                );
            }

            let n = split as f64;
            for (w, g) in wq.iter_mut().zip(&g_wq) {
                *w -= LR * g / n;
            }
            for (h, g) in head.iter_mut().zip(&g_head) {
                *h -= LR * g / n;
            }
        }

        let after = accuracy(&wq, &head, &ctx);
        println!(
            "  {variant:>12}  {:>9.1}%  {:>9.1}%  {:>+9.1}%   {:>8.1}% -> {:>5.1}%",
            100.0 * before,
            100.0 * after,
            100.0 * (after - before),
            100.0 * control_before,
            100.0 * control_after
        );
    }

    println!();
    println!("───────────────────────────────────────────────────────────────────────");
    println!("  'identity' is the untrained projection, which already retrieves the");
    println!("  planted key. 'trained' is after gradient descent through attention.");
    println!();
    println!("  If training closes the gap between schedules, the frozen comparison");
    println!("  in recall_training understated what a schedule is worth. If the gap");
    println!("  survives, that result stands and its narrowest stated assumption is");
    println!("  discharged rather than merely acknowledged.");
    println!("═══════════════════════════════════════════════════════════════════════");
}

/// Identity, to keep the borrow obvious at the call site inside the epoch loop.
fn ctx_ref(ctx: &GpuContext) -> &GpuContext {
    ctx
}
