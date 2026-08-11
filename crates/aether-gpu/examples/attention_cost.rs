//! Where does a scheduled-attention call actually spend its time?
//!
//! Run:
//!   cargo run -p aether-gpu --example attention_cost --release
//!
//! `FEATURES.md` records the backward pass at 493 s against the CPU reference's
//! 216 s on an end-to-end training run, and attributes the gap to "the per-call
//! operand upload and four dispatches". That attribution was written down
//! without being measured, which makes it a guess with a mechanism attached —
//! the same shape as several claims this repository has already had to withdraw.
//!
//! The two candidates predict different numbers, so one measurement separates
//! them. Per call the forward uploads `3·seq·head_dim` floats and records one
//! dispatch; the backward uploads `4·seq·head_dim + 3·seq` plus the schedule and
//! records four. If dispatch count dominates, the backward costs about four
//! times the forward. If transfer volume dominates, about one and a third.
//!
//! Neither would be a surprise, and that is the point of measuring rather than
//! asserting: both stories are plausible, they differ by a factor of three, and
//! only one of them makes buffer reuse the right next change.

use std::time::Instant;

use aether_core::scheduled::{dense_causal_block_schedule, BlockSchedule};
use aether_gpu::GpuContext;

const HEAD_DIM: usize = 16;
const BLOCK: usize = 8;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

/// Median wall time of `reps` calls, in milliseconds.
///
/// A median rather than a mean because this machine's per-call timings carry a
/// long right tail — recorded at length elsewhere in `FEATURES.md`, where a
/// matmul ratio moved by a factor of five between runs. A mean would report the
/// tail; the median reports the common case, and the comparison here is between
/// two medians measured in the same conditions.
fn median_ms(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut t = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        f();
        t.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    t.sort_by(f64::total_cmp);
    t[t.len() / 2]
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("No GPU adapter: {e}");
            std::process::exit(1);
        }
    };

    let info = ctx.adapter_info();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Forward against backward: dispatch count or transfer volume?");
    println!("  adapter {}  |  {}", info.name, info.backend);
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
    println!("  If dispatch count dominates, backward/forward is about 4.0.");
    println!("  If transfer volume dominates, about 1.3.");
    println!();
    println!(
        "  {:>5}  {:>11}  {:>11}  {:>8}  {:>11}  {:>9}",
        "seq", "forward ms", "backward ms", "ratio", "upload ms", "upload %"
    );
    println!(
        "  {:->5}  {:->11}  {:->11}  {:->8}  {:->11}  {:->9}",
        "", "", "", "", "", ""
    );

    for seq in [64usize, 128, 256, 512] {
        let span = seq * HEAD_DIM;
        let q = fill(span, 1);
        let k = fill(span, 2);
        let v = fill(span, 3);
        let d_out = fill(span, 4);
        let schedule: BlockSchedule = dense_causal_block_schedule(seq / BLOCK);

        // Warm the pipelines and the allocator before either measurement.
        let _ = ctx
            .scheduled_attention(&q, &k, &v, seq, HEAD_DIM, &schedule, BLOCK)
            .expect("warmup");

        let reps = if seq >= 256 { 20 } else { 40 };

        let forward = median_ms(reps, || {
            let _ = ctx
                .scheduled_attention(&q, &k, &v, seq, HEAD_DIM, &schedule, BLOCK)
                .expect("forward");
        });

        let backward = median_ms(reps, || {
            let (dq, _, _) = ctx
                .scheduled_attention_backward_resident(
                    &q, &k, &v, seq, HEAD_DIM, &schedule, BLOCK, &d_out,
                )
                .expect("backward");
            // Read one gradient so the batch is actually submitted. Without it
            // the recorded work sits in the pending encoder and the timing
            // measures how fast commands can be queued.
            let _ = ctx.read(&dq).expect("read");
        });

        // The upload alone, at the backward's operand size. Isolating it says
        // whether the transfer is the cost or merely the largest thing named in
        // the sentence that guessed at it.
        let packed = fill(span * 4 + seq * 3, 5);
        let upload = median_ms(reps, || {
            let _ = ctx.upload(&packed, 1, packed.len()).expect("upload");
        });

        println!(
            "  {seq:>5}  {forward:>11.3}  {backward:>11.3}  {:>7.2}x  {upload:>11.3}  {:>8.1}%",
            backward / forward,
            100.0 * upload / backward
        );
    }

    println!();
    println!("───────────────────────────────────────────────────────────────────────");
    println!("  'upload ms' allocates and fills a buffer the size of the backward's");
    println!("  packed operands and does nothing else, so it bounds what buffer");
    println!("  reuse could remove. A small percentage means the four dispatches");
    println!("  are the cost and caching buffers would be work spent on the wrong");
    println!("  half.");
    println!("═══════════════════════════════════════════════════════════════════════");
}
