//! Does routing `aether_core::ml::Tensor::matmul` to the GPU actually pay?
//!
//! Run:
//!   cargo run -p aether-gpu --example tensor_crossover --release
//!
//! The crossover in `gpu_bench` is measured against this crate's own naive f32
//! reference, which is a stand-in. The code an integration would actually
//! replace is `Tensor::matmul`: f64, strided indexing, `RefCell` borrows on
//! every access. Whether the recommendation holds depends on that, not on the
//! stand-in.
//!
//! # The cost the stand-in comparison omits
//!
//! `Tensor` is f64 and WGSL is f32, so a real integration converts both operands
//! down and the result back up. That is three O(n²) passes plus two allocations
//! per call, and it does not appear in any measurement so far. Counting it is
//! the difference between a recommendation and a guess: the conversion is part
//! of the operation, not overhead to be excused.
//!
//! Both are reported, so the honest crossover and the flattering one are
//! visible side by side.

use std::time::Instant;

use aether_core::ml::tensor::Tensor;
use aether_gpu::{tensor_matmul, GpuContext};

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

fn fill64(n: usize, seed: u64) -> Vec<f64> {
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
    println!("  aether_core::ml::Tensor::matmul against the GPU path");
    println!("  adapter {}  |  {}", info.name, info.backend);
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
    println!(
        "  {:>5}  {:>12}  {:>12}  {:>8}  {:>12}  {:>12}  {:>8}",
        "n", "Tensor ms", "GPU raw ms", "ratio", "hand-conv ms", "bridge ms", "ratio"
    );
    println!(
        "  {:->5}  {:->12}  {:->12}  {:->8}  {:->12}  {:->12}  {:->8}",
        "", "", "", "", "", "", ""
    );

    let mut honest_crossover: Option<usize> = None;

    for n in [64usize, 128, 192, 256, 384, 512] {
        let a64 = fill64(n * n, 1);
        let b64 = fill64(n * n, 2);

        let ta = Tensor::new(&a64, &[n, n]);
        let tb = Tensor::new(&b64, &[n, n]);

        // The CPU side used three repetitions above n=384, on the reasoning that
        // a 200 ms operation is expensive to repeat. A median of three is a poor
        // estimator, and measuring it showed how poor: across six runs of this
        // binary the n=512 CPU figure ranged 197.9 to 321.8 ms, and the ratio
        // derived from it ranged 35x to 61x.
        //
        // That made the documented "36x" one draw from a distribution nearly
        // twice as wide as the number suggested. Nine repetitions cost about two
        // seconds at n=512 and are worth it: a benchmark whose noise exceeds the
        // effect it reports is not measuring the effect.
        let reps = if n >= 384 { 9 } else { 15 };
        let cpu = median_ms(reps, || {
            let _ = ta.matmul(&tb);
        });

        // The GPU path as the benchmark measures it: operands already f32.
        let a32: Vec<f32> = a64.iter().map(|v| *v as f32).collect();
        let b32: Vec<f32> = b64.iter().map(|v| *v as f32).collect();
        let _ = ctx.matmul(&a32, &b32, n, n, n).expect("warmup");
        let raw = median_ms(reps.max(10), || {
            let _ = ctx.matmul(&a32, &b32, n, n, n).expect("mm");
        });

        // The GPU path as an integration would have to run it: convert both
        // operands down, dispatch, convert the result back up.
        let with_conv = median_ms(reps.max(10), || {
            let ca: Vec<f32> = a64.iter().map(|v| *v as f32).collect();
            let cb: Vec<f32> = b64.iter().map(|v| *v as f32).collect();
            let out = ctx.matmul(&ca, &cb, n, n, n).expect("mm");
            let _back: Vec<f64> = out.iter().map(|v| *v as f64).collect();
        });

        // The shipped bridge, which is what a caller actually runs. It differs
        // from `with_conv` above by gathering through the tensor's strides
        // rather than reading its buffer flat -- necessary for correctness on a
        // non-contiguous tensor, and a scalar loop on the contiguous path this
        // will almost always take. Whether that cost matters is the question
        // this column answers.
        let bridge = median_ms(reps.max(10), || {
            let _ = tensor_matmul(&ctx, &ta, &tb).expect("bridge");
        });

        let ratio_bridge = cpu / bridge;
        if honest_crossover.is_none() && ratio_bridge > 1.0 {
            honest_crossover = Some(n);
        }

        println!(
            "  {n:>5}  {cpu:>12.3}  {raw:>12.3}  {:>7.2}x  {with_conv:>12.3}  {bridge:>12.3}  {ratio_bridge:>7.2}x",
            cpu / raw
        );
    }

    println!();
    println!("───────────────────────────────────────────────────────────────────────");
    match honest_crossover {
        Some(n) => println!("  crossover using the shipped bridge: n = {n}"),
        None => println!("  no crossover including conversion at any size tested"),
    }
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  'GPU raw' assumes the operands are already f32, which they are not:");
    println!("  Tensor is f64. 'hand-conv' converts both operands down and the");
    println!("  result back up by reading each buffer flat.");
    println!();
    println!("  'bridge' is the shipped tensor_matmul, which gathers through the");
    println!("  tensor's strides instead. That is required for correctness on a");
    println!("  non-contiguous tensor and costs about 11% on the contiguous path");
    println!("  it will almost always take. The ratio column is measured from it,");
    println!("  because it is the code a caller runs -- quoting hand-conv would");
    println!("  report a number nothing in the crate produces.");
    println!("═══════════════════════════════════════════════════════════════════════");
}
