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

/// Median of per-pair ratios, alternating the two measurements.
///
/// Measuring every repetition of A and then every repetition of B puts any
/// drift between the two blocks straight into the ratio. On a machine whose
/// clocks move — a laptop under load, which is what produced every figure in
/// this file — that is most of the noise, and it is a flaw in the measurement
/// rather than in the thing measured.
///
/// **This did not work, and that is the useful part.** It was written expecting
/// the interleaving to cancel drift within each pair. Measured across five runs
/// at n=512, the paired ratio spread 130% against the unpaired 96% — worse, not
/// better.
///
/// The reason is visible in `--samples`, which prints individual timings rather
/// than a median. Across six runs at n=512 the CPU term moved 1.6× and the GPU
/// term moved 5.2×, so the variance is almost entirely on one side. Pairing
/// cancels *common-mode* noise, and there is no common mode here — which is also
/// why it hurt: it puts the GPU's full variance into every ratio sample instead
/// of leaving it to be averaged down across a block.
///
/// Kept regardless. It is the correct design for estimating a ratio, and the
/// reason to hold it is that it is right rather than that it helped here;
/// reverting to a method known to be worse because a better one showed no gain
/// would be tuning the number rather than the measurement.
///
/// Returns `(median ratio, median a, median b)`.
fn paired_ratio(reps: usize, mut a: impl FnMut(), mut b: impl FnMut()) -> (f64, f64, f64) {
    let mut ratios = Vec::with_capacity(reps);
    let mut a_times = Vec::with_capacity(reps);
    let mut b_times = Vec::with_capacity(reps);

    for _ in 0..reps {
        let t0 = Instant::now();
        a();
        let ta = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        b();
        let tb = t1.elapsed().as_secs_f64() * 1000.0;

        ratios.push(ta / tb);
        a_times.push(ta);
        b_times.push(tb);
    }

    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };

    (med(&mut ratios), med(&mut a_times), med(&mut b_times))
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

/// Dump every individual timing at one size instead of a median.
///
/// The aggregate view cannot tell where the variance lives. Three attempts to
/// stabilise the ratio failed, and the conclusion that the noise is
/// short-timescale came from pairing not helping — an inference from a negative
/// result rather than an identification.
///
/// Raw samples answer it directly, and the answer was neither of the aggregation
/// fixes tried before: the CPU term is comparatively steady while the GPU term
/// swings by a factor of five, so the ratio inherits the GPU's variance and no
/// way of combining the two terms can remove it.
///
/// Reading the output: compare the two spread blocks. A run whose samples ramp
/// and then flatten is showing a warmup transient, and the settled tail is the
/// steady-state figure. A plateau ratio that agrees across runs while the
/// all-samples ratio does not means the transient was the problem; both
/// disagreeing means the variance is between runs and outside this process.
fn dump_samples(ctx: &GpuContext, n: usize, reps: usize) {
    let a64 = fill64(n * n, 1);
    let b64 = fill64(n * n, 2);
    let ta = Tensor::new(&a64, &[n, n]);
    let tb = Tensor::new(&b64, &[n, n]);

    let _ = tensor_matmul(ctx, &ta, &tb).expect("warmup");

    println!("raw samples at n={n}, {reps} alternating pairs");
    println!(
        "  {:>4}  {:>12}  {:>12}  {:>9}",
        "i", "Tensor ms", "bridge ms", "ratio"
    );

    let mut cpu = Vec::new();
    let mut gpu = Vec::new();

    for i in 0..reps {
        let t0 = Instant::now();
        let _ = ta.matmul(&tb);
        let c = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let _ = tensor_matmul(ctx, &ta, &tb).expect("bridge");
        let g = t1.elapsed().as_secs_f64() * 1000.0;

        println!("  {i:>4}  {c:>12.3}  {g:>12.3}  {:>8.2}x", c / g);
        cpu.push(c);
        gpu.push(g);
    }

    let stats = |v: &[f64]| -> (f64, f64, f64) {
        let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        (mn, mx, 100.0 * (mx - mn) / mean)
    };

    let (cmin, cmax, cspread) = stats(&cpu);
    let (gmin, gmax, gspread) = stats(&gpu);

    println!();
    println!("  all {reps} samples");
    println!("    Tensor  {cmin:.3} – {cmax:.3} ms   {cspread:.1}%");
    println!("    bridge  {gmin:.3} – {gmax:.3} ms   {gspread:.1}%");

    // This used to say the samples ramp and then flatten, and to report the last
    // third as the settled state. Both halves of that were wrong, and measuring
    // it said so. Means of the three thirds, one run: CPU 195.8, 216.0, 237.5 ms.
    // It does not flatten -- it is still climbing at the end -- and it climbs
    // rather than falls, so the last third is the slowest window, not the
    // settled one. Three runs back to back reproduced the direction every time.
    //
    // The drift is CPU-only. Across seven runs the CPU third-to-third drift was
    // positive every time, +1.5% to +21%. The bridge, once the machine is given
    // idle time, drifts +11.6, -7.4, +2.4, -5.0% over four runs -- the sign
    // varies, so within a run the bridge has no trend and that spread is noise.
    //
    // What the bridge does have is a second effect, visible only across runs and
    // invisible to any single one. Last-third means over three consecutive runs
    // with no gap: 6.16, 13.48, 20.97 ms. Ninety seconds of idle between the same
    // three: 5.74, 5.91, 7.51 ms. It degrades under back-to-back load and recovers
    // when left alone, so the earlier observation of a 4.9-25.7 ms spread was not
    // a noisy run, it was a run that came third. The first version of this code
    // told the reader to run it several times, which is how that state is reached.
    //
    // Both windows are printed rather than one. The first is a burst on a cold
    // machine, the last a sustained load on a warm one, and which a caller wants
    // depends on their workload. The drift between them is printed too, because
    // it is the part a single number hides.
    let tail = reps / 3;
    let (tcmin, tcmax, tcspread) = stats(&cpu[reps - tail..]);
    let (tgmin, tgmax, tgspread) = stats(&gpu[reps - tail..]);

    let (hcmin, hcmax, hcspread) = stats(&cpu[..tail]);
    let (hgmin, hgmax, hgspread) = stats(&gpu[..tail]);

    println!();
    println!("  first {tail} samples -- burst, cold");
    println!("    Tensor  {hcmin:.3} – {hcmax:.3} ms   {hcspread:.1}%");
    println!("    bridge  {hgmin:.3} – {hgmax:.3} ms   {hgspread:.1}%");
    println!();
    println!("  last {tail} samples -- sustained, hot");
    println!("    Tensor  {tcmin:.3} – {tcmax:.3} ms   {tcspread:.1}%");
    println!("    bridge  {tgmin:.3} – {tgmax:.3} ms   {tgspread:.1}%");

    let median = |lo: usize, hi: usize| -> f64 {
        let mut r: Vec<f64> = cpu[lo..hi]
            .iter()
            .zip(&gpu[lo..hi])
            .map(|(c, g)| c / g)
            .collect();
        r.sort_by(f64::total_cmp);
        r[r.len() / 2]
    };
    let head_ratio = median(0, tail);
    let tail_ratio = median(reps - tail, reps);
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let cpu_drift = 100.0 * (mean(&cpu[reps - tail..]) / mean(&cpu[..tail]) - 1.0);
    let gpu_drift = 100.0 * (mean(&gpu[reps - tail..]) / mean(&gpu[..tail]) - 1.0);

    println!();
    println!("  ratio   cold {head_ratio:.2}x   hot {tail_ratio:.2}x");
    println!("  drift   Tensor {cpu_drift:+.1}%   bridge {gpu_drift:+.1}%");
    println!();
    println!("  Tensor drift ran positive in 7 of 7 runs, so the CPU is slower");
    println!("  late in a run and neither window is a plateau. Bridge drift");
    println!("  changed sign across runs, so treat it as noise, not a trend.");
    println!();
    println!("  Leave 90 s idle before rerunning. Three back-to-back runs took");
    println!("  the bridge from 6.16 to 20.97 ms in this window; the same three");
    println!("  spaced out stayed near 6. Rerunning immediately measures that,");
    println!("  not the kernel.");
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("No GPU adapter: {e}");
            std::process::exit(1);
        }
    };

    if std::env::args().any(|a| a == "--samples") {
        dump_samples(&ctx, 512, 24);
        return;
    }

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

        // Paired: alternate the CPU and the bridge so each pair sees the same
        // thermal state, and take the median of the per-pair ratios. This is
        // the figure the recommendation rests on.
        let (ratio_bridge, _, bridge) = paired_ratio(
            reps,
            || {
                let _ = ta.matmul(&tb);
            },
            || {
                let _ = tensor_matmul(&ctx, &ta, &tb).expect("bridge");
            },
        );
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
