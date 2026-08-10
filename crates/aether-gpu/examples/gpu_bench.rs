//! Measures what the resident-tensor and tiling changes actually bought.
//!
//! Run:
//!   cargo run -p aether-gpu --example gpu_bench --release
//!
//! Three paths over identical inputs:
//!
//!   round-trip naive   upload -> naive kernel -> read back, per call
//!   round-trip tiled   same transfers, tiled kernel
//!   resident tiled     upload once, chain on device, read once
//!
//! The gap between the first two is the kernel. The gap between the last two is
//! the bus. Separating them matters because they are fixed by different work,
//! and a single "GPU is faster now" number hides which one was the problem.

use std::time::Instant;

use aether_gpu::{cpu_matmul, cpu_pairwise_sqdist, GpuContext};

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

/// Median of `reps` timings. Median rather than mean because the first
/// dispatch after an allocation is reliably slower and a mean lets that one
/// sample set the number.
fn median_ms(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
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
    println!(
        "  adapter {}  |  {}  |  {}",
        info.name, info.backend, info.device_type
    );
    println!("  all timings are medians; f32; single run of this binary");
    println!("═══════════════════════════════════════════════════════════════════════");

    // ── Square matmul ─────────────────────────────────────────────────────────
    println!();
    println!("  Square matmul, median of 20");
    println!(
        "  {:>5}  {:>12}  {:>12}  {:>12}  {:>12}",
        "size", "CPU ms", "rt-naive ms", "rt-tiled ms", "resident ms"
    );
    println!(
        "  {:->5}  {:->12}  {:->12}  {:->12}  {:->12}",
        "", "", "", "", ""
    );

    for size in [64usize, 128, 256, 512, 1024] {
        let a = fill(size * size, 1);
        let b = fill(size * size, 2);

        // CPU gets fewer reps at 1024: 1024^3 is ~1.07e9 multiply-adds and the
        // naive loop takes seconds. Timing it 20 times would dominate the run.
        let cpu_reps = if size >= 512 { 1 } else { 3 };
        let cpu = median_ms(cpu_reps, || {
            let _ = cpu_matmul(&a, &b, size, size, size);
        });

        let _ = ctx.matmul(&a, &b, size, size, size).expect("warmup");
        let rt_naive = median_ms(20, || {
            let _ = ctx.matmul(&a, &b, size, size, size).expect("rt naive");
        });

        let ga = ctx.upload(&a, size, size).expect("upload");
        let gb = ctx.upload(&b, size, size).expect("upload");

        // Round-trip tiled: pay upload and readback every call.
        let rt_tiled = median_ms(20, || {
            let ua = ctx.upload(&a, size, size).expect("upload");
            let ub = ctx.upload(&b, size, size).expect("upload");
            let c = ctx.matmul_resident(&ua, &ub).expect("tiled");
            let _ = ctx.read(&c).expect("read");
        });

        // Resident: operands already on device, one readback.
        let resident = median_ms(20, || {
            let c = ctx.matmul_resident(&ga, &gb).expect("tiled");
            let _ = ctx.read(&c).expect("read");
        });

        println!("  {size:>5}  {cpu:>12.3}  {rt_naive:>12.3}  {rt_tiled:>12.3}  {resident:>12.3}");
    }

    // ── Chained matmuls: where residency actually pays ────────────────────────
    println!();
    println!("  Chain of 8 matmuls at 256x256, median of 10");

    let size = 256;
    let a = fill(size * size, 3);
    let w = fill(size * size, 4);

    let chained_rt = median_ms(10, || {
        let mut cur = a.clone();
        for _ in 0..8 {
            cur = ctx.matmul(&cur, &w, size, size, size).expect("rt");
        }
    });

    let gw = ctx.upload(&w, size, size).expect("upload");
    let chained_res = median_ms(10, || {
        let mut cur = ctx.upload(&a, size, size).expect("upload");
        for _ in 0..8 {
            cur = ctx.matmul_resident(&cur, &gw).expect("resident");
        }
        let _ = ctx.read(&cur).expect("read");
    });

    println!("  round-trip every step   {chained_rt:>10.3} ms");
    println!("  resident, one readback  {chained_res:>10.3} ms");
    println!(
        "  speedup                 {:>10.2}x",
        chained_rt / chained_res
    );

    // ── Pairwise distance: the op this repository actually needs ──────────────
    println!();
    println!("  Pairwise squared distance, [n, 3] cloud, median of 10");
    println!(
        "  {:>6}  {:>12}  {:>12}  {:>8}  {:>12}",
        "n", "CPU ms", "GPU ms", "ratio", "worst diff"
    );
    println!(
        "  {:->6}  {:->12}  {:->12}  {:->8}  {:->12}",
        "", "", "", "", ""
    );

    for n in [256usize, 512, 1024, 2048] {
        let d = 3;
        let pts = fill(n * d, 5);

        let cpu_reps = if n >= 1024 { 1 } else { 5 };
        let cpu = median_ms(cpu_reps, || {
            let _ = cpu_pairwise_sqdist(&pts, n, d);
        });

        let g = ctx.upload(&pts, n, d).expect("upload");
        let _ = ctx.pairwise_sqdist_resident(&g).expect("warmup");

        let gpu = median_ms(10, || {
            let m = ctx.pairwise_sqdist_resident(&g).expect("sqdist");
            let _ = ctx.read(&m).expect("read");
        });

        let gm = ctx.pairwise_sqdist_resident(&g).expect("sqdist");
        let got = ctx.read(&gm).expect("read");
        let want = cpu_pairwise_sqdist(&pts, n, d);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);

        println!(
            "  {n:>6}  {cpu:>12.3}  {gpu:>12.3}  {:>7.2}x  {worst:>12.3e}",
            cpu / gpu
        );
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  CPU columns are the naive single-threaded reference this crate ships");
    println!("  as its parity baseline. Not a tuned BLAS. The ratios compare this");
    println!("  crate's own two paths and are not a GPU-versus-optimised-CPU claim.");
    println!("═══════════════════════════════════════════════════════════════════════");
}
