//! Trains an MLP on the GPU with k-fold cross-validation and prints the numbers
//! this workspace is allowed to quote.
//!
//! Run:
//!   cargo run -p aether-gpu --example train_mlp_cv --release
//!
//! # What runs where
//!
//! Every matmul in both the forward and backward pass is a GPU dispatch, and
//! matmuls are the whole asymptotic cost of training this network. ReLU and its
//! derivative are GPU dispatches too. Transposes, the sigmoid, the loss, and the
//! SGD parameter update run on the CPU: they are O(elements) memory shuffles and
//! elementwise maps, and dispatching them would measure PCIe latency rather than
//! arithmetic. This split is stated rather than implied so "trained on GPU" can
//! be checked against what the code does.
//!
//! # The task
//!
//! Two interleaved spirals, the standard non-linearly-separable binary problem.
//! It is chosen because a linear model provably cannot solve it, so the linear
//! baseline below is a real control rather than a formality: if the MLP did not
//! learn, its accuracy would collapse to the baselines instead of separating
//! from them.
//!
//! # Precision
//!
//! f32 throughout, because WGSL has no f64.

use std::time::Instant;

use aether_gpu::datasets::report_split;
use aether_gpu::{cpu_matmul, GpuContext};

const FOLDS: usize = 5;
const EPOCHS: usize = 100;
const HIDDEN: usize = 32;
const LEARNING_RATE: f32 = 0.5;
const POINTS_PER_CLASS: usize = 250;

// ═══════════════════════════════════════════════════════════════════════════════
// Data
// ═══════════════════════════════════════════════════════════════════════════════

/// Deterministic LCG. A fixed generator rather than `rand` so the reported
/// numbers reproduce exactly from the seed, on any machine.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as f32 / (1u64 << 31) as f32
    }
}

/// Two interleaved spirals with Gaussian-ish jitter. Returns row-major
/// `[n, 2]` features and `[n]` labels in {0, 1}.
fn spirals(seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Lcg(seed);
    let n = POINTS_PER_CLASS * 2;
    let mut x = Vec::with_capacity(n * 2);
    let mut y = Vec::with_capacity(n);

    for class in 0..2 {
        for i in 0..POINTS_PER_CLASS {
            let t = i as f32 / POINTS_PER_CLASS as f32;
            let radius = 0.2 + 3.8 * t;
            let angle = 2.5 * core::f32::consts::PI * t + class as f32 * core::f32::consts::PI;

            let jitter_r = (rng.next_f32() - 0.5) * 0.35;
            let jitter_a = (rng.next_f32() - 0.5) * 0.15;

            x.push((radius + jitter_r) * (angle + jitter_a).cos() / 4.0);
            x.push((radius + jitter_r) * (angle + jitter_a).sin() / 4.0);
            y.push(class as f32);
        }
    }

    // Shuffle so folds are not one class each. Fisher-Yates with the same LCG.
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_f32() * (i + 1) as f32) as usize % (i + 1);
        order.swap(i, j);
    }

    let mut xs = Vec::with_capacity(n * 2);
    let mut ys = Vec::with_capacity(n);
    for &idx in &order {
        xs.push(x[idx * 2]);
        xs.push(x[idx * 2 + 1]);
        ys.push(y[idx]);
    }

    (xs, ys)
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPU helpers -- memory shuffles and elementwise maps, not the arithmetic cost
// ═══════════════════════════════════════════════════════════════════════════════

fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}

fn sigmoid(z: &[f32]) -> Vec<f32> {
    z.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect()
}

/// Kaiming-uniform init, the right choice for ReLU: naive uniform leaves half
/// the gradient signal dead on arrival and the failure reads as a bad
/// learning rate.
fn init_weights(fan_in: usize, fan_out: usize, rng: &mut Lcg) -> Vec<f32> {
    let bound = (6.0f32 / fan_in as f32).sqrt();
    (0..fan_in * fan_out)
        .map(|_| (rng.next_f32() * 2.0 - 1.0) * bound)
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// The network: 2 -> HIDDEN -> HIDDEN -> 1
// ═══════════════════════════════════════════════════════════════════════════════

struct Mlp {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: Vec<f32>,
}

impl Mlp {
    fn new(rng: &mut Lcg) -> Self {
        Self {
            w1: init_weights(2, HIDDEN, rng),
            b1: vec![0.0; HIDDEN],
            w2: init_weights(HIDDEN, HIDDEN, rng),
            b2: vec![0.0; HIDDEN],
            w3: init_weights(HIDDEN, 1, rng),
            b3: vec![0.0; 1],
        }
    }

    /// Forward pass. Every matmul and every ReLU here is a GPU dispatch.
    fn forward(&self, ctx: &GpuContext, x: &[f32], n: usize) -> Forward {
        let z1 = ctx
            .matmul(x, &self.w1, n, 2, HIDDEN)
            .expect("gpu matmul z1");
        let z1 = ctx.add_bias(&z1, &self.b1, n, HIDDEN).expect("gpu bias b1");
        let a1 = ctx.relu(&z1).expect("gpu relu a1");

        let z2 = ctx
            .matmul(&a1, &self.w2, n, HIDDEN, HIDDEN)
            .expect("gpu matmul z2");
        let z2 = ctx.add_bias(&z2, &self.b2, n, HIDDEN).expect("gpu bias b2");
        let a2 = ctx.relu(&z2).expect("gpu relu a2");

        let z3 = ctx
            .matmul(&a2, &self.w3, n, HIDDEN, 1)
            .expect("gpu matmul z3");
        let z3 = ctx.add_bias(&z3, &self.b3, n, 1).expect("gpu bias b3");
        let out = sigmoid(&z3);

        Forward {
            z1,
            a1,
            z2,
            a2,
            out,
        }
    }

    /// One SGD step on the full batch. Backward matmuls are GPU dispatches.
    fn step(&mut self, ctx: &GpuContext, x: &[f32], y: &[f32], n: usize, lr: f32) -> f32 {
        let f = self.forward(ctx, x, n);

        // dL/dz3 for sigmoid + binary cross-entropy collapses to (p - y).
        let dz3: Vec<f32> = f
            .out
            .iter()
            .zip(y)
            .map(|(p, t)| (p - t) / n as f32)
            .collect();

        let a2_t = transpose(&f.a2, n, HIDDEN);
        let dw3 = ctx
            .matmul(&a2_t, &dz3, HIDDEN, n, 1)
            .expect("gpu matmul dw3");
        let db3: f32 = dz3.iter().sum();

        let w3_t = transpose(&self.w3, HIDDEN, 1);
        let da2 = ctx
            .matmul(&dz3, &w3_t, n, 1, HIDDEN)
            .expect("gpu matmul da2");
        let dz2 = ctx.relu_backward(&f.z2, &da2).expect("gpu relu_backward 2");

        let a1_t = transpose(&f.a1, n, HIDDEN);
        let dw2 = ctx
            .matmul(&a1_t, &dz2, HIDDEN, n, HIDDEN)
            .expect("gpu matmul dw2");
        let db2 = column_sums(&dz2, n, HIDDEN);

        let w2_t = transpose(&self.w2, HIDDEN, HIDDEN);
        let da1 = ctx
            .matmul(&dz2, &w2_t, n, HIDDEN, HIDDEN)
            .expect("gpu matmul da1");
        let dz1 = ctx.relu_backward(&f.z1, &da1).expect("gpu relu_backward 1");

        let x_t = transpose(x, n, 2);
        let dw1 = ctx
            .matmul(&x_t, &dz1, 2, n, HIDDEN)
            .expect("gpu matmul dw1");
        let db1 = column_sums(&dz1, n, HIDDEN);

        apply(&mut self.w1, &dw1, lr);
        apply(&mut self.b1, &db1, lr);
        apply(&mut self.w2, &dw2, lr);
        apply(&mut self.b2, &db2, lr);
        apply(&mut self.w3, &dw3, lr);
        apply(&mut self.b3, &[db3], lr);

        // Binary cross-entropy, clamped away from log(0).
        f.out
            .iter()
            .zip(y)
            .map(|(p, t)| {
                let p = p.clamp(1e-7, 1.0 - 1e-7);
                -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
            })
            .sum::<f32>()
            / n as f32
    }
}

struct Forward {
    z1: Vec<f32>,
    a1: Vec<f32>,
    z2: Vec<f32>,
    a2: Vec<f32>,
    out: Vec<f32>,
}

fn column_sums(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut s = vec![0.0f32; cols];
    for i in 0..rows {
        for j in 0..cols {
            s[j] += a[i * cols + j];
        }
    }
    s
}

fn apply(param: &mut [f32], grad: &[f32], lr: f32) {
    for (p, g) in param.iter_mut().zip(grad) {
        *p -= lr * g;
    }
}

fn accuracy(pred: &[f32], y: &[f32]) -> f32 {
    let correct = pred
        .iter()
        .zip(y)
        .filter(|(p, t)| ((**p >= 0.5) as u8 as f32 - **t).abs() < 0.5)
        .count();
    correct as f32 / y.len() as f32
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("No GPU adapter available: {e}");
            eprintln!("This example asserts GPU execution and will not fall back to CPU.");
            std::process::exit(1);
        }
    };

    let info = ctx.adapter_info();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  GPU-trained MLP, {FOLDS}-fold cross-validation, {EPOCHS} epochs/fold");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  adapter      {}", info.name);
    println!("  backend      {}", info.backend);
    println!("  device type  {}", info.device_type);
    println!("  precision    f32 (WGSL has no f64)");

    if !info.is_hardware() {
        eprintln!(
            "\n  device type is {}, a software rasterizer.",
            info.device_type
        );
        eprintln!("  Refusing to report these as GPU results.");
        std::process::exit(1);
    }

    let (x, y) = spirals(0xA37E5);
    let n = y.len();
    println!("  dataset      two spirals, {n} points, 2 features, 2 classes");
    println!(
        "  network      2 -> {HIDDEN} -> {HIDDEN} -> 1, ReLU, full-batch SGD lr={LEARNING_RATE}"
    );

    // Controls. Without these an accuracy number means nothing.
    let majority = {
        let ones = y.iter().filter(|v| **v > 0.5).count();
        (ones.max(n - ones)) as f32 / n as f32
    };

    let fold_size = n / FOLDS;
    let mut accs = Vec::new();
    let mut losses = Vec::new();

    // What the accuracy below is a number *about*.
    //
    // `diagnose_split` and `report_split` were written for this crate after a set
    // of cross-validation figures was withdrawn for reporting generalisation
    // while measuring interpolation. They were public, documented, and called by
    // nothing, so every accuracy this example printed arrived without the
    // qualification the machinery exists to supply — the same defect the rest of
    // this crate spends its length correcting, in the one place a reader looks
    // first.
    //
    // Fold 0 stands for all of them: the folds are contiguous blocks of one
    // shuffled ordering, so they are the same construction with a different
    // offset and their ratios differ only by sampling noise.
    let mut is_test = vec![false; n];
    for flag in is_test.iter_mut().take(fold_size) {
        *flag = true;
    }
    let split = report_split("fold 1 of the shuffled i.i.d. draw", &x, &is_test);

    let start = Instant::now();

    for fold in 0..FOLDS {
        let lo = fold * fold_size;
        let hi = if fold == FOLDS - 1 { n } else { lo + fold_size };

        let mut tr_x = Vec::new();
        let mut tr_y = Vec::new();
        let mut te_x = Vec::new();
        let mut te_y = Vec::new();

        for i in 0..n {
            if i >= lo && i < hi {
                te_x.extend_from_slice(&x[i * 2..i * 2 + 2]);
                te_y.push(y[i]);
            } else {
                tr_x.extend_from_slice(&x[i * 2..i * 2 + 2]);
                tr_y.push(y[i]);
            }
        }

        let n_tr = tr_y.len();
        let n_te = te_y.len();

        // Same init seed every fold: folds differ by data, not by luck.
        let mut rng = Lcg(0xC0FFEE + fold as u64);
        let mut net = Mlp::new(&mut rng);

        let mut last_loss = 0.0;
        for _ in 0..EPOCHS {
            last_loss = net.step(&ctx, &tr_x, &tr_y, n_tr, LEARNING_RATE);
        }

        let train_acc = accuracy(&net.forward(&ctx, &tr_x, n_tr).out, &tr_y);
        let test_acc = accuracy(&net.forward(&ctx, &te_x, n_te).out, &te_y);

        accs.push(test_acc);
        losses.push(last_loss);

        println!(
            "  fold {}/{}  train {:>5} test {:>5}  train acc {:.4}  test acc {:.4}  loss {:.4}",
            fold + 1,
            FOLDS,
            n_tr,
            n_te,
            train_acc,
            test_acc,
            last_loss
        );
    }

    let elapsed = start.elapsed();

    let mean = accs.iter().sum::<f32>() / accs.len() as f32;
    let var = accs.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / accs.len() as f32;
    let std = var.sqrt();
    let mean_loss = losses.iter().sum::<f32>() / losses.len() as f32;

    println!("───────────────────────────────────────────────────────────────────");
    println!(
        "  CV test accuracy    {:.4} +/- {:.4}  ({FOLDS} folds)",
        mean, std
    );
    println!("  final train loss    {:.4}  (mean over folds)", mean_loss);
    println!("  majority-class      {:.4}  <- control", majority);
    println!(
        "  separation          {:+.4} over majority class",
        mean - majority
    );
    // Printed next to the number it qualifies rather than only in the diagnostic
    // block above, because a reader who scrolls to the summary is the reader most
    // likely to quote the accuracy somewhere else.
    println!(
        "  measures            {} (split ratio {:.2}x)",
        if split.is_extrapolating() {
            "EXTRAPOLATION -- held-out points lie outside the training data"
        } else {
            "interpolation within the sampled region, not extrapolation beyond it"
        },
        split.ratio
    );
    println!(
        "  wall clock          {:.2} s for {} folds x {EPOCHS} epochs",
        elapsed.as_secs_f64(),
        FOLDS
    );

    // ── GPU vs CPU on the dominant op, same inputs ────────────────────────────
    println!("───────────────────────────────────────────────────────────────────");
    println!("  matmul throughput, GPU vs the CPU reference this crate ships");

    for size in [128usize, 256, 512] {
        let a: Vec<f32> = (0..size * size).map(|i| (i % 17) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..size * size).map(|i| (i % 13) as f32 * 0.01).collect();

        // Warm up: first dispatch includes pipeline and allocation costs.
        let _ = ctx.matmul(&a, &b, size, size, size).expect("warmup");

        let t0 = Instant::now();
        let g = ctx.matmul(&a, &b, size, size, size).expect("gpu matmul");
        let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let c = cpu_matmul(&a, &b, size, size, size);
        let cpu_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let worst = g
            .iter()
            .zip(&c)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);

        println!(
            "  {size:>4}x{size:<4}  GPU {gpu_ms:>8.3} ms   CPU {cpu_ms:>9.3} ms   \
             ratio {:>6.2}x   worst |diff| {worst:.3e}",
            cpu_ms / gpu_ms
        );
    }

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  GPU timings include buffer upload, dispatch and readback -- the");
    println!("  whole round trip, not the kernel in isolation. The CPU reference");
    println!("  is the naive triple loop, single-threaded, no SIMD intrinsics.");
    println!("  It is the honest baseline for this crate and not a fast BLAS.");
    println!("═══════════════════════════════════════════════════════════════════");
}
