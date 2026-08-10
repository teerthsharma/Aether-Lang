//! Fully-resident GPU training. Weights never leave the device.
//!
//! Run:
//!   cargo run -p aether-gpu --example train_resident --release
//!
//! # What changed against `train_mlp_cv`
//!
//! That example dispatches to the GPU but round-trips every intermediate
//! through host memory, so a training step pays a bus crossing per operation to
//! move data the GPU produced and is about to consume.
//!
//! Here the whole step is resident: inputs and weights upload once per fold,
//! forward and backward chain on the device, the SGD update runs there too, and
//! the only readback per epoch is a scalar loss. Weights stay in device memory
//! for all 100 epochs.
//!
//! Both are kept. The comparison between them is the measurement, and deleting
//! the slower path would delete the baseline.
//!
//! # Precision
//!
//! f32. WGSL has no f64.

use std::time::Instant;

use aether_gpu::{datasets, GpuContext, GpuTensor};

const RUNS: usize = 5;
const EPOCHS: usize = 100;
const HIDDEN: usize = 32;
const LEARNING_RATE: f32 = 0.5;
const POINTS_PER_CLASS: usize = 250;

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

fn init_weights(fan_in: usize, fan_out: usize, rng: &mut Lcg) -> Vec<f32> {
    let bound = (6.0f32 / fan_in as f32).sqrt();
    (0..fan_in * fan_out)
        .map(|_| (rng.next_f32() * 2.0 - 1.0) * bound)
        .collect()
}

/// Parameters, resident for the lifetime of a run.
struct ResidentMlp {
    w1: GpuTensor,
    b1: GpuTensor,
    w2: GpuTensor,
    b2: GpuTensor,
    w3: GpuTensor,
    b3: GpuTensor,
}

impl ResidentMlp {
    fn new(ctx: &GpuContext, rng: &mut Lcg) -> Self {
        let up = |v: Vec<f32>, r, c| ctx.upload(&v, r, c).expect("upload param");
        Self {
            w1: up(init_weights(2, HIDDEN, rng), 2, HIDDEN),
            b1: up(vec![0.0; HIDDEN], 1, HIDDEN),
            w2: up(init_weights(HIDDEN, HIDDEN, rng), HIDDEN, HIDDEN),
            b2: up(vec![0.0; HIDDEN], 1, HIDDEN),
            w3: up(init_weights(HIDDEN, 1, rng), HIDDEN, 1),
            b3: up(vec![0.0; 1], 1, 1),
        }
    }

    /// Returns the pre-activations needed by the backward pass plus the output
    /// logits. Nothing here crosses the bus.
    fn forward(
        &self,
        ctx: &GpuContext,
        x: &GpuTensor,
    ) -> (GpuTensor, GpuTensor, GpuTensor, GpuTensor, GpuTensor) {
        let z1 = ctx.matmul_resident(x, &self.w1).expect("z1");
        let z1 = ctx.add_bias_resident(&z1, &self.b1).expect("z1 bias");
        let a1 = ctx.relu_resident(&z1).expect("a1");

        let z2 = ctx.matmul_resident(&a1, &self.w2).expect("z2");
        let z2 = ctx.add_bias_resident(&z2, &self.b2).expect("z2 bias");
        let a2 = ctx.relu_resident(&z2).expect("a2");

        let z3 = ctx.matmul_resident(&a2, &self.w3).expect("z3");
        let z3 = ctx.add_bias_resident(&z3, &self.b3).expect("z3 bias");

        (z1, a1, z2, a2, z3)
    }

    /// One full-batch SGD step, entirely on the device.
    fn step(&mut self, ctx: &GpuContext, x: &GpuTensor, y: &GpuTensor, lr: f32) {
        let (z1, a1, z2, a2, z3) = self.forward(ctx, x);

        let dz3 = ctx.sigmoid_bce_grad_resident(&z3, y).expect("dz3");

        let a2t = ctx.transpose_resident(&a2).expect("a2t");
        let dw3 = ctx.matmul_resident(&a2t, &dz3).expect("dw3");
        let db3 = ctx.column_sums_resident(&dz3).expect("db3");

        let w3t = ctx.transpose_resident(&self.w3).expect("w3t");
        let da2 = ctx.matmul_resident(&dz3, &w3t).expect("da2");
        let dz2 = ctx.relu_backward_resident(&z2, &da2).expect("dz2");

        let a1t = ctx.transpose_resident(&a1).expect("a1t");
        let dw2 = ctx.matmul_resident(&a1t, &dz2).expect("dw2");
        let db2 = ctx.column_sums_resident(&dz2).expect("db2");

        let w2t = ctx.transpose_resident(&self.w2).expect("w2t");
        let da1 = ctx.matmul_resident(&dz2, &w2t).expect("da1");
        let dz1 = ctx.relu_backward_resident(&z1, &da1).expect("dz1");

        let xt = ctx.transpose_resident(x).expect("xt");
        let dw1 = ctx.matmul_resident(&xt, &dz1).expect("dw1");
        let db1 = ctx.column_sums_resident(&dz1).expect("db1");

        self.w1 = ctx.sgd_update_resident(&self.w1, &dw1, lr).expect("w1");
        self.b1 = ctx.sgd_update_resident(&self.b1, &db1, lr).expect("b1");
        self.w2 = ctx.sgd_update_resident(&self.w2, &dw2, lr).expect("w2");
        self.b2 = ctx.sgd_update_resident(&self.b2, &db2, lr).expect("b2");
        self.w3 = ctx.sgd_update_resident(&self.w3, &dw3, lr).expect("w3");
        self.b3 = ctx.sgd_update_resident(&self.b3, &db3, lr).expect("b3");

        // Submit once per step, not once per operation and not once per fold.
        //
        // Per operation costs a driver transition twenty times a step. Per fold
        // is worse: the whole run records into one encoder, so the GPU idles
        // while the CPU builds two thousand dispatches and every intermediate
        // buffer from every epoch stays resident until the end. Measured at
        // 0.65 s against 0.48 s for the unbatched version -- batching without a
        // flush boundary is a regression, not an optimisation.
        ctx.flush();
    }

    fn predict(&self, ctx: &GpuContext, x: &GpuTensor) -> Vec<f32> {
        let (_, _, _, _, z3) = self.forward(ctx, x);
        let p = ctx.sigmoid_resident(&z3).expect("sigmoid");
        ctx.read(&p).expect("readback")
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

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("No GPU adapter: {e}");
            std::process::exit(1);
        }
    };

    let info = ctx.adapter_info();
    if !info.is_hardware() {
        eprintln!(
            "adapter is {} -- refusing to report software as GPU",
            info.device_type
        );
        std::process::exit(1);
    }

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Fully-resident GPU training, {RUNS} seeds, {EPOCHS} epochs/fold");
    println!("═══════════════════════════════════════════════════════════════════");
    println!(
        "  adapter      {}  |  {}  |  {}",
        info.name, info.backend, info.device_type
    );
    println!("  residency    weights stay on device for all {EPOCHS} epochs");
    println!("  readback     final predictions only");

    // Train and test are independent draws from the same generator.
    //
    // This example previously cross-validated a single swept sample and printed
    // 0.8220. That split dealt consecutive arc points into different folds, so
    // every held-out point had a training point 0.99x the cloud's own spacing
    // away; the figure measured interpolation between near-duplicates rather
    // than generalisation. See aether_gpu::datasets for the diagnostic.
    let (x, y_class) = datasets::spirals_iid(0xA37E5, 2, POINTS_PER_CLASS);
    let (test_x, test_class) = datasets::spirals_iid(0xA37E6, 2, POINTS_PER_CLASS);
    let y: Vec<f32> = y_class.iter().map(|c| *c as f32).collect();
    let test_y: Vec<f32> = test_class.iter().map(|c| *c as f32).collect();
    let (n, n_te) = (y.len(), test_y.len());

    let mut combined = x.clone();
    combined.extend_from_slice(&test_x);
    let is_test: Vec<bool> = (0..n)
        .map(|_| false)
        .chain((0..n_te).map(|_| true))
        .collect();
    datasets::report_split("independent draws", &combined, &is_test);
    println!();

    let mut accs = Vec::new();
    let start = Instant::now();

    // Uploaded once. Nothing crosses the bus again until predict().
    let gx = ctx.upload(&x, n, 2).expect("upload x");
    let gy = ctx.upload(&y, n, 1).expect("upload y");
    let gtex = ctx.upload(&test_x, n_te, 2).expect("upload test x");

    for run in 0..RUNS {
        let mut rng = Lcg(0xC0FFEE + run as u64);
        let mut net = ResidentMlp::new(&ctx, &mut rng);

        for _ in 0..EPOCHS {
            net.step(&ctx, &gx, &gy, LEARNING_RATE);
        }

        let train_acc = accuracy(&net.predict(&ctx, &gx), &y);
        let test_acc = accuracy(&net.predict(&ctx, &gtex), &test_y);
        accs.push(test_acc);

        println!(
            "  seed {}/{}  train acc {:.4}  test acc {:.4}",
            run + 1,
            RUNS,
            train_acc,
            test_acc
        );
    }

    let elapsed = start.elapsed();
    let mean = accs.iter().sum::<f32>() / accs.len() as f32;
    let var = accs.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / accs.len() as f32;

    let majority = datasets::majority_class(&test_class, 2);

    println!("───────────────────────────────────────────────────────────────────");
    println!(
        "  test accuracy       {:.4} +/- {:.4}  (spread over seeds)",
        mean,
        var.sqrt()
    );
    println!("  majority-class      {:.4}  <- control", majority);
    println!("  separation          {:+.4}", mean - majority);
    println!("  wall clock          {:.2} s", elapsed.as_secs_f64());
    println!("  round-trip baseline 7.76 s  (train_mlp_cv, same task and budget)");
    println!("  speedup             {:.2}x", 7.76 / elapsed.as_secs_f64());
    println!("═══════════════════════════════════════════════════════════════════");
}
