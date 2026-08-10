//! Multi-class GPU training, evaluated on an independently drawn test set.
//!
//! Run:
//!   cargo run -p aether-gpu --example train_multiclass --release
//!
//! # What this adds over `train_resident`
//!
//! Softmax over three classes instead of a sigmoid over one, and reporting that
//! can detect the failure mode accuracy hides. A three-class model that ignores
//! one class entirely and splits the rest still posts a respectable accuracy;
//! the confusion matrix and per-class recall show it immediately, which is why
//! both are printed rather than a single headline number.
//!
//! Train and test are separate i.i.d. draws from the same generator, not two
//! halves of one sample. An earlier version cross-validated a single swept
//! sample and reported 0.8067; that split placed a training point 0.99x the
//! cloud's own spacing from every held-out point, so the figure described
//! interpolation between near-duplicates. See aether_gpu::datasets.
//!
//! # Honest description of the data
//!
//! Three interleaved spirals, generated. It is a synthetic benchmark, not a
//! dataset from the world, and no claim here transfers to real data. It is used
//! because it is non-linearly separable -- a linear model cannot do better than
//! the majority-class control -- so the comparison against that control means
//! something.

use std::time::Instant;

use aether_gpu::{datasets, GpuContext, GpuTensor};

const RUNS: usize = 5;
const EPOCHS: usize = 100;
const HIDDEN: usize = 64;
const CLASSES: usize = 3;
const LEARNING_RATE: f32 = 0.5;
const PER_CLASS: usize = 300;

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

struct Net {
    w1: GpuTensor,
    b1: GpuTensor,
    w2: GpuTensor,
    b2: GpuTensor,
    w3: GpuTensor,
    b3: GpuTensor,
}

impl Net {
    fn new(ctx: &GpuContext, rng: &mut Lcg) -> Self {
        let up = |v: Vec<f32>, r, c| ctx.upload(&v, r, c).expect("upload");
        Self {
            w1: up(init_weights(2, HIDDEN, rng), 2, HIDDEN),
            b1: up(vec![0.0; HIDDEN], 1, HIDDEN),
            w2: up(init_weights(HIDDEN, HIDDEN, rng), HIDDEN, HIDDEN),
            b2: up(vec![0.0; HIDDEN], 1, HIDDEN),
            w3: up(init_weights(HIDDEN, CLASSES, rng), HIDDEN, CLASSES),
            b3: up(vec![0.0; CLASSES], 1, CLASSES),
        }
    }

    fn forward(
        &self,
        ctx: &GpuContext,
        x: &GpuTensor,
    ) -> (GpuTensor, GpuTensor, GpuTensor, GpuTensor, GpuTensor) {
        let z1 = ctx.matmul_resident(x, &self.w1).expect("z1");
        let z1 = ctx.add_bias_resident(&z1, &self.b1).expect("z1b");
        let a1 = ctx.relu_resident(&z1).expect("a1");
        let z2 = ctx.matmul_resident(&a1, &self.w2).expect("z2");
        let z2 = ctx.add_bias_resident(&z2, &self.b2).expect("z2b");
        let a2 = ctx.relu_resident(&z2).expect("a2");
        let z3 = ctx.matmul_resident(&a2, &self.w3).expect("z3");
        let z3 = ctx.add_bias_resident(&z3, &self.b3).expect("z3b");
        (z1, a1, z2, a2, z3)
    }

    fn step(&mut self, ctx: &GpuContext, x: &GpuTensor, y: &GpuTensor, lr: f32) {
        let (z1, a1, z2, a2, z3) = self.forward(ctx, x);

        let dz3 = ctx.softmax_xent_grad_resident(&z3, y).expect("dz3");

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

        // Batch within the step, submit per step. See FEATURES.md: flushing
        // per fold instead measured slower than not batching at all.
        ctx.flush();
    }

    fn predict(&self, ctx: &GpuContext, x: &GpuTensor, rows: usize) -> Vec<usize> {
        let (_, _, _, _, z3) = self.forward(ctx, x);
        let p = ctx
            .read(&ctx.softmax_resident(&z3).expect("softmax"))
            .expect("read");

        (0..rows)
            .map(|r| {
                let row = &p[r * CLASSES..(r + 1) * CLASSES];
                let mut best = 0;
                for j in 1..CLASSES {
                    if row[j] > row[best] {
                        best = j;
                    }
                }
                best
            })
            .collect()
    }
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
        eprintln!("adapter is {}, refusing to report as GPU", info.device_type);
        std::process::exit(1);
    }

    // Train and test are drawn independently from the same generator.
    //
    // This example previously cross-validated a single swept sample, which
    // measured interpolation between near-duplicates rather than
    // generalisation: dealing consecutive arc points into different folds put a
    // training point 0.99x the cloud's own spacing from every held-out point.
    // The figure it printed, 0.8067, was not what it was labelled.
    //
    // See aether_gpu::datasets for the diagnostic and the full argument.
    let (x, y) = datasets::spirals_iid(0x5B1FA1, CLASSES, PER_CLASS);
    let (test_x, test_y) = datasets::spirals_iid(0x5B1FA2, CLASSES, PER_CLASS);
    let n = y.len();
    let n_te = test_y.len();
    let y_hot = datasets::one_hot(&y, CLASSES);

    // Assert the split is real rather than trusting that it is.
    let mut combined = x.clone();
    combined.extend_from_slice(&test_x);
    let is_test: Vec<bool> = (0..n)
        .map(|_| false)
        .chain((0..n_te).map(|_| true))
        .collect();
    datasets::report_split("independent draws", &combined, &is_test);
    println!();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Multi-class GPU training, {RUNS} seeds, {EPOCHS} epochs");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  adapter    {}  |  {}", info.name, info.backend);
    println!("  train      {n} points, drawn i.i.d. (synthetic {CLASSES}-spiral)");
    println!("  test       {n_te} points, drawn independently from the same generator");
    println!("  network    2 -> {HIDDEN} -> {HIDDEN} -> {CLASSES}, ReLU, softmax + cross-entropy");

    // Confusion pooled over runs. The spread below is over initialisations,
    // not over folds: there are no folds now, because train and test are
    // separate draws rather than two halves of one sample.
    let mut confusion = vec![vec![0usize; CLASSES]; CLASSES];
    let mut accs = Vec::new();
    let start = Instant::now();

    let gx = ctx.upload(&x, n, 2).expect("x");
    let gy = ctx.upload(&y_hot, n, CLASSES).expect("y");
    let gtex = ctx.upload(&test_x, n_te, 2).expect("test x");

    for run in 0..RUNS {
        let mut rng = Lcg(0xBEEF + run as u64);
        let mut net = Net::new(&ctx, &mut rng);

        for _ in 0..EPOCHS {
            net.step(&ctx, &gx, &gy, LEARNING_RATE);
        }

        let pred = net.predict(&ctx, &gtex, n_te);
        let correct = pred.iter().zip(&test_y).filter(|(p, t)| p == t).count();
        let acc = correct as f32 / n_te as f32;
        accs.push(acc);

        for (p, t) in pred.iter().zip(&test_y) {
            confusion[*t][*p] += 1;
        }

        println!("  seed {}/{}  test accuracy {:.4}", run + 1, RUNS, acc);
    }

    let elapsed = start.elapsed();
    let mean = accs.iter().sum::<f32>() / accs.len() as f32;
    let sd = (accs.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / accs.len() as f32).sqrt();

    println!("───────────────────────────────────────────────────────────────────");
    println!("  Confusion matrix (rows = true, cols = predicted)");
    print!("            ");
    for c in 0..CLASSES {
        print!("  pred {c}");
    }
    println!();
    for t in 0..CLASSES {
        print!("  true {t}    ");
        for p in 0..CLASSES {
            print!("{:>8}", confusion[t][p]);
        }
        println!();
    }

    println!("───────────────────────────────────────────────────────────────────");
    println!(
        "  {:<8}{:>10}{:>10}{:>10}",
        "class", "precision", "recall", "F1"
    );

    let mut macro_f1 = 0.0;
    for c in 0..CLASSES {
        let tp = confusion[c][c] as f32;
        let fn_ = (0..CLASSES).map(|p| confusion[c][p]).sum::<usize>() as f32 - tp;
        let fp = (0..CLASSES).map(|t| confusion[t][c]).sum::<usize>() as f32 - tp;

        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        macro_f1 += f1 / CLASSES as f32;

        println!("  {c:<8}{precision:>10.4}{recall:>10.4}{f1:>10.4}");
    }

    let majority = {
        let mut counts = vec![0usize; CLASSES];
        for &c in &y {
            counts[c] += 1;
        }
        *counts.iter().max().unwrap() as f32 / n as f32
    };

    println!("───────────────────────────────────────────────────────────────────");
    println!("  test accuracy     {mean:.4} +/- {sd:.4}  (spread over {RUNS} seeds)");
    println!("  macro F1          {macro_f1:.4}");
    println!("  majority-class    {majority:.4}  <- control");
    println!("  separation        {:+.4}", mean - majority);
    println!("  wall clock        {:.2} s", elapsed.as_secs_f64());
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Macro F1 is reported next to accuracy because a model that");
    println!("  abandons one class entirely still posts a plausible accuracy.");
    println!("  The confusion matrix above is what shows whether it did.");
    println!("═══════════════════════════════════════════════════════════════════");
}
