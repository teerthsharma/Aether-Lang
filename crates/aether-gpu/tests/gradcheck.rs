//! Finite-difference verification of the GPU backward pass.
//!
//! A forward-correct kernel with a wrong backward does not crash. It trains, the
//! loss goes down, and the network lands on a worse optimum than it should have.
//! Nothing in a loss curve distinguishes that from a slightly unlucky run, which
//! is why this file exists and why end-to-end convergence is not a substitute.
//!
//! # The three-way check
//!
//! Finite differences in f32 are close to useless: the step has to be large
//! enough to survive f32 rounding of the loss and small enough for the
//! difference quotient to approximate the derivative, and for these networks no
//! step satisfies both comfortably. So the check is staged:
//!
//!   1. central differences in f64  vs  analytic gradient in f64
//!      -- validates the reference implementation's own maths
//!   2. analytic in f64             vs  analytic on the GPU in f32
//!      -- validates the kernels, at f32 tolerance
//!
//! Neither step alone is sufficient. The first cannot see the GPU at all; the
//! second would happily agree if both implementations shared a sign error.
//!
//! # One reference, several shapes
//!
//! This started as four hand-written reference networks -- one and two hidden
//! layers, sigmoid and softmax heads, small and tile-crossing sizes -- and the
//! two combinations that were still missing would have made six. Four
//! near-identical backward passes is a worse liability than the gap they were
//! covering: they drift, and a fix applied to one is not applied to the others.
//!
//! `Net` below is depth-generic, head-generic and size-generic, so every
//! combination is a table row rather than another transcription of the chain
//! rule.

use aether_gpu::{GpuContext, GpuTensor};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Head {
    /// Sigmoid with binary cross-entropy. One output unit.
    Sigmoid,
    /// Softmax with categorical cross-entropy. One-hot targets.
    Softmax,
}

fn context() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP: no usable GPU adapter ({e})");
            None
        }
    }
}

fn fill(n: usize, seed: u64) -> Vec<f64> {
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

/// A dense ReLU stack with a choice of output head, in f64.
///
/// `dims` is `[inputs, hidden.., outputs]`, so `dims.len() - 1` is the number of
/// weight matrices and every hidden layer carries a ReLU. The kernels are
/// matched exactly, including the ReLU derivative being zero at exactly zero and
/// the max subtraction inside softmax.
struct Net {
    rows: usize,
    dims: Vec<usize>,
    w: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    head: Head,
}

/// Per-layer pre-activations and activations, plus the output probabilities.
struct Fwd {
    z: Vec<Vec<f64>>,
    a: Vec<Vec<f64>>,
    p: Vec<f64>,
}

impl Net {
    fn new(rows: usize, dims: &[usize], head: Head, seed: u64) -> Self {
        let layers = dims.len() - 1;
        let mut w = Vec::with_capacity(layers);
        let mut b = Vec::with_capacity(layers);
        for l in 0..layers {
            w.push(fill(dims[l] * dims[l + 1], seed + l as u64 * 7 + 1));
            b.push(fill(dims[l + 1], seed + l as u64 * 7 + 2));
        }
        Self {
            rows,
            dims: dims.to_vec(),
            w,
            b,
            head,
        }
    }

    fn layers(&self) -> usize {
        self.dims.len() - 1
    }

    fn classes(&self) -> usize {
        *self.dims.last().unwrap()
    }

    fn forward(&self, x: &[f64]) -> Fwd {
        let mut z = Vec::with_capacity(self.layers());
        let mut a = vec![x.to_vec()];

        for l in 0..self.layers() {
            let (fan_in, fan_out) = (self.dims[l], self.dims[l + 1]);
            let input = &a[l];
            let mut zl = vec![0.0; self.rows * fan_out];
            for i in 0..self.rows {
                for j in 0..fan_out {
                    let mut s = self.b[l][j];
                    for k in 0..fan_in {
                        s += input[i * fan_in + k] * self.w[l][k * fan_out + j];
                    }
                    zl[i * fan_out + j] = s;
                }
            }
            // Every layer but the last is followed by a ReLU; the last feeds
            // the head.
            if l + 1 < self.layers() {
                a.push(zl.iter().map(|v| v.max(0.0)).collect());
            }
            z.push(zl);
        }

        let logits = z.last().unwrap();
        let p = match self.head {
            Head::Sigmoid => logits
                .iter()
                .map(|v| 1.0 / (1.0 + (-v).exp()))
                .collect::<Vec<_>>(),
            Head::Softmax => {
                let c = self.classes();
                let mut out = vec![0.0; self.rows * c];
                for i in 0..self.rows {
                    let row = &logits[i * c..(i + 1) * c];
                    let mx = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let sum: f64 = row.iter().map(|l| (l - mx).exp()).sum();
                    for j in 0..c {
                        out[i * c + j] = (row[j] - mx).exp() / sum;
                    }
                }
                out
            }
        };

        Fwd { z, a, p }
    }

    /// Mean loss. Both heads reduce to a mean over the batch, which is what
    /// makes the fused gradient `(p - y) / rows` in either case.
    fn loss(&self, x: &[f64], y: &[f64]) -> f64 {
        let f = self.forward(x);
        match self.head {
            Head::Sigmoid => {
                f.p.iter()
                    .zip(y)
                    .map(|(p, t)| {
                        let p = p.clamp(1e-15, 1.0 - 1e-15);
                        -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
                    })
                    .sum::<f64>()
                    / self.rows as f64
            }
            Head::Softmax => {
                let mut total = 0.0;
                for i in 0..f.p.len() {
                    if y[i] > 0.5 {
                        total -= f.p[i].clamp(1e-15, 1.0).ln();
                    }
                }
                total / self.rows as f64
            }
        }
    }

    /// Analytic gradients, one entry per weight matrix and bias vector.
    fn grads(&self, x: &[f64], y: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let f = self.forward(x);
        let layers = self.layers();

        let mut dw = vec![Vec::new(); layers];
        let mut db = vec![Vec::new(); layers];

        // Both heads collapse to the same expression, which is the whole reason
        // the kernels fuse them.
        let mut dz: Vec<f64> =
            f.p.iter()
                .zip(y)
                .map(|(p, t)| (p - t) / self.rows as f64)
                .collect();

        for l in (0..layers).rev() {
            let (fan_in, fan_out) = (self.dims[l], self.dims[l + 1]);
            let input = &f.a[l];

            let mut dwl = vec![0.0; fan_in * fan_out];
            for k in 0..fan_in {
                for j in 0..fan_out {
                    for i in 0..self.rows {
                        dwl[k * fan_out + j] += input[i * fan_in + k] * dz[i * fan_out + j];
                    }
                }
            }
            dw[l] = dwl;

            let mut dbl = vec![0.0; fan_out];
            for j in 0..fan_out {
                for i in 0..self.rows {
                    dbl[j] += dz[i * fan_out + j];
                }
            }
            db[l] = dbl;

            if l == 0 {
                break;
            }

            // Propagate through W[l], then gate on the *previous* layer's
            // pre-activation. Gating on the wrong tensor here is the defect the
            // stacked fixtures exist to catch.
            let prev_out = self.dims[l];
            let mut prev_dz = vec![0.0; self.rows * prev_out];
            for i in 0..self.rows {
                for k in 0..prev_out {
                    let mut s = 0.0;
                    for j in 0..fan_out {
                        s += dz[i * fan_out + j] * self.w[l][k * fan_out + j];
                    }
                    let gate = if f.z[l - 1][i * prev_out + k] > 0.0 {
                        1.0
                    } else {
                        0.0
                    };
                    prev_dz[i * prev_out + k] = s * gate;
                }
            }
            dz = prev_dz;
        }

        (dw, db)
    }
}

/// Targets: alternating labels for the sigmoid head, one-hot for softmax.
fn targets(net: &Net) -> Vec<f64> {
    match net.head {
        Head::Sigmoid => (0..net.rows).map(|i| (i % 2) as f64).collect(),
        Head::Softmax => {
            let c = net.classes();
            let mut out = vec![0.0; net.rows * c];
            for i in 0..net.rows {
                out[i * c + (i % c)] = 1.0;
            }
            out
        }
    }
}

/// Stage 1: analytic gradients against central differences.
///
/// Sweeps every parameter when the network is small and samples on a fixed
/// stride when it is not. The full sweep at the tile-crossing sizes is
/// thousands of parameters at two forward passes each, which is affordable in
/// release and not in debug, where these tests normally run.
///
/// Parameters near the ReLU kink are not skipped. The fixture is asserted clear
/// of it instead, so a seed that drifts onto a kink fails loudly rather than
/// quietly reducing what is checked.
fn stage1(label: &str, rows: usize, dims: &[usize], head: Head, seed: u64, max_checks: usize) {
    let mut net = Net::new(rows, dims, head, seed);
    let x = fill(rows * dims[0], seed);
    let y = targets(&net);

    // A central difference is only invalid if perturbing the parameter flips a
    // ReLU gate. Perturbing one weight by `h` moves a pre-activation by about
    // `h * |input|`, which for `h = 1e-6` and inputs in [-0.5, 0.5] is under
    // 1e-6. The guard sits an order of magnitude above that.
    //
    // It was 1e-4 initially, which is 200 times the perturbation and rejected
    // seeds that were perfectly usable. With 1584 pre-activations in the larger
    // fixtures, a bound that strict fails on most seeds -- and reseeding until
    // an over-strict guard passes is choosing the fixture to fit the check
    // rather than the check to fit the maths.
    const KINK_MARGIN: f64 = 1e-5;

    let f = net.forward(&x);
    for (l, zl) in f.z.iter().enumerate().take(net.layers() - 1) {
        let closest = zl.iter().map(|z| z.abs()).fold(f64::INFINITY, f64::min);
        assert!(
            closest > KINK_MARGIN,
            "{label}: layer {l} has a pre-activation {closest:.3e} from the ReLU \
             kink, inside the {KINK_MARGIN:.0e} margin; reseed rather than \
             loosen the gradient tolerance"
        );
    }

    let (dw, db) = net.grads(&x, &y);
    let h = 1e-6;

    let total: usize =
        dw.iter().map(|v| v.len()).sum::<usize>() + db.iter().map(|v| v.len()).sum::<usize>();
    let stride = (total / max_checks).max(1);

    let mut checked = 0;
    let mut seen = 0;

    for l in 0..net.layers() {
        for (is_bias, analytic) in [(false, &dw[l]), (true, &db[l])] {
            for idx in 0..analytic.len() {
                seen += 1;
                if seen % stride != 0 {
                    continue;
                }

                let a = analytic[idx];
                let orig = if is_bias {
                    net.b[l][idx]
                } else {
                    net.w[l][idx]
                };

                if is_bias {
                    net.b[l][idx] = orig + h;
                } else {
                    net.w[l][idx] = orig + h;
                }
                let up = net.loss(&x, &y);

                if is_bias {
                    net.b[l][idx] = orig - h;
                } else {
                    net.w[l][idx] = orig - h;
                }
                let down = net.loss(&x, &y);

                if is_bias {
                    net.b[l][idx] = orig;
                } else {
                    net.w[l][idx] = orig;
                }

                let fd = (up - down) / (2.0 * h);
                let denom = a.abs().max(fd.abs()).max(1e-8);
                let rel = (a - fd).abs() / denom;

                assert!(
                    rel < 1e-5,
                    "{label}: layer {l} {} index {idx}: analytic {a:.12e} vs \
                     finite diff {fd:.12e}, relative error {rel:.3e}",
                    if is_bias { "bias" } else { "weight" }
                );
                checked += 1;
            }
        }
    }

    assert!(
        checked >= 10,
        "{label}: only {checked} parameters checked out of {total}"
    );
    println!("{label}: stage 1 matched {checked} of {total} parameters");
}

/// Stage 2: the GPU kernels against the verified f64 reference, running the
/// same operation sequence the training examples run.
fn stage2(
    ctx: &GpuContext,
    label: &str,
    rows: usize,
    dims: &[usize],
    head: Head,
    seed: u64,
    tol: f64,
) {
    let net = Net::new(rows, dims, head, seed);
    let x64 = fill(rows * dims[0], seed);
    let y64 = targets(&net);
    let (r_dw, r_db) = net.grads(&x64, &y64);

    let f32v = |v: &[f64]| -> Vec<f32> { v.iter().map(|x| *x as f32).collect() };
    let layers = net.layers();

    let gy = ctx.upload(&f32v(&y64), rows, net.classes()).expect("y");

    let gw: Vec<GpuTensor> = (0..layers)
        .map(|l| {
            ctx.upload(&f32v(&net.w[l]), dims[l], dims[l + 1])
                .expect("w")
        })
        .collect();
    let gb: Vec<GpuTensor> = (0..layers)
        .map(|l| ctx.upload(&f32v(&net.b[l]), 1, dims[l + 1]).expect("b"))
        .collect();

    // Forward, keeping each layer's pre-activation and input for the backward.
    let mut z: Vec<GpuTensor> = Vec::with_capacity(layers);
    let mut a: Vec<GpuTensor> = Vec::with_capacity(layers);
    let mut cur = ctx.upload(&f32v(&x64), rows, dims[0]).expect("a0");

    for l in 0..layers {
        let zl = ctx.matmul_resident(&cur, &gw[l]).expect("z");
        let zl = ctx.add_bias_resident(&zl, &gb[l]).expect("zb");
        a.push(cur);
        if l + 1 < layers {
            cur = ctx.relu_resident(&zl).expect("relu");
        } else {
            cur = ctx.upload(&[0.0], 1, 1).expect("placeholder");
        }
        z.push(zl);
    }

    // Backward.
    let mut dz = match head {
        Head::Sigmoid | Head::Softmax => {
            let logits = z.last().unwrap();
            if head == Head::Sigmoid {
                ctx.sigmoid_bce_grad_resident(logits, &gy).expect("dz")
            } else {
                ctx.softmax_xent_grad_resident(logits, &gy).expect("dz")
            }
        }
    };

    let mut got_dw: Vec<Vec<f32>> = vec![Vec::new(); layers];
    let mut got_db: Vec<Vec<f32>> = vec![Vec::new(); layers];

    for l in (0..layers).rev() {
        let at = ctx.transpose_resident(&a[l]).expect("a^T");
        let dwl = ctx.matmul_resident(&at, &dz).expect("dw");
        let dbl = ctx.column_sums_resident(&dz).expect("db");

        got_dw[l] = ctx.read(&dwl).expect("read dw");
        got_db[l] = ctx.read(&dbl).expect("read db");

        if l == 0 {
            break;
        }

        let wt = ctx.transpose_resident(&gw[l]).expect("w^T");
        let da = ctx.matmul_resident(&dz, &wt).expect("da");
        dz = ctx.relu_backward_resident(&z[l - 1], &da).expect("dz");
    }

    let mut worst = 0.0f64;
    let mut entries = 0;

    for l in 0..layers {
        for (name, got, want) in [("dw", &got_dw[l], &r_dw[l]), ("db", &got_db[l], &r_db[l])] {
            assert_eq!(got.len(), want.len(), "{label}: layer {l} {name} length");
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                let denom = w.abs().max(1e-5);
                let rel = ((*g as f64) - w).abs() / denom;
                worst = worst.max(rel);
                entries += 1;
                assert!(
                    rel < tol,
                    "{label}: layer {l} {name}[{i}]: gpu {g:.8e} vs reference \
                     {w:.8e}, relative error {rel:.3e}"
                );
            }
        }
    }

    println!("{label}: stage 2 matched {entries} entries, worst relative error {worst:.3e}");
}

// ═══════════════════════════════════════════════════════════════════════════════
// The matrix
//
// Depth x head x size. Every row was previously a separate transcription of the
// chain rule; the last two rows did not exist at all, because writing a fifth
// and sixth reference by hand was not worth it.
//
// The tile-crossing dimensions are deliberate: 33 = 2*16+1, 17 = 16+1,
// 48 = 3*16, so every matmul spans several of the kernel's 16-wide tiles and
// leaves a partial tail. A mutation run showed the suite reports a clean pass on
// a kernel with a data race when every case fits inside one tile.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn one_layer_sigmoid_small() {
    stage1("1-layer sigmoid 5x3x4", 5, &[3, 4, 1], Head::Sigmoid, 1, 64);
    if let Some(ctx) = context() {
        stage2(
            &ctx,
            "1-layer sigmoid 5x3x4",
            5,
            &[3, 4, 1],
            Head::Sigmoid,
            1,
            2e-3,
        );
    }
}

#[test]
fn two_layer_sigmoid_small() {
    stage1(
        "2-layer sigmoid 5x3x4x4",
        5,
        &[3, 4, 4, 1],
        Head::Sigmoid,
        11,
        64,
    );
    if let Some(ctx) = context() {
        stage2(
            &ctx,
            "2-layer sigmoid 5x3x4x4",
            5,
            &[3, 4, 4, 1],
            Head::Sigmoid,
            11,
            2e-3,
        );
    }
}

#[test]
fn one_layer_softmax_small() {
    stage1(
        "1-layer softmax 5x3x4x3",
        5,
        &[3, 4, 3],
        Head::Softmax,
        21,
        64,
    );
    if let Some(ctx) = context() {
        stage2(
            &ctx,
            "1-layer softmax 5x3x4x3",
            5,
            &[3, 4, 3],
            Head::Softmax,
            21,
            2e-3,
        );
    }
}

#[test]
fn one_layer_sigmoid_tile_crossing() {
    stage1(
        "1-layer sigmoid 33x17x48",
        33,
        &[17, 48, 1],
        Head::Sigmoid,
        31,
        32,
    );
    if let Some(ctx) = context() {
        stage2(
            &ctx,
            "1-layer sigmoid 33x17x48",
            33,
            &[17, 48, 1],
            Head::Sigmoid,
            31,
            1e-3,
        );
    }
}

/// Depth and tile-crossing together, which no earlier fixture covered: the
/// stacked cases were all small and the large case was all one layer.
#[test]
fn two_layer_sigmoid_tile_crossing() {
    stage1(
        "2-layer sigmoid 33x17x48x48",
        33,
        &[17, 48, 48, 1],
        Head::Sigmoid,
        41,
        32,
    );
    if let Some(ctx) = context() {
        stage2(
            &ctx,
            "2-layer sigmoid 33x17x48x48",
            33,
            &[17, 48, 48, 1],
            Head::Sigmoid,
            41,
            1e-3,
        );
    }
}

/// The softmax head at a size that crosses tiles, and stacked. Previously the
/// multi-class gradient was verified only at 5x3x4.
#[test]
fn two_layer_softmax_tile_crossing() {
    stage1(
        "2-layer softmax 33x17x48x48x5",
        33,
        &[17, 48, 48, 5],
        Head::Softmax,
        51,
        32,
    );
    if let Some(ctx) = context() {
        stage2(
            &ctx,
            "2-layer softmax 33x17x48x48x5",
            33,
            &[17, 48, 48, 5],
            Head::Softmax,
            51,
            1e-3,
        );
    }
}

/// A gradient merely proportional to the truth passes a direction check but
/// trains at the wrong effective rate, and a missing `1/batch` is the usual
/// cause. The mean of `(p - y)` is bounded by 1, so its magnitude catches it.
#[test]
fn the_gradient_is_scaled_by_the_batch_size_not_merely_proportional() {
    let net = Net::new(5, &[3, 4, 1], Head::Sigmoid, 61);
    let x = fill(5 * 3, 61);
    let y = targets(&net);
    let (_, db) = net.grads(&x, &y);

    assert!(
        db[1][0].abs() <= 1.0,
        "output bias gradient {} exceeds 1, so the 1/batch scaling is missing",
        db[1][0]
    );
}
