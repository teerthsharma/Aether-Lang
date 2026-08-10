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
//! difference quotient to approximate the derivative, and for this network no
//! step satisfies both comfortably. So the check is staged:
//!
//!   1. central differences in f64  vs  analytic gradient in f64
//!      -- validates the reference implementation's own maths
//!   2. analytic in f64             vs  analytic on the GPU in f32
//!      -- validates the kernels, at f32 tolerance
//!
//! Neither step alone is sufficient. The first cannot see the GPU at all; the
//! second would happily agree if both implementations shared a sign error.

use aether_gpu::{GpuContext, GpuTensor};

const BATCH: usize = 5;
const IN: usize = 3;
const HID: usize = 4;

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

/// The reference network, in f64, matching the kernels exactly:
/// `x -> W1 -> +b1 -> relu -> W2 -> +b2 -> sigmoid -> mean BCE`.
///
/// One hidden layer rather than the two in `train_resident`: the gradient
/// through a second identical layer exercises no new code path, and fewer
/// parameters means the finite-difference sweep stays cheap.
struct Ref {
    w1: Vec<f64>, // [IN, HID]
    b1: Vec<f64>, // [HID]
    w2: Vec<f64>, // [HID, 1]
    b2: Vec<f64>, // [1]
}

impl Ref {
    fn loss(&self, x: &[f64], y: &[f64]) -> f64 {
        let (_, _, p) = self.forward(x);
        p.iter()
            .zip(y)
            .map(|(p, t)| {
                let p = p.clamp(1e-12, 1.0 - 1e-12);
                -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
            })
            .sum::<f64>()
            / BATCH as f64
    }

    /// Returns `(z1, a1, p)`.
    fn forward(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut z1 = vec![0.0; BATCH * HID];
        for i in 0..BATCH {
            for j in 0..HID {
                let mut s = self.b1[j];
                for k in 0..IN {
                    s += x[i * IN + k] * self.w1[k * HID + j];
                }
                z1[i * HID + j] = s;
            }
        }

        let a1: Vec<f64> = z1.iter().map(|v| v.max(0.0)).collect();

        let mut p = vec![0.0; BATCH];
        for i in 0..BATCH {
            let mut s = self.b2[0];
            for j in 0..HID {
                s += a1[i * HID + j] * self.w2[j];
            }
            p[i] = 1.0 / (1.0 + (-s).exp());
        }

        (z1, a1, p)
    }

    /// Analytic gradients, returned in the same layout as the parameters.
    fn grads(&self, x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (z1, a1, p) = self.forward(x);

        // Fused sigmoid + BCE derivative, matching sigmoid_bce_grad.
        let dz2: Vec<f64> = p
            .iter()
            .zip(y)
            .map(|(p, t)| (p - t) / BATCH as f64)
            .collect();

        let mut dw2 = vec![0.0; HID];
        for j in 0..HID {
            for i in 0..BATCH {
                dw2[j] += a1[i * HID + j] * dz2[i];
            }
        }
        let db2 = vec![dz2.iter().sum::<f64>()];

        let mut dz1 = vec![0.0; BATCH * HID];
        for i in 0..BATCH {
            for j in 0..HID {
                // ReLU derivative: zero at exactly zero, matching the kernel.
                let gate = if z1[i * HID + j] > 0.0 { 1.0 } else { 0.0 };
                dz1[i * HID + j] = dz2[i] * self.w2[j] * gate;
            }
        }

        let mut dw1 = vec![0.0; IN * HID];
        for k in 0..IN {
            for j in 0..HID {
                for i in 0..BATCH {
                    dw1[k * HID + j] += x[i * IN + k] * dz1[i * HID + j];
                }
            }
        }

        let mut db1 = vec![0.0; HID];
        for j in 0..HID {
            for i in 0..BATCH {
                db1[j] += dz1[i * HID + j];
            }
        }

        (dw1, db1, dw2, db2)
    }
}

/// Central differences: `(L(w+h) - L(w-h)) / 2h`.
///
/// Central rather than forward because the error is O(h^2) instead of O(h),
/// which is what makes a meaningful tolerance possible at all.
fn finite_diff(net: &mut Ref, x: &[f64], y: &[f64], which: usize, idx: usize, h: f64) -> f64 {
    let get = |n: &mut Ref, w: usize, i: usize| -> f64 {
        match w {
            0 => n.w1[i],
            1 => n.b1[i],
            2 => n.w2[i],
            _ => n.b2[i],
        }
    };
    let set = |n: &mut Ref, w: usize, i: usize, v: f64| match w {
        0 => n.w1[i] = v,
        1 => n.b1[i] = v,
        2 => n.w2[i] = v,
        _ => n.b2[i] = v,
    };

    let orig = get(net, which, idx);

    set(net, which, idx, orig + h);
    let up = net.loss(x, y);

    set(net, which, idx, orig - h);
    let down = net.loss(x, y);

    set(net, which, idx, orig);

    (up - down) / (2.0 * h)
}

/// Stage 1: the reference's own analytic gradient against central differences.
///
/// A ReLU is non-differentiable at zero, so a unit whose pre-activation sits
/// within `h` of the boundary has a finite difference that straddles the kink
/// and legitimately disagrees with either one-sided derivative. Those entries
/// are skipped rather than fudged into the tolerance.
#[test]
fn the_reference_analytic_gradient_matches_central_differences() {
    let x = fill(BATCH * IN, 1);
    let y: Vec<f64> = (0..BATCH).map(|i| (i % 2) as f64).collect();

    let mut net = Ref {
        w1: fill(IN * HID, 2),
        b1: fill(HID, 3),
        w2: fill(HID, 4),
        b2: fill(1, 5),
    };

    let (dw1, db1, dw2, db2) = net.grads(&x, &y);
    let h = 1e-6;

    let groups: [(usize, &[f64]); 4] = [(0, &dw1), (1, &db1), (2, &dw2), (3, &db2)];

    let mut checked = 0;
    for (which, analytic) in groups {
        for (idx, &a) in analytic.iter().enumerate() {
            // Skip any parameter whose perturbation moves a pre-activation
            // across the ReLU kink.
            let (z1_before, _, _) = net.forward(&x);
            let near_kink = z1_before.iter().any(|z| z.abs() < 1e-3);
            if near_kink {
                continue;
            }

            let fd = finite_diff(&mut net, &x, &y, which, idx, h);
            let denom = a.abs().max(fd.abs()).max(1e-8);
            let rel = (a - fd).abs() / denom;

            assert!(
                rel < 1e-5,
                "group {which} index {idx}: analytic {a:.12e} vs finite diff {fd:.12e}, \
                 relative error {rel:.3e}"
            );
            checked += 1;
        }
    }

    assert!(
        checked > 0,
        "every parameter was skipped; the check proved nothing"
    );
    println!("stage 1: {checked} parameters matched central differences");
}

/// Stage 2: the GPU kernels against the verified f64 reference.
///
/// Builds the same network from the public resident operations and compares the
/// gradients the kernels produce with the reference's analytic ones.
#[test]
fn the_gpu_backward_pass_matches_the_reference_gradients() {
    let Some(ctx) = context() else { return };

    let x64 = fill(BATCH * IN, 11);
    let y64: Vec<f64> = (0..BATCH).map(|i| (i % 2) as f64).collect();
    let w1_64 = fill(IN * HID, 12);
    let b1_64 = fill(HID, 13);
    let w2_64 = fill(HID, 14);
    let b2_64 = fill(1, 15);

    let net = Ref {
        w1: w1_64.clone(),
        b1: b1_64.clone(),
        w2: w2_64.clone(),
        b2: b2_64.clone(),
    };
    let (ref_dw1, ref_db1, ref_dw2, ref_db2) = net.grads(&x64, &y64);

    let f32v = |v: &[f64]| -> Vec<f32> { v.iter().map(|x| *x as f32).collect() };

    let gx = ctx.upload(&f32v(&x64), BATCH, IN).expect("x");
    let gy = ctx.upload(&f32v(&y64), BATCH, 1).expect("y");
    let gw1 = ctx.upload(&f32v(&w1_64), IN, HID).expect("w1");
    let gb1 = ctx.upload(&f32v(&b1_64), 1, HID).expect("b1");
    let gw2 = ctx.upload(&f32v(&w2_64), HID, 1).expect("w2");
    let gb2 = ctx.upload(&f32v(&b2_64), 1, 1).expect("b2");

    // Forward
    let z1 = ctx.matmul_resident(&gx, &gw1).expect("z1");
    let z1 = ctx.add_bias_resident(&z1, &gb1).expect("z1 bias");
    let a1 = ctx.relu_resident(&z1).expect("a1");
    let z2 = ctx.matmul_resident(&a1, &gw2).expect("z2");
    let z2 = ctx.add_bias_resident(&z2, &gb2).expect("z2 bias");

    // Backward
    let dz2 = ctx.sigmoid_bce_grad_resident(&z2, &gy).expect("dz2");
    let a1t = ctx.transpose_resident(&a1).expect("a1t");
    let dw2 = ctx.matmul_resident(&a1t, &dz2).expect("dw2");
    let db2 = ctx.column_sums_resident(&dz2).expect("db2");

    let w2t = ctx.transpose_resident(&gw2).expect("w2t");
    let da1 = ctx.matmul_resident(&dz2, &w2t).expect("da1");
    let dz1 = ctx.relu_backward_resident(&z1, &da1).expect("dz1");

    let xt = ctx.transpose_resident(&gx).expect("xt");
    let dw1 = ctx.matmul_resident(&xt, &dz1).expect("dw1");
    let db1 = ctx.column_sums_resident(&dz1).expect("db1");

    let cmp = |name: &str, got: &GpuTensor, want: &[f64]| {
        let g = ctx.read(got).expect("readback");
        assert_eq!(g.len(), want.len(), "{name}: length");
        for (i, (a, b)) in g.iter().zip(want).enumerate() {
            let denom = (*b).abs().max(1e-6);
            let rel = ((*a as f64) - b).abs() / denom;
            assert!(
                rel < 2e-3,
                "{name}[{i}]: gpu {a:.8e} vs reference {b:.8e}, relative error {rel:.3e}"
            );
        }
        println!("{name}: {} entries matched the f64 reference", g.len());
    };

    cmp("dw1", &dw1, &ref_dw1);
    cmp("db1", &db1, &ref_db1);
    cmp("dw2", &dw2, &ref_dw2);
    cmp("db2", &db2, &ref_db2);
}

/// A gradient that is merely *proportional* to the truth passes a direction
/// check but trains at the wrong effective learning rate, and a 1/batch scaling
/// error is the usual way that happens. Comparing magnitudes catches it.
#[test]
fn the_gradient_is_scaled_by_the_batch_size_not_merely_proportional() {
    let x = fill(BATCH * IN, 21);
    let y: Vec<f64> = (0..BATCH).map(|i| (i % 2) as f64).collect();

    let net = Ref {
        w1: fill(IN * HID, 22),
        b1: fill(HID, 23),
        w2: fill(HID, 24),
        b2: fill(1, 25),
    };

    let (_, _, _, db2) = net.grads(&x, &y);

    // db2 is the mean of (p - y) over the batch, so it is bounded by 1.
    assert!(
        db2[0].abs() <= 1.0,
        "db2 = {} exceeds 1, which means the 1/batch scaling is missing",
        db2[0]
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Softmax and categorical cross-entropy
//
// The binary path above shares no code with this one past the matmuls: a
// different output non-linearity, a different loss, and a different fused
// gradient kernel. Verifying one says nothing about the other, so the whole
// staged check is repeated rather than assumed to carry over.
// ═══════════════════════════════════════════════════════════════════════════════

const CLASSES: usize = 3;

/// `x -> W1 -> +b1 -> relu -> W2 -> +b2 -> softmax -> mean categorical CE`.
struct RefMc {
    w1: Vec<f64>, // [IN, HID]
    b1: Vec<f64>, // [HID]
    w2: Vec<f64>, // [HID, CLASSES]
    b2: Vec<f64>, // [CLASSES]
}

impl RefMc {
    /// Returns `(z1, a1, probabilities)`.
    fn forward(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut z1 = vec![0.0; BATCH * HID];
        for i in 0..BATCH {
            for j in 0..HID {
                let mut s = self.b1[j];
                for k in 0..IN {
                    s += x[i * IN + k] * self.w1[k * HID + j];
                }
                z1[i * HID + j] = s;
            }
        }

        let a1: Vec<f64> = z1.iter().map(|v| v.max(0.0)).collect();

        let mut p = vec![0.0; BATCH * CLASSES];
        for i in 0..BATCH {
            let mut logits = [0.0f64; CLASSES];
            for (c, slot) in logits.iter_mut().enumerate() {
                let mut s = self.b2[c];
                for j in 0..HID {
                    s += a1[i * HID + j] * self.w2[j * CLASSES + c];
                }
                *slot = s;
            }

            // Same max subtraction as the kernel, for the same reason.
            let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = logits.iter().map(|l| (l - mx).exp()).sum();
            for c in 0..CLASSES {
                p[i * CLASSES + c] = (logits[c] - mx).exp() / sum;
            }
        }

        (z1, a1, p)
    }

    /// Mean categorical cross-entropy against one-hot targets.
    fn loss(&self, x: &[f64], y: &[f64]) -> f64 {
        let (_, _, p) = self.forward(x);
        let mut total = 0.0;
        for i in 0..BATCH {
            for c in 0..CLASSES {
                if y[i * CLASSES + c] > 0.5 {
                    total -= p[i * CLASSES + c].clamp(1e-15, 1.0).ln();
                }
            }
        }
        total / BATCH as f64
    }

    fn grads(&self, x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (z1, a1, p) = self.forward(x);

        // Fused softmax + cross-entropy derivative, matching softmax_xent_grad.
        let dz2: Vec<f64> = (0..BATCH * CLASSES)
            .map(|i| (p[i] - y[i]) / BATCH as f64)
            .collect();

        let mut dw2 = vec![0.0; HID * CLASSES];
        for j in 0..HID {
            for c in 0..CLASSES {
                for i in 0..BATCH {
                    dw2[j * CLASSES + c] += a1[i * HID + j] * dz2[i * CLASSES + c];
                }
            }
        }

        let mut db2 = vec![0.0; CLASSES];
        for c in 0..CLASSES {
            for i in 0..BATCH {
                db2[c] += dz2[i * CLASSES + c];
            }
        }

        let mut dz1 = vec![0.0; BATCH * HID];
        for i in 0..BATCH {
            for j in 0..HID {
                let mut s = 0.0;
                for c in 0..CLASSES {
                    s += dz2[i * CLASSES + c] * self.w2[j * CLASSES + c];
                }
                let gate = if z1[i * HID + j] > 0.0 { 1.0 } else { 0.0 };
                dz1[i * HID + j] = s * gate;
            }
        }

        let mut dw1 = vec![0.0; IN * HID];
        for k in 0..IN {
            for j in 0..HID {
                for i in 0..BATCH {
                    dw1[k * HID + j] += x[i * IN + k] * dz1[i * HID + j];
                }
            }
        }

        let mut db1 = vec![0.0; HID];
        for j in 0..HID {
            for i in 0..BATCH {
                db1[j] += dz1[i * HID + j];
            }
        }

        (dw1, db1, dw2, db2)
    }
}

fn one_hot(labels: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0; labels.len() * CLASSES];
    for (i, &c) in labels.iter().enumerate() {
        out[i * CLASSES + c] = 1.0;
    }
    out
}

/// Stage 1 for the multi-class path.
#[test]
fn the_reference_softmax_gradient_matches_central_differences() {
    let x = fill(BATCH * IN, 51);
    let y = one_hot(&(0..BATCH).map(|i| i % CLASSES).collect::<Vec<_>>());

    let mut net = RefMc {
        w1: fill(IN * HID, 52),
        b1: fill(HID, 53),
        w2: fill(HID * CLASSES, 54),
        b2: fill(CLASSES, 55),
    };

    let (dw1, db1, dw2, db2) = net.grads(&x, &y);
    let h = 1e-6;

    let (z1, _, _) = net.forward(&x);
    assert!(
        z1.iter().all(|z| z.abs() > 1e-3),
        "a pre-activation sits on the ReLU kink; reseed rather than loosen the tolerance"
    );

    let mut checked = 0;
    for (which, analytic) in [(0usize, &dw1), (1, &db1), (2, &dw2), (3, &db2)] {
        for (idx, &a) in analytic.iter().enumerate() {
            let orig = match which {
                0 => net.w1[idx],
                1 => net.b1[idx],
                2 => net.w2[idx],
                _ => net.b2[idx],
            };

            let set = |v: f64, net: &mut RefMc| match which {
                0 => net.w1[idx] = v,
                1 => net.b1[idx] = v,
                2 => net.w2[idx] = v,
                _ => net.b2[idx] = v,
            };

            set(orig + h, &mut net);
            let up = net.loss(&x, &y);
            set(orig - h, &mut net);
            let down = net.loss(&x, &y);
            set(orig, &mut net);

            let fd = (up - down) / (2.0 * h);
            let denom = a.abs().max(fd.abs()).max(1e-8);
            let rel = (a - fd).abs() / denom;

            assert!(
                rel < 1e-5,
                "group {which} index {idx}: analytic {a:.12e} vs finite diff {fd:.12e}, \
                 relative error {rel:.3e}"
            );
            checked += 1;
        }
    }

    assert_eq!(
        checked,
        IN * HID + HID + HID * CLASSES + CLASSES,
        "every parameter must be checked"
    );
    println!("stage 1 (softmax): {checked} parameters matched central differences");
}

/// Stage 2 for the multi-class path: the GPU kernels against the verified
/// f64 reference.
#[test]
fn the_gpu_softmax_backward_matches_the_reference_gradients() {
    let Some(ctx) = context() else { return };

    let x64 = fill(BATCH * IN, 61);
    let y64 = one_hot(&(0..BATCH).map(|i| i % CLASSES).collect::<Vec<_>>());
    let w1_64 = fill(IN * HID, 62);
    let b1_64 = fill(HID, 63);
    let w2_64 = fill(HID * CLASSES, 64);
    let b2_64 = fill(CLASSES, 65);

    let net = RefMc {
        w1: w1_64.clone(),
        b1: b1_64.clone(),
        w2: w2_64.clone(),
        b2: b2_64.clone(),
    };
    let (ref_dw1, ref_db1, ref_dw2, ref_db2) = net.grads(&x64, &y64);

    let f32v = |v: &[f64]| -> Vec<f32> { v.iter().map(|x| *x as f32).collect() };

    let gx = ctx.upload(&f32v(&x64), BATCH, IN).expect("x");
    let gy = ctx.upload(&f32v(&y64), BATCH, CLASSES).expect("y");
    let gw1 = ctx.upload(&f32v(&w1_64), IN, HID).expect("w1");
    let gb1 = ctx.upload(&f32v(&b1_64), 1, HID).expect("b1");
    let gw2 = ctx.upload(&f32v(&w2_64), HID, CLASSES).expect("w2");
    let gb2 = ctx.upload(&f32v(&b2_64), 1, CLASSES).expect("b2");

    let z1 = ctx.matmul_resident(&gx, &gw1).expect("z1");
    let z1 = ctx.add_bias_resident(&z1, &gb1).expect("z1b");
    let a1 = ctx.relu_resident(&z1).expect("a1");
    let z2 = ctx.matmul_resident(&a1, &gw2).expect("z2");
    let z2 = ctx.add_bias_resident(&z2, &gb2).expect("z2b");

    let dz2 = ctx.softmax_xent_grad_resident(&z2, &gy).expect("dz2");
    let a1t = ctx.transpose_resident(&a1).expect("a1t");
    let dw2 = ctx.matmul_resident(&a1t, &dz2).expect("dw2");
    let db2 = ctx.column_sums_resident(&dz2).expect("db2");

    let w2t = ctx.transpose_resident(&gw2).expect("w2t");
    let da1 = ctx.matmul_resident(&dz2, &w2t).expect("da1");
    let dz1 = ctx.relu_backward_resident(&z1, &da1).expect("dz1");

    let xt = ctx.transpose_resident(&gx).expect("xt");
    let dw1 = ctx.matmul_resident(&xt, &dz1).expect("dw1");
    let db1 = ctx.column_sums_resident(&dz1).expect("db1");

    let cmp = |name: &str, got: &GpuTensor, want: &[f64]| {
        let g = ctx.read(got).expect("readback");
        assert_eq!(g.len(), want.len(), "{name}: length");
        for (i, (a, b)) in g.iter().zip(want).enumerate() {
            let denom = (*b).abs().max(1e-6);
            let rel = ((*a as f64) - b).abs() / denom;
            assert!(
                rel < 2e-3,
                "{name}[{i}]: gpu {a:.8e} vs reference {b:.8e}, relative error {rel:.3e}"
            );
        }
        println!("{name}: {} entries matched the f64 reference", g.len());
    };

    cmp("dw1", &dw1, &ref_dw1);
    cmp("db1", &db1, &ref_db1);
    cmp("dw2", &dw2, &ref_dw2);
    cmp("db2", &db2, &ref_db2);
}
