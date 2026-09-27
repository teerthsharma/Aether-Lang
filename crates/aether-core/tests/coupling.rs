//! Invariant tests for `aether_core::coupling`.
//!
//! The coupling operator is small enough that a bug in it does not crash: a
//! transposed block, a lift column out of order, a spectral radius read where a
//! spectral norm was meant, or a `<` where `≤` belongs all return plausible
//! numbers. Each test below is a property one of those bugs violates.
//!
//! Properties asserted here:
//!
//! 1. Naive reference: `step` equals the explicit product `W φ(z, a)` bit for
//!    bit, and the ridge fit equals a Gauss-Jordan normal-equation solve.
//! 2. Operator algebra: affine in each argument, not jointly; not linear;
//!    rollouts compose.
//! 3. Reduction: zero coupling is the uncoupled dynamics.
//! 4. Contraction: ‖Tz − Tw‖ ≤ ρ‖z − w‖, attained, and geometric convergence.
//! 5. Certificate: the scalar bound is the geometric series, holds on bounded
//!    residuals and is attained; the directional estimate matches its closed
//!    form.
//! 6. Relabelling: the fit is equivariant under permuting state components,
//!    islands under permuting points, strength under permuting bodies.
//! 7. Islands: equal to a brute-force flood fill, and to β₀ of the H₀ barcode.
//! 8. Refusal: non-finite and malformed input is refused, never propagated.

use aether_core::coupling::{
    island_labels, CouplingError, CouplingOperator, IslandMetric, RolloutCertificate,
};
use aether_core::manifold::ManifoldPoint;
use aether_core::persistence::{persistent_homology, PersistenceConfig};

// ═══════════════════════════════════════════════════════════════════════════════
// Deterministic sampling
// ═══════════════════════════════════════════════════════════════════════════════

/// xorshift64*. Seeded per test so a failure is reproducible from the seed alone.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in [-1, 1).
    fn signed(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }

    /// Standard normal via Box-Muller.
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }

    fn vector<const N: usize>(&mut self, scale: f64) -> [f64; N] {
        core::array::from_fn(|_| scale * self.normal())
    }

    fn matrix<const N: usize>(&mut self, scale: f64) -> [[f64; N]; N] {
        core::array::from_fn(|_| self.vector(scale))
    }

    /// Fisher-Yates permutation of `0..n`.
    fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            p.swap(i, j);
        }
        p
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Naive references
// ═══════════════════════════════════════════════════════════════════════════════

fn random_operator<const D: usize, const K: usize>(
    rng: &mut Rng,
    scale: f64,
) -> CouplingOperator<D, K> {
    CouplingOperator {
        state: rng.matrix(scale),
        bilinear: core::array::from_fn(|_| rng.matrix(scale)),
        input: core::array::from_fn(|_| rng.vector(scale)),
        bias: rng.vector(scale),
    }
}

/// `φ(z, a) = [z ; a ⊗ z ; a ; 1]`, written out.
fn naive_lift<const D: usize, const K: usize>(z: &[f64; D], a: &[f64; K]) -> Vec<f64> {
    let mut phi = z.to_vec();
    for ak in a {
        for zj in z {
            phi.push(ak * zj);
        }
    }
    phi.extend_from_slice(a);
    phi.push(1.0);
    phi
}

/// The dense `D × L` matrix `W` whose column blocks are the operator's blocks.
fn naive_w<const D: usize, const K: usize>(op: &CouplingOperator<D, K>) -> Vec<Vec<f64>> {
    (0..D)
        .map(|i| {
            let mut row = op.state[i].to_vec();
            for t in &op.bilinear {
                row.extend_from_slice(&t[i]);
            }
            row.extend_from_slice(&op.input[i]);
            row.push(op.bias[i]);
            row
        })
        .collect()
}

/// Solve `a x = b` for a matrix of right-hand sides by Gauss-Jordan elimination.
fn gauss_jordan(mut a: Vec<Vec<f64>>, mut b: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))
            .unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        let lead = a[col][col];
        for v in a[col].iter_mut() {
            *v /= lead;
        }
        for v in b[col].iter_mut() {
            *v /= lead;
        }
        for r in 0..n {
            if r != col {
                let f = a[r][col];
                let (arow, brow) = (a[col].clone(), b[col].clone());
                for (v, p) in a[r].iter_mut().zip(&arow) {
                    *v -= f * p;
                }
                for (v, p) in b[r].iter_mut().zip(&brow) {
                    *v -= f * p;
                }
            }
        }
    }
    b
}

/// `σ_max` by 20 000 rounds of power iteration on `AᵀA`, and the top right
/// singular vector it converged to.
fn power_sigma_max<const D: usize>(a: &[[f64; D]; D]) -> (f64, [f64; D]) {
    let mut v = [1.0 / (D as f64).sqrt(); D];
    let mut lambda = 0.0;
    for _ in 0..20_000 {
        let av: [f64; D] = core::array::from_fn(|i| (0..D).map(|j| a[i][j] * v[j]).sum());
        let w: [f64; D] = core::array::from_fn(|j| (0..D).map(|i| a[i][j] * av[i]).sum());
        let norm = norm(&w);
        lambda = norm;
        v = core::array::from_fn(|i| w[i] / norm);
    }
    (lambda.sqrt(), v)
}

/// A random orthogonal matrix by modified Gram-Schmidt, so nothing is
/// accidentally axis-aligned.
fn orthogonal<const D: usize>(rng: &mut Rng) -> [[f64; D]; D] {
    let mut q = rng.matrix::<D>(1.0);
    for i in 0..D {
        for j in 0..i {
            let prev = q[j];
            let dot: f64 = q[i].iter().zip(&prev).map(|(a, b)| a * b).sum();
            for (x, p) in q[i].iter_mut().zip(&prev) {
                *x -= dot * p;
            }
        }
        let n = norm(&q[i]);
        for x in q[i].iter_mut() {
            *x /= n;
        }
    }
    q
}

fn scaled<const D: usize>(m: &[[f64; D]; D], c: f64) -> [[f64; D]; D] {
    core::array::from_fn(|i| core::array::from_fn(|j| c * m[i][j]))
}

fn matvec<const D: usize>(m: &[[f64; D]; D], v: &[f64; D]) -> [f64; D] {
    core::array::from_fn(|i| (0..D).map(|j| m[i][j] * v[j]).sum())
}

fn norm<const D: usize>(v: &[f64; D]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn rms<const D: usize>(v: &[f64; D]) -> f64 {
    norm(v) / (D as f64).sqrt()
}

fn sub<const D: usize>(a: &[f64; D], b: &[f64; D]) -> [f64; D] {
    core::array::from_fn(|i| a[i] - b[i])
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()).max(1.0)
}

/// Island labels by breadth-first flood fill over the explicit adjacency matrix,
/// the quadratic construction `mujoco#3396` replaced with a disjoint-set union.
fn flood_fill<const N: usize>(
    points: &[ManifoldPoint<N>],
    r: f64,
    metric: IslandMetric,
) -> Vec<usize> {
    let n = points.len();
    let dist = |i: usize, j: usize| -> f64 {
        let (a, b) = (&points[i].coords, &points[j].coords);
        match metric {
            IslandMetric::Euclidean => a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt(),
            IslandMetric::Geodesic => a
                .iter()
                .zip(b)
                .map(|(x, y)| x * y)
                .sum::<f64>()
                .clamp(-1.0, 1.0)
                .acos(),
        }
    };
    let mut labels = vec![usize::MAX; n];
    let mut next = 0;
    for start in 0..n {
        if labels[start] != usize::MAX {
            continue;
        }
        labels[start] = next;
        let mut queue = vec![start];
        while let Some(i) = queue.pop() {
            for (j, label) in labels.iter_mut().enumerate() {
                if *label == usize::MAX && i != j && dist(i.min(j), i.max(j)) <= r {
                    *label = next;
                    queue.push(j);
                }
            }
        }
        next += 1;
    }
    labels
}

/// Points in `k` gaussian clusters, with every fifth point duplicated exactly so
/// coincident points exercise the closed ball at radius zero.
fn clustered_cloud(n: usize, k: usize, rng: &mut Rng) -> Vec<ManifoldPoint<3>> {
    let centers: Vec<[f64; 3]> = (0..k)
        .map(|_| core::array::from_fn(|_| 3.0 * rng.signed()))
        .collect();
    let mut out: Vec<ManifoldPoint<3>> = Vec::with_capacity(n);
    for i in 0..n {
        if i % 5 == 4 {
            out.push(out[i - 1]);
            continue;
        }
        let c = centers[(rng.next_u64() % k as u64) as usize];
        out.push(ManifoldPoint::new(core::array::from_fn(|d| {
            c[d] + 0.3 * rng.normal()
        })));
    }
    out
}

fn on_sphere(points: &[ManifoldPoint<3>]) -> Vec<ManifoldPoint<3>> {
    points
        .iter()
        .map(|p| {
            let n = norm(&p.coords);
            ManifoldPoint::new(core::array::from_fn(|d| p.coords[d] / n))
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Naive reference
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn step_matches_the_naive_lifted_matvec_bit_for_bit() {
    // The operator is defined as W φ(z, a). `step` never forms φ, so a column
    // block read in the wrong order (a ⊗ z indexed j·K + k, say) is invisible
    // unless compared against the product written out.
    let mut rng = Rng::new(11);
    for _ in 0..50 {
        let op: CouplingOperator<4, 3> = random_operator(&mut rng, 1.0);
        let z: [f64; 4] = rng.vector(3.0);
        let a: [f64; 3] = rng.vector(2.0);
        let (w, phi) = (naive_w(&op), naive_lift(&z, &a));
        assert_eq!(phi.len(), 4 + 3 * 4 + 3 + 1);

        let got = op.step(&z, &a);
        for i in 0..4 {
            let mut acc = 0.0;
            for (wl, pl) in w[i].iter().zip(&phi) {
                acc += wl * pl;
            }
            assert_eq!(
                got[i].to_bits(),
                acc.to_bits(),
                "row {i}: {} vs {acc}",
                got[i]
            );
        }
    }
}

#[test]
fn ridge_fit_matches_a_naive_normal_equation_solve() {
    // W = (XᵀX + λI)⁻¹ XᵀY, solved here by Gauss-Jordan on the explicit design
    // matrix. Targets are unrelated to the inputs, so this is regression and not
    // recovery: nothing but the solve itself can make the two agree.
    let mut rng = Rng::new(23);
    let n = 60;
    let states: Vec<[f64; 3]> = (0..n).map(|_| rng.vector(1.0)).collect();
    let actions: Vec<[f64; 2]> = (0..n).map(|_| rng.vector(1.0)).collect();
    let next: Vec<[f64; 3]> = (0..n).map(|_| rng.vector(1.0)).collect();
    let ridge = 0.05;

    let fit = CouplingOperator::<3, 2>::fit(&states, &next, &actions, ridge, None).unwrap();

    let x: Vec<Vec<f64>> = (0..n)
        .map(|t| naive_lift(&states[t], &actions[t]))
        .collect();
    let l = x[0].len();
    let gram: Vec<Vec<f64>> = (0..l)
        .map(|p| {
            (0..l)
                .map(|q| {
                    (0..n).map(|t| x[t][p] * x[t][q]).sum::<f64>()
                        + if p == q { ridge } else { 0.0 }
                })
                .collect()
        })
        .collect();
    let xty: Vec<Vec<f64>> = (0..l)
        .map(|p| {
            (0..3)
                .map(|i| (0..n).map(|t| x[t][p] * next[t][i]).sum())
                .collect()
        })
        .collect();
    let wt = gauss_jordan(gram, xty);

    let got = naive_w(&fit.operator);
    for i in 0..3 {
        for p in 0..l {
            assert!(
                close(got[i][p], wt[p][i], 1e-10),
                "W[{i}][{p}] = {} vs naive {}",
                got[i][p],
                wt[p][i]
            );
        }
    }

    let mut energy = 0.0;
    for t in 0..n {
        for i in 0..3 {
            let pred: f64 = (0..l).map(|p| wt[p][i] * x[t][p]).sum();
            energy += (next[t][i] - pred).powi(2);
        }
    }
    let naive_rmse = (energy / (3 * n) as f64).sqrt();
    assert!(
        close(fit.step_rmse, naive_rmse, 1e-10),
        "{} vs {naive_rmse}",
        fit.step_rmse
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Operator algebra
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn step_is_affine_in_each_argument_and_not_jointly() {
    // f(z, a) = T₀z + Σ a_k T_k z + Ba + c. Affine in z at fixed a, affine in a at
    // fixed z, never linear (f(z₁+z₂) − f(z₁) − f(z₂) = −c), and jointly the
    // midpoint misses the average by exactly −¼ Σ_k (a₁−a₂)_k T_k (z₁−z₂). A
    // missing or transposed bilinear block breaks the last identity.
    let mut rng = Rng::new(31);
    for _ in 0..20 {
        let op: CouplingOperator<3, 2> = random_operator(&mut rng, 1.0);
        let (z1, z2): ([f64; 3], [f64; 3]) = (rng.vector(1.0), rng.vector(1.0));
        let (a1, a2): ([f64; 2], [f64; 2]) = (rng.vector(1.0), rng.vector(1.0));
        let lam = 0.3;
        let mix = |x: &[f64; 3], y: &[f64; 3]| -> [f64; 3] {
            core::array::from_fn(|i| lam * x[i] + (1.0 - lam) * y[i])
        };
        let mix_a: [f64; 2] = core::array::from_fn(|k| lam * a1[k] + (1.0 - lam) * a2[k]);

        let in_z = op.step(&mix(&z1, &z2), &a1);
        let expected_z = mix(&op.step(&z1, &a1), &op.step(&z2, &a1));
        let in_a = op.step(&z1, &mix_a);
        let expected_a = mix(&op.step(&z1, &a1), &op.step(&z1, &a2));

        let zero = [0.0; 2];
        let sum: [f64; 3] = core::array::from_fn(|i| z1[i] + z2[i]);
        let (fs, f1, f2) = (
            op.step(&sum, &zero),
            op.step(&z1, &zero),
            op.step(&z2, &zero),
        );

        let mid_z: [f64; 3] = core::array::from_fn(|i| 0.5 * (z1[i] + z2[i]));
        let mid_a: [f64; 2] = core::array::from_fn(|k| 0.5 * (a1[k] + a2[k]));
        let at_mid = op.step(&mid_z, &mid_a);
        let (g1, g2) = (op.step(&z1, &a1), op.step(&z2, &a2));
        let dz = sub(&z1, &z2);
        let cross: [f64; 3] = core::array::from_fn(|i| {
            (0..2)
                .map(|k| {
                    (a1[k] - a2[k]) * (0..3).map(|j| op.bilinear[k][i][j] * dz[j]).sum::<f64>()
                })
                .sum()
        });

        for i in 0..3 {
            assert!(close(in_z[i], expected_z[i], 1e-12), "not affine in z");
            assert!(close(in_a[i], expected_a[i], 1e-12), "not affine in a");
            assert!(
                close(fs[i] - f1[i] - f2[i], -op.bias[i], 1e-12),
                "linearity defect is not −c"
            );
            let gap = at_mid[i] - 0.5 * (g1[i] + g2[i]);
            assert!(
                close(gap, -0.25 * cross[i], 1e-12),
                "joint defect {gap} vs {}",
                -0.25 * cross[i]
            );
        }
        assert!(
            norm(&cross) > 1e-3,
            "the bilinear defect must be visible to be tested"
        );
    }
}

#[test]
fn rollout_is_the_composition_of_single_steps() {
    // rollout(n) = step ∘ … ∘ step, and a rollout split at m resumes exactly
    // where it stopped: an off-by-one in the action index shifts every step.
    let mut rng = Rng::new(37);
    let op: CouplingOperator<3, 1> = random_operator(&mut rng, 0.4);
    let z0: [f64; 3] = rng.vector(1.0);
    let actions: Vec<[f64; 1]> = (0..20).map(|_| rng.vector(1.0)).collect();

    let mut out = [[0.0; 3]; 20];
    op.rollout(&z0, &actions, &mut out).unwrap();

    let mut z = z0;
    for (i, a) in actions.iter().enumerate() {
        z = op.step(&z, a);
        assert_eq!(out[i], z, "step {i}");
    }

    let (mut head, mut tail) = ([[0.0; 3]; 7], [[0.0; 3]; 13]);
    op.rollout(&z0, &actions[..7], &mut head).unwrap();
    op.rollout(&head[6], &actions[7..], &mut tail).unwrap();
    assert_eq!(&head[..], &out[..7]);
    assert_eq!(&tail[..], &out[7..]);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Reduction to the uncoupled dynamics
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn zero_coupling_reduces_to_the_uncoupled_dynamics() {
    let mut rng = Rng::new(41);

    // No action coupling: every action gives the autonomous step.
    let (a0, c0) = (rng.matrix::<3>(0.5), rng.vector::<3>(1.0));
    let driven = CouplingOperator::<3, 2>::autonomous(a0, c0);
    let free = CouplingOperator::<3, 0>::autonomous(a0, c0);
    for _ in 0..20 {
        let (z, a): ([f64; 3], [f64; 2]) = (rng.vector(1.0), rng.vector(5.0));
        assert_eq!(driven.step(&z, &a), free.step(&z, &[]));
    }

    // No cross-body blocks: the joint rollout is two independent rollouts, and
    // the coupling strength between the bodies is exactly zero.
    let (b1, b2) = (
        scaled(&orthogonal::<3>(&mut rng), 0.7),
        scaled(&orthogonal::<3>(&mut rng), 0.9),
    );
    let (c1, c2) = (rng.vector::<3>(1.0), rng.vector::<3>(1.0));
    let mut joint_state = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            joint_state[i][j] = b1[i][j];
            joint_state[i + 3][j + 3] = b2[i][j];
        }
    }
    let joint = CouplingOperator::<6, 0>::autonomous(
        joint_state,
        core::array::from_fn(|i| if i < 3 { c1[i] } else { c2[i - 3] }),
    );
    let (z1, z2) = (rng.vector::<3>(1.0), rng.vector::<3>(1.0));
    let z0: [f64; 6] = core::array::from_fn(|i| if i < 3 { z1[i] } else { z2[i - 3] });

    let mut together = [[0.0; 6]; 25];
    let (mut alone1, mut alone2) = ([[0.0; 3]; 25], [[0.0; 3]; 25]);
    joint.rollout(&z0, &[], &mut together).unwrap();
    CouplingOperator::<3, 0>::autonomous(b1, c1)
        .rollout(&z1, &[], &mut alone1)
        .unwrap();
    CouplingOperator::<3, 0>::autonomous(b2, c2)
        .rollout(&z2, &[], &mut alone2)
        .unwrap();
    for t in 0..25 {
        assert_eq!(
            together[t][..3],
            alone1[t],
            "body 1 felt body 2 at step {t}"
        );
        assert_eq!(
            together[t][3..],
            alone2[t],
            "body 2 felt body 1 at step {t}"
        );
    }

    let mut strength = [f64::NAN; 4];
    assert_eq!(joint.coupling_strength(3, &mut strength), Ok(2));
    assert_eq!((strength[1], strength[2]), (0.0, 0.0));
    assert!(
        close(strength[0], 0.7 * 3f64.sqrt(), 1e-12),
        "‖0.7 Q‖_F = 0.7 √3, got {}",
        strength[0]
    );
    assert!(close(strength[3], 0.9 * 3f64.sqrt(), 1e-12));

    // Identity coupling with no offset carries the state forward unchanged.
    let mut identity = [[0.0; 4]; 4];
    for (i, row) in identity.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let carry = CouplingOperator::<4, 0>::autonomous(identity, [0.0; 4]);
    let z: [f64; 4] = rng.vector(1.0);
    let mut out = [[0.0; 4]; 10];
    carry.rollout(&z, &[], &mut out).unwrap();
    assert!(out.iter().all(|s| *s == z));
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Spectral norm, projection, contraction
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn spectral_norm_matches_power_iteration_and_closed_forms() {
    let mut rng = Rng::new(43);
    for _ in 0..10 {
        let a = rng.matrix::<6>(1.0);
        let (reference, _) = power_sigma_max(&a);
        let got = CouplingOperator::<6, 0>::autonomous(a, [0.0; 6]).spectral_norm();
        assert!(
            close(got, reference, 1e-9),
            "σ_max {got} vs power iteration {reference}"
        );
    }

    // σ_max(cQ) = |c| for orthogonal Q, whatever the sign of c.
    let q = orthogonal::<5>(&mut rng);
    for c in [0.3, -1.7, 4.0] {
        let got = CouplingOperator::<5, 0>::autonomous(scaled(&q, c), [0.0; 5]).spectral_norm();
        assert!(close(got, c.abs(), 1e-12), "σ_max({c}·Q) = {got}");
    }

    // A norm, not a spectral radius: [[1, 10], [0, 1]] has both eigenvalues 1 and
    // σ_max = (√104 + 10)/2. Reading ρ off the eigenvalues would certify an
    // operator that stretches some vector tenfold.
    let shear = CouplingOperator::<2, 0>::autonomous([[1.0, 10.0], [0.0, 1.0]], [0.0; 2]);
    let expected = (104f64.sqrt() + 10.0) / 2.0;
    assert!(
        close(shear.spectral_norm(), expected, 1e-12),
        "{}",
        shear.spectral_norm()
    );
}

#[test]
fn spectral_projection_is_a_uniform_rescale_onto_the_ceiling() {
    // Scaling every singular value by ρ_max/ρ is scaling T₀ by it. Singular
    // vectors, the action blocks and the bias must not move.
    let mut rng = Rng::new(47);
    let original: CouplingOperator<5, 1> = random_operator(&mut rng, 1.0);
    let rho = original.spectral_norm();

    let mut projected = original;
    assert_eq!(projected.project_spectral(rho / 2.0), Ok(rho / 2.0));
    for i in 0..5 {
        for j in 0..5 {
            assert!(close(
                projected.state[i][j],
                0.5 * original.state[i][j],
                1e-14
            ));
        }
    }
    assert_eq!(
        (projected.bilinear, projected.input, projected.bias),
        (original.bilinear, original.input, original.bias)
    );
    assert!(close(projected.spectral_norm(), rho / 2.0, 1e-12));

    let mut untouched = original;
    assert_eq!(untouched.project_spectral(2.0 * rho), Ok(rho));
    assert_eq!(untouched, original, "a ceiling above ρ must change nothing");

    // Through the fit: a system with ρ = 3 comes out at exactly the ceiling.
    let states: Vec<[f64; 8]> = (0..200).map(|_| rng.vector(1.0)).collect();
    let next: Vec<[f64; 8]> = states
        .iter()
        .map(|z| core::array::from_fn(|i| 3.0 * z[i]))
        .collect();
    let fit = CouplingOperator::<8, 0>::fit(&states, &next, &[], 1e-3, Some(0.9)).unwrap();
    assert_eq!(fit.rho, 0.9);
    assert!(fit.certificate().contractive());
    assert!(fit.operator.spectral_norm() <= 0.9 + 1e-9);
}

#[test]
fn contraction_inequality_holds_on_seeded_random_states_and_is_attained() {
    // ‖T z − T w‖ ≤ ρ ‖z − w‖ for every pair, with equality along the top right
    // singular vector. The equality half catches an over-shrinking projection,
    // such as one that divided by the Frobenius norm instead of σ_max.
    let mut rng = Rng::new(53);
    let mut op: CouplingOperator<6, 0> = random_operator(&mut rng, 1.0);
    op.project_spectral(0.8).unwrap();
    let rho = 0.8;

    for _ in 0..500 {
        let (z, w): ([f64; 6], [f64; 6]) = (rng.vector(2.0), rng.vector(2.0));
        let lhs = norm(&sub(&op.step(&z, &[]), &op.step(&w, &[])));
        assert!(
            lhs <= rho * norm(&sub(&z, &w)) * (1.0 + 1e-12),
            "{lhs} > ρ‖z − w‖"
        );
    }

    let (_, v) = power_sigma_max(&op.state);
    let stretched = norm(&matvec(&op.state, &v));
    assert!(
        close(stretched, rho, 1e-9),
        "‖T₀v‖ = {stretched}, expected ρ = {rho}"
    );

    let star = op.fixed_point().unwrap();
    assert!(star.converged);
    let z0: [f64; 6] = rng.vector(5.0);
    let mut out = [[0.0; 6]; 60];
    op.rollout(&z0, &[], &mut out).unwrap();
    let start = norm(&sub(&z0, &star.point));
    for (n, z) in out.iter().enumerate() {
        let bound = rho.powi(n as i32 + 1) * start + 1e-12;
        assert!(
            norm(&sub(z, &star.point)) <= bound,
            "step {} left the Banach envelope",
            n + 1
        );
    }
}

#[test]
fn fixed_point_is_fixed_and_convergence_is_measured_not_assumed() {
    let mut rng = Rng::new(59);

    // z_{t+1} = 0.5 z_t + 0.1 has z* = 0.2 in every coordinate.
    let states: Vec<[f64; 6]> = (0..300).map(|_| rng.vector(1.0)).collect();
    let next: Vec<[f64; 6]> = states
        .iter()
        .map(|z| core::array::from_fn(|i| 0.5 * z[i] + 0.1))
        .collect();
    let fit = CouplingOperator::<6, 0>::fit(&states, &next, &[], 1e-3, None).unwrap();
    let star = fit.operator.fixed_point().unwrap();
    assert!(star.converged);
    let image = fit.operator.step(&star.point, &[]);
    for (i, (moved, fixed)) in image.iter().zip(&star.point).enumerate() {
        assert!((moved - fixed).abs() < 1e-12, "T(z*) ≠ z*");
        assert!((fixed - 0.2).abs() < 1e-4, "z*[{i}] = {fixed}");
    }

    // Expansive: the algebraic fixed point exists, iteration never reaches it.
    let q = orthogonal::<4>(&mut rng);
    let expansive = CouplingOperator::<4, 0>::autonomous(scaled(&q, 1.5), rng.vector(1.0));
    let star = expansive.fixed_point().unwrap();
    assert!(!star.converged, "ρ = 1.5 cannot converge by iteration");
    assert!(norm(&sub(&expansive.step(&star.point, &[]), &star.point)) < 1e-12);

    // I − T₀ singular: no isolated fixed point, and none is invented.
    let mut identity = [[0.0; 3]; 3];
    for (i, row) in identity.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    assert!(
        CouplingOperator::<3, 0>::autonomous(identity, [1.0, 0.0, 0.0])
            .fixed_point()
            .is_none()
    );

    // The fixed point belongs to the a = 0 map: action blocks do not enter it.
    let mut driven: CouplingOperator<3, 2> = random_operator(&mut rng, 1.0);
    driven.state = scaled(&orthogonal::<3>(&mut rng), 0.6);
    let free = CouplingOperator::<3, 0>::autonomous(driven.state, driven.bias);
    assert_eq!(driven.fixed_point(), free.fixed_point());
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. Certificate
// ═══════════════════════════════════════════════════════════════════════════════

fn certificate<const D: usize>(
    rho: f64,
    eps: f64,
    a: [[f64; D]; D],
    sigma: [[f64; D]; D],
) -> RolloutCertificate<D> {
    RolloutCertificate {
        rho,
        step_rmse: eps,
        state_block: a,
        residual_cov: sigma,
    }
}

#[test]
fn scalar_bound_is_the_geometric_series_holds_on_bounded_residuals_and_is_attained() {
    let zero = [[0.0; 1]; 1];
    let c = certificate(0.9, 0.3, zero, zero);
    assert_eq!(c.error_bound(0), 0.0);
    let mut last = 0.0;
    for n in 1..200 {
        let closed = 0.3 * (1.0 - 0.9f64.powi(n as i32)) / 0.1;
        assert!(close(c.error_bound(n), closed, 1e-13), "n={n}");
        assert!(
            c.error_bound(n) > last,
            "the bound must grow with the horizon"
        );
        last = c.error_bound(n);
    }
    assert!(c.error_bound(100_000) <= 3.0 + 1e-12, "limit is ε/(1 − ρ)");
    assert_eq!(certificate(1.0, 0.3, zero, zero).error_bound(7), 0.3 * 7.0);
    assert!(close(
        certificate(1.3, 0.3, zero, zero).error_bound(9),
        0.3 * 9.0 * 1.3f64.powi(8),
        1e-13
    ));
    assert_eq!(
        certificate(2.0, 0.3, zero, zero).error_bound(5000),
        f64::INFINITY
    );

    // Simulate z_{t+1} = T₀ z_t + c + r_t with rms(r_t) ≤ ε and roll the model
    // out without residuals. The error never exceeds the bound; with residuals
    // aligned to the rotation (r_i ∝ Qⁱu) every term adds in phase and the bound
    // is met exactly, which rules out a bound that is merely loose.
    let mut rng = Rng::new(61);
    let q = orthogonal::<5>(&mut rng);
    let eps = 0.05;
    for rho in [0.9, 1.3] {
        let op = CouplingOperator::<5, 0>::autonomous(scaled(&q, rho), rng.vector(1.0));
        let cert = certificate(op.spectral_norm(), eps, op.state, [[0.0; 5]; 5]);
        for trial in 0..40 {
            let aligned = trial == 0;
            let z0: [f64; 5] = rng.vector(1.0);
            let mut truth = z0;
            let mut model = [[0.0; 5]; 30];
            op.rollout(&z0, &[], &mut model).unwrap();
            let u: [f64; 5] = rng.vector(1.0);
            let mut direction = u;
            for (n, predicted) in model.iter().enumerate() {
                let r: [f64; 5] = if aligned {
                    let s = eps / rms(&direction);
                    core::array::from_fn(|i| s * direction[i])
                } else {
                    let raw: [f64; 5] = rng.vector(1.0);
                    let s = eps * rng.unit() / rms(&raw);
                    core::array::from_fn(|i| s * raw[i])
                };
                direction = matvec(&q, &direction);
                let stepped = op.step(&truth, &[]);
                truth = core::array::from_fn(|i| stepped[i] + r[i]);

                let err = rms(&sub(&truth, predicted));
                let bound = cert.error_bound(n + 1);
                assert!(
                    err <= bound * (1.0 + 1e-10),
                    "ρ={rho} n={}: {err} > {bound}",
                    n + 1
                );
                if aligned && rho < 1.0 {
                    assert!(
                        close(err, bound, 1e-9),
                        "aligned residuals must attain the bound: {err} vs {bound}"
                    );
                }
            }
        }
    }
}

#[test]
fn directional_estimate_matches_its_closed_form_and_the_fit_statistics() {
    // T₀ = ρQ, Σ = s²I: C_n = s² Σ_{i<n} ρ^{2i} I, so the estimate is
    // s √((1 − ρ^{2n})/(1 − ρ²)) exactly. Tight because it is right, or wrong.
    let mut rng = Rng::new(67);
    let (rho, s) = (0.9, 0.3);
    let mut sigma = [[0.0; 8]; 8];
    for (i, row) in sigma.iter_mut().enumerate() {
        row[i] = s * s;
    }
    let cert = certificate(rho, s, scaled(&orthogonal::<8>(&mut rng), rho), sigma);
    assert_eq!(cert.directional_error(0), 0.0);
    for n in [1usize, 4, 16] {
        let closed = s * ((1.0 - rho.powi(2 * n as i32)) / (1.0 - rho * rho)).sqrt();
        assert!(
            close(cert.directional_error(n), closed, 1e-12),
            "n={n}: {} vs {closed}",
            cert.directional_error(n)
        );
    }

    // From a fit: tr(Σ)/D = step_rmse², so both estimates agree at one step and
    // any later difference is propagation, not calibration.
    let simulate = |a: [[f64; 4]; 4], phi: f64, rng: &mut Rng| -> Vec<[f64; 4]> {
        let (mut z, mut r) = ([0.0; 4], [0.0; 4]);
        let mut out = Vec::new();
        for _ in 0..6000 {
            out.push(z);
            let w: [f64; 4] = rng.vector(0.2 * (1.0 - phi * phi).sqrt());
            r = core::array::from_fn(|i| phi * r[i] + w[i]);
            let az = matvec(&a, &z);
            z = core::array::from_fn(|i| az[i] + r[i]);
        }
        out.split_off(1000)
    };
    let iid = simulate(scaled(&orthogonal::<4>(&mut rng), 0.7), 0.0, &mut rng);
    let fit =
        CouplingOperator::<4, 0>::fit(&iid[..iid.len() - 1], &iid[1..], &[], 1e-8, None).unwrap();
    let trace: f64 = (0..4).map(|i| fit.residual_cov[i][i]).sum();
    assert!(close(trace / 4.0, fit.step_rmse * fit.step_rmse, 1e-12));
    let c = fit.certificate();
    assert!(close(c.directional_error(1), c.error_bound(1), 1e-12));
    assert!(
        fit.residual_autocorr.abs() < 0.05,
        "iid residuals read {}",
        fit.residual_autocorr
    );

    // Step-correlated residuals violate the estimate's assumption, and the
    // autocorrelation is the diagnostic that says so. T₀ = 0.9·I with AR(1)
    // residuals of persistence 0.9 is the source's configuration.
    let mut slow = [[0.0; 4]; 4];
    for (i, row) in slow.iter_mut().enumerate() {
        row[i] = 0.9;
    }
    let ar = simulate(slow, 0.9, &mut rng);
    let fit =
        CouplingOperator::<4, 0>::fit(&ar[..ar.len() - 1], &ar[1..], &[], 1e-8, None).unwrap();
    assert!(
        fit.residual_autocorr > 0.5,
        "AR(1) residuals read {}",
        fit.residual_autocorr
    );
}

#[test]
fn safe_horizon_inverts_the_bound_and_treats_nan_as_unsafe() {
    let mut rng = Rng::new(71);
    let mut sigma = [[0.0; 3]; 3];
    for (i, row) in sigma.iter_mut().enumerate() {
        row[i] = 0.09;
    }
    let c = certificate(0.9, 0.3, scaled(&orthogonal::<3>(&mut rng), 0.9), sigma);
    for tol in [0.1, 0.25, 0.5, 1.0, 2.0, 2.9] {
        let h = c.safe_horizon(tol, false);
        assert!(
            h == 0 || c.error_bound(h) <= tol,
            "tol {tol}: bound at h={h} exceeds it"
        );
        assert!(
            h == 4096 || c.error_bound(h + 1) > tol,
            "tol {tol}: h={h} is not the largest"
        );
        let hd = c.safe_horizon(tol, true);
        assert!(hd >= h, "the estimate is never charged more than the bound");
        assert!(hd == 0 || c.directional_error(hd) <= tol);
        assert!(hd == 4096 || c.directional_error(hd + 1) > tol);
    }
    assert_eq!(
        c.safe_horizon(0.1, false),
        0,
        "a tolerance below ε allows nothing"
    );
    assert_eq!(
        c.safe_horizon(3.0 + 1e-9, false),
        4096,
        "past ε/(1 − ρ) every horizon is safe, up to the cap"
    );

    let nan_eps = certificate(0.9, f64::NAN, c.state_block, sigma);
    assert_eq!(nan_eps.safe_horizon(1e9, false), 0);
    let nan_sigma = certificate(0.9, 0.3, c.state_block, [[f64::NAN; 3]; 3]);
    assert_eq!(nan_sigma.safe_horizon(1e9, true), 0);
    assert_eq!(c.safe_horizon(f64::NAN, false), 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. Relabelling
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn fit_is_equivariant_under_relabelling_of_state_components() {
    // Permuting coordinates by σ permutes every block: T₀'[i][j] = T₀[σi][σj].
    // The ridge penalty ‖W‖²_F is permutation invariant, so nothing else may
    // change — ρ, ε, the residual moments and the fixed point included.
    let mut rng = Rng::new(73);
    let truth: CouplingOperator<4, 1> = random_operator(&mut rng, 0.5);
    let n = 200;
    let states: Vec<[f64; 4]> = (0..n).map(|_| rng.vector(1.0)).collect();
    let actions: Vec<[f64; 1]> = (0..n).map(|_| rng.vector(1.0)).collect();
    let next: Vec<[f64; 4]> = (0..n)
        .map(|t| {
            let y = truth.step(&states[t], &actions[t]);
            let noise: [f64; 4] = rng.vector(0.05);
            core::array::from_fn(|i| y[i] + noise[i])
        })
        .collect();
    let sigma = [2usize, 0, 3, 1];
    let permute = |v: &[f64; 4]| -> [f64; 4] { core::array::from_fn(|i| v[sigma[i]]) };

    let base = CouplingOperator::<4, 1>::fit(&states, &next, &actions, 1e-3, None).unwrap();
    let states_p: Vec<[f64; 4]> = states.iter().map(permute).collect();
    let next_p: Vec<[f64; 4]> = next.iter().map(permute).collect();
    let moved = CouplingOperator::<4, 1>::fit(&states_p, &next_p, &actions, 1e-3, None).unwrap();

    let (a, b) = (&base.operator, &moved.operator);
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                close(b.state[i][j], a.state[sigma[i]][sigma[j]], 1e-9),
                "T₀[{i}][{j}]"
            );
            assert!(
                close(b.bilinear[0][i][j], a.bilinear[0][sigma[i]][sigma[j]], 1e-9),
                "T₁[{i}][{j}]"
            );
            assert!(close(
                moved.residual_cov[i][j],
                base.residual_cov[sigma[i]][sigma[j]],
                1e-9
            ));
        }
        assert!(close(b.input[i][0], a.input[sigma[i]][0], 1e-9));
        assert!(close(b.bias[i], a.bias[sigma[i]], 1e-9));
    }
    assert!(close(moved.rho, base.rho, 1e-9));
    assert!(close(moved.step_rmse, base.step_rmse, 1e-9));
    assert!(close(moved.residual_autocorr, base.residual_autocorr, 1e-9));
    let (za, zb) = (
        a.fixed_point().unwrap().point,
        b.fixed_point().unwrap().point,
    );
    for i in 0..4 {
        assert!(close(zb[i], za[sigma[i]], 1e-9), "z*[{i}]");
    }
}

#[test]
fn coupling_strength_recovers_a_driver_follower_pair_under_either_labelling() {
    // Body 0 follows body 1: b0_{t+1} = 0.5 b0_t + 0.5 b1_t, exactly. The rows
    // of body 0 are noiseless, so their blocks are 0.5·I₄ each, of Frobenius norm
    // 1. A transposed block reports the coupling from the wrong body.
    let mut rng = Rng::new(79);
    let steps = 400;
    let mut b1 = vec![[0.0; 4]; steps];
    for t in 1..steps {
        let d: [f64; 4] = rng.vector(0.1);
        b1[t] = core::array::from_fn(|i| b1[t - 1][i] + d[i]);
    }
    let peak = b1.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs())) + 1e-9;
    for row in b1.iter_mut() {
        for x in row.iter_mut() {
            *x /= peak;
        }
    }
    let mut b0 = vec![[0.0; 4]; steps];
    for t in 1..steps {
        b0[t] = core::array::from_fn(|i| 0.5 * b0[t - 1][i] + 0.5 * b1[t - 1][i]);
    }
    let join = |x: &[[f64; 4]], y: &[[f64; 4]]| -> Vec<[f64; 8]> {
        x.iter()
            .zip(y)
            .map(|(p, q)| core::array::from_fn(|i| if i < 4 { p[i] } else { q[i - 4] }))
            .collect()
    };

    let strength = |joint: &[[f64; 8]]| -> [f64; 4] {
        let fit = CouplingOperator::<8, 0>::fit(&joint[..steps - 1], &joint[1..], &[], 1e-6, None)
            .unwrap();
        let mut s = [0.0; 4];
        assert_eq!(fit.operator.coupling_strength(4, &mut s), Ok(2));
        s
    };
    let s = strength(&join(&b0, &b1));
    assert!(
        (s[1] - 1.0).abs() < 1e-3,
        "follower ← driver block {}",
        s[1]
    );
    assert!((s[0] - 1.0).abs() < 1e-3, "follower self block {}", s[0]);

    // Swap the bodies: the matrix is conjugated by the swap, nothing else.
    let swapped = strength(&join(&b1, &b0));
    let expected = [s[3], s[2], s[1], s[0]];
    for (got, want) in swapped.iter().zip(expected) {
        assert!(close(*got, want, 1e-9), "{swapped:?} vs {expected:?}");
    }
}

#[test]
fn island_partition_is_invariant_under_relabelling_of_points() {
    for seed in [3u64, 17, 29, 101] {
        let mut rng = Rng::new(seed);
        let points = clustered_cloud(25, 4, &mut rng);
        let perm = rng.permutation(points.len());
        let shuffled: Vec<_> = perm.iter().map(|&i| points[i]).collect();

        for r in [0.0, 0.3, 0.9, 2.5] {
            let (mut before, mut after) = (vec![0; 25], vec![0; 25]);
            let n0 = island_labels(&points, r, IslandMetric::Euclidean, &mut before).unwrap();
            let n1 = island_labels(&shuffled, r, IslandMetric::Euclidean, &mut after).unwrap();
            assert_eq!(
                n0, n1,
                "seed {seed}, r {r}: island count changed under relabelling"
            );
            for i in 0..25 {
                for j in 0..25 {
                    assert_eq!(
                        after[i] == after[j],
                        before[perm[i]] == before[perm[j]],
                        "seed {seed}, r {r}: points {i}, {j} changed islands"
                    );
                }
            }
            // Canonical: ids appear in ascending order of first occurrence.
            let mut seen = 0;
            for &l in &after {
                assert!(l <= seen, "label {l} before label {seen}");
                if l == seen {
                    seen += 1;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. Islands
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn island_labels_match_a_brute_force_flood_fill() {
    for seed in [1u64, 2, 5, 8, 13, 21] {
        let mut rng = Rng::new(seed);
        let points = clustered_cloud(30, 5, &mut rng);
        let sphere = on_sphere(&points);
        let cases = [
            (
                &points,
                IslandMetric::Euclidean,
                [0.0, 0.1, 0.4, 0.8, 1.5, 3.0, 10.0],
            ),
            (
                &sphere,
                IslandMetric::Geodesic,
                [0.0, 0.05, 0.2, 0.5, 1.0, 2.0, core::f64::consts::PI],
            ),
        ];
        for (cloud, metric, radii) in cases {
            for r in radii {
                let mut labels = vec![0; cloud.len()];
                let count = island_labels(cloud, r, metric, &mut labels).unwrap();
                let reference = flood_fill(cloud, r, metric);
                assert_eq!(labels, reference, "seed {seed}, {metric:?}, r {r}");
                assert_eq!(count, reference.iter().max().unwrap() + 1);
            }
        }
    }
    let mut none: [usize; 0] = [];
    assert_eq!(
        island_labels::<3>(&[], 1.0, IslandMetric::Euclidean, &mut none),
        Ok(0)
    );
}

#[test]
fn island_count_equals_h0_bars_alive_past_the_radius() {
    // β₀(r) = #{H₀ bars with death > r}, the essential bar included. Checked at
    // every death value itself, where the closed ball d ≤ r and the strict
    // death > r must agree: a `<` in the island test fails exactly there.
    for seed in [4u64, 9, 16, 25, 36] {
        let mut rng = Rng::new(seed);
        let points: Vec<ManifoldPoint<3>> = clustered_cloud(24, 3, &mut rng);
        let diagram = persistent_homology(&points, PersistenceConfig::h0_only()).unwrap();
        let deaths: Vec<Option<f64>> = diagram
            .pairs
            .iter()
            .filter(|p| p.dimension == 0)
            .map(|p| p.death)
            .collect();
        assert_eq!(deaths.len(), points.len());

        let mut radii: Vec<f64> = deaths.iter().flatten().copied().collect();
        radii.extend(radii.clone().iter().map(|d| d * 1.0001 + 1e-9));
        radii.extend([0.0, 1e3]);
        for r in radii {
            let mut labels = vec![0; points.len()];
            let islands = island_labels(&points, r, IslandMetric::Euclidean, &mut labels).unwrap();
            let alive = deaths.iter().filter(|d| d.is_none_or(|d| d > r)).count();
            assert_eq!(islands, alive, "seed {seed}, r {r}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 8. Refusal
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn malformed_and_non_finite_input_is_refused() {
    let mut rng = Rng::new(83);
    let states: Vec<[f64; 2]> = (0..10).map(|_| rng.vector(1.0)).collect();
    let next: Vec<[f64; 2]> = (0..10).map(|_| rng.vector(1.0)).collect();
    let actions: Vec<[f64; 1]> = (0..10).map(|_| rng.vector(1.0)).collect();
    let fit = |s: &[[f64; 2]], y: &[[f64; 2]], a: &[[f64; 1]], ridge: f64, ceiling: Option<f64>| {
        CouplingOperator::<2, 1>::fit(s, y, a, ridge, ceiling).map(|_| ())
    };
    assert_eq!(fit(&states, &next, &actions, 1e-3, None), Ok(()));

    for poison in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let (mut s, mut y, mut a) = (states.clone(), next.clone(), actions.clone());
        s[3][1] = poison;
        assert_eq!(
            fit(&s, &next, &actions, 1e-3, None),
            Err(CouplingError::NonFinite)
        );
        y[7][0] = poison;
        assert_eq!(
            fit(&states, &y, &actions, 1e-3, None),
            Err(CouplingError::NonFinite)
        );
        a[0][0] = poison;
        assert_eq!(
            fit(&states, &next, &a, 1e-3, None),
            Err(CouplingError::NonFinite)
        );
        assert_eq!(
            fit(&states, &next, &actions, poison, None),
            Err(CouplingError::InvalidRidge)
        );
        assert_eq!(
            fit(&states, &next, &actions, 1e-3, Some(poison)),
            Err(CouplingError::InvalidCeiling)
        );
    }
    assert_eq!(
        fit(&states, &next, &actions, 0.0, None),
        Err(CouplingError::InvalidRidge)
    );
    assert_eq!(
        fit(&states, &next, &actions, -1.0, None),
        Err(CouplingError::InvalidRidge)
    );
    assert_eq!(
        fit(&states, &next, &actions, 1e-3, Some(-0.5)),
        Err(CouplingError::InvalidCeiling)
    );
    assert_eq!(
        fit(&states, &next[..9], &actions, 1e-3, None),
        Err(CouplingError::LengthMismatch)
    );
    assert_eq!(
        fit(&states, &next, &actions[..9], 1e-3, None),
        Err(CouplingError::LengthMismatch)
    );
    assert_eq!(
        fit(&states[..1], &next[..1], &actions[..1], 1e-3, None),
        Err(CouplingError::TooFewTransitions)
    );

    let op: CouplingOperator<2, 1> = random_operator(&mut rng, 0.5);
    let mut out = [[0.0; 2]; 3];
    assert_eq!(
        op.rollout(&[f64::NAN, 0.0], &[[0.0]; 3], &mut out),
        Err(CouplingError::NonFinite)
    );
    assert_eq!(
        op.rollout(&[0.0, 0.0], &[[0.0], [f64::NAN], [0.0]], &mut out),
        Err(CouplingError::NonFinite)
    );
    assert_eq!(
        op.rollout(&[0.0, 0.0], &[[0.0]; 2], &mut out),
        Err(CouplingError::LengthMismatch)
    );

    let mut projected = op;
    assert_eq!(
        projected.project_spectral(f64::NAN),
        Err(CouplingError::InvalidCeiling)
    );
    assert_eq!(
        projected.project_spectral(-1.0),
        Err(CouplingError::InvalidCeiling)
    );
    assert_eq!(
        projected, op,
        "a refused projection must not touch the operator"
    );

    let mut poisoned = op;
    poisoned.state[1][0] = f64::NAN;
    assert!(poisoned.spectral_norm().is_nan());
    assert!(
        !certificate(poisoned.spectral_norm(), 0.1, poisoned.state, [[0.0; 2]; 2]).contractive()
    );

    let mut s = [0.0; 4];
    assert_eq!(
        op.coupling_strength(0, &mut s),
        Err(CouplingError::IndivisibleBodies)
    );
    let wide = CouplingOperator::<3, 0>::autonomous([[0.0; 3]; 3], [0.0; 3]);
    assert_eq!(
        wide.coupling_strength(2, &mut s),
        Err(CouplingError::IndivisibleBodies)
    );
    assert_eq!(
        op.coupling_strength(1, &mut s[..3]),
        Err(CouplingError::LengthMismatch)
    );

    let cloud = [
        ManifoldPoint::new([0.0, 0.0]),
        ManifoldPoint::new([1.0, 0.0]),
    ];
    let mut labels = [0; 2];
    for r in [f64::NAN, f64::INFINITY, -1e-12] {
        assert_eq!(
            island_labels(&cloud, r, IslandMetric::Euclidean, &mut labels),
            Err(CouplingError::InvalidRadius)
        );
    }
    let bad = [
        ManifoldPoint::new([0.0, f64::NAN]),
        ManifoldPoint::new([1.0, 0.0]),
    ];
    assert_eq!(
        island_labels(&bad, 1.0, IslandMetric::Euclidean, &mut labels),
        Err(CouplingError::NonFinite)
    );
    assert_eq!(
        island_labels(&cloud, 1.0, IslandMetric::Euclidean, &mut labels[..1]),
        Err(CouplingError::LengthMismatch)
    );
}
