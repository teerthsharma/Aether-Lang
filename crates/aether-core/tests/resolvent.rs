//! Correctness contracts for `aether_core::resolvent`.
//!
//! Each Lean theorem the module mirrors is a numerical test here over seeded
//! inputs with a stated tolerance, named in the test's comment. The references
//! are written in this file and share no code with the module: the corners are
//! checked against the expressions the theorems state, the softmax corner also
//! against the crate's own `attention::sparse_attention`, and the path product
//! against an explicit triple loop in the order the source fixes.
//!
//! Where a check is bitwise it is because both sides do the same arithmetic in
//! the same order; `libm` is used in those references for that reason.

use aether_core::attention::sparse_attention;
use aether_core::resolvent::{
    blend, chain_label, gate, operator, path_product, readout, Complex, NonFinite, Switches,
};
use core::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════════════════
// Deterministic inputs and independent references
// ═══════════════════════════════════════════════════════════════════════════════

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
    /// Uniform on `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform on `[-1, 1)`.
    fn signed(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }
}

fn matrix(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..rows * cols).map(|_| rng.signed()).collect()
}

/// Raw gate magnitudes on `[0, 1)` and phases on `[-π, π)`.
fn heads(seq: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let u = (0..seq).map(|_| rng.unit()).collect();
    let th = (0..seq).map(|_| rng.signed() * PI).collect();
    (u, th)
}

fn op(
    q: &[f64],
    k: &[f64],
    u: &[f64],
    th: &[f64],
    seq: usize,
    d: usize,
    sw: Switches,
) -> Vec<Complex> {
    operator(q, k, u, th, seq, d, sw).expect("finite inputs are accepted")
}

fn sw(beta: f64, qk: f64, g: f64) -> Switches {
    Switches { beta, qk, g }
}

fn cmul(x: Complex, y: Complex) -> Complex {
    Complex {
        re: x.re * y.re - x.im * y.im,
        im: x.re * y.im + x.im * y.re,
    }
}

fn modulus(c: Complex) -> f64 {
    c.re.hypot(c.im)
}

/// `s_ij = <q_i, k_j> / sqrt(d)`, with the dot product summed in index order.
fn logits(q: &[f64], k: &[f64], seq: usize, d: usize) -> Vec<f64> {
    let scale = 1.0 / libm::sqrt(d as f64);
    let mut s = vec![0.0; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            let mut dot = 0.0;
            for t in 0..d {
                dot += q[i * d + t] * k[j * d + t];
            }
            s[i * seq + j] = dot * scale;
        }
    }
    s
}

/// `a_k = u_k e^{iθ_k}` at `g = 1`, built with `libm` directly.
fn gates(u: &[f64], th: &[f64]) -> Vec<Complex> {
    u.iter()
        .zip(th)
        .map(|(&m, &t)| Complex {
            re: m * libm::cos(t),
            im: m * libm::sin(t),
        })
        .collect()
}

/// `prod_{k=j+1}^{i} a_k`, each entry from scratch, `k` running from `i` down.
fn naive_path_product(a: &[Complex]) -> Vec<Complex> {
    let n = a.len();
    let mut out = vec![Complex { re: 0.0, im: 0.0 }; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut p = Complex { re: 1.0, im: 0.0 };
            for kk in (j + 1..=i).rev() {
                p = cmul(p, a[kk]);
            }
            out[i * n + j] = p;
        }
    }
    out
}

fn causal_mask(seq: usize) -> Vec<bool> {
    (0..seq * seq).map(|x| x % seq <= x / seq).collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. The three corners (three_corners_containment)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn the_softmax_corner_is_causal_softmax_and_matches_the_crate_reference() {
    // corner_softmax: Hop 1 g0 qk = softmaxAttn. The gates are present and the
    // `g = 0` switch must remove them. Tolerance 1e-14 on entries in [0, 1];
    // the module shifts by the row maximum and the reference does not.
    for (seq, d) in [(1usize, 4usize), (5, 3), (8, 8), (13, 7)] {
        let (q, k, v) = (matrix(seq, d, 42), matrix(seq, d, 43), matrix(seq, d, 44));
        let (u, th) = heads(seq, 7);
        let w = op(&q, &k, &u, &th, seq, d, Switches::SOFTMAX);
        let s = logits(&q, &k, seq, d);
        for i in 0..seq {
            let z: f64 = (0..=i).map(|j| s[i * seq + j].exp()).sum();
            for j in 0..seq {
                let want = if j <= i {
                    s[i * seq + j].exp() / z
                } else {
                    0.0
                };
                let got = w[i * seq + j];
                assert!(
                    (got.re - want).abs() < 1e-14,
                    "seq {seq} ({i},{j}): {} vs {want}",
                    got.re
                );
                assert_eq!(got.im, 0.0, "seq {seq} ({i},{j}): imaginary part at g = 0");
            }
        }
        let o = readout(&w, &v, seq, d);
        let reference = sparse_attention(&q, &k, &v, seq, d, &causal_mask(seq));
        for (x, (got, want)) in o.iter().zip(&reference).enumerate() {
            assert!(
                (got.re - want).abs() < 1e-14,
                "seq {seq} element {x}: {} vs {want}",
                got.re
            );
            assert_eq!(got.im, 0.0);
        }
    }
}

#[test]
fn the_unnormalized_kernel_corner_is_the_bare_exponential_bitwise() {
    // corner_linear: Hop 0 g0 qk = linearAttn qk = exp(qk i j), no normalizer.
    // `β = 0` must give Z^0 = 1 for every Z, so this is bitwise.
    let (seq, d) = (9usize, 5usize);
    let (q, k) = (matrix(seq, d, 1), matrix(seq, d, 2));
    let (u, th) = heads(seq, 3);
    let w = op(&q, &k, &u, &th, seq, d, Switches::UNNORMALIZED_KERNEL);
    let s = logits(&q, &k, seq, d);
    for i in 0..seq {
        for j in 0..seq {
            let want = if j <= i {
                libm::exp(s[i * seq + j])
            } else {
                0.0
            };
            assert_eq!(w[i * seq + j], Complex { re: want, im: 0.0 }, "({i},{j})");
        }
    }
}

#[test]
fn the_path_product_corner_is_the_gate_product_bitwise() {
    // corner_path_product, on the closed support of prefix_logit_mask_restated:
    // closed (m = 0) and band (m = 1) gates are included. The reference is an
    // explicit triple loop in the stated association order; q and k are live
    // inputs that `qk = 0` must ignore.
    let (seq, d) = (12usize, 3usize);
    let (q, k) = (matrix(seq, d, 5), matrix(seq, d, 6));
    let (mut u, th) = heads(seq, 7);
    u[4] = 0.0;
    u[8] = 1.0;
    let w = op(&q, &k, &u, &th, seq, d, Switches::PATH_PRODUCT);
    let reference = naive_path_product(&gates(&u, &th));
    for (x, (got, want)) in w.iter().zip(&reference).enumerate() {
        assert_eq!(got, want, "entry ({}, {})", x / seq, x % seq);
    }
    assert_eq!(path_product(&gates(&u, &th)), reference);
}

#[test]
fn the_three_corners_are_distinct_operators() {
    // corners_are_distinct: at g ≡ 0 and qk off, entry (1, 0) reads 1/2 at β = 1
    // and 1 at β = 0. Then on random inputs every pair of corners differs at
    // O(1), not at an ulp: a containment whose corners coincide is empty.
    let z = vec![0.0; 2];
    let one = op(&z, &z, &[1.0, 1.0], &[0.0, 0.0], 2, 1, sw(1.0, 0.0, 0.0));
    let zero = op(&z, &z, &[1.0, 1.0], &[0.0, 0.0], 2, 1, sw(0.0, 0.0, 0.0));
    assert!((one[2].re - 0.5).abs() < 1e-15, "β = 1 reads {}", one[2].re);
    assert_eq!(zero[2].re, 1.0);

    let (seq, d) = (8usize, 4usize);
    let (q, k) = (matrix(seq, d, 11), matrix(seq, d, 12));
    let (u, th) = heads(seq, 13);
    let c1 = op(&q, &k, &u, &th, seq, d, Switches::SOFTMAX);
    let c2 = op(&q, &k, &u, &th, seq, d, Switches::UNNORMALIZED_KERNEL);
    let c3 = op(&q, &k, &u, &th, seq, d, Switches::PATH_PRODUCT);
    for (name, a, b) in [("1-2", &c1, &c2), ("1-3", &c1, &c3), ("2-3", &c2, &c3)] {
        let gap = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                modulus(Complex {
                    re: x.re - y.re,
                    im: x.im - y.im,
                })
            })
            .fold(0.0, f64::max);
        assert!(gap > 0.5, "corners {name} differ by only {gap}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Row-stochasticity: β, not g, decides it
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn beta_alone_decides_row_stochasticity() {
    // softmax_row_sum_one and beta_one_row_is_one: at β = 1 the rows of |W| sum
    // to 1 with the gates on or off (tolerance 1e-14). gate_zero_beta_zero_row_not_one:
    // at g ≡ 0, β = 0, qk off, row i sums to exactly i + 1.
    let (seq, d) = (10usize, 4usize);
    let (q, k) = (matrix(seq, d, 21), matrix(seq, d, 22));
    let (u, th) = heads(seq, 23);
    let row = |w: &[Complex], i: usize, f: fn(Complex) -> f64| {
        (0..seq).map(|j| f(w[i * seq + j])).sum::<f64>()
    };

    let gated = op(&q, &k, &u, &th, seq, d, sw(1.0, 1.0, 1.0));
    let plain = op(&q, &k, &u, &th, seq, d, Switches::SOFTMAX);
    let bare = op(&q, &k, &u, &th, seq, d, sw(0.0, 0.0, 0.0));
    let unnormalized = op(&q, &k, &u, &th, seq, d, sw(0.0, 1.0, 1.0));
    let mut off_by = 0.0f64;
    for i in 0..seq {
        assert!(
            (row(&gated, i, modulus) - 1.0).abs() < 1e-14,
            "gated row {i}"
        );
        assert!(
            (row(&plain, i, |c| c.re) - 1.0).abs() < 1e-14,
            "plain row {i}"
        );
        assert_eq!(row(&bare, i, |c| c.re), (i + 1) as f64, "bare row {i}");
        off_by = off_by.max((row(&unnormalized, i, modulus) - 1.0).abs());
    }
    assert!(
        off_by > 0.1,
        "β = 0 rows should not be stochastic; worst gap {off_by}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. The path product on the closed support (prefix_logit_mask_restated)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn the_path_product_modulus_is_the_product_of_magnitudes() {
    // Clauses 1-3: |G_ij| = prod m_k (pathProd_abs), |G_ij| <= 1, and |G_ij| = 1
    // iff every magnitude on the path is 1. Magnitudes on [0, 0.9] except a band
    // of exact ones at 3..=6, every phase random. Tolerance 1e-14.
    let seq = 12usize;
    let mut rng = Rng::new(31);
    let mut m: Vec<f64> = (0..seq).map(|_| 0.9 * rng.unit()).collect();
    for x in &mut m[3..=6] {
        *x = 1.0;
    }
    let th: Vec<f64> = (0..seq).map(|_| rng.signed() * PI).collect();
    let g = path_product(&gates(&m, &th));
    for i in 0..seq {
        for j in 0..=i {
            let prod: f64 = m[j + 1..=i].iter().product();
            let abs = modulus(g[i * seq + j]);
            assert!(
                (abs - prod).abs() < 1e-14,
                "({i},{j}): |G| {abs} vs prod m {prod}"
            );
            assert!(abs <= 1.0 + 1e-15, "({i},{j}): |G| = {abs} exceeds 1");
            let on_band = m[j + 1..=i].iter().all(|&x| x == 1.0);
            assert_eq!(
                (abs - 1.0).abs() < 1e-14,
                on_band,
                "({i},{j}): band membership, |G| = {abs}"
            );
        }
    }
}

#[test]
fn a_closed_gate_annihilates_every_path_through_it_exactly() {
    // pathProd_eq_zero_iff (clause 4): G_ij = 0 exactly iff a zero magnitude is
    // on the path. That zero must survive every switch setting as an exact zero
    // of W, which no exp(C_i - C_j) form can produce
    // (no_prefix_scan_represents_a_zero_gate). And the gate at position 0 is
    // never on any path, so moving it moves nothing.
    let (seq, d) = (10usize, 3usize);
    let (q, k) = (matrix(seq, d, 41), matrix(seq, d, 42));
    let (mut u, th) = heads(seq, 43);
    for x in &mut u {
        *x = 0.1 + 0.9 * *x;
    }
    u[5] = 0.0;
    for s in [
        sw(0.0, 0.0, 1.0),
        sw(0.0, 1.0, 1.0),
        sw(0.5, 1.0, 1.0),
        sw(1.0, 1.0, 1.0),
    ] {
        let w = op(&q, &k, &u, &th, seq, d, s);
        for i in 0..seq {
            for j in 0..=i {
                let e = w[i * seq + j];
                if j < 5 && 5 <= i {
                    assert!(
                        e.re == 0.0 && e.im == 0.0,
                        "{s:?} ({i},{j}) = {e:?}, not exactly 0"
                    );
                } else {
                    assert!(
                        modulus(e) > 0.0,
                        "{s:?} ({i},{j}) annihilated with no closed gate"
                    );
                }
            }
        }
        let (mut u0, mut th0) = (u.clone(), th.clone());
        u0[0] = 0.0;
        th0[0] = 2.5;
        assert_eq!(
            op(&q, &k, &u0, &th0, seq, d, s),
            w,
            "{s:?}: the position-0 gate was read"
        );
    }
}

#[test]
fn the_bedm_values_are_ordinary_points_of_the_gate() {
    // bedM_gate_exact and negative_draw_is_on_the_band: -1, 0, +1 are
    // gateOf(1, π), gateOf(0, 0), gateOf(1, 0). The real parts are exact; sin(π)
    // leaves 1.2246e-16 in the imaginary part, bounded at 1.3e-16.
    let minus = gate(1.0, PI);
    assert_eq!(minus.re, -1.0);
    assert!(minus.im.abs() <= 1.3e-16, "im(gate(1, π)) = {}", minus.im);
    assert!((modulus(minus) - 1.0).abs() < 1e-16);
    assert_eq!(gate(0.0, 0.0), Complex { re: 0.0, im: 0.0 });
    assert_eq!(gate(0.0, PI).re, 0.0);
    assert_eq!(gate(1.0, 0.0), Complex { re: 1.0, im: 0.0 });
}

#[test]
fn a_constant_phase_is_the_rope_kernel_and_every_phase_schedule_separates() {
    // constant_phase_gate_is_rope: m ≡ 1, θ ≡ ω gives e^{iω(i-j)}, so G is
    // shift-invariant (Toeplitz). cumulative_phase_is_separable: for any θ,
    // G_ij = e^{i(Θ_i - Θ_j)} with Θ the prefix sum. Tolerance 1e-13 over up to
    // 15 accumulated multiplies.
    let seq = 16usize;
    let ones = vec![1.0; seq];
    let omega = 0.37;
    let g = path_product(&gates(&ones, &vec![omega; seq]));
    for i in 0..seq {
        for j in 0..=i {
            let angle = omega * (i - j) as f64;
            let e = g[i * seq + j];
            assert!(
                (e.re - angle.cos()).abs() < 1e-13 && (e.im - angle.sin()).abs() < 1e-13,
                "({i},{j})"
            );
            if i + 1 < seq {
                let shifted = g[(i + 1) * seq + j + 1];
                assert!(
                    modulus(Complex {
                        re: e.re - shifted.re,
                        im: e.im - shifted.im
                    }) < 1e-13,
                    "shift ({i},{j})"
                );
            }
        }
    }

    let (_, th) = heads(seq, 51);
    let prefix: Vec<f64> = th
        .iter()
        .scan(0.0, |acc, t| {
            *acc += t;
            Some(*acc)
        })
        .collect();
    let g = path_product(&gates(&ones, &th));
    for i in 0..seq {
        for j in 0..=i {
            let angle = prefix[i] - prefix[j];
            let e = g[i * seq + j];
            assert!(
                (e.re - angle.cos()).abs() < 1e-13 && (e.im - angle.sin()).abs() < 1e-13,
                "({i},{j})"
            );
        }
    }
}

#[test]
fn scalar_gates_commute_within_a_window() {
    // PhaseH.scalar_gate_commutes: permuting the gates inside a window leaves the
    // product unchanged. Exact in C, 1e-15 in float64, where the fixed
    // association order is the only thing that moves. Windows that do not touch
    // the permuted positions are bitwise unchanged.
    let seq = 12usize;
    let (u, th) = heads(seq, 61);
    let a = gates(&u, &th);
    let mut permuted = a.clone();
    permuted[3..=7].reverse();
    let (g, gp) = (path_product(&a), path_product(&permuted));
    for i in 0..seq {
        for j in 0..=i {
            let (x, y) = (g[i * seq + j], gp[i * seq + j]);
            if j < 3 && i >= 7 {
                assert!(
                    modulus(Complex {
                        re: x.re - y.re,
                        im: x.im - y.im
                    }) < 1e-15,
                    "({i},{j})"
                );
            } else if i < 3 || j >= 7 {
                assert_eq!(x, y, "({i},{j}) sits outside the permuted positions");
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. The path-product corner is the resolvent of the gate shift
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn the_path_product_corner_reads_out_the_chain_and_inverts_the_gate_shift() {
    // chain_eq_sum / chain_path_product: at β = 0, qk off, the readout on values
    // b (b_0 = 0) is the chain y_i = a_i y_{i-1} + b_i. Checked on BED-M's own
    // support {-1, 0, +1} and on random complex gates, tolerance 1e-12 (the
    // source measures under 1e-13 against a 1e-6 bar). Then G (I - A) = I with
    // A_{i,i-1} = a_i (occupancy_is_exact_inverse), tolerance 1e-14. A planted
    // negative: β = 1 at the same setting must break the label at O(1).
    let seq = 24usize;
    let mut rng = Rng::new(71);
    let bedm: Vec<(f64, f64)> = (0..seq)
        .map(|_| match rng.next_u64() % 3 {
            0 => (1.0, PI),
            1 => (0.0, 0.0),
            _ => (1.0, 0.0),
        })
        .collect();
    let (ur, thr) = heads(seq, 72);
    let draws = [
        (
            bedm.iter().map(|p| p.0).collect::<Vec<_>>(),
            bedm.iter().map(|p| p.1).collect::<Vec<_>>(),
        ),
        (ur, thr),
    ];
    let mut b = matrix(seq, 1, 73);
    b[0] = 0.0;
    let z = vec![0.0; seq];
    for (u, th) in &draws {
        let a: Vec<Complex> = u.iter().zip(th).map(|(&m, &t)| gate(m, t)).collect();
        let y = chain_label(&a, &b);
        let honest = readout(
            &op(&z, &z, u, th, seq, 1, Switches::PATH_PRODUCT),
            &b,
            seq,
            1,
        );
        let wrong = readout(&op(&z, &z, u, th, seq, 1, sw(1.0, 0.0, 1.0)), &b, seq, 1);
        let gap = |o: &[Complex]| {
            o.iter()
                .zip(&y)
                .map(|(p, t)| {
                    modulus(Complex {
                        re: p.re - t.re,
                        im: p.im - t.im,
                    })
                })
                .fold(0.0, f64::max)
        };
        assert!(
            gap(&honest) < 1e-12,
            "path-product readout misses the chain by {}",
            gap(&honest)
        );
        assert!(
            gap(&wrong) > 0.1,
            "β = 1 should break the chain; gap {}",
            gap(&wrong)
        );

        let g = path_product(&a);
        for i in 0..seq {
            for j in 0..seq {
                let mut e = g[i * seq + j];
                if j + 1 < seq {
                    let t = cmul(g[i * seq + j + 1], a[j + 1]);
                    e = Complex {
                        re: e.re - t.re,
                        im: e.im - t.im,
                    };
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    modulus(Complex {
                        re: e.re - want,
                        im: e.im
                    }) < 1e-14,
                    "G(I - A) at ({i},{j}) = {e:?}"
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. Causality and the switches
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn perturbing_the_future_never_moves_the_past() {
    // Row i reads q_i, k_{<=i} and the gates in (j, i] only. Perturb every input
    // after position t and require rows <= t of W and the readout to be bitwise
    // unchanged, at every corner and at an interior setting. W above the
    // diagonal must be exactly 0. The perturbation must move some later row.
    let (seq, d, t) = (10usize, 4usize, 4usize);
    let (q, k, v) = (matrix(seq, d, 81), matrix(seq, d, 82), matrix(seq, d, 83));
    let (u, th) = heads(seq, 84);
    let (mut q2, mut k2, mut v2, mut u2, mut th2) =
        (q.clone(), k.clone(), v.clone(), u.clone(), th.clone());
    for p in t + 1..seq {
        for x in 0..d {
            q2[p * d + x] += 0.75;
            k2[p * d + x] -= 0.5;
            v2[p * d + x] *= -3.0;
        }
        u2[p] = 1.0 - u2[p];
        th2[p] += 1.0;
    }
    for s in [
        Switches::SOFTMAX,
        Switches::UNNORMALIZED_KERNEL,
        Switches::PATH_PRODUCT,
        sw(0.5, 0.7, 0.6),
    ] {
        let (w, w2) = (
            op(&q, &k, &u, &th, seq, d, s),
            op(&q2, &k2, &u2, &th2, seq, d, s),
        );
        let (o, o2) = (readout(&w, &v, seq, d), readout(&w2, &v2, seq, d));
        assert_eq!(
            w[..(t + 1) * seq],
            w2[..(t + 1) * seq],
            "{s:?}: a past row of W moved"
        );
        assert_eq!(
            o[..(t + 1) * d],
            o2[..(t + 1) * d],
            "{s:?}: a past readout moved"
        );
        assert_ne!(
            o[(t + 1) * d..],
            o2[(t + 1) * d..],
            "{s:?}: the perturbation moved nothing"
        );
        for i in 0..seq {
            for j in i + 1..seq {
                assert_eq!(
                    w2[i * seq + j],
                    Complex { re: 0.0, im: 0.0 },
                    "{s:?}: ({i},{j}) above the diagonal"
                );
            }
        }
    }
}

#[test]
fn an_off_switch_deletes_its_term_exactly() {
    // g = 0 is the gate-free hop exactly (m = 1, θ = 0), whatever u and θ are;
    // qk = 0 makes W independent of q and k exactly. Both switches must also be
    // live when on. blend keeps m in the closed [0, 1] for every g, because the
    // cap is applied after the blend.
    let (seq, d) = (8usize, 4usize);
    let (q, k) = (matrix(seq, d, 91), matrix(seq, d, 92));
    let (q2, k2) = (matrix(seq, d, 93), matrix(seq, d, 94));
    let (u, th) = heads(seq, 95);
    let (ones, zeros) = (vec![1.0; seq], vec![0.0; seq]);

    let off = op(&q, &k, &u, &th, seq, d, sw(0.7, 1.0, 0.0));
    assert_eq!(
        off,
        op(&q, &k, &ones, &zeros, seq, d, sw(0.7, 1.0, 0.0)),
        "g = 0 read the gates"
    );
    let on = op(&q, &k, &u, &th, seq, d, sw(0.7, 1.0, 1.0));
    assert!(
        off.iter().zip(&on).any(|(a, b)| (a.re - b.re).abs() > 0.1),
        "g = 1 changed nothing"
    );

    let blind = op(&q, &k, &u, &th, seq, d, sw(0.3, 0.0, 1.0));
    assert_eq!(
        blind,
        op(&q2, &k2, &u, &th, seq, d, sw(0.3, 0.0, 1.0)),
        "qk = 0 read q or k"
    );
    let seeing = op(&q, &k, &u, &th, seq, d, sw(0.3, 1.0, 1.0));
    assert!(blind.iter().zip(&seeing).any(|(a, b)| modulus(Complex {
        re: a.re - b.re,
        im: a.im - b.im
    }) > 0.1));

    let mut rng = Rng::new(96);
    for _ in 0..4096 {
        let raw = rng.signed() * 3.0;
        for g in [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 25.0] {
            let (m, _) = blend(raw, 0.3, g);
            assert!((0.0..=1.0).contains(&m), "blend({raw}, g = {g}) = {m}");
        }
        assert_eq!(blend(raw, 0.3, 1.0), (raw.clamp(0.0, 1.0), 0.3));
        assert_eq!(blend(raw, 0.3, 0.0), (1.0, 0.0));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. Large logits, refusals, degenerate inputs
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn logits_of_ten_thousand_stay_finite_at_beta_one() {
    // Logits reach past ±1e4. The unshifted reference overflows to inf/inf (the
    // planted negative); the module's log-domain evaluation must stay finite,
    // keep |W| rows stochastic to 1e-9 with gates off and on, and agree with the
    // max-shifted sparse_attention to 1e-9. At β = 0, logits of -1e4 underflow
    // to finite zeros, never NaN.
    let (seq, d) = (8usize, 4usize);
    let (q, k, v) = (
        matrix(seq, d, 101),
        matrix(seq, d, 102),
        matrix(seq, d, 103),
    );
    let (q, k): (Vec<f64>, Vec<f64>) = (
        q.iter().map(|x| x * 300.0).collect(),
        k.iter().map(|x| x * 300.0).collect(),
    );
    let (u, th) = heads(seq, 104);
    let s = logits(&q, &k, seq, d);
    assert!(
        s.iter().any(|x| x.abs() > 1e4),
        "the draw never reached |logit| > 1e4"
    );
    let naive_overflows = (0..seq).any(|i| {
        let z: f64 = (0..=i).map(|j| s[i * seq + j].exp()).sum();
        (0..=i).any(|j| !(s[i * seq + j].exp() / z).is_finite())
    });
    assert!(
        naive_overflows,
        "the unshifted reference should overflow on this draw"
    );

    for g in [0.0, 1.0] {
        let w = op(&q, &k, &u, &th, seq, d, sw(1.0, 1.0, g));
        assert!(
            w.iter().all(|e| e.re.is_finite() && e.im.is_finite()),
            "g = {g}: non-finite entry"
        );
        for i in 0..seq {
            let sum: f64 = (0..seq).map(|j| modulus(w[i * seq + j])).sum();
            assert!((sum - 1.0).abs() < 1e-9, "g = {g}: row {i} sums to {sum}");
        }
        let o = readout(&w, &v, seq, d);
        assert!(o.iter().all(|e| e.re.is_finite() && e.im.is_finite()));
        if g == 0.0 {
            let reference = sparse_attention(&q, &k, &v, seq, d, &causal_mask(seq));
            for (got, want) in o.iter().zip(&reference) {
                assert!((got.re - want).abs() < 1e-9, "{} vs {want}", got.re);
            }
        }
    }

    let neg: Vec<f64> = k.iter().map(|x| -x.abs()).collect();
    let pos: Vec<f64> = q.iter().map(|x| x.abs()).collect();
    let w = op(&pos, &neg, &u, &th, seq, d, Switches::UNNORMALIZED_KERNEL);
    assert!(w
        .iter()
        .all(|e| e.re.is_finite() && e.re >= 0.0 && e.im == 0.0));
}

#[test]
fn an_overflowing_logit_never_becomes_nan() {
    // At β = 0 a logit of 2e4 has no finite value: live entries read +inf, which
    // is the value. What must not happen is NaN: a dead entry (closed gate on the
    // path) stays exactly 0 and a zero imaginary lane stays exactly 0 rather than
    // becoming 0 · inf, the source's gate-kill mask.
    let (seq, d) = (8usize, 2usize);
    let q = vec![100.0; seq * d];
    let k = vec![100.0; seq * d];
    let mut u = vec![0.8; seq];
    u[2] = 0.0;
    u[5] = 0.0;
    let th = vec![0.0; seq];
    let w = op(&q, &k, &u, &th, seq, d, sw(0.0, 1.0, 1.0));
    assert!(
        w.iter().all(|e| !e.re.is_nan() && !e.im.is_nan()),
        "NaN in the operator"
    );
    assert!(w.iter().all(|e| e.im == 0.0), "a zero imaginary lane moved");
    assert!(
        w.iter().any(|e| e.re == f64::INFINITY),
        "the draw never overflowed"
    );
    for i in 0..seq {
        for j in 0..=i {
            if (j < 2 && 2 <= i) || (j < 5 && 5 <= i) {
                assert_eq!(
                    w[i * seq + j].re,
                    0.0,
                    "dead ({i},{j}) = {:?}",
                    w[i * seq + j]
                );
            }
        }
    }
}

#[test]
fn non_finite_input_is_refused() {
    // A NaN or infinite query, key, gate, phase or switch is refused, and so is
    // a finite input whose logit or blended phase overflows. With qk = 0 the
    // overflowing dot product is not a logit and the call is accepted.
    let (seq, d) = (4usize, 2usize);
    let (q, k) = (matrix(seq, d, 111), matrix(seq, d, 112));
    let (u, th) = heads(seq, 113);
    let bad = |x: &[f64], i: usize, val: f64| {
        let mut y = x.to_vec();
        y[i] = val;
        y
    };
    let base = Switches::SOFTMAX;
    assert_eq!(
        operator(&bad(&q, 3, f64::NAN), &k, &u, &th, seq, d, base),
        Err(NonFinite)
    );
    assert_eq!(
        operator(&q, &bad(&k, 0, f64::INFINITY), &u, &th, seq, d, base),
        Err(NonFinite)
    );
    assert_eq!(
        operator(&q, &k, &bad(&u, 1, f64::NAN), &th, seq, d, base),
        Err(NonFinite)
    );
    assert_eq!(
        operator(&q, &k, &u, &bad(&th, 2, f64::NEG_INFINITY), seq, d, base),
        Err(NonFinite)
    );
    for s in [
        sw(f64::NAN, 1.0, 0.0),
        sw(1.0, f64::INFINITY, 0.0),
        sw(1.0, 1.0, f64::NAN),
    ] {
        assert_eq!(
            operator(&q, &k, &u, &th, seq, d, s),
            Err(NonFinite),
            "{s:?}"
        );
    }
    let huge = vec![1e200; seq * d];
    assert_eq!(
        operator(&huge, &huge, &u, &th, seq, d, base),
        Err(NonFinite),
        "overflowing logit"
    );
    assert!(operator(&huge, &huge, &u, &th, seq, d, Switches::PATH_PRODUCT).is_ok());
    let wide = vec![1e300; seq];
    assert_eq!(
        operator(&q, &k, &u, &wide, seq, d, sw(1.0, 1.0, 1e10)),
        Err(NonFinite),
        "overflowing phase"
    );
}

#[test]
fn an_empty_sequence_is_empty_and_no_row_is_ever_all_masked() {
    // seq = 0 is an empty operator, readout, chain and path product. Every row
    // keeps its diagonal (G_ii = 1, the empty product), so closing every gate
    // leaves the identity at β = 1 and at the path-product corner: no row is
    // ever empty, and Z_i > 0 for every input.
    assert_eq!(
        operator(&[], &[], &[], &[], 0, 3, Switches::SOFTMAX),
        Ok(vec![])
    );
    assert!(readout(&[], &[], 0, 2).is_empty());
    assert!(chain_label(&[], &[]).is_empty());
    assert!(path_product(&[]).is_empty());

    let (seq, d) = (6usize, 3usize);
    let (q, k) = (matrix(seq, d, 121), matrix(seq, d, 122));
    let (closed, th) = (vec![0.0; seq], heads(seq, 123).1);
    for s in [sw(1.0, 1.0, 1.0), Switches::PATH_PRODUCT] {
        let w = op(&q, &k, &closed, &th, seq, d, s);
        for i in 0..seq {
            for j in 0..seq {
                let want = if i == j { 1.0 } else { 0.0 };
                assert_eq!(
                    w[i * seq + j],
                    Complex { re: want, im: 0.0 },
                    "{s:?} ({i},{j})"
                );
            }
        }
    }
    let single = op(&q[..d], &k[..d], &[0.3], &[1.0], 1, d, Switches::SOFTMAX);
    assert_eq!(single, vec![Complex { re: 1.0, im: 0.0 }]);
}
