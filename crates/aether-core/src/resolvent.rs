//! One causal attention head with three switches: softmax attention,
//! unnormalized-kernel attention and the exact path product of a gated chain
//! as settings of a single operator.
//!
//! Ported from `resolvent/ceq/arm_smprime.py` (github.com/teerthsharma/resolvent,
//! commit `8c19735`), whose statements are proved in `resolvent/lean/CEQ/`.
//!
//! # The operator
//!
//! For a causal sequence of length `n`, queries and keys `q_i, k_j` of width
//! `d`, raw gate magnitudes `u_k` and phases `θ_k`, and switches `(β, qk, g)`:
//!
//! ```text
//! s_ij = <q_i, k_j> / sqrt(d)
//! m_k  = clamp(lerp(1, u_k, g), 0, 1)             θ'_k = g θ_k
//! a_k  = m_k e^{i θ'_k}                            the gate
//! G_ij = prod_{k=j+1}^{i} a_k                      the path product, G_ii = 1
//! R_ij = prod_{k=j+1}^{i} m_k = |G_ij|             its modulus
//! Z_i  = sum_{j<=i} R_ij e^{qk s_ij}               the row normalizer
//! W_ij = G_ij e^{qk s_ij} / Z_i^β   (j <= i),      W_ij = 0   (j > i)
//! O_i  = sum_{j<=i} W_ij V_j                       the readout
//! ```
//!
//! # The three corners
//!
//! ```text
//! softmax attention             β = 1, qk = 1, g = 0:   W_ij = e^{s_ij} / sum_{j'<=i} e^{s_ij'}
//! unnormalized-kernel attention β = 0, qk = 1, g = 0:   W_ij = e^{s_ij}
//! path product                  β = 0, qk = 0, g = 1:   W_ij = G_ij = prod_{k=j+1}^{i} a_k
//! ```
//!
//! At the path-product corner with values `b` (`b_0 = 0`) the readout is the
//! chain `y_0 = 0, y_i = a_i y_{i-1} + b_i`. That recurrence is forward
//! substitution of `(I - A) y = b` with `A_{i,i-1} = a_i` strictly lower
//! triangular, so `G = (I - A)^{-1} = sum_{k<n} A^k`: the corner is the
//! resolvent of the gate shift, and the Neumann series terminates because `A`
//! is nilpotent.
//!
//! The `β = 0` kernel corner is not linear attention in the O(n) sense. It is
//! the unnormalized exponential kernel, with no finite feature map, and costs
//! what the softmax corner costs.
//!
//! # What is machine-proved and what is only tested
//!
//! Proved in the source's Lean 4 (`lean/CEQ/V16Domain.lean` unless noted) and
//! mirrored numerically in `tests/resolvent.rs`:
//!
//! - `three_corners_containment` (`corner_softmax`, `corner_linear`,
//!   `corner_path_product`): the family contains the three named operators.
//! - `corners_are_distinct`: at `g ≡ 0, qk = 0`, entry `(1, 0)` reads `1/2`
//!   at `β = 1` and `1` at `β = 0`.
//! - `softmax_row_sum_one`, `beta_one_row_is_one`: at `β = 1` rows sum to 1.
//! - `gate_zero_beta_zero_is_linear_attention`,
//!   `gate_zero_beta_zero_row_not_one`: at `g ≡ 0, β = 0` the head is the
//!   bare kernel, and at `qk = 0` row `i` sums to `i + 1`, so `β`, not `g`,
//!   decides softmax-class membership.
//! - `pathProd_abs`, `pathProd_eq_zero_iff`, `prefix_logit_mask_restated`:
//!   `|G_ij| = prod m_k`, `|G_ij| <= 1`, `|G_ij| = 1` iff every magnitude on
//!   the path is 1, and `G_ij = 0` exactly iff some `m_k = 0` on the path.
//! - `no_prefix_scan_represents_a_zero_gate`: `exp(C_i - C_j)` is never zero,
//!   so the path product is evaluated as a product and never through a log.
//! - `bedM_gate_exact`, `negative_draw_is_on_the_band`: `-1, 0, +1` are the
//!   gates `(m, θ) = (1, π), (0, 0), (1, 0)`.
//! - `constant_phase_gate_is_rope`, `cumulative_phase_is_separable`: a unit
//!   gate of constant phase `ω` is `e^{iω(i-j)}`, and any phase schedule gives
//!   `e^{i(Θ_i - Θ_j)}` with `Θ` the prefix sum.
//! - `PhaseH.scalar_gate_commutes`: a scalar path product is invariant under
//!   permuting the gates inside its window.
//! - `CEQ.V15.chain_path_product`, `CEQ.V15Fork.chain_eq_sum`: the chain equals
//!   `sum_s (prod_{k=s+1}^{i} a_k) b_s`, over the reals, with no hypothesis on
//!   `a`.
//! - `CEQ.Nilpotent.occupancy_is_exact_inverse`: for strictly lower `A`,
//!   `sum_{k<n} A^k` is the exact two-sided inverse of `I - A`.
//!
//! Tested only, not proved:
//!
//! - Lean's `Hop` carries the gate as `exp(C_i - C_j)` over the reals. Corners
//!   1 and 2 (`g = 0`, where both forms are 1 on the causal triangle) are
//!   exactly its statements. Corner 3 with a zero magnitude or a nonzero phase
//!   rests on `pathProd`'s theorems; its identity with this operator is a
//!   bitwise test, as it is in the source.
//! - The complex-gate chain identity (the Lean `chain` is real-valued), and
//!   `G (I - A) = I` as a single matrix statement, which is a corollary of
//!   `chain_path_product` rather than a named theorem.
//! - Every property of the float64 evaluation: exact corners, causality, the
//!   large-logit behaviour below. No statement is made for `β` strictly between
//!   0 and 1, which Lean's `Hop` does not cover either.
//!
//! # Evaluation, refusals and degenerate inputs
//!
//! `Z_i^β` is evaluated in the log domain, `W_ij = G_ij exp(qk s_ij - β log Z_i)`,
//! with `log Z_i` taken by a max-shifted sum over the live entries (`R_ij > 0`).
//! This is the closed form `e^{(1-β)M} o / l^β` of the source's
//! `block_summary`/`read_summary`, with one block and the shift `M` taken over
//! the live logits. It keeps the `β = 1` corner finite (`|W_ij| <= 1`) at
//! logits of `±1e4`, where the dense form overflows.
//!
//! - **All-masked rows cannot occur.** `G_ii = R_ii = 1` (the empty product), so
//!   every row has a live diagonal and `Z_i > 0` for every input. Closing every
//!   gate leaves the identity, not an empty row.
//! - **Non-finite input is refused.** A NaN or infinite query, key, gate
//!   magnitude, phase or switch, a phase `g θ_k` that overflows, or a logit
//!   `qk s_ij` that overflows returns [`NonFinite`]. The source's
//!   `ceqjepa/operator.py::build_operator` refuses non-finite logits the same
//!   way, instead of spreading NaN.
//! - **The empty sequence** is an empty operator, readout and chain.
//! - **Overflow away from `β = 1`.** A live entry whose true value exceeds the
//!   f64 range reads `±inf`; that is the value, not a failure. It never reads
//!   NaN: a dead entry (`R_ij = 0`) is exactly 0, and a lane of `G_ij` that is
//!   exactly 0 stays exactly 0 rather than becoming `0 · inf` (the source's
//!   gate-kill mask, applied here to both lanes). [`readout`] of such an
//!   operator is not defined and is not guarded.
//! - Shape mismatches and `head_dim = 0` with a nonempty sequence panic, as in
//!   [`crate::attention`].

#![warn(missing_docs)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use libm::{cos, exp, fabs, log, sin, sqrt};

/// A complex number, `re + i im`. The path product and the operator are
/// complex because the gate carries its sign and phase in `e^{iθ}`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
const ONE: Complex = Complex { re: 1.0, im: 0.0 };

fn mul(x: Complex, y: Complex) -> Complex {
    Complex {
        re: x.re * y.re - x.im * y.im,
        im: x.re * y.im + x.im * y.re,
    }
}

/// The three switches `(β, qk, g)` of `resolvent/ceq/arm_smprime.py`.
///
/// `g` here is the source's blend scalar (`g = 0` is the gate-free corner).
/// Lean's `Hop` instead takes a log-gate sequence, where `g ≡ 0` is gate-free;
/// the two agree at that corner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Switches {
    /// Normalizer exponent: `W = G e^{qk s} / Z^β`. `β = 1` is intensive (rows
    /// of `|W|` sum to 1), `β = 0` is extensive (no normalizer).
    pub beta: f64,
    /// Content scale on the logit. `qk = 0` deletes the content term exactly.
    pub qk: f64,
    /// Gate blend. `g = 0` sets every gate to `m = 1, θ = 0`; `g = 1` passes
    /// the raw gate through the closed cap unchanged.
    pub g: f64,
}

impl Switches {
    /// Causal softmax attention: `β = 1, qk = 1, g = 0`. Lean `corner_softmax`.
    pub const SOFTMAX: Switches = Switches {
        beta: 1.0,
        qk: 1.0,
        g: 0.0,
    };
    /// Unnormalized-kernel attention: `β = 0, qk = 1, g = 0`. Lean
    /// `corner_linear`, whose operator is named `linearAttn`.
    pub const UNNORMALIZED_KERNEL: Switches = Switches {
        beta: 0.0,
        qk: 1.0,
        g: 0.0,
    };
    /// The exact path product: `β = 0, qk = 0, g = 1`. Lean
    /// `corner_path_product`, re-stated on the closed support by
    /// `prefix_logit_mask_restated`.
    pub const PATH_PRODUCT: Switches = Switches {
        beta: 0.0,
        qk: 0.0,
        g: 1.0,
    };
}

/// Refusal: an input, a blended phase or a logit was NaN or infinite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonFinite;

/// The `g` switch applied to one raw gate: `(m, θ') = (clamp(lerp(1, u, g), 0,
/// 1), g θ)`.
///
/// Ported from `resolvent/ceq/arm_smprime.py::blend` and `::magnitude`. The cap
/// is applied after the blend, so `m ∈ [0, 1]` for every `g`; `lerp` is
/// `torch.lerp`'s two-branch form, which makes `g = 0` return `m = 1` and
/// `g = 1` return `clamp(u, 0, 1)` exactly.
pub fn blend(u: f64, theta: f64, g: f64) -> (f64, f64) {
    let lerp = if fabs(g) < 0.5 {
        1.0 + g * (u - 1.0)
    } else {
        u - (u - 1.0) * (1.0 - g)
    };
    (lerp.clamp(0.0, 1.0), theta * g)
}

/// One gate in polar form, `m e^{iθ}`.
///
/// Ported from `resolvent/ceq/arm_smprime.py::gate`; Lean `gateOf`. No
/// logarithm is taken, so `m = 0` and `m = 1` are ordinary points
/// (`bedM_gate_exact`).
pub fn gate(m: f64, theta: f64) -> Complex {
    Complex {
        re: m * cos(theta),
        im: m * sin(theta),
    }
}

/// The path product `G_ij = prod_{k=j+1}^{i} a_k` for `j <= i`, 0 above the
/// diagonal, as a row-major `[n, n]` matrix.
///
/// Ported from `resolvent/ceq/arm_smprime.py::path_product`; Lean `pathProd`.
/// Accumulated from `k = i` down to `k = j + 1`, the order the source fixes,
/// because a complex product is not associative in floating point. `a_0` is
/// never read. A single `a_k = 0` sends every entry whose window contains `k`
/// to exactly 0 (`pathProd_eq_zero_iff`).
pub fn path_product(a: &[Complex]) -> Vec<Complex> {
    let n = a.len();
    let mut out = vec![ZERO; n * n];
    for i in 0..n {
        let mut p = ONE;
        out[i * n + i] = p;
        for j in (0..i).rev() {
            p = mul(p, a[j + 1]);
            out[i * n + j] = p;
        }
    }
    out
}

/// The operator `W_ij = G_ij e^{qk s_ij} / Z_i^β`, row-major `[seq, seq]`,
/// exactly 0 above the diagonal.
///
/// Ported from `resolvent/ceq/arm_smprime.py::operator` (with `hop` and
/// `numerator`), evaluated in the log domain of `block_summary`/`read_summary`;
/// Lean `Hop`. `q` and `k` are row-major `[seq, head_dim]`; `u` and `theta` are
/// the raw per-position gate magnitudes and phases, `[seq]` each.
///
/// Refuses non-finite inputs, phases and logits with [`NonFinite`]; see the
/// module documentation for every refusal and degenerate case.
pub fn operator(
    q: &[f64],
    k: &[f64],
    u: &[f64],
    theta: &[f64],
    seq: usize,
    head_dim: usize,
    switches: Switches,
) -> Result<Vec<Complex>, NonFinite> {
    assert_eq!(q.len(), seq * head_dim, "q must be [seq, head_dim]");
    assert_eq!(k.len(), seq * head_dim, "k must be [seq, head_dim]");
    assert_eq!(u.len(), seq, "u must be [seq]");
    assert_eq!(theta.len(), seq, "theta must be [seq]");
    if seq == 0 {
        return Ok(Vec::new());
    }
    assert!(head_dim > 0, "head_dim must be positive");

    let Switches { beta, qk, g } = switches;
    let finite = |x: &f64| x.is_finite();
    if ![beta, qk, g].iter().all(finite) || !q.iter().chain(k).chain(u).chain(theta).all(finite) {
        return Err(NonFinite);
    }

    let mut gates = Vec::with_capacity(seq);
    let mut magnitudes = Vec::with_capacity(seq);
    for t in 0..seq {
        let (m, phase) = blend(u[t], theta[t], g);
        if !phase.is_finite() {
            return Err(NonFinite);
        }
        gates.push(gate(m, phase));
        magnitudes.push(Complex { re: m, im: 0.0 });
    }
    // The modulus row is the path product of the real magnitudes, computed
    // directly (clause 1's right-hand side), as the source's `hop` does.
    let hop = path_product(&gates);
    let modulus = path_product(&magnitudes);

    let scale = 1.0 / sqrt(head_dim as f64);
    let mut w = vec![ZERO; seq * seq];
    let mut logits = vec![0.0f64; seq];
    for i in 0..seq {
        let row = i * seq;
        let mut max = f64::NEG_INFINITY;
        for j in 0..=i {
            // `qk = 0` deletes the content term exactly, including when the dot
            // product itself overflows.
            let s = if qk == 0.0 {
                0.0
            } else {
                qk * (dot(q, k, i, j, head_dim) * scale)
            };
            if !s.is_finite() {
                return Err(NonFinite);
            }
            logits[j] = s;
            if modulus[row + j].re > 0.0 && s > max {
                max = s;
            }
        }
        // The diagonal is live (R_ii = 1), so `max` is finite and `sum >= R_ij*`
        // for the maximizing live j: `log Z_i` is always finite.
        let mut sum = 0.0;
        for j in 0..=i {
            let r = modulus[row + j].re;
            if r > 0.0 {
                sum += r * exp(logits[j] - max);
            }
        }
        let log_z = max + log(sum);
        for j in 0..=i {
            if modulus[row + j].re > 0.0 {
                let f = exp(logits[j] - beta * log_z);
                let h = hop[row + j];
                w[row + j] = Complex {
                    re: lane(h.re, f),
                    im: lane(h.im, f),
                };
            }
        }
    }
    Ok(w)
}

/// A lane of `G_ij` that is exactly 0 stays exactly 0 against an overflowed
/// factor, instead of becoming `0 · inf = NaN`.
fn lane(x: f64, f: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x * f
    }
}

fn dot(q: &[f64], k: &[f64], i: usize, j: usize, head_dim: usize) -> f64 {
    let mut sum = 0.0;
    for d in 0..head_dim {
        sum += q[i * head_dim + d] * k[j * head_dim + d];
    }
    sum
}

/// The readout `O_i = sum_{j<=i} W_ij V_j`, row-major `[seq, value_dim]`.
///
/// Ported from `resolvent/ceq/arm_smprime.py::readout`, with the operator
/// passed in rather than rebuilt. `w` is `[seq, seq]` from [`operator`] and `v`
/// is real, `[seq, value_dim]`. No validation beyond shapes.
pub fn readout(w: &[Complex], v: &[f64], seq: usize, value_dim: usize) -> Vec<Complex> {
    assert_eq!(w.len(), seq * seq, "w must be [seq, seq]");
    assert_eq!(v.len(), seq * value_dim, "v must be [seq, value_dim]");
    let mut out = vec![ZERO; seq * value_dim];
    for i in 0..seq {
        for j in 0..=i {
            let wij = w[i * seq + j];
            for d in 0..value_dim {
                let x = v[j * value_dim + d];
                out[i * value_dim + d].re += wij.re * x;
                out[i * value_dim + d].im += wij.im * x;
            }
        }
    }
    out
}

/// The gated chain `y_0 = 0, y_i = a_i y_{i-1} + b_i`, by its own recurrence.
///
/// Ported from `resolvent/ceq/arm_phase.py::chain_label`; Lean `CEQ.V15.chain`
/// with `y₀ = 0`. This is forward substitution of `(I - A) y = b` with
/// `A_{i,i-1} = a_i`, the resolvent that the path-product corner's readout
/// equals (`CEQ.V15Fork.chain_eq_sum`). `b_0` is not read.
pub fn chain_label(a: &[Complex], b: &[f64]) -> Vec<Complex> {
    assert_eq!(a.len(), b.len(), "a and b must have the same length");
    let mut y = vec![ZERO; a.len()];
    for i in 1..a.len() {
        let p = mul(a[i], y[i - 1]);
        y[i] = Complex {
            re: p.re + b[i],
            im: p.im,
        };
    }
    y
}
