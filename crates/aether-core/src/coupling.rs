//! Topological coupling operators: an action-conditioned affine map on world
//! states, its Banach fixed point, its rollout error bound, and the island
//! partition that says which bodies are connected at a stated radius.
//!
//! Ported from `sigmoid` (github.com/teerthsharma/sigmoid): the operator from
//! `sigmoid/operator.py`, block coupling strength from `sigmoid/nbody.py`, and
//! island partitions from `sigmoid/mujoco/island.py`. The persistence encoder
//! that produces the state in that engine is not ported; any fixed-length state
//! vector in ℝ^D, such as [`SystemState::vector`], is a valid input.
//!
//! # The operator
//!
//! A state `z ∈ ℝ^D` and an action `a ∈ ℝ^K` are lifted by the bilinear map
//!
//! ```text
//! φ(z, a) = [ z ; a ⊗ z ; a ; 1 ] ∈ ℝ^L,     L = D + K·D + K + 1,
//! (a ⊗ z)_{k·D + j} = a_k · z_j
//! ```
//!
//! and one step of the operator `W ∈ ℝ^{D×L}` is
//!
//! ```text
//! z_{t+1} = W φ(z_t, a_t) = T₀ z_t + Σ_k a_{t,k} T_k z_t + B a_t + c.
//! ```
//!
//! The map is affine in `z` for fixed `a`, affine in `a` for fixed `z`, and not
//! affine in `(z, a)` jointly: the bilinear blocks `T_k` are the only source of
//! that failure, and `T_k = 0, B = 0` reduces the operator to the autonomous
//! dynamics `z ↦ T₀ z + c` for every action.
//!
//! `W` is fitted by ridge regression in closed form,
//!
//! ```text
//! W = argmin_W Σ_t ‖ W φ_t − z_{t+1} ‖² + λ ‖W‖²_F  =  Yᵀ X (XᵀX + λI)⁻¹,
//! ```
//!
//! solved here by a Cholesky factorisation of `XᵀX + λI`, which is symmetric
//! positive definite for every `λ > 0`.
//!
//! # Fixed points and stability
//!
//! Let `ρ = σ_max(T₀)`. The autonomous map `F(z) = T₀ z + c` (the `a = 0` map)
//! has the fixed point `z* = (I − T₀)⁻¹ c` whenever `I − T₀` is invertible. If
//! `ρ < 1` then `F` is a contraction in the Euclidean norm,
//!
//! ```text
//! ‖F(z) − F(w)‖ ≤ ρ ‖z − w‖,     ‖Fⁿ(z₀) − z*‖ ≤ ρⁿ ‖z₀ − z*‖,
//! ```
//!
//! so by the Banach fixed-point theorem `z*` is unique and every rollout
//! converges to it geometrically. If each one-step residual has RMS norm at most
//! `ε`, the accumulated error after `n` steps satisfies
//!
//! ```text
//! E(n) ≤ ε Σ_{i<n} ρ^i = ε (1 − ρⁿ) / (1 − ρ)   (ρ < 1),
//! E(n) ≤ ε n ρ^{n−1}                           (ρ ≥ 1).
//! ```
//!
//! An optional spectral ceiling `ρ_max` rescales every singular value of `T₀`
//! by `ρ_max / ρ`, which is exactly `T₀ ← (ρ_max / ρ) T₀`.
//!
//! The directional estimate propagates the residual second moment `Σ` instead
//! of the scalar `ε`:
//!
//! ```text
//! C_n = T₀ C_{n−1} T₀ᵀ + Σ,   C₀ = 0,     estimate(n) = √( tr C_n / D ).
//! ```
//!
//! # Islands
//!
//! For points `x_i` and an absolute radius `r ≥ 0`, points `i` and `j` are in
//! the same island when they are joined by a chain of pairs at distance `≤ r`.
//! The island count equals `β₀` of the Vietoris–Rips complex at `r`, which is
//! the number of `H₀` bars with death `> r` (the essential bar included): the
//! single-linkage merge heights are the `H₀` death times. The multi-body
//! coupling diagnostic is the matrix of block Frobenius norms
//! `S_ij = ‖T₀[body i, body j]‖_F`.
//!
//! # Proved versus measured
//!
//! Proved: the contraction inequality and the Banach fixed point when `ρ < 1`,
//! for the autonomous map only (with actions the per-step state map is
//! `T₀ + Σ_k a_k T_k`, whose norm `ρ` does not bound); the scalar error bound,
//! under the hypothesis that every one-step residual has RMS norm at most `ε`;
//! the spectral ceiling producing `σ_max = ρ_max`; the identity between island
//! count and `β₀`.
//!
//! Not proved: the fitted `ε` is an in-sample root mean square, not a supremum,
//! so a certificate built from a fit inherits the hypothesis rather than
//! discharging it. The directional estimate is an expectation under zero-mean,
//! step-uncorrelated residuals and linear dynamics; it is not a bound, and the
//! source measured it under-bounding when residuals are autocorrelated, which is
//! why the fit reports the lag-1 residual autocorrelation beside it. The source
//! reports the scalar bound as vacuous on transformer activations and the
//! certificate as not established as useful there. Convergence of the Banach
//! iteration is measured on every call rather than asserted.
//!
//! The ceiling is never applied by default. A chaotic system has `ρ > 1`, and
//! clipping it buys a certificate by misreporting the dynamics; the source
//! measured a loss of one-step accuracy from doing so and made it opt-in.
//!
//! # Refusals
//!
//! Fitting refuses a ridge that is not finite and positive, a ceiling that is
//! negative or not finite, fewer than two transitions, mismatched lengths, any
//! non-finite datum, and a normal matrix whose Cholesky pivot is not positive in
//! floating point. Rollouts refuse a non-finite start or action. Island
//! partitions refuse a negative or non-finite radius and non-finite points.
//! [`CouplingOperator::fixed_point`] returns `None` when `I − T₀` is exactly
//! singular. A non-finite operator reports `ρ = NaN`, which no certificate
//! treats as contractive or safe.

#![warn(missing_docs)]

#[cfg(feature = "alloc")]
use alloc::vec;

use libm::{acos, fabs, hypot, pow, sqrt};

use crate::manifold::ManifoldPoint;
use crate::state::SystemState;

/// Iteration cap for the measured Banach iteration, as in the source.
const BANACH_ITERATIONS: usize = 512;
/// Step-size tolerance at which the Banach iteration is declared converged.
const BANACH_TOLERANCE: f64 = 1e-12;
/// Distance to the solved fixed point within which convergence is accepted.
const BANACH_AGREEMENT: f64 = 1e-6;
/// Longest horizon [`RolloutCertificate::safe_horizon`] will report.
const SAFE_HORIZON_CAP: usize = 4096;
/// Sweep cap for the Jacobi eigenvalue iteration behind the spectral norm.
const JACOBI_SWEEPS: usize = 100;

/// Why an operation on a coupling operator was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CouplingError {
    /// A state, action, point or target contained a NaN or an infinity.
    NonFinite,
    /// The ridge `λ` was not finite and strictly positive. `λ > 0` is the only
    /// guarantee that `XᵀX + λI` is invertible.
    InvalidRidge,
    /// The spectral ceiling `ρ_max` was negative or not finite.
    InvalidCeiling,
    /// The island radius was negative or not finite.
    InvalidRadius,
    /// Two slices that must agree in length did not.
    LengthMismatch,
    /// Fewer than two transitions were supplied.
    TooFewTransitions,
    /// A Cholesky pivot of `XᵀX + λI` was not positive in floating point, which
    /// happens when `λ` is negligible against a rank-deficient design.
    NotPositiveDefinite,
    /// The body dimension was zero or did not divide the state dimension.
    IndivisibleBodies,
}

/// An action-conditioned affine operator on `ℝ^D` with `K` action channels.
///
/// One step is `T₀ z + Σ_k a_k T_k z + B a + c`. `K = 0` is the autonomous
/// operator `z ↦ T₀ z + c`.
///
/// Ported from `CouplingOperator` in sigmoid/sigmoid/operator.py, where the four
/// blocks are the column blocks of `W` acting on `[z ; a ⊗ z ; a ; 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CouplingOperator<const D: usize, const K: usize> {
    /// `T₀`, the state-to-state block. The only block that feeds back, and so
    /// the only one whose spectral norm governs stability.
    pub state: [[f64; D]; D],
    /// `T_k`, one `D × D` block per action channel, multiplying `a_k z`.
    pub bilinear: [[[f64; D]; D]; K],
    /// `B`, the direct action input, row `i` holding the weights of `a` on `z_i`.
    pub input: [[f64; K]; D],
    /// `c`, the constant offset.
    pub bias: [f64; D],
}

/// The fixed point of the autonomous map and whether iteration reached it.
///
/// Ported from `CouplingOperator._power_iterate` in sigmoid/sigmoid/operator.py.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedPoint<const D: usize> {
    /// `z* = (I − T₀)⁻¹ c`, solved directly.
    pub point: [f64; D],
    /// Whether the Banach iteration `z ← T₀ z + c` from the origin reached
    /// `z*`. An expansive operator can have an algebraic fixed point that no
    /// iteration reaches; this is `false` then.
    pub converged: bool,
}

/// A ridge-fitted operator with the residual statistics measured on its data.
///
/// Ported from `CouplingOperator.fit` in sigmoid/sigmoid/operator.py.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fit<const D: usize, const K: usize> {
    /// The fitted operator, after the spectral ceiling if one was requested.
    pub operator: CouplingOperator<D, K>,
    /// `σ_max(T₀)`, or exactly `ρ_max` when the ceiling was applied.
    pub rho: f64,
    /// In-sample one-step residual RMS over every coordinate of every
    /// transition. `step_rmse² = tr(residual_cov) / D`.
    pub step_rmse: f64,
    /// `Σ = RᵀR / N`, the second moment of the residual about zero, so that a
    /// persistent bias is charged rather than centred away.
    pub residual_cov: [[f64; D]; D],
    /// Lag-1 autocorrelation of the residual sequence,
    /// `Σ_t r_t·r_{t−1} / Σ_t ‖r_t‖²`, assuming rows are in temporal order. Near
    /// zero supports the directional estimate's step-independence assumption;
    /// large values invalidate it.
    pub residual_autocorr: f64,
}

/// A-priori error estimates for an `n`-step rollout.
///
/// Ported from `RolloutCertificate` in sigmoid/sigmoid/operator.py.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RolloutCertificate<const D: usize> {
    /// Spectral norm of the state block. Contractive iff `< 1`.
    pub rho: f64,
    /// One-step residual RMS `ε`.
    pub step_rmse: f64,
    /// `T₀`, needed to propagate the residual second moment.
    pub state_block: [[f64; D]; D],
    /// `Σ`, the one-step residual second moment.
    pub residual_cov: [[f64; D]; D],
}

/// The distance under which island membership is decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IslandMetric {
    /// `‖x − y‖₂`, computed by subtracting before squaring.
    Euclidean,
    /// Great-circle distance `acos(clamp(x·y, −1, 1))`, the metric of the
    /// `mujoco#3396` corpus. Meaningful for unit vectors only; points are not
    /// normalised.
    Geodesic,
}

impl<const D: usize, const K: usize> CouplingOperator<D, K> {
    /// The operator `z ↦ T₀ z + c` with every action block zero.
    ///
    /// Ported from sigmoid/sigmoid/operator.py, where an operator fitted without
    /// actions has no `a ⊗ z` or `a` columns.
    pub fn autonomous(state: [[f64; D]; D], bias: [f64; D]) -> Self {
        Self {
            state,
            bilinear: [[[0.0; D]; D]; K],
            input: [[0.0; K]; D],
            bias,
        }
    }

    /// One step: `T₀ z + Σ_k a_k T_k z + B a + c`.
    ///
    /// Accumulates in the column order of `φ(z, a) = [z ; a ⊗ z ; a ; 1]`, so the
    /// result is bit-identical to the explicit product `W φ(z, a)`. Performs no
    /// validation; [`rollout`](Self::rollout) is the checked entry point.
    ///
    /// Ported from `CouplingOperator.step` in sigmoid/sigmoid/operator.py.
    pub fn step(&self, z: &[f64; D], action: &[f64; K]) -> [f64; D] {
        let mut out = [0.0; D];
        for (i, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (w, x) in self.state[i].iter().zip(z) {
                acc += w * x;
            }
            for (t, ak) in self.bilinear.iter().zip(action) {
                for (w, x) in t[i].iter().zip(z) {
                    acc += w * (ak * x);
                }
            }
            for (w, ak) in self.input[i].iter().zip(action) {
                acc += w * ak;
            }
            *o = acc + self.bias[i];
        }
        out
    }

    /// Roll `out.len()` steps forward from `z0`, writing step `i + 1` to `out[i]`.
    ///
    /// `actions[i]` drives step `i + 1` and must have the same length as `out`
    /// when `K > 0`; it is ignored when `K = 0`. Refuses a non-finite start or
    /// action. An expansive operator may overflow to infinity; that is reported,
    /// not refused.
    ///
    /// Ported from `CouplingOperator.rollout` in sigmoid/sigmoid/operator.py.
    pub fn rollout(
        &self,
        z0: &[f64; D],
        actions: &[[f64; K]],
        out: &mut [[f64; D]],
    ) -> Result<(), CouplingError> {
        if K > 0 && actions.len() != out.len() {
            return Err(CouplingError::LengthMismatch);
        }
        if !finite(z0) || !actions.iter().all(finite) {
            return Err(CouplingError::NonFinite);
        }
        let mut z = *z0;
        for (i, slot) in out.iter_mut().enumerate() {
            z = self.step(&z, &action_at(actions, i));
            *slot = z;
        }
        Ok(())
    }

    /// `ρ = σ_max(T₀)`, the largest singular value of the state block.
    ///
    /// Computed as the square root of the largest eigenvalue of `T₀ᵀT₀` by cyclic
    /// Jacobi rotation, which does not under-estimate the way a truncated power
    /// iteration does. Returns NaN for a non-finite state block.
    ///
    /// Ported from `CouplingOperator._project_spectral` in
    /// sigmoid/sigmoid/operator.py, which reads `ρ` off the SVD.
    pub fn spectral_norm(&self) -> f64 {
        spectral_norm(&self.state)
    }

    /// Enforce `σ_max(T₀) ≤ ρ_max` and return the resulting `ρ`.
    ///
    /// When `ρ > ρ_max`, every singular value is scaled by `ρ_max / ρ`, which is
    /// exactly `T₀ ← (ρ_max / ρ) T₀`; singular vectors, and the action and bias
    /// blocks, are untouched. Otherwise nothing changes. This overwrites the
    /// measured dynamics and is never applied implicitly.
    ///
    /// Ported from `CouplingOperator._project_spectral` in
    /// sigmoid/sigmoid/operator.py.
    pub fn project_spectral(&mut self, rho_max: f64) -> Result<f64, CouplingError> {
        if !(rho_max >= 0.0 && rho_max.is_finite()) {
            return Err(CouplingError::InvalidCeiling);
        }
        let rho = self.spectral_norm();
        if rho.is_nan() || rho <= rho_max {
            return Ok(rho);
        }
        let scale = rho_max / rho;
        for row in self.state.iter_mut() {
            for w in row.iter_mut() {
                *w *= scale;
            }
        }
        Ok(rho_max)
    }

    /// The fixed point of the autonomous map `z ↦ T₀ z + c`, with convergence of
    /// the Banach iteration measured rather than assumed.
    ///
    /// Solves `(I − T₀) z* = c` by Gaussian elimination with partial pivoting,
    /// returning `None` when a pivot is exactly zero or the solution is not
    /// finite. Then iterates `z ← T₀ z + c` from the origin for at most 512
    /// steps, stopping when a step moves less than `1e-12`; `converged` is set
    /// only if that happens and the iterate lies within `1e-6` of `z*`.
    ///
    /// Ported from `CouplingOperator._power_iterate` in
    /// sigmoid/sigmoid/operator.py.
    pub fn fixed_point(&self) -> Option<FixedPoint<D>> {
        let mut system = [[0.0; D]; D];
        for (i, row) in system.iter_mut().enumerate() {
            for (j, m) in row.iter_mut().enumerate() {
                *m = if i == j { 1.0 } else { 0.0 } - self.state[i][j];
            }
        }
        let point = solve_dense(system, self.bias)?;

        let zero = [0.0; K];
        let mut z = [0.0; D];
        let mut converged = false;
        for _ in 0..BANACH_ITERATIONS {
            let next = self.step(&z, &zero);
            if !finite(&next) {
                break;
            }
            if distance(&next, &z) < BANACH_TOLERANCE {
                z = next;
                converged = true;
                break;
            }
            z = next;
        }
        Some(FixedPoint {
            point,
            converged: converged && distance(&z, &point) < BANACH_AGREEMENT,
        })
    }

    /// Block Frobenius norms of `T₀` over bodies of `body_dim` coordinates each.
    ///
    /// Writes `S_ij = ‖T₀[body i, body j]‖_F` to `out[i · n + j]`, where
    /// `n = D / body_dim`, and returns `n`. An off-diagonal entry is the
    /// coupling of body `i`'s next state to body `j`'s current state; a zero
    /// entry means none.
    ///
    /// Ported from `MultiBodyCoupling.coupling_strength` in
    /// sigmoid/sigmoid/nbody.py.
    pub fn coupling_strength(
        &self,
        body_dim: usize,
        out: &mut [f64],
    ) -> Result<usize, CouplingError> {
        if body_dim == 0 || !D.is_multiple_of(body_dim) {
            return Err(CouplingError::IndivisibleBodies);
        }
        let n = D / body_dim;
        if out.len() != n * n {
            return Err(CouplingError::LengthMismatch);
        }
        for (cell, s) in out.iter_mut().enumerate() {
            let (bi, bj) = (cell / n, cell % n);
            let mut sum = 0.0;
            for row in &self.state[bi * body_dim..(bi + 1) * body_dim] {
                for w in &row[bj * body_dim..(bj + 1) * body_dim] {
                    sum += w * w;
                }
            }
            *s = sqrt(sum);
        }
        Ok(n)
    }

    /// Ridge-fit an operator on paired transitions `states[t] → next_states[t]`
    /// under `actions[t]`, optionally enforcing a spectral ceiling.
    ///
    /// Solves `(XᵀX + λI) Wᵀ = XᵀY` by Cholesky, where row `t` of `X` is
    /// `φ(states[t], actions[t])`. `actions` must match `states` in length when
    /// `K > 0` and is ignored when `K = 0`. When `rho_max` is given the ceiling is
    /// applied before the residuals are measured, so the reported statistics
    /// describe the operator returned.
    ///
    /// Ported from `CouplingOperator.fit` in sigmoid/sigmoid/operator.py.
    #[cfg(feature = "alloc")]
    pub fn fit(
        states: &[[f64; D]],
        next_states: &[[f64; D]],
        actions: &[[f64; K]],
        ridge: f64,
        rho_max: Option<f64>,
    ) -> Result<Fit<D, K>, CouplingError> {
        if !(ridge > 0.0 && ridge.is_finite()) {
            return Err(CouplingError::InvalidRidge);
        }
        if matches!(rho_max, Some(r) if !(r >= 0.0 && r.is_finite())) {
            return Err(CouplingError::InvalidCeiling);
        }
        let n = states.len();
        if next_states.len() != n || (K > 0 && actions.len() != n) {
            return Err(CouplingError::LengthMismatch);
        }
        if n < 2 {
            return Err(CouplingError::TooFewTransitions);
        }
        if !states.iter().all(finite)
            || !next_states.iter().all(finite)
            || !actions.iter().all(finite)
        {
            return Err(CouplingError::NonFinite);
        }

        let lift = D + K * D + K + 1;
        let mut gram = vec![0.0; lift * lift];
        let mut rhs = vec![0.0; lift * D];
        let mut phi = vec![0.0; lift];
        for (t, (z, y)) in states.iter().zip(next_states).enumerate() {
            let a = action_at(actions, t);
            phi[..D].copy_from_slice(z);
            for (k, ak) in a.iter().enumerate() {
                for (j, zj) in z.iter().enumerate() {
                    phi[D + k * D + j] = ak * zj;
                }
            }
            phi[D + K * D..lift - 1].copy_from_slice(&a);
            phi[lift - 1] = 1.0;
            for (p, fp) in phi.iter().enumerate() {
                for (g, fq) in gram[p * lift..(p + 1) * lift].iter_mut().zip(&phi) {
                    *g += fp * fq;
                }
                for (r, yi) in rhs[p * D..(p + 1) * D].iter_mut().zip(y) {
                    *r += fp * yi;
                }
            }
        }
        for p in 0..lift {
            gram[p * lift + p] += ridge;
        }
        cholesky_solve(&mut gram, lift, &mut rhs, D)?;

        // rhs is now Wᵀ: rhs[p·D + i] = W[i][p].
        let w = |i: usize, p: usize| rhs[p * D + i];
        let mut operator = Self::autonomous([[0.0; D]; D], [0.0; D]);
        for i in 0..D {
            for j in 0..D {
                operator.state[i][j] = w(i, j);
            }
            for k in 0..K {
                for j in 0..D {
                    operator.bilinear[k][i][j] = w(i, D + k * D + j);
                }
                operator.input[i][k] = w(i, D + K * D + k);
            }
            operator.bias[i] = w(i, lift - 1);
        }
        let rho = match rho_max {
            Some(ceiling) => operator.project_spectral(ceiling)?,
            None => operator.spectral_norm(),
        };

        let mut residual_cov = [[0.0; D]; D];
        let (mut energy, mut lagged) = (0.0, 0.0);
        let mut previous: Option<[f64; D]> = None;
        for (t, (z, y)) in states.iter().zip(next_states).enumerate() {
            let predicted = operator.step(z, &action_at(actions, t));
            let mut r = [0.0; D];
            for i in 0..D {
                r[i] = y[i] - predicted[i];
            }
            for i in 0..D {
                energy += r[i] * r[i];
                for j in 0..D {
                    residual_cov[i][j] += r[i] * r[j];
                }
            }
            if let Some(prev) = previous {
                lagged += r.iter().zip(&prev).map(|(x, y)| x * y).sum::<f64>();
            }
            previous = Some(r);
        }
        for row in residual_cov.iter_mut() {
            for c in row.iter_mut() {
                *c /= n as f64;
            }
        }

        Ok(Fit {
            operator,
            rho,
            step_rmse: sqrt(energy / (n * D) as f64),
            residual_cov,
            residual_autocorr: if energy > 0.0 { lagged / energy } else { 0.0 },
        })
    }
}

impl<const D: usize, const K: usize> Fit<D, K> {
    /// The rollout certificate of the fitted operator.
    ///
    /// Ported from `CouplingOperator.certificate` in sigmoid/sigmoid/operator.py.
    pub fn certificate(&self) -> RolloutCertificate<D> {
        RolloutCertificate {
            rho: self.rho,
            step_rmse: self.step_rmse,
            state_block: self.operator.state,
            residual_cov: self.residual_cov,
        }
    }
}

impl<const D: usize> RolloutCertificate<D> {
    /// Whether `ρ < 1`. False for a NaN `ρ`.
    pub fn contractive(&self) -> bool {
        self.rho < 1.0
    }

    /// The worst-case accumulated error after `steps` applications.
    ///
    /// `ε (1 − ρⁿ)/(1 − ρ)` when `ρ < 1 − 10⁻¹²`, otherwise `ε n ρ^{n−1}` with
    /// `ρ` floored at 1, and infinity when `ρ^{n−1}` overflows. Zero at zero
    /// steps. A guarantee only under the hypothesis that every one-step residual
    /// has RMS norm at most `ε`.
    ///
    /// Ported from `RolloutCertificate.error_bound` in
    /// sigmoid/sigmoid/operator.py.
    pub fn error_bound(&self, steps: usize) -> f64 {
        if steps == 0 {
            return 0.0;
        }
        let n = steps as f64;
        if self.rho >= 1.0 - 1e-12 {
            let growth = pow(self.rho.max(1.0), n - 1.0);
            if !growth.is_finite() {
                return f64::INFINITY;
            }
            return self.step_rmse * n * growth;
        }
        self.step_rmse * (1.0 - pow(self.rho, n)) / (1.0 - self.rho)
    }

    /// The covariance-propagated RMS error after `steps` steps. An estimate, not
    /// a bound.
    ///
    /// `√(tr C_n / D)` with `C_n = T₀ C_{n−1} T₀ᵀ + Σ`, `C₀ = 0`: the expected
    /// squared error for zero-mean, step-uncorrelated residuals under linear
    /// dynamics, so roughly half of individual rollouts exceed it. Returns
    /// infinity once the trace stops being finite.
    ///
    /// Ported from `RolloutCertificate.directional_error` in
    /// sigmoid/sigmoid/operator.py.
    pub fn directional_error(&self, steps: usize) -> f64 {
        let mut second_moment = [[0.0; D]; D];
        let mut rms = 0.0;
        for _ in 0..steps {
            rms = self.propagate(&mut second_moment);
            if rms == f64::INFINITY {
                break;
            }
        }
        rms
    }

    /// The largest `k ≤ 4096` whose error stays within `tolerance`.
    ///
    /// Uses [`error_bound`](Self::error_bound), or with `directional` the
    /// [`directional_error`](Self::directional_error) estimate, which is longer
    /// and inherits that estimate's assumptions. A NaN error counts as unsafe.
    ///
    /// Ported from `CouplingOperator.safe_horizon` in
    /// sigmoid/sigmoid/operator.py.
    pub fn safe_horizon(&self, tolerance: f64, directional: bool) -> usize {
        let mut second_moment = [[0.0; D]; D];
        for k in 1..=SAFE_HORIZON_CAP {
            let err = if directional {
                self.propagate(&mut second_moment)
            } else {
                self.error_bound(k)
            };
            // NaN on either side is unsafe, exactly as `!(err <= tolerance)`.
            if err.is_nan() || tolerance.is_nan() || err > tolerance {
                return k - 1;
            }
        }
        SAFE_HORIZON_CAP
    }

    /// Advance `C ← T₀ C T₀ᵀ + Σ` once and return `√(tr C / D)`.
    fn propagate(&self, c: &mut [[f64; D]; D]) -> f64 {
        let a = &self.state_block;
        let mut ac = [[0.0; D]; D];
        for (i, row) in ac.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = (0..D).map(|k| a[i][k] * c[k][j]).sum();
            }
        }
        let mut trace = 0.0;
        for i in 0..D {
            for j in 0..D {
                c[i][j] = (0..D).map(|k| ac[i][k] * a[j][k]).sum::<f64>() + self.residual_cov[i][j];
            }
            trace += c[i][i];
        }
        if !trace.is_finite() {
            return f64::INFINITY;
        }
        sqrt(trace.max(0.0) / D as f64)
    }
}

/// Label each point with its island at the absolute `radius`, returning the
/// island count.
///
/// Points `i` and `j` share an island when a chain of pairs at distance
/// `≤ radius` joins them. Components are merged by a disjoint-set union and the
/// labels canonicalised as in `mj_island`: each island is represented by its
/// minimum index, and ids ascend with that representative, so `labels[0] = 0`
/// and the first point of each new island receives the next id. The count is
/// `β₀` of the Vietoris–Rips complex at `radius`. `labels` must be as long as
/// `points`; an empty cloud has zero islands.
///
/// Ported from `island_labels` and `island_count` in
/// sigmoid/sigmoid/mujoco/island.py.
pub fn island_labels<const N: usize>(
    points: &[ManifoldPoint<N>],
    radius: f64,
    metric: IslandMetric,
    labels: &mut [usize],
) -> Result<usize, CouplingError> {
    if !(radius >= 0.0 && radius.is_finite()) {
        return Err(CouplingError::InvalidRadius);
    }
    if labels.len() != points.len() {
        return Err(CouplingError::LengthMismatch);
    }
    if !points.iter().all(|p| finite(&p.coords)) {
        return Err(CouplingError::NonFinite);
    }

    // Union-find in place. Every parent index is at most its child's, so each
    // root is the minimum index of its island.
    for (i, l) in labels.iter_mut().enumerate() {
        *l = i;
    }
    for i in 0..points.len() {
        for j in i + 1..points.len() {
            let d = match metric {
                IslandMetric::Euclidean => points[i].distance(&points[j]),
                IslandMetric::Geodesic => {
                    let dot: f64 = points[i]
                        .coords
                        .iter()
                        .zip(&points[j].coords)
                        .map(|(x, y)| x * y)
                        .sum();
                    acos(dot.clamp(-1.0, 1.0))
                }
            };
            if d <= radius {
                let (a, b) = (find_root(labels, i), find_root(labels, j));
                labels[a.max(b)] = a.min(b);
            }
        }
    }
    for i in 0..labels.len() {
        labels[i] = find_root(labels, i);
    }
    // Every root precedes its members, so it is relabelled first.
    let mut count = 0;
    for i in 0..labels.len() {
        labels[i] = if labels[i] == i {
            count += 1;
            count - 1
        } else {
            labels[labels[i]]
        };
    }
    Ok(count)
}

/// Root of `x` with path halving. Parents never exceed their children.
fn find_root(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn finite<const M: usize>(v: &[f64; M]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn action_at<const K: usize>(actions: &[[f64; K]], t: usize) -> [f64; K] {
    actions.get(t).copied().unwrap_or([0.0; K])
}

/// Euclidean distance between two states, via the kernel's state vector.
fn distance<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    SystemState::new(*a, 0).deviation(&SystemState::new(*b, 0))
}

/// `σ_max(a)` via cyclic Jacobi on the symmetric `aᵀa`.
fn spectral_norm<const D: usize>(a: &[[f64; D]; D]) -> f64 {
    if !a.iter().all(finite) {
        return f64::NAN;
    }
    let mut s = [[0.0; D]; D];
    for i in 0..D {
        for j in 0..D {
            s[i][j] = (0..D).map(|k| a[k][i] * a[k][j]).sum();
        }
    }
    for _ in 0..JACOBI_SWEEPS {
        let mut off = 0.0;
        let mut total = 0.0;
        for (i, row) in s.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                total += v * v;
                if i != j {
                    off += v * v;
                }
            }
        }
        if off <= f64::EPSILON * f64::EPSILON * total {
            break;
        }
        for p in 0..D {
            for q in p + 1..D {
                if s[p][q] == 0.0 {
                    continue;
                }
                let theta = (s[q][q] - s[p][p]) / (2.0 * s[p][q]);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (fabs(theta) + hypot(theta, 1.0));
                let c = 1.0 / hypot(t, 1.0);
                let sn = t * c;
                for row in s.iter_mut() {
                    let (kp, kq) = (row[p], row[q]);
                    row[p] = c * kp - sn * kq;
                    row[q] = sn * kp + c * kq;
                }
                let (rp, rq) = (s[p], s[q]);
                for k in 0..D {
                    s[p][k] = c * rp[k] - sn * rq[k];
                    s[q][k] = sn * rp[k] + c * rq[k];
                }
            }
        }
    }
    let largest = (0..D).map(|i| s[i][i]).fold(0.0, f64::max);
    sqrt(largest)
}

/// Solve `m x = rhs` by Gaussian elimination with partial pivoting. `None` on an
/// exactly zero pivot or a non-finite solution, matching LAPACK `gesv`.
fn solve_dense<const D: usize>(mut m: [[f64; D]; D], mut rhs: [f64; D]) -> Option<[f64; D]> {
    for col in 0..D {
        let pivot = (col..D).max_by(|&a, &b| fabs(m[a][col]).total_cmp(&fabs(m[b][col])))?;
        if m[pivot][col] == 0.0 {
            return None;
        }
        m.swap(col, pivot);
        rhs.swap(col, pivot);
        let lead = m[col];
        for r in col + 1..D {
            let f = m[r][col] / lead[col];
            for c in col..D {
                m[r][c] -= f * lead[c];
            }
            rhs[r] -= f * rhs[col];
        }
    }
    let mut x = [0.0; D];
    for i in (0..D).rev() {
        let tail: f64 = (i + 1..D).map(|c| m[i][c] * x[c]).sum();
        x[i] = (rhs[i] - tail) / m[i][i];
    }
    finite(&x).then_some(x)
}

/// Solve `g X = rhs` for symmetric positive definite `g` (`l × l`, row major)
/// and `m` right-hand sides (`rhs` is `l × m`, row major, overwritten).
#[cfg(feature = "alloc")]
fn cholesky_solve(g: &mut [f64], l: usize, rhs: &mut [f64], m: usize) -> Result<(), CouplingError> {
    for j in 0..l {
        let d = g[j * l + j] - (0..j).map(|k| g[j * l + k] * g[j * l + k]).sum::<f64>();
        if d.is_nan() || d <= 0.0 {
            return Err(CouplingError::NotPositiveDefinite);
        }
        let d = sqrt(d);
        g[j * l + j] = d;
        for i in j + 1..l {
            let s = g[i * l + j] - (0..j).map(|k| g[i * l + k] * g[j * l + k]).sum::<f64>();
            g[i * l + j] = s / d;
        }
    }
    for c in 0..m {
        for i in 0..l {
            let s = rhs[i * m + c] - (0..i).map(|k| g[i * l + k] * rhs[k * m + c]).sum::<f64>();
            rhs[i * m + c] = s / g[i * l + i];
        }
        for i in (0..l).rev() {
            let s = rhs[i * m + c]
                - (i + 1..l)
                    .map(|k| g[k * l + i] * rhs[k * m + c])
                    .sum::<f64>();
            rhs[i * m + c] = s / g[i * l + i];
        }
    }
    Ok(())
}
