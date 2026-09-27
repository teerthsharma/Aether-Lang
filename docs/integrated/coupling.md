# Coupling Operator

Module: `aether_core::coupling`, in `crates/aether-core/src/coupling.rs`. It provides four things:

- an action-conditioned affine map on world states;
- the Banach fixed point of that map;
- an error bound for rollouts;
- the island partition that says which bodies are connected at a stated radius.

Evidence: 18 tests in `tests/coupling.rs`.

## The operator

A state \(z \in \mathbb{R}^D\) and an action \(a \in \mathbb{R}^K\) are lifted by the bilinear map

\[
\varphi(z, a) = \begin{bmatrix} z \\ a \otimes z \\ a \\ 1 \end{bmatrix} \in \mathbb{R}^{L},
\qquad L = D + KD + K + 1,
\qquad (a \otimes z)_{kD + j} = a_k z_j .
\]

One step of the operator \(W \in \mathbb{R}^{D \times L}\) is

\[
z_{t+1} = W\,\varphi(z_t, a_t) = T_0 z_t + \sum_{k=1}^{K} a_{t,k}\, T_k z_t + B a_t + c,
\]

where \(T_0 \in \mathbb{R}^{D\times D}\) is the state block, \(T_k \in \mathbb{R}^{D\times D}\) the bilinear blocks, \(B \in \mathbb{R}^{D\times K}\) the direct action input and \(c \in \mathbb{R}^D\) the offset. The map is affine in \(z\) for fixed \(a\) and affine in \(a\) for fixed \(z\). It is not affine in \((z, a)\) jointly, and the bilinear blocks \(T_k\) are the only source of that failure. With \(T_k = 0\) and \(B = 0\), the operator reduces to the autonomous dynamics \(z \mapsto T_0 z + c\) for every action.

**Fit.** \(W\) is fitted by ridge regression in closed form. Here \(X\) stacks the lifted rows \(\varphi_t\), \(Y\) stacks the targets \(z_{t+1}\), and \(\lambda > 0\) is the ridge:

\[
W = \operatorname*{arg\,min}_W \sum_t \big\lVert W\varphi_t - z_{t+1}\big\rVert^2 + \lambda \lVert W\rVert_F^2 = Y^\top X\,(X^\top X + \lambda I)^{-1}.
\]

It is solved by a Cholesky factorisation of \(X^\top X + \lambda I\), which is symmetric positive definite for every \(\lambda > 0\).

## Fixed points and stability

Let \(\rho = \sigma_{\max}(T_0)\) be the largest singular value of the state block. It is computed as the square root of the largest eigenvalue of \(T_0^\top T_0\) by cyclic Jacobi rotation. The autonomous map \(F(z) = T_0 z + c\) has the fixed point \(z^* = (I - T_0)^{-1} c\) whenever \(I - T_0\) is invertible. If \(\rho < 1\), then \(F\) is a contraction in the Euclidean norm:

\[
\lVert F(z) - F(w)\rVert \le \rho\,\lVert z - w\rVert,
\qquad
\lVert F^n(z_0) - z^*\rVert \le \rho^n\,\lVert z_0 - z^*\rVert .
\]

By the Banach fixed-point theorem, \(z^*\) is then unique and every rollout converges to it geometrically.

**Scalar rollout bound.** Suppose every one-step residual has RMS norm at most \(\varepsilon\). The accumulated error after \(n\) steps then satisfies

\[
E(n) \le \varepsilon \sum_{i < n} \rho^i = \varepsilon\,\frac{1 - \rho^n}{1 - \rho} \quad (\rho < 1),
\qquad
E(n) \le \varepsilon\, n\,\rho^{n-1} \quad (\rho \ge 1).
\]

**Directional estimate.** This propagates the residual second moment \(\Sigma\) instead of the scalar \(\varepsilon\):

\[
C_n = T_0\, C_{n-1}\, T_0^\top + \Sigma, \qquad C_0 = 0, \qquad \operatorname{estimate}(n) = \sqrt{\operatorname{tr} C_n / D}.
\]

**Spectral ceiling.** An optional ceiling \(\rho_{\max}\) rescales every singular value of \(T_0\) by \(\rho_{\max}/\rho\). That is exactly \(T_0 \leftarrow (\rho_{\max}/\rho)\,T_0\).

## Islands

For points \(x_i\) and an absolute radius \(r \ge 0\), points \(i\) and \(j\) lie in the same island when a chain of pairs at distance at most \(r\) joins them. The island count equals \(\beta_0\) of the Vietoris–Rips complex at \(r\). That is the number of \(H_0\) bars with death \(> r\), the essential bar included, because the single-linkage merge heights are the \(H_0\) death times. The multi-body coupling diagnostic is the matrix of block Frobenius norms

\[
S_{ij} = \big\lVert T_0[\text{body } i,\ \text{body } j]\big\rVert_F .
\]

## Proved versus measured

**Proved:**

- The contraction inequality and the Banach fixed point when \(\rho < 1\), for the autonomous map only. With actions, the per-step state map is \(T_0 + \sum_k a_k T_k\), whose norm \(\rho\) does not bound.
- The scalar error bound, under the hypothesis that every one-step residual has RMS norm at most \(\varepsilon\).
- That the spectral ceiling produces \(\sigma_{\max} = \rho_{\max}\).
- The identity between island count and \(\beta_0\).

**Not proved:**

- The fitted \(\varepsilon\) is an in-sample root mean square, not a supremum. A certificate built from a fit inherits the hypothesis rather than discharging it.
- The directional estimate is an expectation under zero-mean, step-uncorrelated residuals and linear dynamics. It is not a bound. The source measured it under-bounding when residuals are autocorrelated, which is why the fit reports the lag-1 residual autocorrelation beside it.
- Convergence of the Banach iteration. It is measured on every call rather than asserted.
- Usefulness on transformer activations. The source reports the scalar bound as vacuous there, and the certificate as not established as useful.

**The ceiling is never applied by default.** A chaotic system has \(\rho > 1\). Clipping \(\rho\) buys a certificate by misreporting the dynamics. The source measured a loss of one-step accuracy from doing so and made it opt-in.

## Declined claim: the Lyapunov-gain descent condition

The source also carries a `LyapunovGain` claim. It concerns a correction law with proportional gain \(\alpha\), derivative gain \(\beta\) and time step \(\mathrm{d}t\), and states that \(\alpha + \beta/\mathrm{d}t < 1\) gives energy descent. The port declined to carry it, because it is false in general. Under that law, the state error \(\tilde e_t\) follows the closed loop

\[
\tilde e_t = (1 - \mathrm{d}t\,\alpha - \beta)\,T_0\,\tilde e_{t-1} + \beta\,T_0\,\tilde e_{t-2}.
\]

Whether the error decays depends on the roots of this recurrence's characteristic polynomial, and those roots involve \(T_0\) as well as the gains. The port report gives the counterexample. At \(T_0 = 10I\) with the source's default gains, the recurrence has the root \((3 + \sqrt{17})/2 \approx 3.56\), outside the unit circle, so the error grows. The module documentation records the adjacent decisions: the ceiling is opt-in, and the contraction statement is restricted to the autonomous map.

## Refusals

| Variant | Condition |
| --- | --- |
| `NonFinite` | A state, action, point or target contains a NaN or an infinity |
| `InvalidRidge` | \(\lambda\) is not finite and strictly positive |
| `InvalidCeiling` | \(\rho_{\max}\) is negative or not finite |
| `InvalidRadius` | The island radius is negative or not finite |
| `LengthMismatch` | Two slices that must agree in length do not |
| `TooFewTransitions` | Fewer than two transitions were supplied |
| `NotPositiveDefinite` | A Cholesky pivot of \(X^\top X + \lambda I\) is not positive in floating point |
| `IndivisibleBodies` | The body dimension is zero or does not divide \(D\) |

`fixed_point` returns `None` when \(I - T_0\) is exactly singular. A non-finite operator reports \(\rho = \mathrm{NaN}\), which no certificate treats as contractive or safe. An expansive rollout may overflow to infinity; that is reported, not refused.

## Rust API

```rust
pub struct CouplingOperator<const D: usize, const K: usize> {
    pub state: [[f64; D]; D],           // T₀
    pub bilinear: [[[f64; D]; D]; K],   // T_k
    pub input: [[f64; K]; D],           // B
    pub bias: [f64; D],                 // c
}
impl<const D: usize, const K: usize> CouplingOperator<D, K> {
    pub fn autonomous(state: [[f64; D]; D], bias: [f64; D]) -> Self;
    pub fn step(&self, z: &[f64; D], action: &[f64; K]) -> [f64; D];
    pub fn rollout(&self, z0: &[f64; D], actions: &[[f64; K]], out: &mut [[f64; D]])
        -> Result<(), CouplingError>;
    pub fn spectral_norm(&self) -> f64;
    pub fn project_spectral(&mut self, rho_max: f64) -> Result<f64, CouplingError>;
    pub fn fixed_point(&self) -> Option<FixedPoint<D>>;
    pub fn coupling_strength(&self, body_dim: usize, out: &mut [f64]) -> Result<usize, CouplingError>;
    pub fn fit(states: &[[f64; D]], next_states: &[[f64; D]], actions: &[[f64; K]],
               ridge: f64, rho_max: Option<f64>) -> Result<Fit<D, K>, CouplingError>;
}

pub struct FixedPoint<const D: usize> { pub point: [f64; D], pub converged: bool }
pub struct Fit<const D: usize, const K: usize> {
    pub operator: CouplingOperator<D, K>, pub rho: f64, pub step_rmse: f64,
    pub residual_cov: [[f64; D]; D], pub residual_autocorr: f64,
}
impl<const D: usize, const K: usize> Fit<D, K> { pub fn certificate(&self) -> RolloutCertificate<D>; }

pub struct RolloutCertificate<const D: usize> {
    pub rho: f64, pub step_rmse: f64, pub state_block: [[f64; D]; D], pub residual_cov: [[f64; D]; D],
}
impl<const D: usize> RolloutCertificate<D> {
    pub fn contractive(&self) -> bool;                 // ρ < 1; false for NaN
    pub fn error_bound(&self, steps: usize) -> f64;
    pub fn directional_error(&self, steps: usize) -> f64;
    pub fn safe_horizon(&self, tolerance: f64, directional: bool) -> usize;
}

pub enum IslandMetric { Euclidean, Geodesic }
pub fn island_labels<const N: usize>(points: &[ManifoldPoint<N>], radius: f64, metric: IslandMetric,
                                     labels: &mut [usize]) -> Result<usize, CouplingError>;
```

`step` accumulates in the column order of \(\varphi(z, a)\), so its result is bit-identical to the explicit product \(W\varphi(z, a)\). The Banach iteration in `fixed_point` runs from the origin for at most 512 steps. It stops when a step moves less than \(10^{-12}\), and sets `converged` only if the iterate then lies within \(10^{-6}\) of \(z^*\). `safe_horizon` reports at most 4096 steps and treats a NaN error as unsafe. `Geodesic` is the great-circle metric of the `mujoco#3396` corpus, \(\arccos(\operatorname{clamp}(x \cdot y, -1, 1))\), and is meaningful for unit vectors only.

## Test evidence

`tests/coupling.rs` holds 18 `#[test]` functions. The operator is small enough that a bug in it does not crash. A transposed block, a lift column out of order, a spectral radius read where a spectral norm was meant, or a `<` where `≤` belongs all return plausible numbers. Each test is a property that one of those bugs violates.

```bash
cargo test -p aether-core --test coupling
```

| Test | Pins |
| --- | --- |
| `step_matches_the_naive_lifted_matvec_bit_for_bit` | `step` equals the explicit \(W\varphi(z, a)\), bitwise |
| `ridge_fit_matches_a_naive_normal_equation_solve` | The Cholesky fit agrees with a Gauss–Jordan normal-equation solve |
| `step_is_affine_in_each_argument_and_not_jointly` | Affine in \(z\), affine in \(a\), not jointly, and not linear |
| `rollout_is_the_composition_of_single_steps` | Rollouts compose |
| `zero_coupling_reduces_to_the_uncoupled_dynamics` | \(T_k = 0\), \(B = 0\) gives the autonomous map |
| `spectral_norm_matches_power_iteration_and_closed_forms` | Jacobi \(\sigma_{\max}\) against power iteration and closed forms |
| `spectral_projection_is_a_uniform_rescale_onto_the_ceiling` | \(T_0 \leftarrow (\rho_{\max}/\rho)\,T_0\) |
| `contraction_inequality_holds_on_seeded_random_states_and_is_attained` | \(\lVert Tz - Tw\rVert \le \rho\lVert z - w\rVert\), attained |
| `fixed_point_is_fixed_and_convergence_is_measured_not_assumed` | \(z^*\) is fixed. `converged` is false where iteration cannot reach it |
| `scalar_bound_is_the_geometric_series_holds_on_bounded_residuals_and_is_attained` | \(E(n)\) is the geometric series, holds, and is attained |
| `directional_estimate_matches_its_closed_form_and_the_fit_statistics` | \(C_n\) recursion and the fit's residual statistics |
| `safe_horizon_inverts_the_bound_and_treats_nan_as_unsafe` | Horizon inversion; NaN counts as unsafe |
| `fit_is_equivariant_under_relabelling_of_state_components` | Permuting state components permutes the fit |
| `coupling_strength_recovers_a_driver_follower_pair_under_either_labelling` | \(S_{ij}\) identifies the driver under either body order |
| `island_partition_is_invariant_under_relabelling_of_points` | Seeds 3, 17, 29, 101 |
| `island_labels_match_a_brute_force_flood_fill` | Seeds 1, 2, 5, 8, 13, 21 |
| `island_count_equals_h0_bars_alive_past_the_radius` | Against `persistence::persistent_homology`, seeds 4, 9, 16, 25, 36 |
| `malformed_and_non_finite_input_is_refused` | Every refusal variant |

## Not ported

The persistence encoder that produces the state in sigmoid's engine is not ported. Any fixed-length state vector in \(\mathbb{R}^D\) is a valid input, for example `SystemState::vector`. The `LyapunovGain` descent claim is declined, as recorded above.

## Provenance

[sigmoid](https://github.com/teerthsharma/sigmoid): the operator from `sigmoid/operator.py` (`CouplingOperator`: `step`, `rollout`, `fit`, `certificate`, `safe_horizon`, `_project_spectral`, `_power_iterate`), block coupling strength from `sigmoid/nbody.py` (`MultiBodyCoupling.coupling_strength`), and island partitions from `sigmoid/mujoco/island.py` (`island_labels`, `island_count`), whose canonical labelling follows MuJoCo's `mj_island`.
