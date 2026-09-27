# Monodromy Deciders

Module: `aether_core::monodromy`, in `crates/aether-core/src/monodromy.rs`. It decides four properties of a map, or of a cloud sampled from one: injectivity, symmetry, dimension and sensitivity. The decisions read sampled geometry rather than \(\det DF\). Evidence: 18 tests in `tests/monodromy.rs`.

!!! note "Where the Jacobian is and is not used"
    Only the **injectivity decision** is Jacobian-free. It reads two points at a time and differentiates nothing. The **Lyapunov spectrum** is the growth rate of the tangent cocycle, so it consumes \(DF\) along an orbit, exactly as the source's `dynamics.map_spectrum` does. "Without the Jacobian" refers to the injectivity decision, where the determinant is provably silent. It does not refer to the chaos estimator.

## Why the determinant is not the instrument

The standard applied certificate of invertibility is local: \(\det DF(x) \ne 0\) at every sampled \(x\). That establishes a local diffeomorphism and nothing more. The counterexamples to the Jacobian conjecture (arXiv:2608.00222) are everywhere unramified and still not injective. So no function of \(DF\) evaluated one point at a time decides the global question.

The decision thresholds below are the source's, each carried with the calibration it was set from. None was re-fitted here.

## 1. Injectivity: the two-point ratio

For a map \(F\) and a finite sample \(X\):

\[
\lambda(F; X) = \min_{p \ne q \in X} \frac{\lVert F(p) - F(q)\rVert}{\lVert p - q\rVert},
\qquad
\rho_{\text{free}}(F; X) = \frac{\lambda(F; X)}{\operatorname{median}_{p \ne q \in X} \lVert F(p) - F(q)\rVert / \lVert p - q\rVert},
\]

\[
\ln \tilde\lambda(n) = a + b \ln n,
\]

where \(\tilde\lambda(n)\) is the median of \(\lambda\) over repeated draws of size \(n\), and \(b\) is the fitted slope with standard error \(\operatorname{se}(b)\). \(F\) is injective on a set exactly when the infimum of the ratio over that set is positive. A sampled \(\lambda\) is an **upper** estimate of that infimum, because a minimum over a subset is never below the minimum over the whole. \(\rho_{\text{free}}\) normalises it by the median of the same ratio over the same pairs.

**Decided:** only "collision exhibited", with the witnessing pair. The complement is "no collision at this sampling", never "injective". The verdict type has no variant for the stronger claim. The rule:

- fewer than `MIN_SIZES` sizes, or a size span below `MIN_SPAN`: undecided;
- largest size at least `FREE_MIN_N`: collision iff \(\rho_{\text{free}} < \texttt{FREE\_LEVEL}\);
- otherwise the slope decides: collision iff \(b < \texttt{DECAY}\) and \(b + 3\operatorname{se}(b) < 0\).

**Estimated:** \(\lambda\), \(\rho_{\text{free}}\), the slope and its standard error.

**Failure regime.**

- \(\rho_{\text{free}}\) is a minimum over \(O(n^2)\) pairs and falls with \(n\). `FREE_LEVEL` is therefore valid only at or above `FREE_MIN_N`. The source measured a 3-D non-injective map at \(2.050 \times 10^{-2}\) at \(n = 200\), above the level, and missed it.
- The level was set from eight 2-D maps (gap factor 3.02) and checked on three 3-D maps, which is thin.
- A collision confined to a region the sampler never reaches, or escaping to infinity as in arXiv:2608.00222, is invisible to any bounded sample.
- An injective map with a **critical point** reads as a collision. Near a zero of \(DF\), the ratio of two adjacent samples tends to \(\lvert DF\rvert \to 0\), exactly as it does across a fibre. \(x \mapsto x^3\) at the origin is the minimal case. The witness distinguishes the two situations (adjacent points, against separated points with a common image), but the verdict does not.
- The slope is noisy. The source measured \(-0.261\) over \(n = 100..400\) and \(-0.657\) over \(n = 100..800\) on one map.

## 2. Symmetry: defects that vanish on the symmetry set

For a cloud \(X\), a map \(g\) and its image \(gX\):

\[
\begin{aligned}
D_{\text{ch}}(X, gX) &= \operatorname{mean}_{p \in X} d(p, gX) + \operatorname{mean}_{q \in gX} d(q, X),\\
D_{\text{PH}}(g; X) &= \max_{k \in \{0, 1\}} d_B\big(\mathrm{PH}_k(X),\ \mathrm{PH}_k(X \cup gX)\big),\\
E(g; F, X) &= \max_{x \in X} \lVert F(gx) - g\,F(x)\rVert_\infty,\\
\delta(\theta) &= D_{\text{ch}}(X, R_\theta X), \qquad R_\theta \text{ a rotation about the centroid, } \theta \text{ on a uniform grid},\\
s(n) &= \frac{\sum_{m \ge 1} P(mn)}{\sum_{k \ge 1} P(k)}, \qquad P = \big\lvert \mathrm{DFT}(\delta - \operatorname{mean}\delta)\big\rvert^2 .
\end{aligned}
\]

\(d(p, Y)\) is the distance from \(p\) to the nearest point of \(Y\), \(d_B\) is the bottleneck distance and \(\mathrm{PH}_k\) is the degree-\(k\) persistence diagram. \(D_{\text{ch}}\) and \(D_{\text{PH}}\) vanish exactly when \(gX = X\) as sets. Comparing \(\mathrm{PH}(X)\) with \(\mathrm{PH}(gX)\) would not work: Rips persistence depends only on the distance matrix, so the difference is zero for every isometry. The union is what moves. \(E\) is a statement about a map rather than a cloud. It is zero exactly when \(g\) commutes with \(F\).

**Decided:** the rotation order \(n\) is the largest candidate whose harmonic series carries \(s(n) \ge \texttt{HARMONIC\_TAU}\), scaled by \(360^\circ/\text{span}\). Three further checks follow:

- The order is kept only if the defect at \(360^\circ/n\) is within `GRAIN_LEVEL` mean nearest-neighbour spacings. Periodicity is not symmetry.
- A multiple \(mn\), for \(m \le 6\), is promoted when the off-grid defect at \(360^\circ/(mn)\) falls below `PROMOTE_DEPTH` times the profile median.
- Reflections are declared when an order above 1 exists and the reflection profile dips at least `REFLECTION_DEPTH` as deep as the rotation profile.

**Estimated:** the power fraction, the grain ratio and the depth ratio.

**Failure regime.** The procedure is sound, not complete. On regular-polygon outlines with Gaussian jitter of \(\sigma\) = pct \(\times\) mean radius, the source measured the following recovery:

| Jitter | 0 % | 2 % | 5 % | 8 % | 10 % |
| --- | --- | --- | --- | --- | --- |
| Recovered | 6/6 | 5/6 | 4/6 | 2/6 | 2/6 |

Every miss was order 1 or a proper divisor of the true order. None of 24 symmetryless Gaussian clouds was assigned a group. Order 1 therefore means "nothing found", not "no symmetry". Chamfer is the default for robustness and cost: \(D_{\text{PH}}\) is read through a supremum, so one displaced point moves it by that point's own displacement. The source measured 0.1 s against 846.77 s per profile at \(n = 240\).

## 3. Dimension: Schweinhart's persistent-homology dimension

\[
T_\alpha(X) = \sum_{\text{finite } H_0 \text{ bars}} (\text{death} - \text{birth})^\alpha,
\qquad
\mathbb{E}\,T_\alpha(X_n) \sim n^{(d - \alpha)/d},
\]

\[
\ln T_\alpha(n) = c + s \ln n, \qquad \hat d = \frac{\alpha}{1 - s}, \qquad 0 < \alpha < d .
\]

\(X_n\) is a sample of \(n\) points, \(d\) the dimension being estimated, and \(s\) the fitted slope. At degree 0 the Rips barcode is the minimum spanning tree, so this is Steele's theorem on the \(\alpha\)-weighted MST. The 95 % interval is the image of \(s \pm 1.96\operatorname{se}(s)\) under \(s \mapsto \alpha/(1 - s)\).

**Decided:** Mandelbrot's predicate \(d_H > d_{\text{top}}\), and only with the interval. The verdict is `Fractal` when the lower end exceeds \(d_{\text{top}}\), `NotFractal` when the upper end does not, and `Undecided` otherwise. **Estimated:** \(\hat d\) and its interval.

**Failure regime.**

- The estimate is biased downward in high dimension, and more data does not cure it. The source measured 8.660 at \(n \le 400\) and 8.531 at \(n \le 1600\) on the 10-cube. Trust it to intrinsic dimension about 3, treat 3 to 6 as indicative, and do not read an absolute value above 6.
- The interval models sampling noise, not bias. At small \(n\) a set with boundary reads high. The expected MST length of \(n\) uniform points on a unit segment is \(1 - 2/(n + 1)\), so the slope is positive, \(\hat d\) sits just above 1, and the interval can exclude 1. The predicate then calls a segment fractal against \(d_{\text{top}} = 1\). `tests/monodromy.rs` pins this failure.
- The value moves with the size grid. For the Sierpinski gasket the source reports 1.6246, 1.6163 and 1.6033 on three grids, against \(\log 3/\log 2 = 1.5850\).
- Consecutive iterates of a map are correlated draws, which biases the estimate low.

## 4. Sensitivity: the Lyapunov spectrum by QR (Benettin)

For Jacobians \(J_1, \ldots, J_N\) along an orbit, time step \(\mathrm{d}t\) and state dimension \(D\):

\[
Q_0 = I, \qquad J_k Q_{k-1} = Q_k R_k, \quad (R_k)_{ii} > 0, \qquad
\lambda_i = \frac{1}{N\,\mathrm{d}t} \sum_{k=1}^{N} \ln (R_k)_{ii},
\]

\[
D_{KY} = j + \frac{\sum_{i \le j} \lambda_i}{\lvert\lambda_{j+1}\rvert}, \qquad j \text{ the largest index with } \textstyle\sum_{i \le j} \lambda_i \ge 0 .
\]

Multiplying the Jacobians directly collapses every column onto the leading direction. Re-orthonormalising at each step keeps all \(D\) directions.

**Decided:** nothing. The spectrum and \(D_{KY}\) are estimates. The source attaches no threshold to the sign of \(\lambda_1\).

**Failure regime.**

- A finite \(N\) is not the limit that Oseledets' theorem speaks of, and that theorem's hypotheses (an invariant measure, a genuine cocycle, integrability) are not checked.
- The trace identity \(\sum \lambda_i = \operatorname{mean} \ln\lvert\det J\rvert\) holds even for an implementation that drops the QR frame entirely. So it does not test the individual exponents.
- A singular step is clamped at \(\ln 10^{-300} \approx -690\) rather than producing NaN.
- The Kaplan–Yorke dimension equals the information dimension only under the Kaplan–Yorke conjecture.

## Thresholds and their calibration

| Constant | Value | Calibration in the source |
| --- | --- | --- |
| `MAX_N` | 4000 | Memory wall: two \(n \times n\) distance matrices at \(n = 4000\) are 256 MB |
| `FREE_LEVEL` | \(8.8 \times 10^{-3}\) | Geometric midpoint of the gap on eight 2-D maps: injective \(1.53 \times 10^{-2}\) to \(7.06 \times 10^{-1}\), non-injective \(1.04 \times 10^{-3}\) to \(5.07 \times 10^{-3}\) |
| `FREE_MIN_N` | 400 | The level was set at \(n \ge 400\) |
| `DECAY` | \(-0.25\) | Between measured slopes of \(-0.070\) (injective) and \(-0.657\) (non-injective) |
| `SLOPE_SIGMA` | 3.0 | Standard errors the slope must clear zero by |
| `MIN_SIZES` | 4 | Fewest sample sizes for a verdict |
| `MIN_SPAN` | 8.0 | With less range the slope straddled any usable cutoff |
| `HARMONIC_TAU` | 0.85 | Swept from 0.80 to 0.90 on eight clean outlines and twelve symmetryless clouds. 0.85 lost no clean shape and admitted no false one |
| `PROMOTE_DEPTH` | 0.5 | Gap: 0.0000 for true multiples, 0.8741 for the deepest spurious one |
| `GRAIN_LEVEL` | 2.0 | Gap: true symmetries 0.0000 to 1.6223, swiss-roll modulations 2.3737 to 2.8243 |
| `REFLECTION_DEPTH` | 0.90 | Gap: regular polygons 1.0000, deepest chiral pinwheel 0.8123 |
| `DEFAULT_GRID_STEP_DEG` | 2.0 | Grid step at which the recovery envelope was measured |

## Refusals

A refusal is a `MonodromyError`. Each one marks an input on which the statistic would otherwise return a plausible number that means nothing.

| Part | Refused inputs |
| --- | --- |
| Injectivity | Fewer than two points, or no pair of distinct points. More than `MAX_N` points. An empty size list or zero repeats |
| Symmetry | Fewer than two points. A non-finite coordinate. Zero diameter. A grid step that is not positive or gives fewer than four samples. Clouds of different sizes where one must be the image of the other. Anything the persistence engine refuses under `PersistenceConfig::h1_dense` |
| Dimension | \(\alpha\) not positive and finite. Fewer than three draws with positive total persistence. A sampler that delivers one size for every request. The regression uses the sizes actually delivered, never the sizes requested |
| Sensitivity | An empty Jacobian sequence, or \(\mathrm{d}t\) not positive. A non-finite spectrum passed to `kaplan_yorke_dimension` is refused: NaN compares false against every bound, and would otherwise read as the legitimate answer 0.0 |

## Rust API

```rust
// 1. Injectivity
pub fn lower_ratio_witness<const D: usize, F>(f: F, x: &[ManifoldPoint<D>])
    -> Result<Witness<D>, MonodromyError> where F: Fn(&[f64; D]) -> [f64; D];
pub fn free_ratio<const D: usize, F>(f: F, x: &[ManifoldPoint<D>]) -> Result<f64, MonodromyError>
    where F: Fn(&[f64; D]) -> [f64; D];
pub fn ratio_scaling<const D: usize, F, S>(f: F, sampler: S, sizes: &[usize], repeats: usize, seed: u64)
    -> Result<RatioScaling, MonodromyError>
    where F: Fn(&[f64; D]) -> [f64; D], S: FnMut(usize, u64) -> Vec<ManifoldPoint<D>>;
pub fn collision_certificate<const D: usize, F, S>(f: F, sampler: S, sizes: &[usize], repeats: usize, seed: u64)
    -> Result<CollisionCertificate<D>, MonodromyError>;
pub enum CollisionVerdict { CollisionExhibited, NoCollisionAtThisSampling, Undecided }

// 2. Symmetry
pub fn rotation(theta_deg: f64) -> [[f64; 2]; 2];
pub fn reflection(theta_deg: f64) -> [[f64; 2]; 2];
pub fn chamfer_defect<const D: usize>(x: &[ManifoldPoint<D>], gx: &[ManifoldPoint<D>]) -> Result<f64, MonodromyError>;
pub fn persistence_defect<const D: usize>(x: &[ManifoldPoint<D>], gx: &[ManifoldPoint<D>]) -> Result<f64, MonodromyError>;
pub fn equivariance_defect<const D: usize, F>(step: F, g: &[[f64; D]; D], x: &[ManifoldPoint<D>])
    -> Result<f64, MonodromyError>;
pub fn recover_cyclic(x: &[ManifoldPoint<2>], step_deg: f64) -> Result<GroupReport, MonodromyError>;
pub fn recover_dihedral(x: &[ManifoldPoint<2>], step_deg: f64) -> Result<GroupReport, MonodromyError>;

// 3. Dimension
pub fn alpha_total_persistence<const D: usize>(x: &[ManifoldPoint<D>], alpha: f64) -> Result<f64, MonodromyError>;
pub fn ph_dimension<const D: usize, S>(sampler: S, alpha: f64, sizes: &[usize], repeats: usize, seed: u64)
    -> Result<PhDimension, MonodromyError>;
pub fn is_fractal<const D: usize, S>(sampler: S, topological_dim: f64, alpha: f64, sizes: &[usize],
    repeats: usize, seed: u64) -> Result<(FractalVerdict, PhDimension), MonodromyError>;
pub enum FractalVerdict { Fractal, NotFractal, Undecided }

// 4. Sensitivity
pub fn lyapunov_spectrum<const D: usize, I>(jacobians: I, dt: f64) -> Result<[f64; D], MonodromyError>
    where I: IntoIterator<Item = [[f64; D]; D]>;
pub fn map_spectrum<const D: usize, St, Ja>(step: St, jac: Ja, x0: [f64; D], n: usize, burn: usize)
    -> Result<[f64; D], MonodromyError>;
pub fn kaplan_yorke_dimension<const D: usize>(lam: [f64; D]) -> Result<f64, MonodromyError>;
```

The symmetry and dimension parts reuse the crate's own `persistence::persistent_homology` and `diagram::bottleneck_distance`.

## Test evidence

`tests/monodromy.rs` holds 18 `#[test]` functions. Every assertion is scored against an answer known in closed form, not against a previously recorded output of the estimator.

```bash
cargo test -p aether-core --test monodromy
```

| Test | Pins |
| --- | --- |
| `a_fold_collides_across_its_fibre_and_the_witness_is_the_fibre` | \(x \mapsto x^2\) collides, and the witness pair is \(\{p, -p\}\) |
| `the_wrapped_exp_map_collides_although_its_determinant_never_vanishes` | The exp map on two periods collides, with \(\det DF = e^{2x} > 0\) everywhere |
| `injective_controls_are_cleared_and_the_ratio_is_scale_free` | Injective controls are not flagged. The ratio is scale-free |
| `a_critical_point_reads_as_a_collision_and_the_witness_says_so` | \(x \mapsto x^3\) is flagged at its critical point, and the witness shows adjacent points |
| `short_size_ranges_and_degenerate_samples_are_refused` | `Undecided` below `MIN_SIZES`/`MIN_SPAN`, and the degenerate refusals |
| `regular_polygons_recover_their_order_wherever_they_sit` | \(C_n\) is recovered under translation and scale |
| `mirrors_are_found_on_a_polygon_and_absent_on_a_pinwheel` | Polygons are dihedral. Pinwheels are chiral |
| `clouds_without_symmetry_are_assigned_no_group` | Gaussian clouds get order 1 |
| `set_defects_vanish_exactly_on_symmetries` | \(D_{\text{ch}}\) and \(D_{\text{PH}}\) are zero exactly on symmetries |
| `only_the_odd_map_commutes_with_the_point_reflection` | The Holmes map commutes with \(-I\). Hénon does not |
| `flat_sets_have_their_integer_dimension` | Line within 0.1 of 1, square within 0.15 of 2 |
| `curved_manifolds_have_their_intrinsic_dimension` | Circle within 0.1 of 1, torus within 0.2 of 2 |
| `dimension_is_invariant_under_rigid_motion_scale_and_weight` | A similarity moves the square's estimate by less than \(10^{-9}\). \(\alpha = 1/2\) gives the square within 0.3 of 2; a formula that dropped \(\alpha\) would give about 4 |
| `the_fractal_predicate_is_decided_on_the_interval` | The Sierpinski gasket is `Fractal`, and a square against \(d_{\text{top}} = 2\) is `NotFractal`. A segment is `Fractal` against \(d_{\text{top}} = 1\): the documented bias, pinned |
| `the_logistic_map_at_r4_has_exponent_ln_2` | \(\lambda = \ln 2\) within 0.01 |
| `a_periodic_orbit_has_its_closed_form_negative_exponent` | At \(r = 3.2\), \(\lambda = \tfrac12 \ln 0.16\) within \(10^{-3}\) |
| `henon_exponents_match_sprott_one_by_one_not_just_in_sum` | \(+0.41922\) and \(-1.62319\) within 0.01 each, and \(D_{KY} = 1.25827\) within 0.01. The trace identity is only a control |
| `closed_form_cocycles_are_reproduced_exactly` | Diagonal, orthogonal and scaled cocycles to \(10^{-12}\). Kaplan–Yorke at its edges |

## Not ported

The source's remedy for the critical-point false positive is the étale branch, which computes \(\rho = \lambda / \sigma_{\min}(DF)\). It is not ported, because it needs \(DF\), and the injectivity decision here is Jacobian-free by design. The witness's domain gap (`CollisionCertificate::domain_gap`) is reported, so a caller can separate the two situations by hand.

## Provenance

[monodromy](https://github.com/teerthsharma/monodromy), version 0.1.0 (Zenodo DOI 10.5281/zenodo.22064739). Every public item names the source function it reproduces, for example `phdim.py::is_fractal` and `dynamics.map_spectrum`.
