//! Injectivity, symmetry, dimension and sensitivity decided from sampled
//! geometry rather than from `det DF`.
//!
//! Ported from monodromy (github.com/teerthsharma/monodromy, version 0.1.0,
//! Zenodo DOI 10.5281/zenodo.22064739). Every public item names the source
//! function it reproduces. The decision thresholds are the source's, carried
//! with the calibration each was set from; none was re-fitted here.
//!
//! # Why the determinant is not the instrument
//!
//! The standard applied certificate of invertibility is local: `det DF(x) != 0`
//! at every sampled `x`. That establishes a local diffeomorphism and nothing
//! more. The counterexamples to the Jacobian conjecture (arXiv:2608.00222) are
//! everywhere unramified and still not injective, so no function of `DF` at one
//! point at a time decides the global question. The injectivity decider below
//! reads two points at a time and differentiates nothing.
//!
//! The sensitivity estimator is the exception, and is stated as such: the
//! Lyapunov spectrum is the growth rate of the tangent cocycle, so it consumes
//! `DF` along an orbit, exactly as the source's `dynamics.map_spectrum` does.
//! "Without the Jacobian" refers to the injectivity decision, where the
//! determinant is provably silent, not to the chaos estimator.
//!
//! # 1. Injectivity: the two-point ratio
//!
//! ```text
//! λ(F; X)      = min_{p ≠ q ∈ X}  |F(p) − F(q)| / |p − q|
//! ρ_free(F; X) = λ(F; X) / median_{p ≠ q ∈ X} |F(p) − F(q)| / |p − q|
//! ln λ̃(n)      = a + b · ln n          (λ̃(n): median of λ over repeats)
//! ```
//!
//! `F` is injective on a set exactly when the infimum of the ratio over that set
//! is positive. A sampled `λ` is an **upper** estimate of that infimum, because a
//! minimum over a subset is never below the minimum over the whole, so the raw
//! value has no absolute scale. `ρ_free` normalises it by the median of the same
//! ratio over the same pairs, so numerator and denominator are both distances
//! between sampled points.
//!
//! **Decided:** only "collision exhibited", with the witnessing pair. The
//! complement is "no collision at this sampling", never "injective"; the
//! verdict type has no variant for the stronger claim. The rule is
//!
//! * fewer than [`MIN_SIZES`] sizes, or a span below [`MIN_SPAN`]: undecided;
//! * largest size at least [`FREE_MIN_N`]: collision iff `ρ_free < FREE_LEVEL`;
//! * otherwise the slope decides: collision iff `b < DECAY` and
//!   `b + 3·se(b) < 0`.
//!
//! **Estimated:** `λ`, `ρ_free`, the slope and its standard error.
//!
//! **Bias and failure regime.** `ρ_free` is a minimum over `O(n²)` pairs and
//! falls with `n`, which is why [`FREE_LEVEL`] is valid only at or above
//! [`FREE_MIN_N`]; the source measured a 3-D non-injective map at 2.050e-2 at
//! `n = 200`, above the level, and missed. The level was set from eight 2-D maps
//! (gap factor 3.02) and checked on three 3-D maps, which is thin. A collision
//! confined to a region the sampler never reaches, or escaping to infinity as in
//! arXiv:2608.00222, is invisible to any bounded sample. An injective map with a
//! **critical point** is read as a collision: near a zero of `DF` the ratio of
//! two adjacent samples tends to `|DF| → 0` exactly as it does across a fibre.
//! `x ↦ x³` at the origin is the minimal case. The witness separates the two
//! situations (adjacent points against separated points with a common image),
//! but the verdict does not; the source's remedy, the étale branch through
//! `ρ = λ / σ_min(DF)`, needs `DF` and is not ported. The slope is noisy: the
//! source measured −0.261 over `n = 100..400` and −0.657 over `n = 100..800` on
//! one map, which is why it only decides below [`FREE_MIN_N`].
//!
//! **Refusals:** fewer than two points, or no pair of distinct points; more than
//! [`MAX_N`] points (the pairwise work is `O(n²)` in memory); an empty size list
//! or zero repeats.
//!
//! # 2. Symmetry: defects that vanish on the symmetry set
//!
//! ```text
//! D_ch(X, gX) = mean_{p∈X} d(p, gX) + mean_{q∈gX} d(q, X)
//! D_PH(g; X)  = max_{k∈{0,1}} d_B( PH_k(X), PH_k(X ∪ gX) )
//! E(g; F, X)  = max_{x∈X} ‖F(gx) − g F(x)‖_∞
//! δ(θ)        = D_ch(X, R_θ X),  R_θ acting about the centroid, θ on a uniform grid
//! s(n)        = Σ_{m≥1} P(m n) / Σ_{k≥1} P(k),   P = |DFT(δ − mean δ)|²
//! ```
//!
//! `D_ch` and `D_PH` vanish exactly when `gX = X` as sets. Comparing `PH(X)`
//! with `PH(gX)` would not: Rips persistence depends only on the distance
//! matrix, so it is identically zero for every isometry; the union is what
//! moves. `E` is a statement about a map rather than a cloud: zero exactly when
//! `g` commutes with `F`.
//!
//! **Decided:** the rotation order `n` is the largest candidate whose harmonic
//! series carries `s(n) ≥ HARMONIC_TAU`, scaled by `360° / span`. It is kept only
//! if the defect at `360°/n` is within [`GRAIN_LEVEL`] mean nearest-neighbour
//! spacings (periodicity is not symmetry), and multiples `m·n`, `m ≤ 6`, are
//! promoted when the off-grid defect at `360°/(m n)` falls below
//! [`PROMOTE_DEPTH`] times the profile median (a grid cannot place a minimum it
//! does not sample). Reflections are declared when an order above 1 exists and
//! the reflection profile dips at least [`REFLECTION_DEPTH`] as deep as the
//! rotation profile, depth being `(median − min) / median` with both minima
//! refined off the grid.
//!
//! **Estimated:** the power fraction, the grain ratio, the depth ratio.
//!
//! **Bias and failure regime.** Sound, not complete. On regular-polygon outlines
//! with Gaussian jitter of σ = pct × mean radius the source measured 6/6, 5/6,
//! 4/6, 2/6 and 2/6 recovered at 0, 2, 5, 8 and 10 % jitter; every miss was
//! order 1 or a proper divisor of the true order, and 0 of 24 symmetryless
//! Gaussian clouds were assigned a group. Order 1 therefore means "nothing
//! found", not "no symmetry". `D_PH` is read through a supremum, so one
//! displaced point moves it by its own displacement; the chamfer mean is the
//! source's default for that reason and for cost (0.1 s against 846.77 s per
//! profile at `n = 240` in the source's measurement).
//!
//! **Refusals:** fewer than two points, a non-finite coordinate, zero diameter
//! (every map is then trivially a symmetry and a defect of 0 carries no
//! information), a grid step that is not positive or yields fewer than four
//! samples, clouds of different sizes where one must be the image of the other,
//! and whatever the persistence engine refuses under
//! [`PersistenceConfig::h1_dense`].
//!
//! # 3. Dimension: Schweinhart's persistent-homology dimension
//!
//! ```text
//! T_α(X) = Σ_{finite H0 bars} (death − birth)^α
//! E[T_α(X_n)] ~ n^{(d − α)/d}
//! ln T_α(n) = c + s · ln n,     d̂ = α / (1 − s),     0 < α < d
//! ```
//!
//! At degree 0 the Rips barcode is the minimum spanning tree, so this is
//! Steele's theorem on the α-weighted MST. The interval is the image of
//! `s ± 1.96·se(s)` under `s ↦ α/(1 − s)`.
//!
//! **Decided:** Mandelbrot's predicate `d_H > d_top`, and only with the
//! interval: fractal when the lower end exceeds `d_top`, not fractal when the
//! upper end does not, undecided otherwise. **Estimated:** `d̂` and its interval.
//!
//! **Bias and failure regime.** Downward in high dimension, and not cured by
//! more data: the source measured 8.660 at `n ≤ 400` and 8.531 at `n ≤ 1600` on
//! the 10-cube. Trust it to intrinsic dimension about 3, treat 3 to 6 as
//! indicative, do not read an absolute value above 6. Where the bias is
//! downward, an estimate that already exceeds `d_top` is evidence for the
//! predicate and one below it is not evidence against. The interval models
//! sampling noise and not bias, and at small `n` a set with boundary reads
//! high: the expected MST length of `n` uniform points on a unit segment is
//! `1 − 2/(n + 1)`, so the slope is positive, `d̂` sits just above 1, and the
//! interval can exclude 1, in which case the predicate calls a segment fractal
//! against `d_top = 1` (pinned in `tests/monodromy.rs`). The value also moves
//! with the size grid (Sierpinski gasket: 1.6246, 1.6163, 1.6033 on three grids
//! in the source, against log 3 / log 2 = 1.5850), and consecutive iterates of
//! a map are correlated draws, which biases it low.
//!
//! **Refusals:** `α` not positive and finite; fewer than three draws with
//! positive total persistence; a sampler that delivers one size for every
//! request (there is no scaling to fit). The regression uses the sizes actually
//! delivered, never the sizes requested.
//!
//! # 4. Sensitivity: the Lyapunov spectrum by QR (Benettin)
//!
//! ```text
//! Q_0 = I,     J_k Q_{k−1} = Q_k R_k,     (R_k)_ii > 0
//! λ_i = (1 / (N·dt)) Σ_{k=1}^{N} ln (R_k)_ii
//! D_KY = j + (Σ_{i≤j} λ_i) / |λ_{j+1}|,  j the largest index with Σ_{i≤j} λ_i ≥ 0
//! ```
//!
//! Multiplying the Jacobians directly collapses every column onto the leading
//! direction; re-orthonormalising at each step keeps all `D` directions.
//!
//! **Decided:** nothing. The spectrum and `D_KY` are estimates; the source
//! attaches no threshold to the sign of `λ_1`.
//!
//! **Bias and failure regime.** A finite `N` is not the limit Oseledets'
//! theorem speaks of, and the hypotheses of that theorem (an invariant measure,
//! a genuine cocycle, integrability) are not checked. The trace identity
//! `Σ λ_i = mean ln |det J|` is conserved by an implementation that drops the
//! frame entirely, so it is not a test of the individual exponents. A singular
//! step is clamped at `ln 1e-300 ≈ −690` rather than producing NaN. The
//! Kaplan–Yorke dimension equals the information dimension only under the
//! Kaplan–Yorke conjecture.
//!
//! **Refusals:** an empty Jacobian sequence, `dt` not positive, a non-finite
//! spectrum passed to [`kaplan_yorke_dimension`] (NaN compares false against
//! every bound, and would otherwise read as the legitimate answer 0.0).

#![warn(missing_docs)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use libm::{ceil, cos, log, pow, round, sin, sqrt};

use crate::diagram::bottleneck_distance;
use crate::manifold::ManifoldPoint;
use crate::persistence::{persistent_homology, ComplexKind, PersistenceConfig, PersistenceError};

// ═══════════════════════════════════════════════════════════════════════════════
// Thresholds, each with the calibration it came from
// ═══════════════════════════════════════════════════════════════════════════════

/// Largest sample the pairwise statistics accept. The source's memory wall:
/// two `n × n` distance matrices at `n = 4000` are 256 MB.
pub const MAX_N: usize = 4000;

/// `ρ_free` below this exhibits a collision. The geometric midpoint of the
/// source's measured gap on eight 2-D maps (injective 1.53e-2 to 7.06e-1,
/// non-injective 1.04e-3 to 5.07e-3).
pub const FREE_LEVEL: f64 = 8.8e-3;

/// Smallest sample at which [`FREE_LEVEL`] is comparable. Below it the slope
/// decides, because `ρ_free` falls with `n` and the level was set at `n ≥ 400`.
pub const FREE_MIN_N: usize = 400;

/// Slope of `ln λ̃` against `ln n` below which `λ` counts as still falling.
/// Between the source's measured −0.070 (injective) and −0.657 (non-injective).
pub const DECAY: f64 = -0.25;

/// Standard errors the slope must clear zero by before it counts as steep.
pub const SLOPE_SIGMA: f64 = 3.0;

/// Fewest sample sizes from which a verdict is issued.
pub const MIN_SIZES: usize = 4;

/// Smallest ratio of largest to smallest sample size from which a verdict is
/// issued. With less range the source's slope straddled any usable cutoff.
pub const MIN_SPAN: f64 = 8.0;

/// Share of spectral power a candidate order's harmonic series must carry.
/// The source swept 0.80 to 0.90 on eight clean outlines and twelve
/// symmetryless clouds; 0.85 lost no clean shape and admitted no false one.
pub const HARMONIC_TAU: f64 = 0.85;

/// A multiple of the reported order is promoted when its off-grid defect falls
/// to this fraction of the profile median. Measured gap in the source: 0.0000
/// for true multiples against 0.8741 for the deepest spurious one.
pub const PROMOTE_DEPTH: f64 = 0.5;

/// Largest defect at `360°/n`, in mean nearest-neighbour spacings, that still
/// counts as a symmetry. Source gap: true symmetries 0.0000 to 1.6223, swiss-roll
/// modulations 2.3737 to 2.8243.
pub const GRAIN_LEVEL: f64 = 2.0;

/// Reflection depth over rotation depth at or above which a dihedral group is
/// declared. Source gap: regular polygons 1.0000, deepest chiral pinwheel 0.8123.
pub const REFLECTION_DEPTH: f64 = 0.90;

/// The grid step, in degrees, at which the source's recovery envelope was
/// measured.
pub const DEFAULT_GRID_STEP_DEG: f64 = 2.0;

/// Clamp for `(R_k)_ii` before its logarithm: a singular step reads as a dead
/// direction (`≈ −690`) instead of `−∞`.
const R_FLOOR: f64 = 1e-300;

// ═══════════════════════════════════════════════════════════════════════════════
// Refusals
// ═══════════════════════════════════════════════════════════════════════════════

/// Why a decider refused to answer.
///
/// A refusal is a verdict. Each variant marks an input on which the statistic
/// would otherwise return a plausible number that means nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MonodromyError {
    /// Fewer points, or Jacobians, than the statistic is defined on.
    TooFewPoints {
        /// Items supplied.
        actual: usize,
        /// Items required.
        min: usize,
    },
    /// More points than the pairwise work is budgeted for.
    TooManyPoints {
        /// Points supplied.
        actual: usize,
        /// The cap, [`MAX_N`].
        max: usize,
    },
    /// A coordinate is NaN or infinite.
    NonFinite,
    /// Every point coincides, so no pair of distinct points exists.
    ZeroDiameter,
    /// A cloud that must be the image of another has a different size.
    ShapeMismatch,
    /// A parameter lies outside its domain: a non-positive `α`, `dt` or grid
    /// step, zero repeats, or an empty size list.
    InvalidParameter,
    /// Fewer than three draws had positive total persistence, so no line can be
    /// fitted with an error estimate.
    NotEnoughSamples,
    /// The sampler delivered a single size for every request, so there is no
    /// scaling to fit.
    NoScaling,
    /// The persistence engine refused the filtration.
    Persistence(PersistenceError),
}

impl From<PersistenceError> for MonodromyError {
    fn from(err: PersistenceError) -> Self {
        Self::Persistence(err)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shared arithmetic
// ═══════════════════════════════════════════════════════════════════════════════

/// Median as numpy defines it: the mean of the two middle values for even length.
fn median(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// Ordinary least squares: (slope, intercept, residual sum of squares, Sxx).
fn ols(x: &[f64], y: &[f64]) -> (f64, f64, f64, f64) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxx: f64 = x.iter().map(|v| (v - mx) * (v - mx)).sum();
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let rss = x
        .iter()
        .zip(y)
        .map(|(a, b)| {
            let r = b - intercept - slope * a;
            r * r
        })
        .sum();
    (slope, intercept, rss, sxx)
}

/// The source's `check_cloud`: enough points, finite, and not all coincident.
fn check_cloud<const D: usize>(
    x: &[ManifoldPoint<D>],
    min_points: usize,
) -> Result<(), MonodromyError> {
    if x.len() < min_points {
        return Err(MonodromyError::TooFewPoints {
            actual: x.len(),
            min: min_points,
        });
    }
    if x.iter().any(|p| p.coords.iter().any(|c| !c.is_finite())) {
        return Err(MonodromyError::NonFinite);
    }
    let spread = (0..D)
        .map(|i| {
            let (lo, hi) = x
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
                    (lo.min(p.coords[i]), hi.max(p.coords[i]))
                });
            hi - lo
        })
        .fold(0.0, f64::max);
    if spread <= 0.0 {
        return Err(MonodromyError::ZeroDiameter);
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Injectivity
// ═══════════════════════════════════════════════════════════════════════════════

/// The smallest two-point ratio and the pair of sample points attaining it.
#[derive(Debug, Clone, Copy)]
pub struct Witness<const D: usize> {
    /// `λ(F; X)`.
    pub lambda: f64,
    /// One point of the minimising pair.
    pub p: ManifoldPoint<D>,
    /// The other point of the minimising pair.
    pub q: ManifoldPoint<D>,
}

/// Every distinct-pair ratio, with the minimising pair. Pairs of coincident
/// sample points are skipped: `λ` is defined over `p ≠ q`.
fn pair_ratios<const D: usize, F>(
    f: &F,
    x: &[ManifoldPoint<D>],
) -> Result<(Vec<f64>, Witness<D>), MonodromyError>
where
    F: Fn(&[f64; D]) -> [f64; D],
{
    let n = x.len();
    if n > MAX_N {
        return Err(MonodromyError::TooManyPoints {
            actual: n,
            max: MAX_N,
        });
    }
    if n < 2 {
        return Err(MonodromyError::TooFewPoints { actual: n, min: 2 });
    }
    let y: Vec<ManifoldPoint<D>> = x.iter().map(|p| ManifoldPoint::new(f(&p.coords))).collect();

    let mut ratios = Vec::with_capacity(n * (n - 1) / 2);
    let mut best = (f64::INFINITY, 0, 0);
    for i in 0..n {
        for j in i + 1..n {
            let domain = x[i].distance(&x[j]);
            if domain == 0.0 {
                continue;
            }
            let r = y[i].distance(&y[j]) / domain;
            if r < best.0 {
                best = (r, i, j);
            }
            ratios.push(r);
        }
    }
    if ratios.is_empty() {
        return Err(MonodromyError::ZeroDiameter);
    }
    let witness = Witness {
        lambda: best.0,
        p: x[best.1],
        q: x[best.2],
    };
    Ok((ratios, witness))
}

/// `λ(F; X) = min_{p ≠ q} |F(p) − F(q)| / |p − q|`, with the pair attaining it.
///
/// Zero exactly when two distinct sample points share an image. The pair is the
/// evidence: an exhibited pair is a proof of non-injectivity that needs no
/// threshold.
///
/// Ported from monodromy/injectivity.py `lower_ratio_witness`.
pub fn lower_ratio_witness<const D: usize, F>(
    f: F,
    x: &[ManifoldPoint<D>],
) -> Result<Witness<D>, MonodromyError>
where
    F: Fn(&[f64; D]) -> [f64; D],
{
    pair_ratios(&f, x).map(|(_, witness)| witness)
}

/// `ρ_free(F; X) = λ(F; X) / median_{p ≠ q} |F(p) − F(q)| / |p − q|`.
///
/// Both terms are distances between sampled points; nothing is differentiated.
/// Returns infinity when the median ratio is zero. Only comparable with
/// [`FREE_LEVEL`] at `n ≥ FREE_MIN_N`.
///
/// Ported from monodromy/injectivity.py `free_ratio`.
pub fn free_ratio<const D: usize, F>(f: F, x: &[ManifoldPoint<D>]) -> Result<f64, MonodromyError>
where
    F: Fn(&[f64; D]) -> [f64; D],
{
    let (mut ratios, witness) = pair_ratios(&f, x)?;
    let med = median(&mut ratios);
    Ok(if med > 0.0 {
        witness.lambda / med
    } else {
        f64::INFINITY
    })
}

/// The log-log fit of `λ̃(n)` against `n`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RatioScaling {
    /// `b` in `ln λ̃(n) = a + b ln n`. Near zero once `λ` has found a positive
    /// infimum; steeply negative while it is still falling toward a collision.
    pub slope: f64,
    /// Standard error of `b`, infinite with fewer than three sizes.
    pub se_slope: f64,
    /// `a` in the same fit.
    pub intercept: f64,
}

/// How `λ` falls as the sample grows: the median `λ` over `repeats` draws at
/// each size, fitted in log-log.
///
/// `sampler(n, seed)` draws `n` points; repeat `r` uses seed `seed·1000 + r`.
/// The domain is passed in rather than inferred, because it is modelling.
///
/// Ported from monodromy/injectivity.py `ratio_scaling`.
pub fn ratio_scaling<const D: usize, F, S>(
    f: F,
    mut sampler: S,
    sizes: &[usize],
    repeats: usize,
    seed: u64,
) -> Result<RatioScaling, MonodromyError>
where
    F: Fn(&[f64; D]) -> [f64; D],
    S: FnMut(usize, u64) -> Vec<ManifoldPoint<D>>,
{
    scaling_fit(&f, &mut sampler, sizes, repeats, seed)
}

fn scaling_fit<const D: usize, F, S>(
    f: &F,
    sampler: &mut S,
    sizes: &[usize],
    repeats: usize,
    seed: u64,
) -> Result<RatioScaling, MonodromyError>
where
    F: Fn(&[f64; D]) -> [f64; D],
    S: FnMut(usize, u64) -> Vec<ManifoldPoint<D>>,
{
    if sizes.is_empty() || repeats == 0 {
        return Err(MonodromyError::InvalidParameter);
    }
    let mut xs = Vec::with_capacity(sizes.len());
    let mut ys = Vec::with_capacity(sizes.len());
    for &n in sizes {
        let mut lambdas = Vec::with_capacity(repeats);
        for r in 0..repeats {
            let draw = sampler(n, seed.wrapping_mul(1000).wrapping_add(r as u64));
            lambdas.push(pair_ratios(f, &draw)?.1.lambda);
        }
        xs.push(log(n as f64));
        ys.push(log(median(&mut lambdas)));
    }
    let (slope, intercept, rss, sxx) = ols(&xs, &ys);
    let se_slope = if xs.len() > 2 {
        sqrt(rss / (xs.len() - 2) as f64 / sxx)
    } else {
        f64::INFINITY
    };
    Ok(RatioScaling {
        slope,
        se_slope,
        intercept,
    })
}

/// What the collision certificate concluded. There is deliberately no
/// `Injective` variant: a bounded sample cannot establish it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionVerdict {
    /// A pair of distinct points with (near-)coincident images was exhibited.
    CollisionExhibited,
    /// None was found at this sampling. Not evidence of injectivity.
    NoCollisionAtThisSampling,
    /// The size range was too short to read; see [`MIN_SIZES`], [`MIN_SPAN`].
    Undecided,
}

/// The Jacobian-free collision certificate and everything it was read from.
#[derive(Debug, Clone, Copy)]
pub struct CollisionCertificate<const D: usize> {
    /// The verdict.
    pub verdict: CollisionVerdict,
    /// `λ` on one draw at the largest size, and the pair attaining it.
    pub witness: Witness<D>,
    /// `|F(p) − F(q)|` for the witness pair.
    pub image_gap: f64,
    /// `|p − q|` for the witness pair. Small for a critical point, bounded
    /// below for a genuine fibre collision.
    pub domain_gap: f64,
    /// `ρ_free` on the same draw.
    pub free_ratio: f64,
    /// Whether the largest size reaches [`FREE_MIN_N`], so that `ρ_free`
    /// decides rather than the slope.
    pub free_readable: bool,
    /// The log-log fit of `λ̃(n)`.
    pub scaling: RatioScaling,
    /// `slope < DECAY` and `slope + SLOPE_SIGMA · se < 0`.
    pub slope_steep: bool,
    /// At least [`MIN_SIZES`] sizes spanning at least [`MIN_SPAN`].
    pub size_range_adequate: bool,
    /// Whether `ρ_free` and the slope point the same way. Surfaced, not
    /// averaged.
    pub signals_agree: bool,
}

impl<const D: usize> CollisionCertificate<D> {
    /// Whether a colliding pair was exhibited.
    pub fn collision_found(&self) -> bool {
        self.verdict == CollisionVerdict::CollisionExhibited
    }
}

/// Report whether a colliding pair was exhibited, without a Jacobian.
///
/// `sampler(n, seed)` draws `n` points. The slope is fitted over `sizes` with
/// `repeats` draws each; the witness and `ρ_free` come from one draw of
/// `max(sizes)` points at `seed`. The source's default call is
/// `sizes = [100, 200, 400, 800]`, `repeats = 5`, `seed = 0`.
///
/// Ported from monodromy/injectivity.py `collision_certificate`, the branch
/// taken when no `jacobian` is supplied (`basis = "free"`).
pub fn collision_certificate<const D: usize, F, S>(
    f: F,
    mut sampler: S,
    sizes: &[usize],
    repeats: usize,
    seed: u64,
) -> Result<CollisionCertificate<D>, MonodromyError>
where
    F: Fn(&[f64; D]) -> [f64; D],
    S: FnMut(usize, u64) -> Vec<ManifoldPoint<D>>,
{
    let scaling = scaling_fit(&f, &mut sampler, sizes, repeats, seed)?;

    let n_used = sizes.iter().copied().max().unwrap_or(0);
    let lo = sizes.iter().copied().min().unwrap_or(0);
    let draw = sampler(n_used, seed);
    let (mut ratios, witness) = pair_ratios(&f, &draw)?;
    let med = median(&mut ratios);
    let free = if med > 0.0 {
        witness.lambda / med
    } else {
        f64::INFINITY
    };

    let image_gap = ManifoldPoint::new(f(&witness.p.coords))
        .distance(&ManifoldPoint::new(f(&witness.q.coords)));
    let domain_gap = witness.p.distance(&witness.q);

    let slope_steep = scaling.slope < DECAY && scaling.slope + SLOPE_SIGMA * scaling.se_slope < 0.0;
    let size_range_adequate =
        sizes.len() >= MIN_SIZES && lo > 0 && n_used as f64 / lo as f64 >= MIN_SPAN;
    let free_readable = n_used >= FREE_MIN_N;
    let found = if free_readable {
        free < FREE_LEVEL
    } else {
        slope_steep
    };

    let verdict = if !size_range_adequate {
        CollisionVerdict::Undecided
    } else if found {
        CollisionVerdict::CollisionExhibited
    } else {
        CollisionVerdict::NoCollisionAtThisSampling
    };

    Ok(CollisionCertificate {
        verdict,
        witness,
        image_gap,
        domain_gap,
        free_ratio: free,
        free_readable,
        scaling,
        slope_steep,
        size_range_adequate,
        signals_agree: (free < FREE_LEVEL) == slope_steep,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Symmetry
// ═══════════════════════════════════════════════════════════════════════════════

/// Rotation by `theta_deg` degrees, row-major.
///
/// Ported from monodromy/discover.py `rotation`.
pub fn rotation(theta_deg: f64) -> [[f64; 2]; 2] {
    let t = theta_deg.to_radians();
    let (c, s) = (cos(t), sin(t));
    [[c, -s], [s, c]]
}

/// Reflection in the line through the origin at angle `theta_deg / 2`,
/// row-major.
///
/// Ported from monodromy/discover.py `reflection`.
pub fn reflection(theta_deg: f64) -> [[f64; 2]; 2] {
    let t = theta_deg.to_radians();
    let (c, s) = (cos(t), sin(t));
    [[c, s], [s, -c]]
}

/// Apply the linear map `g` about the cloud's centroid, not about the origin.
///
/// Rotation about the origin is not rotation about the shape; without the
/// centring the source measured a hexagon translated by 5 recovering order 1.
///
/// Ported from monodromy/functional.py `apply_about_centroid`.
pub fn apply_about_centroid<const D: usize>(
    x: &[ManifoldPoint<D>],
    g: &[[f64; D]; D],
) -> Result<Vec<ManifoldPoint<D>>, MonodromyError> {
    check_cloud(x, 1)?;
    let mut c = [0.0; D];
    for p in x {
        for (ci, v) in c.iter_mut().zip(&p.coords) {
            *ci += v;
        }
    }
    for v in &mut c {
        *v /= x.len() as f64;
    }
    Ok(x.iter()
        .map(|p| {
            let mut out = c;
            for (i, row) in g.iter().enumerate() {
                for (j, gij) in row.iter().enumerate() {
                    out[i] += gij * (p.coords[j] - c[j]);
                }
            }
            ManifoldPoint::new(out)
        })
        .collect())
}

/// Mean distance from each point of `a` to its nearest point of `b`.
fn mean_nearest<const D: usize>(a: &[ManifoldPoint<D>], b: &[ManifoldPoint<D>]) -> f64 {
    a.iter()
        .map(|p| {
            b.iter()
                .map(|q| p.distance(q))
                .fold(f64::INFINITY, f64::min)
        })
        .sum::<f64>()
        / a.len() as f64
}

/// `D_ch(X, gX) = mean_{p∈X} d(p, gX) + mean_{q∈gX} d(q, X)`.
///
/// Zero exactly when `gX = X` as sets; both directions are required, since one
/// direction alone vanishes on containment. A mean rather than a supremum, so
/// one displaced point contributes `1/n` of its displacement.
///
/// Ported from monodromy/discover.py `chamfer_defect`.
pub fn chamfer_defect<const D: usize>(
    x: &[ManifoldPoint<D>],
    gx: &[ManifoldPoint<D>],
) -> Result<f64, MonodromyError> {
    check_cloud(x, 2)?;
    check_cloud(gx, 2)?;
    Ok(mean_nearest(x, gx) + mean_nearest(gx, x))
}

/// `D_PH(g; X) = max_{k∈{0,1}} d_B(PH_k(X), PH_k(X ∪ gX))` over finite bars.
///
/// Zero exactly when `gX = X` as sets. The maximum over degrees matters: a
/// collinear cloud has an empty `H1` diagram, so degree 1 alone returns 0 for
/// every `g`. Built on [`persistent_homology`] under
/// [`PersistenceConfig::h1_dense`], so the union is refused above 128 points.
///
/// Ported from monodromy/functional.py `combined_defect`.
pub fn persistence_defect<const D: usize>(
    x: &[ManifoldPoint<D>],
    gx: &[ManifoldPoint<D>],
) -> Result<f64, MonodromyError> {
    check_cloud(x, 2)?;
    check_cloud(gx, 2)?;
    if x.len() != gx.len() {
        return Err(MonodromyError::ShapeMismatch);
    }
    let config = PersistenceConfig::h1_dense();
    let base = persistent_homology(x, config)?;
    let union: Vec<ManifoldPoint<D>> = x.iter().chain(gx).copied().collect();
    let union = persistent_homology(&union, config)?;
    Ok((0..=1)
        .map(|k| bottleneck_distance(&base, &union, k))
        .fold(0.0, f64::max))
}

/// `E(g; F, X) = max_{x∈X} ‖F(gx) − g F(x)‖_∞`: zero exactly when `g` commutes
/// with the map on the sample.
///
/// A statement about the dynamics rather than the cloud. The Lyapunov spectrum
/// cannot stand in for it: the spectrum is a conjugacy invariant, so it is
/// unchanged under every invertible `g`, symmetry or not.
///
/// Ported from monodromy/dynamics.py `equivariance_defect`.
pub fn equivariance_defect<const D: usize, F>(
    step: F,
    g: &[[f64; D]; D],
    x: &[ManifoldPoint<D>],
) -> Result<f64, MonodromyError>
where
    F: Fn(&[f64; D]) -> [f64; D],
{
    if x.is_empty() {
        return Err(MonodromyError::TooFewPoints { actual: 0, min: 1 });
    }
    let apply = |v: &[f64; D]| {
        let mut out = [0.0; D];
        for (o, row) in out.iter_mut().zip(g) {
            *o = row.iter().zip(v).map(|(a, b)| a * b).sum();
        }
        out
    };
    let mut worst: f64 = 0.0;
    for p in x {
        let lhs = step(&apply(&p.coords));
        let rhs = apply(&step(&p.coords));
        for (a, b) in lhs.iter().zip(&rhs) {
            worst = worst.max((a - b).abs());
        }
    }
    Ok(worst)
}

/// What the group recovery found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupReport {
    /// Rotation order. 1 means nothing was found, which is not evidence that
    /// the cloud has no symmetry.
    pub order: usize,
    /// Harmonic share `s(order)` of the rotation profile; 0 when nothing was
    /// found.
    pub power_fraction: f64,
    /// Whether reflections were found as well, making the group `D_n`.
    pub dihedral: bool,
    /// Defect at `360°/order` over the mean nearest-neighbour spacing.
    pub grain_ratio: f64,
    /// Reflection depth over rotation depth. `None` from [`recover_cyclic`],
    /// which does not test reflections.
    pub depth_ratio: Option<f64>,
}

/// `np.arange(0, 360, step)`, refused when it cannot carry a spectrum.
fn angle_grid(step_deg: f64) -> Result<Vec<f64>, MonodromyError> {
    if !step_deg.is_finite() || step_deg <= 0.0 {
        return Err(MonodromyError::InvalidParameter);
    }
    let count = ceil(360.0 / step_deg) as usize;
    if count < 4 {
        return Err(MonodromyError::InvalidParameter);
    }
    Ok((0..count).map(|k| k as f64 * step_deg).collect())
}

/// The defect as a function of the group parameter.
fn defect_profile(
    x: &[ManifoldPoint<2>],
    grid: &[f64],
    action: fn(f64) -> [[f64; 2]; 2],
) -> Result<Vec<f64>, MonodromyError> {
    grid.iter()
        .map(|&t| chamfer_defect(x, &apply_about_centroid(x, &action(t))?))
        .collect()
}

/// Group order as the largest `n` whose harmonic series `{n, 2n, ...}` carries
/// at least `tau` of the profile's spectral power, scaled by `360 / span`.
fn harmonic_order(profile: &[f64], tau: f64, span_deg: f64) -> (usize, f64) {
    let len = profile.len();
    let mean = profile.iter().sum::<f64>() / len as f64;
    // rfft power: bins 0..=len/2. A direct DFT; the profile is a few hundred
    // samples at most.
    let bins = len / 2 + 1;
    let mut power = vec![0.0; bins];
    for (k, pk) in power.iter_mut().enumerate().skip(1) {
        let (mut re, mut im) = (0.0, 0.0);
        for (t, v) in profile.iter().enumerate() {
            let angle = core::f64::consts::TAU * ((k * t) % len) as f64 / len as f64;
            re += (v - mean) * cos(angle);
            im -= (v - mean) * sin(angle);
        }
        *pk = re * re + im * im;
    }
    let total: f64 = power.iter().sum();
    if total.is_nan() || total <= 0.0 {
        return (1, 0.0);
    }
    let (mut best, mut best_share) = (1, 0.0);
    for n in 2..=bins / 2 {
        let share = (n..bins).step_by(n).map(|k| power[k]).sum::<f64>() / total;
        if share >= tau {
            best = n;
            best_share = share;
        }
    }
    let order = best as f64 * 360.0 / span_deg;
    (
        if order >= 1.0 {
            round(order) as usize
        } else {
            1
        },
        best_share,
    )
}

/// Mean nearest-neighbour spacing: the finest structure the sample resolves.
fn sample_grain(x: &[ManifoldPoint<2>]) -> f64 {
    let n = x.len();
    (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| x[i].distance(&x[j]))
                .fold(f64::INFINITY, f64::min)
        })
        .sum::<f64>()
        / n as f64
}

fn rotation_defect(x: &[ManifoldPoint<2>], theta_deg: f64) -> Result<f64, MonodromyError> {
    chamfer_defect(x, &apply_about_centroid(x, &rotation(theta_deg))?)
}

/// Is the defect at `360°/order` inside the sample's grain? Periodicity of the
/// profile is necessary for a symmetry and not sufficient.
fn symmetry_is_real(x: &[ManifoldPoint<2>], order: usize) -> Result<(bool, f64), MonodromyError> {
    if order <= 1 {
        return Ok((true, 0.0));
    }
    let d = rotation_defect(x, 360.0 / order as f64)?;
    let grain = sample_grain(x);
    let ratio = if grain > 0.0 {
        d / grain
    } else {
        f64::INFINITY
    };
    Ok((ratio <= GRAIN_LEVEL, ratio))
}

/// Promote the order to a multiple whose off-grid defect sits at the floor.
fn promote_aliased_order(
    x: &[ManifoldPoint<2>],
    order: usize,
    profile: &[f64],
) -> Result<usize, MonodromyError> {
    let med = median(&mut profile.to_vec());
    if med.is_nan() || med <= 0.0 {
        return Ok(order);
    }
    let mut best = order;
    for m in 2..=6 {
        let candidate = order * m;
        if rotation_defect(x, 360.0 / candidate as f64)? <= PROMOTE_DEPTH * med {
            best = candidate;
        }
    }
    Ok(best)
}

/// Smallest defect near the best (masked) grid angle, refined off the grid by a
/// bounded golden-section search over one grid step either side, to 1e-9°.
/// The source calls scipy's bounded Brent minimiser over the same bracket and
/// tolerance; golden section is that method without its parabolic steps.
fn refined_min(
    x: &[ManifoldPoint<2>],
    action: fn(f64) -> [[f64; 2]; 2],
    grid: &[f64],
    profile: &[f64],
    exclude_identity: bool,
) -> Result<f64, MonodromyError> {
    let keep = |i: &usize| !exclude_identity || (grid[*i] % 360.0).abs() > 1e-8;
    let Some(k) = (0..profile.len())
        .filter(keep)
        .min_by(|&a, &b| profile[a].total_cmp(&profile[b]))
    else {
        return Ok(profile.iter().copied().fold(f64::INFINITY, f64::min));
    };
    let step = grid[1] - grid[0];
    let f = |t: f64| chamfer_defect(x, &apply_about_centroid(x, &action(t))?);

    let inv_phi = (sqrt(5.0) - 1.0) / 2.0;
    let (mut a, mut b) = (grid[k] - step, grid[k] + step);
    let (mut c, mut d) = (b - inv_phi * (b - a), a + inv_phi * (b - a));
    let (mut fc, mut fd) = (f(c)?, f(d)?);
    while b - a > 1e-9 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - inv_phi * (b - a);
            fc = f(c)?;
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + inv_phi * (b - a);
            fd = f(d)?;
        }
    }
    Ok(profile[k].min(fc).min(fd))
}

/// Recover the rotation order `C_n` of a planar cloud. The caller never supplies
/// `n`.
///
/// Chamfer defect profile over `[0°, 360°)` at `step_deg`, read by harmonic
/// share, promoted across grid aliasing, and kept only inside the grain. See
/// the module documentation for the measured envelope: sound, not complete.
///
/// Ported from monodromy/discover.py `recover_cyclic` (default `reader =
/// "harmonic"`, `metric = "chamfer"`).
pub fn recover_cyclic(
    x: &[ManifoldPoint<2>],
    step_deg: f64,
) -> Result<GroupReport, MonodromyError> {
    check_cloud(x, 2)?;
    let grid = angle_grid(step_deg)?;
    let profile = defect_profile(x, &grid, rotation)?;
    let span = grid.len() as f64 * step_deg;
    let (mut order, mut power_fraction) = harmonic_order(&profile, HARMONIC_TAU, span);
    if order > 1 {
        order = promote_aliased_order(x, order, &profile)?;
    }
    let (real, grain_ratio) = symmetry_is_real(x, order)?;
    if !real {
        order = 1;
        power_fraction = 0.0;
    }
    Ok(GroupReport {
        order,
        power_fraction,
        dihedral: false,
        grain_ratio,
        depth_ratio: None,
    })
}

/// Recover `C_n` or `D_n`: rotations as in [`recover_cyclic`], plus a
/// reflection profile compared by depth below its own median.
///
/// A ratio of minima divides by a quantity meant to vanish; depth does not.
/// Both minima are refined off the grid, since a coarse grid can hold a shape's
/// rotations while missing its mirrors. Unlike [`recover_cyclic`], the source
/// does not apply aliasing promotion here.
///
/// Ported from monodromy/discover.py `recover_dihedral`.
pub fn recover_dihedral(
    x: &[ManifoldPoint<2>],
    step_deg: f64,
) -> Result<GroupReport, MonodromyError> {
    check_cloud(x, 2)?;
    let grid = angle_grid(step_deg)?;
    let rot = defect_profile(x, &grid, rotation)?;
    let refl = defect_profile(x, &grid, reflection)?;
    let span = grid.len() as f64 * step_deg;

    let (mut order, mut power_fraction) = harmonic_order(&rot, HARMONIC_TAU, span);
    let (real, grain_ratio) = symmetry_is_real(x, order)?;
    if !real {
        order = 1;
        power_fraction = 0.0;
    }

    let rot_min = refined_min(x, rotation, &grid, &rot, true)?;
    let ref_min = refined_min(x, reflection, &grid, &refl, false)?;
    let mut non_identity: Vec<f64> = grid
        .iter()
        .zip(&rot)
        .filter(|(g, _)| (*g % 360.0).abs() > 1e-8)
        .map(|(_, v)| *v)
        .collect();
    let rot_mid = if non_identity.is_empty() {
        median(&mut rot.clone())
    } else {
        median(&mut non_identity)
    };
    let ref_mid = median(&mut refl.clone());
    let depth = |mid: f64, min: f64| if mid > 0.0 { (mid - min) / mid } else { 0.0 };
    let (rot_depth, ref_depth) = (depth(rot_mid, rot_min), depth(ref_mid, ref_min));
    let depth_ratio = if rot_depth > 0.0 {
        ref_depth / rot_depth
    } else {
        0.0
    };

    Ok(GroupReport {
        order,
        power_fraction,
        dihedral: order > 1 && depth_ratio >= REFLECTION_DEPTH,
        grain_ratio,
        depth_ratio: Some(depth_ratio),
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Dimension
// ═══════════════════════════════════════════════════════════════════════════════

/// `T_α(X) = Σ (death − birth)^α` over the finite `H0` bars of the Rips
/// filtration, which are the edges of the minimum spanning tree.
///
/// Computed by [`persistent_homology`] at degree 0, with the point and simplex
/// caps set to exactly what an `n`-point `H0` filtration needs.
///
/// Ported from monodromy/phdim.py `total_persistence` (`dim = 0`).
pub fn alpha_total_persistence<const D: usize>(
    x: &[ManifoldPoint<D>],
    alpha: f64,
) -> Result<f64, MonodromyError> {
    let n = x.len();
    let config = PersistenceConfig {
        max_homology_dim: 0,
        max_points: n,
        max_simplices: n + n * n.saturating_sub(1) / 2,
        max_radius: f64::INFINITY,
        complex_kind: ComplexKind::VietorisRips,
    };
    let diagram = persistent_homology(x, config)?;
    Ok(diagram
        .pairs
        .iter()
        .filter(|pair| pair.dimension == 0)
        .filter_map(|pair| pair.death.map(|death| pow(death - pair.birth, alpha)))
        .sum())
}

/// Schweinhart's persistent-homology dimension with its fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhDimension {
    /// `d̂ = α / (1 − s)`; infinite when `s ≥ 1`.
    pub dimension: f64,
    /// `s` in `ln T_α = c + s ln n`.
    pub slope: f64,
    /// Standard error of `s`.
    pub se_slope: f64,
    /// Image of `s ± 1.96·se` under `s ↦ α/(1 − s)`, ordered.
    pub ci95: (f64, f64),
    /// Draws that entered the fit.
    pub n_obs: usize,
    /// Draws whose delivered size differed from the requested one.
    pub n_clamped: usize,
    /// Decades of `n` the fit spans. Below one, the interval is wide.
    pub decades: f64,
}

/// Estimate the PH dimension of the measure `sampler` draws from.
///
/// `sampler(n, seed)` returns a cloud; draw `k` uses seed `seed·1000 + k`, so
/// calls with different `alpha` see the same clouds. Draws are fresh rather than
/// subsets of one cloud, which saturate as `n` approaches its size, and the
/// regression uses the size each draw actually delivered. Requires
/// `0 < alpha < d`. The source's default call is `alpha = 1`,
/// `sizes = [100, 200, 400, 800]`, `repeats = 3`, `seed = 0`.
///
/// Ported from monodromy/phdim.py `ph_dimension` (`dim = 0`,
/// `return_fit = True`).
pub fn ph_dimension<const D: usize, S>(
    mut sampler: S,
    alpha: f64,
    sizes: &[usize],
    repeats: usize,
    seed: u64,
) -> Result<PhDimension, MonodromyError>
where
    S: FnMut(usize, u64) -> Vec<ManifoldPoint<D>>,
{
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(MonodromyError::InvalidParameter);
    }
    let (mut xs, mut ys, mut delivered) = (Vec::new(), Vec::new(), Vec::new());
    let mut n_clamped = 0;
    let mut k: u64 = 0;
    for &n in sizes {
        for _ in 0..repeats {
            let draw = sampler(n, seed.wrapping_mul(1000).wrapping_add(k));
            k += 1;
            let got = draw.len();
            if got != n {
                n_clamped += 1;
            }
            let tp = if got >= 2 {
                alpha_total_persistence(&draw, alpha)?
            } else {
                0.0
            };
            if tp > 0.0 {
                xs.push(log(got as f64));
                ys.push(log(tp));
                delivered.push(got);
            }
        }
    }
    if xs.len() < 3 {
        return Err(MonodromyError::NotEnoughSamples);
    }
    if delivered.iter().all(|&g| g == delivered[0]) {
        return Err(MonodromyError::NoScaling);
    }

    let (slope, _, rss, sxx) = ols(&xs, &ys);
    let dof = (xs.len() - 2).max(1) as f64;
    let se_slope = if sxx > 0.0 {
        sqrt(rss / dof / sxx)
    } else {
        f64::INFINITY
    };
    let invert = |s: f64| {
        if s < 1.0 {
            alpha / (1.0 - s)
        } else {
            f64::INFINITY
        }
    };
    let (lo, hi) = (
        invert(slope - 1.96 * se_slope),
        invert(slope + 1.96 * se_slope),
    );
    let (x_min, x_max) = xs
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });

    Ok(PhDimension {
        dimension: invert(slope),
        slope,
        se_slope,
        ci95: (lo.min(hi), lo.max(hi)),
        n_obs: xs.len(),
        n_clamped,
        decades: (x_max - x_min) / core::f64::consts::LN_10,
    })
}

/// Mandelbrot's predicate `d_H > d_top`, decided on an interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FractalVerdict {
    /// The whole 95 % interval lies above the topological dimension.
    Fractal,
    /// The whole interval lies at or below it.
    NotFractal,
    /// The interval straddles it.
    Undecided,
}

/// Decide `d_H > d_top` from [`ph_dimension`], rounding nothing: a straddling
/// interval is [`FractalVerdict::Undecided`].
///
/// Ported from monodromy/phdim.py `is_fractal`.
pub fn is_fractal<const D: usize, S>(
    sampler: S,
    topological_dim: f64,
    alpha: f64,
    sizes: &[usize],
    repeats: usize,
    seed: u64,
) -> Result<(FractalVerdict, PhDimension), MonodromyError>
where
    S: FnMut(usize, u64) -> Vec<ManifoldPoint<D>>,
{
    let fit = ph_dimension(sampler, alpha, sizes, repeats, seed)?;
    let verdict = if fit.ci95.0 > topological_dim {
        FractalVerdict::Fractal
    } else if fit.ci95.1 <= topological_dim {
        FractalVerdict::NotFractal
    } else {
        FractalVerdict::Undecided
    };
    Ok((verdict, fit))
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Sensitivity
// ═══════════════════════════════════════════════════════════════════════════════

fn dot<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Remove from `v` its components along the first `k` columns of `q`, twice
/// (one pass of Gram–Schmidt loses orthogonality on ill-conditioned frames).
fn orthogonalise<const D: usize>(v: &mut [f64; D], q: &[[f64; D]; D], k: usize) {
    for _ in 0..2 {
        for col in &q[..k] {
            let r = dot(col, v);
            for (vi, ci) in v.iter_mut().zip(col) {
                *vi -= r * ci;
            }
        }
    }
}

/// One re-orthonormalised step: `J Q = Q' R`, `R_ii ≥ 0`, accumulating
/// `ln max(R_ii, R_FLOOR)`. `q[k]` is the k-th column of the frame.
fn qr_step<const D: usize>(j: &[[f64; D]; D], q: &mut [[f64; D]; D], acc: &mut [f64; D]) {
    let mut next = [[0.0; D]; D];
    for k in 0..D {
        let mut v = [0.0; D];
        for (vi, row) in v.iter_mut().zip(j) {
            *vi = dot(row, &q[k]);
        }
        orthogonalise(&mut v, &next, k);
        let r = sqrt(dot(&v, &v));
        if r > R_FLOOR {
            for vi in &mut v {
                *vi /= r;
            }
        } else {
            // A direction died this step. Complete the frame with whichever
            // coordinate axis survives orthogonalisation best, as a Householder
            // factorisation would, so later steps still see D directions.
            let mut best = ([0.0; D], -1.0);
            for axis in 0..D {
                let mut e = [0.0; D];
                e[axis] = 1.0;
                orthogonalise(&mut e, &next, k);
                let norm = sqrt(dot(&e, &e));
                if norm > best.1 {
                    best = (e, norm);
                }
            }
            v = best.0;
            for vi in &mut v {
                *vi /= best.1;
            }
        }
        next[k] = v;
        acc[k] += log(r.max(R_FLOOR));
    }
    *q = next;
}

/// Lyapunov exponents of a sequence of Jacobians, applied in the order given,
/// per unit time. Row-major: `J[i][j] = ∂F_i/∂x_j`.
///
/// Returned in the QR column order, which Benettin's iteration drives to
/// descending order along a generic orbit; not re-sorted, as in the source.
/// Streams the sequence, so its length costs no memory.
///
/// Ported from monodromy/_cocycle_vendored.py `lyapunov_spectrum`.
pub fn lyapunov_spectrum<const D: usize, I>(
    jacobians: I,
    dt: f64,
) -> Result<[f64; D], MonodromyError>
where
    I: IntoIterator<Item = [[f64; D]; D]>,
{
    if !dt.is_finite() || dt <= 0.0 {
        return Err(MonodromyError::InvalidParameter);
    }
    let mut q = [[0.0; D]; D];
    for (k, col) in q.iter_mut().enumerate() {
        col[k] = 1.0;
    }
    let mut acc = [0.0; D];
    let mut steps = 0usize;
    for j in jacobians {
        qr_step(&j, &mut q, &mut acc);
        steps += 1;
    }
    if steps == 0 {
        return Err(MonodromyError::TooFewPoints { actual: 0, min: 1 });
    }
    for a in &mut acc {
        *a /= steps as f64 * dt;
    }
    Ok(acc)
}

/// Lyapunov spectrum of a map: `burn` steps discarded, then the Jacobians of
/// `n` consecutive orbit points fed to [`lyapunov_spectrum`] with `dt = 1`.
///
/// Ported from monodromy/dynamics.py `map_spectrum` (with `jacobians_along`).
pub fn map_spectrum<const D: usize, St, Ja>(
    step: St,
    jac: Ja,
    x0: [f64; D],
    n: usize,
    burn: usize,
) -> Result<[f64; D], MonodromyError>
where
    St: Fn(&[f64; D]) -> [f64; D],
    Ja: Fn(&[f64; D]) -> [[f64; D]; D],
{
    let mut x = x0;
    for _ in 0..burn {
        x = step(&x);
    }
    lyapunov_spectrum(
        (0..n).map(|_| {
            let j = jac(&x);
            x = step(&x);
            j
        }),
        1.0,
    )
}

/// Kaplan–Yorke dimension `D_KY = j + Σ_{i≤j} λ_i / |λ_{j+1}|`.
///
/// 0 when every exponent is negative (collapse to a fixed point); `D` when
/// every partial sum stays non-negative (nothing to interpolate). Refuses a
/// non-finite spectrum rather than returning 0, which is a legitimate answer.
///
/// Ported from monodromy/_attractor_vendored.py `kaplan_yorke_dimension`.
pub fn kaplan_yorke_dimension<const D: usize>(mut lam: [f64; D]) -> Result<f64, MonodromyError> {
    if D == 0 {
        return Err(MonodromyError::TooFewPoints { actual: 0, min: 1 });
    }
    if lam.iter().any(|v| !v.is_finite()) {
        return Err(MonodromyError::NonFinite);
    }
    lam.sort_unstable_by(|a, b| b.total_cmp(a));
    if lam[0] < 0.0 {
        return Ok(0.0);
    }
    let mut cumulative = [0.0; D];
    let mut running = 0.0;
    for (c, l) in cumulative.iter_mut().zip(&lam) {
        running += l;
        *c = running;
    }
    let j = cumulative.iter().take_while(|&&c| c >= 0.0).count();
    if j >= D {
        return Ok(D as f64);
    }
    let denom = lam[j].abs();
    if denom < 1e-300 {
        return Ok(j as f64);
    }
    Ok(if j > 0 {
        j as f64 + cumulative[j - 1] / denom
    } else {
        0.0
    })
}
