//! Rounding certificates for top-k, argmin and threshold decisions.
//!
//! A score evaluated in floating point is not the value its formula defines. When
//! two candidates lie closer together than the rounding of the kernel that scored
//! them, the arithmetic, not the data, decided which one was returned. This module
//! either proves that a top-k set, an argmin or a threshold decision is the one
//! exact arithmetic returns on the stored inputs, or refuses and names the indices
//! it cannot separate.
//!
//! Ported from `separatrix` (Teerth Sharma, github.com/teerthsharma/separatrix):
//! `separatrix/enclose.py` (unit roundoff, γ, η, preconditions, both kernels'
//! radii), `separatrix/decide.py` (the rule), `separatrix/api.py` (argmin and the
//! threshold trit), `separatrix/exact.py` (the frontier set) and
//! `separatrix/verdict.py` (the frontier and the refusal catalogue).
//!
//! # Error model
//!
//! For a binary format with unit roundoff `u` (2⁻¹¹ binary16, 2⁻²⁴ binary32,
//! 2⁻⁵³ binary64), every operation whose result is normal satisfies
//!
//! ```text
//! fl(a ∘ b) = (a ∘ b)(1 + δ),    |δ| ≤ u,
//! ```
//!
//! and a product of at most `n` such factors lies in `[1 − γₙ, 1 + γₙ]`, where
//! (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Lemma 3.1)
//!
//! ```text
//! γₙ = n·u / (1 − n·u),    evaluated in binary64 and rounded upward.
//! ```
//!
//! A product whose result is subnormal carries an absolute error of at most half
//! the smallest subnormal `σ`, and an addition whose exact result is subnormal is
//! exact, so every radius carries the unconditional term
//!
//! ```text
//! η_d = 4 (d + 2) σ.
//! ```
//!
//! # Bounds
//!
//! Let `x, q ∈ Tᵈ` be the stored vectors and `s` the exact real value of the named
//! formula on them. The Gram identity `D = fl((‖q‖² + ‖x‖²) − 2⟨x, q⟩)` routes
//! every summand through at most `d + 2` roundings, so by Higham's Theorem 3.1
//! (the absolute value sits inside the sum, which is what survives cancellation)
//!
//! ```text
//! |D − s| ≤ γ_{d+2} (‖x‖² + ‖q‖² + 2 Σₗ |xₗ||qₗ|)      (tight)
//!         ≤ γ_{d+2} (‖x‖ + ‖q‖)²                      (cheap, by Cauchy–Schwarz)
//! ```
//!
//! The direct sum `D = fl(Σₗ fl(qₗ − xₗ)²)` has no cancellation, so its bound is
//! relative. The rounded difference enters squared, which is two roundings; the
//! product is one more and the sum at most `d − 1`, so
//!
//! ```text
//! |D − s| ≤ γ_{d+2} · s ≤ γ_{d+2} / (1 − γ_{d+2}) · D.
//! ```
//!
//! The source uses `γ_{d+1}` here, which counts the rounded difference once; the
//! test `direct_radius_counts_the_rounded_difference_twice` exhibits `γ_{d+1}`
//! radii escaped by exact arithmetic at `d = 1` and `d = 2`, and none escaping
//! this one.
//!
//! Norms are evaluated in binary64. The radius's own rounding (a binary64 reduction
//! of length `d`, a square root, a sum, a square and a scale) is absorbed by the
//! final inflation, derived in the source:
//!
//! ```text
//! R ← next_up( R · (1 + γ_{d+2}(binary64) + 8 u₆₄) + η_d ).
//! ```
//!
//! # Certificate
//!
//! Given `D` and `R` with `|Dᵢ − sᵢ| ≤ Rᵢ` for every `i`, let `T` be the `k`
//! smallest entries of `D`. `T` is certified when
//!
//! ```text
//! max_{i ∈ T} (Dᵢ + Rᵢ)  <  min_{j ∉ T} (Dⱼ − Rⱼ),        (strict)
//! ```
//!
//! and then `T` is the top-k set of every vector in the box `∏ᵢ [Dᵢ − Rᵢ, Dᵢ + Rᵢ]`,
//! the exact scores among them. Any other evaluation of the same formula on the same
//! stored inputs whose error the bound covers (another reduction order, blocking,
//! fused multiply-add, thread count) therefore returns the same `T`.
//!
//! - The argmin is `k = 1`. The largest-k set and the argmax negate the scores and
//!   reuse the one rule; radii are unsigned and are carried unchanged.
//! - Comparing only the rank-k and rank-(k+1) enclosures is unsound whenever the
//!   radii vary: scores `[0, 1, 2, 10]` with radii `[12, 0, 0, 0]` and `k = 2`
//!   present a disjoint boundary pair, while `(11, 1, 2, 10)` lies in the box and
//!   has top-2 `{1, 2}`. The rule above refuses that input.
//! - Order within `T` is certified only on request, by requiring the `k − 1`
//!   adjacent enclosures to be disjoint; transitivity covers the remaining pairs.
//! - A threshold `t`, in score units, is decided per score: above when
//!   `Dᵢ − Rᵢ > t`, below when `Dᵢ + Rᵢ < t`, undetermined otherwise.
//! - Every endpoint `Dᵢ ± Rᵢ` is rounded outward by one unit in the last place
//!   before it is compared, so no certificate rests on the rounding of an endpoint.
//!   The source compares round-to-nearest endpoints.
//!
//! # What is certified, and what is not
//!
//! Certified: the rounding of the named formula, on the stored inputs, in the
//! declared precision, did not choose this set, this index or this side. Not
//! certified: that the stored inputs are correct, or anything about how they were
//! produced; an embedding from a binary16 forward pass carries error orders of
//! magnitude above the binary32 rounding certified here. Order within a top-k set
//! is not certified unless requested. Scores and radii supplied to the rule from
//! outside [`enclose_scores`] are certified only insofar as the radii bound the
//! scores, which the rule cannot check. A refusal is not a finding: it states that
//! this enclosure does not decide the boundary, not that the exact answer differs.
//!
//! # Refusals
//!
//! - [`Refusal::BoundaryUndetermined`]: the rule fails. Carries the extreme pair
//!   and every index whose interval crosses the separatrix.
//! - [`Refusal::NonFiniteInput`] (precondition P1): a non-finite coordinate,
//!   score, radius or threshold, located. No enclosure is defined over one.
//! - [`Refusal::BoundVacuous`] (P3): `n·u > 1/2`, where the a-priori bound carries
//!   no information, or `γ ≥ 1` in the direct kernel's relative form, where the
//!   radius would be negative and would certify every input.
//! - [`Refusal::RangeUnsafe`] (P2): `(‖q‖ + maxⱼ ‖xⱼ‖)²` exceeds the largest finite
//!   value of the working format. It dominates every intermediate of both kernels
//!   and is checked before any score is evaluated.
//! - [`Refusal::Usage`]: `k` outside `0 < k < n` (which includes empty input),
//!   radii of the wrong length, a negative radius, or `γ₀`. The source raises these
//!   as usage errors outside its refusal catalogue: they describe the call, not the
//!   data.
//!
//! Not ported: the scaled-integer escalation of `exact.py`, which needs integers
//! of up to 2·1074 bits and therefore a dependency this crate does not admit; the
//! 4×4 canary (P4), which tests a black-box BLAS path, whereas this module
//! evaluates its own scores; and the per-row radius collapse, which saves memory
//! only across many queries. The rule accepts a single radius for every score,
//! which is that collapse supplied by the caller.

#![warn(missing_docs)]

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use core::ops::{Add, Mul, Sub};

use libm::{nextafter, sqrt};

// ═══════════════════════════════════════════════════════════════════════════════
// Precision and the constants of the error model
// ═══════════════════════════════════════════════════════════════════════════════

/// A binary floating-point format, identified by its unit roundoff.
///
/// Ported from the `dtype` argument of `separatrix/separatrix/enclose.py`.
/// `F16` has no [`Working`] implementation, since Rust has no stable binary16
/// type; it is reachable through [`unit_roundoff`], [`gamma`] and [`eta`] for
/// scores evaluated elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// IEEE 754 binary16: `u = 2⁻¹¹`.
    F16,
    /// IEEE 754 binary32: `u = 2⁻²⁴`.
    F32,
    /// IEEE 754 binary64: `u = 2⁻⁵³`.
    F64,
}

/// The unit roundoff `u = ε / 2` of a format.
///
/// Ported from `unit_roundoff` in `separatrix/separatrix/enclose.py`.
pub fn unit_roundoff(p: Precision) -> f64 {
    match p {
        Precision::F16 => 1.0 / 2048.0,
        Precision::F32 => f32::EPSILON as f64 / 2.0,
        Precision::F64 => f64::EPSILON / 2.0,
    }
}

fn smallest_subnormal(p: Precision) -> f64 {
    match p {
        Precision::F16 => 1.0 / 16_777_216.0,
        Precision::F32 => f32::from_bits(1) as f64,
        Precision::F64 => f64::from_bits(1),
    }
}

fn largest_finite(p: Precision) -> f64 {
    match p {
        Precision::F16 => 65504.0,
        Precision::F32 => f32::MAX as f64,
        Precision::F64 => f64::MAX,
    }
}

fn up(x: f64) -> f64 {
    nextafter(x, f64::INFINITY)
}

fn down(x: f64) -> f64 {
    nextafter(x, f64::NEG_INFINITY)
}

/// `γₙ = n·u / (1 − n·u)`, evaluated in binary64 and rounded upward.
///
/// Refuses with [`Refusal::BoundVacuous`] when `n·u > 1/2`: past that point the
/// bound carries no information, and past `n·u = 1` it turns negative. `n = 0` is
/// a usage error, as in the source.
///
/// Ported from `gamma` in `separatrix/separatrix/enclose.py`.
pub fn gamma(n: usize, p: Precision) -> Result<f64, Refusal> {
    if n == 0 {
        return Err(Refusal::Usage("gamma needs a positive reduction length"));
    }
    let nu = n as f64 * unit_roundoff(p);
    if nu > 0.5 {
        return Err(Refusal::BoundVacuous { n });
    }
    Ok(up(nu / (1.0 - nu)))
}

/// The unconditional underflow term `η_d = 4 (d + 2) σ`, in score units, where `σ`
/// is the smallest subnormal of the format.
///
/// Ported from `eta` in `separatrix/separatrix/enclose.py`.
pub fn eta(d: usize, p: Precision) -> f64 {
    4.0 * (d as f64 + 2.0) * smallest_subnormal(p)
}

// ═══════════════════════════════════════════════════════════════════════════════
// The enclosure
// ═══════════════════════════════════════════════════════════════════════════════

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// A format this module stores vectors in and evaluates scores in.
///
/// Sealed: the certificate is only as sound as `PRECISION`, and an implementation
/// declaring a unit roundoff its arithmetic does not honour would certify nothing.
pub trait Working:
    sealed::Sealed + Copy + Into<f64> + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
    /// The format whose unit roundoff bounds every operation on `Self`.
    const PRECISION: Precision;
    /// The additive identity, where every reduction starts.
    const ZERO: Self;
}

impl Working for f32 {
    const PRECISION: Precision = Precision::F32;
    const ZERO: Self = 0.0;
}

impl Working for f64 {
    const PRECISION: Precision = Precision::F64;
    const ZERO: Self = 0.0;
}

/// The score formula, and the bound its radius is built from.
///
/// Ported from the `kernel` and `bound` arguments of
/// `separatrix/separatrix/enclose.py`. Every kernel scores squared Euclidean
/// distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// Gram identity `(‖q‖² + ‖x‖²) − 2⟨x, q⟩`, radius `γ_{d+2} (‖x‖ + ‖q‖)²`.
    GramCheap,
    /// Gram identity, radius `γ_{d+2} (‖x‖² + ‖q‖² + 2 Σ |xₗ||qₗ|)`. Never wider
    /// than `GramCheap`, at the cost of one binary64 pass over `|x|·|q|`.
    GramTight,
    /// Direct sum `Σ (qₗ − xₗ)²`, relative radius `γ_{d+2} / (1 − γ_{d+2}) · D`.
    Direct,
}

/// Scores and radii for one query against a corpus, with `|scoresᵢ − sᵢ| ≤ radiiᵢ`
/// for the exact value `sᵢ` of the kernel's formula on the stored inputs.
///
/// Ported from `Enclosure` in `separatrix/separatrix/enclose.py`.
#[derive(Debug, Clone, PartialEq)]
pub struct Enclosure {
    /// The scores as evaluated in the working format, widened losslessly to
    /// binary64.
    pub scores: Vec<f64>,
    /// One radius per score.
    pub radii: Vec<f64>,
}

/// Evaluates `kernel` for `query` against every row of `corpus` in the working
/// format `T`, together with a radius bounding each score's rounding.
///
/// Preconditions run in the source's order, and all of them before any score is
/// evaluated: P1 (every coordinate finite), P3 (`γ_{d+2}` informative), P2 (no
/// intermediate can overflow `T`).
///
/// Ported from `enclose_scores`, `gram_scores`, `direct_scores`, `gram_radii`
/// (per-pair rungs) and `direct_radii` in `separatrix/separatrix/enclose.py`, with
/// the direct kernel's constant corrected from `γ_{d+1}` to `γ_{d+2}` (see the
/// module documentation).
pub fn enclose_scores<T: Working, const D: usize>(
    corpus: &[[T; D]],
    query: &[T; D],
    kernel: Kernel,
) -> Result<Enclosure, Refusal> {
    // P1
    let finite = |v: &T| (*v).into().is_finite();
    if let Some(row) = corpus.iter().position(|x| !x.iter().all(finite)) {
        return Err(Refusal::NonFiniteInput {
            operand: "corpus",
            index: row,
        });
    }
    if let Some(l) = query.iter().position(|v| !finite(v)) {
        return Err(Refusal::NonFiniteInput {
            operand: "query",
            index: l,
        });
    }

    // P3
    let p = T::PRECISION;
    let g = gamma(D + 2, p)?;
    if kernel == Kernel::Direct && g >= 1.0 {
        return Err(Refusal::BoundVacuous { n: D + 2 });
    }

    // P2, on binary64 norms so that the check itself cannot overflow.
    let norm2 = |v: &[T; D]| {
        v.iter().fold(0.0f64, |s, &a| {
            let a: f64 = a.into();
            s + a * a
        })
    };
    let qn2 = norm2(query);
    let qn = sqrt(qn2);
    let xn2: Vec<f64> = corpus.iter().map(norm2).collect();
    let max_xn = sqrt(xn2.iter().fold(0.0f64, |m, &v| m.max(v)));
    let headroom = (qn + max_xn) * (qn + max_xn);
    let limit = largest_finite(p);
    if headroom > limit {
        return Err(Refusal::RangeUnsafe { headroom, limit });
    }

    let push = gamma(D + 2, Precision::F64)? + 8.0 * unit_roundoff(Precision::F64);
    let floor = eta(D, p);
    let qn2_t = query.iter().fold(T::ZERO, |s, &a| s + a * a);

    let mut scores = Vec::with_capacity(corpus.len());
    let mut radii = Vec::with_capacity(corpus.len());
    for (x, &x2) in corpus.iter().zip(&xn2) {
        let (score, radius) = match kernel {
            Kernel::GramCheap | Kernel::GramTight => {
                let xn2_t = x.iter().fold(T::ZERO, |s, &a| s + a * a);
                let dot = x.iter().zip(query).fold(T::ZERO, |s, (&a, &b)| s + a * b);
                let score: f64 = ((qn2_t + xn2_t) - (dot + dot)).into();
                let bound = if kernel == Kernel::GramCheap {
                    let sum = qn + sqrt(x2);
                    sum * sum
                } else {
                    let absdot = x.iter().zip(query).fold(0.0f64, |s, (&a, &b)| {
                        let (a, b): (f64, f64) = (a.into(), b.into());
                        s + a.abs() * b.abs()
                    });
                    qn2 + x2 + 2.0 * absdot
                };
                (score, g * bound)
            }
            Kernel::Direct => {
                let score: f64 = x
                    .iter()
                    .zip(query)
                    .fold(T::ZERO, |s, (&a, &b)| {
                        let diff = b - a;
                        s + diff * diff
                    })
                    .into();
                (score, score.max(0.0) * (g / (1.0 - g)))
            }
        };
        scores.push(score);
        radii.push(up(radius * (1.0 + push) + floor));
    }
    Ok(Enclosure { scores, radii })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Refusals
// ═══════════════════════════════════════════════════════════════════════════════

/// The pair whose enclosures decide a refused boundary, and by how much they fail.
///
/// Intervals are reported in the caller's orientation and rounded outward, exactly
/// as the rule compared them.
///
/// Ported from `Frontier` in `separatrix/separatrix/verdict.py`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frontier {
    /// The member of the returned set with the extreme inner endpoint (largest
    /// upper endpoint for the smallest-k set). In an ordered refusal, the
    /// higher-ranked member of the adjacent pair.
    pub inside: usize,
    /// The non-member with the extreme outer endpoint (smallest lower endpoint for
    /// the smallest-k set). In an ordered refusal, the next-ranked member.
    pub outside: usize,
    /// Lower endpoint of `inside`'s interval.
    pub inside_lo: f64,
    /// Upper endpoint of `inside`'s interval.
    pub inside_hi: f64,
    /// Lower endpoint of `outside`'s interval.
    pub outside_lo: f64,
    /// Upper endpoint of `outside`'s interval.
    pub outside_hi: f64,
    /// Separation of the two scores in the ranking direction.
    pub gap: f64,
    /// The sum of the two radii: how far apart the scores needed to be.
    pub width: f64,
}

impl Frontier {
    /// `gap − width`, in score units. Non-positive for a refused pair, up to the
    /// one-ulp outward rounding the decision itself is made on.
    ///
    /// Ported from `Frontier.deficit` in `separatrix/separatrix/verdict.py`.
    pub fn deficit(&self) -> f64 {
        self.gap - self.width
    }
}

/// Why no certificate was formed.
///
/// Ported from the refusal catalogue in `separatrix/separatrix/verdict.py`.
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    /// `BOUNDARY_UNDETERMINED`: the enclosures at the decision boundary overlap.
    BoundaryUndetermined {
        /// The pair the rule compared.
        frontier: Frontier,
        /// Ascending indices of every interval that crosses the separatrix: the
        /// members whose upper endpoint reaches the smallest outer lower endpoint,
        /// and the non-members whose lower endpoint reaches the largest inner upper
        /// endpoint (`exact.py`'s escalation frontier). Every index whose membership
        /// differs at some point of the box is among them. For an ordered refusal,
        /// the adjacent pair.
        straddling: Vec<usize>,
    },
    /// `NONFINITE_INPUT` (P1): a NaN or an infinity.
    NonFiniteInput {
        /// `"corpus"`, `"query"`, `"scores"`, `"radii"` or `"threshold"`.
        operand: &'static str,
        /// The corpus row, query coordinate, or score index; 0 for the threshold.
        index: usize,
    },
    /// `BOUND_VACUOUS` (P3): the a-priori bound carries no information.
    BoundVacuous {
        /// The reduction length at which `γₙ` became vacuous.
        n: usize,
    },
    /// `RANGE_UNSAFE` (P2): an intermediate of the kernel can overflow.
    RangeUnsafe {
        /// `(‖q‖ + maxⱼ ‖xⱼ‖)²`, evaluated in binary64.
        headroom: f64,
        /// The largest finite value of the working format.
        limit: f64,
    },
    /// A malformed call rather than a property of the data.
    Usage(&'static str),
}

// ═══════════════════════════════════════════════════════════════════════════════
// The rule
// ═══════════════════════════════════════════════════════════════════════════════

/// Validates a bring-your-own enclosure: one radius or one per score, all finite,
/// none negative. Ported from `broadcast_radius` in `decide.py` and the radius
/// check in `api.certified_threshold`, with P1 applied to the scores.
fn check_enclosure(scores: &[f64], radii: &[f64]) -> Result<(), Refusal> {
    if radii.len() != 1 && radii.len() != scores.len() {
        return Err(Refusal::Usage(
            "radii must hold one radius, or one per score",
        ));
    }
    if let Some(i) = scores.iter().position(|v| !v.is_finite()) {
        return Err(Refusal::NonFiniteInput {
            operand: "scores",
            index: i,
        });
    }
    if let Some(i) = radii.iter().position(|v| !v.is_finite()) {
        return Err(Refusal::NonFiniteInput {
            operand: "radii",
            index: i,
        });
    }
    if radii.iter().any(|&r| r < 0.0) {
        return Err(Refusal::Usage("a radius cannot be negative"));
    }
    Ok(())
}

fn radius(radii: &[f64], i: usize) -> f64 {
    radii[if radii.len() == 1 { 0 } else { i }]
}

/// The `k` smallest (or largest) indices of `scores`, in rank order, ties broken
/// by index. Not a certificate.
///
/// Ported from `topk_set` in `separatrix/separatrix/decide.py`.
///
/// ponytail: a full O(n log n) sort; `select_nth_unstable_by` then a sort of the
/// first `k` if a profile ever shows this beside the O(nd) scoring.
pub fn topk_set(scores: &[f64], k: usize, largest: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        let o = scores[a].total_cmp(&scores[b]);
        if largest { o.reverse() } else { o }.then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

/// Certifies the top-k set of `scores` against the enclosure `radii`, or refuses
/// and names the boundary.
///
/// `radii` holds one radius per score, or a single radius for all of them, and
/// must bound `|scoresᵢ − sᵢ|`; this function cannot check that. `largest`
/// selects the `k` largest. Requires `0 < k < n`.
///
/// Returns the set in ascending index order, since its order is not certified;
/// with `ordered`, the order is certified as well and the set is returned in rank
/// order.
///
/// Ported from `topk_determined` in `separatrix/separatrix/decide.py`, with the
/// refusal's straddling set from `escalate_row` in `separatrix/separatrix/exact.py`.
pub fn certified_topk(
    scores: &[f64],
    radii: &[f64],
    k: usize,
    largest: bool,
    ordered: bool,
) -> Result<Vec<usize>, Refusal> {
    check_enclosure(scores, radii)?;
    let n = scores.len();
    if k == 0 || k >= n {
        return Err(Refusal::Usage("k must satisfy 0 < k < n"));
    }

    // Negate and reuse: one rule, one place to be wrong. In the signed
    // orientation T always holds the k smallest.
    let sgn = if largest { -1.0 } else { 1.0 };
    let hi = |i: usize| up(sgn * scores[i] + radius(radii, i));
    let lo = |i: usize| down(sgn * scores[i] - radius(radii, i));
    let frontier = |inside: usize, outside: usize| {
        let (ri, ro) = (radius(radii, inside), radius(radii, outside));
        Frontier {
            inside,
            outside,
            inside_lo: down(scores[inside] - ri),
            inside_hi: up(scores[inside] + ri),
            outside_lo: down(scores[outside] - ro),
            outside_hi: up(scores[outside] + ro),
            gap: sgn * (scores[outside] - scores[inside]),
            width: ri + ro,
        }
    };

    let mut set = topk_set(scores, k, largest);
    let mut member = vec![false; n];
    for &i in &set {
        member[i] = true;
    }
    let inside = set
        .iter()
        .copied()
        .max_by(|&a, &b| hi(a).total_cmp(&hi(b)))
        .expect("k > 0");
    let outside = (0..n)
        .filter(|&j| !member[j])
        .min_by(|&a, &b| lo(a).total_cmp(&lo(b)))
        .expect("k < n");
    let (max_in, min_out) = (hi(inside), lo(outside));
    if max_in >= min_out {
        let straddling = (0..n)
            .filter(|&i| {
                if member[i] {
                    hi(i) >= min_out
                } else {
                    lo(i) <= max_in
                }
            })
            .collect();
        return Err(Refusal::BoundaryUndetermined {
            frontier: frontier(inside, outside),
            straddling,
        });
    }

    if ordered {
        if let Some(w) = set.windows(2).find(|w| hi(w[0]) >= lo(w[1])) {
            return Err(Refusal::BoundaryUndetermined {
                frontier: frontier(w[0], w[1]),
                straddling: vec![w[0].min(w[1]), w[0].max(w[1])],
            });
        }
    } else {
        set.sort_unstable();
    }
    Ok(set)
}

/// Certifies the index of the smallest score, or refuses and names the boundary.
/// The argmax is `certified_topk(scores, radii, 1, true, false)`.
///
/// Ported from `certified_argmin` in `separatrix/separatrix/api.py`: the top-k rule
/// at `k = 1`, so there is no second rule. Like the source, it requires at least
/// two scores.
pub fn certified_argmin(scores: &[f64], radii: &[f64]) -> Result<usize, Refusal> {
    certified_topk(scores, radii, 1, false, false).map(|set| set[0])
}

/// Which side of a threshold a score lies on, or that its enclosure straddles it.
///
/// Ported from the `int8` trit of `certified_threshold` in
/// `separatrix/separatrix/api.py`. A trit rather than a boolean, so that an
/// undetermined score cannot be consumed as either side without the caller
/// naming the case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum Trit {
    /// `D + R < t`: the exact score is below the threshold.
    Below = -1,
    /// The enclosure contains the threshold: this score is on the boundary.
    Undetermined = 0,
    /// `D − R > t`: the exact score is above the threshold.
    Above = 1,
}

/// Decides each score against `threshold`, in score units (squared, if the scores
/// are squared), so the threshold itself carries no rounding.
///
/// The [`Trit::Undetermined`] entries name the boundary.
///
/// Ported from `certified_threshold` in `separatrix/separatrix/api.py`.
pub fn certified_threshold(
    scores: &[f64],
    radii: &[f64],
    threshold: f64,
) -> Result<Vec<Trit>, Refusal> {
    check_enclosure(scores, radii)?;
    if !threshold.is_finite() {
        return Err(Refusal::NonFiniteInput {
            operand: "threshold",
            index: 0,
        });
    }
    Ok(scores
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            let r = radius(radii, i);
            if down(d - r) > threshold {
                Trit::Above
            } else if up(d + r) < threshold {
                Trit::Below
            } else {
                Trit::Undetermined
            }
        })
        .collect())
}
