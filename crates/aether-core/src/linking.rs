//! Gauss linking number, writhe and knot determinant of closed polygons in
//! `R^3`, and the rule under which a floating-point linking number may be
//! reported as an integer.
//!
//! Ported from `nerve/crates/nerve-topo/src/lib.rs` (closed-form Gauss linking
//! number and writhe), `nerve/crates/nerve-melt/src/lib.rs` (Alexander
//! determinant at `t = -1`) and `tangle/tangle/certify.py` (the one-directional
//! `LINKED` certificate). The error bound that licenses rounding appears in
//! neither source; it is derived below for this port.
//!
//! # Linking number in closed form
//!
//! For disjoint closed curves `A` and `B` the Gauss double integral
//!
//! ```text
//! Lk(A, B) = 1/(4 pi) oint_A oint_B (r_A - r_B) . (dr_A x dr_B) / |r_A - r_B|^3
//! ```
//!
//! is an integer and an invariant of the link under isotopy. Over two polygons
//! it splits into segment pairs, and for straight segments `p1 -> p2` of `A` and
//! `p3 -> p4` of `B` each term has a closed form: the signed area of the
//! spherical quadrilateral swept by the Gauss map, whose vertices are the
//! directions of `r13, r14, r24, r23`, with `rij = pj - pi`. The quadrilateral
//! is fanned from `r13` and each triangle of unit vectors `a, b, c` is evaluated
//! by the Van Oosterom–Strackee formula:
//!
//! ```text
//! Omega(a, b, c) = 2 atan2( a . (b x c),  1 + a.b + a.c + b.c )
//! omega_ij       = -[ Omega(r13^, r14^, r24^) + Omega(r13^, r24^, r23^) ]
//! Lk             = 1/(4 pi) sum_i sum_j omega_ij
//! ```
//!
//! There is no quadrature term and no length constant. The leading minus sign
//! orients the sum to the integral above. nerve fixed that sign against an
//! independent midpoint quadrature rather than analytically, and
//! `linking_number_matches_midpoint_quadrature` in `tests/linking.rs` pins it
//! again here.
//!
//! Writhe is the same integral taken over one curve against itself,
//! `Wr = 2/(4 pi) sum_{i<j} omega_ij`, over non-adjacent segment pairs.
//! Adjacent segments are coplanar with their shared vertex and contribute zero.
//! Writhe depends on the embedding, is not an invariant, and is never rounded.
//!
//! # Error bound
//!
//! For closed polygons `Lk` is an integer, so rounding the computed value `Lk^`
//! is justified exactly when its floating-point error is known to be below half
//! a unit. nerve measured the deviation from the integer on (2, 2n) torus links
//! (at most `2.16e-13` at 1024 segments, growing with the segment count) and
//! rounded on that evidence. This port carries a running error bound instead.
//! The bound is first order in the unit roundoff `u = 2^-53` (terms of order
//! `u^2` are dropped). It assumes IEEE-754 binary64 round-to-nearest, a
//! correctly rounded `sqrt`, and an `atan2` accurate to 2 ulp. The vertices are
//! taken as exact: the bound covers the arithmetic, not the process that
//! produced the coordinates.
//!
//! 1. Each difference `pj - pi` is rounded componentwise, and normalising it
//!    costs at most `5u` per component, so each unit direction lies within `7u`
//!    of the exact direction.
//! 2. With directions perturbed by `7u`, the computed numerator `N` and
//!    denominator `D` of each triangle lie within `47u` and `60u` of their exact
//!    values. Both are bounded here by `K = 128u`.
//! 3. If the box of half-width `K` about `(D, N)` does not meet the branch cut
//!    `{N = 0, D <= 0}` of `atan2`, and `rho = hypot(N, D) > 2K`, then `atan2`
//!    is smooth on that box, and the triangle's error is
//!    `e = 2 (sqrt(2) K / (rho - sqrt(2) K) + 8u)`.
//! 4. Recursive summation of `n` terms adds at most `gamma_n sum |omega_ij|`,
//!    with `gamma_n = n u / (1 - n u)` (Higham, *Accuracy and Stability of
//!    Numerical Algorithms*, ch. 4). The final division by `4 pi` adds
//!    `2u |Lk^|`.
//!
//! ```text
//! B = ( sum_ij e_ij + gamma_n sum_ij |omega_ij| ) / (4 pi)  +  2u |Lk^|
//! ```
//!
//! Here `e_ij` sums the two triangle errors of pair `ij` and the rounding of
//! their sum, `u |Omega_1 + Omega_2|`.
//!
//! # What is certified
//!
//! [`GaussLinking::certify`] rounds `Lk^` to `n` only when
//! `|Lk^ - n| + B < 1/2`. The interval `[Lk^ - B, Lk^ + B]` then lies inside
//! `(n - 1/2, n + 1/2)`, so `n` is the only integer it can contain. Half a unit
//! is the threshold `nerve/crates/nerve-periodic/src/lib.rs` names as the only
//! defensible one for a closed polygon.
//!
//! - [`LinkVerdict::Linked`]: `Lk = lk != 0`. A split link has `Lk = 0`, so no
//!   isotopy that keeps the curves disjoint can separate them. This is tangle's
//!   theorem T3, read contrapositively.
//! - [`LinkVerdict::ZeroLinking`]: `Lk = 0` is proven, and this certifies
//!   nothing. The Whitehead link has `Lk = 0` and is not split. No outcome
//!   reads "unlinked", because the linking number cannot supply the converse of
//!   the certificate.
//! - [`LinkVerdict::Undetermined`]: the bound does not prove the rounding. The
//!   estimate is returned and no claim is made.
//!
//! The certificate is conditional on the first-order bound above; it is not
//! interval arithmetic. Writhe and the knot determinant carry no certificate.
//!
//! # Knot determinant
//!
//! For a knot diagram with crossing `c`, over-arc `o`, incoming under-arc `a`
//! and outgoing under-arc `b`, the Alexander matrix rows are
//!
//! ```text
//! positive:  a -> t,   b -> -1,   o -> 1 - t
//! negative:  a -> 1,   b -> -t,   o -> t - 1
//! ```
//!
//! At `t = -1` these become `(-1, -1, 2)` and `(1, 1, -2)`, which are exact
//! negatives. Negating a row leaves `|det|` unchanged, so `|Delta(-1)|` is the
//! absolute determinant of any `(c - 1) x (c - 1)` minor, and no crossing sign
//! is computed at all. The minor is evaluated exactly in `i128` by Bareiss
//! elimination. The diagram comes from a projection along `z`. When that
//! projection is not generic, it is retried over eight fixed reorientations.
//! The projected coordinates use the source's absolute tolerance `1e-9`, which
//! is not scale-free. The determinant is an invariant but not a complete one:
//! `4_1` and `5_1` both give 5, and a value of 1 does not certify the unknot.
//!
//! # Refusals
//!
//! Every refusal is a [`LinkingError`], never a number:
//!
//! - a curve with fewer than three vertices (nerve returned `0.0` or `1`);
//! - a non-finite coordinate;
//! - two segments that meet, or come within rounding of meeting. This is the
//!   branch-cut condition of step 3, and it involves no length scale. The four
//!   fan directions are vertices of the Minkowski difference of the two
//!   segments. An exact intersection puts the origin in that parallelogram,
//!   which forces `N = 0, D <= 0` on one fan triangle. Conversely, a refusal
//!   means the segments meet to within rounding;
//! - a difference vector whose squared length is not a normal binary64 number,
//!   where the analysis above no longer holds;
//! - for the determinant, no generic projection among the eight
//!   reorientations.

#![warn(missing_docs)]

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use core::f64::consts::{PI, SQRT_2};

use libm::{atan2, hypot, round, sqrt};
#[cfg(feature = "alloc")]
use libm::{cos, sin};

type V3 = [f64; 3];

const FOUR_PI: f64 = 4.0 * PI;

/// Unit roundoff of binary64, `u = 2^-53`.
const U: f64 = f64::EPSILON / 2.0;

/// Absolute error admitted on each computed Van Oosterom–Strackee numerator
/// and denominator. Step 2 of the module docs derives `47u` and `60u`.
const K: f64 = 128.0 * U;

/// Why a curve, or a pair of curves, was refused.
///
/// Segment `i` of a curve runs from vertex `i` to vertex `(i + 1) % len`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkingError {
    /// A closed polygon needs at least three vertices.
    TooFewVertices {
        /// 0 for the first curve argument, 1 for the second.
        curve: usize,
        /// Vertices supplied.
        len: usize,
    },
    /// A coordinate is NaN or infinite.
    NonFinite {
        /// 0 for the first curve argument, 1 for the second.
        curve: usize,
        /// Index of the first offending vertex.
        vertex: usize,
    },
    /// Two segments meet, or come within rounding error of meeting, so the
    /// Gauss integral is undefined or its sign cannot be resolved. For writhe
    /// both indices refer to the one curve.
    Intersecting {
        /// Segment of the first curve.
        segment_a: usize,
        /// Segment of the second curve.
        segment_b: usize,
    },
    /// A vertex difference between these segments has a squared length
    /// that overflows or underflows binary64 normal range, where the error
    /// bound does not hold. Lengths outside roughly `[1.5e-154, 1.3e154]`.
    OutOfRange {
        /// Segment of the first curve.
        segment_a: usize,
        /// Segment of the second curve.
        segment_b: usize,
    },
    /// No generic projection with an `i128`-representable determinant was
    /// found among the eight fixed reorientations. The causes are a vertex on
    /// a strand, strands collinear or touching in projection (which a
    /// self-intersecting curve forces in every projection), Bareiss overflow,
    /// or a singular minor. The source does not distinguish these causes.
    NoGenericProjection,
}

/// A linking number evaluated in floating point, with a bound on its error.
///
/// Returned by [`linking_number`]. Ported from
/// `nerve/crates/nerve-topo/src/lib.rs` (`linking_number_closed`), with the
/// error bound of the module docs added.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussLinking {
    /// The computed double sum `Lk^`. For closed polygons the exact value is
    /// an integer.
    pub value: f64,
    /// Upper bound on `|value - Lk|`, first order in the unit roundoff.
    pub error_bound: f64,
}

/// What the linking number proves about two closed curves.
///
/// No variant reads "unlinked": `Lk = 0` holds for links that cannot be
/// separated, the Whitehead link among them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinkVerdict {
    /// `Lk` is proven to equal `lk`, which is nonzero, so the curves cannot be
    /// separated by any isotopy that keeps them disjoint.
    Linked {
        /// The exact linking number. Its sign depends on both orientations.
        lk: i64,
    },
    /// `Lk` is proven to be exactly zero. This certifies nothing about
    /// separability.
    ZeroLinking,
    /// The error bound does not prove any rounding, so no claim is made.
    Undetermined {
        /// The computed, unrounded value.
        lk_estimate: f64,
        /// Its error bound.
        error_bound: f64,
    },
}

impl GaussLinking {
    /// Round to an integer if, and only if, the error bound proves it.
    ///
    /// Rounds `value` to `n` when `|value - n| + error_bound < 1/2`, then maps
    /// `n != 0` to [`LinkVerdict::Linked`] and `n = 0` to
    /// [`LinkVerdict::ZeroLinking`]. Everything else, including NaN, is
    /// [`LinkVerdict::Undetermined`]. The outcomes follow `certify` in
    /// `tangle/tangle/certify.py`: `CERTIFIED LINKED` when `lk != 0`, and
    /// `NOT CERTIFIED LK_ZERO` when `lk = 0`.
    pub fn certify(&self) -> LinkVerdict {
        let n = round(self.value);
        if (self.value - n).abs() + self.error_bound < 0.5 {
            if n == 0.0 {
                LinkVerdict::ZeroLinking
            } else {
                LinkVerdict::Linked { lk: n as i64 }
            }
        } else {
            LinkVerdict::Undetermined {
                lk_estimate: self.value,
                error_bound: self.error_bound,
            }
        }
    }
}

/// Gauss linking number of two closed polygons, with its error bound.
///
/// Each slice is read cyclically: the segment from the last vertex back to the
/// first is included. `O(|a| |b|)` segment pairs, no allocation. Reversing one
/// curve negates the result; swapping the curves leaves it unchanged.
///
/// Ported from `nerve/crates/nerve-topo/src/lib.rs` (`linking_number_closed`,
/// `omega`, `solid_angle`). nerve returns `0.0` for fewer than three vertices
/// and for degenerate segment pairs. Here both are refusals.
pub fn linking_number(a: &[[f64; 3]], b: &[[f64; 3]]) -> Result<GaussLinking, LinkingError> {
    validate(a, 0)?;
    validate(b, 1)?;
    gauss_sum(a, b, false)
}

/// Writhe of one closed polygon.
///
/// Real-valued and not a topological invariant, so it is never rounded and
/// carries no certificate. Non-adjacent segments that meet are refused.
/// `O(|a|^2)`, no allocation.
///
/// Ported from `nerve/crates/nerve-topo/src/lib.rs` (`writhe_closed`). Adjacent
/// pairs, which contribute zero exactly, are skipped rather than summed as
/// roundoff.
pub fn writhe(a: &[[f64; 3]]) -> Result<f64, LinkingError> {
    validate(a, 0)?;
    gauss_sum(a, a, true).map(|g| 2.0 * g.value)
}

/// Knot determinant `|Delta(-1)|` of one closed polygon.
///
/// The curve is read cyclically. The result is exact integer arithmetic on a
/// diagram, but extracting the diagram uses the source's absolute projection
/// tolerance `1e-9`, with no error bound. An invariant, but not a complete
/// one: see the module docs.
///
/// Ported from `nerve/crates/nerve-melt/src/lib.rs` (`alexander_det_minus_one`,
/// `det_for_orientation`, `det_bareiss`). nerve returns `Some(1)` for fewer
/// than three vertices. Here that is a refusal.
#[cfg(feature = "alloc")]
pub fn knot_determinant(curve: &[[f64; 3]]) -> Result<u128, LinkingError> {
    validate(curve, 0)?;
    // Fixed, seed-free reorientation ladder, so the result is deterministic.
    (0..8)
        .find_map(|attempt| det_for_orientation(curve, attempt))
        .ok_or(LinkingError::NoGenericProjection)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Gauss sum
// ═══════════════════════════════════════════════════════════════════════════════

fn validate(curve: &[V3], which: usize) -> Result<(), LinkingError> {
    if curve.len() < 3 {
        return Err(LinkingError::TooFewVertices {
            curve: which,
            len: curve.len(),
        });
    }
    match curve.iter().position(|p| !p.iter().all(|x| x.is_finite())) {
        Some(vertex) => Err(LinkingError::NonFinite {
            curve: which,
            vertex,
        }),
        None => Ok(()),
    }
}

/// `sum omega_ij / (4 pi)` with its error bound. `same` selects writhe: pairs
/// `i < j` of one curve, adjacent pairs skipped.
fn gauss_sum(a: &[V3], b: &[V3], same: bool) -> Result<GaussLinking, LinkingError> {
    let (n, m) = (a.len(), b.len());
    let (mut sum, mut abs_sum, mut err, mut terms) = (0.0, 0.0, 0.0, 0usize);
    for i in 0..n {
        let (p1, p2) = (a[i], a[(i + 1) % n]);
        let first = if same { i + 2 } else { 0 };
        for j in first..m {
            if same && i == 0 && j == n - 1 {
                continue; // adjacent through the closing segment
            }
            let (p3, p4) = (b[j], b[(j + 1) % m]);
            let (w, e) = omega(p1, p2, p3, p4).map_err(|fault| fault.at(i, j))?;
            sum += w;
            abs_sum += w.abs();
            err += e;
            terms += 1;
        }
    }
    let nu = terms as f64 * U;
    let value = sum / FOUR_PI;
    let error_bound = if nu < 1.0 {
        (err + nu / (1.0 - nu) * abs_sum) / FOUR_PI + 2.0 * U * value.abs()
    } else {
        f64::INFINITY
    };
    Ok(GaussLinking { value, error_bound })
}

enum Fault {
    Touching,
    Range,
}

impl Fault {
    fn at(self, segment_a: usize, segment_b: usize) -> LinkingError {
        match self {
            Fault::Touching => LinkingError::Intersecting {
                segment_a,
                segment_b,
            },
            Fault::Range => LinkingError::OutOfRange {
                segment_a,
                segment_b,
            },
        }
    }
}

/// `omega` for segments `p1 -> p2` and `p3 -> p4`, and its error `e` (steps
/// 1–3 of the module docs).
fn omega(p1: V3, p2: V3, p3: V3, p4: V3) -> Result<(f64, f64), Fault> {
    let r13 = direction(sub(p3, p1))?;
    let r14 = direction(sub(p4, p1))?;
    let r24 = direction(sub(p4, p2))?;
    let r23 = direction(sub(p3, p2))?;
    let (w1, e1) = triangle(r13, r14, r24)?;
    let (w2, e2) = triangle(r13, r24, r23)?;
    let w = w1 + w2;
    Ok((-w, e1 + e2 + U * w.abs()))
}

fn direction(v: V3) -> Result<V3, Fault> {
    // Gradual underflow makes `x - y == 0` exactly when `x == y`, so this is a
    // shared vertex and never an underflowed difference.
    if v == [0.0; 3] {
        return Err(Fault::Touching);
    }
    let s = dot(v, v);
    if !s.is_normal() {
        return Err(Fault::Range);
    }
    let n = sqrt(s);
    Ok([v[0] / n, v[1] / n, v[2] / n])
}

/// Van Oosterom–Strackee solid angle of the unit-vector triangle `a, b, c`,
/// and its error, refusing where `atan2` is not smooth within rounding.
fn triangle(a: V3, b: V3, c: V3) -> Result<(f64, f64), Fault> {
    let num = dot(a, cross(b, c));
    let den = 1.0 + dot(a, b) + dot(a, c) + dot(b, c);
    let rho = hypot(num, den);
    if (num.abs() <= K && den <= K) || rho <= 2.0 * K {
        return Err(Fault::Touching);
    }
    let theta_err = SQRT_2 * K / (rho - SQRT_2 * K) + 8.0 * U;
    Ok((2.0 * atan2(num, den), 2.0 * theta_err))
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ═══════════════════════════════════════════════════════════════════════════════
// Knot determinant
// ═══════════════════════════════════════════════════════════════════════════════

/// One projection's `|Delta(-1)|`, or `None` when it is not generic or the
/// determinant overflows `i128`.
#[cfg(feature = "alloc")]
fn det_for_orientation(pts: &[V3], attempt: usize) -> Option<u128> {
    const EPS: f64 = 1e-9;
    let n = pts.len();
    let a = attempt as f64;
    // Rotate about x then y by incommensurate angles; attempt 0 is the identity.
    let (s1, c1) = (sin(0.7137 * a), cos(0.7137 * a));
    let (s2, c2) = (sin(0.3110 * a), cos(0.3110 * a));
    let p: Vec<V3> = pts
        .iter()
        .map(|q| {
            let (y, z) = (c1 * q[1] - s1 * q[2], s1 * q[1] + c1 * q[2]);
            let (x, z) = (c2 * q[0] - s2 * z, s2 * q[0] + c2 * z);
            [x, y, z]
        })
        .collect();

    // (segment, parameter, crossing id, is_over)
    let mut passes: Vec<(usize, f64, usize, bool)> = Vec::new();
    let mut n_cross = 0usize;
    for i in 0..n {
        let (a0, a1) = (p[i], p[(i + 1) % n]);
        for j in (i + 1)..n {
            // Segments sharing a vertex cannot cross transversally.
            if j == i + 1 || (i == 0 && j == n - 1) {
                continue;
            }
            let (b0, b1) = (p[j], p[(j + 1) % n]);
            let d0 = [a1[0] - a0[0], a1[1] - a0[1]];
            let d1 = [b1[0] - b0[0], b1[1] - b0[1]];
            let den = d0[0] * d1[1] - d0[1] * d1[0];
            let scale = (d0[0].abs() + d0[1].abs()) * (d1[0].abs() + d1[1].abs());
            let r = [b0[0] - a0[0], b0[1] - a0[1]];
            if den.abs() <= EPS * scale.max(1e-30) {
                // Parallel in projection: no transversal crossing. Only the
                // collinear case is degenerate; a convex curve has many
                // anti-parallel pairs, and rejecting those would reject the
                // unknot.
                let off = r[0] * d0[1] - r[1] * d0[0];
                if off.abs() <= EPS * (d0[0].abs() + d0[1].abs()).max(1e-30) {
                    return None;
                }
                continue;
            }
            let s = (r[0] * d1[1] - r[1] * d1[0]) / den;
            let u = (r[0] * d0[1] - r[1] * d0[0]) / den;
            if !(EPS..=1.0 - EPS).contains(&s) || !(EPS..=1.0 - EPS).contains(&u) {
                // Outside the segments, or a vertex on a strand; only the
                // second is degenerate.
                let near_end = |t: f64| (t > -EPS && t < EPS) || (t > 1.0 - EPS && t < 1.0 + EPS);
                if near_end(s) || near_end(u) {
                    return None;
                }
                continue;
            }
            let za = a0[2] + s * (a1[2] - a0[2]);
            let zb = b0[2] + u * (b1[2] - b0[2]);
            if (za - zb).abs() <= EPS {
                return None; // strands touch in projection
            }
            let c = n_cross;
            n_cross += 1;
            passes.push((i, s, c, za > zb));
            passes.push((j, u, c, zb > za));
        }
    }
    if n_cross == 0 {
        return Some(1); // no crossings: the unknot
    }

    // Walk the curve in order and label arcs. Arcs run between consecutive
    // under-passes, so there are exactly n_cross of them; position 0 lies in
    // the arc that follows the last under-pass.
    passes.sort_by(|x, y| x.0.cmp(&y.0).then(x.1.total_cmp(&y.1)));
    let mut arc = n_cross - 1;
    let mut inn = vec![usize::MAX; n_cross];
    let mut out = vec![usize::MAX; n_cross];
    let mut over = vec![usize::MAX; n_cross];
    for &(_, _, c, is_over) in &passes {
        if is_over {
            over[c] = arc;
        } else {
            inn[c] = arc;
            arc = (arc + 1) % n_cross;
            out[c] = arc;
        }
    }
    if inn.iter().chain(&out).chain(&over).any(|&v| v == usize::MAX) {
        return None; // a crossing missing a pass: the walk was inconsistent
    }

    // At t = -1 both crossing signs give the same row up to -1, which does not
    // change |det|, so no crossing sign is computed anywhere above.
    let mut mat = vec![vec![0i128; n_cross]; n_cross];
    for (c, row) in mat.iter_mut().enumerate() {
        row[inn[c]] -= 1;
        row[out[c]] -= 1;
        row[over[c]] += 2;
    }
    // Delete one row and one column; any choice gives the same |det|.
    let k = n_cross - 1;
    let minor: Vec<Vec<i128>> = mat[..k].iter().map(|r| r[..k].to_vec()).collect();
    det_bareiss(minor).map(|d| d.unsigned_abs())
}

/// Fraction-free (Bareiss) integer determinant. `None` on `i128` overflow
/// rather than a wrapped value, and when no pivot exists: a singular minor,
/// which no knot diagram has, since a knot's determinant is odd.
#[cfg(feature = "alloc")]
fn det_bareiss(mut a: Vec<Vec<i128>>) -> Option<i128> {
    let n = a.len();
    if n == 0 {
        return Some(1);
    }
    let mut sign = 1i128;
    let mut prev = 1i128;
    for k in 0..n - 1 {
        if a[k][k] == 0 {
            let piv = (k + 1..n).find(|&r| a[r][k] != 0)?;
            a.swap(k, piv);
            sign = -sign;
        }
        for i in k + 1..n {
            for j in k + 1..n {
                let t = a[i][j]
                    .checked_mul(a[k][k])?
                    .checked_sub(a[i][k].checked_mul(a[k][j])?)?;
                a[i][j] = t / prev; // exact in Bareiss
            }
        }
        prev = a[k][k];
    }
    Some(sign * a[n - 1][n - 1])
}
