//! Invariant tests for `aether_core::linking`.
//!
//! Each test states a theorem about the Gauss linking number, or about the rule
//! that turns it into a certificate, as an executable assertion. A wrong
//! linking number still looks like a real number close to an integer; these
//! are the properties such a number violates.
//!
//! Theorems asserted here:
//!   1. Ground truth     — Hopf |lk| = 1, (2, 2k) torus |lk| = k, flat rings 0.
//!   2. Whitehead link   — lk = 0 on a non-split link: zero certifies nothing.
//!   3. Orientation      — midpoint quadrature pins the sign and the 1/(4π).
//!   4. Error bound      — the bound covers the measured distance to the integer.
//!   5. Certification    — round only when |lk^ − n| + B < 1/2.
//!   6. Invariance       — rotation, translation, scaling, start vertex, swap.
//!   7. Antisymmetry     — reversing one orientation, or reflecting, negates lk.
//!   8. Refusals         — intersection, too few vertices, non-finite, range.
//!   9. Writhe           — real-valued, odd under reflection, refuses crossings.
//!  10. Knot determinant — classical table, projection and mirror, 4_1 = 5_1.
//!
//! Generators `twisted_band`, `rotate`, `torus_knot` and `figure_eight` are
//! nerve's (`nerve-topo/tests/topo.rs`, `nerve-melt/src/lib.rs`); braid words are
//! tangle's (`tests/test_certify.py`, `tests/test_alexander.py`).

use aether_core::linking::{
    knot_determinant, linking_number, writhe, GaussLinking, LinkVerdict, LinkingError,
};

use std::f64::consts::{PI, TAU};

type Vec3 = [f64; 3];

// ═══════════════════════════════════════════════════════════════════════════════
// Geometry
// ═══════════════════════════════════════════════════════════════════════════════

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mul(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn map_all(v: &[Vec3], f: impl Fn(Vec3) -> Vec3) -> Vec<Vec3> {
    v.iter().map(|p| f(*p)).collect()
}

fn rotate(p: Vec3, ax: f64, ay: f64, az: f64) -> Vec3 {
    let (s, c) = ax.sin_cos();
    let p = [p[0], c * p[1] - s * p[2], s * p[1] + c * p[2]];
    let (s, c) = ay.sin_cos();
    let p = [c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2]];
    let (s, c) = az.sin_cos();
    [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]
}

fn lk(a: &[Vec3], b: &[Vec3]) -> GaussLinking {
    linking_number(a, b).expect("an admissible pair of closed polygons")
}

fn certified_lk(a: &[Vec3], b: &[Vec3]) -> i64 {
    let g = lk(a, b);
    match g.certify() {
        LinkVerdict::Linked { lk } => lk,
        other => panic!("expected a LINKED certificate, got {other:?} from {g:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Curve generators
// ═══════════════════════════════════════════════════════════════════════════════

/// Boundary curves of an annulus carrying `n_twist` full twists: the (2, 2n)
/// torus link, whose components have linking number exactly `n`. `n = 0` gives
/// two concentric coplanar circles, `n = 1` the Hopf link.
fn twisted_band(n_twist: i32, r: f64, a: f64, m: usize) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut ca = Vec::with_capacity(m);
    let mut cb = Vec::with_capacity(m);
    for k in 0..m {
        let t = TAU * (k as f64) / (m as f64);
        let core = [r * t.cos(), r * t.sin(), 0.0];
        let e_r = [t.cos(), t.sin(), 0.0];
        let nt = (n_twist as f64) * t;
        let u = add(mul(e_r, nt.cos()), [0.0, 0.0, nt.sin()]);
        ca.push(add(core, mul(u, a)));
        cb.push(sub(core, mul(u, a)));
    }
    (ca, cb)
}

fn circle(r: f64, m: usize) -> Vec<Vec3> {
    (0..m)
        .map(|k| {
            let t = TAU * (k as f64) / (m as f64);
            [r * t.cos(), r * t.sin(), 0.0]
        })
        .collect()
}

/// The `(p, q)` torus knot as an `n`-gon; the `(2, q)` family has determinant `q`.
fn torus_knot(p: usize, q: usize, n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|k| {
            let t = TAU * k as f64 / n as f64;
            let r = 2.0 + (q as f64 * t).cos();
            [r * (p as f64 * t).cos(), r * (p as f64 * t).sin(), (q as f64 * t).sin()]
        })
        .collect()
}

/// The figure-eight knot 4_1 as an `n`-gon. Determinant 5.
fn figure_eight(n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|k| {
            let t = TAU * k as f64 / n as f64;
            let r = 2.0 + (2.0 * t).cos();
            [r * (3.0 * t).cos(), r * (3.0 * t).sin(), (4.0 * t).sin()]
        })
        .collect()
}

/// Closure of a braid word as closed polygons, one per component.
///
/// Slot `p` at braid time `s + f` sits at angle `2π (s + f) / len` on the circle
/// of radius `3 + p`. Generator `±i` swaps slots `i − 1` and `i` along a
/// half-cosine. The strand moving outward rises to `z = +0.5` for `+i` and dips
/// to `−0.5` for `−i`, and the other strand does the opposite. Nine samples per
/// slot keep every crossing off a vertex in the `z`-projection.
fn braid_closure(word: &[i32], strands: usize) -> Vec<Vec<Vec3>> {
    const SAMPLES: usize = 9;
    let len = word.len() as f64;
    let mut visited = vec![false; strands];
    let mut components = Vec::new();
    for start in 0..strands {
        if visited[start] {
            continue;
        }
        let mut curve = Vec::new();
        let mut p = start;
        loop {
            visited[p] = true;
            for (s, &g) in word.iter().enumerate() {
                let i = g.unsigned_abs() as usize;
                let lift = if g > 0 { 0.5 } else { -0.5 };
                let (to, z) = if p + 1 == i {
                    (i, lift)
                } else if p == i {
                    (i - 1, -lift)
                } else {
                    (p, 0.0)
                };
                for k in 0..SAMPLES {
                    let f = k as f64 / SAMPLES as f64;
                    let phi = TAU * (s as f64 + f) / len;
                    let r = 3.0 + p as f64 + (to as f64 - p as f64) * (1.0 - (PI * f).cos()) / 2.0;
                    curve.push([r * phi.cos(), r * phi.sin(), z * (PI * f).sin()]);
                }
                p = to;
            }
            if p == start {
                break;
            }
        }
        components.push(curve);
    }
    components
}

/// Unit square in the plane `z = 0`.
fn square_a() -> Vec<Vec3> {
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
}

/// A 1 × 2 rectangle in the plane `x = 0.5` whose top edge sits at `y = dy`.
/// For `dy > 0` the edge `y = 0` of [`square_a`] pierces it once; for `dy < 0`
/// nothing does; at `dy = 0` its first segment passes through that edge.
fn square_b(dy: f64) -> Vec<Vec3> {
    vec![
        [0.5, dy, -1.0],
        [0.5, dy, 1.0],
        [0.5, dy - 1.0, 1.0],
        [0.5, dy - 1.0, -1.0],
    ]
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1–2. Ground truth
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn hopf_link_certifies_linked_with_lk_of_magnitude_one() {
    let (a, b) = twisted_band(1, 1.0, 0.3, 200);
    assert_eq!(certified_lk(&a, &b).abs(), 1);
}

#[test]
fn torus_link_2_2k_certifies_lk_k_with_one_sign_across_the_family() {
    // k = 0: every difference vector lies in z = 0, so every triangle's
    // numerator is exactly zero and the sum is exactly zero, not merely small.
    let (a, b) = twisted_band(0, 1.0, 0.2, 256);
    let flat = lk(&a, &b);
    assert_eq!(flat.value, 0.0, "coplanar rings gave {flat:?}");
    assert_eq!(flat.certify(), LinkVerdict::ZeroLinking);

    let mut sign = 0;
    for k in 1..=4i64 {
        let (a, b) = twisted_band(k as i32, 1.0, 0.2, 256);
        let got = certified_lk(&a, &b);
        assert_eq!(got.abs(), k, "(2,{}) torus link certified lk = {got}", 2 * k);
        if sign == 0 {
            sign = got.signum();
        }
        assert_eq!(got.signum(), sign, "one generator must give one sign, k = {k}");
    }
}

/// tangle's `test_whitehead_is_not_split_by_an_independent_invariant` shows the
/// diagram of this braid word has determinant 8, and every split link has
/// determinant 0, so its components do not come apart although `lk = 0`.
#[test]
fn whitehead_link_has_lk_zero_and_is_never_certified_as_separable() {
    // Control: the same generator must reproduce nonzero linking, or the zero
    // below could be the builder's fault rather than the link's.
    for (word, want) in [(&[1, 1][..], 1i64), (&[1, 1, 1, 1][..], 2)] {
        let c = braid_closure(word, 2);
        assert_eq!(c.len(), 2, "braid {word:?} should close to two components");
        assert_eq!(certified_lk(&c[0], &c[1]).abs(), want, "braid {word:?}");
    }

    let c = braid_closure(&[1, -2, 1, -2, -2], 3);
    assert_eq!(c.len(), 2, "the Whitehead link has two components");
    let g = lk(&c[0], &c[1]);
    assert!(
        g.value.abs() <= g.error_bound,
        "true lk is 0, so the bound must cover the value: {g:?}"
    );
    // The proven zero is the whole verdict. There is no variant that could
    // read as "separable", and none is returned.
    assert_eq!(g.certify(), LinkVerdict::ZeroLinking, "Whitehead link: {g:?}");
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3–5. Orientation, error bound, certification rule
// ═══════════════════════════════════════════════════════════════════════════════

/// Midpoint rule for the same double integral. Slow, low order, and correct by
/// inspection. Pins the sign convention and the 1/(4π) of the closed form,
/// which no invariance test can see: every invariance holds for `-Lk` too.
fn lk_midpoint_quadrature(a: &[Vec3], b: &[Vec3]) -> f64 {
    let dot = |x: Vec3, y: Vec3| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let cross = |x: Vec3, y: Vec3| {
        [
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        ]
    };
    let mut s = 0.0;
    for i in 0..a.len() {
        let (a0, a1) = (a[i], a[(i + 1) % a.len()]);
        for j in 0..b.len() {
            let (b0, b1) = (b[j], b[(j + 1) % b.len()]);
            let r = sub(mul(add(a0, a1), 0.5), mul(add(b0, b1), 0.5));
            let d = dot(r, r).sqrt();
            s += dot(r, cross(sub(a1, a0), sub(b1, b0))) / (d * d * d);
        }
    }
    s / (4.0 * PI)
}

#[test]
fn linking_number_matches_midpoint_quadrature() {
    for n in 0..=2 {
        let (a, b) = twisted_band(n, 1.0, 0.3, 500);
        let exact = lk(&a, &b).value;
        let quad = lk_midpoint_quadrature(&a, &b);
        assert!(
            (exact - quad).abs() < 2e-2,
            "n_twist={n}: closed form {exact} vs quadrature {quad}"
        );
    }
}

#[test]
fn error_bound_covers_the_measured_distance_to_the_integer() {
    for twists in 1..=3 {
        for m in [64usize, 256, 512] {
            let (a, b) = twisted_band(twists, 1.0, 0.2, m);
            let g = lk(&a, &b);
            let n = g.value.round();
            let dev = (g.value - n).abs();
            assert_eq!(n.abs(), twists as f64, "twists={twists} m={m}: {g:?}");
            assert!(
                dev <= g.error_bound,
                "twists={twists} m={m}: deviation {dev:.3e} exceeds the bound {:.3e}",
                g.error_bound
            );
            assert!(
                g.error_bound < 1e-6,
                "twists={twists} m={m}: bound {:.3e} is too loose to certify anything",
                g.error_bound
            );
        }
    }
}

#[test]
fn certification_rounds_only_when_the_bound_proves_the_rounding() {
    let verdict = |value, error_bound| GaussLinking { value, error_bound }.certify();
    let undetermined = |v: LinkVerdict| matches!(v, LinkVerdict::Undetermined { .. });

    assert_eq!(verdict(1.0 + 1e-13, 1e-10), LinkVerdict::Linked { lk: 1 });
    assert_eq!(verdict(-2.0 - 3e-12, 1e-9), LinkVerdict::Linked { lk: -2 });
    assert_eq!(verdict(1e-14, 1e-10), LinkVerdict::ZeroLinking);
    assert_eq!(verdict(-0.0, 0.0), LinkVerdict::ZeroLinking);
    // [0.7, 1.1] holds one integer, so the rounding is proven.
    assert_eq!(verdict(0.9, 0.2), LinkVerdict::Linked { lk: 1 });
    // [0.5, 0.9] reaches the half-unit boundary: not proven.
    assert!(undetermined(verdict(0.7, 0.2)));
    assert!(undetermined(verdict(0.5, 1e-12)));
    // A value on the integer with a bound of half a unit proves nothing.
    assert!(undetermined(verdict(3.0, 0.5)));
    assert!(undetermined(verdict(f64::NAN, 0.0)));
    assert!(undetermined(verdict(1.0, f64::NAN)));
    assert!(undetermined(verdict(1.0, f64::INFINITY)));
    assert_eq!(
        verdict(0.7, 0.2),
        LinkVerdict::Undetermined {
            lk_estimate: 0.7,
            error_bound: 0.2
        }
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6–7. Invariance and antisymmetry
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn linking_number_is_invariant_under_rigid_rotation() {
    let (a, b) = twisted_band(2, 1.0, 0.3, 200);
    let base = lk(&a, &b);
    let (ax, ay, az) = (0.731, -1.219, 2.443);
    let rot = lk(
        &map_all(&a, |p| rotate(p, ax, ay, az)),
        &map_all(&b, |p| rotate(p, ax, ay, az)),
    );
    assert!((base.value - rot.value).abs() < 1e-10, "{base:?} vs {rot:?}");
    assert_eq!(rot.certify(), base.certify());
}

#[test]
fn linking_number_is_invariant_under_translation() {
    let (a, b) = twisted_band(2, 1.0, 0.3, 200);
    let base = lk(&a, &b);
    let t = [123.5, -7.25, 4096.0];
    let moved = lk(&map_all(&a, |p| add(p, t)), &map_all(&b, |p| add(p, t)));
    assert!((base.value - moved.value).abs() < 1e-10, "{base:?} vs {moved:?}");
    assert_eq!(moved.certify(), base.certify());
}

/// Powers of two scale binary64 coordinates without rounding, so the value
/// and its bound must come back bitwise. A length constant anywhere in the
/// integrator or the refusal test would show up here and nowhere else.
#[test]
fn linking_number_and_bound_are_bitwise_invariant_under_power_of_two_scaling() {
    let (a, b) = twisted_band(2, 1.0, 0.3, 200);
    let base = lk(&a, &b);
    for c in [0.25, 0.5, 2.0, 1024.0, 1.0 / 1024.0, 2f64.powi(400)] {
        let got = lk(&map_all(&a, |p| mul(p, c)), &map_all(&b, |p| mul(p, c)));
        assert_eq!(base.value.to_bits(), got.value.to_bits(), "scale {c}");
        assert_eq!(base.error_bound.to_bits(), got.error_bound.to_bits(), "scale {c}");
    }
    for c in [1e-6, 0.037, 3.7, 1.9e5] {
        let got = lk(&map_all(&a, |p| mul(p, c)), &map_all(&b, |p| mul(p, c)));
        assert!((base.value - got.value).abs() < 1e-10, "scale {c}: {got:?}");
        assert_eq!(got.certify(), base.certify(), "scale {c}");
    }
}

#[test]
fn linking_number_is_invariant_under_cyclic_shift_of_the_start_vertex() {
    let (a, b) = twisted_band(3, 1.0, 0.25, 300);
    let base = lk(&a, &b);
    for k in [1usize, 77, 299] {
        let mut sa = a.clone();
        sa.rotate_left(k);
        let mut sb = b.clone();
        sb.rotate_left((2 * k) % b.len());
        let got = lk(&sa, &sb);
        assert!((base.value - got.value).abs() < 1e-12, "shift {k}: {got:?}");
        assert_eq!(got.certify(), base.certify(), "shift {k}");
    }
}

#[test]
fn reversing_one_orientation_or_reflecting_negates_lk() {
    let (a, b) = twisted_band(2, 1.0, 0.3, 200);
    let base = certified_lk(&a, &b);
    assert_eq!(base.abs(), 2);

    let mut rb = b.clone();
    rb.reverse();
    assert_eq!(certified_lk(&a, &rb), -base, "reversing B");
    let mut ra = a.clone();
    ra.reverse();
    assert_eq!(certified_lk(&ra, &b), -base, "reversing A");
    assert_eq!(certified_lk(&ra, &rb), base, "reversing both");

    let mirror = |p: Vec3| [p[0], p[1], -p[2]];
    assert_eq!(
        certified_lk(&map_all(&a, mirror), &map_all(&b, mirror)),
        -base,
        "reflection"
    );
}

#[test]
fn swapping_the_curves_keeps_lk() {
    let (a, b) = twisted_band(3, 1.0, 0.25, 300);
    let ab = lk(&a, &b);
    let ba = lk(&b, &a);
    assert!((ab.value - ba.value).abs() < 1e-12, "{ab:?} vs {ba:?}");
    assert_eq!(ab.certify(), ba.certify());
}

// ═══════════════════════════════════════════════════════════════════════════════
// 8. Refusals
// ═══════════════════════════════════════════════════════════════════════════════

/// The one place where the linking number is undefined, bracketed from both
/// sides: a hair inside, exactly on, and a hair outside the edge.
#[test]
fn a_strand_through_the_other_curve_is_refused_and_either_side_is_certified() {
    let a = square_a();
    let hair = 1.0 / 1024.0;
    assert_eq!(certified_lk(&a, &square_b(hair)).abs(), 1);
    assert_eq!(
        linking_number(&a, &square_b(0.0)),
        Err(LinkingError::Intersecting {
            segment_a: 0,
            segment_b: 0
        })
    );
    assert_eq!(lk(&a, &square_b(-hair)).certify(), LinkVerdict::ZeroLinking);

    // A shared vertex is an intersection too, and is not summed as zero.
    let mut b = square_b(-hair);
    b[0] = a[1];
    assert_eq!(
        linking_number(&a, &b),
        Err(LinkingError::Intersecting {
            segment_a: 0,
            segment_b: 0
        })
    );
}

#[test]
fn too_few_vertices_and_non_finite_coordinates_are_refused_everywhere() {
    let ring = circle(1.0, 16);
    let few = LinkingError::TooFewVertices { curve: 0, len: 2 };
    assert_eq!(linking_number(&ring[..2], &ring), Err(few));
    assert_eq!(
        linking_number(&ring, &[]),
        Err(LinkingError::TooFewVertices { curve: 1, len: 0 })
    );
    assert_eq!(writhe(&ring[..2]), Err(few));
    assert_eq!(knot_determinant(&ring[..2]), Err(few));

    for bad_value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut bad = ring.clone();
        bad[5][1] = bad_value;
        assert_eq!(
            linking_number(&ring, &bad),
            Err(LinkingError::NonFinite { curve: 1, vertex: 5 })
        );
        let first = LinkingError::NonFinite { curve: 0, vertex: 5 };
        assert_eq!(linking_number(&bad, &ring), Err(first));
        assert_eq!(writhe(&bad), Err(first));
        assert_eq!(knot_determinant(&bad), Err(first));
    }
}

/// Without the range check an overflowing squared length makes every
/// direction the zero vector, and a Hopf link sums to exactly 0: a proven-
/// looking zero that is false.
#[test]
fn out_of_range_differences_are_refused_rather_than_summed_as_zero() {
    let (a, b) = twisted_band(1, 1.0, 0.3, 64);
    for c in [2f64.powi(600), 2f64.powi(-600)] {
        let got = linking_number(&map_all(&a, |p| mul(p, c)), &map_all(&b, |p| mul(p, c)));
        assert!(
            matches!(got, Err(LinkingError::OutOfRange { .. })),
            "scale {c:e}: {got:?}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9. Writhe
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn writhe_is_real_valued_odd_under_reflection_and_refuses_self_intersection() {
    let w_flat = writhe(&circle(1.0, 256)).unwrap();
    assert!(w_flat.abs() < 1e-12, "planar circle writhe {w_flat}");

    let (a, _) = twisted_band(3, 1.0, 0.3, 300);
    let w = writhe(&a).unwrap();
    assert!(w.abs() > 0.1, "degenerate test: writhe {w}");
    assert!(
        (w - w.round()).abs() > 1e-6,
        "writhe {w} is suspiciously integral; it is not an invariant"
    );
    let reflected = writhe(&map_all(&a, |p| [p[0], p[1], -p[2]])).unwrap();
    assert!((w + reflected).abs() < 1e-10, "{w} vs reflected {reflected}");

    // Lemniscate of Gerono: crosses itself at the origin, off every vertex.
    let lemniscate: Vec<Vec3> = (0..64)
        .map(|k| {
            let t = TAU * (k as f64 + 0.5) / 64.0;
            [t.sin(), t.sin() * t.cos(), 0.0]
        })
        .collect();
    let got = writhe(&lemniscate);
    assert!(
        matches!(got, Err(LinkingError::Intersecting { .. })),
        "self-intersecting curve gave {got:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 10. Knot determinant
// ═══════════════════════════════════════════════════════════════════════════════

/// Classical values, from two generators that share no code: nerve's smooth
/// parametrisations, and closures of tangle's braid words.
#[test]
fn knot_determinant_matches_the_classical_table() {
    let cases: [(&str, Vec<Vec3>, u128); 5] = [
        ("unknot", circle(1.0, 120), 1),
        ("trefoil 3_1 = (2,3)", torus_knot(2, 3, 240), 3),
        ("figure-eight 4_1", figure_eight(240), 5),
        ("5_1 = (2,5)", torus_knot(2, 5, 240), 5),
        ("7_1 = (2,7)", torus_knot(2, 7, 240), 7),
    ];
    for (name, k, want) in cases {
        assert_eq!(knot_determinant(&k), Ok(want), "{name}");
    }

    for (name, word, strands, want) in [
        ("braid trefoil", &[1, 1, 1][..], 2, 3u128),
        ("braid figure-eight", &[1, -2, 1, -2][..], 3, 5),
    ] {
        let c = braid_closure(word, strands);
        assert_eq!(c.len(), 1, "{name} should close to one component");
        assert_eq!(knot_determinant(&c[0]), Ok(want), "{name}");
    }
}

#[test]
fn knot_determinant_is_projection_invariant_mirror_blind_and_incomplete() {
    let k = torus_knot(2, 3, 240);
    for (ax, ay, az) in [(0.7, 0.0, 0.0), (0.0, 1.9, 0.0), (0.0, 0.0, 2.6), (3.3, 0.4, -1.2)] {
        let r = map_all(&k, |p| rotate(p, ax, ay, az));
        assert_eq!(knot_determinant(&r), Ok(3), "rotation ({ax}, {ay}, {az})");
    }
    // The determinant cannot see chirality: the mirror trefoil gives 3 as well.
    let mirror = map_all(&k, |p| [p[0], p[1], -p[2]]);
    assert_eq!(knot_determinant(&mirror), Ok(3));
    // The ceiling: distinct knots 4_1 and 5_1 share determinant 5.
    assert_eq!(
        knot_determinant(&figure_eight(240)),
        knot_determinant(&torus_knot(2, 5, 240))
    );
}
