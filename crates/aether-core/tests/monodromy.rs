//! Ground-truth tests for `aether_core::monodromy`.
//!
//! Every assertion is scored against an answer known in closed form, not against
//! a previously recorded output of the estimator. A decider bug rarely crashes:
//! it returns a plausible verdict. These are the verdicts a plausible-but-wrong
//! implementation gets wrong.
//!
//! Asserted here:
//!
//! 1. Injectivity. x ↦ x² collides across the fibre {p, −p}; the exp map on two
//!    periods collides with det DF = e^{2x} > 0 everywhere; injective controls
//!    are cleared; x ↦ x³ is cleared where it is étale and read as a collision at
//!    its critical point, with the witness showing which; short size ranges are
//!    undecided.
//! 2. Symmetry. Regular polygons recover C_n wherever they sit and at any scale;
//!    polygons are dihedral, pinwheels chiral; Gaussian clouds get no group; set
//!    defects vanish exactly on symmetries; the Holmes map commutes with −I and
//!    Hénon does not.
//! 3. Dimension. Line, square, circle and torus give 1, 2, 1, 2; the estimate is
//!    invariant under rigid motion, scaling and the weight α; the Sierpinski
//!    gasket is decided fractal, and a segment is too (the documented bias).
//! 4. Sensitivity. Logistic r = 4 gives ln 2; the period-2 orbit at r = 3.2 gives
//!    ½ ln 0.16; Hénon matches Sprott exponent by exponent; diagonal, orthogonal
//!    and scaled cocycles are exact.

use aether_core::manifold::ManifoldPoint;
use aether_core::monodromy::{
    apply_about_centroid, chamfer_defect, collision_certificate, equivariance_defect, free_ratio,
    is_fractal, kaplan_yorke_dimension, lower_ratio_witness, lyapunov_spectrum, map_spectrum,
    persistence_defect, ph_dimension, recover_cyclic, recover_dihedral, rotation, CollisionVerdict,
    FractalVerdict, MonodromyError, DEFAULT_GRID_STEP_DEG, MAX_N,
};

use core::f64::consts::{LN_2, TAU};

// ═══════════════════════════════════════════════════════════════════════════════
// Deterministic sampling
// ═══════════════════════════════════════════════════════════════════════════════

/// xorshift64*. Seeded per draw so a failure is reproducible from the seed alone.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Spread small consecutive seeds before the first step.
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
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

    /// Uniform in [lo, hi).
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// Standard normal via Box-Muller (one of the two variates; the other is dropped).
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Samplers: `(n, seed) -> cloud`, the protocol the estimators take
// ═══════════════════════════════════════════════════════════════════════════════

fn interval(lo: f64, hi: f64) -> impl Fn(usize, u64) -> Vec<ManifoldPoint<1>> {
    move |n, seed| {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| ManifoldPoint::new([rng.uniform(lo, hi)]))
            .collect()
    }
}

fn rect(x: (f64, f64), y: (f64, f64)) -> impl Fn(usize, u64) -> Vec<ManifoldPoint<2>> {
    move |n, seed| {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| ManifoldPoint::new([rng.uniform(x.0, x.1), rng.uniform(y.0, y.1)]))
            .collect()
    }
}

/// A unit segment in the plane, at an angle, so no coordinate is degenerate.
fn line(n: usize, seed: u64) -> Vec<ManifoldPoint<2>> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let t = rng.unit();
            ManifoldPoint::new([0.6 * t, 0.8 * t])
        })
        .collect()
}

fn circle(n: usize, seed: u64) -> Vec<ManifoldPoint<2>> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let t = TAU * rng.unit();
            ManifoldPoint::new([t.cos(), t.sin()])
        })
        .collect()
}

/// Torus of revolution in R³, radii 2 and 1, sampled uniformly in its angles.
fn torus(n: usize, seed: u64) -> Vec<ManifoldPoint<3>> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let (u, v) = (TAU * rng.unit(), TAU * rng.unit());
            let r = 2.0 + v.cos();
            ManifoldPoint::new([r * u.cos(), r * u.sin(), v.sin()])
        })
        .collect()
}

/// Sierpinski gasket by the chaos game. Hausdorff dimension log 3 / log 2.
fn sierpinski(n: usize, seed: u64) -> Vec<ManifoldPoint<2>> {
    let v = [[0.0, 0.0], [1.0, 0.0], [0.5, 3f64.sqrt() / 2.0]];
    let mut rng = Rng::new(seed);
    let mut p = [rng.unit(), rng.unit()];
    let mut out = Vec::with_capacity(n);
    for i in 0..n + 30 {
        let c = v[(rng.next_u64() % 3) as usize];
        p = [(p[0] + c[0]) / 2.0, (p[1] + c[1]) / 2.0];
        if i >= 30 {
            out.push(ManifoldPoint::new(p));
        }
    }
    out
}

/// The source's `polygon(n, per_edge)`: an outline sampled evenly along each edge.
fn polygon(n: usize, per_edge: usize) -> Vec<ManifoldPoint<2>> {
    let v: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let t = TAU * i as f64 / n as f64;
            [t.cos(), t.sin()]
        })
        .collect();
    let mut out = Vec::new();
    for i in 0..n {
        let (a, b) = (v[i], v[(i + 1) % n]);
        for k in 0..per_edge {
            let s = k as f64 / per_edge as f64;
            out.push(ManifoldPoint::new([
                a[0] + (b[0] - a[0]) * s,
                a[1] + (b[1] - a[1]) * s,
            ]));
        }
    }
    out
}

/// The source's chiral `pinwheel(k, per, skew)`: C_k rotations, no reflection.
fn pinwheel(k: usize, per: usize, skew: f64) -> Vec<ManifoldPoint<2>> {
    let mut out = Vec::new();
    for j in 0..k {
        let a = TAU * j as f64 / k as f64;
        for i in 0..per {
            let s = 0.25 + 0.75 * i as f64 / (per - 1) as f64;
            let t = a + skew * s;
            out.push(ManifoldPoint::new([s * t.cos(), s * t.sin()]));
        }
    }
    out
}

fn gaussian_cloud(n: usize, seed: u64) -> Vec<ManifoldPoint<2>> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| ManifoldPoint::new([rng.normal(), rng.normal()]))
        .collect()
}

/// Rotate by `theta`, scale by `c`, translate by `(dx, dy)`.
fn similarity(
    points: &[ManifoldPoint<2>],
    theta: f64,
    c: f64,
    dx: f64,
    dy: f64,
) -> Vec<ManifoldPoint<2>> {
    let (s, co) = theta.sin_cos();
    points
        .iter()
        .map(|p| {
            let [x, y] = p.coords;
            ManifoldPoint::new([c * (co * x - s * y) + dx, c * (s * x + co * y) + dy])
        })
        .collect()
}

/// The source's default size grid for the collision certificate.
const SIZES: [usize; 4] = [100, 200, 400, 800];

/// Size grid for the dimension estimator. The source's default stops at 800;
/// stopping at 400 keeps each estimate under a second through the Rips engine.
const DIM_SIZES: [usize; 4] = [50, 100, 200, 400];

// ═══════════════════════════════════════════════════════════════════════════════
// Maps with closed-form answers
// ═══════════════════════════════════════════════════════════════════════════════

/// (x, y) ↦ (eˣ cos y, eˣ sin y). det DF = e^{2x} > 0 everywhere, and
/// F(x, y) = F(x, y + 2π) identically.
fn exp_polar(p: &[f64; 2]) -> [f64; 2] {
    let r = p[0].exp();
    [r * p[1].cos(), r * p[1].sin()]
}

/// (x, y) ↦ (x, y + x²). A polynomial automorphism with det DF ≡ 1: the
/// Jacobian conjecture's hypothesis, satisfied by an injective map.
fn tame(p: &[f64; 2]) -> [f64; 2] {
    [p[0], p[1] + p[0] * p[0]]
}

fn henon(p: &[f64; 2]) -> [f64; 2] {
    [1.0 - 1.4 * p[0] * p[0] + p[1], 0.3 * p[0]]
}

fn henon_jacobian(p: &[f64; 2]) -> [[f64; 2]; 2] {
    [[-2.8 * p[0], 1.0], [0.3, 0.0]]
}

/// The Holmes cubic map. Every term is odd, so F(−p) = −F(p) identically.
fn holmes(p: &[f64; 2]) -> [f64; 2] {
    [p[1], -0.2 * p[0] + 2.77 * p[1] - p[1] * p[1] * p[1]]
}

fn logistic(r: f64) -> impl Fn(&[f64; 1]) -> [f64; 1] {
    move |x| [r * x[0] * (1.0 - x[0])]
}

fn logistic_jacobian(r: f64) -> impl Fn(&[f64; 1]) -> [[f64; 1]; 1] {
    move |x| [[r * (1.0 - 2.0 * x[0])]]
}

/// `n` orbit points of `step` after a burn-in, thinned by 10.
fn orbit(step: fn(&[f64; 2]) -> [f64; 2], n: usize) -> Vec<ManifoldPoint<2>> {
    let mut x = [0.1, 0.1];
    for _ in 0..1_000 {
        x = step(&x);
    }
    (0..n * 10)
        .filter_map(|i| {
            x = step(&x);
            (i % 10 == 0).then_some(ManifoldPoint::new(x))
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Injectivity
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn a_fold_collides_across_its_fibre_and_the_witness_is_the_fibre() {
    // x ↦ x² identifies p with −p. Ground truth first, with no estimation: two
    // distinct points with one image give λ = 0 exactly.
    let square = |x: &[f64; 1]| [x[0] * x[0]];
    let pair = [ManifoldPoint::new([0.3]), ManifoldPoint::new([-0.3])];
    assert_eq!(lower_ratio_witness(square, &pair).unwrap().lambda, 0.0);

    let c = collision_certificate(square, interval(-1.0, 1.0), &SIZES, 5, 0).unwrap();
    assert_eq!(c.verdict, CollisionVerdict::CollisionExhibited, "{c:?}");
    assert!(c.free_readable && c.free_ratio < 1e-3, "{c:?}");

    // The evidence is the pair, so check the pair: it straddles the fibre
    // (q ≈ −p) and is well separated in the domain, which is what distinguishes
    // a collision from two neighbouring samples.
    let (p, q) = (c.witness.p.coords[0], c.witness.q.coords[0]);
    assert!(
        (p + q).abs() < 1e-3,
        "witness {p}, {q} does not straddle the fibre"
    );
    assert!(
        c.domain_gap > 0.05,
        "witness {p}, {q} is adjacent, not a fibre"
    );
    assert!(c.image_gap < 1e-3 * c.domain_gap, "{c:?}");
}

#[test]
fn the_wrapped_exp_map_collides_although_its_determinant_never_vanishes() {
    // On y ∈ [0, 4π] the exp map covers the punctured plane twice. Every local
    // check passes, since det DF = e^{2x} > 0; two points one period apart
    // still share an image.
    let c = collision_certificate(exp_polar, rect((-1.0, 1.0), (0.0, 2.0 * TAU)), &SIZES, 5, 0)
        .unwrap();
    assert_eq!(c.verdict, CollisionVerdict::CollisionExhibited, "{c:?}");
    let (p, q) = (c.witness.p.coords, c.witness.q.coords);
    assert!(
        ((p[1] - q[1]).abs() - TAU).abs() < 0.05 && (p[0] - q[0]).abs() < 0.05,
        "witness {p:?}, {q:?} is not one period apart in y at equal radius"
    );
}

#[test]
fn injective_controls_are_cleared_and_the_ratio_is_scale_free() {
    // The same exp map on less than one period, the det ≡ 1 automorphism, and a
    // linear isomorphism. None collides, and no verdict can say "injective":
    // the enum has no such variant, so the one-sidedness is enforced by type.
    let wrap_free = collision_certificate(
        exp_polar,
        rect((-1.0, 1.0), (0.0, 0.75 * TAU)),
        &SIZES,
        5,
        0,
    )
    .unwrap();
    let automorphism =
        collision_certificate(tame, rect((-2.0, 2.0), (-2.0, 2.0)), &SIZES, 5, 0).unwrap();
    let linear = collision_certificate(
        |p: &[f64; 2]| [2.0 * p[0] + p[1], 3.0 * p[1]],
        rect((-2.0, 2.0), (-2.0, 2.0)),
        &SIZES,
        5,
        0,
    )
    .unwrap();
    for (name, c) in [
        ("exp on 0.75 periods", wrap_free),
        ("tame", automorphism),
        ("linear", linear),
    ] {
        assert_eq!(
            c.verdict,
            CollisionVerdict::NoCollisionAtThisSampling,
            "{name}: {c:?}"
        );
        assert!(
            c.free_ratio > 3.0 * 8.8e-3,
            "{name}: ρ_free {} sits near the level",
            c.free_ratio
        );
    }

    // ρ_free is a ratio of ratios, so a similarity of the codomain and a scaling
    // of the domain (with the map conjugated to match) leave it unchanged.
    let x = rect((-2.0, 2.0), (-2.0, 2.0))(500, 3);
    let base = free_ratio(tame, &x).unwrap();
    let moved = free_ratio(
        |p: &[f64; 2]| {
            let t = tame(&[p[0] / 7.0, p[1] / 7.0]);
            let (s, c) = 0.61f64.sin_cos();
            [
                0.2 * (c * t[0] - s * t[1]) + 9.0,
                0.2 * (s * t[0] + c * t[1]) - 4.0,
            ]
        },
        &similarity(&x, 0.0, 7.0, 0.0, 0.0),
    )
    .unwrap();
    assert!(
        (moved / base - 1.0).abs() < 1e-9,
        "ρ_free {base} became {moved}"
    );
}

#[test]
fn a_critical_point_reads_as_a_collision_and_the_witness_says_so() {
    // x ↦ x³ is injective on [−1, 1] and ramified at 0, where DF vanishes. Away
    // from 0 it is étale and is cleared.
    let cube = |x: &[f64; 1]| [x[0] * x[0] * x[0]];
    let etale = collision_certificate(cube, interval(0.25, 1.0), &SIZES, 5, 0).unwrap();
    assert_eq!(
        etale.verdict,
        CollisionVerdict::NoCollisionAtThisSampling,
        "{etale:?}"
    );

    // Across 0 the two-point ratio of neighbouring samples tends to 3x² → 0 just
    // as a fibre's does, and the Jacobian-free verdict fires. This is the
    // documented failure regime, pinned so it cannot be described away. The
    // witness separates the cases: adjacent points at the critical point, where
    // the fold's witness was a pair a whole fibre apart.
    let ramified = collision_certificate(cube, interval(-1.0, 1.0), &SIZES, 5, 0).unwrap();
    assert_eq!(
        ramified.verdict,
        CollisionVerdict::CollisionExhibited,
        "{ramified:?}"
    );
    let (p, q) = (ramified.witness.p.coords[0], ramified.witness.q.coords[0]);
    assert!(
        p.abs() < 0.05 && q.abs() < 0.05 && ramified.domain_gap < 0.05,
        "witness {p}, {q} should be two neighbours at the critical point"
    );
}

#[test]
fn short_size_ranges_and_degenerate_samples_are_refused() {
    let square = |x: &[f64; 1]| [x[0] * x[0]];
    // Three sizes spanning 4x straddled the slope cutoff in the source, so no
    // verdict in either direction.
    let short = collision_certificate(square, interval(-1.0, 1.0), &[100, 200, 400], 3, 0).unwrap();
    assert_eq!(short.verdict, CollisionVerdict::Undecided, "{short:?}");
    assert!(!short.size_range_adequate);

    let one = [ManifoldPoint::new([0.5])];
    assert_eq!(
        lower_ratio_witness(square, &one).unwrap_err(),
        MonodromyError::TooFewPoints { actual: 1, min: 2 }
    );
    let same = [ManifoldPoint::new([0.5]); 4];
    assert_eq!(
        free_ratio(square, &same).unwrap_err(),
        MonodromyError::ZeroDiameter
    );
    let wall = vec![ManifoldPoint::new([0.0]); MAX_N + 1];
    assert!(matches!(
        free_ratio(square, &wall),
        Err(MonodromyError::TooManyPoints { .. })
    ));
    assert_eq!(
        collision_certificate(square, interval(-1.0, 1.0), &[], 3, 0).unwrap_err(),
        MonodromyError::InvalidParameter
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Symmetry
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn regular_polygons_recover_their_order_wherever_they_sit() {
    for n in [3, 4, 5, 6, 8] {
        let r = recover_cyclic(&polygon(n, 8), 6.0).unwrap();
        assert_eq!(r.order, n, "C{n} outline: {r:?}");
        assert!(r.power_fraction >= 0.98, "C{n} outline: {r:?}");
    }
    // The action is about the centroid and every gate is a ratio of distances,
    // so a rigid motion and a scaling of the cloud cannot move the answer.
    let moved = similarity(&polygon(6, 8), 0.3, 7.0, 5.0, -3.0);
    assert_eq!(recover_cyclic(&moved, 6.0).unwrap().order, 6);
    // The shipped 2° grid, where an argmax-bin reader returned 1 for a hexagon.
    let fine = recover_cyclic(&polygon(6, 6), DEFAULT_GRID_STEP_DEG).unwrap();
    assert_eq!(fine.order, 6, "{fine:?}");
}

#[test]
fn mirrors_are_found_on_a_polygon_and_absent_on_a_pinwheel() {
    // A hexagon rotated by 1°: its rotations stay on a 6° grid while its
    // mirrors fall between grid points, which is why the minima are refined.
    let hexagon = similarity(&polygon(6, 6), 1f64.to_radians(), 1.0, 0.0, 0.0);
    let d6 = recover_dihedral(&hexagon, 6.0).unwrap();
    assert!(d6.order == 6 && d6.dihedral, "hexagon: {d6:?}");

    let wheel = pinwheel(5, 12, 0.35);
    let c5 = recover_dihedral(&wheel, DEFAULT_GRID_STEP_DEG).unwrap();
    assert!(c5.order == 5 && !c5.dihedral, "pinwheel: {c5:?}");
    assert!(c5.depth_ratio.unwrap() < 0.9, "pinwheel: {c5:?}");
}

#[test]
fn clouds_without_symmetry_are_assigned_no_group() {
    for seed in 0..12 {
        let r = recover_cyclic(&gaussian_cloud(60, seed), DEFAULT_GRID_STEP_DEG).unwrap();
        assert_eq!(r.order, 1, "Gaussian cloud {seed}: {r:?}");
    }
    for seed in 100..103 {
        let r = recover_dihedral(&gaussian_cloud(48, seed), 6.0).unwrap();
        assert!(!r.dihedral && r.order == 1, "Gaussian cloud {seed}: {r:?}");
    }
    // Refusals: a single point, and a cloud whose points all coincide, would
    // otherwise score the defect 0 that means "exact symmetry" at every angle.
    let one = [ManifoldPoint::new([1.0, 2.0])];
    assert!(matches!(
        recover_cyclic(&one, 6.0),
        Err(MonodromyError::TooFewPoints { .. })
    ));
    let same = [ManifoldPoint::new([1.0, 2.0]); 5];
    assert_eq!(
        recover_dihedral(&same, 6.0).unwrap_err(),
        MonodromyError::ZeroDiameter
    );
    let bad = [
        ManifoldPoint::new([0.0, f64::NAN]),
        ManifoldPoint::new([1.0, 0.0]),
    ];
    assert_eq!(
        recover_cyclic(&bad, 6.0).unwrap_err(),
        MonodromyError::NonFinite
    );
    assert_eq!(
        recover_cyclic(&polygon(4, 4), 120.0).unwrap_err(),
        MonodromyError::InvalidParameter
    );
}

#[test]
fn set_defects_vanish_exactly_on_symmetries() {
    let square = [
        ManifoldPoint::new([0.0, 0.0]),
        ManifoldPoint::new([1.0, 0.0]),
        ManifoldPoint::new([1.0, 1.0]),
        ManifoldPoint::new([0.0, 1.0]),
    ];
    for (angle, symmetric) in [(90.0, true), (180.0, true), (17.0, false), (45.0, false)] {
        let g = apply_about_centroid(&square, &rotation(angle)).unwrap();
        let ch = chamfer_defect(&square, &g).unwrap();
        let ph = persistence_defect(&square, &g).unwrap();
        assert_eq!(ch < 1e-6, symmetric, "chamfer {ch} at {angle}°");
        assert_eq!(ph < 1e-6, symmetric, "persistence {ph} at {angle}°");
    }

    // Containment is not equality: a strict superset must score, in both
    // argument orders, or the zero set is wrong.
    let tri = [
        ManifoldPoint::new([0.0, 0.0]),
        ManifoldPoint::new([1.0, 0.0]),
        ManifoldPoint::new([0.0, 1.0]),
    ];
    let bigger = [tri[0], tri[1], tri[2], ManifoldPoint::new([10.0, 10.0])];
    assert!(chamfer_defect(&tri, &bigger).unwrap() > 1.0);
    assert_eq!(
        chamfer_defect(&tri, &bigger).unwrap(),
        chamfer_defect(&bigger, &tri).unwrap()
    );

    // A collinear cloud has no H1, so degree 1 alone reads 0 for every map; the
    // maximum over degrees must still see a translation through H0.
    let segment: Vec<_> = (0..20)
        .map(|i| ManifoldPoint::new([i as f64 / 19.0, 0.0]))
        .collect();
    let far: Vec<_> = segment
        .iter()
        .map(|p| ManifoldPoint::new([p.coords[0] + 100.0, 0.0]))
        .collect();
    assert!(persistence_defect(&segment, &far).unwrap() > 1.0);
}

#[test]
fn only_the_odd_map_commutes_with_the_point_reflection() {
    // Holmes is odd term by term, so F(−p) = −F(p) holds in floating point, not
    // merely approximately. Hénon has no symmetry; a 90° rotation is not one of
    // Holmes's. The source measured 0, 2.61 and 2.92 on attractors of diameter
    // 2.56.
    let minus = [[-1.0, 0.0], [0.0, -1.0]];
    let quarter = rotation(90.0);
    let holmes_orbit = orbit(holmes, 200);
    let henon_orbit = orbit(henon, 200);
    assert_eq!(
        equivariance_defect(holmes, &minus, &holmes_orbit).unwrap(),
        0.0
    );
    assert!(equivariance_defect(henon, &minus, &henon_orbit).unwrap() > 1.0);
    assert!(equivariance_defect(holmes, &quarter, &holmes_orbit).unwrap() > 1.0);
    assert!(matches!(
        equivariance_defect(holmes, &minus, &[]),
        Err(MonodromyError::TooFewPoints { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Dimension
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn flat_sets_have_their_integer_dimension() {
    // The bias is downward (the source's square reads 1.9259), so the interval
    // must reach the truth from below rather than straddle it symmetrically.
    let l = ph_dimension(line, 1.0, &DIM_SIZES, 3, 0).unwrap();
    let s = ph_dimension(rect((0.0, 1.0), (0.0, 1.0)), 1.0, &DIM_SIZES, 3, 0).unwrap();
    assert!((l.dimension - 1.0).abs() < 0.1, "line: {l:?}");
    assert!((s.dimension - 2.0).abs() < 0.15, "square: {s:?}");
    assert!(
        s.dimension > l.dimension + 0.7,
        "line {l:?} vs square {s:?}"
    );
    assert_eq!((l.n_obs, l.n_clamped), (12, 0));
}

#[test]
fn curved_manifolds_have_their_intrinsic_dimension() {
    // Dimension is intrinsic: a circle is 1-dimensional in the plane and a torus
    // 2-dimensional in space, whatever the ambient dimension.
    let c = ph_dimension(circle, 1.0, &DIM_SIZES, 3, 0).unwrap();
    let t = ph_dimension(torus, 1.0, &DIM_SIZES, 3, 0).unwrap();
    assert!((c.dimension - 1.0).abs() < 0.1, "circle: {c:?}");
    assert!((t.dimension - 2.0).abs() < 0.2, "torus: {t:?}");
}

#[test]
fn dimension_is_invariant_under_rigid_motion_scale_and_weight() {
    // A similarity multiplies every bar by c, so T_α by c^α, which shifts the
    // intercept and leaves the slope untouched.
    let square = rect((0.0, 1.0), (0.0, 1.0));
    let base = ph_dimension(&square, 1.0, &DIM_SIZES, 3, 0).unwrap();
    let moved = ph_dimension(
        |n, seed| similarity(&square(n, seed), 0.77, 37.0, -4.5, 17.25),
        1.0,
        &DIM_SIZES,
        3,
        0,
    )
    .unwrap();
    assert!(
        (moved.dimension - base.dimension).abs() < 1e-9,
        "{} became {} under a similarity",
        base.dimension,
        moved.dimension
    );

    // d = α / (1 − s) inverts the scaling law, so the estimate must not move
    // with α. Every test at α = 1 alone cannot see a formula that drops α: it
    // would return d/α, about 4 at α = 1/2.
    let half = ph_dimension(&square, 0.5, &DIM_SIZES, 3, 0).unwrap();
    assert!((half.dimension - 2.0).abs() < 0.3, "α = 0.5: {half:?}");
    assert!(
        (half.dimension - base.dimension).abs() < 0.35,
        "{half:?} vs {base:?}"
    );
}

#[test]
fn the_fractal_predicate_is_decided_on_the_interval() {
    // Sierpinski: log 3 / log 2 = 1.585 against a topological dimension of 1.
    let (verdict, fit) = is_fractal(sierpinski, 1.0, 1.0, &DIM_SIZES, 3, 0).unwrap();
    assert_eq!(verdict, FractalVerdict::Fractal, "{fit:?}");
    assert!(
        (fit.dimension - 3f64.ln() / 2f64.ln()).abs() < 0.15,
        "{fit:?}"
    );
    // A square against its own topological dimension: the downward bias keeps
    // the whole interval at or below 2.
    let (verdict, fit) =
        is_fractal(rect((0.0, 1.0), (0.0, 1.0)), 2.0, 1.0, &DIM_SIZES, 3, 0).unwrap();
    assert_eq!(verdict, FractalVerdict::NotFractal, "{fit:?}");

    // The documented failure regime, pinned so it cannot be described away. The
    // interval models sampling noise, not bias. The expected MST length of n
    // uniform points on a unit segment is 1 − 2/(n + 1), so the fitted slope is
    // positive, d̂ sits just above 1, and at this size grid the interval excludes
    // 1: the predicate calls a segment fractal against d_top = 1.
    let (verdict, fit) = is_fractal(line, 1.0, 1.0, &DIM_SIZES, 3, 0).unwrap();
    assert!(fit.dimension > 1.0 && fit.ci95.0 > 1.0, "{fit:?}");
    assert_eq!(verdict, FractalVerdict::Fractal, "{fit:?}");

    // Refusals. A finite dataset that caps every draw at one size has no
    // scaling to fit; a cloud of coincident points has no bars; α must be
    // positive.
    let capped = |_: usize, seed: u64| circle(60, seed);
    assert_eq!(
        ph_dimension(capped, 1.0, &DIM_SIZES, 2, 0).unwrap_err(),
        MonodromyError::NoScaling
    );
    let collapsed = |n: usize, _: u64| vec![ManifoldPoint::new([0.5, 0.5]); n];
    assert_eq!(
        ph_dimension(collapsed, 1.0, &DIM_SIZES, 2, 0).unwrap_err(),
        MonodromyError::NotEnoughSamples
    );
    assert_eq!(
        ph_dimension(circle, 0.0, &DIM_SIZES, 2, 0).unwrap_err(),
        MonodromyError::InvalidParameter
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Sensitivity
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn the_logistic_map_at_r4_has_exponent_ln_2() {
    // Conjugate to the doubling map, so λ = ln 2 exactly for almost every orbit.
    let [lambda] = map_spectrum(
        logistic(4.0),
        logistic_jacobian(4.0),
        [0.123_456],
        200_000,
        1_000,
    )
    .unwrap();
    assert!((lambda - LN_2).abs() < 0.01, "λ = {lambda}, ln 2 = {LN_2}");
}

#[test]
fn a_periodic_orbit_has_its_closed_form_negative_exponent() {
    // At r = 3.2 the attractor is a period-2 cycle whose multiplier is
    // f'(x₁) f'(x₂) = 4 + 2r − r² = 0.16, so λ = ½ ln 0.16 < 0.
    let r = 3.2;
    let [lambda] = map_spectrum(logistic(r), logistic_jacobian(r), [0.3], 20_000, 1_000).unwrap();
    let exact = 0.5 * (4.0f64 + 2.0 * r - r * r).abs().ln();
    assert!((lambda - exact).abs() < 1e-3, "λ = {lambda}, exact {exact}");
    assert!(lambda < 0.0);
}

#[test]
fn henon_exponents_match_sprott_one_by_one_not_just_in_sum() {
    // The trace identity Σλ = ln|det J| = ln 0.3 survives an implementation that
    // never carries the QR frame forward, so it is asserted only as the control;
    // the exponents are checked individually against Sprott's +0.41922 and
    // −1.62319, and D_KY against his 1.25827.
    let spectrum = map_spectrum(henon, henon_jacobian, [0.1, 0.1], 200_000, 10_000).unwrap();
    assert!(
        (spectrum.iter().sum::<f64>() - 0.3f64.ln()).abs() < 1e-9,
        "{spectrum:?}"
    );
    assert!((spectrum[0] - 0.41922).abs() < 0.01, "{spectrum:?}");
    assert!((spectrum[1] + 1.62319).abs() < 0.01, "{spectrum:?}");
    let d_ky = kaplan_yorke_dimension(spectrum).unwrap();
    assert!((d_ky - 1.25827).abs() < 0.01, "D_KY = {d_ky}");
}

#[test]
fn closed_form_cocycles_are_reproduced_exactly() {
    // A constant diagonal cocycle returns the log of its diagonal, an orthogonal
    // one returns zero, and scaling every Jacobian by c shifts every exponent by
    // ln c. These hold at any length, so they pin the arithmetic, not the limit.
    let diagonal =
        lyapunov_spectrum(core::iter::repeat_n([[3.0, 0.0], [0.0, 0.5]], 50), 1.0).unwrap();
    assert!((diagonal[0] - 3f64.ln()).abs() < 1e-12 && (diagonal[1] - 0.5f64.ln()).abs() < 1e-12);

    let turn = rotation(37.0);
    let orthogonal = lyapunov_spectrum(core::iter::repeat_n(turn, 50), 1.0).unwrap();
    assert!(orthogonal.iter().all(|l| l.abs() < 1e-12), "{orthogonal:?}");

    let base = |k: usize| [[1.0 + 0.1 * (k % 7) as f64, 0.3], [-0.2, 0.9]];
    let plain = lyapunov_spectrum((0..400).map(base), 1.0).unwrap();
    let scaled = lyapunov_spectrum(
        (0..400).map(|k| base(k).map(|row| row.map(|v| 2.5 * v))),
        1.0,
    )
    .unwrap();
    for (a, b) in plain.iter().zip(&scaled) {
        assert!(
            (b - a - 2.5f64.ln()).abs() < 1e-12,
            "{plain:?} vs {scaled:?}"
        );
    }
    // dt rescales to per-unit-time exponents.
    let halved =
        lyapunov_spectrum(core::iter::repeat_n([[3.0, 0.0], [0.0, 0.5]], 50), 2.0).unwrap();
    assert!((halved[0] - 0.5 * 3f64.ln()).abs() < 1e-12);

    // Kaplan–Yorke at its edges, and the refusals.
    assert_eq!(kaplan_yorke_dimension([-0.1, -2.0]).unwrap(), 0.0);
    assert_eq!(kaplan_yorke_dimension([0.5, 0.2]).unwrap(), 2.0);
    assert!((kaplan_yorke_dimension([-1.0, 0.5]).unwrap() - 1.5).abs() < 1e-15);
    assert_eq!(
        kaplan_yorke_dimension([f64::NAN, -1.0]).unwrap_err(),
        MonodromyError::NonFinite
    );
    assert!(matches!(
        lyapunov_spectrum(core::iter::empty::<[[f64; 2]; 2]>(), 1.0),
        Err(MonodromyError::TooFewPoints { .. })
    ));
    assert_eq!(
        lyapunov_spectrum(core::iter::repeat_n(turn, 3), 0.0).unwrap_err(),
        MonodromyError::InvalidParameter
    );
}
