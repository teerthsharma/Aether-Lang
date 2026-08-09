//! Scale tests for `aether_core::persistence`.
//!
//! Written before the implementation that makes them pass. Two claims:
//!
//!   1. The engine handles point counts that topological ML actually uses, not
//!      just the 32 the original `PersistenceConfig` capped at.
//!   2. Correctness *improves* with sample density rather than degrading. The
//!      Rips circle theorem says H₁ dies at exactly √3·r; discretization error is
//!      O(1/n), so a denser circle must land closer to the constant. A test that
//!      only asserts "it finished" would pass on an implementation that returns
//!      garbage faster.
//!
//! Claim 2 is the load-bearing one. Making a wrong implementation fast is the
//! most expensive kind of wrong.

use aether_core::manifold::ManifoldPoint;
use aether_core::persistence::{persistent_homology, ComplexKind, PersistenceConfig};

fn circle(n: usize, r: f64) -> Vec<ManifoldPoint<2>> {
    (0..n)
        .map(|i| {
            let t = core::f64::consts::TAU * i as f64 / n as f64;
            ManifoldPoint::new([r * t.cos(), r * t.sin()])
        })
        .collect()
}

fn config(max_dim: usize, max_points: usize, max_simplices: usize) -> PersistenceConfig {
    PersistenceConfig {
        max_homology_dim: max_dim,
        max_points,
        max_simplices,
        max_radius: f64::INFINITY,
        complex_kind: ComplexKind::VietorisRips,
    }
}

/// Longest finite bar in a dimension, as (birth, death).
fn longest_bar(
    diagram: &aether_core::persistence::PersistenceDiagram,
    dim: usize,
) -> Option<(f64, f64)> {
    diagram
        .pairs
        .iter()
        .filter(|p| p.dimension == dim)
        .filter_map(|p| p.death.map(|d| (p.birth, d)))
        .max_by(|a, b| (a.1 - a.0).total_cmp(&(b.1 - b.0)))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Claim 1: the engine runs past 32 points
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn h0_handles_five_hundred_points() {
    // H0 only needs vertices and edges: 500 points is 125,250 edges. Any
    // implementation that scans linearly for faces makes this quadratic in the
    // simplex count and will not finish.
    let points = circle(500, 1.0);
    let diagram = persistent_homology(&points, config(0, 512, 200_000)).unwrap();

    // 500 points on a circle: one component survives, 499 merge.
    assert_eq!(
        diagram.pairs.iter().filter(|p| p.death.is_none()).count(),
        1,
        "expected exactly one essential H0 class"
    );
    assert_eq!(
        diagram
            .pairs
            .iter()
            .filter(|p| p.dimension == 0 && p.death.is_some())
            .count(),
        499,
        "expected 499 finite H0 bars"
    );
}

#[test]
fn h1_handles_one_hundred_points() {
    // 100 points with max_dim 1 means 4,950 edges and 161,700 triangles.
    let points = circle(100, 1.0);
    let diagram = persistent_homology(&points, config(1, 128, 400_000)).unwrap();

    let long = diagram
        .pairs
        .iter()
        .filter(|p| p.dimension == 1)
        .filter(|p| p.death.map(|d| d - p.birth > 0.5).unwrap_or(false))
        .count();
    assert_eq!(long, 1, "expected exactly one long H1 bar, found {long}");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Claim 2: density improves accuracy — the assertion that a fast wrong
// implementation cannot satisfy
// ═══════════════════════════════════════════════════════════════════════════════

/// Exact Rips death time of the H₁ class of a regular `n`-gon of radius `r`.
///
/// The cycle dies when the first triangle spanning roughly a third of the polygon
/// enters, so the death is the chord subtending `ceil(n/3)` steps:
///
/// ```text
/// death(n, r) = 2 r sin(pi * ceil(n/3) / n)
/// ```
///
/// For `n` divisible by 3 this is exactly `sqrt(3) * r`; otherwise it exceeds
/// `sqrt(3) * r` and decreases toward it as `n` grows. The continuous-circle
/// constant `sqrt(3) * r` is the limit, not the value at finite `n`.
fn ngon_h1_death(n: usize, r: f64) -> f64 {
    let k = n.div_ceil(3);
    2.0 * r * (core::f64::consts::PI * k as f64 / n as f64).sin()
}

#[test]
fn circle_h1_dies_at_the_exact_regular_polygon_chord() {
    // The sharp form of the ground-truth test. A 5%-tolerance assertion passes on
    // an implementation that is systematically off; this one does not.
    for n in [9usize, 10, 11, 12, 13, 17, 24, 48] {
        let points = circle(n, 1.0);
        let diagram = persistent_homology(&points, config(1, 128, 400_000)).unwrap();
        let (birth, death) = longest_bar(&diagram, 1).expect("no finite H1 bar");

        let expected = ngon_h1_death(n, 1.0);
        assert!(
            (death - expected).abs() < 1e-12,
            "n = {n}: H1 bar [{birth}, {death}) should die at {expected} \
             (2 r sin(pi * ceil(n/3) / n)); off by {:e}",
            (death - expected).abs()
        );
    }
}

#[test]
fn circle_h1_death_converges_to_sqrt3_from_above() {
    // The limit statement, tested on the sequence where it is not already exact:
    // n not divisible by 3 must approach sqrt(3) monotonically from above.
    let limit = 3.0f64.sqrt();
    let mut previous_error = f64::INFINITY;

    for n in [10usize, 13, 25, 49, 97] {
        let points = circle(n, 1.0);
        let diagram = persistent_homology(&points, config(1, 128, 400_000)).unwrap();
        let (_, death) = longest_bar(&diagram, 1).expect("no finite H1 bar");

        assert!(
            death >= limit - 1e-12,
            "n = {n}: death {death} fell below the sqrt(3) limit {limit}"
        );
        let error = (death - limit) / limit;
        assert!(
            error < previous_error,
            "n = {n}: relative error {error:.6} did not improve on {previous_error:.6}"
        );

        // The rate, not just the direction. For n = 3m + j the death angle is
        // pi/3 * (1 + O(1/n)), so the relative error decays as O(1/n); the
        // leading term is 2*pi/(3*sqrt(3)*n) ~= 1.21/n. Assert 2/n, which leaves
        // margin without being vacuous: at n = 25 the true error is 0.0503
        // against a bound of 0.08.
        assert!(
            error < 2.0 / n as f64,
            "n = {n}: relative error {error:.6} exceeds the O(1/n) bound {:.6}",
            2.0 / n as f64
        );
        previous_error = error;
    }
}

#[test]
fn dense_sampling_does_not_manufacture_extra_loops() {
    // The failure mode of a broken face lookup: missing faces make columns short,
    // which manufactures spurious cycles. Those multiply with n, so a dense circle
    // is the sharpest place to catch it.
    let points = circle(96, 1.0);
    let diagram = persistent_homology(&points, config(1, 128, 400_000)).unwrap();

    let long = diagram
        .pairs
        .iter()
        .filter(|p| p.dimension == 1)
        .filter(|p| p.death.map(|d| d - p.birth > 0.3).unwrap_or(false))
        .count();
    assert_eq!(
        long, 1,
        "a densely sampled circle has one loop; found {long} bars with persistence > 0.3"
    );

    // And no essential H1: the loop closes.
    assert_eq!(
        diagram
            .pairs
            .iter()
            .filter(|p| p.dimension == 1 && p.death.is_none())
            .count(),
        0,
        "H1 must not survive to infinity on a finite Rips complex"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// The caps still have to bite
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn simplex_cap_still_fails_fast_rather_than_exhausting_memory() {
    use aether_core::persistence::PersistenceError;

    let points = circle(64, 1.0);
    assert_eq!(
        persistent_homology(&points, config(2, 128, 1_000)),
        Err(PersistenceError::TooManySimplices { max: 1_000 }),
        "raising the point cap must not remove the simplex guard"
    );
}

#[test]
fn point_cap_is_still_enforced_when_configured() {
    use aether_core::persistence::PersistenceError;

    let points = circle(64, 1.0);
    assert_eq!(
        persistent_homology(&points, config(1, 16, 400_000)),
        Err(PersistenceError::TooManyPoints {
            actual: 64,
            max: 16
        })
    );
}
