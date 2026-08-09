//! Tests for `aether_core::diagram`: the metrics and vectorizations that turn a
//! persistence diagram into something a model can consume.
//!
//! Written before the module exists. Every assertion is a property the metric or
//! the feature map must satisfy by definition, not a value scraped from a run.

use aether_core::diagram::{
    bottleneck_distance, persistence_image, persistence_landscape, wasserstein_distance,
    ImageConfig, LandscapeConfig,
};
use aether_core::manifold::ManifoldPoint;
use aether_core::persistence::{
    persistent_homology, PersistenceConfig, PersistenceDiagram, PersistencePair,
};

fn finite(pairs: &[(usize, f64, f64)]) -> PersistenceDiagram {
    PersistenceDiagram::new(
        pairs
            .iter()
            .map(|&(dimension, birth, death)| PersistencePair {
                dimension,
                birth,
                death: Some(death),
            })
            .collect(),
    )
}

fn circle(n: usize, r: f64) -> Vec<ManifoldPoint<2>> {
    (0..n)
        .map(|i| {
            let t = core::f64::consts::TAU * i as f64 / n as f64;
            ManifoldPoint::new([r * t.cos(), r * t.sin()])
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Metric axioms
//
// Bottleneck and Wasserstein must be metrics on the space of diagrams. If any of
// these fails, downstream distances, kernels, and clusterings are meaningless
// even though they will still produce numbers.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bottleneck_of_a_diagram_with_itself_is_zero() {
    let d = finite(&[(1, 0.1, 0.9), (1, 0.2, 0.35), (1, 0.0, 1.4)]);
    assert_eq!(bottleneck_distance(&d, &d, 1), 0.0);
}

#[test]
fn bottleneck_is_symmetric() {
    let a = finite(&[(1, 0.1, 0.9), (1, 0.2, 0.35)]);
    let b = finite(&[(1, 0.15, 1.0)]);
    let ab = bottleneck_distance(&a, &b, 1);
    let ba = bottleneck_distance(&b, &a, 1);
    assert!(
        (ab - ba).abs() < 1e-12,
        "d(a,b) = {ab} but d(b,a) = {ba}; a metric is symmetric"
    );
}

#[test]
fn bottleneck_satisfies_the_triangle_inequality() {
    let a = finite(&[(1, 0.0, 1.0)]);
    let b = finite(&[(1, 0.3, 0.9)]);
    let c = finite(&[(1, 0.5, 2.0), (1, 0.1, 0.2)]);

    let ac = bottleneck_distance(&a, &c, 1);
    let ab = bottleneck_distance(&a, &b, 1);
    let bc = bottleneck_distance(&b, &c, 1);
    assert!(
        ac <= ab + bc + 1e-12,
        "triangle inequality violated: d(a,c) = {ac} > {ab} + {bc}"
    );
}

#[test]
fn bottleneck_matches_a_hand_computed_pairing() {
    // One bar each. The L-infinity distance between (0.0, 1.0) and (0.2, 1.1) is
    // max(0.2, 0.1) = 0.2. Matching both to the diagonal would cost
    // max(0.5, 0.45) = 0.5, so the optimal matching pairs them.
    let a = finite(&[(1, 0.0, 1.0)]);
    let b = finite(&[(1, 0.2, 1.1)]);
    assert!((bottleneck_distance(&a, &b, 1) - 0.2).abs() < 1e-12);
}

#[test]
fn bottleneck_projects_an_unmatched_bar_to_the_diagonal() {
    // An extra short bar in `b` with persistence 0.4 costs 0.2 to kill against the
    // diagonal, which is cheaper than distorting the matched pair.
    let a = finite(&[(1, 0.0, 1.0)]);
    let b = finite(&[(1, 0.0, 1.0), (1, 0.5, 0.9)]);
    assert!(
        (bottleneck_distance(&a, &b, 1) - 0.2).abs() < 1e-12,
        "expected the diagonal projection cost 0.2, got {}",
        bottleneck_distance(&a, &b, 1)
    );
}

#[test]
fn wasserstein_sums_where_bottleneck_takes_a_maximum() {
    // Two bars, each displaced by 0.2. Bottleneck reports the max (0.2);
    // 1-Wasserstein reports the sum (0.4). A implementation that returns the same
    // number for both has conflated them.
    let a = finite(&[(1, 0.0, 1.0), (1, 2.0, 3.0)]);
    let b = finite(&[(1, 0.2, 1.2), (1, 2.2, 3.2)]);

    let bottleneck = bottleneck_distance(&a, &b, 1);
    let w1 = wasserstein_distance(&a, &b, 1, 1.0);

    assert!(
        (bottleneck - 0.2).abs() < 1e-12,
        "bottleneck was {bottleneck}"
    );
    assert!((w1 - 0.4).abs() < 1e-9, "1-Wasserstein was {w1}");
}

#[test]
fn wasserstein_is_at_least_bottleneck() {
    // For p >= 1 the p-Wasserstein distance dominates the bottleneck distance,
    // which is its p -> infinity limit.
    let a = finite(&[(1, 0.0, 1.0), (1, 0.4, 0.6), (1, 1.0, 2.5)]);
    let b = finite(&[(1, 0.1, 1.3), (1, 1.1, 2.0)]);

    let bottleneck = bottleneck_distance(&a, &b, 1);
    for p in [1.0, 2.0, 4.0] {
        let w = wasserstein_distance(&a, &b, 1, p);
        assert!(
            w >= bottleneck - 1e-9,
            "p = {p}: Wasserstein {w} fell below bottleneck {bottleneck}"
        );
    }
}

#[test]
fn distances_respect_the_stability_theorem_on_real_diagrams() {
    // The library metric must reproduce the bound the invariant suite asserts:
    // perturbing the cloud by at most eps moves the diagram by at most 2*eps.
    let base = circle(14, 1.0);
    let eps = 0.02;
    let moved: Vec<_> = base
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let theta = 0.7 * i as f64;
            ManifoldPoint::new([
                p.coords[0] + eps * theta.cos(),
                p.coords[1] + eps * theta.sin(),
            ])
        })
        .collect();

    let config = PersistenceConfig::h1_dense();
    let a = persistent_homology(&base, config).unwrap();
    let b = persistent_homology(&moved, config).unwrap();

    for dim in 0..=1 {
        let d = bottleneck_distance(&a, &b, dim);
        assert!(
            d <= 2.0 * eps + 1e-9,
            "H{dim}: bottleneck {d} exceeds the 2*eps bound {}",
            2.0 * eps
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Persistence landscapes
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn a_single_bar_gives_the_expected_tent_function() {
    // The first landscape of one bar [b, d] is the tent
    // lambda(t) = max(0, min(t - b, d - t)), peaking at (b + d) / 2 with height
    // (d - b) / 2. Here [0, 2] peaks at t = 1 with height 1.
    let d = finite(&[(1, 0.0, 2.0)]);
    let config = LandscapeConfig {
        levels: 1,
        resolution: 5,
        min_t: 0.0,
        max_t: 2.0,
    };
    let l = persistence_landscape(&d, 1, config);

    assert_eq!(l.len(), 1, "one level requested");
    assert_eq!(l[0].len(), 5, "five sample points requested");
    // t = 0.0, 0.5, 1.0, 1.5, 2.0
    let expected = [0.0, 0.5, 1.0, 0.5, 0.0];
    for (i, (got, want)) in l[0].iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-12,
            "sample {i}: got {got}, want {want}"
        );
    }
}

#[test]
fn landscape_levels_are_ordered_pointwise() {
    // By construction lambda_1(t) >= lambda_2(t) >= ... at every t.
    //
    // The bars must CROSS, or this test is vacuous: with nested bars the tent
    // values already arrive in descending order and an implementation that skips
    // the per-sample sort passes anyway. (It did — that mutant survived the first
    // version of this test, which used nested bars.)
    //
    // Here bar A = [0.0, 1.2] and bar B = [0.4, 3.0] cross:
    //   at t = 0.6, A = 0.6 and B = 0.2  (A leads)
    //   at t = 1.0, A = 0.2 and B = 0.6  (B leads)
    let d = finite(&[(1, 0.0, 1.2), (1, 0.4, 3.0), (1, 0.8, 2.0)]);
    let config = LandscapeConfig {
        levels: 3,
        resolution: 31,
        min_t: 0.0,
        max_t: 3.0,
    };
    let l = persistence_landscape(&d, 1, config);

    for t in 0..31 {
        for level in 1..3 {
            assert!(
                l[level - 1][t] >= l[level][t] - 1e-12,
                "at sample {t}, level {} ({}) < level {} ({})",
                level,
                l[level - 1][t],
                level + 1,
                l[level][t]
            );
        }
    }
}

#[test]
fn landscape_takes_the_kth_largest_tent_where_bars_cross() {
    // The pointwise-ordering test proves the levels are sorted; this one proves
    // they hold the right values. Bars A = [0.0, 1.2] and B = [0.4, 3.0].
    //
    // At t = 1.0:  A = max(0, min(1.0, 0.2)) = 0.2
    //              B = max(0, min(0.6, 2.0)) = 0.6
    // so lambda_1(1.0) = 0.6 and lambda_2(1.0) = 0.2 — B leads, though A is the
    // earlier bar.
    let d = finite(&[(1, 0.0, 1.2), (1, 0.4, 3.0)]);
    let config = LandscapeConfig {
        levels: 2,
        resolution: 4,
        min_t: 0.0,
        max_t: 3.0, // samples at t = 0.0, 1.0, 2.0, 3.0
    };
    let l = persistence_landscape(&d, 1, config);

    assert!(
        (l[0][1] - 0.6).abs() < 1e-12,
        "lambda_1(1.0) should be 0.6 (bar B), got {}",
        l[0][1]
    );
    assert!(
        (l[1][1] - 0.2).abs() < 1e-12,
        "lambda_2(1.0) should be 0.2 (bar A), got {}",
        l[1][1]
    );
}

#[test]
fn landscape_is_one_lipschitz_in_the_bottleneck_distance() {
    // The stability theorem for landscapes: the sup-norm between two landscapes is
    // at most the bottleneck distance between their diagrams. This is what makes
    // the landscape a usable ML feature rather than a lossy sketch.
    let a = finite(&[(1, 0.0, 2.0), (1, 0.5, 1.2)]);
    let b = finite(&[(1, 0.1, 2.1), (1, 0.55, 1.1)]);

    let config = LandscapeConfig {
        levels: 2,
        resolution: 65,
        min_t: 0.0,
        max_t: 2.5,
    };
    let la = persistence_landscape(&a, 1, config);
    let lb = persistence_landscape(&b, 1, config);

    let sup = la
        .iter()
        .zip(lb.iter())
        .flat_map(|(x, y)| x.iter().zip(y.iter()).map(|(p, q)| (p - q).abs()))
        .fold(0.0f64, f64::max);

    let bottleneck = bottleneck_distance(&a, &b, 1);
    assert!(
        sup <= bottleneck + 1e-9,
        "landscape sup-norm {sup} exceeds bottleneck {bottleneck}; not 1-Lipschitz"
    );
}

#[test]
fn an_empty_diagram_gives_a_zero_landscape() {
    let d = finite(&[]);
    let config = LandscapeConfig {
        levels: 2,
        resolution: 8,
        min_t: 0.0,
        max_t: 1.0,
    };
    let l = persistence_landscape(&d, 1, config);
    assert!(l.iter().flatten().all(|&v| v == 0.0), "got {l:?}");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Persistence images
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn persistence_image_has_the_requested_shape_and_is_nonnegative() {
    let d = finite(&[(1, 0.2, 1.4), (1, 0.5, 0.7)]);
    let config = ImageConfig {
        width: 8,
        height: 6,
        sigma: 0.1,
        min_birth: 0.0,
        max_birth: 2.0,
        max_persistence: 2.0,
    };
    let image = persistence_image(&d, 1, config);

    assert_eq!(image.len(), 6 * 8, "height * width entries");
    assert!(image.iter().all(|&v| v >= 0.0), "a density is nonnegative");
    assert!(
        image.iter().sum::<f64>() > 0.0,
        "two bars must deposit mass"
    );
}

#[test]
fn persistence_image_weights_long_bars_more_than_short_ones() {
    // The standard weighting is linear in persistence, so noise near the diagonal
    // contributes little. Without it the image is dominated by sampling artifacts.
    let long = finite(&[(1, 0.5, 1.9)]);
    let short = finite(&[(1, 0.5, 0.55)]);
    let config = ImageConfig {
        width: 12,
        height: 12,
        sigma: 0.15,
        min_birth: 0.0,
        max_birth: 2.0,
        max_persistence: 2.0,
    };

    let long_mass: f64 = persistence_image(&long, 1, config).iter().sum();
    let short_mass: f64 = persistence_image(&short, 1, config).iter().sum();
    assert!(
        long_mass > short_mass * 5.0,
        "long bar mass {long_mass} should dominate short bar mass {short_mass}"
    );
}

#[test]
fn sigma_controls_the_kernel_width() {
    // Without this, an implementation that hardcodes the Gaussian width passes
    // every other image test — shape, non-negativity, weighting and translation
    // equivariance all hold for any fixed sigma. (That mutant survived until this
    // test existed.)
    //
    // A narrow kernel concentrates the bar's mass into few pixels; a wide one
    // spreads it. Measure concentration as the share of total mass in the single
    // brightest pixel.
    let d = finite(&[(1, 0.9, 1.9)]);
    let base = ImageConfig {
        width: 24,
        height: 24,
        sigma: 0.0,
        min_birth: 0.0,
        max_birth: 2.0,
        max_persistence: 2.0,
    };

    let concentration = |sigma: f64| {
        let image = persistence_image(&d, 1, ImageConfig { sigma, ..base });
        let total: f64 = image.iter().sum();
        let peak = image.iter().copied().fold(0.0f64, f64::max);
        assert!(total > 0.0, "sigma {sigma} deposited no mass");
        peak / total
    };

    let narrow = concentration(0.05);
    let wide = concentration(0.6);
    assert!(
        narrow > wide * 3.0,
        "sigma = 0.05 concentration {narrow:.4} should far exceed sigma = 0.6 \
         concentration {wide:.4}; sigma is not affecting the kernel"
    );
}

#[test]
fn persistence_image_is_translation_equivariant_in_birth() {
    // Shifting every bar's birth by a constant and shifting the window by the same
    // constant must give the identical image. Catches an absolute coordinate leak.
    let a = finite(&[(1, 0.3, 1.1), (1, 0.6, 0.9)]);
    let shift = 0.4;
    let b = finite(&[(1, 0.3 + shift, 1.1 + shift), (1, 0.6 + shift, 0.9 + shift)]);

    let config_a = ImageConfig {
        width: 10,
        height: 10,
        sigma: 0.12,
        min_birth: 0.0,
        max_birth: 2.0,
        max_persistence: 1.5,
    };
    let config_b = ImageConfig {
        min_birth: shift,
        max_birth: 2.0 + shift,
        ..config_a
    };

    let ia = persistence_image(&a, 1, config_a);
    let ib = persistence_image(&b, 1, config_b);
    for (i, (x, y)) in ia.iter().zip(ib.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-12,
            "pixel {i}: {x} vs {y} under a birth shift of {shift}"
        );
    }
}
