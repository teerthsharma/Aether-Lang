//! Does an f32 GPU path change the persistence diagram?
//!
//! This is the question standing between the GPU backend and the thing this
//! repository is actually about. `pairwise_sqdist` exists because the O(n^2 d)
//! distance computation dominates every Vietoris-Rips filtration in
//! `aether-core`, but the engine computes in f64 and WGSL has no f64, so
//! routing distances through the GPU means computing the filtration from f32
//! values. Whether that is acceptable is a topological question, not a
//! numerical one, and it has an answer the engine can be asked directly.
//!
//! # The prediction
//!
//! Rounding coordinates to f32 moves every point by at most half an f32 ulp.
//! The Cohen-Steiner-Edelsbrunner-Harer stability theorem, in the Rips form
//! this engine's own invariant suite asserts, bounds the resulting change:
//!
//!     d_B(Dgm(X), Dgm(X')) <= 2 * eps      where eps = max ||x - x'||
//!
//! So the diagrams must agree to within twice the rounding displacement. This
//! is not a tolerance chosen to make a test pass; it is the theorem the engine
//! is already tested against, applied to a perturbation whose size is known in
//! advance.
//!
//! A failure here would mean either that f32 is not viable for the filtration
//! or that the engine violates stability. Those are very different findings,
//! which is why the perturbation is measured rather than assumed.

use aether_core::diagram::bottleneck_distance;
use aether_core::manifold::ManifoldPoint;
use aether_core::persistence::{persistent_homology, ComplexKind, PersistenceConfig};

/// Deterministic cloud generator, f64 throughout.
fn cloud(n: usize, seed: u64) -> Vec<ManifoldPoint<3>> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };
    (0..n)
        .map(|_| ManifoldPoint::new([next(), next(), next()]))
        .collect()
}

/// A circle, the fixture the engine's own invariant suite uses for H1.
fn circle(n: usize, radius: f64) -> Vec<ManifoldPoint<3>> {
    (0..n)
        .map(|i| {
            let t = 2.0 * core::f64::consts::PI * i as f64 / n as f64;
            ManifoldPoint::new([radius * t.cos(), radius * t.sin(), 0.0])
        })
        .collect()
}

/// Round every coordinate through f32 and back, which is exactly what a GPU
/// path does to the inputs before it ever computes a distance.
fn through_f32(points: &[ManifoldPoint<3>]) -> Vec<ManifoldPoint<3>> {
    points
        .iter()
        .map(|p| {
            ManifoldPoint::new([
                p.coords[0] as f32 as f64,
                p.coords[1] as f32 as f64,
                p.coords[2] as f32 as f64,
            ])
        })
        .collect()
}

/// Largest displacement introduced by the rounding, which is the `eps` the
/// stability bound is stated in terms of.
fn max_displacement(a: &[ManifoldPoint<3>], b: &[ManifoldPoint<3>]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(p, q)| {
            let d: f64 = (0..3)
                .map(|k| (p.coords[k] - q.coords[k]).powi(2))
                .sum::<f64>();
            d.sqrt()
        })
        .fold(0.0, f64::max)
}

fn h0_config(max_points: usize) -> PersistenceConfig {
    PersistenceConfig {
        max_homology_dim: 0,
        max_points,
        max_simplices: 1_000_000,
        max_radius: f64::INFINITY,
        complex_kind: ComplexKind::VietorisRips,
    }
}

fn h1_config(max_points: usize) -> PersistenceConfig {
    PersistenceConfig {
        max_homology_dim: 1,
        max_points,
        max_simplices: 1_000_000,
        max_radius: f64::INFINITY,
        complex_kind: ComplexKind::VietorisRips,
    }
}

/// H0 on random clouds: the f32 diagram must sit inside the stability bound.
#[test]
fn f32_coordinates_move_the_h0_diagram_within_the_stability_bound() {
    for seed in [1u64, 2, 3, 5, 8] {
        let exact = cloud(64, seed);
        let rounded = through_f32(&exact);

        let eps = max_displacement(&exact, &rounded);
        let cfg = h0_config(128);

        let da = persistent_homology(&exact, cfg).expect("exact");
        let db = persistent_homology(&rounded, cfg).expect("rounded");

        let d = bottleneck_distance(&da, &db, 0);
        let bound = 2.0 * eps;

        assert!(
            d <= bound + 1e-15,
            "seed {seed}: bottleneck {d:e} exceeds the stability bound {bound:e} \
             for a displacement of {eps:e}"
        );
        println!("seed {seed}: eps {eps:.3e}, bottleneck {d:.3e}, bound {bound:.3e}");
    }
}

/// H1 on a circle, where there is a long bar with a known death time and a
/// perturbation therefore has something meaningful to disturb.
#[test]
fn f32_coordinates_move_the_h1_diagram_within_the_stability_bound() {
    for n in [12usize, 18, 24] {
        let exact = circle(n, 1.0);
        let rounded = through_f32(&exact);

        let eps = max_displacement(&exact, &rounded);
        let cfg = h1_config(64);

        let da = persistent_homology(&exact, cfg).expect("exact");
        let db = persistent_homology(&rounded, cfg).expect("rounded");

        let d = bottleneck_distance(&da, &db, 1);
        let bound = 2.0 * eps;

        assert!(
            d <= bound + 1e-15,
            "n={n}: H1 bottleneck {d:e} exceeds the stability bound {bound:e}"
        );
        println!("circle n={n}: eps {eps:.3e}, H1 bottleneck {d:.3e}, bound {bound:.3e}");
    }
}

/// The Betti numbers -- the discrete part, and the part the language's seal
/// loop terminates on -- must not change at all.
///
/// This is the assertion that decides whether an f32 path is usable for this
/// project specifically. Bar endpoints moving by 1e-7 is irrelevant to a
/// convergence rule that compares integers; a Betti number changing is not.
#[test]
fn f32_coordinates_do_not_change_the_betti_numbers() {
    for seed in [1u64, 2, 3, 5, 8, 13] {
        let exact = cloud(48, seed);
        let rounded = through_f32(&exact);
        let cfg = h1_config(64);

        let da = persistent_homology(&exact, cfg).expect("exact");
        let db = persistent_homology(&rounded, cfg).expect("rounded");

        // Sweep the filtration rather than checking one radius, since a
        // disagreement could hide between sample points.
        for step in 1..40 {
            let r = step as f64 * 0.05;
            let ba = da.betti_at(r);
            let bb = db.betti_at(r);
            assert_eq!(
                (ba.beta_0, ba.beta_1),
                (bb.beta_0, bb.beta_1),
                "seed {seed}: Betti numbers differ at radius {r}"
            );
        }
    }
    println!("Betti numbers identical across 39 radii, 6 seeds");
}

/// The same question for the shape the persistence engine is usually asked
/// about: does f32 rounding change how many long bars a circle has?
#[test]
fn f32_coordinates_preserve_the_circle_h1_bar_count() {
    for n in [12usize, 15, 18, 24] {
        let exact = circle(n, 1.0);
        let rounded = through_f32(&exact);
        let cfg = h1_config(64);

        let long = |d: &aether_core::persistence::PersistenceDiagram| -> usize {
            d.pairs
                .iter()
                .filter(|p| p.dimension == 1)
                .filter(|p| match p.death {
                    Some(dd) => dd - p.birth > 0.3,
                    None => true,
                })
                .count()
        };

        let da = persistent_homology(&exact, cfg).expect("exact");
        let db = persistent_homology(&rounded, cfg).expect("rounded");

        assert_eq!(
            long(&da),
            long(&db),
            "n={n}: f32 rounding changed the number of long H1 bars"
        );
    }
}
