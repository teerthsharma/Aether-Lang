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
//!
//! # The answer, in two halves
//!
//! **Coordinates:** fine. Diagrams stay inside `2 * eps`, Betti numbers are
//! identical across every radius and seed tested. Bars move by about 2e-8.
//!
//! **Accumulation:** not fine at the sizes that matter. The kernel also rounds
//! every subtraction, square and partial sum, at roughly f32 epsilon regardless
//! of n — while a cloud of n points packs n(n-1)/2 distances into a bounded
//! range, so the smallest gap between two distinct distances shrinks
//! quadratically. The two cross between n=32 and n=64, and past that point two
//! distances can swap and the filtration combinatorics are no longer identical
//! by construction. `h0_only` admits 512 points.
//!
//! So the usable claim is the weak one: filtration values move by at most the
//! distance error, so bars shift by about 1e-7. Whether a bar appears or
//! disappears would have to be measured per cloud rather than derived.

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

// ═══════════════════════════════════════════════════════════════════════════════
// The other half: f32 accumulation, not just f32 inputs
//
// Everything above rounds the coordinates and then computes in f64. A real GPU
// path also rounds every subtraction, every square and every partial sum, and
// that error is not bounded by the coordinate displacement. These tests run the
// actual kernel.
// ═══════════════════════════════════════════════════════════════════════════════

use aether_gpu::{cpu_pairwise_sqdist, GpuContext};

fn context() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP: no usable GPU adapter ({e})");
            None
        }
    }
}

fn flat(points: &[ManifoldPoint<3>]) -> Vec<f32> {
    points
        .iter()
        .flat_map(|p| p.coords.iter().map(|c| *c as f32))
        .collect()
}

/// Exact pairwise distances in f64, the reference the kernel is measured
/// against.
fn exact_distances(points: &[ManifoldPoint<3>]) -> Vec<f64> {
    let n = points.len();
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let d: f64 = (0..3)
                .map(|k| (points[i].coords[k] - points[j].coords[k]).powi(2))
                .sum();
            out[i * n + j] = d.sqrt();
        }
    }
    out
}

/// Where the filtration ordering stops being guaranteed, measured.
///
/// The ordering is the sharper criterion for a Vietoris-Rips filtration, which
/// is determined by the order simplices enter rather than by their values: a
/// simplex enters at the maximum of its pairwise distances, and the reduction
/// processes them sorted. If every distance moves by less than half the smallest
/// gap between two distinct distances, no pair can swap and the combinatorics
/// are identical *by construction*.
///
/// **That guarantee does not survive growing n**, which is the finding this test
/// records rather than asserts away. The kernel's error is roughly constant at
/// f32 epsilon, but a cloud of n points has n(n-1)/2 distances packed into a
/// bounded range, so the smallest gap between distinct distances shrinks
/// quadratically. The two cross:
///
///     n=32   error 6.690e-8   smallest gap 3.857e-6   ratio 0.017   safe
///     n=64   error 1.099e-7   smallest gap 9.380e-9   ratio 11.7    not safe
///
/// At n=64 -- an eighth of the 512 points `h0_only` admits -- two distances can
/// already swap. So an f32 distance path cannot promise identical combinatorics
/// at the sizes this engine is configured for, and any claim that Betti numbers
/// are unchanged has to be measured on the specific cloud rather than derived.
///
/// What survives is the weaker bound, asserted below: the filtration values move
/// by at most the distance error, so the diagram moves by at most that in
/// bottleneck distance. Bars shift by 1e-7; whether a bar appears or disappears
/// is no longer guaranteed either way.
#[test]
fn the_filtration_ordering_guarantee_fails_as_the_cloud_grows() {
    let Some(ctx) = context() else { return };

    let mut safe_upto = 0usize;

    for (n, seed) in [(16usize, 0u64), (32, 1), (64, 2), (96, 3)] {
        let points = cloud(n, seed);
        let host = flat(&points);

        let g = ctx.upload(&host, n, 3).expect("upload");
        let sq = ctx
            .read(&ctx.pairwise_sqdist_resident(&g).expect("sqdist"))
            .expect("read");

        let exact = exact_distances(&points);

        let mut worst = 0.0f64;
        for i in 0..n * n {
            let got = (sq[i].max(0.0) as f64).sqrt();
            worst = worst.max((got - exact[i]).abs());
        }

        // Ties are excluded: equal distances cannot swap, they are equal.
        let mut sorted: Vec<f64> = exact.clone();
        sorted.sort_by(f64::total_cmp);
        let mut min_gap = f64::INFINITY;
        for w in sorted.windows(2) {
            let gap = w[1] - w[0];
            if gap > 1e-12 {
                min_gap = min_gap.min(gap);
            }
        }

        let ordering_safe = worst < min_gap / 2.0;
        if ordering_safe {
            safe_upto = n;
        }

        println!(
            "n={n:>3}: error {worst:.3e}  min gap {min_gap:.3e}  ratio {:>9.2e}  ordering {}",
            worst / min_gap,
            if ordering_safe {
                "guaranteed"
            } else {
                "NOT guaranteed"
            }
        );

        // The bound that does hold at every size: filtration values move by at
        // most the distance error, so the diagram moves by at most that.
        assert!(
            worst < 1e-6,
            "n={n}: distance error {worst:e} is larger than f32 arithmetic explains"
        );
    }

    // The guarantee holds somewhere and fails somewhere; pinning both ends stops
    // this from silently becoming vacuous if the kernel's precision changes.
    assert!(
        safe_upto >= 16,
        "the ordering guarantee failed even at the smallest cloud tested"
    );
    assert!(
        safe_upto < 96,
        "the ordering guarantee now holds at n=96; this test documents that it \
         does not, and the claim in aether-gpu FEATURES.md needs revisiting"
    );

    println!("ordering guaranteed up to n={safe_upto}, not beyond");
}

/// The kernel's error against the CPU reference this crate ships, as a plain
/// magnitude rather than a ratio.
///
/// Reported separately because the ordering test above passes or fails on a
/// property of the *data* -- how close together its distances happen to be --
/// and a reader needs the kernel's own error to judge whether that will hold
/// on a different cloud.
#[test]
fn the_distance_kernel_error_is_bounded_by_f32_epsilon() {
    let Some(ctx) = context() else { return };

    let n = 64;
    let points = cloud(n, 7);
    let host = flat(&points);

    let g = ctx.upload(&host, n, 3).expect("upload");
    let sq = ctx
        .read(&ctx.pairwise_sqdist_resident(&g).expect("sqdist"))
        .expect("read");
    let cpu_sq = cpu_pairwise_sqdist(&host, n, 3);

    // Against the f32 CPU reference: the two should agree to a few ulp, since
    // both accumulate three terms in f32 with the same associativity.
    let mut worst_vs_cpu = 0.0f32;
    for i in 0..n * n {
        worst_vs_cpu = worst_vs_cpu.max((sq[i] - cpu_sq[i]).abs());
    }

    // Against exact f64: this is the real precision cost of the kernel.
    let exact = exact_distances(&points);
    let mut worst_rel = 0.0f64;
    for i in 0..n * n {
        let got = (sq[i].max(0.0) as f64).sqrt();
        if exact[i] > 1e-6 {
            worst_rel = worst_rel.max((got - exact[i]).abs() / exact[i]);
        }
    }

    println!("kernel vs f32 CPU reference: {worst_vs_cpu:.3e} (squared units)");
    println!("kernel vs exact f64:         {worst_rel:.3e} relative");

    assert!(
        worst_vs_cpu < 1e-6,
        "kernel and CPU f32 reference disagree by {worst_vs_cpu:e}"
    );

    // A three-term f32 dot product plus a sqrt should land within a few times
    // f32 epsilon, which is 1.19e-7.
    assert!(
        worst_rel < 1e-5,
        "relative distance error {worst_rel:e} is larger than f32 arithmetic explains"
    );
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
