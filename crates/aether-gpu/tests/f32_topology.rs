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
use aether_core::persistence::{
    persistent_homology, persistent_homology_from_distances, ComplexKind, PersistenceConfig,
};

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

/// Displace every point by a random vector of exactly `delta`, which perturbs
/// pairwise distances by at most `2 * delta` and typically by about `delta`.
fn perturb(points: &[ManifoldPoint<3>], delta: f64, seed: u64) -> Vec<ManifoldPoint<3>> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };

    points
        .iter()
        .map(|p| {
            let (a, b, c) = (next(), next(), next());
            let norm = (a * a + b * b + c * c).sqrt().max(1e-300);
            ManifoldPoint::new([
                p.coords[0] + delta * a / norm,
                p.coords[1] + delta * b / norm,
                p.coords[2] + delta * c / norm,
            ])
        })
        .collect()
}

/// How often a distance perturbation the size of the kernel's error actually
/// changes a Betti number.
///
/// The ordering analysis says the combinatorics stop being guaranteed identical
/// past about n=32. That is a statement about the worst case: two distances
/// *can* swap. It says nothing about whether a swap changes the homology, and
/// most swaps do not — exchanging the entry order of two simplices that are not
/// both pivotal leaves the barcode alone except for endpoint shifts.
///
/// This measures the practical rate directly. Points are displaced by the
/// measured kernel error, which moves distances by a comparable amount, and the
/// Betti numbers are compared across the whole filtration.
///
/// The distinction matters because "not guaranteed" and "observed to break" are
/// different findings, and only one of them blocks using the kernel.
#[test]
fn how_often_a_kernel_sized_perturbation_changes_the_betti_numbers() {
    // The relative error measured from the kernel, scaled to the cloud's extent
    // of roughly 1 unit.
    const DELTA: f64 = 4e-7;

    let betti_differs = |a: &aether_core::persistence::PersistenceDiagram,
                         b: &aether_core::persistence::PersistenceDiagram,
                         dim: usize|
     -> bool {
        (1..60).any(|step| {
            let r = step as f64 * 0.03;
            let ba = a.betti_at(r);
            let bb = b.betti_at(r);
            if dim == 0 {
                ba.beta_0 != bb.beta_0
            } else {
                ba.beta_0 != bb.beta_0 || ba.beta_1 != bb.beta_1
            }
        })
    };

    let mut checked = 0usize;
    let mut changed = 0usize;

    for (n, dim, cfg, seeds) in [
        (32usize, 1usize, h1_config(128), 50u64),
        (64, 1, h1_config(128), 50),
        (128, 0, h0_config(512), 50),
        (256, 0, h0_config(512), 30),
    ] {
        let mut n_changed = 0usize;

        for seed in 0..seeds {
            let exact = cloud(n, 100 + seed);
            let moved = perturb(&exact, DELTA, 900 + seed);

            let da = persistent_homology(&exact, cfg).expect("exact");
            let db = persistent_homology(&moved, cfg).expect("perturbed");

            checked += 1;
            if betti_differs(&da, &db, dim) {
                changed += 1;
                n_changed += 1;
            }
        }

        println!("n={n:>3} H{dim}: {n_changed}/{seeds} clouds changed a Betti number");
    }

    println!("total at {DELTA:.0e}: {changed}/{checked} clouds affected");

    // With zero events in `checked` trials the rule of three puts the 95%
    // upper bound at 3/checked, which is what this number is worth: not "it
    // never happens" but "below this rate".
    if changed == 0 {
        println!(
            "  95% upper bound on the rate: {:.1}% (rule of three)",
            300.0 / checked as f64
        );
    }

    assert!(checked >= 150, "only {checked} clouds were compared");
}

/// Where a perturbation *does* start changing the topology, and how far that is
/// from the error the kernel actually produces.
///
/// The previous test reports a rate at one displacement. This finds the
/// displacement at which the rate stops being zero, which is the number that
/// says whether the kernel has margin or is merely lucky.
#[test]
fn the_perturbation_a_diagram_survives_has_margin_over_the_kernel_error() {
    let cfg = h1_config(128);
    let n = 64;
    let seeds = 20u64;

    let betti_differs = |a: &aether_core::persistence::PersistenceDiagram,
                         b: &aether_core::persistence::PersistenceDiagram|
     -> bool {
        (1..60).any(|step| {
            let r = step as f64 * 0.03;
            let ba = a.betti_at(r);
            let bb = b.betti_at(r);
            ba.beta_0 != bb.beta_0 || ba.beta_1 != bb.beta_1
        })
    };

    let mut largest_clean = 0.0f64;

    for exp in [-7i32, -6, -5, -4, -3, -2] {
        let delta = 10f64.powi(exp);
        let mut n_changed = 0usize;

        for seed in 0..seeds {
            let exact = cloud(n, 100 + seed);
            let moved = perturb(&exact, delta, 900 + seed);

            let da = persistent_homology(&exact, cfg).expect("exact");
            let db = persistent_homology(&moved, cfg).expect("perturbed");

            if betti_differs(&da, &db) {
                n_changed += 1;
            }
        }

        println!("displacement {delta:.0e}: {n_changed}/{seeds} clouds changed");
        if n_changed == 0 {
            largest_clean = delta;
        }
    }

    // The kernel's measured relative error, on a cloud of extent ~1.
    const KERNEL_ERROR: f64 = 4e-7;
    let margin = largest_clean / KERNEL_ERROR;

    println!(
        "largest displacement with no observed change: {largest_clean:.0e}, \
         kernel error {KERNEL_ERROR:.0e}, margin {margin:.0}x"
    );

    assert!(
        largest_clean >= KERNEL_ERROR,
        "the topology changes at displacements at or below the kernel's own error"
    );
}

/// The distance-matrix entry point must agree exactly with the point-based one.
///
/// It exists so a filtration can be built from distances the engine did not
/// compute. Its first obligation is to compute the same thing as the path it
/// generalises, on the same input — otherwise every result obtained through it
/// describes a different engine.
#[test]
fn the_distance_matrix_path_reproduces_the_point_path_exactly() {
    for (n, cfg) in [
        (24usize, h1_config(64)),
        (40, h1_config(64)),
        (64, h0_config(128)),
    ] {
        let points = cloud(n, 55);
        let exact = exact_distances(&points);

        let from_points = persistent_homology(&points, cfg).expect("points");
        let from_matrix = persistent_homology_from_distances(&exact, n, cfg).expect("matrix");

        assert_eq!(
            from_points.pairs.len(),
            from_matrix.pairs.len(),
            "n={n}: different number of pairs"
        );
        for (a, b) in from_points.pairs.iter().zip(&from_matrix.pairs) {
            assert_eq!(a.dimension, b.dimension, "n={n}: dimension");
            assert!((a.birth - b.birth).abs() < 1e-15, "n={n}: birth");
            match (a.death, b.death) {
                (Some(x), Some(y)) => assert!((x - y).abs() < 1e-15, "n={n}: death"),
                (None, None) => {}
                _ => panic!("n={n}: one bar is essential and the other is not"),
            }
        }
    }
    println!("distance-matrix path matches the point path bar for bar");
}

/// A matrix that is not a distance matrix must be rejected, not filtered.
///
/// An asymmetric matrix still produces a barcode: the reduction runs, bars come
/// out, and nothing indicates that the filtration was never a filtration. That
/// is the failure worth guarding, and it is why the entry point validates
/// rather than trusting its caller.
#[test]
fn the_distance_matrix_path_rejects_matrices_that_are_not_metrics() {
    let cfg = h0_config(64);
    let n = 4;

    let mut asymmetric = vec![0.0; n * n];
    asymmetric[1] = 1.0;
    asymmetric[n] = 2.0;
    assert!(persistent_homology_from_distances(&asymmetric, n, cfg).is_err());

    let mut nonzero_diagonal = vec![1.0; n * n];
    for i in 0..n {
        nonzero_diagonal[i * n + i] = 0.5;
    }
    assert!(persistent_homology_from_distances(&nonzero_diagonal, n, cfg).is_err());

    let mut negative = vec![0.0; n * n];
    negative[1] = -1.0;
    negative[n] = -1.0;
    assert!(persistent_homology_from_distances(&negative, n, cfg).is_err());

    let mut not_finite = vec![0.0; n * n];
    not_finite[1] = f64::NAN;
    not_finite[n] = f64::NAN;
    assert!(persistent_homology_from_distances(&not_finite, n, cfg).is_err());

    assert!(persistent_homology_from_distances(&[0.0; 9], 4, cfg).is_err());
}

/// End to end: a diagram computed from the GPU kernel's own f32 distances,
/// against one computed from exact f64 distances.
///
/// Every earlier result here was a proxy. The coordinate test rounded inputs and
/// computed in f64; the perturbation study displaced points to *model* the
/// kernel's error. This runs `pairwise_sqdist`, takes the square root, and hands
/// the result to the persistence engine.
#[test]
fn a_diagram_built_from_gpu_distances_matches_one_built_from_exact_distances() {
    let Some(ctx) = context() else { return };

    for (n, dim, cfg) in [
        (32usize, 1usize, h1_config(64)),
        (64, 1, h1_config(128)),
        (128, 0, h0_config(256)),
    ] {
        let points = cloud(n, 77);
        let host = flat(&points);

        let g = ctx.upload(&host, n, 3).expect("upload");
        let sq = ctx
            .read(&ctx.pairwise_sqdist_resident(&g).expect("sqdist"))
            .expect("read");

        // The kernel returns squared distances; the filtration wants distances.
        //
        // Taken directly, with no symmetrisation. An earlier version averaged
        // d(i,j) with d(j,i) on the assumption that f32 rounding made them
        // differ in the last bit. It does not: IEEE-754 subtraction is exactly
        // antisymmetric, so `a - b` and `b - a` differ only in sign bit, their
        // squares are bitwise identical, and the kernel accumulates both orders
        // over the same range. `the_distance_matrix_is_symmetric_with_a_zero_diagonal`
        // now asserts that bitwise rather than to a tolerance.
        //
        // The averaging was harmless and wrong: it hid the property instead of
        // relying on it, and would have masked a genuine indexing bug.
        let gpu: Vec<f64> = sq.iter().map(|v| (v.max(0.0) as f64).sqrt()).collect();

        let exact = exact_distances(&points);

        let da = persistent_homology_from_distances(&exact, n, cfg).expect("exact");
        let db = persistent_homology_from_distances(&gpu, n, cfg).expect("gpu");

        let bottleneck = bottleneck_distance(&da, &db, dim);

        let mut worst = 0.0f64;
        for i in 0..n * n {
            worst = worst.max((gpu[i] - exact[i]).abs());
        }

        // Perturbing every filtration value by at most `worst` moves the
        // diagram by at most `worst` in bottleneck distance.
        assert!(
            bottleneck <= worst + 1e-12,
            "n={n}: bottleneck {bottleneck:e} exceeds the distance error {worst:e}"
        );

        let mut betti_same = true;
        for step in 1..60 {
            let r = step as f64 * 0.03;
            let ba = da.betti_at(r);
            let bb = db.betti_at(r);
            if ba.beta_0 != bb.beta_0 || (dim == 1 && ba.beta_1 != bb.beta_1) {
                betti_same = false;
                break;
            }
        }

        println!(
            "n={n:>3} H{dim}: distance error {worst:.3e}, bottleneck {bottleneck:.3e}, \
             Betti {}",
            if betti_same { "identical" } else { "DIFFERENT" }
        );

        assert!(
            betti_same,
            "n={n}: GPU distances changed a Betti number end to end"
        );
    }
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
