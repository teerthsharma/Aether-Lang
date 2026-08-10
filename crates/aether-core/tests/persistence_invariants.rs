//! Invariant tests for `aether_core::persistence`.
//!
//! Each test is one theorem turned into an executable assertion. A persistent
//! homology bug does not crash and rarely looks wrong: it returns a plausible
//! diagram. These properties are what a plausible-but-wrong diagram violates.
//!
//! Theorems asserted here:
//!   1. Permutation invariance  — a diagram is a multiset over a point *set*.
//!   2. Isometry invariance     — rotation/translation preserve pairwise distance.
//!   3. Scale equivariance      — scaling by c scales every birth/death by c.
//!   4. Stability (CSEH)        — ‖perturbation‖∞ ≤ ε ⇒ bottleneck ≤ 2ε for Rips.
//!   5. Ground truth            — circle S¹ has one H₁ bar dying at √3·r.
//!   6. Negative control        — a Gaussian blob has no long H₁ bar.
//!   7. Elder rule              — H₀ matches an independent union-find.

use aether_core::manifold::ManifoldPoint;
use aether_core::persistence::{
    persistent_homology, ComplexKind, PersistenceConfig, PersistenceDiagram, PersistencePair,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Deterministic sampling
// ═══════════════════════════════════════════════════════════════════════════════

/// xorshift64*. Seeded per test so a failure is reproducible from the seed alone.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
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

    /// Uniform in [-1, 1).
    fn signed(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }

    /// Standard normal via Box-Muller (one of the two variates; the other is dropped).
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }

    /// Fisher-Yates permutation of `0..n`.
    fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            p.swap(i, j);
        }
        p
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Point cloud generators
// ═══════════════════════════════════════════════════════════════════════════════

/// `n` points evenly spaced on a circle of radius `r`, with optional jitter.
fn circle(n: usize, r: f64, jitter: f64, rng: &mut Rng) -> Vec<ManifoldPoint<2>> {
    (0..n)
        .map(|i| {
            let t = core::f64::consts::TAU * i as f64 / n as f64;
            ManifoldPoint::new([
                r * t.cos() + jitter * rng.signed(),
                r * t.sin() + jitter * rng.signed(),
            ])
        })
        .collect()
}

/// An isotropic Gaussian blob. The negative control: this has no topology.
fn blob(n: usize, sigma: f64, rng: &mut Rng) -> Vec<ManifoldPoint<2>> {
    (0..n)
        .map(|_| ManifoldPoint::new([sigma * rng.normal(), sigma * rng.normal()]))
        .collect()
}

/// `k` tight clusters whose centers are `separation` apart along a line.
fn clusters(k: usize, per: usize, separation: f64, rng: &mut Rng) -> Vec<ManifoldPoint<2>> {
    let mut out = Vec::with_capacity(k * per);
    for c in 0..k {
        for _ in 0..per {
            out.push(ManifoldPoint::new([
                c as f64 * separation + 0.02 * rng.signed(),
                0.02 * rng.signed(),
            ]));
        }
    }
    out
}

fn rotate_translate(
    points: &[ManifoldPoint<2>],
    theta: f64,
    dx: f64,
    dy: f64,
) -> Vec<ManifoldPoint<2>> {
    let (s, c) = theta.sin_cos();
    points
        .iter()
        .map(|p| {
            let [x, y] = p.coords;
            ManifoldPoint::new([c * x - s * y + dx, s * x + c * y + dy])
        })
        .collect()
}

fn scale(points: &[ManifoldPoint<2>], factor: f64) -> Vec<ManifoldPoint<2>> {
    points
        .iter()
        .map(|p| ManifoldPoint::new([p.coords[0] * factor, p.coords[1] * factor]))
        .collect()
}

/// Perturb every point by at most `eps` in Euclidean norm (rejection-free: sample
/// a direction and a radius in [0, eps]).
fn perturb(points: &[ManifoldPoint<2>], eps: f64, rng: &mut Rng) -> Vec<ManifoldPoint<2>> {
    points
        .iter()
        .map(|p| {
            let theta = core::f64::consts::TAU * rng.unit();
            let rho = eps * rng.unit().sqrt();
            ManifoldPoint::new([
                p.coords[0] + rho * theta.cos(),
                p.coords[1] + rho * theta.sin(),
            ])
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Diagram comparison
// ═══════════════════════════════════════════════════════════════════════════════

fn config(max_dim: usize, max_points: usize) -> PersistenceConfig {
    PersistenceConfig {
        max_homology_dim: max_dim,
        max_points,
        max_simplices: 200_000,
        max_radius: f64::INFINITY,
        complex_kind: ComplexKind::VietorisRips,
    }
}

fn finite_in_dim(diagram: &PersistenceDiagram, dim: usize) -> Vec<(f64, f64)> {
    let mut v: Vec<(f64, f64)> = diagram
        .pairs
        .iter()
        .filter(|p| p.dimension == dim)
        .filter_map(|p| p.death.map(|d| (p.birth, d)))
        .collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    v
}

fn essential_in_dim(diagram: &PersistenceDiagram, dim: usize) -> Vec<f64> {
    let mut v: Vec<f64> = diagram
        .pairs
        .iter()
        .filter(|p| p.dimension == dim && p.death.is_none())
        .map(|p| p.birth)
        .collect();
    v.sort_by(f64::total_cmp);
    v
}

/// Longest-lived finite bar in a dimension, as (birth, death, persistence).
fn longest_bar(diagram: &PersistenceDiagram, dim: usize) -> Option<(f64, f64, f64)> {
    finite_in_dim(diagram, dim)
        .into_iter()
        .map(|(b, d)| (b, d, d - b))
        .max_by(|a, b| a.2.total_cmp(&b.2))
}

/// Bars in a dimension whose persistence exceeds `threshold`.
fn long_bars(diagram: &PersistenceDiagram, dim: usize, threshold: f64) -> usize {
    finite_in_dim(diagram, dim)
        .iter()
        .filter(|(b, d)| d - b > threshold)
        .count()
}

/// Exact bottleneck distance between the finite parts of two diagrams in one
/// dimension, plus a hard requirement that the essential (infinite) classes agree
/// in count. Computed the textbook way: binary search over the candidate cost set,
/// feasibility by Kuhn's augmenting-path matching on the threshold graph.
///
/// Costs follow the standard construction on `n + m` nodes per side:
///   real_i  ↔ real_j   = L∞ between the two points of the diagram
///   real_i  ↔ diagonal = (death_i - birth_i) / 2
///   diagonal ↔ diagonal = 0
fn bottleneck(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    let (n, m) = (a.len(), b.len());
    if n == 0 && m == 0 {
        return 0.0;
    }
    let size = n + m;

    // cost[i][j], i indexes (a-reals then diagonal-slots-for-b), j the mirror.
    let mut cost = vec![vec![0.0f64; size]; size];
    for i in 0..size {
        for j in 0..size {
            cost[i][j] = match (i < n, j < m) {
                (true, true) => {
                    let (b0, d0) = a[i];
                    let (b1, d1) = b[j];
                    (b0 - b1).abs().max((d0 - d1).abs())
                }
                (true, false) => (a[i].1 - a[i].0) / 2.0,
                (false, true) => (b[j].1 - b[j].0) / 2.0,
                (false, false) => 0.0,
            };
        }
    }

    let mut candidates: Vec<f64> = cost.iter().flatten().copied().collect();
    candidates.sort_by(f64::total_cmp);
    candidates.dedup();

    // Binary search for the smallest threshold admitting a perfect matching.
    let (mut lo, mut hi) = (0usize, candidates.len() - 1);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if perfect_matching_exists(&cost, candidates[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    candidates[lo]
}

/// Kuhn's algorithm on the graph of edges with `cost <= threshold`.
fn perfect_matching_exists(cost: &[Vec<f64>], threshold: f64) -> bool {
    let size = cost.len();
    let mut match_right: Vec<Option<usize>> = vec![None; size];

    fn try_augment(
        left: usize,
        cost: &[Vec<f64>],
        threshold: f64,
        seen: &mut [bool],
        match_right: &mut [Option<usize>],
    ) -> bool {
        for right in 0..cost.len() {
            if seen[right] || cost[left][right] > threshold {
                continue;
            }
            seen[right] = true;
            let free = match match_right[right] {
                None => true,
                Some(other) => try_augment(other, cost, threshold, seen, match_right),
            };
            if free {
                match_right[right] = Some(left);
                return true;
            }
        }
        false
    }

    for left in 0..size {
        let mut seen = vec![false; size];
        if !try_augment(left, cost, threshold, &mut seen, &mut match_right) {
            return false;
        }
    }
    true
}

/// Bottleneck across every dimension up to `max_dim`, with essential-class counts
/// required to match exactly (an implementation that drops an essential class has
/// a real defect that a finite-only comparison would hide).
fn diagram_distance(a: &PersistenceDiagram, b: &PersistenceDiagram, max_dim: usize) -> f64 {
    let mut worst: f64 = 0.0;
    for dim in 0..=max_dim {
        let ea = essential_in_dim(a, dim);
        let eb = essential_in_dim(b, dim);
        assert_eq!(
            ea.len(),
            eb.len(),
            "essential class count differs in H{dim}: {ea:?} vs {eb:?}"
        );
        for (x, y) in ea.iter().zip(eb.iter()) {
            worst = worst.max((x - y).abs());
        }
        worst = worst.max(bottleneck(&finite_in_dim(a, dim), &finite_in_dim(b, dim)));
    }
    worst
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Permutation invariance
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn diagram_is_invariant_under_input_permutation() {
    // A persistence diagram is a multiset over a point SET. Row order is not data.
    // A violation means insertion order leaks into the filtration ordering or the
    // tie-break in `compare_simplex`.
    for seed in [1u64, 7, 42, 1337, 90210] {
        let mut rng = Rng::new(seed);
        let points = circle(14, 1.0, 0.05, &mut rng);

        let perm = rng.permutation(points.len());
        let shuffled: Vec<_> = perm.iter().map(|&i| points[i]).collect();

        let base = persistent_homology(&points, config(1, 32)).unwrap();
        let permuted = persistent_homology(&shuffled, config(1, 32)).unwrap();

        for dim in 0..=1 {
            let (x, y) = (finite_in_dim(&base, dim), finite_in_dim(&permuted, dim));
            assert_eq!(
                x.len(),
                y.len(),
                "seed {seed}: H{dim} bar count changed under permutation"
            );
            for (p, q) in x.iter().zip(y.iter()) {
                assert!(
                    (p.0 - q.0).abs() < 1e-9 && (p.1 - q.1).abs() < 1e-9,
                    "seed {seed}: H{dim} bar {p:?} became {q:?} under permutation"
                );
            }
            assert_eq!(
                essential_in_dim(&base, dim),
                essential_in_dim(&permuted, dim),
                "seed {seed}: H{dim} essential classes changed under permutation"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Isometry invariance
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn diagram_is_invariant_under_rotation_and_translation() {
    // Rips depends only on pairwise distances, which isometries preserve exactly.
    // A violation means a coordinate-dependent distance or a centering step that
    // leaked into the filtration.
    for seed in [3u64, 11, 2024] {
        let mut rng = Rng::new(seed);
        let points = circle(14, 1.3, 0.04, &mut rng);
        let moved = rotate_translate(&points, 0.9128, -4.5, 17.25);

        let base = persistent_homology(&points, config(1, 32)).unwrap();
        let isometric = persistent_homology(&moved, config(1, 32)).unwrap();

        let d = diagram_distance(&base, &isometric, 1);
        assert!(
            d < 1e-9,
            "seed {seed}: isometry moved the diagram by {d:e} (expected 0)"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Scale equivariance
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn diagram_scales_linearly_with_the_point_cloud() {
    // Scaling the cloud by c > 0 must scale every birth and death by exactly c.
    // A violation means a hardcoded epsilon or an absolute threshold pretending to
    // be a relative one — the bug that makes every result silently dataset-dependent.
    let mut rng = Rng::new(555);
    let points = circle(14, 1.0, 0.03, &mut rng);
    let base = persistent_homology(&points, config(1, 32)).unwrap();

    for factor in [0.125, 0.5, 2.0, 37.0] {
        let scaled = persistent_homology(&scale(&points, factor), config(1, 32)).unwrap();

        for dim in 0..=1 {
            let expected: Vec<(f64, f64)> = finite_in_dim(&base, dim)
                .into_iter()
                .map(|(b, d)| (b * factor, d * factor))
                .collect();
            let actual = finite_in_dim(&scaled, dim);

            assert_eq!(
                expected.len(),
                actual.len(),
                "factor {factor}: H{dim} bar count changed under scaling"
            );
            for (e, a) in expected.iter().zip(actual.iter()) {
                let tol = 1e-9 * factor.max(1.0);
                assert!(
                    (e.0 - a.0).abs() < tol && (e.1 - a.1).abs() < tol,
                    "factor {factor}: H{dim} expected {e:?}, got {a:?}"
                );
            }
        }
    }
}

/// An edge just beyond the radius cap must be excluded, with no slack.
///
/// This test exists because a mutant survived every suite in the workspace. The
/// filtration's admission test is `r <= config.max_radius`; adding an absolute
/// `+0.001` to that bound changed nothing any assertion could see.
///
/// The cap was not untested — `a_scaled_radius_cap_selects_the_same_complex`
/// below exercises it at 1.2 and at 1.2 × 6, and an absolute epsilon breaks the
/// proportionality that test asserts. It survived anyway, because on a
/// twelve-point circle no pairwise distance happens to fall in the thousandth
/// of a unit above either cap. The property was right and the fixture never put
/// it under load, which is the failure this repository's own mutation section
/// warns about in the sentence about fixtures chosen for convenience rather than
/// for discrimination.
///
/// Three collinear points place one distance exactly on the cap and one 5e-4
/// above it, so the boundary is the only thing the answer depends on. Correct
/// code leaves two components; any positive slack merges them into one, and the
/// essential H0 count says which happened.
#[test]
fn an_edge_just_beyond_the_radius_cap_is_excluded() {
    // d(a,b) = 1.0 exactly, d(b,c) = 1.0005, d(a,c) = 2.0005.
    let points = [
        ManifoldPoint::new([0.0, 0.0]),
        ManifoldPoint::new([1.0, 0.0]),
        ManifoldPoint::new([2.0005, 0.0]),
    ];

    let mut capped = config(1, 32);
    capped.max_radius = 1.0;
    let diagram = persistent_homology(&points, capped).unwrap();

    let components = essential_in_dim(&diagram, 0).len();
    assert_eq!(
        components, 2,
        "with a cap of exactly 1.0 the edge at 1.0005 must be excluded, leaving \
         two components; {components} means the admission test carries slack"
    );

    // The control. Raising the cap past 1.0005 must merge them, or the assertion
    // above would also hold on an implementation that admits no edges at all.
    let mut wider = capped;
    wider.max_radius = 1.001;
    let merged = persistent_homology(&points, wider).unwrap();
    assert_eq!(
        essential_in_dim(&merged, 0).len(),
        1,
        "a cap of 1.001 admits the edge at 1.0005 and must leave one component; \
         if it does not, the test above passes for the wrong reason"
    );
}

#[test]
fn a_scaled_radius_cap_selects_the_same_complex() {
    // `max_radius` is an absolute length. Scaling the cloud without scaling the cap
    // is a user error; scaling both must reproduce the diagram exactly, scaled.
    let mut rng = Rng::new(556);
    let points = circle(12, 1.0, 0.0, &mut rng);
    let factor = 6.0;

    let mut capped = config(1, 32);
    capped.max_radius = 1.2;
    let base = persistent_homology(&points, capped).unwrap();

    let mut capped_scaled = capped;
    capped_scaled.max_radius = 1.2 * factor;
    let scaled = persistent_homology(&scale(&points, factor), capped_scaled).unwrap();

    for dim in 0..=1 {
        let expected: Vec<(f64, f64)> = finite_in_dim(&base, dim)
            .into_iter()
            .map(|(b, d)| (b * factor, d * factor))
            .collect();
        let actual = finite_in_dim(&scaled, dim);
        assert_eq!(expected.len(), actual.len(), "H{dim} bar count changed");
        for (e, a) in expected.iter().zip(actual.iter()) {
            assert!(
                (e.0 - a.0).abs() < 1e-8 && (e.1 - a.1).abs() < 1e-8,
                "H{dim} expected {e:?}, got {a:?}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Stability — the load-bearing test
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bottleneck_distance_respects_the_stability_bound() {
    // Cohen-Steiner-Edelsbrunner-Harer, in the Vietoris-Rips form: if no point
    // moves more than eps, the diagram moves at most 2*eps in bottleneck distance.
    // The constant is 2*eps via the Hausdorff bound for Rips; eps would produce
    // spurious failures and 4*eps would pass a broken implementation.
    //
    // This is the single most valuable assertion in the file. Wrong pairing, a
    // dropped bar, a mishandled infinite death, and an early-terminating reduction
    // all surface here.
    for seed in [2u64, 19, 404, 65_537] {
        for eps in [1e-3, 1e-2, 5e-2] {
            let mut rng = Rng::new(seed);
            let points = circle(12, 1.0, 0.02, &mut rng);
            let moved = perturb(&points, eps, &mut rng);

            let base = persistent_homology(&points, config(1, 32)).unwrap();
            let noisy = persistent_homology(&moved, config(1, 32)).unwrap();

            let d = diagram_distance(&base, &noisy, 1);
            assert!(
                d <= 2.0 * eps + 1e-9,
                "seed {seed}, eps {eps}: bottleneck {d:e} exceeds the 2*eps bound {:e}",
                2.0 * eps
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. Known-manifold ground truth
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn circle_has_exactly_one_long_h1_bar_dying_at_sqrt3_times_the_radius() {
    // For the Vietoris-Rips complex of a circle of radius r, the single H1 class
    // dies at exactly sqrt(3) * r. This constant is sharp and non-obvious: an
    // implementation that gets the qualitative shape right and this number wrong
    // has a real defect.
    let radius = 1.0;
    let mut rng = Rng::new(31_337);
    let points = circle(18, radius, 0.0, &mut rng);

    let diagram = persistent_homology(&points, config(1, 32)).unwrap();

    let long = long_bars(&diagram, 1, 0.5 * radius);
    assert_eq!(long, 1, "expected exactly one long H1 bar, found {long}");

    let (birth, death, _) = longest_bar(&diagram, 1).expect("no finite H1 bar at all");
    let expected_death = 3.0f64.sqrt() * radius;

    // Discretization error for n points on the circle is O(1/n); at n = 18 the
    // chord spacing is ~0.35, so 5% is the honest tolerance, not a fudge factor.
    let relative_error = (death - expected_death).abs() / expected_death;
    assert!(
        relative_error < 0.05,
        "H1 bar [{birth}, {death}) should die at sqrt(3)*r = {expected_death}; \
         relative error {relative_error:.4} exceeds 5%"
    );

    // The class must be born well before it dies, or the "loop" is an artifact.
    assert!(
        birth < 0.5 * expected_death,
        "H1 born at {birth}, too late relative to death {death}"
    );
}

#[test]
fn separated_clusters_produce_one_h0_bar_each_until_the_gap_closes() {
    // k tight clusters separated by `separation` must show exactly k connected
    // components at any radius between the intra-cluster spread and the gap.
    let (k, per, separation) = (3usize, 5usize, 4.0);
    let mut rng = Rng::new(808);
    let points = clusters(k, per, separation, &mut rng);

    let diagram = persistent_homology(&points, config(0, 32)).unwrap();

    let betti = diagram.betti_at(1.0);
    assert_eq!(
        betti.beta_0, k as u32,
        "expected {k} components at radius 1.0 (intra-spread 0.04, gap {separation}), got {}",
        betti.beta_0
    );

    // Past the gap everything is one component.
    assert_eq!(
        diagram.betti_at(separation + 1.0).beta_0,
        1,
        "clusters should merge into one component past the gap"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. Negative control
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn gaussian_noise_produces_no_long_h1_bar() {
    // The negative control is worth as much as the positive cases: a pipeline that
    // finds structure in noise will find it everywhere, and every downstream claim
    // built on it is unfalsifiable.
    for seed in [5u64, 23, 97, 6_700_417] {
        let mut rng = Rng::new(seed);
        let points = blob(16, 1.0, &mut rng);

        let diagram = persistent_homology(&points, config(1, 32)).unwrap();

        // Compare against the circle's signal: a real loop in a radius-1 circle
        // persists for ~0.7. Noise bars must stay well under that.
        let long = long_bars(&diagram, 1, 0.6);
        assert_eq!(
            long,
            0,
            "seed {seed}: Gaussian blob reported {long} long H1 bar(s); longest was {:?}",
            longest_bar(&diagram, 1)
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. Elder rule / independent H0 reference
// ═══════════════════════════════════════════════════════════════════════════════

/// Independent, obviously-correct H₀ via union-find over edges in increasing
/// length. This is the elder rule stated directly: when two components merge, the
/// younger one dies at that edge length.
fn h0_reference(points: &[ManifoldPoint<2>]) -> Vec<f64> {
    let n = points.len();
    let mut edges: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            edges.push((points[i].distance(&points[j]), i, j));
        }
    }
    edges.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            let root = find(parent, parent[x]);
            parent[x] = root;
        }
        parent[x]
    }

    let mut deaths = Vec::new();
    for (length, i, j) in edges {
        let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
        if ri != rj {
            parent[ri] = rj;
            deaths.push(length);
        }
    }
    deaths.sort_by(f64::total_cmp);
    deaths
}

#[test]
fn h0_matches_an_independent_union_find() {
    // Every H0 class is born at 0 and dies when its component merges. The set of
    // death times must equal the minimum spanning tree's edge lengths exactly.
    for seed in [13u64, 77, 999] {
        let mut rng = Rng::new(seed);
        let points = blob(14, 1.0, &mut rng);

        let diagram = persistent_homology(&points, config(0, 32)).unwrap();

        let mut actual: Vec<f64> = finite_in_dim(&diagram, 0)
            .into_iter()
            .map(|(birth, death)| {
                assert_eq!(birth, 0.0, "seed {seed}: H0 class born at {birth}, not 0");
                death
            })
            .collect();
        actual.sort_by(f64::total_cmp);

        let expected = h0_reference(&points);
        assert_eq!(
            actual.len(),
            expected.len(),
            "seed {seed}: {} H0 deaths, union-find says {}",
            actual.len(),
            expected.len()
        );
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < 1e-9,
                "seed {seed}: H0 death {a} vs union-find {e}"
            );
        }

        // Exactly one component survives forever, on a connected complex.
        assert_eq!(
            essential_in_dim(&diagram, 0).len(),
            1,
            "seed {seed}: expected exactly one essential H0 class"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regression guards on the error surface
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn a_single_point_has_one_essential_component_and_nothing_else() {
    let points = [ManifoldPoint::<2>::new([0.0, 0.0])];
    let diagram = persistent_homology(&points, config(2, 32)).unwrap();

    assert_eq!(diagram.pairs.len(), 1, "got {:?}", diagram.pairs);
    assert_eq!(
        diagram.pairs[0],
        PersistencePair {
            dimension: 0,
            birth: 0.0,
            death: None
        }
    );
}

#[test]
fn duplicate_points_do_not_break_the_reduction() {
    // Coincident points give zero-length edges: a degenerate filtration value that
    // breaks tie-breaking in a naive implementation.
    let points = [
        ManifoldPoint::<2>::new([0.0, 0.0]),
        ManifoldPoint::<2>::new([0.0, 0.0]),
        ManifoldPoint::<2>::new([0.0, 0.0]),
        ManifoldPoint::<2>::new([2.0, 0.0]),
    ];
    let diagram = persistent_homology(&points, config(1, 32)).unwrap();

    assert_eq!(
        diagram.betti_at(0.0).beta_0,
        2,
        "three coincident + one apart"
    );
    assert_eq!(diagram.betti_at(3.0).beta_0, 1);
    assert_eq!(essential_in_dim(&diagram, 0).len(), 1);
}
