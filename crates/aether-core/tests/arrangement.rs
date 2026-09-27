//! Closed-form tests for `aether_core::arrangement`.
//!
//! Truth comes from each figure's construction, derived on paper, never from the
//! module under test. The figures and refusal cases follow planimeter's
//! `tests/figures.py` and `tests/test_arrange.py`. A counting bug does not crash:
//! it returns a plausible integer. These are the properties a plausible wrong
//! integer violates.
//!
//! Asserted here:
//!   1. Closed-form families — square, triangle, disjoint and nested squares,
//!      split diagonals, figure-eight, the n×m grid, T-junction, outside tangent,
//!      collinear overlap and duplicates.
//!   2. The snap rule — the merge-nothing window of a clean figure; a vertex-pair
//!      gap below the radius merges and one above it stays open; the feature/ρ
//!      boundary; a collapsed edge falls through to the next window.
//!   3. Invariance — rigid motion, uniform scaling (window scaled with it, and a
//!      scaled grid), segment permutation, segment reversal.
//!   4. Refusals — a crossing with no vertex, a vertex near an edge, no stable
//!      scale, non-finite and empty input, the vertex budget.
//!
//! Every certified answer is also checked against the identity
//! `faces = E − V + C`, `χ = V − E = pieces − faces`.

use aether_core::arrangement::{
    arrange, ArrangementConfig, Chi, RadiusSource, Refusal, RefusalKind, Segment,
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
// Figures, with their truth derived from the construction
// ═══════════════════════════════════════════════════════════════════════════════

fn seg(ax: f64, ay: f64, bx: f64, by: f64) -> Segment {
    [[ax, ay], [bx, by]]
}

fn square(x: f64, y: f64, side: f64) -> Vec<Segment> {
    vec![
        seg(x, y, x + side, y),
        seg(x + side, y, x + side, y + side),
        seg(x + side, y + side, x, y + side),
        seg(x, y + side, x, y),
    ]
}

fn join(parts: &[Vec<Segment>]) -> Vec<Segment> {
    parts.concat()
}

fn triangle() -> Vec<Segment> {
    vec![
        seg(0.0, 0.0, 10.0, 0.0),
        seg(10.0, 0.0, 5.0, 8.66),
        seg(5.0, 8.66, 0.0, 0.0),
    ]
}

/// `n` columns by `m` rows of unit cells, one segment per cell side.
/// V = (n+1)(m+1), E = n(m+1) + m(n+1), faces = nm, χ = 1 − nm.
fn grid(n: usize, m: usize) -> Vec<Segment> {
    let mut out = Vec::new();
    for j in 0..=m {
        for i in 0..n {
            out.push(seg(i as f64, j as f64, (i + 1) as f64, j as f64));
        }
    }
    for i in 0..=n {
        for j in 0..m {
            out.push(seg(i as f64, j as f64, i as f64, (j + 1) as f64));
        }
    }
    out
}

/// A square with a crosswall between the midpoints of two sides. 1 piece, 2 faces.
fn t_junction() -> Vec<Segment> {
    join(&[square(0.0, 0.0, 10.0), vec![seg(5.0, 0.0, 5.0, 10.0)]])
}

/// A stub touching the bottom side from outside. 1 piece, 1 face, 1 dangle.
fn outside_stub() -> Vec<Segment> {
    join(&[square(0.0, 0.0, 10.0), vec![seg(5.0, 0.0, 5.0, -5.0)]])
}

/// Both diagonals as four half-diagonals meeting at a written centre. 4 faces.
fn split_diagonals() -> Vec<Segment> {
    join(&[
        square(0.0, 0.0, 10.0),
        vec![
            seg(0.0, 0.0, 5.0, 5.0),
            seg(5.0, 5.0, 10.0, 10.0),
            seg(10.0, 0.0, 5.0, 5.0),
            seg(5.0, 5.0, 0.0, 10.0),
        ],
    ])
}

/// Two squares sharing the corner (10, 10). 1 piece, 2 faces.
fn figure_eight() -> Vec<Segment> {
    join(&[square(0.0, 0.0, 10.0), square(10.0, 10.0, 10.0)])
}

/// A square whose first corner is written twice, one copy displaced by `(d, d)`.
fn jittered_corner(d: f64) -> Vec<Segment> {
    let mut s = square(0.0, 0.0, 10.0);
    s[0][0] = [d, d];
    s
}

/// A square whose closing side stops `gap` short of the first corner.
fn open_corner(gap: f64) -> Vec<Segment> {
    let mut s = square(0.0, 0.0, 10.0);
    s[3][1] = [0.0, gap];
    s
}

/// A square and a segment of length 1e-4 inside it. 2 pieces, 1 face.
fn square_with_speck() -> Vec<Segment> {
    join(&[square(0.0, 0.0, 10.0), vec![seg(5.0, 5.0, 5.0001, 5.0)]])
}

// ═══════════════════════════════════════════════════════════════════════════════
// Harness
// ═══════════════════════════════════════════════════════════════════════════════

fn with_grid(grid: f64) -> ArrangementConfig {
    ArrangementConfig {
        grid: Some(grid),
        ..ArrangementConfig::default()
    }
}

/// Certify, and check the identity and the window on every answer.
fn certify_with(segments: &[Segment], config: ArrangementConfig) -> Chi {
    let c = arrange(segments, &config).unwrap_or_else(|r| panic!("refused: {r:?}"));
    assert_eq!(
        c.faces + c.vertices,
        c.edges + c.pieces,
        "faces != E - V + C: {c:?}"
    );
    assert_eq!(
        c.chi,
        c.vertices as i64 - c.edges as i64,
        "chi != V - E: {c:?}"
    );
    assert_eq!(
        c.chi,
        c.pieces as i64 - c.faces as i64,
        "chi != pieces - faces: {c:?}"
    );
    assert!(c.ratio >= config.rho, "window narrower than rho: {c:?}");
    assert!(
        c.t_below <= c.radius && c.radius < c.t_above,
        "radius outside window: {c:?}"
    );
    c
}

fn certify(segments: &[Segment]) -> Chi {
    certify_with(segments, ArrangementConfig::default())
}

fn refuse(segments: &[Segment], config: ArrangementConfig) -> Refusal {
    match arrange(segments, &config) {
        Ok(c) => panic!("certified where a refusal was due: {c:?}"),
        Err(r) => r,
    }
}

/// (V, E, pieces, faces, χ).
fn integers(c: &Chi) -> (usize, usize, usize, usize, i64) {
    (c.vertices, c.edges, c.pieces, c.faces, c.chi)
}

fn map(segments: &[Segment], f: impl Fn([f64; 2]) -> [f64; 2]) -> Vec<Segment> {
    segments.iter().map(|s| [f(s[0]), f(s[1])]).collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Closed-form families
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn a_clean_square_certifies_one_face_at_the_merge_nothing_window() {
    let c = certify(&square(0.0, 0.0, 10.0));
    assert_eq!(integers(&c), (4, 4, 1, 1, 0));
    assert_eq!(c.dangles, 0);
    // The only window is [δ, 10): the floor up to the smallest real separation.
    let floor = 4096.0 * f64::EPSILON * 10.0;
    assert_eq!(c.t_below, floor);
    assert_eq!(c.t_above, 10.0);
    assert_eq!(c.radius, (floor * 10.0).sqrt());
    assert_eq!(c.radius_source, RadiusSource::Derived);
    assert_eq!((c.n_merged, c.n_candidates), (0, 1));

    // The apex is 8.66 from its own base; a vertex-pair-only spectrum would read
    // that as a band and refuse an ordinary triangle.
    assert_eq!(integers(&certify(&triangle())), (3, 3, 1, 1, 0));
}

#[test]
fn disjoint_and_nested_squares_add_pieces_and_faces() {
    let two = join(&[square(0.0, 0.0, 10.0), square(100.0, 0.0, 10.0)]);
    assert_eq!(integers(&certify(&two)), (8, 8, 2, 2, 0));

    let nested = join(&[square(0.0, 0.0, 30.0), square(10.0, 10.0, 10.0)]);
    assert_eq!(integers(&certify(&nested)), (8, 8, 2, 2, 0));

    // A disjoint union is additive in every integer.
    let far = map(&t_junction(), |[x, y]| [x + 1000.0, y]);
    let both = certify(&join(&[square(0.0, 0.0, 10.0), far]));
    assert_eq!(integers(&both), (4 + 6, 4 + 7, 1 + 1, 1 + 2, -1));
}

#[test]
fn diagonals_meeting_at_a_written_centre_make_four_faces() {
    let c = certify(&split_diagonals());
    assert_eq!(integers(&c), (5, 8, 1, 4, -3));
    assert_eq!(c.n_subdivided, 0);
}

#[test]
fn a_figure_eight_shares_one_vertex_between_two_faces() {
    let c = certify(&figure_eight());
    assert_eq!(integers(&c), (7, 8, 1, 2, -1));
    assert_eq!(c.dangles, 0);
}

#[test]
fn an_n_by_m_grid_has_n_times_m_faces() {
    for n in 1..=4 {
        for m in 1..=5 {
            let c = certify(&grid(n, m));
            let v = (n + 1) * (m + 1);
            let e = n * (m + 1) + m * (n + 1);
            assert_eq!(
                integers(&c),
                (v, e, 1, n * m, 1 - (n * m) as i64),
                "grid {n}x{m}"
            );
        }
    }
}

#[test]
fn an_endpoint_on_a_segment_interior_subdivides_it_without_crossing() {
    // Inside: the crosswall splits the square. A planarity test alone passes
    // this input and reads two pieces and one face.
    let t = certify(&t_junction());
    assert_eq!(integers(&t), (6, 7, 1, 2, -1));
    assert_eq!(t.n_subdivided, 2);

    // Outside: tangent to the bottom side, it adds a vertex and a dangle, no face.
    let stub = certify(&outside_stub());
    assert_eq!(integers(&stub), (6, 6, 1, 1, 0));
    assert_eq!((stub.n_subdivided, stub.dangles), (1, 1));

    // H: a rung between the interiors of two rails is a tree.
    let h = vec![
        seg(0.0, 0.0, 0.0, 20.0),
        seg(10.0, 0.0, 10.0, 20.0),
        seg(0.0, 10.0, 10.0, 10.0),
    ];
    assert_eq!(integers(&certify(&h)), (6, 5, 1, 0, 1));
}

#[test]
fn collinear_overlaps_and_duplicates_leave_one_edge_per_vertex_pair() {
    let nested = certify(&[seg(0.0, 0.0, 10.0, 0.0), seg(3.0, 0.0, 7.0, 0.0)]);
    assert_eq!(integers(&nested), (4, 3, 1, 0, 1));
    assert_eq!(nested.n_dup_edges, 1);

    let staggered = certify(&[seg(0.0, 0.0, 6.0, 0.0), seg(4.0, 0.0, 10.0, 0.0)]);
    assert_eq!(integers(&staggered), (4, 3, 1, 0, 1));

    for twin in [seg(0.0, 0.0, 10.0, 0.0), seg(10.0, 0.0, 0.0, 0.0)] {
        let dup = certify(&[seg(0.0, 0.0, 10.0, 0.0), twin]);
        assert_eq!(integers(&dup), (2, 1, 1, 0, 1));
        assert_eq!(dup.n_dup_edges, 1);
    }

    // A doubled wall would otherwise inflate E, hence faces, by one.
    let doubled = certify(&join(&[
        square(0.0, 0.0, 10.0),
        vec![seg(10.0, 0.0, 0.0, 0.0)],
    ]));
    assert_eq!(integers(&doubled), (4, 4, 1, 1, 0));
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. The snap rule
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn a_vertex_pair_gap_below_the_radius_merges_and_one_above_it_stays_open() {
    let gap = 1e-5 * 2f64.sqrt();

    // Derived: the genuine gap [√2·1e-5, 9.99999) is tried before merge-nothing.
    let merged = certify(&jittered_corner(1e-5));
    assert_eq!(integers(&merged), (4, 4, 1, 1, 0));
    assert_eq!(merged.n_merged, 1);
    // The window opens at the merge height itself and the radius sits above it.
    assert!(
        (merged.t_below - gap).abs() < 1e-20 && merged.radius > gap,
        "{merged:?}"
    );

    // A supplied radius above the gap closes the corner; one below it does not.
    let above = certify_with(&jittered_corner(1e-5), with_grid(1e-3));
    assert_eq!(integers(&above), (4, 4, 1, 1, 0));
    assert_eq!(
        (above.radius, above.radius_source),
        (1e-3, RadiusSource::User)
    );
    let below = certify_with(&jittered_corner(1e-5), with_grid(1e-8));
    assert_eq!(integers(&below), (5, 4, 1, 0, 1));
    assert_eq!((below.n_merged, below.dangles), (0, 2));

    // A gap of 1 on a 10-unit square sits above the derived radius: open path.
    let open = certify(&open_corner(1.0));
    assert_eq!(integers(&open), (5, 4, 1, 0, 1));
    assert!(open.radius < 1.0);

    // [1, 9) has ratio 9: a radius of 5 closes it only once ρ is lowered to 5.
    let lowered = ArrangementConfig {
        rho: 5.0,
        ..with_grid(5.0)
    };
    let closed = certify_with(&open_corner(1.0), lowered);
    assert_eq!(integers(&closed), (4, 4, 1, 1, 0));
    assert_eq!(
        (closed.t_below, closed.t_above, closed.n_merged),
        (1.0, 9.0, 1)
    );
}

#[test]
fn a_gap_one_tenth_of_the_feature_is_the_boundary_of_the_stated_rule() {
    // planimeter RESULTS.md §8: the widest window is (gap, 10) with ratio 10/gap,
    // so a gap of exactly 1 merges the facing corners and 1.01 does not.
    let touching = certify(&join(&[square(0.0, 0.0, 10.0), square(11.0, 0.0, 10.0)]));
    assert_eq!(integers(&touching), (6, 7, 1, 2, -1));
    assert_eq!((touching.n_merged, touching.n_dup_edges), (2, 1));

    let apart = certify(&join(&[square(0.0, 0.0, 10.0), square(11.01, 0.0, 10.0)]));
    assert_eq!(integers(&apart), (8, 8, 2, 2, 0));
    assert_eq!(apart.n_merged, 0);
}

#[test]
fn a_collapsed_edge_falls_through_to_the_next_window() {
    // Derived: [1e-4, 4.9999) swallows the speck, so merge-nothing is tried next.
    let c = certify(&square_with_speck());
    assert_eq!(integers(&c), (6, 5, 2, 1, 1));
    assert_eq!((c.n_candidates, c.n_merged), (2, 0));

    // A supplied radius above the speck's length has no next window.
    let r = refuse(&square_with_speck(), with_grid(0.01));
    assert_eq!(
        r,
        Refusal::EdgeCollapsed {
            segment: 4,
            count: 1
        }
    );
    assert_eq!(r.kind(), RefusalKind::Geometry);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Invariance
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn rigid_motions_preserve_the_integers() {
    // Exact in float64: quarter turn, reflection, dyadic translation. Exact
    // incidences stay exact, so the T-junction and the stub are included.
    let exact: [fn([f64; 2]) -> [f64; 2]; 3] = [
        |[x, y]| [-y, x],
        |[x, y]| [-x, y],
        |[x, y]| [x + 123.25, y - 47.5],
    ];
    for figure in [
        t_junction(),
        outside_stub(),
        split_diagonals(),
        figure_eight(),
        grid(3, 2),
    ] {
        let base = integers(&certify(&figure));
        for f in exact {
            assert_eq!(integers(&certify(&map(&figure, f))), base);
        }
    }

    // General rotations round every coordinate, but a shared endpoint rounds to
    // the same value everywhere it is written, so figures without interior
    // incidences still certify with the same integers.
    for figure in [
        square(0.0, 0.0, 10.0),
        join(&[square(0.0, 0.0, 10.0), square(100.0, 0.0, 10.0)]),
        split_diagonals(),
        figure_eight(),
        grid(3, 2),
        jittered_corner(1e-5),
    ] {
        let base = integers(&certify(&figure));
        for theta in [0.3f64, 1.1, 2.5, 4.0] {
            let (s, c) = theta.sin_cos();
            let moved = map(&figure, |[x, y]| {
                [c * x - s * y + 123.25, s * x + c * y - 47.5]
            });
            assert_eq!(integers(&certify(&moved)), base, "theta {theta}");
        }
    }
}

#[test]
fn uniform_scaling_scales_the_window_with_it() {
    // A power of two is exact in binary floating point, so the window moves by
    // exactly the factor and the integers do not move at all.
    for figure in [
        square(0.0, 0.0, 10.0),
        t_junction(),
        jittered_corner(1e-5),
        figure_eight(),
        grid(3, 2),
    ] {
        let a = certify(&figure);
        let b = certify(&map(&figure, |[x, y]| [x * 1024.0, y * 1024.0]));
        assert_eq!(integers(&a), integers(&b));
        assert_eq!(b.t_below, a.t_below * 1024.0);
        assert_eq!(b.t_above, a.t_above * 1024.0);
        assert_eq!(b.radius, a.radius * 1024.0);
    }

    // A non-dyadic factor, with a supplied radius scaled by the same factor.
    let tripled = map(&jittered_corner(1e-5), |[x, y]| [3.0 * x, 3.0 * y]);
    assert_eq!(integers(&certify(&tripled)), (4, 4, 1, 1, 0));
    assert_eq!(
        integers(&certify_with(&tripled, with_grid(3e-3))),
        (4, 4, 1, 1, 0)
    );
    assert_eq!(
        integers(&certify_with(&tripled, with_grid(3e-8))),
        (5, 4, 1, 0, 1)
    );
    assert_eq!(
        integers(&certify(&map(&t_junction(), |[x, y]| [3.0 * x, 3.0 * y]))),
        (6, 7, 1, 2, -1)
    );
}

#[test]
fn segment_order_and_orientation_do_not_move_the_answer_or_the_window() {
    let figures = [
        t_junction(),
        outside_stub(),
        split_diagonals(),
        jittered_corner(1e-5),
        grid(3, 4),
        square_with_speck(),
        join(&[square(0.0, 0.0, 10.0), square(11.0, 0.0, 10.0)]),
        vec![seg(0.0, 0.0, 10.0, 0.0), seg(3.0, 0.0, 7.0, 0.0)],
    ];
    for figure in &figures {
        let base = certify(figure);
        for seed in [1u64, 7, 42, 1337, 90210] {
            let perm = Rng::new(seed).permutation(figure.len());
            let shuffled: Vec<Segment> = perm.iter().map(|&i| figure[i]).collect();
            assert_eq!(certify(&shuffled), base, "seed {seed}");
        }
        let reversed: Vec<Segment> = figure.iter().map(|s| [s[1], s[0]]).collect();
        assert_eq!(certify(&reversed), base);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Refusals
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn crossing_segments_refuse_rather_than_invent_the_point() {
    let diagonals = join(&[
        square(0.0, 0.0, 10.0),
        vec![seg(0.0, 0.0, 10.0, 10.0), seg(10.0, 0.0, 0.0, 10.0)],
    ]);
    let r = refuse(&diagonals, ArrangementConfig::default());
    assert_eq!(
        r,
        Refusal::EdgesCross {
            first: 4,
            second: 5,
            count: 1
        }
    );
    assert_eq!(r.kind(), RefusalKind::Geometry);

    // A raw pentagram crosses at five points the input does not contain.
    let p: Vec<[f64; 2]> = (0..5)
        .map(|i| {
            let a = (90.0 + 72.0 * i as f64).to_radians();
            [a.cos(), a.sin()]
        })
        .collect();
    let star: Vec<Segment> = (0..5).map(|i| [p[i], p[(i + 2) % 5]]).collect();
    assert!(matches!(
        refuse(&star, ArrangementConfig::default()),
        Refusal::EdgesCross { count: 5, .. }
    ));
}

#[test]
fn a_vertex_near_an_edge_refuses_and_names_the_distance() {
    // A crosswall whose end misses the floor by 2.2e-6: no two points are close,
    // so only the vertex-to-edge distances in the spectrum can see it.
    let near = join(&[square(0.0, 0.0, 10.0), vec![seg(5.0, 2.2e-6, 5.0, 10.0)]]);
    match refuse(&near, ArrangementConfig::default()) {
        Refusal::VertexNearEdge {
            vertex,
            segment,
            distance,
            count,
        } => {
            assert_eq!((vertex, segment, count), ([5.0, 2.2e-6], 0, 1));
            assert!((distance - 2.2e-6).abs() < 1e-15, "{distance}");
        }
        other => panic!("expected VertexNearEdge, got {other:?}"),
    }

    // Both resolutions certify: onto the wall, or clearly away from it.
    assert_eq!(integers(&certify(&t_junction())), (6, 7, 1, 2, -1));
    let away = join(&[square(0.0, 0.0, 10.0), vec![seg(5.0, 2.0, 5.0, 10.0)]]);
    assert_eq!(integers(&certify(&away)), (6, 6, 1, 1, 0));
}

#[test]
fn no_stable_scale_refuses_rather_than_choosing() {
    // At magnitude 1e8 the floor is 9.1e-5, so separations of 1e-4 to 2e-4 have
    // no gap of ratio 10 anywhere.
    let chain: Vec<Segment> = (0..3)
        .map(|i| seg(1e8 + i as f64 * 1e-4, 0.0, 1e8 + (i + 1) as f64 * 1e-4, 0.0))
        .collect();
    let r = refuse(&chain, ArrangementConfig::default());
    assert!(
        matches!(r, Refusal::NoStableScale { gap: Some(_) }),
        "{r:?}"
    );
    assert_eq!(r.kind(), RefusalKind::Geometry);

    // A supplied radius inside a gap narrower than ρ, above every separation, or
    // below the floor is refused the same way.
    let open = open_corner(1.0);
    assert_eq!(
        refuse(&open, with_grid(2.0)),
        Refusal::NoStableScale {
            gap: Some([1.0, 9.0])
        }
    );
    assert_eq!(
        refuse(&open, with_grid(1e9)),
        Refusal::NoStableScale { gap: None }
    );
    assert_eq!(
        refuse(&open, with_grid(1e-300)),
        Refusal::NoStableScale { gap: None }
    );

    // A lone zero-length segment has no separation at all.
    let point = [seg(1.0, 1.0, 1.0, 1.0)];
    assert_eq!(
        refuse(&point, ArrangementConfig::default()),
        Refusal::NoStableScale { gap: None }
    );
}

#[test]
fn non_finite_and_empty_input_are_refused_by_type() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let r = refuse(&[seg(0.0, 0.0, bad, 1.0)], ArrangementConfig::default());
        assert_eq!(r, Refusal::NonFinite { values: 1 });
        assert_eq!(r.kind(), RefusalKind::Input);

        let r = refuse(&square(0.0, 0.0, 10.0), with_grid(bad));
        assert_eq!(r, Refusal::InvalidGrid);
        assert_eq!(r.kind(), RefusalKind::Input);
    }
    let two = [
        seg(f64::NAN, 0.0, 1.0, 1.0),
        seg(0.0, f64::INFINITY, 1.0, 1.0),
    ];
    assert_eq!(
        refuse(&two, ArrangementConfig::default()),
        Refusal::NonFinite { values: 2 }
    );

    let r = refuse(&[], ArrangementConfig::default());
    assert_eq!(r, Refusal::NoGeometry);
    assert_eq!(r.kind(), RefusalKind::Geometry);
}

#[test]
fn the_vertex_ceiling_is_a_budget_refusal_not_a_geometry_one() {
    let g = grid(3, 3); // 16 distinct endpoints
    let tight = ArrangementConfig {
        max_vertices: 15,
        ..ArrangementConfig::default()
    };
    let r = refuse(&g, tight);
    assert_eq!(
        r,
        Refusal::TooManyVertices {
            actual: 16,
            max: 15
        }
    );
    assert_eq!(r.kind(), RefusalKind::Budget);

    // At the ceiling every quadratic pass fits in 4·16² pairs.
    let exact = ArrangementConfig {
        max_vertices: 16,
        ..ArrangementConfig::default()
    };
    assert_eq!(integers(&certify_with(&g, exact)), (16, 24, 1, 9, -8));
}
