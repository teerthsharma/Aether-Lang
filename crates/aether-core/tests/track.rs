//! Invariant tests for `aether_core::track`.
//!
//! A tracking bug rarely crashes: it returns a plausible forest with one swapped
//! identity, one fork in the wrong frame, or one link the gate should have
//! refused. Each test below is a property a plausible-but-wrong graph violates.
//!
//! Properties asserted here:
//!   1. Ground truth       — constant-velocity tracks, at the closed-form cost.
//!   2. Split              — one division, one fork, right parent, priced correctly.
//!   3. Gap budget         — a dropout ends a track; cleave has no gap closing.
//!   4. Gate               — priced inside the assignment, applied in micrometres.
//!   5. Separation         — crossing tracks beyond the gate stay distinct.
//!   6. Forest invariants  — degrees, one frame per edge, certificate (CleaveProofs §7-8).
//!   7. Invariance         — rigid motion of every frame, permutation within a frame.
//!   8. Refusals           — non-finite input, bad gate or scale, uncalibrated costs.

use aether_core::track::{
    flow_track, link_frames, link_sequence, physical_distance, FlowConfig, Node, Point, TrackError,
    TrackGraph, SCALE_UM,
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
// Scenes
// ═══════════════════════════════════════════════════════════════════════════════

const UNIT: [f64; 3] = [1.0, 1.0, 1.0];

/// Cells on a random walk in a 60 um box, with divisions, dropouts and clutter.
/// Coordinates are micrometres, so `UNIT` is the spacing.
fn scene(rng: &mut Rng, frames: usize) -> Vec<Vec<Point>> {
    let mut cells: Vec<Point> = (0..7)
        .map(|_| [60.0 * rng.unit(), 60.0 * rng.unit(), 60.0 * rng.unit()])
        .collect();
    let mut out = Vec::with_capacity(frames);
    for _ in 0..frames {
        let mut det: Vec<Point> = cells.iter().copied().filter(|_| rng.unit() > 0.1).collect();
        for _ in 0..rng.next_u64() % 3 {
            det.push([60.0 * rng.unit(), 60.0 * rng.unit(), 60.0 * rng.unit()]);
        }
        out.push(det);

        let mut next = Vec::with_capacity(cells.len() + 2);
        for c in &cells {
            // Each daughter takes its own step. Mirrored daughters `c +- step` would
            // be exactly equidistant from the parent, an exact cost tie whose winner
            // is decided by rounding, which no invariance can survive.
            let children = if rng.unit() < 0.12 { 2 } else { 1 };
            for _ in 0..children {
                let step = [2.0 * rng.normal(), 2.0 * rng.normal(), 2.0 * rng.normal()];
                next.push([c[0] + step[0], c[1] + step[1], c[2] + step[2]]);
            }
        }
        cells = next;
    }
    out
}

/// Divisions are cheap here (cleave's own `flow.py` demo value), so forks are
/// frequent; still strictly calibrated, since c_app + c_det = -40 < -30.
fn eager() -> FlowConfig {
    FlowConfig {
        c_div: -30.0,
        max_um: 8.0,
        ..FlowConfig::default()
    }
}

/// Edges as `(parent, child)` node pairs, sorted: comparable across graphs whose
/// node lists differ.
fn edge_nodes(g: &TrackGraph) -> Vec<(Node, Node)> {
    let mut v: Vec<_> = g
        .edges
        .iter()
        .map(|&(p, c)| (g.nodes[p], g.nodes[c]))
        .collect();
    v.sort();
    v
}

fn degrees(g: &TrackGraph) -> (Vec<usize>, Vec<usize>) {
    let mut indeg = vec![0; g.nodes.len()];
    let mut outdeg = vec![0; g.nodes.len()];
    for &(p, c) in &g.edges {
        outdeg[p] += 1;
        indeg[c] += 1;
    }
    (indeg, outdeg)
}

/// Every invariant the module guarantees, on one graph.
fn assert_forest(
    g: &TrackGraph,
    frames: &[Vec<Point>],
    max_out: usize,
    gate: f64,
    scale: [f64; 3],
) {
    let (indeg, outdeg) = degrees(g);
    for (v, node) in g.nodes.iter().enumerate() {
        assert!(
            node.index < frames[node.frame].len(),
            "node {node:?} names no detection"
        );
        assert!(
            indeg[v] <= 1,
            "node {node:?} has {} parents: a merge",
            indeg[v]
        );
        assert!(
            outdeg[v] <= max_out,
            "node {node:?} has {} children",
            outdeg[v]
        );
    }
    for &(p, c) in &g.edges {
        assert!(
            p < g.nodes.len() && c < g.nodes.len(),
            "edge ({p}, {c}) out of range"
        );
        let (a, b) = (g.nodes[p], g.nodes[c]);
        assert_eq!(
            b.frame,
            a.frame + 1,
            "edge {a:?} -> {b:?} does not advance one frame"
        );
        let d = physical_distance(frames[a.frame][a.index], frames[b.frame][b.index], scale);
        assert!(
            d <= gate,
            "edge {a:?} -> {b:?} is {d} um, beyond the {gate} um gate"
        );
    }
    let forks: Vec<usize> = (0..g.nodes.len()).filter(|&v| outdeg[v] == 2).collect();
    assert_eq!(
        g.divisions, forks,
        "divisions must be exactly the out-degree-2 nodes"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Ground truth
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn constant_velocity_particles_are_tracked_exactly_at_the_closed_form_cost() {
    // Five particles 25 um apart, each with its own constant velocity of at most
    // 3.4 um/frame, over eight frames. Every step beats the gate and every
    // cross-link is farther than it, so the true tracks are the unique optimum.
    let (k, frames) = (5usize, 8usize);
    let velocity = |p: usize| [0.5 * p as f64, 3.0 - 0.7 * p as f64, 1.0];
    let seq: Vec<Vec<Point>> = (0..frames)
        .map(|t| {
            (0..k)
                .map(|p| {
                    let v = velocity(p);
                    [
                        v[0] * t as f64,
                        25.0 * p as f64 + v[1] * t as f64,
                        v[2] * t as f64,
                    ]
                })
                .collect()
        })
        .collect();

    let truth: Vec<(Node, Node)> = {
        let mut v: Vec<_> = (0..frames - 1)
            .flat_map(|t| {
                (0..k).map(move |p| {
                    (
                        Node { frame: t, index: p },
                        Node {
                            frame: t + 1,
                            index: p,
                        },
                    )
                })
            })
            .collect();
        v.sort();
        v
    };

    let linked = link_sequence(&seq, 8.0, UNIT).unwrap();
    assert_eq!(edge_nodes(&linked), truth, "assignment linker lost a track");
    assert!(linked.divisions.is_empty());

    let cfg = FlowConfig::default();
    let solved = flow_track(&seq, &cfg, UNIT).unwrap();
    assert_eq!(edge_nodes(&solved.graph), truth, "circulation lost a track");
    assert!(solved.graph.divisions.is_empty());
    assert_eq!(
        solved.graph.nodes.len(),
        k * frames,
        "every detection must be used"
    );

    // One appearance and one disappearance per track, every detection used once,
    // plus the path length. Each arc is rounded to 1/1000, hence the tolerance.
    let path: f64 = (0..k)
        .map(|p| {
            let v = velocity(p);
            (frames - 1) as f64 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
        })
        .sum();
    let expected = k as f64 * (cfg.c_app + cfg.c_dis) + (k * frames) as f64 * cfg.c_det + path;
    let slack = (k * (frames - 1)) as f64 * 0.5e-3;
    assert!(
        (solved.cost - expected).abs() <= slack,
        "cost {} against closed form {expected}",
        solved.cost
    );
    assert!(solved.certified());
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Split
// ═══════════════════════════════════════════════════════════════════════════════

/// One cell drifting in x that divides between frames 2 and 3, the daughters
/// leaving at `offset` um either side in y, plus a bystander 100 um away.
fn dividing_scene(offset: f64) -> Vec<Vec<Point>> {
    (0..6)
        .map(|t| {
            let x = t as f64;
            let mut frame = if t <= 2 {
                vec![[0.0, 0.0, x]]
            } else {
                let y = offset + (t - 3) as f64;
                vec![[0.0, y, x], [0.0, -y, x]]
            };
            frame.push([0.0, 100.0, x]);
            frame
        })
        .collect()
}

#[test]
fn a_single_division_produces_exactly_one_split_on_the_correct_parent() {
    let seq = dividing_scene(2.0);
    let result = flow_track(&seq, &FlowConfig::default(), UNIT).unwrap();
    let g = &result.graph;
    assert!(
        result.certified(),
        "{} certificate violations",
        result.violations
    );
    assert_eq!(
        g.divisions.len(),
        1,
        "expected one split, got {:?}",
        g.divisions
    );

    // The fork belongs on the last single-cell frame, and its children are the two
    // daughters, not the bystander.
    let parent = g.divisions[0];
    assert_eq!(g.nodes[parent], Node { frame: 2, index: 0 });
    let children: Vec<Node> = g
        .edges
        .iter()
        .filter(|&&(p, _)| p == parent)
        .map(|&(_, c)| g.nodes[c])
        .collect();
    assert_eq!(
        children,
        vec![Node { frame: 3, index: 0 }, Node { frame: 3, index: 1 }]
    );
    assert_forest(g, &seq, 2, 12.0, UNIT);

    // The assignment linker is one-to-one, so the same scene has no split at all.
    assert!(link_sequence(&seq, 12.0, UNIT)
        .unwrap()
        .divisions
        .is_empty());
}

#[test]
fn a_split_is_taken_only_when_it_beats_a_new_lineage() {
    // The second daughter joins the parent iff c_div + d < c_app, i.e. d < 5 um at
    // the default costs. Daughters leave at `offset` in y and advance 1 um in x, so
    // d = sqrt(offset^2 + 1): below 5 for offsets 2 and 4, above it for 5 and 7,
    // all inside the 12 um gate.
    for (offset, splits) in [(2.0, 1), (4.0, 1), (5.0, 0), (7.0, 0)] {
        let result = flow_track(&dividing_scene(offset), &FlowConfig::default(), UNIT).unwrap();
        assert_eq!(
            result.graph.divisions.len(),
            splits,
            "offset {offset}: d = {}",
            (offset * offset + 1.0f64).sqrt()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Gap budget
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn a_dropout_ends_the_track_and_nothing_is_renumbered() {
    // cleave implements no gap closing: a one-frame dropout is two fragments.
    let a: Vec<Point> = vec![[10.0, 40.0, 20.0], [10.0, 40.0, 60.0]];
    let b: Vec<Point> = vec![[10.0, 40.0, 22.0], [10.0, 40.0, 62.0]];
    let empty: Vec<Point> = Vec::new();
    let gapped = vec![a.clone(), empty, b];

    let linked = link_sequence(&gapped, 8.0, SCALE_UM).unwrap();
    let frames: Vec<usize> = linked.nodes.iter().map(|n| n.frame).collect();
    assert_eq!(frames, vec![0, 0, 2, 2], "timepoints must not renumber");
    assert!(linked.edges.is_empty(), "an edge crossed an empty frame");

    let solved = flow_track(&gapped, &FlowConfig::default(), SCALE_UM).unwrap();
    assert!(
        solved.graph.edges.is_empty(),
        "an edge crossed an empty frame"
    );
    assert!(solved.graph.nodes.iter().all(|n| n.frame != 1));

    // A single missing detection, the rest of the frame present: the particle's
    // track is cut into two, and its neighbour is not borrowed to bridge it.
    let track = |t: usize| [10.0, 40.0, 20.0 + 2.0 * t as f64];
    let other = |t: usize| [10.0, 40.0, 60.0 + 2.0 * t as f64];
    let seq: Vec<Vec<Point>> = (0..5)
        .map(|t| {
            if t == 2 {
                vec![other(t)]
            } else {
                vec![track(t), other(t)]
            }
        })
        .collect();
    for g in [
        link_sequence(&seq, 8.0, SCALE_UM).unwrap(),
        flow_track(&seq, &FlowConfig::default(), SCALE_UM)
            .unwrap()
            .graph,
    ] {
        assert_forest(&g, &seq, 2, 8.0, SCALE_UM);
        assert_eq!(
            g.edges.len(),
            4 + 2,
            "four links for the full track, two + two for the cut one"
        );
        let into_gap = edge_nodes(&g)
            .into_iter()
            .filter(|(p, c)| p.frame == 1 && c.frame == 2)
            .count();
        assert_eq!(into_gap, 1, "only the unbroken track links into frame 2");
    }

    let nothing: Vec<Vec<Point>> = Vec::new();
    assert_eq!(
        link_sequence(&nothing, 8.0, UNIT).unwrap(),
        TrackGraph::default()
    );
    assert_eq!(
        flow_track(&nothing, &FlowConfig::default(), UNIT)
            .unwrap()
            .graph,
        TrackGraph::default()
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Gate
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn the_gate_is_priced_inside_the_assignment_not_applied_after_it() {
    // On a line, gate 5: a0 = 0, a1 = 10, b0 = 4, b1 = -6. The raw assignment
    // takes the swap a0-b1 (6) + a1-b0 (6) = 12 over a0-b0 (4) + a1-b1 (16) = 20,
    // and gating it afterwards discards both. Pricing the dummy at the gate keeps
    // a0-b0, the one pair inside it.
    let a: Vec<Point> = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]];
    let b: Vec<Point> = vec![[0.0, 0.0, 4.0], [0.0, 0.0, -6.0]];
    assert_eq!(link_frames(&a, &b, 5.0, UNIT).unwrap(), vec![(0, 0)]);
}

#[test]
fn the_assignment_is_global_not_greedy() {
    // cleave's contention fixture: both sources are nearest to target 0, and a
    // greedy pass would give it to one and orphan the other.
    let src: Vec<Point> = vec![[10.0, 40.0, 40.0], [10.0, 40.0, 44.0]];
    let dst: Vec<Point> = vec![[10.0, 40.0, 42.0], [10.0, 40.0, 47.0]];
    assert_eq!(
        link_frames(&src, &dst, 8.0, SCALE_UM).unwrap(),
        vec![(0, 0), (1, 1)]
    );
}

#[test]
fn the_gate_is_applied_in_micrometres_not_voxels() {
    // Four voxels along z is 6.5 um on the benchmark grid; along x it is 1.625 um.
    let a: Vec<Point> = vec![[0.0, 40.0, 40.0]];
    let along_z: Vec<Point> = vec![[4.0, 40.0, 40.0]];
    let along_x: Vec<Point> = vec![[0.0, 40.0, 44.0]];
    assert_eq!(physical_distance(a[0], along_z[0], SCALE_UM), 6.5);
    assert_eq!(physical_distance(a[0], along_x[0], SCALE_UM), 1.625);
    assert_eq!(
        link_frames(&a, &along_z, 8.0, SCALE_UM).unwrap(),
        vec![(0, 0)]
    );
    assert!(link_frames(&a, &along_z, 6.0, SCALE_UM).unwrap().is_empty());
    assert_eq!(
        link_frames(&a, &along_x, 6.0, SCALE_UM).unwrap(),
        vec![(0, 0)]
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. Separation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn crossing_tracks_separated_above_the_gate_stay_distinct() {
    // Two particles whose (y, x) paths cross head-on at frame 5, 8 voxels apart in
    // z: 13 um on the benchmark grid, above the 8 um gate, so no link may join one
    // particle to the other. A metric that dropped z would see them coincide at
    // the crossing, where every swap ties with the true link.
    let seq: Vec<Vec<Point>> = (0..11)
        .map(|t| {
            let s = 4.0 * t as f64;
            vec![[10.0, 40.0, 20.0 + s], [18.0, 40.0, 60.0 - s]]
        })
        .collect();
    let lane = |n: Node| n.index;
    for g in [
        link_sequence(&seq, 8.0, SCALE_UM).unwrap(),
        flow_track(
            &seq,
            &FlowConfig {
                max_um: 8.0,
                ..FlowConfig::default()
            },
            SCALE_UM,
        )
        .unwrap()
        .graph,
    ] {
        assert_forest(&g, &seq, 2, 8.0, SCALE_UM);
        assert_eq!(g.edges.len(), 2 * 10, "a track broke at the crossing");
        for (p, c) in edge_nodes(&g) {
            assert_eq!(lane(p), lane(c), "identities swapped at {p:?} -> {c:?}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. Forest invariants
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn graph_invariants_hold_on_seeded_random_scenes() {
    let mut forks = 0;
    for seed in 0..24u64 {
        let mut rng = Rng::new(0x7AC0 + seed);
        let seq = scene(&mut rng, 6);

        let linked = link_sequence(&seq, 8.0, UNIT).unwrap();
        assert_forest(&linked, &seq, 1, 8.0, UNIT);
        let total: usize = seq.iter().map(Vec::len).sum();
        assert_eq!(
            linked.nodes.len(),
            total,
            "seed {seed}: the linker drops no detection"
        );

        for cfg in [
            FlowConfig {
                max_um: 8.0,
                ..FlowConfig::default()
            },
            eager(),
        ] {
            let solved = flow_track(&seq, &cfg, UNIT).unwrap();
            assert_forest(&solved.graph, &seq, 2, 8.0, UNIT);
            // CleaveProofs §8: under strict calibration no optimum violates.
            assert!(
                solved.certified(),
                "seed {seed}: {} violations",
                solved.violations
            );
            forks += solved.graph.divisions.len();
        }
    }
    assert!(
        forks > 0,
        "no scene exercised a split; the out-degree bound went untested"
    );
}

#[test]
fn a_barely_calibrated_division_price_still_certifies() {
    // c_div one unit above c_app + c_det: a violating flow is only 1.0 dearer than
    // its reroute, so an inexact solver would hand one back. The exact one cannot.
    let cfg = FlowConfig {
        c_div: -39.0,
        max_um: 8.0,
        ..FlowConfig::default()
    };
    for seed in 0..12u64 {
        let seq = scene(&mut Rng::new(0xCE27 + seed), 6);
        let solved = flow_track(&seq, &cfg, UNIT).unwrap();
        assert!(
            solved.certified(),
            "seed {seed}: {} violations",
            solved.violations
        );
        assert_forest(&solved.graph, &seq, 2, 8.0, UNIT);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. Invariance
// ═══════════════════════════════════════════════════════════════════════════════

/// A proper rotation from three angles (z-y-x Euler), then a translation.
fn rigid(seq: &[Vec<Point>], angles: [f64; 3], shift: Point) -> Vec<Vec<Point>> {
    let (sa, ca) = angles[0].sin_cos();
    let (sb, cb) = angles[1].sin_cos();
    let (sc, cc) = angles[2].sin_cos();
    let r = [
        [ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
        [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
        [-sb, cb * sc, cb * cc],
    ];
    seq.iter()
        .map(|frame| {
            frame
                .iter()
                .map(|p| {
                    let mut q = shift;
                    for (i, row) in r.iter().enumerate() {
                        q[i] += row[0] * p[0] + row[1] * p[1] + row[2] * p[2];
                    }
                    q
                })
                .collect()
        })
        .collect()
}

#[test]
fn a_rigid_motion_of_every_frame_leaves_the_graph_unchanged() {
    for seed in 0..10u64 {
        let mut rng = Rng::new(0x51D + seed);
        let seq = scene(&mut rng, 6);
        let angles = [6.3 * rng.unit(), 6.3 * rng.unit(), 6.3 * rng.unit()];
        let shift = [
            500.0 * rng.normal(),
            500.0 * rng.normal(),
            500.0 * rng.normal(),
        ];
        let moved = rigid(&seq, angles, shift);

        assert_eq!(
            link_sequence(&seq, 8.0, UNIT).unwrap(),
            link_sequence(&moved, 8.0, UNIT).unwrap(),
            "seed {seed}: assignment changed under a rigid motion"
        );
        let (before, after) = (
            flow_track(&seq, &eager(), UNIT).unwrap(),
            flow_track(&moved, &eager(), UNIT).unwrap(),
        );
        assert_eq!(
            before.graph, after.graph,
            "seed {seed}: circulation changed"
        );
        assert_eq!(before.violations, after.violations);
    }
}

#[test]
fn permuting_detections_within_a_frame_permutes_the_graph() {
    for seed in 0..10u64 {
        let mut rng = Rng::new(0x9E7 + seed);
        let seq = scene(&mut rng, 6);
        // shuffled[t][i] = seq[t][perm[t][i]]
        let perm: Vec<Vec<usize>> = seq.iter().map(|f| rng.permutation(f.len())).collect();
        let shuffled: Vec<Vec<Point>> = seq
            .iter()
            .zip(&perm)
            .map(|(f, p)| p.iter().map(|&i| f[i]).collect())
            .collect();
        let back = |g: &TrackGraph| {
            let mut v: Vec<(Node, Node)> = edge_nodes(g)
                .into_iter()
                .map(|(a, b)| {
                    (
                        Node {
                            frame: a.frame,
                            index: perm[a.frame][a.index],
                        },
                        Node {
                            frame: b.frame,
                            index: perm[b.frame][b.index],
                        },
                    )
                })
                .collect();
            v.sort();
            v
        };

        let linked = link_sequence(&seq, 8.0, UNIT).unwrap();
        assert_eq!(
            edge_nodes(&linked),
            back(&link_sequence(&shuffled, 8.0, UNIT).unwrap()),
            "seed {seed}: assignment depends on detection order"
        );
        let solved = flow_track(&seq, &eager(), UNIT).unwrap();
        let solved_shuffled = flow_track(&shuffled, &eager(), UNIT).unwrap();
        assert_eq!(
            edge_nodes(&solved.graph),
            back(&solved_shuffled.graph),
            "seed {seed}: circulation depends on detection order"
        );
        assert_eq!(
            solved.graph.divisions.len(),
            solved_shuffled.graph.divisions.len()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 8. Refusals
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn non_finite_coordinates_are_refused_with_their_location() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let seq: Vec<Vec<Point>> = vec![
            vec![[0.0, 0.0, 0.0]],
            vec![[0.0, 0.0, 1.0], [0.0, bad, 2.0]],
        ];
        let want = TrackError::NonFiniteCoordinate { frame: 1, index: 1 };
        assert_eq!(link_sequence(&seq, 8.0, UNIT), Err(want));
        assert_eq!(
            flow_track(&seq, &FlowConfig::default(), UNIT).map(|r| r.graph),
            Err(want)
        );
        assert_eq!(link_frames(&seq[0], &seq[1], 8.0, UNIT), Err(want));
    }
}

#[test]
fn a_gate_or_spacing_that_is_not_finite_and_positive_is_refused() {
    let a: Vec<Point> = vec![[0.0, 0.0, 0.0]];
    let seq = vec![a.clone(), a.clone()];
    for gate in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            link_frames(&a, &a, gate, UNIT),
            Err(TrackError::InvalidGate)
        );
        assert_eq!(
            link_sequence(&seq, gate, UNIT),
            Err(TrackError::InvalidGate)
        );
        let cfg = FlowConfig {
            max_um: gate,
            ..FlowConfig::default()
        };
        assert_eq!(
            flow_track(&seq, &cfg, UNIT).map(|r| r.cost),
            Err(TrackError::InvalidGate)
        );
    }
    for scale in [[0.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, f64::NAN]] {
        assert_eq!(
            link_frames(&a, &a, 8.0, scale),
            Err(TrackError::InvalidScale)
        );
        let solved = flow_track(&seq, &FlowConfig::default(), scale);
        assert_eq!(solved.map(|r| r.cost), Err(TrackError::InvalidScale));
    }
}

#[test]
fn an_uncalibrated_division_price_is_refused_at_the_bound_exactly() {
    // c_app + c_det = -40 at the defaults. Equality is admissible (the reroute is
    // then cost-neutral, `reroute_not_worse`); one below it is not.
    let at = FlowConfig {
        c_div: -40.0,
        ..FlowConfig::default()
    };
    let below = FlowConfig {
        c_div: -40.5,
        ..FlowConfig::default()
    };
    assert_eq!(at.validate(), Ok(()));
    assert_eq!(below.validate(), Err(TrackError::Uncalibrated));
    let seq: Vec<Vec<Point>> = vec![vec![[0.0, 0.0, 0.0]]];
    assert_eq!(
        flow_track(&seq, &below, UNIT).map(|r| r.cost),
        Err(TrackError::Uncalibrated)
    );

    let none = FlowConfig {
        n_neighbours: 0,
        ..FlowConfig::default()
    };
    assert_eq!(none.validate(), Err(TrackError::InvalidNeighbours));
    for cfg in [
        FlowConfig {
            c_det: f64::NAN,
            ..FlowConfig::default()
        },
        FlowConfig {
            weight_scale: 0.0,
            ..FlowConfig::default()
        },
        FlowConfig {
            weight_scale: 1e12,
            ..FlowConfig::default()
        },
    ] {
        assert_eq!(cfg.validate(), Err(TrackError::InvalidCost), "{cfg:?}");
    }
}
