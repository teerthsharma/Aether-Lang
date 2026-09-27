//! Cell tracking on point detections: one-to-one frame linking, divisions by
//! min-cost circulation, and the certificate that makes the relaxation exact.
//!
//! Ported from `cleave` (Teerth Sharma), a tracker for 3D+time light-sheet
//! microscopy: the linker of `cleave/cleave/link.py`, the division-aware
//! circulation of `cleave/cleave/flow.py`, the division count of
//! `cleave/cleave/zebrahub.py::division_parents`, and the certificate lemmas of
//! `cleave/proofs/CleaveProofs.lean` sections 7 and 8. The input is a sequence
//! of frames, each a set of detection centroids in voxel coordinates `(z, y, x)`.
//! The output is a [`TrackGraph`]: nodes are `(frame, detection index)`, edges
//! run parent to child, and divisions are listed explicitly.
//!
//! # Metric
//!
//! Every distance is physical. With voxel spacing `s = (s_z, s_y, s_x)`,
//!
//! ```text
//! d(p, q) = \| \mathrm{diag}(s) (p - q) \|_2
//! ```
//!
//! On the benchmark grid ([`SCALE_UM`]) z is four times coarser than y and x, so
//! an unscaled distance would link along z four times too eagerly.
//!
//! # Linking rule: [`link_frames`], [`link_sequence`]
//!
//! Frames `A = {a_i}` (`n_A` points) and `B = {b_j}` (`n_B` points) are linked by
//! the optimal assignment on a padded cost matrix with gate `r` (`max_um`):
//!
//! ```text
//! C_{ij} = d(a_i, b_j)   for j < n_B,        C_{ij} = r   for n_B <= j < n_B + n_A,
//! \sigma^* = \arg\min_{\sigma\ \mathrm{injective}} \sum_i C_{i \sigma(i)},
//! E = \{ (i, \sigma^*(i)) : \sigma^*(i) < n_B,\ d(a_i, b_{\sigma^*(i)}) \le r \}.
//! ```
//!
//! The `n_A` dummy columns price ending a track at exactly `r`, so a real
//! successor is taken only when it is nearer than the gate, and the solver never
//! spends a cell on a partner the gate would discard. Equivalently, `E` is a
//! maximum-weight matching under weights `r - d(a_i, b_j)`. Assigning on the raw
//! matrix and gating afterwards is not equivalent: a distant pair chosen by the
//! solver blocks a nearer one, the gate then drops the distant pair, and one bad
//! assignment costs two links (`cleave/tests/test_link_gate.py`).
//!
//! # Split rule: [`flow_track`]
//!
//! Tracking with divisions is posed as a min-cost circulation on a node-split
//! graph (Zhang, Li and Nevatia, CVPR 2008, with a division in-arc added). Each
//! detection `x` becomes `x_in -> x_out`, with source `s` and sink `t`:
//!
//! ```text
//! arc              capacity   cost      meaning
//! x_in  -> x_out   1          c_det     the detection is used
//! s     -> x_in    1          c_app     a lineage starts at x
//! x_out -> t       1          c_dis     a lineage ends at x
//! s     -> x_out   1          c_div     the second daughter's unit enters at x
//! x_out -> y_in    1          d(x, y)   y in frame t+1, one of the k nearest, d <= r
//! t     -> s       unbounded  0         closes the circulation
//! ```
//!
//! The solution is the integral circulation `f` minimising `\sum_e c_e f_e`. A
//! node divides when its `x_out` emits two transitions, which is cleave's count of
//! a parent with exactly two distinct daughters:
//!
//! ```text
//! \mathrm{split}(x) \iff |\{ y : f(x_{out}, y_{in}) = 1 \}| = 2.
//! ```
//!
//! Holding the rest of the flow fixed, a second daughter `y` is attached to `x`
//! rather than started as a new lineage exactly when
//!
//! ```text
//! c_{div} + d(x, y) < c_{app},
//! ```
//!
//! which is 5 um under the default costs. A split needs no post-hoc detector: it
//! is the only way a node can carry two units.
//!
//! ## The certificate
//!
//! The constraint "a cell may only divide where a cell exists" couples flow on
//! different arcs and destroys the total unimodularity that makes a flow
//! relaxation integral (Haubold et al., ECCV 2016). It is checked on the answer
//! instead of encoded:
//!
//! ```text
//! \mathrm{violation}(x) \iff f(s, x_{out}) = 1 \wedge f(x_{in}, x_{out}) = 0,
//! \qquad \text{calibration: } c_{div} \ge c_{app} + c_{det}.
//! ```
//!
//! Rerouting a violating unit from `s -> x_out` onto `s -> x_in -> x_out` changes
//! the cost by `c_app + c_det - c_div <= 0`, and conservation at `x_in` (one
//! in-arc from `s`, one out-arc) forces both arcs of that path to be free
//! (`reroute_not_worse`, `conservation_frees_capacity` in `CleaveProofs.lean`).
//! Under strict calibration the reroute strictly improves the cost
//! (`reroute_strictly_better`), so no optimum violates the certificate and the
//! circulation optimum is the integer-program optimum. [`FlowConfig`] refuses a
//! configuration below the calibration bound, and [`TrackResult::violations`]
//! recomputes the count from the solved flow.
//!
//! ## Solver
//!
//! Costs are rounded to integers, `round(w c)` with `w = weight_scale`, exactly as
//! cleave rounds them for `networkx.network_simplex`. The circulation is solved as
//! a min-cost `s`-`t` flow by successive shortest paths (Bellman-Ford on the
//! residual graph, unit augmentations). The initial graph is acyclic because every
//! arc goes forward in time, so the shortest-path lengths are non-decreasing and
//! stopping at the first non-negative one gives the minimum-cost circulation.
//! Zero-cost augmentations are not taken, so ties resolve toward fewer units.
//!
//! # Guaranteed, by construction
//!
//! * In-degree at most 1 for every node, from both linkers: no merge events.
//! * Out-degree at most 1 from [`link_sequence`] (it cannot emit a division) and
//!   at most 2 from [`flow_track`]; [`TrackGraph::divisions`] lists exactly the
//!   nodes at 2.
//! * Every edge advances exactly one frame and has physical length at most `r`.
//!   The gap budget is one frame: a missing detection, or an empty frame, ends
//!   every track through it, and later frame indices are not renumbered.
//! * With in-degree at most 1 and edges strictly forward in time, the graph is a
//!   forest; each tree is one lineage.
//! * [`link_frames`] returns an exact optimum of the padded assignment;
//!   [`flow_track`] returns an exact optimum of the rounded-cost circulation over
//!   its candidate arcs, with zero violations whenever `c_div > c_app + c_det`.
//! * The output depends on positions only through `d`, so an isometry of `d`
//!   applied to every frame leaves it unchanged, and permuting detections within
//!   a frame permutes it, in both cases up to exact cost ties. Rotation plus
//!   translation is such an isometry when `s` is isotropic; on [`SCALE_UM`] only
//!   rotations in the `(y, x)` plane are.
//!
//! # Heuristic, not guaranteed
//!
//! * The costs themselves. `c_det`, `c_app`, `c_dis`, `c_div` and the gate `r` are
//!   modelling choices; `cleave/proofs/README.md` states that whether tracking
//!   should be this circulation at all is not proved. The assignment gate default
//!   of 8 um is another competitor's measurement recorded in `link.py`.
//! * The `k`-nearest candidate restriction. The true successor can be excluded,
//!   and the optimum is then over the restricted arc set.
//! * Rounding to `1 / w`. Costs closer than that are treated as equal.
//! * Whether a fork is a biological division. The rule prices a geometric
//!   configuration; it does not observe mitosis.
//!
//! # Refusals
//!
//! A non-finite coordinate ([`TrackError::NonFiniteCoordinate`]), a gate that is
//! not finite and positive ([`TrackError::InvalidGate`]), a spacing component that
//! is not finite and positive ([`TrackError::InvalidScale`]), `c_div` below
//! `c_app + c_det` ([`TrackError::Uncalibrated`]), `n_neighbours = 0`
//! ([`TrackError::InvalidNeighbours`]), and a non-finite cost, a non-positive
//! `weight_scale`, or a rounded cost outside `i32` ([`TrackError::InvalidCost`];
//! the bound keeps every path sum exact in `i64`). Empty frames and an empty
//! sequence are accepted, as in cleave: an empty frame must be passed rather than
//! omitted, because omitting it would shift every later frame in time.
//!
//! # Not ported
//!
//! Gap closing: cleave implements none, deliberately (`link.py` records it as
//! measured neutral), so "join" here is the frame-to-frame link and nothing
//! more. The image-side split evidence (`euler3d.division_signature`,
//! `persist.h0_barcode`) is an H0 filtration over voxel intensities; its input is
//! an image, not points, so it is outside this module, and the linker has no
//! filtration that [`crate::persistence`] could supply.

#![warn(missing_docs)]

use alloc::vec;
use alloc::vec::Vec;

use libm::{fabs, round, sqrt};

/// Voxel spacing of the Biohub benchmark grid, micrometres per voxel along
/// `(z, y, x)`. Ported from `cleave/cleave/euler3d.py::SCALE_UM`.
pub const SCALE_UM: [f64; 3] = [1.625, 0.40625, 0.40625];

/// A detection centroid `(z, y, x)` in voxels.
pub type Point = [f64; 3];

/// One detection: its frame and its index within that frame's input slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Node {
    /// Frame (timepoint) index, the position in the input sequence.
    pub frame: usize,
    /// Index of the detection within its frame.
    pub index: usize,
}

/// Detections and their temporal links. Ported from `cleave/cleave/link.py::TrackGraph`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrackGraph {
    /// Nodes in `(frame, index)` order.
    pub nodes: Vec<Node>,
    /// `(parent, child)` indices into `nodes`, sorted; the child is always one
    /// frame after the parent.
    pub edges: Vec<(usize, usize)>,
    /// Indices into `nodes` of every node with exactly two children, sorted: the
    /// division events. A parent with two distinct daughters, the count
    /// `cleave/cleave/zebrahub.py::division_parents` makes.
    pub divisions: Vec<usize>,
}

/// Why a tracking call was refused.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrackError {
    /// A coordinate is NaN or infinite; it cannot be ordered by distance.
    NonFiniteCoordinate {
        /// Frame of the offending detection (0 or 1 for [`link_frames`]).
        frame: usize,
        /// Index of the offending detection within its frame.
        index: usize,
    },
    /// The gate `max_um` is not finite and positive.
    InvalidGate,
    /// A voxel-spacing component is not finite and positive.
    InvalidScale,
    /// `c_div < c_app + c_det`: the certificate has no reason to hold.
    Uncalibrated,
    /// `n_neighbours` is zero, which leaves no transition arcs.
    InvalidNeighbours,
    /// A cost is not finite, `weight_scale` is not finite and positive, or a
    /// rounded cost does not fit in `i32`.
    InvalidCost,
}

/// Physical distance between two voxel coordinates: `|| diag(scale) (a - b) ||_2`.
///
/// Ported from `cleave/cleave/link.py::physical_distances`.
pub fn physical_distance(a: Point, b: Point, scale: [f64; 3]) -> f64 {
    let mut sum = 0.0;
    for k in 0..3 {
        let v = (a[k] - b[k]) * scale[k];
        sum += v * v;
    }
    sqrt(sum)
}

/// One-to-one links from frame `a` to frame `b`: the optimal assignment with
/// non-assignment priced at the gate `max_um`, returned as `(i, j)` pairs sorted
/// by `i`, every pair within the gate.
///
/// Ported from `cleave/cleave/link.py::link_frames`. See the module documentation
/// for the padded cost matrix. An empty frame on either side yields no pairs.
pub fn link_frames(
    a: &[Point],
    b: &[Point],
    max_um: f64,
    scale: [f64; 3],
) -> Result<Vec<(usize, usize)>, TrackError> {
    check_gate(max_um)?;
    check_scale(scale)?;
    check_points(0, a)?;
    check_points(1, b)?;
    if a.is_empty() || b.is_empty() {
        return Ok(Vec::new());
    }

    let (n_a, n_b) = (a.len(), b.len());
    let mut d = Vec::with_capacity(n_a * n_b);
    for p in a {
        for q in b {
            d.push(physical_distance(*p, *q, scale));
        }
    }
    // One dummy column per row, priced at the gate: ending a track costs exactly
    // `max_um`, so a real successor is taken only when it is nearer than that.
    let columns = assign(n_a, n_b + n_a, |i, j| {
        if j < n_b {
            d[i * n_b + j]
        } else {
            max_um
        }
    });
    // Still filtered: a pair exactly at `max_um` ties with its dummy, and the
    // comparison keeps that boundary where cleave keeps it.
    Ok(columns
        .into_iter()
        .enumerate()
        .filter(|&(i, j)| j < n_b && d[i * n_b + j] <= max_um)
        .collect())
}

/// Link a sequence of frames by [`link_frames`] between each consecutive pair.
///
/// Every detection becomes a node. The result has in- and out-degree at most 1,
/// so it never contains a division. Ported from `cleave/cleave/link.py::link_sequence`.
pub fn link_sequence<F: AsRef<[Point]>>(
    frames: &[F],
    max_um: f64,
    scale: [f64; 3],
) -> Result<TrackGraph, TrackError> {
    check_gate(max_um)?;
    check_scale(scale)?;
    let mut nodes = Vec::new();
    let mut offsets = Vec::with_capacity(frames.len());
    for (t, frame) in frames.iter().enumerate() {
        check_points(t, frame.as_ref())?;
        offsets.push(nodes.len());
        nodes.extend((0..frame.as_ref().len()).map(|index| Node { frame: t, index }));
    }

    let mut edges = Vec::new();
    for t in 1..frames.len() {
        for (i, j) in link_frames(frames[t - 1].as_ref(), frames[t].as_ref(), max_um, scale)? {
            edges.push((offsets[t - 1] + i, offsets[t] + j));
        }
    }
    Ok(finish(nodes, edges))
}

/// Arc costs of the division-aware circulation, in the units of the physical
/// distance. Ported from `cleave/cleave/flow.py::FlowConfig`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowConfig {
    /// Cost of using a detection. Negative, or the empty circulation is optimal.
    pub c_det: f64,
    /// Cost of starting a lineage.
    pub c_app: f64,
    /// Cost of ending a lineage.
    pub c_dis: f64,
    /// Cost of a division. Must satisfy `c_div >= c_app + c_det`.
    pub c_div: f64,
    /// Gate on transition length, in micrometres.
    pub max_um: f64,
    /// Transition arcs per detection: the nearest this many within the gate.
    pub n_neighbours: usize,
    /// Integer scaling applied to every cost before the solve.
    pub weight_scale: f64,
}

impl Default for FlowConfig {
    /// cleave's defaults: `c_det = -50`, `c_app = c_dis = 10`, `c_div = 5`,
    /// a 12 um gate, 5 neighbours, costs scaled by 1000.
    fn default() -> Self {
        Self {
            c_det: -50.0,
            c_app: 10.0,
            c_dis: 10.0,
            c_div: 5.0,
            max_um: 12.0,
            n_neighbours: 5,
            weight_scale: 1000.0,
        }
    }
}

impl FlowConfig {
    /// Refuse a configuration the solver or the certificate cannot vouch for.
    ///
    /// The calibration check is the one cleave's `FlowConfig.__post_init__`
    /// makes; the finiteness and `i32` range checks keep the integer solve exact.
    pub fn validate(&self) -> Result<(), TrackError> {
        check_gate(self.max_um)?;
        if self.n_neighbours == 0 {
            return Err(TrackError::InvalidNeighbours);
        }
        let w = self.weight_scale;
        if !w.is_finite() || w <= 0.0 {
            return Err(TrackError::InvalidCost);
        }
        for c in [self.c_det, self.c_app, self.c_dis, self.c_div, self.max_um] {
            if !c.is_finite() || fabs(round(w * c)) > i32::MAX as f64 {
                return Err(TrackError::InvalidCost);
            }
        }
        if self.c_div < self.c_app + self.c_det {
            return Err(TrackError::Uncalibrated);
        }
        Ok(())
    }
}

/// A solved track graph and the evidence that the solve was admissible.
/// Ported from `cleave/cleave/flow.py::TrackResult`.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackResult {
    /// Used detections only, with the solved transitions as edges.
    pub graph: TrackGraph,
    /// Nodes whose division arc carries flow while their detection arc does not.
    pub violations: usize,
    /// Total cost of the circulation, in unscaled units.
    pub cost: f64,
}

impl TrackResult {
    /// True when no node divided where its detection was unused.
    pub fn certified(&self) -> bool {
        self.violations == 0
    }
}

/// Link a sequence of frames into a lineage forest by min-cost circulation.
///
/// Only detections the solver uses become nodes. Transitions join each detection
/// to its `n_neighbours` nearest detections in the next frame within `max_um`
/// (ties by index). Ported from `cleave/cleave/flow.py::flow_track`; the brute
/// candidate search is the full-matrix form cleave's KD-tree query replaced and is
/// documented there as bit-identical to it.
pub fn flow_track<F: AsRef<[Point]>>(
    frames: &[F],
    config: &FlowConfig,
    scale: [f64; 3],
) -> Result<TrackResult, TrackError> {
    config.validate()?;
    check_scale(scale)?;
    let mut offsets = Vec::with_capacity(frames.len());
    let mut n_det = 0;
    for (t, frame) in frames.iter().enumerate() {
        check_points(t, frame.as_ref())?;
        offsets.push(n_det);
        n_det += frame.as_ref().len();
    }

    let w = config.weight_scale;
    let q = |c: f64| round(w * c) as i64;
    let (s, t) = (0, 1);
    let x_in = |n: usize| 2 + 2 * n;
    let x_out = |n: usize| 3 + 2 * n;

    let mut net = Network::new(2 * n_det + 2);
    let mut det_arc = Vec::with_capacity(n_det);
    let mut div_arc = Vec::with_capacity(n_det);
    for n in 0..n_det {
        det_arc.push(net.arc(x_in(n), x_out(n), q(config.c_det)));
        net.arc(s, x_in(n), q(config.c_app));
        net.arc(x_out(n), t, q(config.c_dis));
        div_arc.push(net.arc(s, x_out(n), q(config.c_div)));
    }

    let mut transitions = Vec::new();
    for f in 1..frames.len() {
        let (a, b) = (frames[f - 1].as_ref(), frames[f].as_ref());
        for (j, p) in a.iter().enumerate() {
            let mut near: Vec<(f64, usize)> = b
                .iter()
                .enumerate()
                .map(|(k, y)| (physical_distance(*p, *y, scale), k))
                .filter(|&(d, _)| d <= config.max_um)
                .collect();
            near.sort_by(|u, v| u.0.total_cmp(&v.0).then(u.1.cmp(&v.1)));
            near.truncate(config.n_neighbours);
            for (d, k) in near {
                let (n, m) = (offsets[f - 1] + j, offsets[f] + k);
                transitions.push((net.arc(x_out(n), x_in(m), q(d)), n, m));
            }
        }
    }

    let total = net.min_cost_flow(s, t);

    let used: Vec<bool> = det_arc.iter().map(|&e| net.carries(e)).collect();
    let violations = (0..n_det)
        .filter(|&n| net.carries(div_arc[n]) && !used[n])
        .count();

    let mut row = vec![usize::MAX; n_det];
    let mut nodes = Vec::new();
    for (f, frame) in frames.iter().enumerate() {
        for index in 0..frame.as_ref().len() {
            let n = offsets[f] + index;
            if used[n] {
                row[n] = nodes.len();
                nodes.push(Node { frame: f, index });
            }
        }
    }
    // An edge out of an unused detection can only exist at a violating node; like
    // cleave, keep edges between used detections only.
    let edges = transitions
        .into_iter()
        .filter(|&(e, n, m)| net.carries(e) && used[n] && used[m])
        .map(|(_, n, m)| (row[n], row[m]))
        .collect();

    Ok(TrackResult {
        graph: finish(nodes, edges),
        violations,
        cost: total as f64 / w,
    })
}

fn check_gate(max_um: f64) -> Result<(), TrackError> {
    if max_um.is_finite() && max_um > 0.0 {
        Ok(())
    } else {
        Err(TrackError::InvalidGate)
    }
}

fn check_scale(scale: [f64; 3]) -> Result<(), TrackError> {
    if scale.iter().all(|s| s.is_finite() && *s > 0.0) {
        Ok(())
    } else {
        Err(TrackError::InvalidScale)
    }
}

fn check_points(frame: usize, points: &[Point]) -> Result<(), TrackError> {
    match points.iter().position(|p| !p.iter().all(|c| c.is_finite())) {
        Some(index) => Err(TrackError::NonFiniteCoordinate { frame, index }),
        None => Ok(()),
    }
}

/// Sort the edges and read the division events off the out-degrees.
fn finish(nodes: Vec<Node>, mut edges: Vec<(usize, usize)>) -> TrackGraph {
    edges.sort_unstable();
    let mut out = vec![0usize; nodes.len()];
    for &(parent, _) in &edges {
        out[parent] += 1;
    }
    let divisions = (0..nodes.len()).filter(|&v| out[v] == 2).collect();
    TrackGraph {
        nodes,
        edges,
        divisions,
    }
}

/// Rectangular assignment, `n <= m`: the column given to each row, minimising
/// the total `cost(row, column)`.
///
/// The potentials form of the Hungarian method, as in
/// `crate::diagram::hungarian_min_cost`, which is square and returns only the
/// cost; this one is rectangular and returns the assignment.
fn assign(n: usize, m: usize, cost: impl Fn(usize, usize) -> f64) -> Vec<usize> {
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; m + 1];
    let mut owner = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];

    for i in 1..=n {
        owner[0] = i;
        let mut j0 = 0usize;
        let mut min_cost = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = owner[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=m {
                if used[j] {
                    continue;
                }
                let current = cost(i0 - 1, j - 1) - u[i0] - v[j];
                if current < min_cost[j] {
                    min_cost[j] = current;
                    way[j] = j0;
                }
                if min_cost[j] < delta {
                    delta = min_cost[j];
                    j1 = j;
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[owner[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_cost[j] -= delta;
                }
            }
            j0 = j1;
            if owner[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            owner[j0] = owner[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut columns = vec![0usize; n];
    for j in 1..=m {
        if owner[j] != 0 {
            columns[owner[j] - 1] = j - 1;
        }
    }
    columns
}

/// Unit-capacity residual network. Arc `e` and its reverse `e ^ 1` are stored
/// together; the reverse starts with no residual capacity.
struct Network {
    from: Vec<usize>,
    to: Vec<usize>,
    cost: Vec<i64>,
    residual: Vec<u8>,
    size: usize,
}

impl Network {
    fn new(size: usize) -> Self {
        Self {
            from: Vec::new(),
            to: Vec::new(),
            cost: Vec::new(),
            residual: Vec::new(),
            size,
        }
    }

    fn arc(&mut self, u: usize, v: usize, cost: i64) -> usize {
        let id = self.from.len();
        self.from.extend([u, v]);
        self.to.extend([v, u]);
        self.cost.extend([cost, -cost]);
        self.residual.extend([1, 0]);
        id
    }

    fn carries(&self, arc: usize) -> bool {
        self.residual[arc] == 0
    }

    /// Successive shortest paths from `s` to `t`, one unit at a time, stopping
    /// at the first path whose cost is not negative. Returns the total cost.
    ///
    /// ponytail: Bellman-Ford per augmentation is O(V E) and there are O(V)
    /// augmentations. Replace with Dijkstra on Johnson-reduced costs when frames
    /// carry hundreds of detections; the stopping rule does not change.
    fn min_cost_flow(&mut self, s: usize, t: usize) -> i64 {
        let mut total = 0i64;
        loop {
            let mut dist = vec![i64::MAX; self.size];
            let mut pred = vec![usize::MAX; self.size];
            dist[s] = 0;
            for _ in 0..self.size {
                let mut changed = false;
                for e in 0..self.from.len() {
                    let (a, b) = (self.from[e], self.to[e]);
                    if self.residual[e] > 0
                        && dist[a] != i64::MAX
                        && dist[a] + self.cost[e] < dist[b]
                    {
                        dist[b] = dist[a] + self.cost[e];
                        pred[b] = e;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
            // Unreachable is `i64::MAX`, which is not negative either.
            if dist[t] >= 0 {
                return total;
            }
            total += dist[t];
            let mut node = t;
            while node != s {
                let e = pred[node];
                self.residual[e] -= 1;
                self.residual[e ^ 1] += 1;
                node = self.from[e];
            }
        }
    }
}
