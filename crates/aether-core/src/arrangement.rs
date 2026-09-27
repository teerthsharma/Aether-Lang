//! Certified integer invariants of a planar segment arrangement.
//!
//! Given a finite set of straight segments in the plane, [`arrange`] returns
//! three integers — the number of connected pieces, the number of bounded faces
//! the segments enclose, and the Euler characteristic of their union — together
//! with the snap window that decided which endpoints are one vertex, or a typed
//! [`Refusal`] naming the condition that could not be established.
//!
//! Ported from `planimeter` (Teerth Sharma): `planimeter/count.py` (the counting
//! identity), `planimeter/snap.py` (the snap window) and `planimeter/arrange.py`
//! (the arrangement preconditions and the window-selection loop). The file
//! readers, curve flattening, the digest and the JSON surface are not ported;
//! the input here is a segment list.
//!
//! # Counting
//!
//! For a graph embedded in the plane with `V` vertices, `E` edges and `C`
//! connected components, Euler's formula reads
//!
//! ```text
//! V − E + F = 1 + C
//! ```
//!
//! where `F` counts every face including the single unbounded one. The enclosed
//! faces, the Euler characteristic of the union, and the identity joining them
//! are therefore
//!
//! ```text
//! faces = F − 1 = E − V + C        (the first Betti number of the 1-complex)
//! χ     = V − E = pieces − faces
//! ```
//!
//! `C` ("pieces") is computed by union-find over the final edge list. Given a
//! correct, deduplicated edge list this step is exact integer arithmetic; the
//! entire difficulty lies in producing that edge list.
//!
//! # The snap rule
//!
//! 1. **Exact identification.** Bitwise-equal coordinates (with `−0.0 = +0.0`)
//!    are one point. No tolerance is involved.
//! 2. **Separation spectrum.** Let `M = max |coordinate|` over the distinct
//!    points (`1` if that is zero) and `δ = 4096 · 2⁻⁵² · M`, the
//!    representability floor. The spectrum is the sorted, duplicate-free set
//!
//!    ```text
//!    S = {δ} ∪ {w : w an EMST edge weight, w > δ}
//!            ∪ {d(p, s) : s an input segment, p a point not an endpoint of s, d > δ}
//!      = {s₀ = δ < s₁ < … < s_K}
//!    ```
//!
//!    By Gower and Ross (1969) the single-linkage partition `π(r)` — the
//!    components of the graph joining every pair of points within `r` — changes
//!    only at Euclidean-MST edge weights, so `π(r)` is constant on every
//!    `[s_k, s_{k+1})`. The vertex-to-segment distances enter so that a wall end
//!    missing a floor by a tiny margin shows up as a separation even though no
//!    two points are close.
//! 3. **Candidate windows.** `[s_k, s_{k+1})` is a candidate when
//!    `s_{k+1} / s_k ≥ ρ` (default [`RHO`] `= 10`). Genuine gaps (`k ≥ 1`) are
//!    tried in decreasing ratio; the merge-nothing window (`k = 0`) is tried
//!    last, because its ratio is drawing scale over machine epsilon for
//!    arithmetic reasons alone. At most [`CAND_MAX`] windows are tried, and a
//!    window whose partition has a cluster of more than [`CLUSTER_MAX`] points is
//!    skipped.
//! 4. **Partition and radius.** Clusters are the components of the EMST edges of
//!    weight `≤ t_below`; each is represented by its lexicographically least
//!    point, so no coordinate is ever created. The reported radius is the
//!    log-midpoint `√(t_below · t_above)`. With a user-supplied `grid = r` the
//!    single window is the `[s_k, s_{k+1})` containing `r`, subject to the same
//!    ratio and cluster conditions, and the reported radius is `r`.
//!
//! # Preconditions checked in each window
//!
//! - **Margin.** `t_above > 64 · 2⁻⁵² · M`.
//! - **P1, every edge survives.** No input segment has both endpoints in one
//!   cluster.
//! - Edges between clusters are deduplicated: under P3 two straight edges on one
//!   vertex pair are the same segment, so collapsing them is an identity.
//! - **P2, robustness.** Every (vertex, non-incident edge) distance is exactly
//!   `0` or at least `t_above`. Distance `0` subdivides the edge at that
//!   existing vertex; a distance in `(0, t_above)` refuses.
//! - **P3, plane embedding.** After subdivision and deduplication, no two edges
//!   without a shared vertex properly cross.
//!
//! A margin or P1 failure says the *window* is wrong for this drawing, and the
//! next candidate is tried. A P2 or P3 failure says the *drawing* is ambiguous or
//! unnoded, and ends the run: a finer window would read the near miss as a clean
//! miss, and returning that would be choosing whichever reading certifies. If
//! every window fails, the first window's refusal is returned.
//!
//! No intersection point is ever computed. Two segments crossing at a point the
//! input does not contain is a refusal, not a new vertex: a square with both
//! diagonals drawn as two crossing segments refuses, and the same figure drawn as
//! four half-diagonals meeting at a written centre has four faces.
//!
//! # Certified versus heuristic
//!
//! Certified, for a returned [`Chi`]: `π(r)` is constant for every
//! `r ∈ [t_below, t_above)`; `t_above / t_below ≥ ρ`; by Kruskal's cut property
//! the two closest distinct clusters are exactly the next merge height apart,
//! which is at least `t_above` (equal to it unless `t_above` is a vertex-to-edge
//! distance); the margin, P1, P2 and P3 were checked rather than assumed; and the
//! three integers are exact for the resulting graph.
//!
//! Not certified: that the identification is the one the drawing's author
//! intended — the window is *stable*, not *right*. Uniqueness is not established:
//! at most [`CAND_MAX`] windows are tried and the first that passes wins, so a
//! later window might also pass with different integers. Two 10-unit squares
//! exactly 1 apart read as one piece, 1.01 apart as two. [`RHO`], [`CAND_MAX`],
//! [`FLOOR_ULPS`] and [`CLUSTER_MAX`] are chosen policy constants, not theorems.
//! The margin condition rests on the source's float64 argument (predicate error a
//! small multiple of `2⁻⁵² · M`), not on exact arithmetic. `faces` counts
//! arrangement faces, not the regions a renderer fills: two overlapping
//! rectangles, noded at their crossings, are three faces where a renderer fills
//! two regions.
//!
//! # Refusals
//!
//! Every condition under which [`arrange`] returns a [`Refusal`] instead of
//! integers:
//!
//! - [`Refusal::NonFinite`] — a coordinate is NaN or infinite.
//! - [`Refusal::InvalidGrid`] — a supplied grid radius is NaN or infinite.
//! - [`Refusal::NoGeometry`] — the segment list is empty.
//! - [`Refusal::TooManyVertices`] — more distinct endpoints than `max_vertices`.
//! - [`Refusal::TooManyPairs`] — a quadratic pass would exceed
//!   `4 · max_vertices²` pairs.
//! - [`Refusal::NoStableScale`] — no candidate window exists, or a supplied grid
//!   lies below the floor, at or above every separation, in a gap narrower than
//!   `ρ`, or in a window with a cluster above [`CLUSTER_MAX`].
//! - [`Refusal::MarginTooSmall`] — `t_above` is not above 64 ulps of `M`.
//! - [`Refusal::EdgeCollapsed`] — P1 failed.
//! - [`Refusal::VertexNearEdge`] — P2 failed.
//! - [`Refusal::EdgesCross`] — P3 failed.
//!
//! # Deviations from the source
//!
//! The point-to-segment distance is evaluated with the segment oriented from its
//! lexicographically smaller endpoint, so it is a function of the unordered
//! segment. The source orients by input order in the spectrum and by cluster
//! label in P2, which can differ in the last ulp; here P2 recomputes exactly the
//! values the spectrum holds, and reversing a segment changes nothing. Every
//! pair-budget overrun is reported as [`Refusal::TooManyPairs`]; the source
//! reports the one in the spectrum pass as `TOO_MANY_VERTICES`. The source's
//! `CURVE_UNSTABLE` refusal belongs to curve flattening in its reader and has no
//! counterpart here.

#![warn(missing_docs)]

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use libm::{fabs, hypot, sqrt};

/// Policy: the minimum `t_above / t_below` ratio a window may have.
pub const RHO: f64 = 10.0;
/// Policy: how many candidate windows are tried, widest ratio first.
pub const CAND_MAX: usize = 4;
/// Policy: the representability floor, in ulps of the drawing's magnitude.
pub const FLOOR_ULPS: f64 = 4096.0;
/// Policy: the largest cluster of endpoints treated as one vertex.
pub const CLUSTER_MAX: usize = 16;
/// Budget: the default ceiling on distinct endpoints.
pub const BRUTE_MAX: usize = 2000;
/// The float64 margin, in ulps of the drawing's magnitude, above which the
/// source's argument makes every incidence sign correct.
pub const MARGIN_ULPS: f64 = 64.0;

/// A straight segment as its two endpoints, `[[x0, y0], [x1, y1]]`.
pub type Segment = [[f64; 2]; 2];

type Point = [f64; 2];

/// An EMST edge as `(weight, a, b)`.
type TreeEdge = (f64, usize, usize);

/// Inputs to [`arrange`] beyond the segments themselves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArrangementConfig {
    /// Minimum window ratio `t_above / t_below`. Default [`RHO`].
    pub rho: f64,
    /// How many candidate windows are tried. Default [`CAND_MAX`].
    pub max_candidates: usize,
    /// A user-supplied snap radius. `None` derives the window from the
    /// spectrum; `Some(r)` uses the window containing `r`, verified the same way.
    pub grid: Option<f64>,
    /// Refuses inputs with more distinct endpoints than this, and bounds every
    /// quadratic pass at `4 · max_vertices²` pairs. Default [`BRUTE_MAX`].
    pub max_vertices: usize,
}

impl Default for ArrangementConfig {
    fn default() -> Self {
        Self {
            rho: RHO,
            max_candidates: CAND_MAX,
            grid: None,
            max_vertices: BRUTE_MAX,
        }
    }
}

/// Who chose the snap radius.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadiusSource {
    /// Derived from the separation spectrum.
    Derived,
    /// Supplied through [`ArrangementConfig::grid`].
    User,
}

/// A certified count, with the window that decided it.
///
/// `faces = edges − vertices + pieces` and `chi = pieces − faces = vertices − edges`
/// hold by construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chi {
    /// Vertices of the final graph: clusters of endpoints.
    pub vertices: usize,
    /// Edges of the final graph, after subdivision and deduplication.
    pub edges: usize,
    /// Connected components.
    pub pieces: usize,
    /// Bounded faces, `E − V + C`.
    pub faces: usize,
    /// Euler characteristic of the union, `V − E`.
    pub chi: i64,
    /// Vertices of degree one.
    pub dangles: usize,
    /// Lower end of the snap window.
    pub t_below: f64,
    /// Upper end of the snap window: no two distinct clusters are closer, and
    /// no vertex lies strictly between `0` and this from an edge.
    pub t_above: f64,
    /// `t_above / t_below`.
    pub ratio: f64,
    /// The snap radius: `√(t_below · t_above)`, or the user's grid.
    pub radius: f64,
    /// Whether the radius was derived or supplied.
    pub radius_source: RadiusSource,
    /// Distinct endpoints merged into another: `points − clusters`.
    pub n_merged: usize,
    /// Edges subdivided at an exactly incident vertex.
    pub n_subdivided: usize,
    /// Edges removed as duplicates of another edge on the same vertex pair.
    pub n_dup_edges: usize,
    /// Candidate windows available when this one was chosen.
    pub n_candidates: usize,
}

/// Whether a refusal is about the drawing, the machine, or the call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalKind {
    /// Re-observe the drawing.
    Geometry,
    /// A work budget was exceeded; the drawing was not judged.
    Budget,
    /// The input could not be read; nothing was computed.
    Input,
}

/// One named condition that could not be established.
///
/// A refusal is a statement about the evidence, not about the geometry. Segment
/// indices refer to the caller's slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Refusal {
    /// The segment list is empty.
    NoGeometry,
    /// Coordinates that are NaN or infinite.
    NonFinite {
        /// How many of the `4m` coordinates are not finite.
        values: usize,
    },
    /// The supplied grid radius is NaN or infinite.
    InvalidGrid,
    /// No window with ratio at least `ρ` could be used.
    NoStableScale {
        /// The widest gap in the spectrum as `[t_below, t_above]`, or for a
        /// supplied grid the gap it landed in; `None` when there is no such gap.
        gap: Option<[f64; 2]>,
    },
    /// `t_above` is inside float64 noise at the drawing's magnitude.
    MarginTooSmall {
        /// The window's upper end.
        t_above: f64,
        /// `64 · 2⁻⁵² · M`.
        margin: f64,
    },
    /// An input segment has both endpoints in one cluster.
    EdgeCollapsed {
        /// The first such segment.
        segment: usize,
        /// How many segments collapsed.
        count: usize,
    },
    /// A vertex lies strictly between `0` and `t_above` from an edge it is not
    /// an endpoint of.
    VertexNearEdge {
        /// The closest such vertex.
        vertex: [f64; 2],
        /// The input segment that owns the edge it is near.
        segment: usize,
        /// Its distance from that edge.
        distance: f64,
        /// How many (vertex, edge) pairs sit in the band.
        count: usize,
    },
    /// Two edges cross at a point the input does not contain.
    EdgesCross {
        /// Input segment owning the first edge of the first crossing pair.
        first: usize,
        /// Input segment owning the second edge.
        second: usize,
        /// How many edge pairs cross.
        count: usize,
    },
    /// More distinct endpoints than `max_vertices`.
    TooManyVertices {
        /// Distinct endpoints.
        actual: usize,
        /// The ceiling.
        max: usize,
    },
    /// A quadratic pass would exceed `4 · max_vertices²` pairs.
    TooManyPairs {
        /// Pairs the pass would evaluate.
        pairs: usize,
        /// The budget.
        max: usize,
    },
}

impl Refusal {
    /// Which kind of refusal this is.
    ///
    /// Ported from `planimeter/planimeter/result.py` (`KIND_OF`).
    pub fn kind(&self) -> RefusalKind {
        match self {
            Refusal::NonFinite { .. } | Refusal::InvalidGrid => RefusalKind::Input,
            Refusal::TooManyVertices { .. } | Refusal::TooManyPairs { .. } => RefusalKind::Budget,
            _ => RefusalKind::Geometry,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Union-find and the counting identity (count.py)
// ═══════════════════════════════════════════════════════════════════════════════

/// Union-find with path halving and union by size.
struct Dsu {
    parent: Vec<usize>,
    size: Vec<usize>,
    sets: usize,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
            sets: n,
        }
    }

    fn find(&mut self, mut a: usize) -> usize {
        while self.parent[a] != a {
            self.parent[a] = self.parent[self.parent[a]];
            a = self.parent[a];
        }
        a
    }

    fn union(&mut self, a: usize, b: usize) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            core::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
        self.sets -= 1;
    }

    /// Component id per element, numbered by first appearance so the result
    /// does not depend on union order.
    fn labels(&mut self) -> Vec<usize> {
        let mut seen = BTreeMap::new();
        (0..self.parent.len())
            .map(|i| {
                let root = self.find(i);
                let next = seen.len();
                *seen.entry(root).or_insert(next)
            })
            .collect()
    }
}

struct Counts {
    vertices: usize,
    edges: usize,
    pieces: usize,
    faces: usize,
    chi: i64,
    dangles: usize,
}

/// `faces = E − V + C`, `χ = V − E`. Edges are distinct, loop-free index pairs.
fn count(vertices: usize, edges: &[[usize; 2]]) -> Counts {
    let mut dsu = Dsu::new(vertices);
    let mut degree = vec![0usize; vertices];
    for &[a, b] in edges {
        degree[a] += 1;
        degree[b] += 1;
        dsu.union(a, b);
    }
    Counts {
        vertices,
        edges: edges.len(),
        pieces: dsu.sets,
        faces: edges.len() + dsu.sets - vertices,
        chi: vertices as i64 - edges.len() as i64,
        dangles: degree.iter().filter(|&&d| d == 1).count(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Geometry
// ═══════════════════════════════════════════════════════════════════════════════

fn lex_less(a: Point, b: Point) -> bool {
    a[0] < b[0] || (a[0] == b[0] && a[1] < b[1])
}

fn dist(a: Point, b: Point) -> f64 {
    hypot(a[0] - b[0], a[1] - b[1])
}

/// Distance from `p` to the segment `ab`, and the clamped parameter of the
/// nearest point measured from the lexicographically smaller endpoint.
fn point_segment(p: Point, a: Point, b: Point) -> (f64, f64) {
    let (a, b) = if lex_less(b, a) { (b, a) } else { (a, b) };
    let ab = [b[0] - a[0], b[1] - a[1]];
    let den = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if den > 0.0 {
        (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / den).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (dist(p, [a[0] + t * ab[0], a[1] + t * ab[1]]), t)
}

fn cross(u: Point, v: Point) -> f64 {
    u[0] * v[1] - u[1] * v[0]
}

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1]]
}

/// `M = max |coordinate|`, or `1` for a drawing at the origin.
fn magnitude(pts: &[Point]) -> f64 {
    let m = pts
        .iter()
        .flatten()
        .fold(0.0f64, |m, &x| if fabs(x) > m { fabs(x) } else { m });
    if m == 0.0 {
        1.0
    } else {
        m
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// The snap window (snap.py)
// ═══════════════════════════════════════════════════════════════════════════════

/// Collapse bitwise-equal coordinates. Returns the distinct points and each
/// segment's endpoint indices.
fn dedup_exact(segments: &[Segment]) -> (Vec<Point>, Vec<[usize; 2]>) {
    let mut seen: BTreeMap<(u64, u64), usize> = BTreeMap::new();
    let mut pts = Vec::new();
    let mut index = |p: Point| {
        // `+ 0.0` maps −0.0 to +0.0: the same point written two ways.
        let key = ((p[0] + 0.0).to_bits(), (p[1] + 0.0).to_bits());
        *seen.entry(key).or_insert_with(|| {
            pts.push(p);
            pts.len() - 1
        })
    };
    let ends = segments
        .iter()
        .map(|s| [index(s[0]), index(s[1])])
        .collect();
    (pts, ends)
}

/// Exact Euclidean MST by all-pairs Prim, O(n²).
fn emst(pts: &[Point]) -> Vec<TreeEdge> {
    let n = pts.len();
    let mut inside = vec![false; n];
    let mut best = vec![f64::INFINITY; n];
    let mut src = vec![0usize; n];
    let mut out = Vec::with_capacity(n.saturating_sub(1));
    if n > 0 {
        best[0] = 0.0;
    }
    for k in 0..n {
        let mut u = usize::MAX;
        for i in (0..n).filter(|&i| !inside[i]) {
            if u == usize::MAX || best[i] < best[u] {
                u = i;
            }
        }
        inside[u] = true;
        if k > 0 {
            out.push((dist(pts[src[u]], pts[u]), src[u], u));
        }
        for i in (0..n).filter(|&i| !inside[i]) {
            let d = dist(pts[u], pts[i]);
            if d < best[i] {
                best[i] = d;
                src[i] = u;
            }
        }
    }
    out
}

/// The separation spectrum `S`, sorted and duplicate-free, with `S[0] = δ`.
fn spectrum(pts: &[Point], ends: &[[usize; 2]], tree: &[TreeEdge]) -> Vec<f64> {
    let delta = FLOOR_ULPS * f64::EPSILON * magnitude(pts);
    let mut s = vec![delta];
    s.extend(tree.iter().map(|e| e.0).filter(|&w| w > delta));
    for &[a, b] in ends {
        for (v, &p) in pts.iter().enumerate() {
            if v != a && v != b {
                let d = point_segment(p, pts[a], pts[b]).0;
                if d > delta {
                    s.push(d);
                }
            }
        }
    }
    s.sort_by(f64::total_cmp);
    s.dedup();
    s
}

struct Window {
    t_below: f64,
    t_above: f64,
    radius: f64,
    labels: Vec<usize>,
    reps: Vec<usize>,
    source: RadiusSource,
}

/// The partition at `t_below`, or `None` when a cluster exceeds [`CLUSTER_MAX`].
fn build(pts: &[Point], tree: &[TreeEdge], t_below: f64, t_above: f64) -> Option<Window> {
    let mut dsu = Dsu::new(pts.len());
    for &(w, a, b) in tree {
        if w <= t_below {
            dsu.union(a, b);
        }
    }
    let labels = dsu.labels();
    let mut size = vec![0usize; dsu.sets];
    let mut reps = vec![usize::MAX; dsu.sets];
    for (i, &l) in labels.iter().enumerate() {
        size[l] += 1;
        if reps[l] == usize::MAX || lex_less(pts[i], pts[reps[l]]) {
            reps[l] = i;
        }
    }
    if size.iter().any(|&c| c > CLUSTER_MAX) {
        return None;
    }
    Some(Window {
        t_below,
        t_above,
        radius: sqrt(t_below * t_above),
        labels,
        reps,
        source: RadiusSource::Derived,
    })
}

/// Genuine gaps by decreasing ratio, then the merge-nothing window.
fn candidates(
    pts: &[Point],
    tree: &[TreeEdge],
    s: &[f64],
    config: &ArrangementConfig,
) -> Vec<Window> {
    if s.len() < 2 {
        return Vec::new();
    }
    let ratio = |k: usize| s[k + 1] / s[k];
    let mut order: Vec<usize> = (1..s.len() - 1)
        .filter(|&k| ratio(k) >= config.rho)
        .collect();
    order.sort_by(|&x, &y| ratio(y).total_cmp(&ratio(x)));
    if ratio(0) >= config.rho {
        order.push(0);
    }
    let mut out = Vec::new();
    for k in order {
        if out.len() >= config.max_candidates {
            break;
        }
        out.extend(build(pts, tree, s[k], s[k + 1]));
    }
    out
}

/// The window a user-supplied radius lands in, verified like a derived one.
fn window_from_grid(
    pts: &[Point],
    tree: &[TreeEdge],
    s: &[f64],
    radius: f64,
    rho: f64,
) -> Result<Window, Refusal> {
    if !radius.is_finite() {
        return Err(Refusal::InvalidGrid);
    }
    if radius < s[0] || radius >= s[s.len() - 1] {
        return Err(Refusal::NoStableScale { gap: None });
    }
    let k = s.partition_point(|&x| x <= radius) - 1;
    let (lo, hi) = (s[k], s[k + 1]);
    let refused = Refusal::NoStableScale {
        gap: Some([lo, hi]),
    };
    if hi / lo < rho {
        return Err(refused);
    }
    let mut w = build(pts, tree, lo, hi).ok_or(refused)?;
    w.radius = radius;
    w.source = RadiusSource::User;
    Ok(w)
}

/// The widest gap in the spectrum, for a `NoStableScale` refusal.
fn no_stable_scale(s: &[f64]) -> Refusal {
    let gap = s
        .windows(2)
        .max_by(|x, y| (x[1] / x[0]).total_cmp(&(y[1] / y[0])))
        .map(|w| [w[0], w[1]]);
    Refusal::NoStableScale { gap }
}

// ═══════════════════════════════════════════════════════════════════════════════
// The arrangement (arrange.py)
// ═══════════════════════════════════════════════════════════════════════════════

fn check_pairs(a: usize, b: usize, max: usize) -> Result<(), Refusal> {
    let pairs = a.saturating_mul(b);
    if pairs > max {
        return Err(Refusal::TooManyPairs { pairs, max });
    }
    Ok(())
}

struct Graph {
    counts: Counts,
    n_subdivided: usize,
    n_dup_edges: usize,
}

/// Margin, P1, edge dedup, P2 with subdivision, P3, then count.
fn try_window(
    pts: &[Point],
    ends: &[[usize; 2]],
    w: &Window,
    budget: usize,
) -> Result<Graph, Refusal> {
    let margin = MARGIN_ULPS * f64::EPSILON * magnitude(pts);
    if w.t_above.is_nan() || w.t_above <= margin {
        return Err(Refusal::MarginTooSmall {
            t_above: w.t_above,
            margin,
        });
    }

    // P1: every edge survives.
    let label = |i: usize| w.labels[i];
    let mut collapsed = ends
        .iter()
        .enumerate()
        .filter(|(_, e)| label(e[0]) == label(e[1]));
    if let Some((segment, _)) = collapsed.next() {
        let count = 1 + collapsed.count();
        return Err(Refusal::EdgeCollapsed { segment, count });
    }

    // Edges between clusters, first occurrence owns the edge.
    let mut seen = BTreeSet::new();
    let mut edges: Vec<[usize; 2]> = Vec::new();
    let mut owner = Vec::new();
    for (k, e) in ends.iter().enumerate() {
        let (a, b) = (label(e[0]), label(e[1]));
        let key = (a.min(b), a.max(b));
        if seen.insert(key) {
            edges.push([key.0, key.1]);
            owner.push(k);
        }
    }
    let mut n_dup_edges = ends.len() - edges.len();
    let r: Vec<Point> = w.reps.iter().map(|&i| pts[i]).collect();

    // P2: every (vertex, non-incident edge) distance is 0 or at least t_above.
    check_pairs(r.len(), edges.len(), budget)?;
    let mut splits: Vec<Vec<(f64, usize)>> = vec![Vec::new(); edges.len()];
    let mut nearest: Option<(f64, usize, usize)> = None;
    let mut band = 0;
    for (e, &[a, b]) in edges.iter().enumerate() {
        for (v, &p) in r.iter().enumerate() {
            if v == a || v == b {
                continue;
            }
            let (d, t) = point_segment(p, r[a], r[b]);
            // NaN compares false and drops out, as in the source.
            if d.is_nan() || d >= w.t_above {
                continue;
            }
            if d > 0.0 {
                band += 1;
                if nearest.is_none_or(|(best, _, _)| d < best) {
                    nearest = Some((d, v, e));
                }
            } else {
                splits[e].push((t, v));
            }
        }
    }
    if let Some((distance, v, e)) = nearest {
        return Err(Refusal::VertexNearEdge {
            vertex: r[v],
            segment: owner[e],
            distance,
            count: band,
        });
    }

    // Subdivide at exactly incident vertices; no coordinate is created.
    let n_subdivided = splits.iter().filter(|s| !s.is_empty()).count();
    let mut seen = BTreeSet::new();
    let mut fedges: Vec<[usize; 2]> = Vec::new();
    let mut fowner = Vec::new();
    let mut pieces = 0;
    for (e, split) in splits.iter_mut().enumerate() {
        let [a, b] = edges[e];
        // `t` is measured from the lexicographically smaller endpoint.
        let (start, end) = if lex_less(r[b], r[a]) { (b, a) } else { (a, b) };
        split.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        let chain: Vec<usize> = core::iter::once(start)
            .chain(split.iter().map(|&(_, v)| v))
            .chain(core::iter::once(end))
            .collect();
        for pair in chain.windows(2) {
            pieces += 1;
            let key = (pair[0].min(pair[1]), pair[0].max(pair[1]));
            if seen.insert(key) {
                fedges.push([key.0, key.1]);
                fowner.push(owner[e]);
            }
        }
    }
    n_dup_edges += pieces - fedges.len();

    // P3: no two edges without a shared vertex properly cross.
    check_pairs(fedges.len(), fedges.len(), budget)?;
    let mut crossing: Option<(usize, usize)> = None;
    let mut crossings = 0;
    for (i, &[ia, ib]) in fedges.iter().enumerate() {
        for (j, &[ja, jb]) in fedges.iter().enumerate().skip(i + 1) {
            if ia == ja || ia == jb || ib == ja || ib == jb {
                continue;
            }
            let (ai, bi, aj, bj) = (r[ia], r[ib], r[ja], r[jb]);
            let d1 = cross(sub(bj, aj), sub(ai, aj));
            let d2 = cross(sub(bj, aj), sub(bi, aj));
            let d3 = cross(sub(bi, ai), sub(aj, ai));
            let d4 = cross(sub(bi, ai), sub(bj, ai));
            if (d1 > 0.0) != (d2 > 0.0) && (d3 > 0.0) != (d4 > 0.0) {
                crossings += 1;
                crossing.get_or_insert((i, j));
            }
        }
    }
    if let Some((i, j)) = crossing {
        return Err(Refusal::EdgesCross {
            first: fowner[i],
            second: fowner[j],
            count: crossings,
        });
    }

    Ok(Graph {
        counts: count(r.len(), &fedges),
        n_subdivided,
        n_dup_edges,
    })
}

/// Pieces, enclosed faces and Euler characteristic of a segment arrangement,
/// with the snap window that decided them, or the condition that could not be
/// established.
///
/// Ported from `planimeter/planimeter/arrange.py` (`chi_segments`), with the
/// window search of `planimeter/planimeter/snap.py` (`candidates`,
/// `window_from_grid`) and the count of `planimeter/planimeter/count.py`. The
/// module documentation states the rule, the preconditions and every refusal.
///
/// Cost is O(n²) in distinct endpoints for the spanning tree, O(n·m) for the
/// spectrum and P2, and O(m²) for P3, each bounded by `4 · max_vertices²` pairs.
pub fn arrange(segments: &[Segment], config: &ArrangementConfig) -> Result<Chi, Refusal> {
    let values = segments
        .iter()
        .flatten()
        .flatten()
        .filter(|x| !x.is_finite())
        .count();
    if values > 0 {
        return Err(Refusal::NonFinite { values });
    }
    if segments.is_empty() {
        return Err(Refusal::NoGeometry);
    }

    let (pts, ends) = dedup_exact(segments);
    if pts.len() > config.max_vertices {
        return Err(Refusal::TooManyVertices {
            actual: pts.len(),
            max: config.max_vertices,
        });
    }
    let budget = config
        .max_vertices
        .saturating_mul(config.max_vertices)
        .saturating_mul(4);
    check_pairs(pts.len(), ends.len(), budget)?;

    let tree = emst(&pts);
    let s = spectrum(&pts, &ends, &tree);
    let windows = match config.grid {
        Some(radius) => vec![window_from_grid(&pts, &tree, &s, radius, config.rho)?],
        None => candidates(&pts, &tree, &s, config),
    };

    let mut first = None;
    for w in &windows {
        match try_window(&pts, &ends, w, budget) {
            Ok(g) => {
                return Ok(Chi {
                    vertices: g.counts.vertices,
                    edges: g.counts.edges,
                    pieces: g.counts.pieces,
                    faces: g.counts.faces,
                    chi: g.counts.chi,
                    dangles: g.counts.dangles,
                    t_below: w.t_below,
                    t_above: w.t_above,
                    ratio: w.t_above / w.t_below,
                    radius: w.radius,
                    radius_source: w.source,
                    n_merged: pts.len() - w.reps.len(),
                    n_subdivided: g.n_subdivided,
                    n_dup_edges: g.n_dup_edges,
                    n_candidates: windows.len(),
                });
            }
            // The window is wrong for this drawing: try the next one.
            Err(r @ (Refusal::EdgeCollapsed { .. } | Refusal::MarginTooSmall { .. })) => {
                first.get_or_insert(r);
            }
            // The drawing is ambiguous or unnoded: that is the answer.
            Err(r) => return Err(r),
        }
    }
    Err(first.unwrap_or_else(|| no_stable_scale(&s)))
}
