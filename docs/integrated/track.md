# Cell Tracking

Module: `aether_core::track`, in `crates/aether-core/src/track.rs`. It tracks point detections across frames in two ways: one-to-one frame linking, and linking with divisions by min-cost circulation. It also checks the certificate that makes the circulation relaxation exact. Evidence: 15 tests in `tests/track.rs`.

The input is a sequence of frames. Each frame is a set of detection centroids in voxel coordinates \((z, y, x)\). The output is a `TrackGraph`: its nodes are \((\text{frame}, \text{detection index})\) pairs, its edges run from parent to child, and its divisions are listed explicitly.

## Metric

Every distance is physical. With voxel spacing \(s = (s_z, s_y, s_x)\),

\[
d(p, q) = \big\lVert \operatorname{diag}(s)\,(p - q) \big\rVert_2 .
\]

On the benchmark grid (`SCALE_UM` \(= (1.625,\ 0.40625,\ 0.40625)\) µm per voxel), \(z\) is four times coarser than \(y\) and \(x\). An unscaled distance would link along \(z\) four times too eagerly.

## Linking rule

Consider frames \(A = \{a_i\}\) with \(n_A\) points and \(B = \{b_j\}\) with \(n_B\) points, and a gate \(r\) (`max_um`). They are linked by the optimal assignment on a padded cost matrix:

\[
C_{ij} =
\begin{cases}
d(a_i, b_j) & j < n_B,\\
r & n_B \le j < n_B + n_A,
\end{cases}
\qquad
\sigma^* = \operatorname*{arg\,min}_{\sigma \text{ injective}} \sum_i C_{i\,\sigma(i)},
\]

\[
E = \big\{(i, \sigma^*(i)) : \sigma^*(i) < n_B,\ d(a_i, b_{\sigma^*(i)}) \le r\big\}.
\]

The \(n_A\) dummy columns price ending a track at exactly \(r\). A real successor is therefore taken only when it is nearer than the gate, and the solver never spends a cell on a partner the gate would discard. Equivalently, \(E\) is a maximum-weight matching under weights \(r - d(a_i, b_j)\).

Assigning on the raw matrix and gating afterwards is not equivalent. A distant pair chosen by the solver can block a nearer one. The gate then drops the distant pair, so one bad assignment costs two links.

## Split rule

Tracking with divisions is posed as a min-cost circulation on a node-split graph: the construction of Zhang, Li and Nevatia (CVPR 2008), with a division in-arc added. Each detection \(x\) becomes \(x_{\text{in}} \to x_{\text{out}}\), and there is a source \(s\) and a sink \(t\).

| Arc | Capacity | Cost | Meaning |
| --- | --- | --- | --- |
| \(x_{\text{in}} \to x_{\text{out}}\) | 1 | \(c_{\text{det}}\) | The detection is used |
| \(s \to x_{\text{in}}\) | 1 | \(c_{\text{app}}\) | A lineage starts at \(x\) |
| \(x_{\text{out}} \to t\) | 1 | \(c_{\text{dis}}\) | A lineage ends at \(x\) |
| \(s \to x_{\text{out}}\) | 1 | \(c_{\text{div}}\) | The second daughter's unit enters at \(x\) |
| \(x_{\text{out}} \to y_{\text{in}}\) | 1 | \(d(x, y)\) | \(y\) is in frame \(t{+}1\), among the \(k\) nearest, with \(d \le r\) |
| \(t \to s\) | unbounded | 0 | Closes the circulation |

The solution is the integral circulation \(f\) that minimises \(\sum_e c_e f_e\). A node divides when its \(x_{\text{out}}\) emits two transitions:

\[
\operatorname{split}(x) \iff \big\lvert\{y : f(x_{\text{out}}, y_{\text{in}}) = 1\}\big\rvert = 2 .
\]

Hold the rest of the flow fixed. A second daughter \(y\) is then attached to \(x\), rather than started as a new lineage, exactly when

\[
c_{\text{div}} + d(x, y) < c_{\text{app}},
\]

which is 5 µm under the default costs. A split needs no post-hoc detector: it is the only way a node can carry two units.

## The certificate

The constraint "a cell may only divide where a cell exists" couples flow on different arcs. That coupling destroys the total unimodularity that makes a flow relaxation integral (Haubold et al., ECCV 2016). So the constraint is checked on the answer instead of encoded in the graph:

\[
\operatorname{violation}(x) \iff f(s, x_{\text{out}}) = 1 \,\wedge\, f(x_{\text{in}}, x_{\text{out}}) = 0,
\qquad
\text{calibration: } c_{\text{div}} \ge c_{\text{app}} + c_{\text{det}} .
\]

Reroute a violating unit from \(s \to x_{\text{out}}\) onto \(s \to x_{\text{in}} \to x_{\text{out}}\). The cost changes by \(c_{\text{app}} + c_{\text{det}} - c_{\text{div}} \le 0\). Conservation at \(x_{\text{in}}\), which has one in-arc from \(s\) and one out-arc, forces both arcs of that path to be free (`reroute_not_worse` and `conservation_frees_capacity` in `CleaveProofs.lean`). Under strict calibration the reroute strictly improves the cost (`reroute_strictly_better`). No optimum then violates the certificate, and the circulation optimum is the integer-program optimum. `FlowConfig::validate` refuses a configuration below the calibration bound, and `TrackResult::violations` recomputes the count from the solved flow.

**Solver.** Costs are rounded to integers as \(\operatorname{round}(w\,c)\), with \(w\) = `weight_scale`, exactly as cleave rounds them for `networkx.network_simplex`. The circulation is solved as a min-cost \(s\)–\(t\) flow by successive shortest paths: Bellman–Ford on the residual graph, with unit augmentations. The initial graph is acyclic because every arc goes forward in time. The shortest-path lengths are therefore non-decreasing, and stopping at the first non-negative one gives the minimum-cost circulation. Zero-cost augmentations are not taken, so ties resolve toward fewer units.

## Guaranteed, and heuristic

**Guaranteed by construction:**

- In-degree is at most 1 for every node, from both linkers: there are no merge events.
- Out-degree is at most 1 from `link_sequence`, which cannot emit a division, and at most 2 from `flow_track`. `TrackGraph::divisions` lists exactly the nodes with out-degree 2.
- Every edge advances exactly one frame and has physical length at most \(r\). The gap budget is one frame: a missing detection, or an empty frame, ends every track through it, and later frame indices are not renumbered.
- The graph is a forest, because in-degree is at most 1 and every edge runs strictly forward in time. Each tree is one lineage.
- `link_frames` returns an exact optimum of the padded assignment. `flow_track` returns an exact optimum of the rounded-cost circulation over its candidate arcs, with zero violations whenever \(c_{\text{div}} > c_{\text{app}} + c_{\text{det}}\).
- The output depends on positions only through \(d\). An isometry of \(d\) applied to every frame therefore leaves it unchanged, and permuting detections within a frame permutes it, in both cases up to exact cost ties. Rotation plus translation is such an isometry when \(s\) is isotropic. On `SCALE_UM`, only rotations in the \((y, x)\) plane are.

**Heuristic, not guaranteed:**

- The costs themselves: \(c_{\text{det}}, c_{\text{app}}, c_{\text{dis}}, c_{\text{div}}\) and the gate \(r\) are modelling choices. `cleave/proofs/README.md` states that it is not proved that tracking should be this circulation at all. The assignment gate default of 8 µm is another competitor's measurement, recorded in `link.py`.
- The \(k\)-nearest candidate restriction. It can exclude the true successor, and the optimum is then over the restricted arc set.
- Rounding to \(1/w\). Costs closer together than that are treated as equal.
- Whether a fork is a biological division. The rule prices a geometric configuration. It does not observe mitosis.

## Refusals

| Variant | Condition |
| --- | --- |
| `NonFiniteCoordinate { frame, index }` | A coordinate is NaN or infinite |
| `InvalidGate` | The gate `max_um` is not finite and positive |
| `InvalidScale` | A voxel-spacing component is not finite and positive |
| `Uncalibrated` | \(c_{\text{div}} < c_{\text{app}} + c_{\text{det}}\) |
| `InvalidNeighbours` | `n_neighbours = 0` |
| `InvalidCost` | A non-finite cost, a non-positive `weight_scale`, or a rounded cost outside `i32`. The `i32` bound keeps every path sum exact in `i64` |

Empty frames and an empty sequence are accepted, as in cleave. An empty frame must be passed rather than omitted, because omitting it would shift every later frame in time.

## Rust API

```rust
pub const SCALE_UM: [f64; 3] = [1.625, 0.40625, 0.40625];
pub type Point = [f64; 3];                          // (z, y, x) in voxels
pub struct Node { pub frame: usize, pub index: usize }
pub struct TrackGraph { pub nodes: Vec<Node>, pub edges: Vec<(usize, usize)>, pub divisions: Vec<usize> }

pub fn physical_distance(a: Point, b: Point, scale: [f64; 3]) -> f64;
pub fn link_frames(a: &[Point], b: &[Point], max_um: f64, scale: [f64; 3])
    -> Result<Vec<(usize, usize)>, TrackError>;
pub fn link_sequence<F: AsRef<[Point]>>(frames: &[F], max_um: f64, scale: [f64; 3])
    -> Result<TrackGraph, TrackError>;

pub struct FlowConfig {
    pub c_det: f64, pub c_app: f64, pub c_dis: f64, pub c_div: f64,
    pub max_um: f64, pub n_neighbours: usize, pub weight_scale: f64,
}   // Default: c_det = -50, c_app = c_dis = 10, c_div = 5, 12 µm gate, 5 neighbours, scale 1000
impl FlowConfig { pub fn validate(&self) -> Result<(), TrackError>; }

pub struct TrackResult { pub graph: TrackGraph, pub violations: usize, pub cost: f64 }
impl TrackResult { pub fn certified(&self) -> bool; }   // violations == 0
pub fn flow_track<F: AsRef<[Point]>>(frames: &[F], config: &FlowConfig, scale: [f64; 3])
    -> Result<TrackResult, TrackError>;
```

At the defaults \(c_{\text{app}} + c_{\text{det}} = -40\), so \(c_{\text{div}} = 5\) is strictly calibrated.

## Test evidence

`tests/track.rs` holds 15 `#[test]` functions. A tracking bug rarely crashes. It returns a plausible forest with one swapped identity, one fork in the wrong frame, or one link the gate should have refused. Each test is a property that such a graph violates.

```bash
cargo test -p aether-core --test track
```

| Test | Pins |
| --- | --- |
| `constant_velocity_particles_are_tracked_exactly_at_the_closed_form_cost` | Five particles 25 µm apart, each at a constant velocity of at most 3.4 µm per frame, over eight frames. The true tracks are recovered at the closed-form cost |
| `a_single_division_produces_exactly_one_split_on_the_correct_parent` | One fork, on the last single-cell frame, with the two daughters as children. `link_sequence` on the same scene has no split |
| `a_split_is_taken_only_when_it_beats_a_new_lineage` | With \(d = \sqrt{\text{offset}^2 + 1}\), offsets 2 and 4 (\(d < 5\) µm) split, and offsets 5 and 7 do not |
| `a_dropout_ends_the_track_and_nothing_is_renumbered` | A one-frame dropout gives two fragments. There is no gap closing |
| `the_gate_is_priced_inside_the_assignment_not_applied_after_it` | On a line with gate 5, the raw assignment takes the swap (\(6 + 6 = 12\) against \(4 + 16 = 20\)), and gating afterwards would drop both links. The padded matrix keeps the one pair inside the gate |
| `the_assignment_is_global_not_greedy` | cleave's contention fixture: two sources whose nearest target is the same |
| `the_gate_is_applied_in_micrometres_not_voxels` | 4 voxels along \(z\) are 6.5 µm, and along \(x\) 1.625 µm. A 6 µm gate refuses the \(z\) link |
| `crossing_tracks_separated_above_the_gate_stay_distinct` | Paths crossing head-on 8 voxels apart in \(z\) (13 µm, above an 8 µm gate) stay distinct |
| `graph_invariants_hold_on_seeded_random_scenes` | Degrees, one frame per edge, and a zero certificate on 24 seeded scenes |
| `a_barely_calibrated_division_price_still_certifies` | \(c_{\text{div}} = -39\), one unit above the bound, certifies on 12 seeded scenes. An inexact solver could return a violating flow here |
| `a_rigid_motion_of_every_frame_leaves_the_graph_unchanged` | 10 seeded scenes |
| `permuting_detections_within_a_frame_permutes_the_graph` | 10 seeded scenes |
| `non_finite_coordinates_are_refused_with_their_location` | `NonFiniteCoordinate { frame, index }` |
| `a_gate_or_spacing_that_is_not_finite_and_positive_is_refused` | `InvalidGate`, `InvalidScale` |
| `an_uncalibrated_division_price_is_refused_at_the_bound_exactly` | Equality with \(c_{\text{app}} + c_{\text{det}} = -40\) is admissible, and one below it is `Uncalibrated`. Also `InvalidNeighbours` and `InvalidCost` |

## Not ported

- **Gap closing.** cleave implements none, deliberately: `link.py` records it as measured neutral. "Join" here is the frame-to-frame link and nothing more.
- **The image-side split evidence** (`euler3d.division_signature`, `persist.h0_barcode`). It is an \(H_0\) filtration over voxel intensities. Its input is an image, not points, so it falls outside this module, and the linker has no filtration that `aether_core::persistence` could supply.

**Known ceiling.** Bellman–Ford costs \(O(VE)\) per augmentation, and there are \(O(V)\) augmentations. When frames carry hundreds of detections, the source code names the replacement: Dijkstra on Johnson-reduced costs. The stopping rule does not change.

## Provenance

cleave (Teerth Sharma; private repository), a tracker for 3D+time light-sheet microscopy:

- `cleave/cleave/link.py`: the linker and `TrackGraph`.
- `cleave/cleave/flow.py`: the division-aware circulation, `FlowConfig` and `TrackResult`.
- `cleave/cleave/zebrahub.py::division_parents`: the division count.
- `cleave/cleave/euler3d.py::SCALE_UM`: the benchmark voxel spacing.
- `cleave/proofs/CleaveProofs.lean`, sections 7 and 8: the certificate lemmas.
