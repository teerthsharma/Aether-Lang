# Segment Arrangements

Module: `aether_core::arrangement`, in `crates/aether-core/src/arrangement.rs`. Given a finite set of straight segments in the plane, `arrange` returns three integers:

- the number of connected pieces;
- the number of bounded faces the segments enclose;
- the Euler characteristic of their union.

It also returns the snap window that decided which endpoints count as one vertex. When a needed condition cannot be established, it returns a typed `Refusal` naming that condition instead. Evidence: 18 tests in `tests/arrangement.rs`.

## Object

The input is a list of segments \([[x_0, y_0], [x_1, y_1]]\). After snapping and noding it defines a plane graph with \(V\) vertices (clusters of endpoints), \(E\) edges (after subdivision and deduplication) and \(C\) connected components. Let \(F\) count every face, including the single unbounded one. Euler's formula for a plane graph reads

\[
V - E + F = 1 + C .
\]

The enclosed faces, the Euler characteristic of the union and the identity joining them are therefore

\[
\text{faces} = F - 1 = E - V + C = \beta_1, \qquad \chi = V - E = \text{pieces} - \text{faces},
\]

where \(\beta_1\) is the first Betti number of the 1-complex and pieces \(= C\). \(C\) is computed by union-find over the final edge list. Given a correct, deduplicated edge list, this step is exact integer arithmetic. The whole difficulty lies in producing that edge list.

## The snap rule

1. **Exact identification.** Bitwise-equal coordinates are one point, with \(-0.0 = +0.0\). No tolerance is involved.
2. **Separation spectrum.** Let \(M = \max \lvert\text{coordinate}\rvert\) over the distinct points (\(M = 1\) if that maximum is zero), and let \(\delta = 4096 \cdot 2^{-52} \cdot M\) be the representability floor. The spectrum is the sorted, duplicate-free set
   \[
   S = \{\delta\} \,\cup\, \{w : w \text{ an EMST edge weight},\ w > \delta\} \,\cup\, \{d(p, s) : p \text{ not an endpoint of segment } s,\ d > \delta\} = \{s_0 = \delta < s_1 < \cdots < s_K\}.
   \]
   EMST is the Euclidean minimum spanning tree of the distinct endpoints. By Gower and Ross (1969), the single-linkage partition \(\pi(r)\) changes only at EMST edge weights. \(\pi(r)\) is the set of components of the graph that joins every pair of points within distance \(r\). It is therefore constant on every \([s_k, s_{k+1})\). The vertex-to-segment distances are included so that a wall end missing a floor by a tiny margin shows up as a separation, even though no two points are close.
3. **Candidate windows.** \([s_k, s_{k+1})\) is a candidate when \(s_{k+1}/s_k \ge \rho\), with \(\rho = 10\) by default. Genuine gaps (\(k \ge 1\)) are tried in decreasing ratio. The merge-nothing window (\(k = 0\)) is tried last, because its ratio is drawing scale over machine epsilon for arithmetic reasons alone. At most `CAND_MAX` windows are tried. A window whose partition has a cluster of more than `CLUSTER_MAX` points is skipped.
4. **Partition and radius.** Clusters are the components of the EMST edges of weight at most \(t_{\text{below}}\). Each cluster is represented by its lexicographically least point, so no coordinate is ever created. The reported radius is the log-midpoint \(\sqrt{t_{\text{below}}\, t_{\text{above}}}\). With a supplied `grid = r`, the single window is the \([s_k, s_{k+1})\) that contains \(r\), subject to the same conditions, and the reported radius is \(r\).

### Preconditions checked in each window

| Condition | Statement | On failure |
| --- | --- | --- |
| Margin | \(t_{\text{above}} > 64 \cdot 2^{-52} \cdot M\) | Try the next window |
| P1, every edge survives | No input segment has both endpoints in one cluster | Try the next window |
| P2, robustness | Every (vertex, non-incident edge) distance is exactly \(0\) or at least \(t_{\text{above}}\). A distance of 0 subdivides the edge at that vertex | End the run |
| P3, plane embedding | After subdivision and deduplication, no two edges without a shared vertex properly cross | End the run |

A margin or P1 failure means the window is wrong for this drawing. A P2 or P3 failure means the drawing itself is ambiguous or unnoded. A finer window would read the near miss as a clean miss, and returning that would mean choosing whichever reading certifies. If every window fails, the first window's refusal is returned. No intersection point is ever computed. Two segments that cross at a point the input does not contain produce a refusal, not a new vertex.

## Certified versus heuristic

For a returned `Chi`, the following are certified:

- \(\pi(r)\) is constant for every \(r \in [t_{\text{below}}, t_{\text{above}})\), and \(t_{\text{above}}/t_{\text{below}} \ge \rho\).
- The two closest distinct clusters are exactly the next merge height apart, which is at least \(t_{\text{above}}\). This follows from Kruskal's cut property.
- The margin, P1, P2 and P3 were checked rather than assumed.
- The three integers are exact for the resulting graph.

The following are not certified:

- That the identification is the one the author of the drawing intended. The window is *stable*, not *right*.
- Uniqueness. The first passing window wins, so a later window might pass with different integers. Two 10-unit squares exactly 1 apart read as one piece, and 1.01 apart as two.
- The policy constants. They are chosen, not theorems.
- The margin condition itself. It rests on the source's float64 argument (predicate error a small multiple of \(2^{-52} M\)), not on exact arithmetic.
- The count of rendered regions. `faces` counts arrangement faces: two overlapping rectangles, noded at their crossings, are three faces where a renderer fills two regions.

| Constant | Value | Role |
| --- | --- | --- |
| `RHO` | 10 | Minimum window ratio \(t_{\text{above}}/t_{\text{below}}\) |
| `CAND_MAX` | 4 | Candidate windows tried |
| `FLOOR_ULPS` | 4096 | Representability floor, in ulps of \(M\) |
| `CLUSTER_MAX` | 16 | Largest cluster treated as one vertex |
| `BRUTE_MAX` | 2000 | Default ceiling on distinct endpoints |
| `MARGIN_ULPS` | 64 | Float64 margin, in ulps of \(M\) |

## Refusals

| Variant | Kind | Condition |
| --- | --- | --- |
| `NonFinite { values }` | Input | A coordinate is NaN or infinite |
| `InvalidGrid` | Input | The supplied grid radius is NaN or infinite |
| `NoGeometry` | Geometry | The segment list is empty |
| `TooManyVertices { actual, max }` | Budget | More distinct endpoints than `max_vertices` |
| `TooManyPairs { pairs, max }` | Budget | A quadratic pass would exceed \(4 \cdot \texttt{max\_vertices}^2\) pairs |
| `NoStableScale { gap }` | Geometry | No candidate window exists. Or the supplied grid lies below the floor, at or above every separation, in a gap narrower than \(\rho\), or in a window with an oversized cluster |
| `MarginTooSmall { t_above, margin }` | Geometry | \(t_{\text{above}}\) is not above 64 ulps of \(M\) |
| `EdgeCollapsed { segment, count }` | Geometry | P1 failed |
| `VertexNearEdge { vertex, segment, distance, count }` | Geometry | P2 failed |
| `EdgesCross { first, second, count }` | Geometry | P3 failed |

`Refusal::kind()` returns `Input` (nothing was computed), `Budget` (the drawing was not judged) or `Geometry` (re-observe the drawing).

## Rust API

```rust
pub type Segment = [[f64; 2]; 2];

pub struct ArrangementConfig {
    pub rho: f64,               // default RHO
    pub max_candidates: usize,  // default CAND_MAX
    pub grid: Option<f64>,      // None: derive the window from the spectrum
    pub max_vertices: usize,    // default BRUTE_MAX
}

pub fn arrange(segments: &[Segment], config: &ArrangementConfig) -> Result<Chi, Refusal>;

pub struct Chi {
    pub vertices: usize, pub edges: usize, pub pieces: usize, pub faces: usize, pub chi: i64,
    pub dangles: usize,
    pub t_below: f64, pub t_above: f64, pub ratio: f64, pub radius: f64,
    pub radius_source: RadiusSource,          // Derived | User
    pub n_merged: usize, pub n_subdivided: usize, pub n_dup_edges: usize, pub n_candidates: usize,
}
impl Refusal { pub fn kind(&self) -> RefusalKind; }   // Geometry | Budget | Input
```

Cost is \(O(n^2)\) in distinct endpoints for the spanning tree, \(O(nm)\) for the spectrum and P2, and \(O(m^2)\) for P3. Each is bounded by \(4 \cdot \texttt{max\_vertices}^2\) pairs.

## Test evidence

`tests/arrangement.rs` holds 18 `#[test]` functions. Truth comes from each figure's construction, derived on paper, never from the module under test. Every certified answer is also checked against \(\text{faces} = E - V + C\) and \(\chi = V - E\). Tuples below are \((V, E, \text{pieces}, \text{faces}, \chi)\).

```bash
cargo test -p aether-core --test arrangement
```

| Test | Pins |
| --- | --- |
| `a_clean_square_certifies_one_face_at_the_merge_nothing_window` | Square \((4,4,1,1,0)\) at the window \([\delta, 10)\), with 1 candidate. Triangle \((3,3,1,1,0)\) |
| `disjoint_and_nested_squares_add_pieces_and_faces` | Two squares, disjoint or nested, give \((8,8,2,2,0)\). A disjoint union is additive: \((10, 11, 2, 3, -1)\) |
| `diagonals_meeting_at_a_written_centre_make_four_faces` | \((5,8,1,4,-3)\), with no subdivision |
| `a_figure_eight_shares_one_vertex_between_two_faces` | \((7,8,1,2,-1)\) |
| `an_n_by_m_grid_has_n_times_m_faces` | \(V = (n{+}1)(m{+}1)\), \(E = n(m{+}1) + m(n{+}1)\), faces \(nm\), \(\chi = 1 - nm\), for \(n \le 4\) and \(m \le 5\) |
| `an_endpoint_on_a_segment_interior_subdivides_it_without_crossing` | T-junction \((6,7,1,2,-1)\). Outside stub \((6,6,1,1,0)\) with one dangle. H figure \((6,5,1,0,1)\) |
| `collinear_overlaps_and_duplicates_leave_one_edge_per_vertex_pair` | Overlaps \((4,3,1,0,1)\), duplicates \((2,1,1,0,1)\), and a doubled wall that does not add a face |
| `a_vertex_pair_gap_below_the_radius_merges_and_one_above_it_stays_open` | A corner jittered by \(10^{-5}\) merges. `grid` \(10^{-3}\) closes it and \(10^{-8}\) leaves it open. With \(\rho = 5\), `grid` 5 closes the window \([1, 9)\) |
| `a_gap_one_tenth_of_the_feature_is_the_boundary_of_the_stated_rule` | 10-unit squares with gap 1 give \((6,7,1,2,-1)\). With gap 1.01 they give \((8,8,2,2,0)\) |
| `a_collapsed_edge_falls_through_to_the_next_window` | A \(10^{-4}\) speck gives \((6,5,2,1,1)\) at the second candidate. `grid` 0.01 refuses with `EdgeCollapsed { segment: 4, count: 1 }` |
| `rigid_motions_preserve_the_integers` | A quarter turn, a reflection and a dyadic translation are exact and keep every integer, including the T-junction and stub. General rotations keep the integers of figures without interior incidences |
| `uniform_scaling_scales_the_window_with_it` | Scaling by 1024 multiplies \(t_{\text{below}}\), \(t_{\text{above}}\) and the radius by exactly 1024. A non-dyadic factor with a grid scaled by the same factor keeps the integers |
| `segment_order_and_orientation_do_not_move_the_answer_or_the_window` | Permuting or reversing segments changes neither the answer nor the window |
| `crossing_segments_refuse_rather_than_invent_the_point` | A crossing with no written vertex gives `EdgesCross`. A raw pentagram gives `EdgesCross { count: 5, .. }` |
| `a_vertex_near_an_edge_refuses_and_names_the_distance` | A crosswall that misses the floor by \(2.2 \times 10^{-6}\) gives `VertexNearEdge`, naming vertex \((5, 2.2 \times 10^{-6})\) and the distance to \(10^{-15}\). Either resolution of the drawing certifies |
| `no_stable_scale_refuses_rather_than_choosing` | At magnitude \(10^{8}\), separations of \(10^{-4}\) to \(2 \times 10^{-4}\) have no ratio-10 gap and give `NoStableScale`. So does a supplied radius in a narrow gap, above every separation or below the floor |
| `non_finite_and_empty_input_are_refused_by_type` | `NonFinite` and `InvalidGrid` have kind `Input`. `NoGeometry` has kind `Geometry` |
| `the_vertex_ceiling_is_a_budget_refusal_not_a_geometry_one` | `TooManyVertices` has kind `Budget`. At the ceiling, a \(3 \times 3\) grid still certifies \((16, 24, 1, 9, -8)\) |

## Not ported, and deviations

- **Not ported:** planimeter's file readers, curve flattening, digest and JSON surface. The input here is a segment list. The source's `CURVE_UNSTABLE` refusal belongs to curve flattening in its reader and has no counterpart here.
- **Deviation:** the point-to-segment distance is evaluated with the segment oriented from its lexicographically smaller endpoint, so it is a function of the unordered segment. The source orients by input order in the spectrum and by cluster label in P2, and the two can differ in the last ulp. Here P2 recomputes exactly the values the spectrum holds, and reversing a segment changes nothing.
- **Deviation:** every pair-budget overrun is reported as `TooManyPairs`. The source reports the overrun in its spectrum pass as `TOO_MANY_VERTICES`.

## Provenance

[planimeter](https://github.com/teerthsharma/planimeter): `planimeter/count.py` (the counting identity), `planimeter/snap.py` (the snap window: `candidates`, `window_from_grid`), `planimeter/arrange.py` (the arrangement preconditions and the window-selection loop, `chi_segments`) and `planimeter/result.py` (`KIND_OF`).
