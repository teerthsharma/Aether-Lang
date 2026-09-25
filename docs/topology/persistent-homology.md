# Persistent Homology

Aether's active persistent-homology engine lives in
`crates/aether-core/src/persistence.rs`.

## Input

The engine consumes a point cloud:

\[
X = \{x_1, x_2, \ldots, x_n\}
\]

where each point is a `ManifoldPoint<D>`.

<div class="ts-viz" data-viz="topo-cloud" data-title="Point cloud X" data-caption="Twelve points in the plane (D = 2), the input the engine consumes; drag any point."></div>


## Vietoris-Rips Complex

For radius \(r\), the Vietoris-Rips complex is:

\[
VR_r(X) = \{\sigma \subseteq X : \max_{u,v \in \sigma} d(u,v) \le r\}
\]

Plain meaning:

- vertices enter at radius `0`;
- an edge enters when its endpoints are within the radius;
- a triangle enters when all three edges are within the radius;
- a tetrahedron enters when all six edges are within the radius.

<div class="ts-viz" data-viz="topo-rips" data-title="Vietoris–Rips complex VR_r(X)" data-caption="Edges, triangles and tetrahedra present at radius r, counted from the max-pairwise-distance rule; drag points or sweep r."></div>


The implementation supports homology dimensions 0 through 2, so it builds
simplexes through tetrahedra.

## Lazy Witness Mode

For lower-load DSL runs, Aether can select landmarks and use all points as
witnesses. A simplex filtration value is:

\[
f(\sigma) = \min_{w \in X}
  \left(\max_{\ell \in \sigma} d(w,\ell) - d(w,L)\right)
\]

where \(L\) is the landmark set and \(d(w,L)\) is the distance from witness
\(w\) to its nearest landmark.

This reduces the selected complex size. It is not the same claim as exact
Vietoris-Rips homology over the full point cloud.

<div class="ts-viz" data-viz="topo-witness" data-title="Lazy witness vs Vietoris–Rips" data-caption="Left: Rips on all 40 points. Right: maxmin landmarks with the witness filtration f(σ) evaluated over all 40 witnesses, at the same r."></div>


## Reduction

The engine sorts simplexes by filtration value and dimension, constructs
boundary columns, and reduces them over \(\mathbb{Z}_2\). A reduced empty column
births a feature. A later column with a low pivot kills the feature born by that
pivot.

<div class="ts-viz" data-viz="topo-matrix" data-title="Boundary-matrix reduction over Z₂" data-caption="A 6-point Rips filtration up to triangles (41 simplices) reduced one column at a time: empty columns birth features, low pivots pair them."></div>


The output is a `PersistenceDiagram` containing pairs:

```rust
pub struct PersistencePair {
    pub dimension: usize,
    pub birth: f64,
    pub death: Option<f64>,
}
```

<div class="ts-viz" data-viz="topo-diagram" data-title="PersistenceDiagram and its stability" data-caption="Diagram and barcode computed live from the Rips filtration; drag or jitter points and watch the bottleneck distance stay within twice the largest point move."></div>


## Betti Query

For radius \(r\):

\[
\beta_k(r) = |\{(b_i,d_i) : b_i \le r < d_i,\ dimension=i=k\}|
\]

Essential intervals have no death value and remain live after birth.

<div class="ts-viz" data-viz="topo-betti" data-title="Betti numbers β_k(r)" data-caption="β₀, β₁, β₂ as step functions of r for a two-cluster cloud, read off the barcode by the b ≤ r &lt; d rule."></div>

