# Persistent Homology

Engine: `crates/aether-core/src/persistence.rs`.

## Input

A point cloud \(X = \{x_1, \ldots, x_n\}\), each point a `ManifoldPoint<D>`.

<div class="ts-viz" data-viz="topo-cloud" data-title="Point cloud X" data-caption="Twelve points in the plane (D = 2), the input the engine consumes; drag any point."></div>


## Vietoris-Rips Complex

\[
VR_r(X) = \{\sigma \subseteq X : \max_{u,v \in \sigma} d(u,v) \le r\}
\]

- vertices enter at radius `0`;
- an edge enters when its endpoints are within the radius;
- a triangle enters when all three edges are within the radius;
- a tetrahedron enters when all six edges are within the radius.

<div class="ts-viz" data-viz="topo-rips" data-title="Vietoris–Rips complex VR_r(X)" data-caption="Edges, triangles and tetrahedra present at radius r, counted from the max-pairwise-distance rule; drag points or sweep r."></div>


Homology dimensions 0–2 are supported, so simplexes go up to tetrahedra.

## Lazy Witness Mode

Landmarks \(L\), all points as witnesses: a smaller complex for lower-load DSL runs, not exact Rips homology. [Theory →](../theory.md#th-lazy-witness)

<div class="ts-viz" data-viz="topo-witness" data-title="Lazy witness vs Vietoris–Rips" data-caption="Left: Rips on all 40 points. Right: maxmin landmarks with the witness filtration f(σ) evaluated over all 40 witnesses, at the same r."></div>


## Reduction

Simplexes sorted by filtration value and dimension; boundary columns reduced over \(\mathbb{Z}_2\). [Theory →](../theory.md#th-reduction)

<div class="ts-viz" data-viz="topo-matrix" data-title="Boundary-matrix reduction over Z₂" data-caption="A 6-point Rips filtration up to triangles (41 simplices) reduced one column at a time: empty columns birth features, low pivots pair them."></div>


The output is a `PersistenceDiagram` of pairs:

```rust
pub struct PersistencePair {
    pub dimension: usize,
    pub birth: f64,
    pub death: Option<f64>,
}
```

<div class="ts-viz" data-viz="topo-diagram" data-title="PersistenceDiagram and its stability" data-caption="Diagram and barcode computed live from the Rips filtration; drag or jitter points and watch the bottleneck distance stay within twice the largest point move."></div>


## Betti Query

Essential intervals have no death value and stay live after birth. [Theory →](../theory.md#th-betti)

\[
\beta_k(r) = |\{(b_i,d_i) : b_i \le r < d_i,\ dimension=i=k\}|
\]

<div class="ts-viz" data-viz="topo-betti" data-title="Betti numbers β_k(r)" data-caption="β₀, β₁, β₂ as step functions of r for a two-cluster cloud, read off the barcode by the b ≤ r &lt; d rule."></div>
