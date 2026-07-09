# Persistent Homology

Aether's active persistent-homology engine lives in
`crates/aether-core/src/persistence.rs`.

## Input

The engine consumes a point cloud:

\[
X = \{x_1, x_2, \ldots, x_n\}
\]

where each point is a `ManifoldPoint<D>`.

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

## Reduction

The engine sorts simplexes by filtration value and dimension, constructs
boundary columns, and reduces them over \(\mathbb{Z}_2\). A reduced empty column
births a feature. A later column with a low pivot kills the feature born by that
pivot.

The output is a `PersistenceDiagram` containing pairs:

```rust
pub struct PersistencePair {
    pub dimension: usize,
    pub birth: f64,
    pub death: Option<f64>,
}
```

## Betti Query

For radius \(r\):

\[
\beta_k(r) = |\{(b_i,d_i) : b_i \le r < d_i,\ dimension=i=k\}|
\]

Essential intervals have no death value and remain live after birth.
