# Shape Gates

Aether has two topology surfaces.

## Persistent-Homology Surface

Files:

- `crates/aether-core/src/persistence.rs`;
- `crates/aether-lang/src/interpreter.rs`.

This is the active topological ML surface. The DSL can construct a persistence
diagram from a manifold and query Betti numbers or intervals.

```aether
import topology~
let data = [1.0, 1.0, 1.0, 1.0, 1.0]~
manifold M = embed(data, tau=1)~
let diagram = topology.ph(M, max_dim=2, mode="vr", max_points=16)~
let b = topology.betti(diagram, radius=0.0)~
```

## Binary-Shape Surface

Files:

- `crates/aether-core/src/topology.rs`;
- `crates/aether-kernel/src/loader.rs`.

This surface computes approximate `beta_0`, approximate `beta_1`, density, and
verification results for byte slices.

Current rejection reasons:

- invalid density;
- excessive loops;
- mismatch from a reference shape.

This is a gatekeeping heuristic. It should not be described as proof of binary
safety without external validation, corpora, baselines, and false-positive /
false-negative artifacts.
