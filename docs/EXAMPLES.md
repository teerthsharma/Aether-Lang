# Examples

Example documentation should distinguish parser examples from runtime examples.

## Runtime Example

```aether
let data = [1.0, 2.0, 3.0, 4.0]~
manifold M = embed(data, tau=1)~
print("embedded")~
```

## Topology Example

```aether
import topology~
let data = [1.0, 1.0, 1.0, 1.0, 1.0]~
manifold M = embed(data, tau=1)~
let diagram = topology.ph(M, max_dim=2, mode="vr", max_points=16)~
let b = topology.betti(diagram, radius=0.0)~
```

See [Language syntax](language/syntax.md), [Module contracts](language/modules.md),
and [Status matrix](reference/status.md) for claim boundaries.
