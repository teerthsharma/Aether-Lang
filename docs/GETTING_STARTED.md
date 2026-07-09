# Getting Started

This page is retained for existing links. The current getting-started flow is
documented in the MkDocs site:

- [Home](index.md)
- [Language syntax](language/syntax.md)
- [Execution model](language/execution-model.md)
- [Status matrix](reference/status.md)

## Build And Run

```powershell
cargo build -p aether-cli
cargo run -p aether-cli -- check examples/simple.aegis
cargo run -p aether-cli -- run examples/simple.aegis
cargo run -p aether-cli -- repl
```

The standard script extensions are `.aether` and `.ae`. Some repository examples
use legacy extensions and may print an extension warning.

## Minimal Script

```aether
let data = [1.0, 2.0, 3.0, 4.0]~
manifold M = embed(data, tau=1)~
print("done")~
```

## Topology Script

```aether
import topology~
let data = [1.0, 1.0, 1.0, 1.0, 1.0]~
manifold M = embed(data, tau=1)~
let diagram = topology.ph(M, max_dim=2, mode="vr", max_points=16)~
let b = topology.betti(diagram, radius=0.0)~
```

This calls the active bounded persistent-homology path described in
[Persistent Homology](topology/persistent-homology.md).
