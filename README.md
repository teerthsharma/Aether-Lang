# Aether Lang

Aether Lang is a Rust workspace for a small language runtime, a bounded
topological computation core, ML primitives, and a sparse-event kernel
prototype. The project uses topology as a systems signal: embeddings, residuals,
binary streams, and scheduler state are converted into geometric or topological
objects, then used for execution, diagnostics, convergence checks, or pruning.

The intended benefit is mechanical rather than promotional. When a system
represents structure explicitly, useful behavior can emerge from ordinary
constraints:

```text
source or signal -> typed runtime object -> geometric summary -> topology or bound -> execution decision
```

## Current Working Surface

Rust crates:

- `aether-lang`: lexer, parser, AST, interpreter, Titan bytecode VM, ASCII and
  WebGL exporters.
- `aether-core`: manifold points, time-delay embeddings, bounded persistent
  homology, geometric block metadata, ML primitives, tensors, and governors.
- `aether-cli`: `aether repl`, `aether run`, and `aether check`.
- `aether-kernel`: no_std sparse-event scheduler, loader, allocator, serial,
  boot, and hardware-topology scaffolding.
- `aegis-core` and `aegis-cli`: compatibility crates retained in the workspace.

Active DSL behavior includes variable declarations, assignments, arithmetic,
comparison and logical expressions, lists, `if`, `while`, `for`, `seal until`,
functions, manifold embedding from numeric lists, block extraction, ML
constructors, and topology module calls.

## Quickstart

```powershell
cargo build -p aether-cli
cargo run -p aether-cli -- check examples/simple.aegis
cargo run -p aether-cli -- run examples/simple.aegis
cargo run -p aether-cli -- repl
```

The CLI accepts `.aether` and `.ae` as standard extensions. Some repository
examples still use `.aegis` and `.ag`; those can be parsed by the current CLI but
may print an extension warning.

## Topology Example

```aether
import topology~
let data = [1.0, 1.0, 1.0, 1.0, 1.0]~
manifold M = embed(data, tau=1)~
let diagram = topology.ph(M, max_dim=2, mode="vr", max_points=16)~
let b = topology.betti(diagram, radius=0.0)~
let intervals = topology.intervals(diagram)~
```

`topology.ph` calls the bounded persistent-homology engine in `aether-core`.
`topology.betti` returns `[beta_0, beta_1, beta_2]`.

## Documentation

Hosted documentation: [teerthsharma.github.io/Aether-Lang](https://teerthsharma.github.io/Aether-Lang/)

The documentation is organized as a MkDocs site:

```powershell
python -m pip install -r requirements-docs.txt
python -m mkdocs serve
```

Start with:

- [docs/index.md](docs/index.md)
- [docs/reference/status.md](docs/reference/status.md)
- [docs/benchmarks/index.md](docs/benchmarks/index.md)
- [docs/topology/derivations.md](docs/topology/derivations.md)

## Evidence Policy

Capability claims in this repository should be tied to one of:

- a Rust unit test;
- a CLI smoke check;
- a benchmark artifact with environment and correctness fields;
- a docs-only theory or roadmap label.

Unverified speedups, hardware acceleration, model-quality improvements, and
security guarantees are not treated as active claims.

## Local Checks

```powershell
cargo fmt --all -- --check
cargo test -p aether-core
cargo test -p aether-lang
cargo test -p aether-cli
cargo build -p aether-kernel -Z build-std=core,alloc --target x86_64-unknown-none
cargo build -p aether-core --no-default-features --features no_std -Z build-std=core,alloc --target thumbv7m-none-eabi
python -m mkdocs build --strict
```
