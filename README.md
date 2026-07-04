# Invented by Teerth Sharma — λ Aether-Lang

A Rust-based language runtime for topological machine learning — embeddings live on
3D manifolds, training converges via topological invariants (Betti numbers,
persistent homology), not loss thresholds.

---

## What It Actually Is

```
Aether-Lang = interpreter/VM (Rust) + manifold ML runtime (Rust) + microkernel (Rust)
```

| Crate | Role |
|-------|------|
| `aether-lang` | Lexer → parser → AST → interpreter → VM |
| `aether-core` | Manifold embeddings, bounded persistent homology, TDA, geometric primitives |
| `aether-kernel` | Bare-metal x86_64 microkernel (no_std) |
| `aether-cli` | `aether` REPL and script runner |

---

## The Verified Kernel (Lean 4)

The AETHER Verified Kernel is a **separate, formally verified** project:

> [apoth3osis.io/paper-proof-code/aether-verified-kernel](https://www.apoth3osis.io/paper-proof-code/aether-verified-kernel)

- **23 theorems** — machine-checked by the Lean 4 kernel
- **0 sorry / 0 admit** — every step proven
- **818 lines** of proof code across 4 modules
- Compiles to **verified C** + **safe Rust** artifacts via Curry-Howard
- **6 hostile adversarial audits**, final 2 consecutive CLEAN
- Covers: sparse-event attention, Lyapunov PD governors, Chebyshev GC guards,
  Betti approximation bounds

---

## Quickstart

```bash
# Build CLI
cargo build -p aether-cli --release
./target/release/aether repl

# Run a script
./target/release/aether run examples/hello.aether
```

**Docker (recommended):**
```bash
docker pull teerthsharma/aether
docker run -it teerthsharma/aether repl
```

---

## Language Sample

```aether
// manifold_learn.aether — topological training
manifold M = embed(dataset, dim=3, tau=5)
block B = M.cluster(range=0:64)
centroid C = B.center

// Train until topology stabilizes (Betti₁ convergence), not loss threshold
train M until topological_convergence(betti_threshold=0.95)
```

In the REPL, statements end with `~`. Scripts use `.aether` / `.ae`.

---

## Topological ML Core

AETHER now ships a real bounded persistent-homology engine in `aether-core`.
It builds filtered complexes and reduces them over `Z2` for H0/H1/H2 intervals.
The core is `no_std`-compatible with `alloc`; `std` is only for convenience.

The engine is designed for DSL runs, so it has hard load controls instead of
unbounded simplex expansion:

- caps for points, edges, triangles, tetrahedra, and total simplices
- radius cutoffs for Vietoris-Rips complexes
- optional landmark/witness mode for cheaper topology over larger signals
- low-load defaults exposed through the DSL

```aether
import topology~

let data = [0, 1, 0, -1, 0, 1, 0, -1]~
let M = embed(data, dim=3, tau=1)~
let diagram = topology.ph(M, max_dim=2, mode="witness", landmarks=16)~
let b = topology.betti(diagram, radius=0.5)~
let bars = topology.intervals(diagram)~
```

`topology.betti` returns `[β0, β1, β2]`. This is not a Betti stub: the DSL calls
the same exact filtered-complex reduction used by `aether-core`.

---

## Runtime Performance Notes

The hot paths follow the same rule as the topology core: keep the math real, but
pay only for what the algorithm needs.

- KNN and nearest-centroid rank by squared distance, avoiding unnecessary `sqrt`.
- DBSCAN, auto-k selection, sparse-neighborhood checks, and gossip convergence
  compare squared radii/tolerances in threshold loops.
- Scalar loss and distance reductions borrow tensor data directly instead of
  building temporary tensors.
- The manifold heap Chebyshev guard computes statistics in one pass.

---

## Architecture

```
Source (.ae)
    │
    ▼
┌─────────────┐
│   Lexer     │  → tokens
│  aether-lang│
├─────────────┤
│   Parser    │  → AST
├─────────────┤
│ Interpreter │  → IR
├─────────────┤
│   VM        │  → execution
└─────────────┘
    │
    ▼
┌─────────────┐
│ aether-core │  ← manifold embeddings, persistent homology, TDA, geometry
│ aether-kernel│ ← bare-metal microkernel (x86_64)
└─────────────┘
```

---

## Verified Stack (Full Trust Chain)

```
Lean 4 Proofs (apoth3osis.io)
       │
       ▼  Curry-Howard → Compiled
Verified C (IPFS) + Safe Rust (IPFS)
       │
       ▼  FFI / bindings
Aether-Lang Runtime (this repo)
```

The AETHER Verified Kernel provides the **trusted math foundation** for Aether-Lang's
topological operations — zero false negatives in attention sparsification, energy-based
convergence certificates, GC guard bounds at 1/k², and Betti approximation theorems.

---

## CLI Reference

```bash
aether repl                  # Interactive REPL (~ to end statements)
aether run path/to/file.ae  # Execute script
aether check path/to/file.ae # Type-check only
```

---

## Repository Layout

```
crates/
  aether-lang/    — Lexer, parser, AST, interpreter, VM
  aether-core/     — Manifolds, embeddings, TDA, geometry
  aether-kernel/   — Bare-metal x86_64 microkernel (no_std)
  aether-cli/      — REPL + command-line interface
  aegis-core/      — Compatibility / alternate core
  aegis-cli/       — Compatibility CLI
docs/
  GETTING_STARTED.md
  LANGUAGE.md
  ARCHITECTURE.md
examples/
  *.ae             — Example Aether scripts
```

---

## Building

```bash
rustup install nightly
rustup component add rustfmt clippy

cargo build --release          # full workspace
cargo test --workspace         # all tests
cargo build -p aether-kernel   # bare-metal kernel (nightly required)
```

---

## Links

- **Verified Kernel:** [apoth3osis.io/paper-proof-code/aether-verified-kernel](https://www.apoth3osis.io/paper-proof-code/aether-verified-kernel)
- **Lean 4 Proofs:** IPFS (content-addressed, immutable)
- **Language Docs:** [docs/](docs/)

---

*λ — where manifold topology meets verified foundations.*
