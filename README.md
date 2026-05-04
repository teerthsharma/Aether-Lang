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
| `aether-core` | Manifold embeddings, TDA, geometric primitives |
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
│ aether-core │  ← manifold embeddings, TDA, geometry
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
