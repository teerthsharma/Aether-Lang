# Aether Lang

Aether Lang documents the language runtime and core systems as a set of bounded
contracts. The central idea is that ordinary engineering constraints can expose
useful behavior when the runtime preserves structure:

```mermaid
flowchart LR
  A["Source, signal, tensor, binary, or system state"] --> B["Typed runtime object"]
  B --> C["Embedding, block, graph, or state vector"]
  C --> D["Topology, bound, drift, or threshold"]
  D --> E["Execution, convergence, pruning, or rejection decision"]
```

## What Is Active Today

- Lexer, parser, AST, interpreter, and Titan VM scaffolding in `aether-lang`.
- CLI commands: `aether repl`, `aether run`, and `aether check`.
- Variables, assignments, arithmetic, comparison, logical operators, lists,
  functions, `if`, `while`, `for`, and `seal until`.
- Manifold embedding from numeric lists through a fixed 3D time-delay workspace.
- Block extraction and geometric block metadata.
- Bounded persistent homology over Vietoris-Rips and lazy witness complexes.
- DSL topology calls: `topology.ph`, `topology.betti`, and
  `topology.intervals`.
- ML primitives in `aether-core`: tensors, losses, regression, clustering,
  classification, neural layers, autograd scaffolding, convolution, data
  loading, and gossip consensus.
- Sparse-event scheduler and geometric governor tests in the kernel/core stack.

## What Is Roadmap Or Gated

- Hardware acceleration and GPU claims.
- Production security claims for binary authentication.
- End-to-end benchmark speedups.
- Full language-level type checking.
- Framework parity with PyTorch, TensorFlow, CUDA, or Triton.
- Bare-metal bootability as a user-facing distribution target.

Those surfaces can be implemented, but the docs should not describe them as
active capabilities until tests and artifacts cover the claim.

## Learning Path

1. Read the language pipeline to understand source-to-runtime flow.
2. Read persistent homology and derivations before relying on topology terms.
3. Read the runtime surface and status matrix to separate active behavior from
   scaffolding.
4. Run the local checks before trusting any performance or backend statement.

## Evidence Policy

Every active claim should have one of three forms:

- a unit test or integration test;
- a runnable CLI or benchmark artifact;
- a docs-only theory or roadmap statement clearly labeled as such.

This keeps the project legible without turning planned systems into active
claims.
