<!-- Aether-Lang README -->
<!-- Target: Epsilon-Hollow scale. Every number below was measured. The jokes were not. -->

<h1 align="center">Aether-Lang — The Topological Programming Language</h1>

<p align="center">
  <strong>Programs are point clouds. Convergence is a Betti number.</strong><br>
  A DSL where persistent homology is a language primitive, loops terminate on topological invariants,<br>
  and the whole runtime compiles <code>no_std</code> down to bare metal.
</p>

<p align="center">
  <a href=".github/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/teerthsharma/Aether-Lang/ci.yml?branch=master&label=CI&style=flat-square" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-00aaff?style=flat-square" alt="License: MIT"></a>
  <a href="rust-toolchain.toml"><img src="https://img.shields.io/badge/rust-nightly-orange?style=flat-square&logo=rust" alt="Rust nightly"></a>
  <a href="#results"><img src="https://img.shields.io/badge/tests-223%20passing-brightgreen?style=flat-square" alt="223 tests"></a>
  <a href="#mutation-testing-or-how-i-learned-to-stop-trusting-green-checkmarks"><img src="https://img.shields.io/badge/mutants-8%20injected-purple?style=flat-square" alt="8 mutants"></a>
  <a href="#what-we-got-wrong"><img src="https://img.shields.io/badge/claims%20killed-6-red?style=flat-square" alt="6 claims killed"></a>
  <a href="docs/reference/status.md"><img src="https://img.shields.io/badge/claim%20ledger-live-blue?style=flat-square" alt="Claim ledger"></a>
</p>

<p align="center">
  <a href="#quick-start--five-minutes-to-a-betti-number">🚀 Quick Start</a> •
  <a href="#honest-status-dashboard">📊 Status</a> •
  <a href="#theoretical-foundation">📐 Theory</a> •
  <a href="#results">📈 Results</a> •
  <a href="#what-we-got-wrong">💀 Failures</a> •
  <a href="#the-faq-nobody-asked-for">❓ FAQ</a> •
  <a href="#limitations">⚠️ Limits</a>
</p>

<p align="center">
  <strong>Invented by <a href="https://teerthsharma.vercel.app/">Teerth Sharma</a> · <a href="https://github.com/teerthsharma/Aether-Lang">github.com/teerthsharma/Aether-Lang</a></strong><br>
  <code>teerthsharma@outlook.com</code>
</p>

---

## Table of Contents

**Orientation**
- [Before You Read This](#before-you-read-this)
- [The 30-Second Pitch](#the-30-second-pitch)
- [Abstract](#abstract)
- [Why I Did This To Myself](#why-i-did-this-to-myself)
- [Honest Status Dashboard](#honest-status-dashboard)
- [Quick Start — Five Minutes To A Betti Number](#quick-start--five-minutes-to-a-betti-number)

**The language**
- [The Language](#the-language)
  - [Tildes, Seals, And Other Crimes](#tildes-seals-and-other-crimes)
  - [The Topology Module](#the-topology-module)
  - [Seal Loops](#seal-loops)
  - [The Convergence Condition Is A Three-Way Choice](#the-convergence-condition-is-a-three-way-choice)
  - [The Full Statement Grammar](#the-full-statement-grammar)
  - [The Expression Grammar](#the-expression-grammar)
  - [Compiler Internals, By Size](#compiler-internals-by-size)

**The argument**
- [Background](#background)
  - [Why Topology?](#why-topology)
  - [Why A Language And Not A Library?](#why-a-language-and-not-a-library)
  - [Prior Art](#prior-art)
- [Theoretical Foundation](#theoretical-foundation) — 15 numbered results, each implemented
  - [1) The Vietoris–Rips filtration](#1-the-vietorisrips-filtration) · [2) Persistence via Z₂ column reduction](#2-persistence-via-z-column-reduction) · [3) The stability theorem](#3-the-stability-theorem) · [4) The regular-polygon chord](#4-the-regular-polygon-chord)
  - [5) Bottleneck and Wasserstein](#5-bottleneck-and-wasserstein) · [6) Persistence landscapes](#6-persistence-landscapes) · [7) The elder rule as a block score](#7-the-elder-rule-as-a-block-score) · [8) Takens' embedding theorem](#8-takens-embedding-theorem)
  - [9) The lazy witness complex](#9-the-lazy-witness-complex) · [10) Persistent entropy](#10-persistent-entropy) · [11) Persistence images](#11-persistence-images) · [12) The placement statistic](#12-the-placement-statistic)
  - [13) The routing gap ratio](#13-the-routing-gap-ratio) · [14) Numerically stable softmax](#14-numerically-stable-softmax) · [15) Chebyshev's inequality as an allocator guard](#15-chebyshevs-inequality-as-an-allocator-guard)

**The code**
- [Implementation](#implementation)
- [The Persistence Engine In Detail](#the-persistence-engine-in-detail)
- [The Attention Subsystem In Detail](#the-attention-subsystem-in-detail)
- [The Scheduled Attention Port In Detail](#the-scheduled-attention-port-in-detail)
- [The ML Subsystem](#the-ml-subsystem)
- [The Runtime Substrate](#the-runtime-substrate) — memory · governor · aether · manifold · topology · state
- [The Kernel In Detail](#the-kernel-in-detail)
- [The Lean Formalization](#the-lean-formalization)
- [Complexity Reference](#complexity-reference)
- [The Duplicate Crate Problem](#the-duplicate-crate-problem)
- [Contributing](#contributing)
- [How To Read This Repository](#how-to-read-this-repository)

**The evidence**
- [The Test Suite, Test By Test](#the-test-suite-test-by-test)
- [Results](#results)
- [Mutation Testing, Or How I Learned To Stop Trusting Green Checkmarks](#mutation-testing-or-how-i-learned-to-stop-trusting-green-checkmarks)
- [What We Got Wrong](#what-we-got-wrong)
- [Reproducing Every Number In This Document](#reproducing-every-number-in-this-document)

**The rest**
- [Design Decisions That Seemed Good At 3 AM](#design-decisions-that-seemed-good-at-3-am)
- [The FAQ Nobody Asked For](#the-faq-nobody-asked-for)
- [Worked Examples](#worked-examples)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Building And Testing](#building-and-testing)
- [Limitations](#limitations)
- [Glossary Of Words I Made Up](#glossary-of-words-i-made-up)
- [License](#license)

---

## Before You Read This

> **⚠️ Content warning:** this README contains strong opinions about algebraic topology, an unreasonable number of tables, and at least one section where a headline result gets dismantled by its own test suite. If you believe that a passing test suite means the code is correct, the following may cause discomfort.

Aether-Lang is a programming language where **persistent homology is a builtin**. Not a library you import. A builtin, next to `if` and `while`, with a keyword and a parser rule and everything.

The pitch is one sentence: *some loops should terminate when the shape of the data stops changing, not when a float gets small.* Everything else here is consequence.

**What this README promises:**

- Every number is measured. If it was not run, it is marked unmeasured and stays out of the tables.
- Every claim names what it was compared against. A speedup with no baseline is a vibe.
- The negative results get their own section, near the top, with the numbers that killed them.
- Sarcasm, because I have spent real hours of my finite life discovering that `multiboot2::load` was removed in 0.24.

**What this README does not promise:**

- Brevity. You saw the table of contents.
- That the language is production-ready. It is a research language. It has a seal emoji as a keyword.
- GPU acceleration of the language or the topology engine. There is a real GPU backend — `aether-gpu`, 60 tests, measured on an RTX 4060 — and **nothing calls it**. The cost and precision of both candidate integrations are measured; the integrations are not made. An earlier version of this line said there was no GPU at all, which was true of the `wgpu` dependency it described and is no longer true of the tree.

Grab a coffee. There are 24,180 lines of Rust and 10,474 lines of Lean below, and roughly a third of this document is about the ways I was wrong.

(Those two figures replaced 21,262 and 11,652, which appeared in this README until a rewrite re-counted them. Neither was reproducible. If a document is going to insist that every number carries a command, it has to survive that rule being pointed at itself. The command is in [Reproducing Every Number](#reproducing-every-number-in-this-document).)

---

## The 30-Second Pitch

**What if a `while` loop could ask "has the topology stabilised?" instead of "is the error small?"**

```aether
🦭 until convergence(1e-6) {
    regress { model: "polynomial", escalate: true }~
}
```

That is a **seal loop**. It runs until the Betti numbers of the residual manifold stop changing, not until a scalar dips below a threshold. The distinction matters when your loss is noisy but your *structure* is stable — the loop exits on the shape, and the shape is a discrete invariant, so it does not jitter.

Underneath, `aether-core` computes exact persistent homology over 𝔽₂ for H₀, H₁ and H₂, in `no_std`, with a bounded simplex budget so a runaway workload fails fast instead of eating your RAM.

### Why the seal

Because `🦭` is a valid identifier in the lexer and I thought it was funny at 2 AM. It is still funny. `seal until` also works if your terminal has opinions.

### Key differentiators

| Capability | Aether-Lang | ripser | GUDHI | giotto-tda |
|---|---|---|---|---|
| Persistent homology H₀/H₁/H₂ | ✅ | ✅ | ✅ | ✅ |
| Bottleneck / Wasserstein distance | ✅ | ❌ | ✅ | ✅ |
| Landscapes / images | ✅ | ❌ | ✅ | ✅ |
| **Topology as a language primitive** | ✅ | ❌ | ❌ | ❌ |
| **Convergence on Betti numbers** | ✅ | ❌ | ❌ | ❌ |
| **`no_std`, runs with no OS** | ✅ | ❌ | ❌ | ❌ |
| **Bare-metal x86_64 kernel** | ✅ | ❌ | ❌ | ❌ |
| Core math dependencies | **3 declared** (`libm`, `heapless`, `nalgebra`), 2 used | many | many | many |
| Mature, fast, widely used | ❌ | ✅ | ✅ | ✅ |
| Verified against the others | ❌ **not yet** | — | — | — |

That last row is the important one and it is deliberately in the table. External parity against a pinned `ripser` or `gudhi` is **not done**. The invariant suite is not parity — a self-consistently wrong implementation can satisfy every internal property. It is gated in the ledger and it is the single largest correctness debt in this repository.

---

## Abstract

Aether-Lang is a research programming language in which persistent homology is a first-class primitive rather than a library call. The core insight is that many iterative numerical procedures have a *structural* fixed point that arrives before, and is more stable than, their scalar one: the Betti numbers of the residual point cloud stop changing while the loss is still fluctuating. Aether-Lang exposes that as control flow — a `seal until convergence(ε)` loop terminates on a topological invariant. The implementation is a bounded exact 𝔽₂ persistence engine over Vietoris–Rips and lazy-witness complexes, supporting H₀ through H₂, written in `no_std` Rust with `libm` as its only mathematical dependency, so the same code runs under a CLI on Windows and inside a bare-metal x86_64 kernel with no operating system beneath it. On top of the engine sit the standard diagram metrics (exact bottleneck via threshold-graph matching, exact p-Wasserstein via Hungarian assignment) and vectorizations (landscapes, images, persistent entropy). The engine is validated by 11 property tests encoding the Cohen-Steiner–Edelsbrunner–Harer stability theorem and six other invariants, mutation-tested against 8 injected defects, and checked against a closed-form ground truth it reproduces to 1e-12. Indexing the simplex-face lookup reduced the scale test suite from 29.07 s to 1.10 s, a 26× improvement on identical assertions. A port of the topology-derived sparse attention kernel merged as `triton-lang/kernels#22` reproduces its CSR block schedules exactly and achieves 58.8% block reduction at test scale.

---

## Why I Did This To Myself

*(a short, mildly unhinged history)*

Somebody once looked at a training loop, watched the loss oscillate for the four hundredth epoch, and thought: *the loss is lying to me. The answer stopped changing shape twenty epochs ago.*

That somebody was me. It was not a healthy thought, but it was a correct one.

### The origin

Topological data analysis has a beautiful property almost nobody exploits in *control flow*: Betti numbers are **integers**. They do not wobble. A loss of `0.0341` versus `0.0339` is noise; β₁ going from 3 to 1 is an event. If your convergence criterion is a discrete invariant, you get a stopping rule with no epsilon-tuning ritual attached.

So I built a language where that is a keyword.

### Why not just a Python library

Because I wanted it to run on a kernel with no OS, and `scikit-tda` has opinions about `numpy` that a bare-metal x86_64 target does not share. `aether-core` is `no_std`, computes its mathematics against `libm`, and compiles for `thumbv7m-none-eabi` — a Cortex-M3. Try that with `gudhi`.

(It also declares `nalgebra`, which it does not use. That is a defect, not a feature, and it is [written up below](#6-nalgebra-a-second-phantom-dependency) rather than papered over.)

Also because a library call is a library call, but a *keyword* changes how you think. `🦭 until convergence(1e-6)` reads as a loop. `while not tda.has_converged(diagram, prev, 1e-6):` reads as homework.

### Why not just use ripser

I should. You should. `ripser` is excellent, it is fast, and it has been checked by more people than have ever read this file. If you need production TDA, use ripser.

This repository exists because I wanted the primitives *inside* a runtime that also owns the scheduler and the allocator, and because I wanted to know what happens when topology makes execution decisions rather than describing data after the fact. Different questions. One of them has a market.

### The part where a repository audit ruined my afternoon

Partway through this project I ran an over-engineering audit on my own codebase and discovered:

- The CI workflow triggered on `main` and `develop`. **This repository's branch is `master`.** No CI run had ever executed. Not once, in the entire history of the project.
- Consequently, `aether-kernel` had silently stopped compiling.
- The `Dockerfile` copied crate directories from paths that have never existed.
- `pyproject.toml` was publishing the description **"Current World's Fastest Agentic AI Language"** to PyPI, three files away from a README section titled *Evidence Policy* that explicitly forbids unverified speedup claims.

All fixed; receipts in [PR #177](https://github.com/teerthsharma/Aether-Lang/pull/177). The lesson, offered freely: **a green checkmark you have never seen is not a green checkmark.**

---

## Honest Status Dashboard

The rule: a row is **Active** only if a command in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) produces its evidence. Test count in a file is not evidence if the file never runs.

A third status was needed once the GPU backend arrived. 🖥️ **Hardware-gated** means the tests need an adapter no CI runner has. They were briefly worse than useless: a test that returns early on a missing adapter *passes*, so `cargo test --workspace` reported roughly forty GPU tests green while executing none of them — the exact green checkmark this document spends a section warning about, built by its own author. They are now `#[ignore]`d behind an off-by-default `gpu` feature, so the same run prints `0 passed; 38 ignored`. CI still compiles them with `cargo build -p aether-gpu --tests --features gpu`, so a broken one is caught rather than hidden behind the ignore. `AETHER_REQUIRE_GPU=1` additionally turns a missing adapter into a failure, for a run that must prove the hardware path was exercised. The honest reading of a hardware-gated row is "verified on one developer machine", which is weaker than every other Active row here.

| Subsystem | Status | Evidence |
|---|---|---|
| Lexer, parser, AST | ✅ Active | 11 tests, `cargo test -p aether-lang` |
| Interpreter — assignment, loops, functions, classes | ✅ Active | 11 tests |
| `topology.ph` / `betti` / `intervals` | ✅ Active | interpreter topology tests |
| Persistent homology H₀/H₁/H₂ | ✅ Active | 9 in-module + **11 invariants** |
| Lazy witness complex | ✅ Active | `persistence.rs` test |
| Bottleneck / Wasserstein / landscapes / images | ✅ Active | **17 tests** |
| Sparse attention reference kernel | ✅ Active | **29 contracts** |
| Scheduled attention (Triton port) | ✅ Active | **16 tests** |
| Scale past 32 points | ✅ Active | **7 tests** |
| `no_std` on a real embedded target | ✅ Active | builds `thumbv7m-none-eabi` |
| Kernel compiles bare metal | ✅ Active | builds `x86_64-unknown-none` |
| Titan VM language parity | 🚧 Partial | needs per-construct tests |
| Static type checking | 🚧 Partial | checker exists, diagnostics thin |
| Sparse scheduler | ⛔ **Ungated** | 4 tests exist and **never execute** — `no_std` bin, no test harness |
| Kernel *boots* | ⛔ Ungated | compiles ≠ boots; needs QEMU logs |
| External TDA parity | ⛔ Ungated | no ripser/GUDHI fixture comparison |
| Lean 4 formalization | ⛔ Ungated | 10,474 lines, 48 theorems, **no `lake build` in CI** |
| GPU compute backend (`aether-gpu`) | 🖥️ **Hardware-gated** | **41 tests**, RTX 4060 / Vulkan. `cargo test -p aether-gpu --features gpu --release`. In CI they report as **ignored**, not passed |
| GPU used by `aether-core` | ⛔ **Ungated** | nothing routes through it; cost and precision measured, integration not made |
| Attention backward pass | ❌ Does not exist | forward only; no gradcheck possible |
| Wall-clock speedup claims | ❌ Withdrawn | see [What We Got Wrong](#what-we-got-wrong) |

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WORKSPACE GATE                        branch master, nightly, Win11
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  cargo fmt --all -- --check                                   clean
  cargo clippy -D correctness -D suspicious                     clean
  cargo test --workspace --exclude aether-kernel     223 / 223 passed
  cargo build -p aether-kernel --target x86_64-unknown-none        ok
  cargo build -p aether-core  --target thumbv7m-none-eabi          ok
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Rust lines (crates/)                                        24,180
  Lean lines (Aether/)              10,474   theorems 48   sorry 0
  Test suites gated in CI                                          7
  Claims withdrawn during audit                                    6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Quick Start — Five Minutes To A Betti Number

```bash
git clone https://github.com/teerthsharma/Aether-Lang.git
cd Aether-Lang
cargo build -p aether-cli --release
```

Write `loop.aether`:

```aether
import topology~

let data = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0]~
manifold M = embed(data, tau=1)~

let diagram = topology.ph(M, max_dim=1, mode="vr", max_points=16)~
let b = topology.betti(diagram, radius=0.5)~
print(b)~
```

Run it:

```bash
cargo run -p aether-cli -- run loop.aether
```

```
═══════════════════════════════════════════════════════════════
  🛡️ AEGIS - Running: loop.aether
  Mode: bio
═══════════════════════════════════════════════════════════════
List([Num(4.0), Num(0.0), Num(0.0)])

Bio-Script Execution complete. 🦭
```

β₀ = 4, β₁ = 0, β₂ = 0 at radius 0.5. Four connected components, no loops — the periodic signal, time-delay embedded and resolved into components at that scale.

Yes, `print` on a list emits the runtime's `Debug` form. It is ugly. It is on the list. The list is long.

Other commands:

```bash
cargo run -p aether-cli -- repl              # interactive
cargo run -p aether-cli -- check f.aether    # parse only, no execution
```

---

## The Language

### Tildes, Seals, And Other Crimes

Statements terminate with `~`. Not `;`. There is no deep reason. It looked like a wave, waves are manifolds if you squint, and by the time I reconsidered the lexer had opinions.

```aether
let x = 10~
let xs = [1.0, 2.0, 3.0]~

fn dist(a, b) {
    return (a - b) * (a - b)~
}

if x > 5 { print("big")~ } else { print("small")~ }

for i in xs { print(i)~ }
```

Keywords the lexer actually recognises — a more honest list than "the language supports":

`let` `fn` `return` `if` `else` `for` `while` `in` `break` `continue` `class` `new` `self` `import` `from` `as` `true` `false` `seal` `until` `convergence` `escalate` `manifold` `embed` `block` `cluster` `regress` `render` `project` `dim` `tau` `model` `center` `spread` `color` `axis` `format` `output`

### The Topology Module

```aether
import topology~

manifold M = embed(data, dim=3, tau=5)~

let diagram   = topology.ph(M, max_dim=2, mode="vr", max_points=16)~
let b         = topology.betti(diagram, radius=0.4)~
let intervals = topology.intervals(diagram)~
```

`topology.ph` calls straight into the bounded persistence engine. `mode` selects `"vr"` (Vietoris–Rips) or the lazy-witness complex. `max_points` and the internal simplex cap are a **budget, not a correctness limit** — exceeding them returns `TooManyPoints` or `TooManySimplices` rather than quietly degrading or exhausting memory.

`topology.betti(diagram, radius=r)` returns `[β₀, β₁, β₂]` at that filtration value.

### Seal Loops

```aether
🦭 until convergence(1e-6) {
    regress { model: "polynomial", escalate: true }~
}
```

The body runs; after each iteration the interpreter recomputes the residual manifold's homology and compares Betti numbers to the previous iteration. When they stop changing — and the scalar tolerance is also satisfied — the loop exits. `escalate: true` promotes the regression model when the current one plateaus.

`seal until` is the ASCII spelling, for people whose editors are cowards.

### The convergence condition is a three-way choice

The AST is more honest than the pitch. `ConvergenceCond` has three variants:

```rust
pub enum ConvergenceCond {
    Epsilon(Number),              // scalar tolerance
    BettiStable { epochs: u32 },  // topological
    Custom(Expr),                 // arbitrary predicate
}
```

`convergence(1e-6)` parses to `Epsilon`. The **topological** stopping rule is `BettiStable { epochs }` — hold the Betti vector constant for `epochs` iterations, then exit. That `epochs` count is the [`stability_window`](#convergence--where-topology-becomes-control-flow) discussed in the convergence module, surfaced into the grammar.

Stating the consequence plainly, because the 30-second pitch overstates it: **`convergence(1e-6)` is a scalar criterion.** The seal loop's headline example uses the scalar variant. The topological variant exists, is parsed, and is implemented — but a reader who assumed every seal loop stops on topology would be wrong, and this document would rather say so than let the ambiguity ride.

### The full statement grammar

Every variant the parser actually produces, from `StmtKind`:

| Statement | Form | Notes |
|---|---|---|
| `Var` | `let x = expr~` | |
| `Assign` | `x = expr~` | |
| `Fn` | `fn name(a, b) { ... }` | |
| `Return` | `return expr~` | |
| `If` | `if cond { ... } else { ... }` | |
| `While` | `while cond { ... }` | |
| `For` | `for i in iterable { ... }` | |
| `Loop` | `🦭 until convergence(ε) { ... }` | The seal loop |
| `Break` / `Continue` | `break~` / `continue~` | |
| `Class` | `class Name { ... }` | With `new` and `self` |
| `Import` | `import topology~`, `from x import y~` | |
| `Manifold` | `manifold M = embed(data, dim=3, tau=5)~` | First-class declaration |
| `Block` | `block { ... }` | |
| `Regress` | `regress { model: "polynomial", escalate: true }~` | |
| `Render` | `render { ... }~` | ASCII / WebGL export |
| `Expr` | any expression as a statement | |
| `Empty` | `~` | |

`Manifold`, `Block`, `Regress` and `Render` being **statement kinds rather than function calls** is the concrete meaning of "topology is a language primitive". They have parser rules and AST nodes, not entries in a builtin table.

### The expression grammar

From `ExprKind`:

`Literal` · `Ident` · `BinaryOp` · `UnaryOp` · `FieldAccess` · `Call` · `MethodCall` · `Index` · `Config` · `New` · `Range` · `List`

Binary operators: `+` `-` `*` `/` `%` `==` `!=` `<` `>` `<=` `>=` `&&` `||` — as `Add Sub Mul Div Mod Eq Neq Lt Gt Le Ge And Or`. Unary: `Neg`, `Not`.

Two variants worth calling out:

**`Config(Vec<ConfigPair>)`** is a first-class brace-delimited configuration literal — `{ model: "polynomial", escalate: true }`. It is what makes `regress` read like a declaration rather than a call with six positional arguments.

**`CallArg::Named { name, value }`** means named arguments are in the grammar, not simulated with a config object. `topology.ph(M, max_dim=1, mode="vr", max_points=16)` parses natively, and a reader can tell `max_dim` from `max_points` at the call site without consulting a signature.

### Numbers are represented oddly, deliberately

```rust
pub enum Number {
    Int(i64),
    Float { int_part: i64, frac_part: i64 },
}
```

Floats are carried as **integer and fractional parts** rather than as `f64` at the AST level. This preserves the literal exactly as written through parsing, so `0.1` round-trips as `0.1` rather than as the nearest double's decimal expansion. Evaluation converts to `f64`; the AST keeps the source form.

### Compiler internals, by size

| File | Lines | Role |
|---|---:|---|
| `interpreter.rs` | 1,803 | Tree-walking evaluator; holds the topology builtins, seal-loop convergence, manifold primitives |
| `parser.rs` | 1,107 | Recursive-descent, produces a positioned AST |
| `vm.rs` | 806 | `TitanVM` bytecode VM — behind the interpreter on coverage |
| `lexer.rs` | 444 | Tokeniser, including the four-byte `🦭` codepoint |
| `ast.rs` | 343 | The node definitions above |
| `ascii_render.rs` | 149 | Terminal rendering for `render` |
| `webgl_export.rs` | 105 | WebGL export path for `render` |
| `python.rs` | 70 | `pyo3` surface; the bindings package is currently empty |

The AST carries **source positions**, which is why `aether check` reports `line`, `column`, `message` rather than a panic backtrace. That is a small thing that is disproportionately visible to anyone who actually uses the language.

**Two implementations, honestly ranked.** The tree-walking interpreter is the reference; `TitanVM` is a bytecode VM that is behind it on construct coverage. There is no per-construct parity suite, which is why the [status dashboard](#honest-status-dashboard) lists Titan VM parity as 🚧 Partial rather than green. Two execution engines with no differential test between them is a standing invitation to divergence.

---

## Background

### Why Topology?

Numerical convergence criteria are almost always scalar: watch a residual, wait for it to drop below ε. This works, and it has two well-known failure modes anybody who has trained anything has felt.

**Scalars are noisy where structure is not.** A loss oscillating in the third decimal makes you either tune a patience parameter or add a moving average, both hyperparameters you now have to defend. β₁ going 3 → 3 → 3 → 1 is not noisy. It is an integer sequence with an event in it.

**Scalars are one-dimensional.** They compress the entire state into a single number, then ask you to threshold it. A persistence diagram keeps the multi-scale structure. Two residual fields with identical L₂ norms can have completely different topology, and if what you care about is whether the model found the *shape* of the data, the norm answers a different question than the one you asked.

**Discrete invariants compose.** Betti numbers are homotopy invariants — unchanged under continuous deformation. That is exactly the property you want in a stopping rule: robust to the wiggle, sensitive to the event.

**The cost is bounded and knowable.** For `n` points capped at homology dimension 2, the Rips complex has at most `n + C(n,2) + C(n,3) + C(n,4)` simplices. You can compute that before allocating, which is why the engine takes a budget and fails fast rather than discovering the problem at 40 GB resident.

### Why A Language And Not A Library?

Two reasons, one principled and one petty.

The principled one: **`no_std`**. The core math computes against `libm`, uses `heapless` for fixed-capacity containers, and compiles for targets with no operating system and no allocator by default. That is not retrofitted onto a Python library; it is designed for on line one. The same persistence code backing `topology.ph` in the CLI runs inside `aether-kernel` on bare x86_64, where it informs scheduling.

The petty one: a keyword changes how you reach for a thing. When persistence is a builtin, you use it in a loop condition. When it is `from gudhi import RipsComplex`, you use it in a plot at the end of the notebook.

### Prior Art

Every system below is real, currently maintained, and better than this one at the thing it was built for.

| System | Language | H₀/H₁/H₂ | Metrics | Vectorizations | `no_std` | Topology as control flow | Maturity |
|---|---|---|---|---|---|---|---|
| [ripser](https://github.com/Ripser/ripser) | C++ | ✅ (Rips, fast) | ❌ | ❌ | ❌ | ❌ | Production |
| [GUDHI](https://gudhi.inria.fr/) | C++/Python | ✅ (many complexes) | ✅ | ✅ | ❌ | ❌ | Production |
| [giotto-tda](https://github.com/giotto-ai/giotto-tda) | Python/C++ | ✅ | ✅ | ✅ | ❌ | ❌ | Production |
| [Dionysus](https://www.mrzv.org/software/dionysus2/) | C++/Python | ✅ | ✅ | partial | ❌ | ❌ | Mature |
| **Aether-Lang** | Rust | ✅ (Rips + witness) | ✅ | ✅ | ✅ | ✅ | **Research** |

**There is deliberately no timing column.** I have not run ripser or GUDHI on the same hardware with the same inputs, and inventing a row saying "12× faster than ripser" is exactly the class of claim this repository spent an entire audit removing. When the parity harness exists, numbers go here. Until then the honest statement is:

> Aether-Lang's persistence engine has **not** been benchmarked against ripser, GUDHI, giotto-tda or Dionysus, and has **not** been verified to agree with them on shared fixtures. Its correctness evidence is internal invariants plus mutation testing.

For what it is worth, the measured ceiling below (H₁ at n=300 in 131 s) strongly suggests it is **substantially slower** than ripser, which handles clouds orders of magnitude larger. This engine trades throughput for `no_std` and exactness-with-a-budget. If that is not the trade you need, use ripser.

---

## Theoretical Foundation

Everything here is implemented in [`persistence.rs`](crates/aether-core/src/persistence.rs) and [`diagram.rs`](crates/aether-core/src/diagram.rs). Nothing here is aspirational.

### 1) The Vietoris–Rips filtration

For a finite cloud $X = \{x_1, \dots, x_n\} \subset \mathbb{R}^d$ with the Euclidean metric, the Rips complex at scale $r$ admits a simplex on any vertex set whose pairwise distances are all at most $r$:

$$
\mathrm{VR}_r(X) = \bigl\{\, \sigma \subseteq X \;:\; \mathrm{diam}(\sigma) \le r \,\bigr\}
$$

so a simplex's filtration value is its diameter:

$$
f(\sigma) = \max_{u,v \in \sigma} \lVert u - v \rVert_2
$$

The engine enumerates simplices up to dimension 3 (tetrahedra), giving H₀, H₁ and H₂. Faces always enter no later than their cofaces, since a face's diameter cannot exceed its coface's — `every_face_is_present_and_precedes_its_coface` asserts this on five complexes rather than trusting the argument.

### 2) Persistence via Z₂ column reduction

Order simplices by $(f(\sigma), \dim\sigma, \text{vertices})$, build the boundary matrix $\partial$ over $\mathbb{F}_2$, and reduce left to right so that no two columns share a lowest nonzero row:

$$
R = \partial V, \qquad \mathrm{low}(j) \ne \mathrm{low}(j') \;\; \text{for } j \ne j'
$$

A column reducing to zero births a class; a column with surviving $\mathrm{low}(j) = i$ pairs a birth at $\sigma_i$ with a death at $\sigma_j$, giving the interval $[f(\sigma_i), f(\sigma_j))$. Unpaired classes persist to infinity.

The defining identity of the chain complex,

$$
\partial \circ \partial = 0 \pmod 2
$$

is asserted directly, on every simplex of five complexes, by `boundary_of_boundary_is_zero_over_z2`. If it fails, every rank downstream is meaningless — and it fails *silently*, which is the entire argument for testing it.

### 3) The stability theorem

The result that makes persistence usable as a feature at all. Cohen-Steiner, Edelsbrunner and Harer: for clouds $X, Y$ within Hausdorff distance $\varepsilon$,

$$
d_B\bigl(\mathrm{Dgm}(X), \mathrm{Dgm}(Y)\bigr) \;\le\; 2\varepsilon
$$

for the Rips filtration, where $d_B$ is bottleneck distance. The constant is $2\varepsilon$ via the Hausdorff bound; $\varepsilon$ produces spurious failures and $4\varepsilon$ passes a broken implementation.

`bottleneck_distance_respects_the_stability_bound` asserts exactly this over 12 seed × ε combinations. It is the single most valuable test in the repository: wrong pairing, a dropped bar, a mishandled infinite death, and an early-terminating reduction all surface here and nowhere else.

### 4) The regular-polygon chord

The Rips complex of a circle of radius $r$ has one H₁ class dying at $\sqrt{3}\,r$. That is the **continuous** statement. For $n$ points sampled uniformly, the cycle dies when the first triangle spanning roughly a third of the polygon enters, so the death is the chord subtending $\lceil n/3 \rceil$ steps:

$$
d_{H_1}(n, r) \;=\; 2 r \sin\!\left(\frac{\pi \lceil n/3 \rceil}{n}\right)
$$

equal to $\sqrt{3}\,r$ exactly when $3 \mid n$, and decreasing to it otherwise.

**The engine reproduces this to 1e-12** for $n \in \{9, 10, 11, 12, 13, 17, 24, 48\}$. Worth dwelling on: an earlier test asserted "within 5% of $\sqrt{3}r$", which would have passed an implementation systematically wrong. Deriving the exact finite-$n$ form turned a tolerance into an identity.

### 5) Bottleneck and Wasserstein

For diagrams $D_1, D_2$ and bijections $\eta$ that may match points to the diagonal $\Delta$:

$$
d_B(D_1, D_2) = \inf_{\eta} \; \sup_{x \in D_1} \lVert x - \eta(x) \rVert_\infty
$$

$$
W_p(D_1, D_2) = \left( \inf_{\eta} \; \sum_{x \in D_1} \lVert x - \eta(x) \rVert_\infty^{\,p} \right)^{1/p}
$$

with the diagonal cost being half a bar's persistence, $\lVert x - \Delta \rVert_\infty = (d - b)/2$.

Both **exact**. Bottleneck by binary search over the finite candidate-cost set, feasibility decided by Kuhn's augmenting-path matching on the threshold graph. Wasserstein by the Hungarian algorithm on the same $(n+m)^2$ cost matrix — $O((n+m)^3)$, marked with a `ponytail:` comment naming the ceiling and the upgrade path (auction or Sinkhorn), triggered when a `PersistenceConfig` preset raises `max_points` past 2048. The widest preset admits 512 points, so $(n+m)$ stays in the hundreds today.

Since $d_B = \lim_{p \to \infty} W_p$, the ordering $W_p \ge d_B$ must hold for all $p \ge 1$. Asserted for $p \in \{1, 2, 4\}$.

### 6) Persistence landscapes

Each bar $[b, d]$ contributes a tent

$$
\lambda_{[b,d]}(t) = \max\bigl(0, \min(t - b, \; d - t)\bigr)
$$

and the $k$-th landscape is the pointwise $k$-th largest. Landscapes are **1-Lipschitz** in bottleneck distance,

$$
\lVert \lambda(D_1) - \lambda(D_2) \rVert_\infty \le d_B(D_1, D_2)
$$

which is what makes a landscape a *feature* rather than a hash: small change in, small change out. `landscape_is_one_lipschitz_in_the_bottleneck_distance` asserts it.

### 7) The elder rule as a block score

In H₀, when two components merge the **younger** one dies. Run single-linkage over centroids and record, for each point, the merge distance at which its component was absorbed. That number is an H₀ death time, measuring how long the centroid stayed distinct from everything else.

This is the salience score in the [scheduled attention kernel](#scheduled-attention). Exactly one point scores 0 — the component never absorbed — which follows from an invariant every merge preserves: *each component holds exactly one point that has never been written.*

### 8) Takens' embedding theorem

The justification for turning a scalar time series into a point cloud at all. For a dynamical system with a $d$-dimensional attractor and a generic observation function, the delay map

$$
\Phi_{\tau,m}(t) \;=\; \bigl(x_t,\; x_{t-\tau},\; x_{t-2\tau},\; \dots,\; x_{t-(m-1)\tau}\bigr) \;\in\; \mathbb{R}^{m}
$$

is an embedding of the attractor whenever $m > 2d$ — the reconstructed cloud is **diffeomorphic** to the original attractor, so its topology is the attractor's topology.

This is the theorem that makes `manifold M = embed(data, tau=1)` more than a reshaping trick: it is why $\beta_1$ of the delay cloud says something about the *system* rather than about the windowing. Implemented in `TimeDelayEmbedder<D>`, where `D` is $m$ and `tau` is $\tau$.

**The caveat this project inherits and does not solve:** the theorem is generic and asymptotic. It guarantees an embedding exists for $m > 2d$; it does not tell you $d$, does not tell you $\tau$, and says nothing about finite noisy samples. Choosing $\tau$ badly gives a cloud that is technically an embedding and practically a diagonal smear. Aether-Lang requires you to pass $\tau$ and offers no automatic mutual-information or false-nearest-neighbour selection. That is a real gap, not a design stance.

### 9) The lazy witness complex

Rips on $n$ points enumerates $\binom{n}{k+1}$ candidate $k$-simplices. The witness construction breaks that dependence by choosing $\ell \ll n$ **landmarks** $L$ and letting the remaining points serve as witnesses. A simplex $\sigma \subseteq L$ enters the complex at

$$
f(\sigma) \;=\; \min_{w \in X} \; \max_{v \in \sigma} \; \bigl( d(w, v) - m_w \bigr)
$$

where $m_w$ is the distance from witness $w$ to its nearest landmark. Cost now scales with $\ell$, not $n$, which is what makes `low_load()` — 24 landmarks — viable on a Cortex-M3.

It is an **approximation**. It is not asserted anywhere in this repository to produce diagrams equal to the Rips diagram on the same cloud, and it should not be read as doing so.

### 10) Persistent entropy

Treat the normalised bar lengths of a diagram as a probability distribution. For bars $\{[b_i, d_i)\}$ with lengths $\ell_i = d_i - b_i$ and total $L = \sum_i \ell_i$:

$$
p_i = \frac{\ell_i}{L}, \qquad E(D) \;=\; -\sum_i p_i \log p_i
$$

One long bar and many short ones gives low entropy — a diagram dominated by a single feature. Many comparable bars gives high entropy — no scale dominates. It is a **one-number summary of how concentrated the topological signal is**, which is exactly the thing a stopping rule wants to watch, and it is cheap once the diagram exists.

### 11) Persistence images

The other vectorization. Map each bar to birth–persistence coordinates $(b, \ell)$ with $\ell = d - b$, place a Gaussian at each, weight it by persistence, and integrate over a pixel grid:

$$
\rho(z) \;=\; \sum_i w(\ell_i)\, \frac{1}{2\pi\sigma^{2}} \exp\!\left( -\frac{\lVert z - (b_i, \ell_i)\rVert^{2}}{2\sigma^{2}} \right)
$$

The weight $w$ is **linear in persistence** and vanishes on the diagonal. That last property is not decoration: bars near the diagonal are noise, and a weight that does not vanish there makes the image discontinuous in the input — a small perturbation that creates a tiny bar would produce a jump in the feature vector. `persistence_image_weights_long_bars_more_than_short_ones` pins the weighting, and `sigma_controls_the_kernel_width` pins $\sigma$ — the latter existing only because a mutant that ignored $\sigma$ entirely survived the first version of the suite.

### 12) The placement statistic

Not textbook mathematics, but the measurement instrument every attention ablation in this document is reported against, so it belongs here rather than in a footnote.

Given a selector $S$, a random baseline $R$, and a budget-matched oracle $O$, all measured by recovered attention mass at **equal budget**:

$$
\text{placement}(S) \;=\; \frac{m(S) - m(R)}{m(O) - m(R)}
$$

Placement 0 means indistinguishable from choosing keys uniformly at random. Placement 1 means matching a selector that computed every score before choosing. The normalisation is what makes numbers comparable across input distributions, since $m(S)$ alone drifts with the key distribution and would let a selector look better simply by being tested on easier data.

**Two ways this statistic lies, both encountered here:**

Equal budget is load-bearing. A selector that declines to select — 1.00 keys per row where baselines take 5.53 — posts catastrophic placement without losing on mechanism. [It lost on budget.](#3-the-first-fix-was-also-wrong--measurement-artifact)

The denominator can vanish or invert. With a **dense** fallback, $m(S)$ exceeds $m(O)$, because dense recovers all the mass while the oracle is budget-limited. The ratio then blows up — a printed **+7.614** that would read as a 700% win over an oracle it never competed with. [The two regimes are now scored on different scales deliberately.](#bonus-a-number-that-lied)

### 13) The routing gap ratio

The decision function that makes topological routing a conditional claim rather than a universal one. Given the H₀ barcode of the key directions, compare the merge height at which the cut is taken against the heights immediately below it:

$$
\text{gap ratio} \;=\; \frac{h_{\text{cut}}}{h_{\text{below}}}
$$

A cloud with genuine cluster structure has a **large jump** at the cut — components stay separate until well past the within-cluster scale. A cloud with no structure chains: single-linkage absorbs one point at a time at nearly-equal heights, and the ratio sits near 1.

Measured separation, 6 trials each: **structured minimum 2.70, chained maximum 1.04.** No overlap. `routing_plan` uses this to decide, at runtime, whether to route or fall back to dense.

This is the sharpest form of the project's actual thesis. Not "topology makes attention faster" — that is false on uniform keys. Rather: **the H₀ barcode is a sufficient statistic for deciding whether a topological method will pay**, and it is cheap enough to consult before committing.

### 14) Numerically stable softmax

Implemented in every attention path, and stated here because the tests reference it. The identity

$$
\mathrm{softmax}(x)_i \;=\; \frac{e^{x_i - \max_j x_j}}{\sum_k e^{x_k - \max_j x_j}}
$$

holds exactly — subtracting a constant from every logit leaves the result unchanged — while bounding every exponent at or below zero, so `exp` cannot overflow. `large_logits_do_not_overflow_the_softmax` and `large_logits_stay_finite_across_scheduled_blocks` assert it on both kernels.

The degenerate case the identity does **not** cover is an empty row, where the denominator is a sum over nothing and the result is 0/0. That is a policy decision, not a numerical one: this implementation returns zeros, and `an_all_masked_row_returns_zeros_rather_than_nan` pins the policy so a later refactor cannot quietly turn it into a NaN.

### 15) Chebyshev's inequality as an allocator guard

For any distribution with finite mean $\mu$ and variance $\sigma^2$ — no normality assumption —

$$
\Pr\bigl(\lvert X - \mu\rvert \ge k\sigma\bigr) \;\le\; \frac{1}{k^{2}}
$$

The manifold heap uses this on the liveness distribution across spatial blocks. The reason it is the right inequality here is discussed in [the runtime substrate section](#the-chebyshev-guard): the allocator cannot assume a distribution of object lifetimes, because that would be an assumption about user programs. Chebyshev is loose and unconditional, which is the correct trade for a guard where a false "safe" is a leak.

---

## Implementation

### `aether-core` — the math

Platform-agnostic, `no_std`, one mathematical dependency.

| Module | What it computes | Notable decision |
|---|---|---|
| `persistence` | 𝔽₂ reduction, Rips + lazy witness, H₀–H₂ | Face lookup is a `BTreeMap` on the zero-padded vertex array. It used to be a linear scan; see [The 26x](#the-26x). |
| `diagram` | bottleneck, Wasserstein, landscapes, images, entropy | Exact, not approximate. Hungarian is cubic and says so in a comment. |
| `manifold` | time-delay embedding, sparse attention graphs | Takens embedding with configurable τ |
| `topology` | Betti numbers, shape verification | The cheap path, when you do not need a full diagram |
| `attention` | sparse-attention reference kernel + selectors | CPU reference. There is no GPU. |
| `scheduled` | CSR block-scheduled attention (Triton port) | Working set is one score tile, never `seq × seq` |
| `governor` | PID-on-manifold adaptive thresholds | The calibration knob the physical world needs |
| `memory` | manifold heap, entropy-regulated GC | Chebyshev's guard as a safety protocol |
| `ml` | regression, clustering, classification, MLP, tensors | Written from scratch, `no_std` |

The `no_std` story is not decorative:

```bash
cargo build -p aether-core --no-default-features --features no_std \
  -Z build-std=core,alloc --target thumbv7m-none-eabi
```

That is a Cortex-M3. It builds.

### `aether-lang` — the language

Lexer → AST → parser → tree-walking interpreter, plus a bytecode VM (`TitanVM`) that is behind the interpreter on coverage and is honestly labelled as such.

The interpreter is 1,976 lines and holds the topology builtins, seal-loop convergence, and manifold primitives. The parser is 1,276 lines and produces a real AST with source positions, so `aether check` gives you `line`, `column`, `message` rather than a stack trace.

### `aether-kernel` — the bare metal bit

`no_std` x86_64 microkernel. Sparse event loop waking on deviation Δ ≥ ε rather than a timer tick, `WFI` power management, topological code authentication, and a scheduler using `aether-core` to decide.

```bash
cargo build -p aether-kernel -Z build-std=core,alloc --target x86_64-unknown-none
```

It **compiles**. It is not asserted to **boot** — that needs QEMU logs and a hardware matrix. Compiling and booting are different claims, and conflating them is how projects acquire reputations.

Worth recording: this crate did not compile at all until the audit, because CI had never run. Four defects including `multiboot2::load` removed in 0.24, plus one real bug only visible once the file compiled — `BootInfo::config_root` returned a pointer to the eight-byte RSDP *signature string* where the ACPI root table address was intended.

### `aether-cli` — the part humans touch

`repl`, `run`, `check`. Rustyline for editing, clap for arguments. It accepts `.aether` and `.ae`; repository examples still use `.aegis` and `.ag` from an earlier name and print a warning — technical debt sitting in plain sight rather than hidden.

---

## The Persistence Engine In Detail

Everything in this section is `crates/aether-core/src/persistence.rs`, 783 lines, `no_std`.

### The simplex representation

A simplex is a **zero-padded fixed vertex array plus a vertex count**:

```rust
const SIMPLEX_VERTICES: usize = 4;
```

Four slots, because the engine enumerates up to tetrahedra — which is what H₂ requires and no more. A triangle `{3, 7, 11}` is stored as `([3, 7, 11, 0], 3)`. The padding is not cosmetic: it makes the array a **canonical key**, which is the whole reason the `BTreeMap` face index works. `simplex()` zero-fills the unused slots and `boundary_indices` builds faces the same way, so two routes to the same simplex produce byte-identical keys. Combinations are generated once each, so keys are unique and the map never has to resolve a collision.

This is the single decision that bought the [26× speedup](#the-26x). The prior code linear-scanned `simplices[..before]` for every face of every simplex, which is O(m²) in the simplex count and put a hard ceiling of roughly 32 points on the entire engine.

### Configuration is a budget, not a limit

```rust
pub struct PersistenceConfig {
    pub max_homology_dim: usize,
    pub max_points: usize,
    pub max_simplices: usize,
    pub max_radius: f64,
    pub complex_kind: ComplexKind,
}
```

Four presets ship, and their point caps are **derived from the measured timings in [Results](#measured-scale-ceiling)**, not guessed:

| Preset | `max_homology_dim` | `max_points` | `max_simplices` | Complex | Rationale |
|---|---:|---:|---:|---|---|
| `h2_default()` | 2 | 48 | 1,000,000 | Vietoris–Rips | Tetrahedra are the O(n⁴) term |
| `h1_dense()` | 1 | 128 | 1,000,000 | Vietoris–Rips | Dropping H₂ buys a much larger point budget |
| `h0_only()` | 0 | 512 | 1,000,000 | Vietoris–Rips | Components only; cheapest useful configuration |
| `low_load()` | 1 | 24 | 4,096 | Witness, 24 landmarks | The embedded/kernel profile |

The doc comment on `h2_default` states the policy directly: *the caps are a fail-fast budget, not a statement about correctness. Raise them explicitly when the workload justifies the wait.*

### Failing fast is a feature

```rust
pub enum PersistenceError { /* TooManyPoints, TooManySimplices, ... */ }
```

Exceeding a cap returns an error. It does **not** silently subsample, does not degrade to a coarser complex, and does not proceed until the allocator gives up. Two tests pin this behaviour explicitly — `simplex_cap_still_fails_fast_rather_than_exhausting_memory` and `point_cap_is_still_enforced_when_configured` — because the failure mode being prevented is the one that costs you a machine, and a cap nobody tested is a cap that quietly stopped working three refactors ago.

The simplex count is knowable **before** allocating. For `n` points capped at homology dimension 2, the Rips complex admits at most

$$
n + \binom{n}{2} + \binom{n}{3} + \binom{n}{4}
$$

simplices, so the check is arithmetic on the input, not a discovery made at 40 GB resident.

### The two complexes

`ComplexKind::VietorisRips` is the exact filtration described in [Theory §1](#1-the-vietorisrips-filtration).

`ComplexKind::Witness { max_landmarks }` is the **lazy witness complex**: choose `max_landmarks` landmark points, and admit a simplex on a landmark set when some data point witnesses it. Cost scales with the landmark count rather than the full cloud, which is what makes `low_load()` viable at 24 landmarks on a Cortex-M3. It is an approximation of the Rips complex and is labelled as one — it is not asserted to produce identical diagrams.

### Entry points

```rust
pub fn persistent_homology<const D: usize>(/* ... */) -> Result<PersistenceDiagram, PersistenceError>;
pub fn time_delay_persistence<const D: usize>(/* ... */) -> Result<PersistenceDiagram, PersistenceError>;
```

`time_delay_persistence` is the composition the language actually uses: Takens-embed a scalar series, then run persistence on the resulting cloud. It exists as one function because that pairing is the common case and doing it in two steps invites a τ mismatch between the embedder and the engine.

```rust
pub struct PersistenceDiagram { /* pairs */ }
impl PersistenceDiagram {
    pub fn betti_at(&self, radius: f64) -> BettiNumbers3;
}
pub struct BettiNumbers3 { /* beta_0, beta_1, beta_2 */ }
```

`betti_at` is a **query on a computed diagram**, not a separate computation. Betti numbers at radius `r` are the count of bars whose interval contains `r`, per dimension. This is why `topology.betti(diagram, radius=...)` in the language takes a diagram rather than a point cloud — computing the diagram once and querying it at many radii is the cheap direction, and the API makes the cheap direction the obvious one.

---

## The Attention Subsystem In Detail

`crates/aether-core/src/attention.rs`, 699 lines. This is a **CPU reference kernel**. There is no GPU, no SIMD, and no threading anywhere in it.

### The selector taxonomy

```rust
pub enum Selector {
    Dense,
    Local { window: usize },
    Random { budget: usize, seed: u64 },
    OracleTopK { budget: usize },
    Topological { budget: usize, radius_scale: f64 },
    TopologicalRouted { budget: usize, clusters: usize },
    Adaptive { budget: usize, clusters: usize },
}
```

Seven variants, and the two that are not selection strategies at all are the reason the ablations mean anything:

| Selector | Role | Why it exists |
|---|---|---|
| `Dense` | upper bound on quality, upper bound on cost | The thing every sparse method must beat on cost without losing quality |
| `Local { window }` | the trivial baseline | Sliding window; what you get with no cleverness at all |
| `Random { budget, seed }` | **the floor** | A selector that cannot beat random at equal budget has no mechanism |
| `OracleTopK { budget }` | **the ceiling** | Computes every score, then takes the true top-k. Not deployable — it is a *ruler* |
| `Topological { budget, radius_scale }` | Euclidean-proximity selection | The claim that [measured negative](#2-euclidean-proximity-as-an-attention-mass-proxy--negative) |
| `TopologicalRouted { budget, clusters }` | H₀-cluster routing on key directions | The claim that survived, [conditionally](#4-topological-routing-on-unstructured-keys--not-sparse-at-all) |
| `Adaptive { budget, clusters }` | routed when it pays, dense when it does not | The honest shipping default |

`Random` and `OracleTopK` are what make the placement statistic

$$
\text{placement} = \frac{\text{selector} - \text{random}}{\text{oracle} - \text{random}}
$$

a scale-free number rather than a raw attention mass whose magnitude depends on the input distribution. A selector at placement 0.0 is indistinguishable from random; at 1.0 it has matched an oracle that cheated. `the_oracle_is_priced_as_the_diagnostic_it_is` exists to stop the oracle being read as a result.

### Cost accounting, added after the fact

```rust
pub fn selection_dot_cost(/* ... */) -> f64;
pub fn dense_dot_cost(seq: usize, causal: bool) -> f64;
```

These two functions exist because **every test in the original suite measured how good a selection was and none measured what it cost**. That omission let a selector post a +0.94 placement while examining 0.999× the dense dot-product count — dense attention wearing a clustering hat. The full autopsy is in [What We Got Wrong §4](#4-topological-routing-on-unstructured-keys--not-sparse-at-all).

The lesson generalises past this repository: *a quality metric with no paired cost metric will eventually reward a method for doing more work.*

### The routing plan as a runtime check

```rust
pub struct RoutingPlan { /* gap_ratio, decision, predicted cost, ... */ }
pub fn routing_plan(/* ... */) -> RoutingPlan;
```

`routing_plan` converts the conditional claim into a decision the code makes at runtime rather than an assumption the author makes at design time. It reads the H₀ barcode of the key directions, computes a `gap_ratio`, and decides whether routing will pay **before** committing to it.

The separation is clean and measured: **structured minimum 2.70 against chained maximum 1.04**, 6 trials each, no overlap. Three tests hold that line — `the_plan_predicts_the_cost_it_will_actually_incur`, `the_plan_declines_to_route_exactly_when_routing_would_not_pay`, and `the_h0_barcode_alone_separates_the_two_regimes`.

That last name is the interesting claim: the decision needs **only the barcode**, not the raw keys. The topology is sufficient statistics for the routing decision.

### `single_linkage_clusters` and the theorem it is checked against

```rust
pub fn single_linkage_clusters(
    points: &[f64], count: usize, dim: usize, clusters: usize, normalize: bool,
) -> (Vec<usize>, Vec<f64>);
```

Returns `(assignment, merge_heights)`. Two decisions worth naming:

**Labels are canonicalised to first-occurrence order.** Otherwise the labels depend on union-find internals and two runs on the same data are not comparable. Determinism here is not fastidiousness — it is what makes `repeated_runs_are_bitwise_identical` meaningful.

**Ties are broken by index.** The edge sort compares distance, then `i`, then `j`, so the merge order and therefore the cut do not depend on sort stability.

The merge heights are exactly the finite H₀ deaths of the Vietoris–Rips persistence of the same cloud. Rather than trusting two implementations of one theorem, `single_linkage_merge_heights_equal_the_h0_persistence_deaths` asserts one against the other. This is the most valuable kind of test available in a repository that implements the same mathematics twice for different reasons: **cross-validation between independent code paths that must agree.**

`normalize` projects points to the unit sphere first, making the clustering invariant to per-point rescaling — the property the routed selector needs and the nearest-neighbour selector conspicuously lacks. A zero vector has no direction, so it is left at the origin rather than having one invented for it.

---

## The Scheduled Attention Port In Detail

`crates/aether-core/src/scheduled.rs`, 477 lines. A Rust port of [`triton-lang/kernels#22`](https://github.com/triton-lang/kernels/pull/22).

### Why the split is the point

```rust
pub struct BlockSchedule { /* CSR: indptr + indices */ }
impl BlockSchedule {
    pub fn from_rows(rows: &[Vec<usize>]) -> Result<Self, ScheduleError>;
    pub fn row(&self, q_block: usize) -> &[usize];
    pub fn num_blocks(&self) -> usize;
}
```

The port preserves the original's separation of a **combinatorial** half from a **numeric** half, and that separation is what makes each half checkable alone:

| Half | Nature | Checkable against | Result |
|---|---|---|---|
| CSR block schedule | combinatorial — which blocks | the Python builder, **exactly** | `[0,1,3,6,10]` / `[0,0,1,0,1,2,0,1,2,3]` for 4 blocks |
| Kernel | numeric — what the blocks compute | dense masked attention | max abs diff **< 1e-12**, 4 schedules × 4 geometries |

A monolithic port would only be checkable end to end, where a schedule bug and a numeric bug are indistinguishable from a wrong output. Split, the schedule is compared for **exact integer equality** with the upstream builder — no tolerance, no float comparison, no judgement call.

### The API surface

```rust
pub fn dense_causal_block_schedule(num_blocks: usize) -> BlockSchedule;
pub fn block_salience(/* ... */);
pub fn topology_block_schedule(/* ... */);
pub fn scheduled_attention(/* ... */);
pub fn dense_masked_attention(/* ... */);
pub struct TopologyScheduleConfig { /* local_radius, sink, topk */ }
pub enum ScheduleError { /* ... */ }
```

`dense_masked_attention` is in the shipping crate, not the test file, on purpose: it is the reference the sparse kernel is checked against, and a reference that lives only in tests tends to drift from the thing it references.

### The working-set claim

The schedule keeps one score tile live at a time, never a `seq × seq` matrix. This is the claim that distinguishes block-scheduled attention from masked dense attention with extra steps, and `the_working_set_does_not_grow_with_the_sequence` measures it rather than asserting it in a comment.

### What is reproduced and what is not

Reproduced: the CSR schedule (exactly), the numeric output (1e-12), the **58.8% block reduction** at 16 blocks with `local_radius=1 sink=1 topk=2`, the salience-vs-H₀-deaths agreement (1e-9), and the invariant that exactly one block scores zero.

**Not reproduced:** the upstream wall-clock figures — 56.6% block reduction at seq 1024, 80.9% at seq 4096, and 1.04×–3.48× sparse-vs-dense-CSR — measured on an RTX 4060. This port is a scalar CPU kernel. It reproduces *the answer* and *the block reduction*, not *the speed*, and no table in this document claims otherwise.

---

## The ML Subsystem

`crates/aether-core/src/ml/`, 12 modules, ~3,990 lines, all `no_std`, all written from scratch against `libm`. The README previously gave this subtree a single table row, which undersold roughly a fifth of the codebase.

A blanket caveat that applies to everything in this section, stated once: **these are correct-and-small implementations, not competitive ones.** They exist so the language and the kernel have learning primitives that compile with no operating system underneath. Every one of them is outperformed by its `scikit-learn` equivalent by margins nobody has measured here and nobody should need to.

### `tensor` — the array type

```rust
pub struct Tensor { /* data, shape */ }
```

| Method | Signature | Note |
|---|---|---|
| `from_vec` | `(Vec<f64>, Vec<usize>) -> Self` | Takes ownership |
| `new` | `(&[f64], &[usize]) -> Self` | Copies |
| `zeros` / `ones` | `(&[usize]) -> Self` | |
| `kaiming_uniform` | `(&[usize]) -> Self` | He initialisation for ReLU-family nets |
| `get` / `set` | `(&[usize]) -> f64` / `(&[usize], f64)` | Multi-index |
| `matmul` | `(&Tensor) -> Tensor` | Naive triple loop |
| `add` / `sub` / `mul` / `scale` | elementwise / scalar | |
| `transpose` / `flatten` / `sum` / `map` | | `map` takes `F: Fn(f64) -> f64` |

`kaiming_uniform` rather than plain uniform is the one non-obvious choice: with ReLU activations, naive initialisation puts the network in a regime where half the gradient signal is dead on arrival, and the failure looks like a bad learning rate rather than a bad initialiser.

The `set` method takes `&self` rather than `&mut self` — interior mutability. Worth flagging as a design wrinkle rather than leaving a reader to discover it from a type error.

There is a comment at `ml/tensor.rs:6` reading *"future hooks for wgpu"*. It is the **only** occurrence of `wgpu` in the entire tree, and it is the whole basis of a GPU dependency that ships in `aether-lang`'s default feature set. See the [FAQ](#the-faq-nobody-asked-for).

### `neural` — layers, activations, optimizers

```rust
pub enum Activation { /* ... */ }
impl Activation {
    pub fn apply(&self, x: &Tensor) -> Tensor;
    pub fn apply_scalar(&self, x: f64) -> f64;
    pub fn derivative(&self, x: &Tensor) -> Tensor;
}

pub enum OptimizerConfig { /* ... */ }
pub enum OptimizerState { /* ... */ }

pub struct DenseLayer { /* ... */ }
impl DenseLayer {
    pub fn new(/* ... */) -> Self;
    pub fn init_optimizer(&mut self, config: &OptimizerConfig);
}
```

The `Activation::derivative` method existing as a **peer of `apply`** rather than being folded into a backward pass is the design decision worth naming: it makes each activation's gradient independently testable, which is where activation bugs actually live. A wrong derivative does not crash — it trains to a worse optimum, slowly, and the loss curve looks fine.

`OptimizerConfig` and `OptimizerState` are split so that configuration is `Copy`-cheap and shareable while the per-parameter state (momentum buffers and similar) lives with the layer that owns the parameters.

### `autograd` — reverse-mode differentiation

302 lines. Reverse-mode automatic differentiation over the tensor type.

**There is no gradcheck in CI for this module**, which is stated here rather than left for a reader to notice. An autograd implementation without a finite-difference check against its own forward pass is the definition of an unverified claim, and the same argument this README makes about [the missing attention backward pass](#limitations) applies with equal force here.

### `convolution` — Conv2D

```rust
pub struct Conv2D { /* ... */ }
impl Conv2D {
    pub fn new(/* ... */) -> Self;
    pub fn forward(/* ... */);
}
```

Bounded by compile-time constants, which is what makes it `no_std`-viable with no allocator:

```rust
const MAX_KERNEL_SIZE:  usize = 5;   // 3x3, 5x5
const MAX_CHANNELS_IN:  usize = 3;   // RGB
const MAX_CHANNELS_OUT: usize = 8;
const MAX_IMG_DIM:      usize = 32;
```

**Forward only.** A convolution with no backward pass is a feature extractor, not a trainable layer, and calling it a CNN would be an overclaim.

### `regressor` — manifold regression with model escalation

```rust
pub enum ModelType { /* ... */ }
impl ModelType { pub fn complexity(&self) -> u8; }

pub struct Coefficients { /* max 8 terms */ }
impl Coefficients { pub fn eval_polynomial(&self, x: f64) -> f64; }

pub struct ManifoldRegressor<const D: usize> { /* ... */ }
impl<const D: usize> ManifoldRegressor<D> {
    pub fn fit(&mut self) -> f64;
    pub fn predict(&self, point: &[f64; D]) -> f64;
    pub fn upgrade_model(&mut self);
}
```

This module is what `regress { model: "polynomial", escalate: true }` calls. `upgrade_model` promotes the model along the `complexity()` ordering when the current one plateaus — the machinery behind `escalate: true` in a seal loop.

Capped at `MAX_DEGREE = 8` and `MAX_POINTS = 256`. The degree cap is not arbitrary: fitting a high-degree polynomial through few points is a numerically ill-conditioned way to overfit, and a cap is a cheaper defence than a condition-number check nobody will read.

### `clustering` — three algorithms, and the one the language uses

```rust
pub struct KMeans<const D: usize> { /* builder: with_max_iter, with_tol, with_seed */ }
pub struct DBSCAN<const D: usize> { /* new(epsilon, min_samples) */ }
pub struct AgglomerativeClustering<const D: usize> { /* new(Linkage) */ }
pub enum Linkage { /* ... */ }
pub fn auto_k_selection<const D: usize>(data: &[[f64; D]], n: usize, epsilon: f64) -> usize;
```

`KMeans::with_seed` matters more than it looks: k-means is initialisation-sensitive, and an unseeded k-means makes every downstream test flaky in a way that gets diagnosed as a tolerance problem for a week before anyone checks the initialiser.

`auto_k_selection` picks `k` from the data rather than requiring the caller to know it — the topological argument for this project in miniature, since the number of clusters *is* β₀ at the right scale.

`AgglomerativeClustering::cut_tree(result, k)` cuts the dendrogram to `k` components. This is the same operation as the H₀ cut in `single_linkage_clusters`, reached from the classical direction. Note that these are **two separate implementations of overlapping mathematics** in one crate — a real piece of duplication, distinct from the `aegis`/`aether` copy-paste, and not currently cross-checked against each other the way the attention clustering is cross-checked against the persistence engine.

### `classification` — six classifiers

| Type | Fit signature returns | Predicts |
|---|---|---|
| `LogisticRegression` | `f64` (final loss) | `predict_proba` → `f64`, `predict` → `u32` |
| `KNNClassifier<D>` | — | `u32` |
| `Perceptron` | — | `i32` (sign convention) |
| `GaussianNB` | — | `u32` |
| `DecisionStump` | — | `i32` |
| `AdaBoost` | — | `i32` |

The `u32` / `i32` split in return types is not an inconsistency to be tidied away: `u32` is a class index, `i32` is a **±1 margin label**, and `Perceptron`, `DecisionStump` and `AdaBoost` are margin methods where the sign carries the meaning. `DecisionStump` exists specifically as `AdaBoost`'s weak learner.

Bounded by `MAX_CLASSES = 16`, `MAX_POINTS = 256`, `MAX_FEATURES = 32`.

### `convergence` — where topology becomes control flow

This is the module the entire language premise rests on.

```rust
pub struct BettiNumbers { /* beta_0, beta_1 */ }
impl BettiNumbers {
    pub fn is_singular(&self) -> bool;
    pub fn distance(&self, other: &Self) -> u32;
}

pub struct ConvergenceDetector { /* ... */ }
impl ConvergenceDetector {
    pub fn new(epsilon: f64, stability_window: usize) -> Self;
    pub fn record_epoch(&mut self, betti: BettiNumbers, drift: f64, error: f64);
    pub fn is_converged(&self) -> bool;
    pub fn convergence_score(&self) -> f64;
}

pub struct ResidualAnalyzer<const D: usize> { /* ... */ }
impl<const D: usize> ResidualAnalyzer<D> {
    pub fn set_residuals(&mut self, residuals: &[f64]);
    pub fn compute_betti(&self) -> BettiNumbers;
    pub fn compute_drift(&self) -> f64;
    pub fn is_collapsed(&self, threshold: f64) -> bool;
}

pub struct Answer { /* ... */ }
impl Answer {
    pub fn from_detector(detector: &ConvergenceDetector, coefficients: [f64; 8]) -> Option<Self>;
    pub fn is_perfect(&self, epsilon: f64) -> bool;
}
```

`BettiNumbers::distance` is an **integer** metric — the L₁ distance between Betti vectors. That integrality is the entire pitch: the stopping signal cannot jitter in the third decimal because it has no decimals.

`stability_window` is the honest admission inside the design. Betti numbers are discrete, but a *single* matching pair does not establish stability — one coincidental repeat is not a fixed point. The detector requires the Betti vector to hold across a window. That window is a hyperparameter, which means **topological convergence does not eliminate tuning; it moves it from a continuous threshold to a discrete count.** A count is easier to reason about than an ε, and this project's claim should be read as that and not more.

`ConvergenceDetector` also takes `drift` and `error` alongside the Betti numbers, and `is_converged` consults all three. The seal loop is therefore *not* purely topological — it exits when the topology has stabilised **and** the scalar tolerance is met. This is more honest than the pitch and is stated here so the pitch cannot be read as stronger than the code.

`Answer::from_detector` returns `Option`, so "no answer has emerged yet" is representable rather than being encoded as a sentinel value someone eventually forgets to check.

### `gossip`, `dataloader`, `linalg`, `benchmark`

| Module | Lines | Surface |
|---|---:|---|
| `gossip` | 203 | Distributed averaging, `MAX_DIM = 3` |
| `dataloader` | 127 | `DataLoader`, `BatchIterator<'a>`, batching + shuffle |
| `linalg` | 254 | Scalar reductions and distance primitives underneath the rest |
| `benchmark` | 255 | `EscalatingBenchmark<D>`, `BenchmarkConfig`, `TestFunction`, `generate_test_function` |

`benchmark` drives the model-escalation loop against standard `TestFunction` targets, capped at `MAX_EPOCHS = 1000`. It is an **internal harness for the escalation policy**, not a performance benchmark, and nothing in the [Results](#results) tables comes from it. The name is misleading and is being flagged rather than defended.

---

## The Runtime Substrate

The parts of `aether-core` that are not mathematics: how the runtime allocates, adapts, and prunes. These are the pieces that justify the phrase "a runtime that also owns the scheduler and the allocator".

### `memory` — the manifold heap

480 lines. A `no_std` allocator that organises objects **spatially** rather than by free-list order.

```rust
pub struct Gc<T> { /* index + generation */ }
pub struct ObjectHeader { /* ... */ }
pub enum HeapSlot<T> { /* ... */ }
pub struct SpatialBlock<T> { /* N = 8, contiguous arrays */ }
pub struct SpatialNode { /* aggregates child statistics */ }
pub struct ManifoldHeap<T> { /* ... */ }
pub enum MemoryMode { /* ... */ }
```

`Gc<T>` is a **generational handle**, not a pointer: an index plus a generation counter. A stale handle whose slot has been reused fails the generation check and returns `None` from `get`, instead of aliasing whatever now lives there. `Copy`/`Clone` are implemented manually to avoid an implicit `T: Copy` bound leaking onto the handle — a handle should be copyable whether or not the thing it refers to is.

`SpatialBlock` is fixed at **N = 8** with contiguous arrays, and liveness is stored in the block rather than in the slot specifically so a liveness sweep reads one packed array instead of chasing headers. The comment in the source says this is for SIMD access; there is **no SIMD in the crate today**, so read it as a layout that permits vectorisation, not one that performs it.

```rust
impl<T> ManifoldHeap<T> {
    pub fn alloc(&mut self, data: T) -> Gc<T>;
    pub fn get(&self, handle: Gc<T>) -> Option<&T>;
    pub fn get_mut(&mut self, handle: Gc<T>) -> Option<&mut T>;
    pub fn touch(&mut self, handle: Gc<T>);
    pub fn mark(&mut self, handle: Gc<T>);
    pub fn active_count(&self) -> usize;
    pub fn capacity(&self) -> usize;
}
```

`touch` records access without reading the value — the recency signal the collector uses to decide which branches are cold.

### The Chebyshev guard

```rust
pub struct ChebyshevGuard { /* ... */ }
impl ChebyshevGuard {
    pub fn calculate<T>(heap: &ManifoldHeap<T>) -> Self;
    pub fn is_safe(&self, liveness: f64) -> bool;
    pub fn regulate_entropy<F>(&mut self, tracer: F) -> usize;
}
```

Chebyshev's inequality bounds the fraction of a distribution lying more than `k` standard deviations from its mean, **for any distribution**, with no normality assumption:

$$
\Pr\bigl(\lvert X - \mu \rvert \ge k\sigma\bigr) \;\le\; \frac{1}{k^{2}}
$$

Used here as a **safety protocol**: measure the liveness distribution across spatial blocks, and treat a block whose liveness sits within the Chebyshev bound as normal variation rather than as garbage. Collection triggers only outside the bound.

The reason this is the right inequality and not a stylistic one: the runtime does not know the distribution of object lifetimes, and it cannot assume one. A Gaussian assumption here would be an assumption about *user programs*, which is not knowledge the allocator has. Chebyshev holds regardless, at the cost of being loose — a trade that is correct for a guard, where a false "safe" is a leak and a false "unsafe" is only wasted work.

`regulate_entropy` takes a tracer closure and returns the number of objects reclaimed.

### `governor` — PID control on the state manifold

267 lines. The adaptive threshold controller, and the place this project takes its own advice about the physical world needing a calibration knob.

$$
\varepsilon(t+1) \;=\; \varepsilon(t) \;+\; \alpha\, e(t) \;+\; \beta\, \frac{de}{dt}
$$

where the error is the difference between target and achieved effective tick rate,

$$
e(t) \;=\; R_{\text{target}} - R_{\text{actual}}, \qquad R_{\text{actual}} = \frac{\Delta}{\varepsilon}
$$

Shipped constants, all named and all tunable:

| Constant | Value | Role |
|---|---:|---|
| `TARGET_TICK_RATE` | 1000.0 Hz | The balance point between responsiveness and efficiency |
| `ALPHA` (α) | 0.01 | Proportional gain — response to instantaneous error |
| `BETA` (β) | 0.05 | Derivative gain — damps oscillation |
| `EPSILON_MIN` | 0.001 | Floor; prevents runaway sensitivity |
| `EPSILON_MAX` | 10.0 | Ceiling; prevents the system sleeping through events |
| `EPSILON_INITIAL` | 0.1 | Starting threshold |

```rust
impl GeometricGovernor {
    pub fn with_epsilon(epsilon: f64) -> Self;
    pub fn with_gains(alpha: f64, beta: f64) -> Self;
    pub fn adapt(&mut self, deviation_delta: f64, dt: f64) -> f64;
    pub fn should_trigger(&self, deviation: f64) -> bool;
}
```

There is **no integral term** — this is PD, not PID, despite the source comment saying PID. Defensible for this application (an integral term on a threshold that is already clamped invites windup), but the comment and the code disagree and the code is right.

The stability properties claimed in the doc comment — bounded ε, asymptotic stability around the target, damped oscillation — are **argued, not asserted by a test**. `with_gains` exists precisely because the shipped α and β are a starting point for a real system, not a derivation. A clock drifts, a sensor reads off, and a workload is not the one the gains were picked against.

### `aether` — hierarchical block pruning

448 lines. A three-level tree of geometric summaries supporting query-time pruning.

```rust
const MAX_DIM: usize = 64;
const MAX_BLOCKS: usize = 128;

pub struct BlockMetadata<const D: usize> { /* ... */ }
impl<const D: usize> BlockMetadata<D> {
    pub fn from_points(points: &[[f64; D]]) -> Self;
    pub fn upper_bound_score(&self, query: &[f64; D]) -> f64;
    pub fn can_prune(&self, query: &[f64; D], threshold: f64) -> bool;
}

pub struct HierarchicalBlockTree<const D: usize> { /* 3 levels */ }
impl<const D: usize> HierarchicalBlockTree<D> {
    pub fn build_from_blocks(&mut self, blocks: &[BlockMetadata<D>]);
    pub fn hierarchical_query(&self, query: &[f64; D], threshold: f64) -> [bool; MAX_BLOCKS];
    pub fn pruning_ratio(&self, active_mask: &[bool; MAX_BLOCKS]) -> f64;
}
```

| Level | Granularity | Fan-in |
|---:|---|---|
| 0 | 64-token blocks | finest |
| 1 | 256-token clusters | 4 blocks |
| 2 | 1024-token super-clusters | 16 blocks |

The mechanism is an **admissible upper bound**: `upper_bound_score` computes a value no lower than any true score inside the block, so `can_prune` discards a subtree only when even its optimistic score falls below the threshold. Pruning is therefore exact — it changes cost, never the answer. This is the same argument branch-and-bound rests on, and it is why the bound must be an over-estimate rather than an estimate.

`pruning_ratio` is the honesty function: it reports what fraction was actually skipped, so a "hierarchical" query that prunes nothing is visible as such rather than being assumed effective.

```rust
pub enum CompressionStrategy { /* ... */ }
pub fn select_compression<const D: usize>(meta: &BlockMetadata<D>) -> CompressionStrategy;
pub fn estimate_compression_ratio<const D: usize>(meta: &BlockMetadata<D>) -> f64;

pub struct DriftDetector<const D: usize> { /* ... */ }
impl<const D: usize> DriftDetector<D> {
    pub fn update(&mut self, centroid: &[f64; D]) -> f64;
    pub fn is_drifting(&self, threshold: f64) -> bool;
    pub fn velocity_magnitude(&self) -> f64;
}
```

`estimate_compression_ratio` is named `estimate` because it is one. No measured compression figures appear anywhere in this document.

### `manifold` — embedding and the streaming pipeline

608 lines.

```rust
pub struct ManifoldPoint<const D: usize> { /* coords */ }
pub struct TimeDelayEmbedder<const D: usize> { /* new(tau) */ }
pub struct SparseAttentionGraph<const D: usize> { /* new(epsilon) */ }
pub struct GeometricConcentrator<const D: usize> { /* ... */ }
pub struct TopologicalPipeline<const D: usize> { /* new(tau, epsilon) */ }
```

`TimeDelayEmbedder::embed` returns `Option<ManifoldPoint<D>>` — `None` until enough samples have arrived to fill the delay window. The type encodes the warm-up rather than returning a zero point that silently pollutes the first `(D-1)·τ` diagrams.

`SparseAttentionGraph` maintains an ε-neighbourhood graph incrementally with `adjacency: [u64; MAX_POINTS]` — a **bitset row per point**, capping the graph at 64 neighbours per node by construction. `compute_betti_0` is exact (connected components); `estimate_betti_1` is, as named, an **estimate** — a cycle-rank heuristic on the graph, not persistent homology. The naming distinction is deliberate and is the difference between this cheap path and the [full engine](#the-persistence-engine-in-detail).

`GeometricConcentrator` tracks which axis carries the most variance (`principal_dimension`) and projects to it (`concentrate_1d`), reporting `concentration_ratio` so a caller can tell whether the projection kept anything.

`TopologicalPipeline::push` is the whole streaming path in one call: push a scalar sample, get back `Option<(β₀, β₁, u64)>` once the embedder has warmed up. This is what makes the language's `manifold M = embed(data, tau=1)` a streaming construct rather than a batch one.

### `topology` — the byte-level module, and a naming hazard

356 lines. This module shares vocabulary with the persistence engine and **operates on completely different input**, which is the most confusing thing in the crate and is therefore stated first:

```rust
pub fn compute_betti_0(data: &[u8]) -> u32;
pub fn compute_betti_1(data: &[u8]) -> u32;
pub fn compute_shape(data: &[u8]) -> TopologicalShape;
```

**`&[u8]`.** Raw bytes, not point clouds. These functions compute a coarse topological signature of a *byte sequence* — the machinery behind the kernel's [`verify_binary_topology`](#loader--topological-code-authentication).

| | `topology::compute_betti_0` | `persistence::persistent_homology` |
|---|---|---|
| Input | `&[u8]` byte slice | `&[ManifoldPoint<D>]` |
| Output | `u32` | Full `PersistenceDiagram` |
| Method | Windowed density clustering | Exact 𝔽₂ column reduction |
| Exact? | Heuristic | Exact, invariant-tested |
| Cost | O(len) | See [complexity](#complexity-reference) |

Two functions named `compute_betti_0` exist in this crate — this one, and `SparseAttentionGraph::compute_betti_0`, which is an exact union-find over a point graph. Neither is the persistence engine. **None of the 11 persistence invariants apply to this module**, and a reader who assumed the phrase "Betti number" carried the same guarantees everywhere in the crate would be importing an assurance level that was never claimed.

Shipped constants:

```rust
const CLUSTER_THRESHOLD: i16  = 15;
const WINDOW_SIZE:       usize = 64;
const DENSITY_MIN:       f64   = 0.1;
const DENSITY_MAX:       f64   = 0.6;
const MAX_BETTI_1:       u32   = 10;
```

`MAX_BETTI_1 = 10` is a **saturating cap**, not a measurement — a byte sequence with more than ten detected cycles reports ten. Fine for a signature whose job is comparison against a reference; misleading if read as a homology computation.

```rust
pub struct TopologicalShape { /* betti_0, betti_1, data_len */ }
impl TopologicalShape { pub fn distance(&self, other: &Self) -> f64; }

pub enum VerifyResult { /* ... */ }
pub fn verify_shape(data: &[u8]) -> VerifyResult;
pub fn is_shape_valid(data: &[u8]) -> bool;
pub fn verify_against_reference(/* ... */);
pub fn verify_sliding_window(data: &[u8], window_size: usize) -> Result<(), usize>;
```

`verify_sliding_window` returns `Result<(), usize>` — on failure, the `usize` is **the offset where verification broke**. Returning the position rather than a bare `false` is the difference between a check that tells you something failed and one that tells you where to look.

### `state` — the vector everything else observes

186 lines, and the smallest module carrying real weight.

```rust
pub struct SystemState<const D: usize> { /* vector, timestamp */ }
impl<const D: usize> SystemState<D> {
    pub fn new(vector: [f64; D], timestamp: u64) -> Self;
    pub fn deviation(&self, other: &Self) -> f64;          // L2
    pub fn max_deviation(&self, other: &Self) -> f64;      // L-infinity
    pub fn manhattan_deviation(&self, other: &Self) -> f64; // L1
    pub fn magnitude(&self) -> f64;
    pub fn elapsed_since(&self, other: &Self) -> u64;
}
```

**Three metrics, offered rather than chosen.** L², L^∞ and L¹ each answer a different question about how far the system moved: total displacement, worst single component, and summed component change. The scheduler's trigger condition is Δ ≥ ε, and *which* Δ changes what wakes the machine — L^∞ fires when any one component spikes, L¹ fires on diffuse drift that L^∞ would sleep through.

Exposing all three rather than picking one is the right call for a substrate whose workload is not known at design time. It is also the [calibration knob](#governor--pid-control-on-the-state-manifold) argument again: a real system is tuned against its actual dynamics, and a module that hard-codes L² has made that decision on the operator's behalf with no evidence.

`timestamp` travelling inside the state, with `elapsed_since` as a peer of the deviation metrics, is what lets the governor compute `de/dt` without a separate clock source — relevant in a kernel where "what time is it" is not a free question.

---

## The Kernel In Detail

`crates/aether-kernel`. A `no_std` x86_64 microkernel that links `aether-core` and uses it to make scheduling decisions. It **compiles** for `x86_64-unknown-none`. It is **not asserted to boot**.

That distinction is maintained everywhere in this document because it is the distinction that gets blurred most often, and blurring it is how a project acquires a reputation. Compiling proves the code typechecks and links against a bare-metal target. Booting requires QEMU logs and a hardware matrix, neither of which exists here.

### `allocator` — a bump allocator on a static heap

```rust
const HEAP_SIZE: usize = 64 * 1024;
static mut HEAP: [u8; HEAP_SIZE] = [0; HEAP_SIZE];
static ALLOCATOR: Mutex<BumpAllocator> = Mutex::new(BumpAllocator::new());
#[global_allocator]
static GLOBAL_ALLOCATOR: AegisAllocator = AegisAllocator;
```

**64 KB, bump-allocated, and it does not free.** A bump allocator hands out sequential pointers and reclaims nothing until reset. This is the right allocator for a kernel that boots, sets up, and runs a fixed workload, and the wrong one for anything that allocates in a loop.

It is worth being explicit that this is a *different allocator* from the [manifold heap](#memory--the-manifold-heap) in `aether-core`. The manifold heap is the sophisticated one with generational handles and entropy-regulated collection; the kernel's global allocator is 64 KB of bump. They serve different layers and the naming does not make that obvious.

### `interrupts` — the state manifold lives in an IDT handler

```rust
static IDT: spin::Lazy<InterruptDescriptorTable> = /* ... */;
static CURRENT_STATE: Mutex<SystemState<STATE_DIMENSION>> = /* ... */;
static IRQ_COUNTER: spin::Mutex<u64> = spin::Mutex::new(0);
static TIMESTAMP: spin::Mutex<u64> = spin::Mutex::new(0);

pub fn get_current_state() -> SystemState<STATE_DIMENSION>;
pub fn update_state_component(index: usize, value: f64);
```

The system state vector that the governor and scheduler read is maintained **from interrupt context**, behind spin mutexes. This is the load-bearing piece of the whole "topology makes execution decisions" claim: the state manifold is not a userspace abstraction observing the kernel, it is updated by the interrupt handlers themselves.

It also requires `#![feature(abi_x86_interrupt)]`, which is why the toolchain is pinned to nightly independently of the `-Z build-std` requirement.

### `scheduler` — sparse, deviation-triggered

```rust
const DEFAULT_DT: f64 = 0.001;
const ENTROPY_MULTIPLIER: u64 = 6364136223846793005;
pub struct SparseScheduler<const D: usize> { /* ... */ }
```

The scheduler wakes on **deviation Δ ≥ ε** rather than on a timer tick, with ε supplied by the [geometric governor](#governor--pid-control-on-the-state-manifold). Between events it issues `WFI`. A timer-tick scheduler does fixed work per unit time whether or not anything happened; a deviation-triggered one does work proportional to how much the state actually moved, which is the entire power argument.

`ENTROPY_MULTIPLIER = 6364136223846793005` is the multiplier from Knuth's MMIX / PCG linear congruential generator. Naming it rather than leaving a magic constant matters, because a reader who does not recognise it cannot tell a well-chosen LCG multiplier from a number somebody typed.

**Four tests in this file never execute.** `aether-kernel` is a `no_std` binary with no test harness, so `cargo test` cannot reach them. They are listed as ⛔ Ungated in the [status dashboard](#honest-status-dashboard). Tests that cannot run are documentation with a misleading syntax highlight.

### `loader` — topological code authentication

```rust
const ELF_MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];
pub enum LoadError { /* ... */ }
pub struct ElfInfo { /* ... */ }
pub fn verify_elf(data: &[u8]) -> Result<ElfInfo, LoadError>;
pub fn verify_binary_topology(data: &[u8]) -> bool;
```

`verify_elf` is ordinary ELF header validation. `verify_binary_topology` is the unusual one: it computes a topological signature of the binary's byte distribution and checks it against an expected shape.

**This is not a security mechanism and must not be read as one.** It is a structural plausibility check — it would catch a truncated or corrupted image, not an adversary, because an adversary who knows the check exists can trivially pad a payload to match a byte-distribution signature. Cryptographic signing is what authenticates code. This is a cheap integrity heuristic with an interesting name, and calling it "authentication" in the source is an overclaim that this section is correcting rather than repeating.

### `boot` — BIOS handoff

```rust
pub struct BootInfo { /* ... */ }
pub struct MemoryRegion { /* ... */ }
pub enum MemoryRegionKind { /* ... */ }
pub struct Framebuffer { /* ... */ }
pub struct HardwareTopology { /* ... */ }
pub struct IoCaps { /* ... */ }
```

This is where the [audit found a real bug](#aether-kernel--the-bare-metal-bit), and it is worth recording precisely because it was only visible *after* the crate was made to compile again.

`BootInfo::config_root` returned a pointer to the eight-byte **RSDP signature string** where the ACPI root table address was intended. The RSDP begins with the literal bytes `"RSD PTR "`, so the function returned a pointer to that text rather than to the table it introduces. Any ACPI enumeration built on it would have parsed ASCII as a table header.

The defect had been latent for the entire life of the repository because **CI triggered on `main` and `develop`, and this repository's branch is `master`.** No run had ever executed. The crate had stopped compiling, so the bug beneath the compile error was unreachable. Fixing the workflow surfaced four compile defects — including `multiboot2::load` being removed in 0.24 — and this fifth, genuine one underneath them.

The general lesson is in [Why I Did This To Myself](#the-part-where-a-repository-audit-ruined-my-afternoon): **a green checkmark you have never seen is not a green checkmark.** The narrower one is that a compile error is not a bug's hiding place, it is its roof.

### `serial` — the only output channel

```rust
const COM1: u16 = 0x3F8;
static SERIAL: Mutex<SerialPort> = Mutex::new(SerialPort::new(COM1));
```

COM1 at the standard port. This is how a booting kernel would report anything at all, and it is the channel the missing QEMU logs would arrive on. The path from "compiles" to "asserted to boot" runs through this file.

---

## The Duplicate Crate Problem

The workspace contains six crates. Two of them should not exist.

| Crate | Status | Note |
|---|---|---|
| `aether-core` | real | The math foundation |
| `aether-lang` | real | Lexer, parser, AST, interpreter, Titan VM |
| `aether-kernel` | real | Bare-metal microkernel |
| `aether-cli` | real | `repl` / `run` / `check` |
| `aegis-core` | ⚠ duplicate | Queued for deletion |
| `aegis-cli` | ⚠ duplicate | Queued for deletion |

The project was renamed from AEGIS to AETHER **by copying directories rather than moving them**. The consequences are all still visible:

- `aegis-cli/src/main.rs` differs from `aether-cli/src/main.rs` only in string literals.
- `aegis-cli` does not reference `aegis-core` at all — the duplicate CLI is not even wired to the duplicate core.
- Half the doc comments across the tree still say AEGIS. `ml/tensor.rs` opens with *"AEGIS Tensor: N-dimensional array"*. The kernel's global allocator is `AegisAllocator`. The CLI banner prints `🛡️ AEGIS`.
- Repository examples use `.aegis` and `.ag` extensions; the CLI accepts `.aether` and `.ae` and prints a warning for the old ones.

This is documented here rather than quietly cleaned before publishing because **the state of the repository is part of what a reader is evaluating**, and a reviewer who discovers duplicate crates themselves reasonably wonders what else is unstated. It is the single most embarrassing thing in the tree and it is in the table of contents.

---

## Contributing

The claim ledger at [`docs/reference/status.md`](docs/reference/status.md) is the authority on what is gated, what is ungated, and which command produces each piece of evidence. Read it before opening anything.

Ranked by value, highest first — this ranking is the honest one, not the convenient one:

**1. The external parity harness.** Compare this engine against a pinned `ripser` or `gudhi` on shared fixtures. This is the largest correctness debt in the repository by a wide margin. Every correctness claim currently rests on internal invariants, and internal invariants are satisfiable by a self-consistently wrong implementation. Nothing else on this list is close.

**2. The controlled convergence experiment.** Does stopping on β-stability actually beat stopping on a tuned scalar residual, on real problems? This is the **core premise of the language** and it is [unmeasured](#the-faq-nobody-asked-for). A negative result here would be more valuable to this project than another green test suite.

**3. QEMU boot logs.** Move the kernel from "compiles" to "boots" with evidence.

**4. Gate the Lean tree.** 10,474 lines, 48 theorems, zero `sorry`, and no `lake build` in CI. Either gate it or cut it; the current state is neither.

**5. Delete `aegis-core` and `aegis-cli`.** Mechanical, uncontroversial, and removes the most embarrassing thing in the tree.

**6. Wire `Tensor::matmul` to `aether-gpu`, or decide not to.** The cost is measured (crossover n=128, 38x at n=512 with conversion) and the precision is measured (5e-7 relative, fine for training, not for 1e-9 assertions). What remains is the semantic decision about whether `ml::Tensor` may drop to f32.

**7. A gradcheck for `ml/autograd.rs`.** An autograd with no finite-difference verification is an unverified claim.

**8. Differential tests between the interpreter and `TitanVM`.** Two execution engines with no parity suite will diverge.

The house rule for any contribution that adds a number to this README: **it must come with the command that reproduces it.** A number without a reproduction command does not go in a table, and a benchmark measured on an implementation whose correctness is unestablished is not a result.

---

## The Test Suite, Test By Test

163 tests. Listing the ones that carry the correctness argument, because a count is not evidence and a reader deserves to see what is actually asserted.

### `persistence_invariants.rs` — 11 tests, 607 lines

The suite that makes the engine believable. Every entry is a **property over generated inputs**, not one hand-picked cloud with one expected number.

| Test | What it pins |
|---|---|
| `diagram_is_invariant_under_input_permutation` | Row order cannot change the diagram. 5 seeds, 1e-9 |
| `diagram_is_invariant_under_rotation_and_translation` | Isometry invariance — rotate 0.9128 rad, translate. Bottleneck < 1e-9 |
| `diagram_scales_linearly_with_the_point_cloud` | Scale equivariance for c ∈ {0.125, 0.5, 2, 37}. Exact |
| `a_scaled_radius_cap_selects_the_same_complex` | The cap must scale with the data, catching absolute-threshold bugs |
| `bottleneck_distance_respects_the_stability_bound` | **CSEH stability**, d_B ≤ 2ε, 12 seed×ε combinations |
| `circle_has_exactly_one_long_h1_bar_dying_at_sqrt3_times_the_radius` | The canonical positive control |
| `separated_clusters_produce_one_h0_bar_each_until_the_gap_closes` | β₀ tracks the actual component count |
| `gaussian_noise_produces_no_long_h1_bar` | **The negative control.** 4 seeds, zero long H₁ bars |
| `h0_matches_an_independent_union_find` | Cross-validation against separate code. 1e-9 |
| `a_single_point_has_one_essential_component_and_nothing_else` | Degenerate input |
| `duplicate_points_do_not_break_the_reduction` | Zero-distance pairs, the classic reduction crasher |

Two of these deserve emphasis. `gaussian_noise_produces_no_long_h1_bar` is worth as much as every positive case combined — a pipeline that finds structure in noise finds it everywhere, and every downstream claim built on it is unfalsifiable. And `h0_matches_an_independent_union_find` is why the engine [deliberately does not special-case H₀ to union-find](#design-decisions-that-seemed-good-at-3-am): doing so would make the check tautological, and the check is worth more than the speed.

Not listed above but present in the file: `boundary_of_boundary_is_zero_over_z2` and `every_face_is_present_and_precedes_its_coface`, asserted across five complexes. ∂∂ = 0 fails **silently** — every rank downstream becomes meaningless with no crash — which is the entire argument for testing an identity that is true by construction.

### `diagram_distance.rs` — 17 tests, 381 lines

| Group | Tests |
|---|---|
| Bottleneck as a metric | `bottleneck_of_a_diagram_with_itself_is_zero`, `bottleneck_is_symmetric`, `bottleneck_satisfies_the_triangle_inequality` |
| Bottleneck correctness | `bottleneck_matches_a_hand_computed_pairing`, `bottleneck_projects_an_unmatched_bar_to_the_diagonal` |
| Wasserstein | `wasserstein_sums_where_bottleneck_takes_a_maximum`, `wasserstein_is_at_least_bottleneck` |
| Cross-check | `distances_respect_the_stability_theorem_on_real_diagrams` |
| Landscapes | `a_single_bar_gives_the_expected_tent_function`, `landscape_levels_are_ordered_pointwise`, `landscape_takes_the_kth_largest_tent_where_bars_cross`, `landscape_is_one_lipschitz_in_the_bottleneck_distance`, `an_empty_diagram_gives_a_zero_landscape` |
| Images | `persistence_image_has_the_requested_shape_and_is_nonnegative`, `persistence_image_weights_long_bars_more_than_short_ones`, `sigma_controls_the_kernel_width`, `persistence_image_is_translation_equivariant_in_birth` |

`wasserstein_sums_where_bottleneck_takes_a_maximum` and `landscape_takes_the_kth_largest_tent_where_bars_cross` both exist because a mutant survived. See [Mutation Testing](#mutation-testing-or-how-i-learned-to-stop-trusting-green-checkmarks) — the original level-ordering test used *nested* bars, whose tent values already arrive sorted, so an implementation skipping the sort passed. Crossing bars kill that mutant. `sigma_controls_the_kernel_width` exists because **no test referenced σ at all**, so every image property held for any fixed kernel width whatsoever.

### `attention_contracts.rs` — 29 tests, 1,124 lines

The largest file in the repository, and the shape of it tells the story: nine tests establish contracts every selector must satisfy, and the remaining twenty are the record of a claim being repaired three times.

**Contracts — must hold for every selector:**

`a_full_mask_reproduces_dense_attention_exactly` · `attention_output_is_a_convex_combination_of_values` · `a_uniform_query_averages_the_values` · `a_masked_key_contributes_exactly_nothing` · `the_realized_pattern_equals_the_requested_pattern` · `a_budgeted_selector_respects_its_budget` · `a_single_position_attends_to_itself` · `shapes_around_the_block_boundary_are_handled` · `the_topological_selector_is_scale_equivariant`

**Causality — the class of bug that silently destroys a language model:**

`no_output_position_depends_on_a_later_position` · `causal_selectors_never_select_a_future_key`

Two tests rather than one because they check different things: the first perturbs a future position and asserts the output does not move (behavioural), the second inspects the mask directly (structural). A kernel can pass one and fail the other.

**Numerical safety:**

`an_all_masked_row_returns_zeros_rather_than_nan` · `large_logits_do_not_overflow_the_softmax` · `repeated_runs_are_bitwise_identical`

The all-masked row is the sharp edge: softmax over an empty set is 0/0. Returning zeros is a decision, and a decision that is not tested is a decision that gets refactored into a NaN that surfaces four layers downstream as a loss of `nan` with no stack trace.

**Measurement instruments:**

`oracle_top_k_upper_bounds_every_other_selector_at_the_same_budget` · `the_oracle_is_priced_as_the_diagnostic_it_is` · `the_topological_selector_is_placed_on_the_random_to_oracle_axis`

**The negative results, pinned so they cannot drift back:**

`the_topological_advantage_collapses_when_key_norms_vary` · `routing_is_sparse_only_when_the_keys_have_h0_structure` · `routing_buys_its_quality_at_a_real_discount_on_structured_keys` · `the_routed_selector_survives_the_key_norm_spread_that_broke_the_old_one`

**The routing decision:**

`the_plan_predicts_the_cost_it_will_actually_incur` · `the_plan_declines_to_route_exactly_when_routing_would_not_pay` · `the_h0_barcode_alone_separates_the_two_regimes` · `the_adaptive_selector_never_costs_more_than_dense` · `the_adaptive_selector_keeps_the_quality_of_whichever_path_it_picks`

**Cross-validation against the persistence engine:**

`single_linkage_merge_heights_equal_the_h0_persistence_deaths` · `routing_clusters_are_invariant_to_per_key_rescaling` · `the_routed_selector_obeys_every_contract_the_others_do`

That last one is the guard against a common failure: a new selector added to an enum, tested only on the new behaviour, and never re-run against the contracts the old ones satisfy.

### `scheduled_attention.rs` — 16 tests, 576 lines

| Group | Tests |
|---|---|
| Schedule structure | `the_dense_causal_schedule_is_lower_triangular_csr`, `a_csr_schedule_is_well_formed_at_every_size`, `the_topology_schedule_contains_sink_local_and_salient_blocks`, `the_topology_schedule_visits_fewer_blocks_than_the_dense_one` |
| Salience | `block_salience_is_the_elder_rule_over_centroids`, `the_salience_multiset_is_invariant_to_block_order`, `the_schedule_depends_on_block_order` |
| Numeric parity | `a_dense_schedule_reproduces_full_causal_attention`, `a_sparse_schedule_matches_its_own_dense_masked_reference`, `scheduling_a_block_actually_changes_what_the_row_sees` |
| Safety | `the_kernel_is_deterministic`, `large_logits_stay_finite_across_scheduled_blocks`, `an_empty_row_is_rejected_rather_than_producing_nan` |
| Working set | `the_working_set_does_not_grow_with_the_sequence` |
| Input validation | `the_builder_rejects_malformed_inputs`, `the_kernel_rejects_a_schedule_that_does_not_match_the_sequence` |

`the_salience_multiset_is_invariant_to_block_order` sitting next to `the_schedule_depends_on_block_order` is not a contradiction — it is a caveat that was discovered by a test failing and is now pinned in both directions. The *multiset* of salience scores is invariant, being the H₀ barcode. The per-block *assignment* is not, because under component-size ties the union-find absorbs by index order. The Triton original shares the tie-breaking. [The test was wrong, not the code.](#bonus-my-own-tests-were-wrong-twice)

`scheduling_a_block_actually_changes_what_the_row_sees` is the anti-tautology test: without it, a kernel that ignored the schedule entirely and computed dense attention would pass every numeric-parity check in the file.

### `persistence_scale.rs` — 7 tests, 206 lines, `--release`

`h0_handles_five_hundred_points` · `h1_handles_one_hundred_points` · `circle_h1_dies_at_the_exact_regular_polygon_chord` · `circle_h1_death_converges_to_sqrt3_from_above` · `dense_sampling_does_not_manufacture_extra_loops` · `simplex_cap_still_fails_fast_rather_than_exhausting_memory` · `point_cap_is_still_enforced_when_configured`

This is the file that went from 29.07 s to 1.10 s across the `BTreeMap` refactor on identical assertions. `dense_sampling_does_not_manufacture_extra_loops` is a second negative control: sampling a circle more densely must not invent H₁ classes that are not there.

`circle_h1_death_converges_to_sqrt3_from_above` is the test that [taught its own author a sharper theorem](#bonus-my-own-tests-were-wrong-twice) by failing at n=24.

### What the suite does not cover

Stated plainly, because a test inventory that lists only what exists is an advertisement:

- **No external parity.** Nothing here compares against ripser, GUDHI, giotto-tda or Dionysus. Every test above is internal consistency, and internal consistency is satisfiable by a self-consistently wrong implementation.
- **No gradcheck**, anywhere. `ml/autograd.rs` has no finite-difference verification, and `attention` has no backward pass to check.
- **No `ml/` property tests.** The clustering, classification and neural modules have far thinner coverage than the topology core. K-means is not tested for initialisation sensitivity, and the two independent single-linkage implementations are not cross-checked against each other.
- **Four kernel scheduler tests never execute.** They exist in `aether-kernel/src/scheduler.rs`, but the crate is a `no_std` binary with no test harness.
- **The Lean tree is not built.** 48 theorems, no `lake build` in CI.
- **No fuzzing, no `proptest`, no Miri.** The property tests are hand-rolled loops over seeds, not a shrinking generator.

---

## Results

All measurements: single machine, Windows 11, Rust nightly, single core, release where noted. Single run, **no confidence intervals** — engineering measurements, not a study.

### The 26x

`find_simplex` linear-scanned `simplices[..before]` for every face of every simplex, making the reduction O(m²) in the simplex count. Now a `BTreeMap` keyed on the zero-padded vertex array.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  cargo test -p aether-core --test persistence_scale --release
  Identical assertions either side of commit 27d70fa, same machine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Before (linear face scan)              29.07 s
  After  (BTreeMap index)                 1.10 s      ← 26x reduction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Invariant tests green across the refactor       11 / 11
  Point cap lifted                            32 → 512 (h0_only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The invariant suite existing *before* the refactor is the entire reason it was safe. That is what property tests buy: not bug-finding, permission to change things.

### Measured scale ceiling

`cargo run -p aether-core --example scale_probe --release`, regular circle, single core.

| dim | n | pairs | seconds |
| ---: | ---: | ---: | ---: |
| 0 | 200 | 200 | 0.049 |
| 0 | 1,000 | 1,000 | 5.781 |
| 0 | 4,000 | 4,000 | 335.049 |
| 1 | 60 | 1,771 | 0.117 |
| 1 | 120 | 7,141 | 2.202 |
| 1 | 200 | 19,901 | 20.728 |
| 1 | 300 | 44,851 | 131.343 |
| 2 | 30 | 4,090 | 0.100 |
| 2 | 50 | 19,650 | 1.859 |
| 2 | 70 | 54,810 | 15.338 |

Presets are sized to those timings, not guessed: `h2_default` 48 points, `h1_dense` 128, `h0_only` 512. **These caps are a time budget, not a correctness limit.** Raise them explicitly and wait longer.

For context: ripser routinely handles clouds of tens of thousands of points. This engine does not. The trade purchased is `no_std` with one dependency, and the ability to run inside a kernel.

### Exactness against closed form

| Property | Assertion | Result |
|---|---|---|
| Circle H₁ death | $2r\sin(\pi\lceil n/3\rceil/n)$, 8 values of $n$ | exact to **1e-12** |
| H₀ deaths | equal an independent union-find MST | exact to **1e-9** |
| Permutation invariance | shuffled rows give identical diagram | **1e-9**, 5 seeds |
| Isometry invariance | rotate 0.9128 rad + translate | bottleneck **< 1e-9** |
| Scale equivariance | $c \in \{0.125, 0.5, 2, 37\}$ | exact |
| Stability | $\lVert\delta\rVert \le \varepsilon \Rightarrow d_B \le 2\varepsilon$ | 12 seed×ε cases pass |
| Negative control | Gaussian blob | **0** long H₁ bars, 4 seeds |
| $\partial\circ\partial = 0$ | every simplex, 5 complexes | exact |

The Gaussian blob is worth as much as any positive case. A pipeline that finds structure in noise finds it everywhere, and every downstream claim built on it is unfalsifiable.

### Scheduled attention

A Rust port of the topology-derived sparse attention kernel merged as [`triton-lang/kernels#22`](https://github.com/triton-lang/kernels/pull/22). The Python original requires CUDA; this compiles wherever `aether-core` does.

The port keeps the original's split, which is what makes each half checkable alone:

| Half | Nature | Checked against |
|---|---|---|
| CSR block schedule | combinatorial | **exact** equality with the Python builder — `[0,1,3,6,10]` / `[0,0,1,0,1,2,0,1,2,3]` for 4 blocks |
| Kernel | numeric | dense masked attention, **1e-12**, 4 schedule configs × 4 block geometries |

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Block reduction, 16 blocks, local_radius=1 sink=1 topk=2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Scheduled blocks                     56 / 136          58.8% cut
  Kernel vs dense masked reference             max |Δ| < 1e-12
  Salience vs persistence engine H0 deaths        agrees, 1e-9
  Blocks scoring zero                                 exactly 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The upstream PR measured 56.6% at seq 1024 and 80.9% at seq 4096 on an RTX 4060, with 1.04×–3.48× sparse-vs-dense-CSR wall-clock. **Those timings are not reproduced here and are not claimed here.** This port is a scalar CPU kernel with no SIMD, no threading, no GPU. It reproduces the *answer* and the *block reduction*, not the *speed*.

---

## Mutation Testing, Or How I Learned To Stop Trusting Green Checkmarks

A passing suite tells you nothing about what it would catch. So: inject known defects, one at a time, and count.

**Persistence invariants** — 3 defects:

| Injected defect | New suite | Prior suite |
|---|---|---|
| Triangle filtration drops one of three edges | **4 / 11** | 0 / 6 |
| Hardcoded `+0.001` absolute epsilon in the filtration | **4 / 11** | 0 / 6 |
| Column reduction terminates after one operation | **7 / 11** | 1 / 6 |

The prior six example tests missed two of three defects **entirely**. That is the difference between "one hand-picked cloud with one expected number" and "a property that holds for every input".

**Diagram metrics** — 5 defects, and this is the part I did not enjoy:

| Injected defect | Caught by |
|---|---|
| Landscape skips the per-sample descending sort | 2 / 17 |
| Image drops the linear persistence weight | 1 / 17 |
| Wasserstein returns the max instead of the sum | 1 / 17 |
| Image hardcodes the Gaussian width, ignoring σ | 1 / 17 |
| Bottleneck forbids diagonal projection | not run — infinite costs diverge the matching search |

**Two of those five survived the first version of the suite.**

The level-ordering test used *nested* bars, whose tent values already arrive in descending order, so an implementation skipping the sort passed anyway. And **no test referenced σ at all**, so every image property — shape, non-negativity, weighting, translation equivariance — held for any fixed kernel width whatsoever.

Both were rewritten with crossing bars and a concentration measurement until the mutants died. The lesson generalises unpleasantly:

> Test fixtures chosen for convenience rather than for discrimination produce suites that pass and prove nothing. A suite you have not mutated is a suite of unknown strength.

---

## What We Got Wrong

First-class section, near the top, with the numbers that killed each claim. This is the part of a README I would want to read first about somebody else's project.

### 1. "Current World's Fastest Agentic AI Language"

That string was the `description` field in `pyproject.toml`, **published to PyPI**. No benchmark in this repository supports it. It sat three files from a README section titled *Evidence Policy* which explicitly forbids unverified speedup claims. Removed.

### 2. Euclidean proximity as an attention-mass proxy — **negative**

The claim: keys geometrically near a query carry most of the attention mass, so you can select them without computing scores.

Measured against random and oracle top-k at the same budget, seq 32, head_dim 8, budget 6, 8 trials per row. Placement = (selector − random) / (oracle − random):

| key-norm spread | random | nearest-neighbour | oracle | placement |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.4902 | 0.5667 | 0.5769 | **+0.884** |
| 1.0 | 0.4899 | 0.5613 | 0.6242 | +0.533 |
| 2.0 | 0.4892 | 0.5265 | 0.6723 | +0.202 |
| 4.0 | 0.4873 | 0.4577 | 0.7616 | **−0.109** |
| 8.0 | 0.4848 | 0.3725 | 0.8841 | **−0.285** |

Past spread 4 the selector is **worse than picking keys uniformly at random**.

The reason is a one-line identity: $\lVert q - k \rVert^2 = \lVert q \rVert^2 + \lVert k \rVert^2 - 2 q \cdot k$. When key norms are approximately equal, ranking by distance *is* ranking by dot product, so the +0.884 at spread 0 is close to tautological. Vary the norms and the rankings decouple — a large-norm key becomes simultaneously far away and high-scoring.

The +0.884 is **not reported as a result**. `the_topological_advantage_collapses_when_key_norms_vary` pins both ends so it cannot quietly drift back.

### 3. The first fix was also wrong — **measurement artifact**

The first repair used an **absolute** radius of 0.6 against a median query-key distance of 2.395. It selected **1.00 keys per row** where its same-budget baselines selected **5.53**, and posted placements of **−3.573 to −4.177**.

It did not lose on mechanism. It lost on *budget*, by declining to select. Same bug class as the hardcoded epsilon the persistence scale-equivariance test exists to catch: an absolute threshold pretending to be relative. The radius is now relative to the row's median distance, and the ablation asserts equal mean budget before comparing anything.

### 4. Topological routing on unstructured keys — **not sparse at all**

The repaired selector clusters key *directions* and ranks by exact dot product. Placement held flat at +0.87 across the whole spread curve. Clean win, apparently.

Then the cost was measured:

| key distribution | H₀ component sizes | cost vs dense | placement |
| --- | --- | ---: | ---: |
| uniform random | `[61, 1, 1, 1]` | **0.999** | +0.942 |
| 4 real clusters | `[16, 16, 16, 16]` | **0.449** | +0.990 |
| 8 real clusters | `[8, 8, 8, 8]` | 0.528 | +0.995 |
| 16 real clusters | `[4, 4, 4, 4]` | 0.733 | +0.989 |

On uniform keys it examines **0.999× the dense dot-product count**. It is dense attention with clustering overhead, and its +0.94 placement was bought by looking at every single key.

Single-linkage **chains** on a cloud with no density gaps — 61 of 64 keys in one component. That is not a clustering bug. It is H₀ correctly reporting that uniform data has no structure to route on. The persistence diagram was right; the claim was wrong.

**Every test in the suite measured how *good* a selection was. None measured what it *cost*.** `selection_dot_cost` and `dense_dot_cost` exist because of this, and the honest claim is now conditional:

> Topological routing is a real sparsity win exactly when the key distribution has H₀ structure, and no win at all when it does not.

`routing_plan` makes that a runtime check rather than an assumption. Its `gap_ratio`, derived from the H₀ barcode alone, separates the regimes without overlap: **structured minimum 2.70 against chained maximum 1.04**, 6 trials each.

### 5. The cheap fallback — **indistinguishable from random**

When routing does not pay, `Selector::Adaptive` must do something else. The first fallback was a budget-6 sliding window. It measured placement **+0.014** on unstructured keys — random, to three decimals.

Not a fallback bug. When the keys have no structure there is **no cheap-and-good option**, because finding the top-k without computing the scores is precisely what the structure was supposed to make possible. The fallback is now dense, and the guarantee is *never worse than dense, in cost or in quality*.

### 6. `nalgebra`: a second phantom dependency

Found while rewriting this README, by the unglamorous method of reading `Cargo.toml` instead of trusting the sentence next to it.

`aether-core` — the crate this document repeatedly holds up as having a minimal dependency surface — declares:

```toml
nalgebra = { version = "0.32", default-features = false, features = ["libm"] }
```

**Zero call sites.** `grep -r nalgebra crates/ --include=*.rs` returns only the `Cargo.toml` lines and three stale `cargo tree` dumps checked into `aether-kernel/`. Not one `.rs` file references it.

Until this rewrite, the README asserted in four separate places that the core had *"exactly one dependency, `libm`"* and that `libm` plus `heapless` was *"the whole mathematical surface"*. All four were false, and all four are now corrected.

This is the same defect class as [the `wgpu` finding](#the-faq-nobody-asked-for), with two aggravating differences. `wgpu` at least had a comment gesturing at intent (*"future hooks for wgpu"*); `nalgebra` has nothing. And `wgpu` sits in `aether-lang`, whereas this one sits in the crate whose minimality is a **headline claim** in the prior-art table.

Three things are worth extracting from it:

**The `no_std` claim survives; the minimality claim does not.** `default-features = false` with the `libm` feature is why `thumbv7m-none-eabi` still builds. The Cortex-M3 result stands. The "one dependency" line was decoration on top of a real result, and it was the decoration that was false.

**The duplicate crate was better configured than the real one.** `aegis-core` declares the same dependency as `optional = true`. The [crate slated for deletion](#the-duplicate-crate-problem) got this right and the maintained one did not, which is what happens when a rename is done by copying and then only one copy receives attention.

**A dependency count is a claim, and claims need commands.** Every performance number in this document carries a reproduction command. The dependency count did not, because it read like a fact about the file rather than a measurement — and that is precisely the category of statement that rots, since the file changes and the sentence does not. It now carries one, in [Reproducing Every Number](#reproducing-every-number-in-this-document).

Queued for removal alongside `wgpu`, `pollster` and `bytemuck`.

### Bonus: a number that lied

With the dense fallback, the unstructured placement printed **+7.614**. Meaningless — dense recovers all the attention mass, which sits *above* the budget-limited oracle, so the ratio blows up. Left in a table it would read as a 700% win over an oracle it never competed with. The two regimes are now scored on different scales deliberately.

### Bonus: my own tests were wrong, twice

- A scale test asserted the circle H₁ death *converges* to $\sqrt{3}r$ monotonically. It failed at n=24 because the engine returns $\sqrt{3}r$ **exactly** from n=12 onward whenever $3 \mid n$. Trying to write the test found the sharper theorem.
- A port test asserted per-block salience is permutation-equivariant. It failed at block 2 (`0` vs `1.724`). Under component-size ties the union-find's absorb-smaller choice depends on index order. The *multiset* is invariant, being the H₀ barcode. The Triton original shares the tie-breaking. The test was wrong, not the code — and the caveat is now pinned.

---

## Design Decisions That Seemed Good At 3 AM

**The tilde.** `~` terminates statements. Not `;`, because `;` is what everybody else uses and I wanted the source to *look* different at a glance. Not a technical argument. An aesthetic one, and I stand by it while acknowledging it costs every new reader four seconds.

**The seal emoji.** `🦭 until convergence(1e-6)`. A four-byte codepoint as a control-flow keyword. The lexer handles it fine. Your `grep` may not. `seal until` exists for that reason.

**Two crate families.** The workspace contains `aether-core`/`aether-cli` **and** `aegis-core`/`aegis-cli`, because the project was renamed by *copying* rather than moving. `aegis-cli/src/main.rs` differs from `aether-cli/src/main.rs` only in string literals, and `aegis-cli` does not reference `aegis-core` at all. Documented rather than hidden; deletion queued. The single most embarrassing thing in the tree.

**Not special-casing H₀ to union-find.** Union-find would make H₀ near-linear instead of the 335 s that n=4000 takes. Deliberately not done: the invariant suite tests the engine's H₀ *against* an independent union-find, and making the engine use union-find turns that test tautological. Speed is not worth deleting the check.

**Denying only `correctness` and `suspicious` in clippy.** The workspace carries 16 style-class warnings. A `-D warnings` gate that fails on first contact gets switched off within a week, and then the correctness lints stop being enforced too. Strict where it finds bugs, quiet where it finds taste.

**Keeping a guard that provably never fires.** `boundary_indices` checks `idx < simplex_idx` even though `every_face_is_present_and_precedes_its_coface` proves it always holds. Free, and a malformed complex degrades instead of lying.

**10,474 lines of Lean.** `Aether/` contains a Lean 4 formalization with 48 theorems and zero `sorry`. It is also **not built by CI**, and 8,281 of those lines (`Lexer`, `Parser`, `Pipeline`, `Static`, `VM`) hold **exactly one theorem between them** — a second implementation of the language in Lean, not proofs about the first. Either gate it or cut it. It is in the ledger as ungated, which is the honest interim state.

---

## The FAQ Nobody Asked For

**Is this production-ready?**
No. It is a research language with a seal emoji as a keyword and a persistence engine that takes 335 seconds to do H₀ on 4,000 points. Use ripser.

**Why is it called both AETHER and AEGIS?**
Because it was renamed and the rename was done by copying directories. Half the doc comments say AEGIS, the examples use `.aegis`, and two duplicate crates are still in the workspace. Being fixed. It is in writing above precisely so nobody has to discover it.

**Do the Betti numbers actually help convergence, or is this elaborate numerology?**
Honest answer: **unmeasured.** The seal loop works, the topology is computed correctly, and the machinery is tested. Whether stopping on β-stability beats stopping on a scalar residual, on real problems, against a properly tuned baseline, is a controlled experiment that **has not been run**. It should be. It is the most important missing number in this repository and no table above pretends otherwise.

**Is the persistent homology correct?**
It satisfies 11 invariants including the stability theorem, reproduces a closed-form ground truth to 1e-12, agrees with an independent union-find on H₀, and survives mutation testing. It has **never been compared against ripser or GUDHI**. Those are different levels of assurance and I am not going to blur them.

**Why `no_std`?**
So the same persistence code that runs in the CLI runs in `aether-kernel` on bare metal, where it informs scheduling. Also because a two-dependency mathematical surface is a nice place to be — though the crate currently declares a third it never calls, which is [logged as a defect](#6-nalgebra-a-second-phantom-dependency).

**Does the kernel boot?**
It **compiles** for `x86_64-unknown-none`. Booting is not tested. Different claims.

**Is there GPU acceleration?**
There is a GPU **backend** — `aether-gpu`, 13 WGSL kernels, resident tensors, 60 tests, an RTX 4060 over Vulkan. There is no GPU **acceleration of this project**, because nothing in `aether-core` or `aether-lang` calls it. Both integrations are measured and neither is made: `Tensor::matmul` pays above n=128 and reaches 38× at n=512 with conversion counted; `pairwise_sqdist` never pays, because the persistence reduction is CPU-side and sequential so the matrix has to come back. Wiring the first one in means deciding whether `ml::Tensor` may drop to f32, which is a semantic change no benchmark authorises.

This answer used to read "No", and the `wgpu`/`pollster`/`bytemuck` entries it referred to — in `aether-lang`'s default feature set with zero call sites — have since been deleted. The current backend is a separate crate on a current wgpu.

**Why is there a `CHANGELOG.md` with almost nothing in it?**
Fair.

**How much of this README is the author admitting to mistakes?**
Roughly a third by line count. Intentional. A reader who finds an unstated limitation themselves discounts every other claim in the document.

**Can I contribute?**
Yes. Read [`docs/reference/status.md`](docs/reference/status.md) first — it lists exactly what is gated, what is ungated, and what command produces each piece of evidence. The highest-value contribution by a wide margin is the external parity harness against a pinned ripser. See [Contributing](#contributing) for the ranked list.

**Is `convergence(1e-6)` actually topological?**
No. It parses to `ConvergenceCond::Epsilon`, a scalar tolerance. The topological variant is `BettiStable { epochs }`. Both exist in the grammar and both are implemented, but the headline example in the pitch uses the scalar one, and [the language section says so](#the-convergence-condition-is-a-three-way-choice) rather than letting the ambiguity ride.

**Does topological convergence eliminate hyperparameter tuning?**
No, and claiming so would be the easiest overclaim in this project to make. It replaces a **continuous** threshold ε with a **discrete** stability window — how many iterations the Betti vector must hold. That is still a hyperparameter. The argument for it is that an integer count is easier to reason about and does not interact with the scale of your loss, not that it disappeared.

**Why is `estimate_betti_1` an estimate when there is a full persistence engine right there?**
Because they serve different layers. `SparseAttentionGraph::estimate_betti_1` is a cycle-rank heuristic on an incrementally-maintained ε-neighbourhood graph — cheap enough to run per sample in a streaming pipeline. The [full engine](#the-persistence-engine-in-detail) computes exact persistent homology and is far too slow for that path. The naming distinction is deliberate: one is exact and says so, one is an estimate and says so.

**There are two single-linkage implementations in this crate. Is that a bug?**
It is duplication, not a bug. `attention::single_linkage_clusters` and `ml::clustering::AgglomerativeClustering` implement overlapping mathematics for different callers. The attention one is [cross-checked against the persistence engine](#single_linkage_clusters-and-the-theorem-it-is-checked-against); the `ml` one is not cross-checked against anything. They are also not checked against **each other**, which is the obvious missing test and is listed in [what the suite does not cover](#what-the-suite-does-not-cover).

**Why is the kernel's allocator 64 KB of bump when `aether-core` has a sophisticated heap?**
Different layers, and the naming does not make that clear. `AegisAllocator` is the kernel's `#[global_allocator]`: 64 KB, bump, never frees — correct for a kernel that boots and runs a fixed workload. The [manifold heap](#memory--the-manifold-heap) is a data structure inside `aether-core` with generational handles and entropy-regulated collection. They are unrelated despite sitting in the same repository.

**Is `verify_binary_topology` a security feature?**
No. It is a structural plausibility check that would catch a truncated or corrupted image. An adversary who knows it exists can pad a payload to match a byte-distribution signature trivially. Cryptographic signing authenticates code; this does not. The word "authentication" in the source is an overclaim and [the kernel section corrects it](#loader--topological-code-authentication).

**The governor's doc comment says PID. Is it?**
It is **PD** — proportional and derivative, no integral term. Defensible for a threshold that is already clamped to `[EPSILON_MIN, EPSILON_MAX]`, since an integral term on a saturating actuator invites windup. But the comment and the code disagree, and the code is right.

**Why `-A clippy::style` when the project is this fussy about correctness?**
Because a gate that fails on first contact gets switched off within a week, and when it goes, the **correctness** lints go with it. The workspace carries 16 style-class warnings. Strict where lints find bugs, quiet where they find taste. [Design decisions](#design-decisions-that-seemed-good-at-3-am) has the longer version.

**Why not make H₀ use union-find? It would be enormously faster.**
Because `h0_matches_an_independent_union_find` compares the engine's H₀ **against** a union-find implementation. Making the engine use union-find turns the test tautological. H₀ at n=4,000 takes 335 s and that is the price of keeping a real cross-check. Speed is not worth deleting the check that says the answer is right.

**Is the Lean formalization proving things about the Rust?**
No, and this is the sharpest thing to be clear about. `Aether/` is 10,474 lines with 48 theorems and zero `sorry`. But 47 of those 48 live in `Core.lean` alone, and the **8,281 lines** of `Lexer`, `Parser`, `Pipeline`, `Static` and `VM` hold **exactly one theorem between them**. That is a second implementation of the language written in Lean, not proofs about the first. There is also no `lake build` in CI, so none of it is verified to still compile. Either gate it or cut it.

**How fast is this compared to ripser?**
Unmeasured, and [deliberately absent from the prior-art table](#prior-art). The honest inference from the measured ceiling — H₁ at n=300 in 131 s — is that it is **substantially slower**, since ripser routinely handles clouds orders of magnitude larger. That is an inference, not a benchmark, and it is phrased as one.

**What would change your mind about this whole approach?**
A controlled experiment showing β-stability stopping does no better than a tuned scalar residual on real problems. That experiment [has not been run](#contributing), it is the second-highest-value contribution on the list, and a negative result would be worth more to this project than another green suite.

---

## Repository Layout

```
Aether-Lang/
├── crates/
│   ├── aether-core/          math foundation, no_std, libm + heapless
│   │   ├── src/
│   │   │   ├── persistence.rs    F2 reduction, Rips + witness, H0-H2
│   │   │   ├── diagram.rs        bottleneck, Wasserstein, landscapes, images
│   │   │   ├── attention.rs      sparse attention reference + selectors
│   │   │   ├── scheduled.rs      CSR block-scheduled attention (Triton port)
│   │   │   ├── manifold.rs       time-delay embedding, sparse graphs
│   │   │   ├── topology.rs       Betti numbers, shape verification
│   │   │   ├── governor.rs       PID-on-manifold
│   │   │   ├── memory.rs         manifold heap, entropy GC
│   │   │   └── ml/               regression, clustering, neural, tensors
│   │   ├── tests/            most of the topology tests live here
│   │   └── examples/         scale_probe, routing_cost — reproduce the tables
│   ├── aether-gpu/           wgpu compute backend — 13 WGSL kernels, f32
│   │   ├── src/shaders.wgsl      matmul, tiled matmul, pairwise distance,
│   │   │                         softmax, fused gradients, Adam, SGD
│   │   ├── tests/                60 tests: parity, gradcheck, f32 topology
│   │   ├── mutants.sh            mutation harness, 0 of 10 escape
│   │   └── FEATURES.md           measurements, negative results, corrections
│   ├── aether-lang/          lexer, parser, AST, interpreter, Titan VM
│   ├── aether-kernel/        no_std x86_64 microkernel
│   ├── aether-cli/           repl / run / check
│   ├── aegis-core/           ⚠ duplicate, queued for deletion
│   └── aegis-cli/            ⚠ duplicate, queued for deletion
├── Aether/                   Lean 4 formalization (ungated, see above)
├── docs/                     MkDocs site + the claim ledger
├── examples/                 .aegis / .ag scripts (extension debt)
└── .github/workflows/ci.yml  the gate that had never run until recently
```

### `aether-core` by module, with line counts

The crate the rest of the workspace is built on. 24 files, and the distribution is informative — the topology core and the ML subtree are comparable in size, which the [old single-row table](#implementation) did not convey.

```
crates/aether-core/src/                                    lines
├── lib.rs                                                    51   module exports, feature gates
├── persistence.rs                                           783   F2 reduction, Rips + witness, H0-H2
├── attention.rs                                             699   7 selectors, routing plan, cost model
├── manifold.rs                                              608   Takens embedding, sparse graph, pipeline
├── memory.rs                                                480   manifold heap, Chebyshev guard
├── scheduled.rs                                             477   CSR block schedule (Triton port)
├── aether.rs                                                448   hierarchical block tree, drift
├── diagram.rs                                               417   bottleneck, Wasserstein, landscapes, images
├── topology.rs                                              356   byte-level shape verification  ⚠ not the engine
├── governor.rs                                              267   PD control on the state manifold
├── state.rs                                                 186   SystemState, three deviation metrics
└── ml/                                                    3,992
    ├── classification.rs                                    672   6 classifiers
    ├── clustering.rs                                        558   KMeans, DBSCAN, agglomerative
    ├── neural.rs                                            489   activations, optimizers, dense layers
    ├── convergence.rs                                       339   BettiNumbers, detector, residuals  ← the premise
    ├── regressor.rs                                         328   model escalation
    ├── autograd.rs                                          302   reverse mode  ⚠ no gradcheck
    ├── benchmark.rs                                         255   escalation harness  ⚠ not a benchmark
    ├── linalg.rs                                            254   reductions, distances
    ├── tensor.rs                                            254   N-d array
    ├── gossip.rs                                            203   distributed averaging
    ├── convolution.rs                                       160   Conv2D  ⚠ forward only
    ├── dataloader.rs                                        127   batching, shuffle
    └── mod.rs                                                51
```

```
crates/aether-core/tests/                                  lines
├── attention_contracts.rs                                 1,124   29 tests
├── persistence_invariants.rs                                607   11 tests
├── scheduled_attention.rs                                   576   16 tests
├── diagram_distance.rs                                      381   17 tests
└── persistence_scale.rs                                     206    7 tests  (--release)
                                                          ─────
                                                            2,894
```

**2,894 lines of tests against 7,472 lines of non-`ml` source.** The topology core is well covered; the [4,000-line `ml` subtree has no dedicated test file at all](#what-the-suite-does-not-cover), and that asymmetry is the honest shape of this repository's assurance.

### `aether-lang` and `aether-kernel`

```
crates/aether-lang/src/                                    lines
├── interpreter.rs                                         1,803   the reference implementation
├── parser.rs                                              1,107   recursive descent, positioned AST
├── vm.rs                                                    806   TitanVM  ⚠ no parity suite vs interpreter
├── lexer.rs                                                 444   includes the 4-byte seal codepoint
├── ast.rs                                                   343   node definitions
├── ascii_render.rs                                          149   terminal render
├── webgl_export.rs                                          105   WebGL export
├── python.rs                                                 70   pyo3 surface  ⚠ bindings package empty
├── lib.rs                                                    56
└── mod.rs                                                    41

crates/aether-kernel/src/
├── main.rs                     kernel_main, panic handler
├── interrupts.rs               IDT, CURRENT_STATE behind spin mutexes
├── scheduler.rs                SparseScheduler  ⚠ 4 tests never execute
├── allocator.rs                64 KB bump, #[global_allocator]
├── loader.rs                   ELF verification + byte-topology check
├── serial.rs                   COM1 — the only output channel
└── boot/
    ├── bios.rs                 BootInfo, MemoryRegion, Framebuffer
    └── topology.rs             HardwareTopology, IoCaps
```

---

## How To Read This Repository

Depending on why you are here, different files are the point.

**Evaluating whether the mathematics is right.** Start at [`persistence_invariants.rs`](crates/aether-core/tests/persistence_invariants.rs) — 11 property tests, and the two that matter most are `bottleneck_distance_respects_the_stability_bound` (the CSEH theorem) and `gaussian_noise_produces_no_long_h1_bar` (the negative control). Then `persistence.rs` itself. Then note what is **absent**: no comparison against ripser or GUDHI exists, so every assurance here is internal.

**Evaluating whether the engineering is sound.** [`attention_contracts.rs`](crates/aether-core/tests/attention_contracts.rs), 1,124 lines, is the most informative file in the repository — not because the code is best there, but because the *shape* of the file records a claim being repaired three times, with the cost model that finally falsified it added at the end.

**Deciding whether to use it.** Read [Limitations](#limitations) first, then [Prior Art](#prior-art). If you need production TDA, the answer is ripser and this document says so in four places.

**Looking for the interesting idea.** [Theory §13](#13-the-routing-gap-ratio) — the H₀ barcode is a sufficient statistic for deciding whether a topological method will pay, and it is cheap enough to consult before committing. That is the one result here that generalises past this repository.

**Wanting to contribute.** [`docs/reference/status.md`](docs/reference/status.md), then [Contributing](#contributing). The top two items are worth more than everything else combined.

**Hiring, or otherwise judging the author.** [What We Got Wrong](#what-we-got-wrong) is roughly a third of this document and includes six claims that were killed by their own measurements, two tests that turned out to be wrong rather than the code, and two phantom dependencies found by reading `Cargo.toml` instead of trusting the sentence beside it. That section is the argument. The green checkmarks are not.

---

## The Lean Formalization

`Aether/`, 10,474 lines of Lean 4. **Not built by CI**, which is the first and most important thing to say about it.

### What is actually there

| File | Lines | Theorems | `sorry` | What it is |
|---|---:|---:|---:|---|
| `Core.lean` | 3,356 | **47** | 0 | The formalization proper |
| `VM.lean` | 2,526 | 1 | 0 | A VM model in Lean |
| `Static.lean` | 1,891 | 0 | 0 | A static checker in Lean |
| `Parser.lean` | 1,569 | 0 | 0 | A parser in Lean |
| `Pipeline.lean` | 1,442 | 0 | 0 | A pipeline in Lean |
| `Lexer.lean` | 853 | 0 | 0 | A lexer in Lean |
| **Total** | **10,474** | **48** | **0** | |

The shape of that table is the honest summary: **47 of 48 theorems live in one file.** The other 8,281 lines hold exactly one theorem between them. They are a **second implementation** of the language — lexer, parser, static checker, VM — written in Lean, not proofs about the Rust one.

Zero `sorry` is real and worth having. A formalization with `sorry` scattered through it is a sketch; this one commits.

### What it does not do

**It does not verify the Rust.** There is no extraction, no refinement proof, and no correspondence argument connecting `Core.lean` to `crates/aether-core`. Two independent implementations of overlapping mathematics, one of which has theorems attached, is a useful thing to have. It is not the same as a verified implementation, and the difference is exactly the difference this README keeps insisting on between *compiles* and *boots*, or between *invariants* and *parity*.

**It is not checked to still compile.** No `lake build` runs anywhere in CI. Given that this repository's [CI had never executed at all](#the-part-where-a-repository-audit-ruined-my-afternoon) until recently, an ungated tree should be assumed stale until a build says otherwise.

### The honest disposition

Gate it or cut it. Gating means adding `lake build` to `ci.yml` and accepting the toolchain-pinning cost. Cutting means acknowledging that 8,281 lines of un-theoremed re-implementation are a liability rather than an asset. The current state — present, impressive-looking in a line count, unverified — is the one state that is worse than either decision, and it is listed as ⛔ Ungated in the [status dashboard](#honest-status-dashboard) rather than being quietly counted as a feature.

---

## Complexity Reference

Asymptotics for every operation whose cost a caller can actually feel. `n` is point count, `m` simplex count, `s` sequence length, `d` dimension, `b` bar count.

### Persistence

| Operation | Cost | Note |
|---|---|---|
| Rips simplex enumeration, `max_dim = k` | O(n<sup>k+2</sup>) | Tetrahedra are the O(n⁴) term — the reason `h2_default` caps at 48 points |
| Pairwise distances | O(n²d) | |
| Face lookup (`BTreeMap`) | O(log m) per face | Was O(m) linear scan — [the 26×](#the-26x) |
| Z₂ column reduction | O(m³) worst case | Worst case is rarely hit; the practical driver is m |
| `betti_at(radius)` | O(b) | A query on a computed diagram, not a recomputation |
| Witness complex, ℓ landmarks | O(ℓ<sup>k+2</sup> + nℓ) | Decouples cost from n — [Theory §9](#9-the-lazy-witness-complex) |

### Diagram metrics

| Operation | Cost | Note |
|---|---|---|
| Bottleneck distance | O(b³ log b) | Binary search over candidate costs × Kuhn's matching |
| p-Wasserstein | O(b³) | Hungarian, exact. [`ponytail:` ceiling](#5-bottleneck-and-wasserstein), trigger at `max_points` > 2048 |
| Landscape, r resolution, k levels | O(b·r + r·b log b) | Per-sample descending sort is the second term |
| Persistence image, p pixels | O(b·p) | |
| Persistent entropy | O(b) | |

### Attention

| Operation | Cost | Note |
|---|---|---|
| Dense attention | O(s²d) | The thing everything else is measured against |
| `single_linkage_clusters` | O(n²d + n² log n) | Edge enumeration + sort. [`ponytail:` marked](#single_linkage_clusters-and-the-theorem-it-is-checked-against) — not the binding term, since the mask is already Θ(s²) |
| `select_mask`, any selector | Θ(s²) memory | **The dense `[s, s]` bool mask dominates every selector's asymptotics** |
| `OracleTopK` | O(s²d) | Computes every score. A ruler, not a method |
| `routing_plan` | O(n² + n log n) | H₀ barcode of key directions |
| Scheduled attention, `nnz` blocks | O(nnz · B²d) | B = block size; working set is one tile |

That `select_mask` row is the one worth internalising. **Every selector in this crate materialises a full `[seq, seq]` boolean mask**, so no selector is sub-quadratic in memory regardless of how few keys it picks. The selection savings are real in *dot products* and absent in *allocation*. This is a property of a reference implementation written for checkability, and it is precisely the trigger recorded in the `ponytail:` comment on the clustering routine: a neighbour graph saves nothing until the mask stops being dense.

### Runtime substrate

| Operation | Cost | Note |
|---|---|---|
| `ManifoldHeap::alloc` | O(1) amortised | |
| `ManifoldHeap::get` | O(1) | Index + generation check |
| `ChebyshevGuard::calculate` | O(blocks) | One pass over block statistics |
| `hierarchical_query` | O(active blocks) | Prunes whole subtrees via admissible bound |
| `GeometricGovernor::adapt` | O(1) | PD update |
| `TimeDelayEmbedder::push` | O(1) | Ring buffer |
| `SparseAttentionGraph::add_point` | O(n) | Bitset row per point, 64-neighbour cap |
| `compute_betti_0` | O(n·α(n)) | Union-find, exact |
| `estimate_betti_1` | O(n) | Cycle-rank heuristic, **not** persistent homology |

### Where the time actually goes

Asymptotics are not the measurement. The [measured ceiling](#measured-scale-ceiling) is:

- H₀ at n=4,000 — **335 s**
- H₁ at n=300 — **131 s**
- H₂ at n=70 — **15.3 s**

All single-threaded, all `--release`. The engine is dominated by simplex enumeration and reduction, in that order, and nothing in it is parallelised or vectorised. The presets exist to keep a caller inside those numbers rather than discovering them.

---

## Worked Examples

`examples/` holds 17 programs. They still use the pre-rename `.aegis` and `.ag` extensions and the CLI prints a warning for them — [extension debt](#limitations) sitting in plain sight.

| File | Bytes | What it demonstrates |
|---|---:|---|
| `simple.aegis` | 32 | The smallest program that runs |
| `seal_demo.aegis` | 549 | A seal loop, minimal |
| `llm_benchmark.aegis` | 993 | LLM-shaped workload |
| `llm_demo.ag` | 1,263 | |
| `hello_manifold.aegis` | 1,407 | Embedding a series into a manifold |
| `ml_test.ag` | 1,426 | ML module surface |
| `regression_demo.aegis` | 1,602 | `regress` with model escalation |
| `3d_cluster.aegis` | 1,660 | Clustering in 3D |
| `grand_benchmark.aegis` | 2,724 | |
| `neural_topology.aegis` | 2,895 | Neural net + topology together |
| `benchmark_seal_vs_linear.aegis` | 4,638 | Seal loop against a linear baseline |
| `seal_loop_demo.aegis` | 5,119 | The fullest seal-loop example |
| `visualization_demo.aegis` | 6,191 | `render` to ASCII and WebGL |
| `benchmark_suite.aegis` | 8,163 | |
| `titan_bench.ag` | 615 | Titan VM |
| `benchmark_compare.py` | 2,252 | Python-side comparison harness |
| `llm_benchmark.py` | 1,347 | |

**A caveat that matters more than the table.** Four of these files have `benchmark` in the name, and `benchmark_compare.py` exists to compare against something. **No number from any of them appears anywhere in this README**, and none of them is run by CI. They are demonstration programs and exploratory harnesses, not evidence. A reader who assumed `benchmark_seal_vs_linear.aegis` had produced a published comparison would be wrong — the [core convergence claim is unmeasured](#the-faq-nobody-asked-for), and that is stated in the FAQ, the Limitations, and the Contributing list precisely because a directory listing suggests otherwise.

Run any of them:

```bash
cargo run -p aether-cli -- run examples/hello_manifold.aegis
```

The two Rust examples under `crates/aether-core/examples/` are different in kind — they are the probes that produce numbers this document actually quotes:

```bash
cargo run -p aether-core --example scale_probe   --release   # the scale ceiling table
cargo run -p aether-core --example routing_cost  --release   # the routing cost table
```

`scale_probe.rs` opens with a comment stating its own role: *"Not a test: a probe that prints the numbers the docs are allowed to quote."* That is the distinction this section is drawing, written into the source.

---

## Requirements

**Toolchain.** Rust **nightly**, pinned in `rust-toolchain.toml`. Nightly is required for `-Z build-std` (kernel and embedded targets) and `#![feature(abi_x86_interrupt)]` in the kernel. The CLI itself uses stable features only.

**Components.** `rust-src` and `llvm-tools-preview` for bare-metal builds; `clippy` and `rustfmt` for the gate.

**Targets.**

| Target | Purpose | Extra flags |
|---|---|---|
| host (`x86_64-pc-windows-msvc`, Linux, macOS) | CLI, tests | — |
| `x86_64-unknown-none` | kernel | `-Z build-std=core,alloc` |
| `thumbv7m-none-eabi` | `no_std` verification | `-Z build-std=core,alloc` |

**Dependencies.** `aether-core` declares three: `libm`, `heapless`, and `nalgebra`.

```toml
libm     = "0.2"
heapless = "0.8"
nalgebra = { version = "0.32", default-features = false, features = ["libm"] }
```

`libm` and `heapless` are used. **`nalgebra` has zero call sites** — `grep -r nalgebra crates/ --include=*.rs` returns nothing. It is declared non-optionally, so every build of the core crate compiles a linear-algebra library that no line of code references. It is `default-features = false` with the `libm` feature, which is why the `thumbv7m-none-eabi` build still succeeds; the `no_std` claim survives, the minimality claim does not. See [What We Got Wrong §6](#6-nalgebra-a-second-phantom-dependency).

For contrast, `aegis-core` declares the same dependency as `optional = true` — the duplicate crate is, on this one point, better configured than the real one.

**Feature flags** (`aether-core`):

| Feature | Implies | Meaning |
|---|---|---|
| `std` *(default)* | `alloc` | Standard library; CLI and tests |
| `alloc` | — | `no_std` with an allocator |
| `no_std` | `alloc` | Bare-metal mode |

The CLI adds `clap` and `rustyline`.

`aether-lang` used to pull `wgpu`, `reqwest`, `safetensors`, `pollster` and `bytemuck` through its **default** feature set, all with zero call sites, so every default build compiled an HTTP client and a 2023-era GPU stack that nothing referenced. All five are deleted.

`aether-gpu` is the crate that actually uses a GPU, and it is deliberately separate: `aether-core` is `no_std` and builds for `thumbv7m-none-eabi`, while `wgpu` needs `std` and a driver stack. A feature flag would leave a `no_std` crate whose dependency graph only resolves on hosted targets.

**Optional Python.** `pyproject.toml` builds a `pyo3` extension via `maturin`. The bindings directory is currently an empty package.

---

## Building And Testing

```bash
# the gate, exactly as CI runs it
cargo fmt --all -- --check
cargo clippy --workspace --exclude aether-kernel --all-targets -- \
  -D warnings -D clippy::correctness -D clippy::suspicious \
  -A clippy::style -A clippy::complexity -A clippy::perf
cargo test --workspace --exclude aether-kernel

# bare metal
cargo build -p aether-kernel -Z build-std=core,alloc --target x86_64-unknown-none

# no_std on a real embedded target
cargo build -p aether-core --no-default-features --features no_std \
  -Z build-std=core,alloc --target thumbv7m-none-eabi

# reproduce the tables in this README
cargo run -p aether-core --example scale_probe   --release
cargo run -p aether-core --example routing_cost  --release
```

Convenience aliases in `.cargo/config.toml`: `cargo gate`, `cargo invariants`, `cargo kernel`, `cargo embedded`, `cargo cli`.

### What CI actually runs

Three workflows. [`ci.yml`](.github/workflows/ci.yml) is the gate, and its header records why it exists in its current form:

> *Triggers on the branch this repository actually uses. The previous config listed `main`/`develop`, neither of which exists here, so no run had ever executed and the workspace had drifted out of compiling.*

| Job | Runner | What it proves |
|---|---|---|
| `test` | ubuntu · windows · macos, `fail-fast: false` | fmt (ubuntu only), clippy, `cargo test --workspace --exclude aether-kernel` |
| `invariants` | ubuntu | The 6 topology suites, **named separately so a failure names itself** |
| `no_std_check` | ubuntu | Builds `aether-core` for `thumbv7m-none-eabi` |
| `kernel` | ubuntu | Builds `aether-kernel` for `x86_64-unknown-none` |
| `docker` | ubuntu | Builds the image and runs `docker run --rm aether:test --help` |
| `release` | ubuntu | Tag-gated, `needs:` all five above |

Three details worth naming:

**`invariants` is a separate job on purpose.** It re-runs suites the `test` job already covers. The redundancy buys a **named check** in the PR list: a failure reads `Persistence Invariants` rather than being one line inside a 163-test log. The workflow comment states the reasoning — *a benchmark measured on a wrong implementation is not a result* — which is the [evidence policy](#the-evidence-policy-stated-as-rules) expressed as CI structure.

**`fail-fast: false` on the OS matrix.** A Windows-only failure and a macOS-only failure are different bugs, and fail-fast would hide the second behind the first.

**The docker job runs the binary, it does not just build it.** `docker run --rm aether:test --help` is the difference between an image that builds and an image that works. The Dockerfile had been copying crate directories from paths that never existed, which a build-only job would have caught but a build-and-run job catches more usefully.

[`docs.yml`](.github/workflows/docs.yml) builds the MkDocs site with `mkdocs build --strict` and deploys to GitHub Pages. `--strict` turns broken internal links into build failures, which is the same argument as the rest of this document applied to documentation: an unenforced claim decays.

[`publish.yml`](.github/workflows/publish.yml) handles releases.

**What CI does not run**, restated here because the [status dashboard](#honest-status-dashboard) depends on it: no `lake build` for the Lean tree, no QEMU boot, no external TDA parity, no gradcheck, and no execution of the four kernel scheduler tests.

`aether-kernel` is excluded from the host test job on purpose. It is a `no_std` bare-metal binary with no global allocator or panic handler and **cannot link for a host target**. It has its own CI job on `x86_64-unknown-none`. Running `cargo test --workspace` without the exclusion fails, and that failure is a property of the target, not a bug.

---

## Reproducing Every Number In This Document

The evidence policy in one rule: **a number without a reproduction command does not go in a table.** This section is that rule discharged — every quantitative claim above, mapped to the command that produces it.

| Claim | Where | Command |
|---|---|---|
| 223 / 223 tests pass | [Status](#honest-status-dashboard) | `cargo test --workspace --exclude aether-kernel` |
| 11 persistence invariants | [Theory](#theoretical-foundation) | `cargo test -p aether-core --test persistence_invariants` |
| 17 diagram-metric tests | [Test suite](#diagram_distancers--17-tests-381-lines) | `cargo test -p aether-core --test diagram_distance` |
| 29 attention contracts | [Test suite](#attention_contractsrs--29-tests-1124-lines) | `cargo test -p aether-core --test attention_contracts -- --nocapture` |
| 16 scheduled-attention tests | [Test suite](#scheduled_attentionrs--16-tests-576-lines) | `cargo test -p aether-core --test scheduled_attention -- --nocapture` |
| 7 scale tests, 1.10 s | [The 26×](#the-26x) | `cargo test -p aether-core --test persistence_scale --release` |
| Scale ceiling table | [Results](#measured-scale-ceiling) | `cargo run -p aether-core --example scale_probe --release` |
| Routing cost table | [What We Got Wrong §4](#4-topological-routing-on-unstructured-keys--not-sparse-at-all) | `cargo run -p aether-core --example routing_cost --release` |
| Placement / spread table | [What We Got Wrong §2](#2-euclidean-proximity-as-an-attention-mass-proxy--negative) | `cargo test -p aether-core --test attention_contracts -- --nocapture` |
| 58.8% block reduction | [Scheduled attention](#scheduled-attention) | `cargo test -p aether-core --test scheduled_attention -- --nocapture` |
| gap_ratio 2.70 vs 1.04 | [Theory §13](#13-the-routing-gap-ratio) | `cargo test -p aether-core --test attention_contracts -- --nocapture` |
| Kernel compiles bare metal | [Status](#honest-status-dashboard) | `cargo build -p aether-kernel -Z build-std=core,alloc --target x86_64-unknown-none` |
| `no_std` on Cortex-M3 | [Status](#honest-status-dashboard) | `cargo build -p aether-core --no-default-features --features no_std -Z build-std=core,alloc --target thumbv7m-none-eabi` |
| Formatting clean | [Status](#honest-status-dashboard) | `cargo fmt --all -- --check` |
| Clippy clean | [Status](#honest-status-dashboard) | `cargo clippy --workspace --exclude aether-kernel --all-targets -- -D warnings -D clippy::correctness -D clippy::suspicious -A clippy::style -A clippy::complexity -A clippy::perf` |
| 24,180 Rust lines | [Status](#honest-status-dashboard) | `Get-ChildItem crates -Recurse -Filter *.rs \| Get-Content \| Measure-Object -Line` |
| `nalgebra` has zero call sites | [What We Got Wrong §6](#6-nalgebra-a-second-phantom-dependency) | `grep -rn nalgebra crates/ --include=*.rs` |
| 60 GPU tests, RTX 4060 / Vulkan | [Status](#honest-status-dashboard) | `cargo test -p aether-gpu --release` |
| 0 of 10 GPU mutants escape | [FEATURES.md](crates/aether-gpu/FEATURES.md) | `./crates/aether-gpu/mutants.sh` |
| matmul crossover n=128, 38× at n=512 | [FEATURES.md](crates/aether-gpu/FEATURES.md) | `cargo run -p aether-gpu --example tensor_crossover --release` |
| 10,474 Lean lines, 48 theorems, 0 `sorry` | [Lean](#the-lean-formalization) | `Get-ChildItem Aether -Recurse -Filter *.lean \| Get-Content \| Measure-Object -Line`, then `Select-String "^\s*(theorem\|lemma)\s"` and `Select-String "\bsorry\b"` |

### Numbers that are *not* reproducible from this repository

Listed separately rather than mixed into the table above, because the distinction is the point.

| Claim | Why it cannot be reproduced here | Where it came from |
|---|---|---|
| 56.6% / 80.9% block reduction at seq 1024 / 4096 | Requires CUDA | Upstream [`triton-lang/kernels#22`](https://github.com/triton-lang/kernels/pull/22) |
| 1.04×–3.48× sparse-vs-dense wall clock | Measured on an RTX 4060 | Same |
| 29.07 s before the `BTreeMap` refactor | Requires checking out the parent of `27d70fa` | Historical, same machine |

The upstream GPU figures are **cited, not claimed**. This port is a scalar CPU kernel with no SIMD, no threading and no GPU; it reproduces the answer and the block reduction, not the speed.

### Measurement substrate

Everything measured in this document ran on a **single machine**: Windows 11, Rust nightly, single core, `--release` where noted.

**No confidence intervals. Single runs. No core pinning. No turbo control.** These are engineering measurements taken to size caps and catch regressions, not a study. The 26× figure is robust to all of that because it is a 26× difference on identical assertions on the same machine within one session; the absolute scale-ceiling timings are not, and should be read as order-of-magnitude.

### The evidence policy, stated as rules

1. **Every number is measured.** Projected, estimated and theoretical-peak figures are labelled as such in the same cell, or they are absent.
2. **Every baseline is named.** "Faster than before" is unfalsifiable; "29.07 s → 1.10 s across commit `27d70fa`, identical assertions" is checkable.
3. **Every comparison has a control.** The attention ablations report against `Random` (floor) and `OracleTopK` (ceiling) at equal budget, because a selector measured against nothing is a selector measured against its author's hopes.
4. **Negative results get the same typography as positive ones.** [An entire section](#what-we-got-wrong), near the top, with the numbers that killed each claim.
5. **A count is not evidence.** A test file with 40 tests that never runs contributes nothing. The [status dashboard](#honest-status-dashboard) marks a row Active only when a command in `ci.yml` produces its evidence.
6. **Correctness precedes performance.** A benchmark measured on an implementation whose correctness is unestablished is not a result. This is why the [persistence invariants have their own named CI job](#reproducing-every-number-in-this-document) rather than being folded into the general test run.

---

## Limitations

Longer than most projects' feature lists. That is the point.

**No external parity.** The persistence engine has never been compared against ripser, GUDHI, giotto-tda or Dionysus on shared fixtures. The invariant suite is *not* parity: a self-consistently wrong implementation can satisfy every internal property it tests. This is the largest correctness debt here.

**Scale.** H₀ at n=4,000 takes 335 s. H₁ at n=300 takes 131 s. The reduction is O(m²) in the simplex count and single-threaded. Production TDA libraries handle clouds orders of magnitude larger.

**The core claim is unmeasured.** Whether topological convergence beats scalar convergence on real problems, against a tuned baseline, has not been tested. The machinery is correct; its *value* is unestablished.

**The GPU backend is not used by anything.** `aether-gpu` is real — 13 WGSL
kernels, resident tensors, 60 tests, verified against finite differences and a
mutation matrix — but no line of `aether-core` or `aether-lang` calls it. Both
candidate integrations have been measured and neither has been made:
`Tensor::matmul` crosses over at n=128 and reaches 38× at n=512 with f64↔f32
conversion counted, and `pairwise_sqdist` never pays because the persistence
reduction is CPU-side, so the matrix must come back. See
[`crates/aether-gpu/FEATURES.md`](crates/aether-gpu/FEATURES.md).

**Two phantom dependencies.** `nalgebra` is a non-optional dependency of `aether-core` with zero call sites, and `wgpu`/`pollster`/`bytemuck` ship in `aether-lang`'s default features with one comment between them. Both are queued for removal; both are [written up](#6-nalgebra-a-second-phantom-dependency) rather than quietly deleted before publication.

**Every selector allocates a dense `[seq, seq]` mask.** No selector in `attention` is sub-quadratic in *memory*, however few keys it picks. The savings are real in dot products and absent in allocation. This is a property of a reference implementation built for checkability, and it is the recorded trigger on the clustering `ponytail:` marker.

**The `ml` subtree has no dedicated test file.** 3,992 lines — clustering, classification, neural, autograd, convolution — against 2,894 lines of tests that target the topology core almost exclusively. K-means is not tested for initialisation sensitivity. The two independent single-linkage implementations in this crate are not checked against each other.

**No gradcheck anywhere.** `ml/autograd.rs` has no finite-difference verification against its own forward pass.

**`ml/convolution.rs` is forward-only.** A convolution with no backward pass is a feature extractor, not a trainable layer.

**`topology.rs` shares vocabulary with the persistence engine and shares none of its guarantees.** It computes Betti-like numbers over `&[u8]` by windowed density heuristics, with `MAX_BETTI_1` saturating at 10. None of the 11 persistence invariants apply to it. Three functions in this crate are named some variant of `compute_betti_0` and only one of them is exact persistent homology.

**`verify_binary_topology` is not a security mechanism.** It is a structural integrity heuristic that an adversary can defeat by padding. The word "authentication" in the source is an overclaim.

**The governor is PD, not PID**, despite its doc comment. Its stability properties are argued, not tested.

**`ml/benchmark.rs` is not a benchmark** in the sense this document uses the word. It is an internal harness for the model-escalation policy, and no number in any table comes from it. Four files in `examples/` also have `benchmark` in their names and produce no published number either.

**The Titan VM has no parity suite against the interpreter.** Two execution engines, no differential test.

**No backward pass.** `aether_core::attention` is forward-only, so no gradient check accompanies it. A forward-correct kernel with a wrong backward trains to a plausible worse optimum that loss curves will not reveal, so any backward must land with its `gradcheck` in the same change.

**Attention results are synthetic.** Every ablation number above is measured on synthetic keys. Whether real attention key distributions carry H₀ structure is unmeasured, and it decides whether the routing result transfers at all.

**The Triton port reproduces answers, not timings.** Scalar CPU, no SIMD, no threading, no GPU. The upstream 1.04×–3.48× figures were measured on an RTX 4060 this workspace cannot reach.

**Per-block salience is order-dependent.** Under component-size ties the absorbed component is chosen by index order, so the same centroid scores differently depending on sequence position. The multiset is invariant. A fix needs a tie-break on centroid content.

**The kernel compiles but is not asserted to boot.** No QEMU logs, no hardware matrix.

**Four kernel scheduler tests never execute.** They exist in `scheduler.rs`, but `aether-kernel` is a `no_std` binary with no test harness.

**The Lean tree is ungated.** 10,474 lines, 48 theorems, 0 `sorry`, and 8,281 lines holding one theorem between them. No `lake build` in CI.

**Duplicate crates.** `aegis-core` and `aegis-cli` are near-copies of their `aether` counterparts, left over from a rename done by copying.

**Extension drift.** The CLI accepts `.aether` and `.ae`; the repository examples use `.aegis` and `.ag` and print a warning.

**All timings are single-machine, single-run**, Windows 11, nightly, no confidence intervals, no turbo control, no core pinning.

---

## Glossary Of Words I Made Up

**Seal loop** — a loop terminating on topological convergence rather than a scalar threshold. Named for `🦭`. No further justification available.

**Manifold heap** — the `no_std` allocator in `aether-core::memory`, organising objects spatially and reclaiming cold branches, treating unused memory as entropy.

**Bio mode** — the CLI's default execution mode. It prints a shield emoji. It does nothing biological.

**Titan VM** — the bytecode VM in `aether-lang::vm`. Named at 3 AM. Behind the interpreter on coverage.

**Geometric concentrator** — the component in `manifold.rs` reducing a point cloud to the region carrying the most structure.

**Elder rule** — not made up; this one is real algebraic topology. When two components merge, the younger dies. It is the reason a block salience score is an H₀ death time.

**Placement** — the scale-free statistic every attention ablation is reported against: `(selector − random) / (oracle − random)` at equal budget. 0 is random, 1 is a cheating oracle. [Theory §12](#12-the-placement-statistic) covers the two ways it lies.

**Gap ratio** — the H₀ barcode statistic `routing_plan` uses to decide, at runtime, whether topological routing will pay. Structured clouds measure ≥ 2.70; chained ones ≤ 1.04. [Theory §13](#13-the-routing-gap-ratio).

**Chaining** — the single-linkage failure mode where a cloud with no density gaps gets absorbed one point at a time into one giant component. `[61, 1, 1, 1]` out of 64 keys. Not a bug — H₀ correctly reporting that uniform data has no structure to route on.

**Sink block** — in the scheduled-attention config, the leading block every query attends to regardless of salience. Attention sinks are a real empirical phenomenon in transformers; the scheduler reserves them explicitly rather than hoping salience picks them up.

**Budget** — the number of keys a selector may examine per row. Every ablation in this document holds it equal across selectors, because [a selector that declines to select](#3-the-first-fix-was-also-wrong--measurement-artifact) posts catastrophic numbers without losing on mechanism.

**Essential class** — a homology class that never dies; its bar runs to infinity. A connected cloud has exactly one essential H₀ class. `a_single_point_has_one_essential_component_and_nothing_else` is the degenerate case.

**Witness complex** — the landmark-based approximation to Vietoris–Rips that makes persistence viable at 24 landmarks on a Cortex-M3. An approximation, labelled as one. [Theory §9](#9-the-lazy-witness-complex).

**Generational handle** — `Gc<T>` in the manifold heap: an index plus a generation counter, so a stale handle whose slot was reused fails the check and returns `None` instead of aliasing whatever now lives there.

**Fail-fast budget** — the persistence engine's caps. Exceeding one returns `TooManyPoints` or `TooManySimplices` rather than subsampling silently or exhausting memory. A **time** budget, not a correctness limit; raise it explicitly and wait longer.

**`ponytail:` comment** — the repository's convention for marking a deliberate shortcut with its ceiling and the trigger that should force revisiting it. Two exist, both in `aether-core`, both with concrete triggers. A marker naming a ceiling but no trigger is the kind that silently rots.

**Ungated** — a claim whose evidence exists but whose command does not run in CI. Distinguished throughout from ✅ Active (a CI command produces the evidence) and ❌ Does not exist. The Lean tree, the kernel boot, and four scheduler tests are ungated.

---

## License

MIT. See [LICENSE](LICENSE).

Copyright © 2026 Teerth Sharma. The Lean formalization, the persistence engine, the language, and every mistake catalogued above are original work.

The scheduled-attention module is a port of [`triton-lang/kernels#22`](https://github.com/triton-lang/kernels/pull/22), contributed by the same author to that repository under its license.

---

<p align="center">
  <strong>Invented by <a href="https://teerthsharma.vercel.app/">Teerth Sharma</a></strong><br>
  <a href="https://github.com/teerthsharma/Aether-Lang">github.com/teerthsharma/Aether-Lang</a> · <code>teerthsharma@outlook.com</code>
</p>

<p align="center">
  <em>Every number above was measured. Every claim names its control.<br>
  The section where I am wrong is a third of the document, and that is the feature.</em>
</p>
