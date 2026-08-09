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
  <a href="#results"><img src="https://img.shields.io/badge/tests-163%20passing-brightgreen?style=flat-square" alt="163 tests"></a>
  <a href="#mutation-testing-or-how-i-learned-to-stop-trusting-green-checkmarks"><img src="https://img.shields.io/badge/mutants-8%20injected-purple?style=flat-square" alt="8 mutants"></a>
  <a href="#what-we-got-wrong"><img src="https://img.shields.io/badge/claims%20killed-5-red?style=flat-square" alt="5 claims killed"></a>
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

- [Before You Read This](#before-you-read-this)
- [The 30-Second Pitch](#the-30-second-pitch)
- [Abstract](#abstract)
- [Why I Did This To Myself](#why-i-did-this-to-myself)
- [Honest Status Dashboard](#honest-status-dashboard)
- [Quick Start — Five Minutes To A Betti Number](#quick-start--five-minutes-to-a-betti-number)
- [The Language](#the-language)
  - [Tildes, Seals, And Other Crimes](#tildes-seals-and-other-crimes)
  - [The Topology Module](#the-topology-module)
  - [Seal Loops](#seal-loops)
- [Background](#background)
  - [Why Topology?](#why-topology)
  - [Why A Language And Not A Library?](#why-a-language-and-not-a-library)
  - [Prior Art](#prior-art)
- [Theoretical Foundation](#theoretical-foundation)
- [Implementation](#implementation)
- [Results](#results)
- [Mutation Testing, Or How I Learned To Stop Trusting Green Checkmarks](#mutation-testing-or-how-i-learned-to-stop-trusting-green-checkmarks)
- [What We Got Wrong](#what-we-got-wrong)
- [Design Decisions That Seemed Good At 3 AM](#design-decisions-that-seemed-good-at-3-am)
- [The FAQ Nobody Asked For](#the-faq-nobody-asked-for)
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
- A GPU. There is no GPU. There is a `wgpu` dependency in `Cargo.toml` with *zero call sites*, discovered while auditing this repo and now documented as a lie in the [claim ledger](docs/reference/status.md). More on that in the section where I list what this project used to claim and no longer does.

Grab a coffee. There are 21,262 lines of Rust and 11,652 lines of Lean below, and roughly a third of this document is about the ways I was wrong.

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
| Core math dependencies | **1** (`libm`) | many | many | many |
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

Because I wanted it to run on a kernel with no OS, and `scikit-tda` has opinions about `numpy` that a bare-metal x86_64 target does not share. `aether-core` is `no_std`, depends on `libm` and nothing else, and compiles for `thumbv7m-none-eabi` — a Cortex-M3. Try that with `gudhi`.

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
| Lean 4 formalization | ⛔ Ungated | 11,652 lines, 48 theorems, **no `lake build` in CI** |
| GPU acceleration | ❌ **Does not exist** | `wgpu` declared, zero call sites |
| Attention backward pass | ❌ Does not exist | forward only; no gradcheck possible |
| Wall-clock speedup claims | ❌ Withdrawn | see [What We Got Wrong](#what-we-got-wrong) |

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WORKSPACE GATE                        branch master, nightly, Win11
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  cargo fmt --all -- --check                                   clean
  cargo clippy -D correctness -D suspicious                     clean
  cargo test --workspace --exclude aether-kernel     163 / 163 passed
  cargo build -p aether-kernel --target x86_64-unknown-none        ok
  cargo build -p aether-core  --target thumbv7m-none-eabi          ok
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Rust lines (crates/)                                        21,262
  Lean lines (Aether/)              11,652   theorems 48   sorry 0
  Test suites gated in CI                                          7
  Claims withdrawn during audit                                    5
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

The principled one: **`no_std`**. The core math has exactly one dependency, `libm`, and compiles for targets with no operating system and no allocator by default. That is not retrofitted onto a Python library; it is designed for on line one. The same persistence code backing `topology.ph` in the CLI runs inside `aether-kernel` on bare x86_64, where it informs scheduling.

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

Both **exact**. Bottleneck by binary search over the finite candidate-cost set, feasibility decided by Kuhn's augmenting-path matching on the threshold graph. Wasserstein by the Hungarian algorithm on the same $(n+m)^2$ cost matrix — $O((n+m)^3)$, marked with a `ponytail:` comment naming the ceiling and the upgrade path (auction or Sinkhorn) should diagrams reach thousands of bars.

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

**11,652 lines of Lean.** `Aether/` contains a Lean 4 formalization with 48 theorems and zero `sorry`. It is also **not built by CI**, and 8,281 of those lines (`Lexer`, `Parser`, `Pipeline`, `Static`, `VM`) contain **zero theorems** — a second implementation of the language in Lean, not proofs about the first. Either gate it or cut it. It is in the ledger as ungated, which is the honest interim state.

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
So the same persistence code that runs in the CLI runs in `aether-kernel` on bare metal, where it informs scheduling. Also because `libm` as your only dependency is a nice place to be.

**Does the kernel boot?**
It **compiles** for `x86_64-unknown-none`. Booting is not tested. Different claims.

**There is a `wgpu` dependency. Is there GPU acceleration?**
No. `wgpu`, `pollster` and `bytemuck` are in `aether-lang`'s **default** feature set with **zero call sites**. The only occurrence of `wgpu` in the entire tree is the comment `"future hooks for wgpu"` at `ml/tensor.rs:6`. Every default build compiles a GPU stack for nothing. Documented in the ledger, queued for removal.

**Why is there a `CHANGELOG.md` with almost nothing in it?**
Fair.

**How much of this README is the author admitting to mistakes?**
Roughly a third by line count. Intentional. A reader who finds an unstated limitation themselves discounts every other claim in the document.

**Can I contribute?**
Yes. Read [`docs/reference/status.md`](docs/reference/status.md) first — it lists exactly what is gated, what is ungated, and what command produces each piece of evidence. The highest-value contribution by a wide margin is the external parity harness against a pinned ripser.

---

## Repository Layout

```
Aether-Lang/
├── crates/
│   ├── aether-core/          math foundation, no_std, libm only
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
│   │   ├── tests/            most of the 163 tests live here
│   │   └── examples/         scale_probe, routing_cost — reproduce the tables
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

**Dependencies.** `aether-core` depends on `libm` and `heapless`. That is the whole mathematical surface. The CLI adds `clap` and `rustyline`. The `wgpu`/`reqwest`/`pollster`/`bytemuck` entries in `aether-lang`'s default features have zero call sites and should not be there.

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

`aether-kernel` is excluded from the host test job on purpose. It is a `no_std` bare-metal binary with no global allocator or panic handler and **cannot link for a host target**. It has its own CI job on `x86_64-unknown-none`. Running `cargo test --workspace` without the exclusion fails, and that failure is a property of the target, not a bug.

---

## Limitations

Longer than most projects' feature lists. That is the point.

**No external parity.** The persistence engine has never been compared against ripser, GUDHI, giotto-tda or Dionysus on shared fixtures. The invariant suite is *not* parity: a self-consistently wrong implementation can satisfy every internal property it tests. This is the largest correctness debt here.

**Scale.** H₀ at n=4,000 takes 335 s. H₁ at n=300 takes 131 s. The reduction is O(m²) in the simplex count and single-threaded. Production TDA libraries handle clouds orders of magnitude larger.

**The core claim is unmeasured.** Whether topological convergence beats scalar convergence on real problems, against a tuned baseline, has not been tested. The machinery is correct; its *value* is unestablished.

**No GPU.** None. The dependencies suggesting otherwise have zero call sites.

**No backward pass.** `aether_core::attention` is forward-only, so no gradient check accompanies it. A forward-correct kernel with a wrong backward trains to a plausible worse optimum that loss curves will not reveal, so any backward must land with its `gradcheck` in the same change.

**Attention results are synthetic.** Every ablation number above is measured on synthetic keys. Whether real attention key distributions carry H₀ structure is unmeasured, and it decides whether the routing result transfers at all.

**The Triton port reproduces answers, not timings.** Scalar CPU, no SIMD, no threading, no GPU. The upstream 1.04×–3.48× figures were measured on an RTX 4060 this workspace cannot reach.

**Per-block salience is order-dependent.** Under component-size ties the absorbed component is chosen by index order, so the same centroid scores differently depending on sequence position. The multiset is invariant. A fix needs a tie-break on centroid content.

**The kernel compiles but is not asserted to boot.** No QEMU logs, no hardware matrix.

**Four kernel scheduler tests never execute.** They exist in `scheduler.rs`, but `aether-kernel` is a `no_std` binary with no test harness.

**The Lean tree is ungated.** 11,652 lines, 48 theorems, 0 `sorry`, and 8,281 lines containing no theorems at all. No `lake build` in CI.

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
