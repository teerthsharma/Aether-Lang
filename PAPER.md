<h1 align="center">Aether: the technical paper</h1>

<p align="center">
  The full account behind the <a href="README.md">README</a>: the mathematics the code implements, the language definition,<br>
  the implementation, every measurement with its control, verification, and the claims that did not survive.
</p>

<p align="center">
  <strong>Teerth Sharma</strong> · <a href="https://github.com/teerthsharma/Aether-Lang">github.com/teerthsharma/Aether-Lang</a> · <a href="https://doi.org/10.5281/zenodo.21997728">doi:10.5281/zenodo.21997728</a> · <a href="https://teerthsharma.github.io/Aether-Lang/">documentation</a>
</p>

---

## Abstract

Aether-Lang is a research language in which persistent homology is a language primitive: `topology.ph`, `topology.betti` and `topology.intervals` are builtins, and `seal until` loops terminate on an arbitrary predicate, including the stability of a Betti vector. The premise is that Betti numbers are integers, so a stopping rule defined on them cannot jitter the way a scalar residual does; whether it beats a tuned scalar criterion is unmeasured. The core is a bounded, exact $\mathbb{F}_2$ persistence engine for $H_0$–$H_2$ over Vietoris–Rips and lazy-witness filtrations, written in `no_std` Rust against `libm`, which builds for a Cortex-M3 and a bare-metal x86_64 kernel. Exact diagram metrics and vectorisations sit on top, beside a certified library that decides linking numbers, top-$k$ selections and arrangement invariants, or refuses. Correctness rests on 12 property tests including the Cohen-Steiner–Edelsbrunner–Harer bound, a closed-form circle ground truth reproduced to 1e-12, and 52 injected mutants of which none escape; parity against ripser or GUDHI has not been run. Indexing the face lookup cut the scale suite from 29.07 s to 1.10 s. The Rust rebuild of the author's Triton sparse-attention kernel reproduces its block schedules exactly, and a same-budget ablation finds its topological selection recovers less attention mass than random selection.

**Keywords:** persistent homology · Vietoris–Rips filtration · topological data analysis · certified computation · domain-specific languages · `no_std` Rust · sparse attention · mutation testing

---

## Contents

**Front matter**
- [Abstract](#abstract)

**The paper**
- [1. Introduction](#1-introduction) — [1.1 Motivation](#11-motivation) · [1.2 Why a language and not a library](#12-why-a-language-and-not-a-library) · [1.3 Scope of the claims](#13-scope-of-the-claims) · [1.4 Evidence policy](#14-evidence-policy) · [1.5 Status](#15-status) · [1.6 Reading guide](#16-reading-guide)
- [2. Background and prior art](#2-background-and-prior-art) — [2.1 Scalar and structural stopping criteria](#21-scalar-and-structural-stopping-criteria) · [2.2 Prior art](#22-prior-art) · [2.3 Differentiators](#23-differentiators)
- [3. Theoretical foundation](#3-theoretical-foundation) — 29 numbered results, each with its implementing function
  - Complexes and homology: [3.1 Vietoris–Rips filtration](#31-the-vietorisrips-filtration) · [3.2 Complex size](#32-complex-size-and-the-fail-fast-budget) · [3.3 Boundary operator](#33-chains-and-the-boundary-operator-over-f2) · [3.4 Betti numbers and Euler–Poincaré](#34-homology-betti-numbers-and-the-eulerpoincaré-relation) · [3.5 Column reduction](#35-persistence-by-column-reduction) · [3.6 Diagram size](#36-diagram-size-on-the-complete-complex) · [3.7 Betti numbers from a diagram](#37-betti-numbers-from-a-diagram) · [3.8 Single linkage and the elder rule](#38-h0-single-linkage-and-the-elder-rule) · [3.9 Witness filtration](#39-the-lazy-witness-filtration)
  - Metrics and vectorisations: [3.10 Stability](#310-the-stability-theorem) · [3.11 Bottleneck and Wasserstein](#311-bottleneck-and-wasserstein-distances) · [3.12 Landscapes](#312-persistence-landscapes) · [3.13 Entropy](#313-total-persistence-and-persistent-entropy) · [3.14 Images](#314-persistence-images) · [3.15 Polygon chord](#315-the-regular-polygon-chord) · [3.16 Delay embedding](#316-delay-embedding) · [3.17 Graph Betti numbers](#317-graph-betti-numbers-on-the-streaming-path)
  - Attention: [3.18 Softmax and the online recurrence](#318-numerically-stable-softmax-and-the-online-recurrence) · [3.19 Backward pass](#319-the-scheduled-attention-backward-pass) · [3.20 Block salience and the oracle](#320-block-salience-recovered-mass-and-the-oracle) · [3.21 Placement](#321-the-placement-statistic) · [3.22 Gap ratio](#322-the-routing-gap-ratio)
  - Runtime substrate: [3.23 Admissible pruning](#323-admissible-bounds-for-hierarchical-pruning) · [3.24 Drift](#324-drift-as-a-second-difference) · [3.25 Chebyshev guard](#325-the-chebyshev-guard) · [3.26 Governor](#326-the-governor-control-law) · [3.27 Gossip](#327-ring-gossip-consensus) · [3.28 Learning primitives](#328-learning-primitives) · [3.29 Convergence predicates](#329-the-convergence-predicates) · [3.30 References](#330-references)
- [The certified library](#49-the-certified-library) — ten modules that return a certificate or a typed refusal; full derivations in the [documentation](https://teerthsharma.github.io/Aether-Lang/integrated/)
- [4. The language](#4-the-language) — [4.1 Lexical conventions](#41-lexical-conventions) · [4.2 Statements](#42-statement-grammar) · [4.3 Expressions](#43-expression-grammar) · [4.4 Numeric literals](#44-numeric-literals) · [4.5 The topology module](#45-the-topology-module) · [4.6 Seal loops](#46-seal-loops) · [4.7 The regress statement](#47-the-regress-statement-and-convergencecond) · [4.8 Execution engines](#48-two-execution-engines) · [4.9 The certified library](#49-the-certified-library)
- [5. Implementation](#5-implementation) — [5.1 Workspace](#51-workspace) · [5.2 Persistence engine](#52-the-persistence-engine) · [5.3 Diagram module](#53-the-diagram-module) · [5.4 Attention](#54-the-attention-subsystem) · [5.5 Scheduled attention](#55-scheduled-attention) · [5.6 ML subsystem](#56-the-ml-subsystem) · [5.7 Runtime substrate](#57-the-runtime-substrate) · [5.8 aether-lang](#58-aether-lang) · [5.9 aether-kernel](#59-aether-kernel) · [5.10 aether-cli](#510-aether-cli) · [5.11 aether-gpu](#511-aether-gpu) · [5.12 Duplicate crates](#512-duplicate-crates) · [5.13 Complexity](#513-complexity-reference) · [5.14 Design decisions](#514-design-decisions)
- [6. Evaluation](#6-evaluation) — [6.1 Substrate](#61-measurement-substrate) · [6.2 Face index](#62-the-face-index-refactor) · [6.3 Scale ceiling](#63-measured-scale-ceiling) · [6.4 Closed form](#64-exactness-against-closed-form) · [6.5 Scheduled attention](#65-scheduled-attention)
- [7. Verification](#7-verification) — [7.1 Test inventory](#71-test-inventory) · [7.2 Coverage gaps](#72-what-the-suite-does-not-cover) · [7.3 Mutation testing](#73-mutation-testing) · [7.4 Lean](#74-the-lean-formalization) · [7.5 CI](#75-continuous-integration)
- [8. Negative results: what we got wrong](#8-negative-results-what-we-got-wrong)
- [9. Limitations](#9-limitations)

**Back matter**
- [Quick start](#quick-start) · [Requirements](#requirements) · [Reproducing every number](#reproducing-every-number) · [Repository layout](#repository-layout) · [FAQ](#faq) · [Contributing](#contributing) · [Glossary](#glossary) · [License](#license)

---

## 1. Introduction

### 1.1 Motivation

Iterative numerical procedures almost always stop on a scalar: a residual is watched until it falls below a tolerance $\varepsilon$. The rule works, and it has two failure modes familiar to anyone who has trained a model. A loss oscillating in its third decimal forces either a patience parameter or a moving average, both of which are further hyperparameters to defend; and a scalar compresses the entire state into one number before thresholding it, so two residual fields with identical $L_2$ norms but different structure are indistinguishable to it.

Topological data analysis offers a property that is almost never exploited in *control flow*: Betti numbers are integers. A loss of `0.0341` against `0.0339` is noise; $\beta_1$ moving from 3 to 1 is an event. A stopping rule defined on a discrete invariant has no decimals in which to jitter. Betti numbers are also homotopy invariants, unchanged under continuous deformation, which is the property wanted in a stopping rule: insensitive to the wiggle, sensitive to the event. The price of computing them is bounded and knowable in advance (§3.2), which is why the engine takes a budget and refuses work rather than discovering the problem at 40 GB resident.

The premise is therefore a narrow one: *some loops should terminate when the shape of the data stops changing, not when a float becomes small.* Aether-Lang makes that expressible. Whether it pays — whether stopping on $\beta$-stability beats stopping on a tuned scalar residual on real problems — is a controlled experiment that has not been run (§9), and no table in this document implies otherwise.

### 1.2 Why a language and not a library

**`no_std` from the first line.** The mathematical core computes against `libm`, uses `heapless` for fixed-capacity containers, and compiles for targets with no operating system and no default allocator. It builds for `thumbv7m-none-eabi`, a Cortex-M3. The same persistence code that backs `topology.ph` in the CLI links into `aether-kernel` on bare x86_64. That is not something that can be retrofitted onto a library written against `numpy`, which is why `scikit-tda` and GUDHI are not candidates for this layer.

**A keyword changes how a primitive is reached for.** When persistence is a builtin it appears in loop conditions; when it is `from gudhi import RipsComplex` it appears in a plot at the end of a notebook. `seal until ...` reads as a loop; `while not tda.has_converged(diagram, prev, 1e-6):` reads as bookkeeping. Concretely, `manifold`, `block`, `regress` and `render` are statement kinds with parser rules and AST nodes (§4.2), not entries in a builtin table.

**A runtime that owns the scheduler and the allocator.** The question this repository asks is different from the one ripser answers. ripser is excellent, fast, and checked by far more people than have read this file; for production TDA it is the right tool, and this document says so wherever throughput is discussed. The question here is what happens when topology makes execution decisions inside a runtime that also owns scheduling and allocation, rather than describing data after the fact.

### 1.3 Scope of the claims

The repository contains 53,228 lines of Rust across 105 files in `crates/` and 11,637 lines of Lean in `Aether/`. Both are raw line counts including tests, comments and blank lines, each reproduced by the command in [Reproducing every number](#reproducing-every-number). An earlier revision stated 21,262 and 11,652; neither was reproducible, and the Rust figure was subsequently re-measured at 36,305 before this revision re-measured it again. A document that insists every number carries a command has to survive that rule being applied to itself.

What is claimed: a bounded, exact $\mathbb{F}_2$ persistence engine for $H_0$, $H_1$ and $H_2$; exact diagram metrics and standard vectorisations; a language whose grammar carries topology as statement kinds; a `no_std` build for an embedded target; a kernel that compiles for `x86_64-unknown-none`; a Rust port of a Triton sparse-attention kernel whose schedules reproduce the original exactly; and a certified library of ten modules, each of which states what it certifies, what it only estimates, and when it refuses.

What is not claimed:

- **Production readiness.** This is a research language.
- **GPU acceleration of the language or the topology engine.** A GPU backend exists — `aether-gpu`, 107 tests, measured on an RTX 4060 — and **nothing in `aether-core` or `aether-lang` calls it**. The cost and precision of both candidate integrations are measured; the integrations are not made. An earlier version of this line said there was no GPU at all, which was true of the `wgpu` dependency it described and is no longer true of the tree.
- **Agreement with other TDA libraries.** External parity against a pinned ripser or GUDHI is not done. The invariant suite is not parity — a self-consistently wrong implementation can satisfy every internal property — and this is the largest correctness debt in the repository.
- **That `seal until convergence(ε)` is topological.** It runs (§4.6), and it is a scalar max-norm tolerance on the body's value. The stopping signal the built-in `regress` statement computes is a sign-change count, not persistent homology (§3.29); a program that wants a Betti-number stopping rule writes one with `seal until stable(...)` (§4.6).
- **That the kernel boots.** Compiling and booting are different claims (§5.9).

### 1.4 Evidence policy

Six rules govern every number in this document. They are stated here because every later section is written against them.

1. **Every number is measured.** Projected, estimated and theoretical-peak figures are labelled as such in the same cell, or they are absent.
2. **Every baseline is named.** "Faster than before" is unfalsifiable; "29.07 s → 1.10 s across commit `27d70fa`, identical assertions" is checkable.
3. **Every comparison has a control.** The attention ablations report against `Random` (floor) and `OracleTopK` (ceiling) at equal budget, because a selector measured against nothing is measured against its author's expectations.
4. **Negative results get the same typography as positive ones.** [Section 8](#8-negative-results-what-we-got-wrong) is a first-class section with the numbers that killed each claim.
5. **A count is not evidence.** A test file with 40 tests that never runs contributes nothing. The status table below marks a row Active only when a command in `ci.yml` produces its evidence.
6. **Correctness precedes performance.** A benchmark measured on an implementation whose correctness is unestablished is not a result. This is why the persistence invariants have their own named CI job rather than being folded into the general test run.

### 1.5 Status

A row is **Active** only if a command in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) produces its evidence. A test count in a file is not evidence if the file never runs.

A separate status, **Hardware-gated**, was needed once the GPU backend arrived. Those tests were briefly worse than useless: a test that returns early on a missing adapter *passes*, so `cargo test --workspace` reported roughly forty GPU tests green while executing none of them — the unseen green checkmark this document warns about, produced by its own author. They are now `#[ignore]`d behind an off-by-default `gpu` feature, so the same run prints `80 ignored` across the hardware suites. CI still compiles them with `cargo build -p aether-gpu --tests --features gpu`, so a broken one is caught rather than hidden behind the ignore, and asking for the hardware tests on a machine with no adapter fails rather than skipping, so the feature flag is the only switch and cannot be half-honoured. A hardware-gated row reads as "verified on one developer machine", which is weaker than every Active row.

| Subsystem | Status | Evidence |
|---|---|---|
| Lexer, parser, AST | **Active** | 11 tests (lexer 4, parser 7), `cargo test -p aether-lang` |
| Interpreter — assignment, loops, functions, classes | **Active** | 11 tests |
| `topology.ph` / `betti` / `intervals` | **Active** | interpreter topology tests |
| Persistent homology $H_0$/$H_1$/$H_2$ | **Active** | 9 in-module + **12 invariants** |
| Lazy witness complex | **Active** | `persistence.rs` test |
| Bottleneck / Wasserstein / landscapes / images | **Active** | **17 tests** |
| Sparse attention reference kernel | **Active** | **29 contracts** |
| Scheduled attention (Rust rebuild of the Triton kernel) | **Active** | **16 tests** |
| Same-budget random and oracle schedules | **Active** | 9 tests, `ablation_baselines.rs` |
| Scheduled-attention backward pass | **Active** | 5 tests, central finite differences, `attention_backward.rs`; `aether_core::attention` itself is forward only |
| Activation derivatives, softmax-layer gradient | **Active** | 7 tests, `activation_contracts.rs` |
| Linking number, writhe, knot determinant (`linking`, [`linking` docs](https://teerthsharma.github.io/Aether-Lang/integrated/linking/)) | **Active** | 18 tests |
| Rounding certificates (`certify`, [`certify` docs](https://teerthsharma.github.io/Aether-Lang/integrated/certify/)) | **Active** | 17 tests, checked against exact `i128` arithmetic |
| Planar arrangement invariants (`arrangement`, [`arrangement` docs](https://teerthsharma.github.io/Aether-Lang/integrated/arrangement/)) | **Active** | 18 tests |
| Resolvent attention operator (`resolvent`, [`resolvent` docs](https://teerthsharma.github.io/Aether-Lang/integrated/resolvent/)) | **Active** | 17 tests, each mirroring a Lean-proved identity |
| Orbit-partition bounds (`orbit`, [`orbit` docs](https://teerthsharma.github.io/Aether-Lang/integrated/orbit/)) | **Active** | 15 tests |
| Injectivity, symmetry, dimension, Lyapunov (`monodromy`, [`monodromy` docs](https://teerthsharma.github.io/Aether-Lang/integrated/monodromy/)) | **Active** | 18 tests against closed-form answers |
| Cell tracking with division certificate (`track`, [`track` docs](https://teerthsharma.github.io/Aether-Lang/integrated/track/)) | **Active** | 15 tests |
| Coupling operator, fixed point, islands (`coupling`, [`coupling` docs](https://teerthsharma.github.io/Aether-Lang/integrated/coupling/)) | **Active** | 18 tests |
| Segment witnesses for sparse top-k (`kvwitness`, [`kvwitness` docs](https://teerthsharma.github.io/Aether-Lang/integrated/kvwitness/)) | **Active** | 15 tests; no quality evaluation |
| Offset planning, transitive reduction, islands (`planner`, [`planner` docs](https://teerthsharma.github.io/Aether-Lang/integrated/planner/)) | **Active** | 15 tests |
| Scale past 32 points | **Active** | **7 tests** |
| `no_std` on a real embedded target | **Active** | builds `thumbv7m-none-eabi` |
| Kernel compiles bare metal | **Active** | builds `x86_64-unknown-none` |
| Titan VM language parity | **Partial** | reshape in progress (§4.8): fails closed on what it cannot compile; parity goldens and the ≥3× benchmark gate pending |
| Static type checking | **Partial** | checker exists, diagnostics thin |
| Seal loop spelled `until convergence(ε)` | **Active** | max-norm tolerance on the body's value; `aether-lang/tests/seal_convergence.rs` (§4.6). Broken until this revision |
| Seal loop spelled `until stable(expr)` | **Active** | exact-equality stability of any number, list or record; `aether-lang/tests/integrated_modules.rs` |
| `ConvergenceCond::BettiStable` | **Not wired** | declared in the AST, never constructed by the parser (§4.7) |
| Sparse scheduler | **Ungated** | 4 tests exist and **never execute** — `no_std` bin, no test harness |
| Kernel *boots* | **Ungated** | compiles ≠ boots; needs QEMU logs |
| External TDA parity | **Ungated** | no ripser/GUDHI fixture comparison |
| Lean 4 formalization | **Ungated** | 11,637 lines, 48 theorems, **no `lake build` in CI** |
| GPU compute backend (`aether-gpu`) | **Hardware-gated** | **107 tests**, RTX 4060 / Vulkan. `cargo test -p aether-gpu --features gpu --release`. In CI the gated ones report as **ignored**, not passed |
| GPU used by `aether-core` | **Ungated** | nothing routes through it; cost and precision measured, integration not made |
| Wall-clock speedup claims | **Withdrawn** | see [Section 8](#8-negative-results-what-we-got-wrong) |

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WORKSPACE GATE                        branch master, nightly, Win11
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  cargo fmt --all -- --check                                   clean
  cargo clippy -D correctness -D suspicious                     clean
  cargo test --workspace --exclude aether-kernel   428 passed 80 ignored
  cargo build -p aether-kernel --target x86_64-unknown-none        ok
  cargo build -p aether-core  --target thumbv7m-none-eabi          ok
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Rust lines (crates/)                                        53,228
  Lean lines (Aether/)              11,637   theorems 48   sorry 0
  Test suites gated in CI                                          7
  Claims withdrawn during audit                                    6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The 428 passing tests divide as: `aether-core` 330 (56 unit, 274 integration, of which 166 belong to the certified library), `aether-lang` 64 (32 unit, 23 for the module bindings, 8 for the tolerance seal loop, literals and grouping, 1 running the opening program), `aether-gpu` 31 that need no adapter, `aegis-core` 2, `aether-cli` 1.

### 1.6 Reading guide

Different readers want different files.

**Evaluating whether the mathematics is right.** Start at [`persistence_invariants.rs`](crates/aether-core/tests/persistence_invariants.rs): 12 property tests, of which the two that matter most are `bottleneck_distance_respects_the_stability_bound` (§3.10) and `gaussian_noise_produces_no_long_h1_bar` (the negative control). Then `persistence.rs` itself, read against §3.1–§3.9. Then note what is absent: no comparison against ripser or GUDHI exists, so every assurance here is internal.

**Evaluating whether the engineering is sound.** [`attention_contracts.rs`](crates/aether-core/tests/attention_contracts.rs), 1,255 lines, is the most informative file in the repository — not because the code is best there, but because the shape of the file records a claim being repaired three times, with the cost model that finally falsified it added at the end.

**Deciding whether to use it.** Read [Section 9](#9-limitations) first, then [Section 2.2](#22-prior-art). For production TDA the answer is ripser, and this document says so in four places.

**Looking for the transferable idea.** [Section 3.22](#322-the-routing-gap-ratio): the $H_0$ barcode separates key distributions on which topological routing pays from those on which it does not, and is cheap enough to consult before committing. That is the result here most likely to generalise past this repository.

**Wanting to contribute.** [`docs/reference/status.md`](docs/reference/status.md), then [Contributing](#contributing). The top two items are worth more than everything else combined.

**Judging the author.** [Section 8](#8-negative-results-what-we-got-wrong) records six claims withdrawn on their own measurements, two tests that were wrong rather than the code, two phantom dependencies found by reading `Cargo.toml` instead of the sentence beside it, and the corrections made while re-deriving the mathematics for this revision. That section is the argument; the green checkmarks are not.

---

## 2. Background and prior art

### 2.1 Scalar and structural stopping criteria

**Scalars are noisy where structure is not.** A loss oscillating in the third decimal forces a patience parameter or a moving average. $\beta_1$ reading 3, 3, 3, 1 is an integer sequence with one event in it.

**Scalars are one-dimensional.** A persistence diagram retains multi-scale structure. Two residual fields with identical $L_2$ norms can have entirely different topology, and when the question is whether a model has found the *shape* of the data, the norm answers a different question from the one asked.

**Discrete invariants compose.** Betti numbers are homotopy invariants and, through the stability theorem (§3.10), the diagrams they are read from move by at most the size of the perturbation. That combination — integer-valued readings from an object that is Lipschitz in the input — is what a stopping rule wants.

**The cost is knowable before it is paid.** For $n$ points and homology through dimension $K$, the Rips complex has at most $\sum_{k=0}^{K+1}\binom{n}{k+1}$ simplices (§3.2). That arithmetic is available before allocation, which is why the engine accepts a budget and refuses rather than degrades.

**Topological convergence does not remove tuning.** It moves it. A Betti vector that repeats once is not a fixed point, so any rule on Betti numbers needs a stability window — a discrete count in place of a continuous $\varepsilon$ (§3.29). An integer window is easier to reason about than a tolerance that interacts with the scale of the loss, and the claim should be read as that and no more.

### 2.2 Prior art

Every system below is real, currently maintained, and better than this one at what it was built for.

| System | Language | $H_0$/$H_1$/$H_2$ | Metrics | Vectorisations | `no_std` | Topology as control flow | Maturity |
|---|---|---|---|---|---|---|---|
| [ripser](https://github.com/Ripser/ripser) | C++ | yes (Rips, fast) | no | no | no | no | Production |
| [GUDHI](https://gudhi.inria.fr/) | C++/Python | yes (many complexes) | yes | yes | no | no | Production |
| [giotto-tda](https://github.com/giotto-ai/giotto-tda) | Python/C++ | yes | yes | yes | no | no | Production |
| [Dionysus](https://www.mrzv.org/software/dionysus2/) | C++/Python | yes | yes | partial | no | no | Mature |
| **Aether-Lang** | Rust | yes (Rips + witness) | yes | yes | yes | yes | **Research** |

**There is deliberately no timing column.** ripser and GUDHI have not been run on the same hardware with the same inputs, and a row reading "12× faster than ripser" is exactly the class of claim this repository spent an audit removing. When a parity harness exists, numbers go here. Until then:

> Aether-Lang's persistence engine has **not** been benchmarked against ripser, GUDHI, giotto-tda or Dionysus, and has **not** been verified to agree with them on shared fixtures. Its correctness evidence is internal invariants plus mutation testing.

The measured ceiling (§6.3: $H_1$ at $n=300$ in 131 s) strongly suggests the engine is **substantially slower** than ripser, which routinely handles clouds orders of magnitude larger. That is an inference, not a benchmark. The engine performs the textbook reduction (§3.5) without the clearing, cohomology and apparent-pair optimisations ripser is built on; it trades throughput for `no_std` and exactness under a budget. If that is not the trade required, use ripser.

### 2.3 Differentiators

| Capability | Aether-Lang | ripser | GUDHI | giotto-tda |
|---|---|---|---|---|
| Persistent homology $H_0$/$H_1$/$H_2$ | yes | yes | yes | yes |
| Bottleneck / Wasserstein distance | yes | no | yes | yes |
| Landscapes / images | yes | no | yes | yes |
| **Topology as a language primitive** | yes | no | no | no |
| **Loop termination on a topological predicate** | yes (§3.29) | no | no | no |
| **`no_std`, runs with no OS** | yes | no | no | no |
| **Bare-metal x86_64 kernel** | yes (compiles) | no | no | no |
| Core math dependencies | **3 declared** (`libm`, `heapless`, `nalgebra`), 2 used | many | many | many |
| Mature, fast, widely used | no | yes | yes | yes |
| Verified against the others | **not yet** | — | — | — |

The last row is the important one and is deliberately in the table. It is gated in the claim ledger as the single largest correctness debt in this repository.

---

## 3. Theoretical foundation

Every result in this section is implemented, and each names the function that implements it as `file.rs` → `fn name` (paths under `crates/aether-core/src/` unless stated). Where the code approximates, the approximation and its error bound are stated with the result. Where the code does something other than the textbook construction, the difference is stated rather than smoothed over. Nothing here is aspirational: a formula that does not appear in the source does not appear in this section.

**Notation.** $X = \lbrace x_1,\dots,x_n\rbrace $ is a finite point set with a dissimilarity $d$ (Euclidean on $\mathbb{R}^D$ unless a distance matrix is supplied). $\mathbb{F}_2 = \mathbb{Z}/2$. A simplex $\sigma$ is a non-empty vertex set with $\dim\sigma = \lvert\sigma\rvert - 1$. $K$ denotes both a simplicial complex and, where it is an integer, `max_homology_dim`; context disambiguates. $c_k$ is the number of $k$-simplices, $m$ the total number of simplices, $b$ the number of bars in a diagram. $R$ is `max_radius`.

| § | Result | Implemented in | Evidence |
|---|---|---|---|
| 3.1 | Rips filtration, flag property, face-first order | `persistence.rs` → `fn build_vietoris_rips_from_distances`, `fn compare_simplex` | `every_face_is_present_and_precedes_its_coface` |
| 3.2 | Simplex-count bound; presets never reach their simplex cap | `fn validate_point_cap`, `fn push_simplex` | `simplex_cap_still_fails_fast_rather_than_exhausting_memory` |
| 3.3 | $\partial\circ\partial = 0$ over $\mathbb{F}_2$ | `fn boundary_indices`, `fn xor_sorted` | `boundary_of_boundary_is_zero_over_z2` |
| 3.4 | $\beta_k$ as ranks; Euler–Poincaré | `fn reduce_z2`; `manifold.rs` → `fn estimate_betti_1` | §3.6, §3.17 |
| 3.5 | Column reduction and the pairing lemma | `fn reduce_z2` | `reduced_columns_have_distinct_lowest_ones`, mutation 7/12 |
| 3.6 | Diagram size on the complete complex | `fn reduce_z2` | `scale_probe` output, all 10 rows |
| 3.7 | $\beta_k(r)$ from a diagram | `PersistenceDiagram::betti_at` | `h0_tracks_component_merges` |
| 3.8 | $H_0$ deaths = MST weights | `attention.rs` → `fn single_linkage_clusters` | `h0_matches_an_independent_union_find` |
| 3.9 | Lazy witness filtration | `fn select_landmarks`, `fn witness_filtration` | `witness_mode_uses_landmarks_without_rejecting_full_signal_size` |
| 3.10 | Stability, $d_B \le 2\varepsilon$ | (property of 3.1 + 3.5) | `bottleneck_distance_respects_the_stability_bound` |
| 3.11 | Exact $d_B$ and $W_p$ | `diagram.rs` → `fn bottleneck_distance`, `fn wasserstein_distance` | 8 metric tests |
| 3.12 | Landscapes, 1-Lipschitz | `fn persistence_landscape` | `landscape_is_one_lipschitz_in_the_bottleneck_distance` |
| 3.13 | Total persistence, entropy | `fn total_persistence`, `fn persistent_entropy` | no dedicated test |
| 3.14 | Persistence images, midpoint error | `fn persistence_image` | 4 image tests |
| 3.15 | Regular-polygon chord | (property of 3.1 + 3.5) | `circle_h1_dies_at_the_exact_regular_polygon_chord` |
| 3.16 | Delay embedding | `manifold.rs` → `TimeDelayEmbedder::embed` | `test_embedding` |
| 3.17 | Graph Betti numbers, cycle rank | `SparseAttentionGraph::{compute_betti_0, estimate_betti_1}` | `test_single_component` |
| 3.18 | Stable softmax, online recurrence | `attention.rs` → `fn sparse_attention`; `scheduled.rs` → `fn scheduled_attention` | `large_logits_do_not_overflow_the_softmax` |
| 3.19 | Attention backward pass | `scheduled.rs` → `fn scheduled_attention_backward` | 5 finite-difference tests |
| 3.20 | Block salience; oracle optimality | `fn block_salience`, `fn oracle_block_schedule` | `no_schedule_recovers_more_mass_than_the_oracle` |
| 3.21 | Placement statistic | `attention.rs` → `fn attention_mass_recovered` | `the_topological_selector_is_placed_on_the_random_to_oracle_axis` |
| 3.22 | Routing gap ratio | `fn routing_plan` | `the_h0_barcode_alone_separates_the_two_regimes` |
| 3.23 | Admissible pruning bound | `aether.rs` → `BlockMetadata::upper_bound_score` | none beyond unit tests |
| 3.24 | Drift as a second difference | `DriftDetector::update` | `test_drift_detector` |
| 3.25 | Chebyshev–Cantelli guard | `memory.rs` → `ChebyshevGuard::is_safe` | none |
| 3.26 | Governor law and its stability condition | `governor.rs` → `GeometricGovernor::adapt` | clamp tests only |
| 3.27 | Ring gossip consensus rate | `ml/gossip.rs` → `GossipRing::tick` | in-module test |
| 3.28 | Initialisation, optimisers, boosting, regression | `ml/` | `activation_contracts.rs` |
| 3.29 | The convergence predicates actually implemented | `aether-lang` interpreter; `ml/convergence.rs` | interpreter tests |

### 3.1 The Vietoris–Rips filtration

For a finite $X$ with dissimilarity $d$, the Vietoris–Rips complex at scale $r \ge 0$ and its filtration function are

$$
\mathrm{VR}_r(X) \;=\; \bigl\lbrace \, \sigma \subseteq X,\ \sigma \neq \varnothing \;:\; f(\sigma) \le r \,\bigr\rbrace ,
\qquad
f(\sigma) \;=\; \max_{u,v \in \sigma} d(u,v),
$$

with $f(\lbrace x\rbrace ) = 0$ for vertices. $f(\sigma)$ is the diameter of $\sigma$ and is the value at which $\sigma$ enters the filtration.

The code relies on four properties.

**(i) Monotonicity.** If $\tau \subseteq \sigma$ then $f(\tau) \le f(\sigma)$, since a maximum over a subset of pairs cannot exceed the maximum over all of them. Hence $r \le r'$ implies $\mathrm{VR}_r \subseteq \mathrm{VR}_{r'}$, and $\lbrace \mathrm{VR}_r\rbrace _{r \ge 0}$ is a filtration.

**(ii) Flag property.** $\sigma \in \mathrm{VR}_r$ if and only if every edge of $\sigma$ is in $\mathrm{VR}_r$: $\mathrm{VR}_r$ is the clique complex of its 1-skeleton. The engine therefore never searches for simplices; it evaluates $f$ as a maximum over the $\binom{\lvert\sigma\rvert}{2}$ stored pairwise entries (`max3` for triangles, `max6` for tetrahedra).

**(iii) Truncation.** To compute $H_0,\dots,H_K$ with $K = $ `max_homology_dim` $\le 2$, only the $(K+1)$-skeleton is built, and a simplex is admitted only if $f(\sigma) \le R$. Nothing below $R$ is lost: $H_k$ of a complex depends only on its $(k+1)$-skeleton, because $\ker\partial_k$ lives in the $k$-chains and $\operatorname{im}\partial_{k+1}$ in the $(k+1)$-chains.

**(iv) A face-first total order.** Simplices are sorted by the key $\bigl(f(\sigma),\ \dim\sigma,\ \text{vertex list}\bigr)$, compared lexicographically with `f64::total_cmp` on the first component. If $\tau \subsetneq \sigma$ then $f(\tau) \le f(\sigma)$ and $\dim\tau < \dim\sigma$, so $\tau$ precedes $\sigma$ even when filtration values tie. This is the precondition of the reduction in §3.5: the boundary matrix in this order is strictly upper triangular.

**Implementation.** `persistence.rs` → `fn build_vietoris_rips_from_distances`, `fn compare_simplex`. The Euclidean entry point `fn persistent_homology` fills the $n \times n$ distance matrix and delegates; `fn persistent_homology_from_distances` accepts any matrix that is square, symmetric, zero on the diagonal, non-negative and finite, and rejects a violation with `InvalidRadius`. The triangle inequality is neither required nor checked, since none of (i)–(iv) uses it.

**Evidence.** `every_face_is_present_and_precedes_its_coface` asserts (iv) and the presence of every face on five complexes, one of them a witness complex; `an_edge_just_beyond_the_radius_cap_is_excluded` places one distance exactly on the cap and one $5\times10^{-4}$ above it, pinning that the comparison $f(\sigma) \le R$ carries no slack.

### 3.2 Complex size and the fail-fast budget

The $(K+1)$-skeleton of the simplex on $n$ vertices has

$$
\lvert K_{\le K+1} \rvert \;=\; \sum_{k=0}^{K+1} \binom{n}{k+1}
$$

simplices, and the engine's complex is a subcomplex of it, with equality when $R = \infty$. For $K = 2$ this is $n + \binom{n}{2} + \binom{n}{3} + \binom{n}{4}$, whose $O(n^4)$ term is why `h2_default` caps at 48 points. Evaluated at each shipped preset with $R = \infty$:

| Preset | $K$ | Point cap | Full complex at the cap | `max_simplices` |
|---|---:|---:|---:|---:|
| `h2_default()` | 2 | 48 | $48 + 1{,}128 + 17{,}296 + 194{,}580 = 213{,}052$ | 1,000,000 |
| `h1_dense()` | 1 | 128 | $128 + 8{,}128 + 341{,}376 = 349{,}632$ | 1,000,000 |
| `h0_only()` | 0 | 512 | $512 + 130{,}816 = 131{,}328$ | 1,000,000 |
| `low_load()` | 1 | 24 landmarks | $24 + 276 + 2{,}024 = 2{,}324$ | 4,096 |

Every preset's point cap therefore binds before its simplex cap: at the cap and infinite radius, no preset reaches `max_simplices`. The simplex cap is reached only when a caller raises the point cap without raising the simplex cap — which the language does by default (§4.5). Edges are built even when $K = 0$, which is why `h0_only` counts $\binom{n}{2}$.

**Enforcement.** `fn validate_point_cap` rejects the cloud with `TooManyPoints` before any distance is computed. `fn push_simplex` refuses the simplex that would exceed the cap with `TooManySimplices`, so the simplex vector never holds more than `max_simplices` entries and memory is bounded by the cap rather than by $n$. A refused filtration is never truncated: a diagram of a partial complex is a valid-looking answer about a shape nobody asked about.

### 3.3 Chains and the boundary operator over F2

$C_k(K;\mathbb{F}_2)$ is the $\mathbb{F}_2$-vector space with the $k$-simplices as basis. The boundary operator is

$$
\partial_k\,[v_0,\dots,v_k] \;=\; \sum_{i=0}^{k}\,[v_0,\dots,\widehat{v_i},\dots,v_k],
$$

where $\widehat{v_i}$ marks the omitted vertex. Over $\mathbb{F}_2$ no orientation signs are needed.

**Proposition.** $\partial_{k-1}\circ\partial_k = 0$.

*Proof.* A $(k-2)$-face of $[v_0,\dots,v_k]$ is obtained by deleting two vertices $v_i, v_j$. It arises exactly twice in $\partial\partial$ — delete $v_i$ then $v_j$, or $v_j$ then $v_i$ — and $1 + 1 = 0$ in $\mathbb{F}_2$. $\square$

**Implementation.** A chain is a sorted list of simplex indices, and addition in $C_k$ is symmetric difference: `fn xor_sorted` merges two sorted lists and drops common entries in $O(\lvert a\rvert + \lvert b\rvert)$. `fn boundary_indices` builds the $k+1$ faces by deleting one vertex at a time, zero-padding each into the canonical key `([usize; 4], len)`, and resolves it through a `BTreeMap` in $O(\log m)$. A guard keeps only faces whose index is below the coface's; it never fires on a well-formed filtration, by §3.1 (iv), and exists so that a malformed complex degrades rather than lies.

**Evidence.** `boundary_of_boundary_is_zero_over_z2` evaluates $\partial\partial\sigma$ for every simplex of five complexes. The identity fails *silently* — every rank downstream becomes meaningless without a crash — which is the whole argument for testing an identity true by construction.

### 3.4 Homology, Betti numbers and the Euler–Poincaré relation

$$
H_k(K) \;=\; \ker\partial_k \,/\, \operatorname{im}\partial_{k+1},
\qquad
\beta_k \;=\; \dim H_k \;=\; \dim\ker\partial_k - \operatorname{rank}\partial_{k+1} \;=\; c_k - \operatorname{rank}\partial_k - \operatorname{rank}\partial_{k+1}.
$$

$\beta_0$ counts connected components, $\beta_1$ independent loops, $\beta_2$ enclosed voids.

**Proposition (Euler–Poincaré).**

$$
\chi(K) \;=\; \sum_{k} (-1)^k c_k \;=\; \sum_{k} (-1)^k \beta_k .
$$

*Proof.* By rank–nullity, $c_k = \dim\ker\partial_k + \operatorname{rank}\partial_k$. Substituting $\dim\ker\partial_k = \beta_k + \operatorname{rank}\partial_{k+1}$ gives $\sum_k(-1)^k c_k = \sum_k(-1)^k\beta_k + \sum_k(-1)^k\bigl(\operatorname{rank}\partial_{k+1} + \operatorname{rank}\partial_k\bigr)$, and the last sum telescopes to zero. $\square$

**Where the code uses it.** After the reduction of §3.5, $\operatorname{rank}\partial_k$ equals the number of non-zero reduced columns of dimension $k$, so every $\beta_k$ is read off the pairing without a separate rank computation. The one-dimensional instance $\beta_1(G) = \lvert E\rvert - \lvert V\rvert + \beta_0(G)$ is exactly `estimate_betti_1` (§3.17). The instance on the complete complex fixes the size of every diagram the scale probe prints (§3.6).

### 3.5 Persistence by column reduction

Index the simplices $\sigma_1,\dots,\sigma_m$ in the order of §3.1 (iv) and let $D \in \mathbb{F}_2^{m\times m}$ be the boundary matrix, $D_{ij} = 1$ iff $\sigma_i$ is a codimension-one face of $\sigma_j$. For a non-zero column $R_j$ write $\mathrm{low}(j) = \max\lbrace \, i : R_{ij} = 1 \,\rbrace $. The standard reduction is

$$
\text{for } j = 1,\dots,m:\qquad
\text{while } R_j \neq 0 \ \text{and}\ \exists\, j' < j \text{ with } \mathrm{low}(j') = \mathrm{low}(j):\qquad
R_j \leftarrow R_j + R_{j'} .
$$

It terminates with $R = DV$, where $V$ is upper unitriangular (only earlier columns are ever added to later ones), and with $\mathrm{low}$ injective on the non-zero columns of $R$.

**Pairing lemma.** For the reduced $R$:

1. $R_j = 0$ if and only if $\sigma_j$ is *positive*: its entry creates a homology class of dimension $\dim\sigma_j$.
2. If $\mathrm{low}(j) = i$, then $\sigma_i$ is positive, $\sigma_j$ is *negative*, and the class created by $\sigma_i$ is destroyed by $\sigma_j$. This contributes the interval $[\,f(\sigma_i),\ f(\sigma_j)\,)$ to the diagram in dimension $\dim\sigma_i = \dim\sigma_j - 1$.
3. A positive $\sigma_i$ that is no column's low is *essential*: it contributes $[\,f(\sigma_i),\ \infty)$.

Every simplex is exactly one of: paired positive, negative, essential. The pairing does not depend on which sequence of column additions produced $R$, because it is determined by ranks of submatrices of $D$ alone (Edelsbrunner & Harer 2010, Ch. VII): writing $r_D(i,j)$ for the rank of $D$ restricted to rows $i,\dots,m$ and columns $1,\dots,j$,

$$
\mathrm{low}_R(j) = i \iff r_D(i,j) - r_D(i+1,j) - r_D(i,j-1) + r_D(i+1,j-1) = 1 .
$$

**Implementation.** `persistence.rs` → `fn reduce_z2`. The array `low_owner[i]` records the unique column whose lowest entry is $i$; the inner loop adds `reduced_columns[owner]` until the current low is unowned, so injectivity of $\mathrm{low}$ is an invariant maintained at every step rather than a post-condition checked afterwards. Pairs in dimensions above $K$ are discarded. Zero-length pairs, $f(\sigma_i) = f(\sigma_j)$, are retained — a cloud with duplicate points has $[0,0)$ bars in its diagram — and essential classes carry `death: None`, which is a different statement from dying at infinity. The reduction cannot fail on a well-formed complex, which is why `PersistenceError` has no variant for it.

**Cost.** $O(m^3)$ in the worst case. Not implemented: clearing (Chen & Kerber 2011), reduction of the coboundary instead of the boundary (de Silva, Morozov & Vejdemo-Johansson 2011), and apparent pairs (Bauer 2021) — the three optimisations that account for most of ripser's throughput.

**Evidence.** `reduced_columns_have_distinct_lowest_ones` asserts the observable consequence (no negative-length bar) on five complexes; the mutant "column reduction terminates after one operation" is caught by 7 of the 12 invariant tests (§7.3); `h2_tetrahedron_boundary_has_void_until_tetrahedron_enters` checks the $H_2$ pairing on the smallest void.

### 3.6 Diagram size on the complete complex

With $R = \infty$ the engine's complex is the full $(K+1)$-skeleton of the simplex on $n$ vertices. Its reduced homology vanishes in degrees $0,\dots,K$, and that alone fixes the number of pairs, independently of the geometry.

**Proposition.** With $R = \infty$, the diagram through dimension $K$ has

$$
\lvert\mathrm{Dgm}_{\le 0}\rvert = n,
\qquad
\lvert\mathrm{Dgm}_{\le 1}\rvert = \binom{n}{2} + 1,
\qquad
\lvert\mathrm{Dgm}_{\le 2}\rvert = \binom{n}{3} + n
$$

pairs, counting essential and zero-length pairs.

*Proof.* Let $p_k$ and $q_k$ be the numbers of positive and negative $k$-simplices, so $p_k + q_k = \binom{n}{k+1}$. Vertices are all positive: $p_0 = n$. Exactly one $H_0$ class survives, so $q_1 = n - 1$. For $1 \le k \le K$ every class born in degree $k$ dies, since $\tilde H_k$ of the $(K+1)$-skeleton is zero, and each dies at exactly one negative $(k+1)$-simplex: $q_{k+1} = p_k$. Each positive simplex of dimension at most $K$ contributes exactly one pair, so $\lvert\mathrm{Dgm}_{\le K}\rvert = \sum_{k \le K} p_k$. Then $p_1 = \binom{n}{2} - n + 1$ and $p_2 = \binom{n}{3} - p_1$, and summing gives the three forms. $\square$

The measured scale table (§6.3) prints `d.pairs.len()` with $R = \infty$, and every one of its ten rows equals these closed forms — for example $\binom{300}{2} + 1 = 44{,}851$ and $\binom{70}{3} + 70 = 54{,}810$. The count is a free consistency check on the reduction: a dropped or duplicated pair changes it, while geometry cannot. It is not asserted by a test. The diagram of the quick-start program has $16 = \binom{6}{2} + 1$ pairs for the same reason.

### 3.7 Betti numbers from a diagram

$$
\beta_k(r) \;=\; \#\bigl\lbrace \, (b,d) \in \mathrm{Dgm}_k \;:\; b \le r < d \,\bigr\rbrace ,
\qquad d = \infty \text{ for essential classes.}
$$

By the structure theorem for persistence modules over a field (Zomorodian & Carlsson 2005), the persistent homology of a finite filtration decomposes into interval modules, one per bar, and consequently $\beta_k(r) = \dim H_k(K_r)$ with $K_r = \lbrace \sigma : f(\sigma) \le r\rbrace $, for every $r \le R$. The half-open convention matches the code, which counts a pair when `birth <= radius && radius < death`. Above $R$ the truncated complex is final, so a class that would die beyond the cap is reported alive there.

**Implementation.** `persistence.rs` → `PersistenceDiagram::betti_at`, $O(b)$: a query on a computed diagram, not a recomputation. This is why `topology.betti(diagram, radius=r)` takes a diagram rather than a point cloud — computing once and querying at many radii is the cheap direction, and the API makes it the obvious one.

### 3.8 H0, single linkage and the elder rule

**Proposition.** The finite $H_0$ deaths of $\mathrm{VR}(X)$ are, as a multiset, the edge weights of a minimum spanning tree of the complete graph on $X$ weighted by $d$.

*Proof.* An edge is negative in the sense of §3.5 exactly when, at its entry, its endpoints lie in different components of the current 1-skeleton; otherwise it closes a cycle and is positive. Scanning edges in filtration order and keeping exactly the negative ones is Kruskal's algorithm, so the negative edges form a minimum spanning tree and their filtration values are the deaths. $\square$

Single-linkage clustering is the same construction (Gower & Ross 1969): its merge heights are the MST weights in increasing order, and cutting after $n - c$ merges leaves $c$ clusters.

**The elder rule.** When two components merge, the younger dies and the elder survives. In a Rips filtration every vertex is born at 0, so "younger" is decided by a tie-break; only the multiset of death *values* is canonical, not which vertex is credited with each. §3.20 is where this distinction stops being academic.

**Implementation.** `attention.rs` → `fn single_linkage_clusters`: Kruskal with iterative path-compressed union-find, edges sorted by $(d, i, j)$ so the merge order does not depend on sort stability, labels canonicalised to first-occurrence order so two runs are comparable. `scheduled.rs` → `fn block_salience` applies the same scan to block centroids (§3.20).

**Evidence.** `h0_matches_an_independent_union_find` (to 1e-9) and `single_linkage_merge_heights_equal_the_h0_persistence_deaths` — cross-validation between independent code paths that must agree, the most valuable kind of test available in a crate that implements one theorem twice for different callers.

### 3.9 The lazy witness filtration

Landmarks $L = \lbrace \ell_1,\dots,\ell_{\lvert L\rvert}\rbrace  \subseteq X$ are chosen by maxmin (farthest-point) selection starting from $x_1$:

$$
\ell_1 = x_1,
\qquad
\ell_{j+1} \;=\; \operatorname*{arg\,max}_{x \in X \setminus L_j}\ \min_{\ell \in L_j} d(x,\ell).
$$

Every point of $X$ is a witness. With $m_w = \min_{\ell\in L} d(w,\ell)$ the distance from witness $w$ to its nearest landmark, a simplex $\sigma \subseteq L$ with $\lvert\sigma\rvert \ge 2$ enters at

$$
f_W(\sigma) \;=\; \min_{w \in X}\ \max\Bigl(0,\ \max_{v \in \sigma} d(w,v) - m_w\Bigr),
$$

and landmarks enter at 0. This is the $\nu = 1$ lazy witness value of de Silva & Carlsson (2004). The clamp at 0 is inert, since $d(w,v) \ge m_w$ for every landmark $v$.

**Where the code differs from the textbook.** The lazy witness complex is the flag completion of its witnessed 1-skeleton. The code instead evaluates $f_W$ directly on every simplex up to dimension $K+1$. Monotonicity survives — for fixed $w$ the inner maximum is monotone in $\sigma$, and a minimum of monotone functions is monotone — so §3.1 (iv) applies unchanged. Direct evaluation dominates the flag value, $f_W(\sigma) \ge \max_{e \subseteq \sigma} f_W(e)$, so at every scale the code's complex is a subcomplex of the lazy witness complex, equal to it on the 1-skeleton.

**Cost.** Maxmin selection costs $O(n\lvert L\rvert^2)$ distance evaluations; each candidate simplex scans all $n$ witnesses, so enumeration costs $O\bigl(n\,\lvert L\rvert^{K+2}\bigr)$. The combinatorial term depends on the landmark count rather than on $n$, which is what makes `low_load()` viable at 24 landmarks on a Cortex-M3. The point cap applies to the landmark count, not to $n$.

**Approximation.** None is bounded. The witness diagram is a diagram of the landmark set as witnessed by the rest, not of $X$; no theorem relating it to $\mathrm{Dgm}\,\mathrm{VR}(X)$ is implemented, and no test asserts agreement.

### 3.10 The stability theorem

**Theorem** (Cohen-Steiner, Edelsbrunner & Harer 2007). For tame functions $f, g$ on the same finite simplicial complex,

$$
d_B\bigl(\mathrm{Dgm}_k f,\ \mathrm{Dgm}_k g\bigr) \;\le\; \lVert f - g \rVert_\infty .
$$

**Corollary** (the form the suite asserts). Let $Y = \lbrace y_1,\dots,y_n\rbrace $ with $\lVert y_i - x_i\rVert \le \varepsilon$ for every $i$, and $R = \infty$. Then

$$
d_B\bigl(\mathrm{Dgm}_k\,\mathrm{VR}(X),\ \mathrm{Dgm}_k\,\mathrm{VR}(Y)\bigr) \;\le\; 2\varepsilon .
$$

*Derivation.* Both filtrations live on the same abstract complex, the $(K+1)$-skeleton of the simplex on $n$ labelled vertices. By the triangle inequality $\lvert d(y_i,y_j) - d(x_i,x_j)\rvert \le \lVert y_i - x_i\rVert + \lVert y_j - x_j\rVert \le 2\varepsilon$; a maximum of pairwise terms inherits the bound, so $\lVert f_X - f_Y\rVert_\infty \le 2\varepsilon$, and the theorem applies. $\square$ Through the Gromov–Hausdorff distance the same constant appears as $d_B \le 2\,d_{GH}(X,Y)$ (Chazal, de Silva & Oudot 2014).

The constant is $2\varepsilon$. An assertion at $\varepsilon$ fails spuriously; one at $4\varepsilon$ passes a broken implementation.

**Scope of the check.** `bottleneck_distance` compares finite bars only. That is sound for the tested configuration, because both diagrams carry the same essential classes (one in $H_0$, none in $H_1$ on the complete 2-skeleton). With a finite $R$ the two complexes can differ and the corollary does not apply.

**Evidence.** `bottleneck_distance_respects_the_stability_bound` perturbs a 12-point noisy circle by displacements drawn uniformly from the disc of radius $\varepsilon$ and asserts $d_B \le 2\varepsilon + 10^{-9}$ over 12 seed × $\varepsilon$ combinations. It is the single most valuable test in the repository: wrong pairing, a dropped bar, a mishandled infinite death and an early-terminating reduction all surface here and nowhere else. `distances_respect_the_stability_theorem_on_real_diagrams` repeats the check through `diagram.rs`.

### 3.11 Bottleneck and Wasserstein distances

Treat a diagram as a finite multiset of points $(b,d)$ and let $\Delta$ be the diagonal. With the $L_\infty$ ground cost

$$
c\bigl((b,d),(b',d')\bigr) = \max\bigl(\lvert b-b'\rvert,\ \lvert d-d'\rvert\bigr),
\qquad
c\bigl((b,d),\Delta\bigr) = \tfrac{1}{2}(d-b),
\qquad
c(\Delta,\Delta) = 0,
$$

the two distances are

$$
d_B(D_1,D_2) \;=\; \min_{\eta}\ \max_{x}\ c\bigl(x,\eta(x)\bigr),
\qquad
W_p(D_1,D_2) \;=\; \Bigl(\min_{\eta}\ \sum_{x} c\bigl(x,\eta(x)\bigr)^{p}\Bigr)^{1/p},
\quad p \ge 1,
$$

over bijections $\eta$ between $D_1 \cup \Delta$ and $D_2 \cup \Delta$. $\tfrac12(d-b)$ is the $L_\infty$ distance from $(b,d)$ to the nearest diagonal point.

**Finite reduction.** With $\lvert D_1\rvert = n$ and $\lvert D_2\rvert = m$, build the $(n+m)\times(n+m)$ matrix $C$ whose rows are the points of $D_1$ followed by $m$ diagonal slots and whose columns are the points of $D_2$ followed by $n$ diagonal slots, with entries given by $c$. Perfect matchings of $C$ are exactly the admissible $\eta$, so $d_B$ is a bottleneck assignment and $W_p^p$ a linear assignment on the entrywise power $C^{\circ p}$.

**Exact algorithms.** The bottleneck value of an assignment is one of its entries, so $d_B$ lies in the finite set of distinct entries of $C$. Feasibility of a threshold $t$ — a perfect matching in the bipartite graph $\lbrace C_{ij} \le t\rbrace $ — is monotone in $t$, so binary search over the sorted candidates is exact: $O(\log(n+m))$ feasibility tests, each by Kuhn's augmenting paths in $O\bigl((n+m)^3\bigr)$. $W_p$ is solved by the Hungarian algorithm in its potentials form, $O\bigl((n+m)^3\bigr)$.

**Ordering.** For any fixed matching $\bigl(\sum c^p\bigr)^{1/p} \ge \max c$, so $W_p \ge d_B$; $\lVert\cdot\rVert_p$ is non-increasing in $p$, so $W_p$ is too, and $W_p \to d_B$ as $p \to \infty$.

**Implementation.** `diagram.rs` → `fn cost_matrix`, `fn bottleneck_distance`, `fn perfect_matching_exists`, `fn augment`, `fn wasserstein_distance`, `fn hungarian_min_cost`. Essential classes are excluded from both; there is no finite distance between an essential class and a finite one. The Hungarian solver carries a `ponytail:` comment naming its ceiling: bar count is bounded by `max_points`, 512 in the widest preset, so $(n+m)$ stays in the hundreds; the trigger is any preset raised past 2048, and the upgrade path is an auction or Sinkhorn solver.

**Evidence.** Metric axioms (`bottleneck_of_a_diagram_with_itself_is_zero`, `bottleneck_is_symmetric`, `bottleneck_satisfies_the_triangle_inequality`), a hand-computed pairing, diagonal projection, `wasserstein_is_at_least_bottleneck` for $p \in \lbrace 1, 2, 4\rbrace $, and `wasserstein_sums_where_bottleneck_takes_a_maximum`, written because a mutant returning the maximum instead of the sum survived.

### 3.12 Persistence landscapes

Each finite bar $[b,d]$ contributes the tent

$$
\lambda_{[b,d]}(t) \;=\; \max\bigl(0,\ \min(t - b,\ d - t)\bigr),
$$

and the $k$-th landscape $\lambda_k(t)$ is the $k$-th largest tent value at $t$, or 0 if fewer than $k$ tents are positive there. Levels are pointwise ordered, $\lambda_1 \ge \lambda_2 \ge \cdots$, by construction.

**Theorem** (Bubenik 2015). For every $k$,

$$
\lVert \lambda_k(D_1) - \lambda_k(D_2) \rVert_\infty \;\le\; d_B(D_1, D_2).
$$

*Sketch.* A tent is 1-Lipschitz in its bar under the $L_\infty$ cost, $\lvert\lambda_{[b,d]}(t) - \lambda_{[b',d']}(t)\rvert \le \max(\lvert b-b'\rvert, \lvert d-d'\rvert)$; a bar matched to the diagonal has a tent no higher than $\tfrac12(d-b)$; and the $k$-th order statistic is 1-Lipschitz in the sup norm of its arguments. $\square$ This is what makes a landscape a feature rather than a hash: a small change in, a small change out.

**Discretisation.** The code samples $N$ = `resolution` points $t_s = t_{\min} + s\,(t_{\max} - t_{\min})/(N-1)$, $s = 0,\dots,N-1$. A supremum over grid points never exceeds the supremum over the line, so the bound holds for the sampled vectors unchanged. Nothing is interpolated between samples, so a feature narrower than the grid step can be missed entirely. `landscape_norm` is the Euclidean norm of the flattened samples; it is not multiplied by the step and is therefore a Riemann sum for the $L^2$ norm only up to the factor $\sqrt{\text{step}}$.

**Implementation.** `diagram.rs` → `fn persistence_landscape`, `fn landscape_norm`. **Evidence.** Five landscape tests, including `landscape_is_one_lipschitz_in_the_bottleneck_distance` and `landscape_takes_the_kth_largest_tent_where_bars_cross` — the latter exists because the original level-ordering test used *nested* bars, whose tent values already arrive sorted, so an implementation skipping the sort passed.

### 3.13 Total persistence and persistent entropy

For the finite bars of one dimension, with lengths $\ell_i = d_i - b_i$ and total $L = \sum_i \ell_i$,

$$
\mathrm{TP}(D) \;=\; \sum_i \ell_i,
\qquad
E(D) \;=\; -\sum_{i\,:\,\ell_i > 0} \frac{\ell_i}{L}\,\ln\frac{\ell_i}{L},
$$

with $E(D) = 0$ when $L \le 0$. $E$ is the Shannon entropy of the normalised bar lengths (Chintakunta et al. 2015), in nats.

**Bounds.** With $N_+$ the number of bars of positive length, $0 \le E(D) \le \ln N_+$ (Gibbs' inequality). $E = 0$ exactly when one bar carries all the persistence — a diagram dominated by a single feature — and $E = \ln N_+$ exactly when all positive bars have equal length, so that no scale dominates. Normalisation makes $E$ invariant under rescaling the filtration, $E(cD) = E(D)$ for $c > 0$, while $\mathrm{TP}(cD) = c\,\mathrm{TP}(D)$. It is a one-number summary of how concentrated the topological signal is, which is what a stopping rule would watch, and it costs $O(b)$ once the diagram exists.

**Implementation.** `diagram.rs` → `fn total_persistence`, `fn persistent_entropy` (natural logarithm, `libm::log`). No test in the suite pins either function; the bounds above are properties of the formula, not measured assertions.

### 3.14 Persistence images

Map each finite bar to birth–persistence coordinates $(b_i, \ell_i)$ and deposit a weighted isotropic Gaussian:

$$
\rho(z) \;=\; \sum_{i\,:\,\ell_i > 0} \ell_i\;\frac{1}{2\pi\sigma^{2}}\exp\!\left(-\frac{\lVert z - (b_i,\ell_i)\rVert^{2}}{2\sigma^{2}}\right),
\qquad z = (z_b, z_\ell).
$$

The weight $w(\ell) = \ell$ is linear in persistence and vanishes on the diagonal. That is not decoration: bars near the diagonal are sampling noise, and a weight that does not vanish there makes the image discontinuous in the input, since a perturbation that creates a tiny bar would produce a jump in the feature vector (Adams et al. 2017).

**Discretisation, and where the code differs from the definition.** Adams et al. define each pixel as the integral of $\rho$ over the pixel. The code instead evaluates $\rho$ at the pixel centre $z_P = \bigl(b_{\min} + (c+\tfrac12)h_b,\ (r+\tfrac12)h_\ell\bigr)$, with $h_b = (b_{\max} - b_{\min})/W$ and $h_\ell = \ell_{\max}/H$. The centre value is the midpoint-rule estimate of the pixel *mean* of $\rho$, and for this kernel

$$
\left\lvert\ \frac{1}{\lvert P\rvert}\int_P \rho \;-\; \rho(z_P)\ \right\rvert
\;\le\;
\frac{\mathrm{TP}(D)}{2\pi\sigma^{4}}\left(\frac{h_b^{2} + h_\ell^{2}}{24} + \frac{h_b\,h_\ell}{16\,e}\right).
$$

*Derivation.* Expand $\rho$ to second order about $z_P$. The linear term integrates to zero over the symmetric pixel. For the normalised Gaussian $G$, $\sup\lvert\partial_b^2 G\rvert = \sup\lvert\partial_\ell^2 G\rvert = 1/(2\pi\sigma^4)$, attained at the centre, and $\sup\lvert\partial_b\partial_\ell G\rvert = 1/(2\pi e\,\sigma^4)$. Over a uniform pixel $\mathbb{E}[u_b^2] = h_b^2/12$, $\mathbb{E}[u_\ell^2] = h_\ell^2/12$ and $\mathbb{E}\lvert u_b u_\ell\rvert = h_b h_\ell/16$. Weighting by $\ell_i$ and summing over bars gives the factor $\mathrm{TP}(D)$. $\square$

To recover Adams et al.'s pixel integrals, multiply the output by $h_b h_\ell$. There is no hard crop: every bar deposits on every pixel, so a bar born outside $[b_{\min}, b_{\max}]$ contributes its Gaussian tail rather than nothing. Adams et al. prove stability of the image with respect to $W_1$ for this family of weightings; that result is cited, not tested here.

**Implementation.** `diagram.rs` → `fn persistence_image`, row-major, row 0 the lowest persistence band, values non-negative, returned zero-filled when $\sigma \le 0$ or the grid is empty. **Evidence.** `persistence_image_has_the_requested_shape_and_is_nonnegative`, `persistence_image_weights_long_bars_more_than_short_ones`, `sigma_controls_the_kernel_width`, `persistence_image_is_translation_equivariant_in_birth`. The σ test exists only because a mutant that ignored $\sigma$ survived the first version of the suite.

### 3.15 The regular-polygon chord

The Vietoris–Rips complex of a circle of radius $r$ has one $H_1$ class, dying at $\sqrt{3}\,r$. That is the continuous statement. For $n$ equally spaced points the death is the chord subtending $\lceil n/3\rceil$ steps:

$$
d_{H_1}(n, r) \;=\; 2r\sin\!\left(\frac{\pi\,\lceil n/3\rceil}{n}\right).
$$

*Derivation sketch.* A triangle on the polygon has vertex gaps $(a, b, c)$ steps with $a + b + c = n$, and its side spanning $a$ steps has length $2r\sin(\pi\min(a, n-a)/n)$. A triangle that does not contain the centre has one gap exceeding $n/2$ and lies over an arc; triangles of that kind do not kill the loop (Adamaszek & Adams 2017 establish that the complex is homotopy-equivalent to $S^1$ below the threshold). The cycle dies when the first triangle containing the centre enters. Its longest side spans $\max(a,b,c)$ steps, and the smallest possible maximum of three positive integers summing to $n$ is $\lceil n/3\rceil$. $\square$

Since $2\sin(\pi/3) = \sqrt3$, the death equals $\sqrt3\,r$ exactly when $3 \mid n$. Otherwise $\lceil n/3\rceil/n > 1/3$, the death exceeds $\sqrt3\,r$, and it tends to $\sqrt3\,r$ as $n\to\infty$.

**Evidence.** `circle_h1_dies_at_the_exact_regular_polygon_chord` reproduces the formula to 1e-12 for $n \in \lbrace 9, 10, 11, 12, 13, 17, 24, 48\rbrace $; `circle_h1_death_converges_to_sqrt3_from_above` checks the approach along $n \in \lbrace 10, 13, 25, 49, 97\rbrace $. An earlier test asserted "within 5% of $\sqrt3 r$", which would have passed a systematically wrong implementation; deriving the finite-$n$ form turned a tolerance into an identity (§8).

### 3.16 Delay embedding

The map that turns a scalar series into a point cloud is

$$
\Phi_{\tau,D}(t) \;=\; \bigl(x_t,\ x_{t-\tau},\ x_{t-2\tau},\ \dots,\ x_{t-(D-1)\tau}\bigr) \;\in\; \mathbb{R}^{D},
$$

with delay $\tau \ge 1$ and embedding dimension $D$.

**Theorem** (Takens 1981). For a compact $d$-dimensional manifold $M$, a generic diffeomorphism $\varphi$ and a generic observation $h : M \to \mathbb{R}$, the delay map $p \mapsto \bigl(h(p), h(\varphi^{-1}p), \dots\bigr)$ is an embedding when $D \ge 2d + 1$. Sauer, Yorke & Casdagli (1991) extend this to attractors of box-counting dimension $d_{\mathrm{box}}$ with $D > 2d_{\mathrm{box}}$.

Under an embedding the reconstructed cloud is diffeomorphic to the attractor, so its homology is the attractor's homology. That is what makes `manifold M = embed(data, tau=1)` more than a reshaping trick: it is why $\beta_1$ of the delay cloud says something about the *system* rather than about the windowing.

**Implementation.** `manifold.rs` → `TimeDelayEmbedder::{push, embed}` over a 256-sample ring buffer; `persistence.rs` → `fn time_delay_persistence` composes embedding and persistence in one call so the embedder and the engine cannot disagree about $\tau$. Three details are exact consequences of the code. `embed` returns `None` until $D\tau$ samples have been pushed, although the map needs only $(D-1)\tau + 1$, so the first $\tau - 1$ usable windows are discarded and $N$ samples yield $N - D\tau + 1$ points for $N \ge D\tau$. $D\tau \le 256$ is required, or no point is ever produced. $\tau = 0$ is coerced to 1. The interpreter fixes $D = 3$.

**What is not solved.** The theorem is generic and asymptotic: it does not supply $d$, does not supply $\tau$, and says nothing about finite noisy samples. A badly chosen $\tau$ gives a cloud that is technically an embedding and practically a diagonal smear. The caller must pass $\tau$; no mutual-information or false-nearest-neighbour selection is implemented.

### 3.17 Graph Betti numbers on the streaming path

The streaming pipeline maintains the neighbourhood graph $G_\varepsilon$ on the embedded points, with an edge whenever $d(x_i,x_j) < \varepsilon$ (strict). It reports two numbers:

$$
\beta_0(G_\varepsilon) = \#\text{components},
\qquad
\beta_1(G_\varepsilon) \;=\; \lvert E\rvert - \lvert V\rvert + \beta_0(G_\varepsilon).
$$

The second is the Euler–Poincaré relation of §3.4 for a one-dimensional complex, $\chi = \lvert V\rvert - \lvert E\rvert = \beta_0 - \beta_1$, so it is the *exact* cycle rank of the graph, not an approximation of it.

**Relation to the Rips complex.** $G_\varepsilon$ is the 1-skeleton of the flag complex $\mathrm{VR}_{<\varepsilon}$. Adding 2-simplices can only kill 1-cycles, so $H_1$ of the flag complex is a quotient of $H_1(G_\varepsilon)$ and

$$
\beta_1\bigl(\mathrm{VR}_{<\varepsilon}\bigr) \;\le\; \beta_1(G_\varepsilon).
$$

Three mutually adjacent points give $\beta_1(G) = 1$ while the filled triangle has $\beta_1 = 0$. `estimate_betti_1` is therefore named correctly as an estimate of the Rips quantity — it is an upper bound on it — while being exact for the graph.

**Implementation, and its range of validity.** `manifold.rs` → `SparseAttentionGraph::{add_point, compute_betti_0, estimate_betti_1}`. Adjacency is a `u64` bitset per point, so edges are recorded only to the first 64 points, and for a point with index 64 or above the reverse edge is written to bit `idx % 64`, aliasing it onto an earlier point. `compute_betti_0` is an iterative depth-first search with a fixed 64-entry stack that drops pushes when full. Both numbers are exact for graphs of at most 64 points whose traversal stays within the stack. `ml::clustering::auto_k_selection` computes the same $\beta_0$ on an $\varepsilon$-graph, with the same stack, as its choice of $k$ — the number of clusters *is* $\beta_0$ at the right scale.

### 3.18 Numerically stable softmax and the online recurrence

For scores $s_j = q\cdot k_j/\sqrt{d_h}$ over a non-empty set of permitted keys,

$$
\mathrm{softmax}(s)_j \;=\; \frac{e^{\,s_j - m}}{\sum_{k} e^{\,s_k - m}},
\qquad m = \max_k s_k .
$$

Subtracting $m$ leaves the value unchanged (numerator and denominator scale by $e^{-m}$), bounds every exponent at or below zero so `exp` cannot overflow, and guarantees a denominator of at least 1, because the maximal term contributes $e^0$. The case the identity does not cover is an empty row, where the denominator is a sum over nothing; that is a policy decision, and the reference kernel returns zeros for it.

**Online recurrence.** The scheduled kernel never holds a full row of scores. Visiting score tiles in turn with running state $(m, l, a)$ — running maximum, denominator and value accumulator — it applies (Milakov & Gimelshein 2018; Dao et al. 2022)

$$
m' = \max\Bigl(m,\ \max_{j\in\text{tile}} s_j\Bigr),\quad
\alpha = e^{\,m - m'},\quad
l' = \alpha\, l + \sum_{j\in\text{tile}} e^{\,s_j - m'},\quad
a' = \alpha\, a + \sum_{j\in\text{tile}} e^{\,s_j - m'}\, v_j .
$$

**Invariant.** After any prefix $B$ of tiles, $m = \max_{j\in B} s_j$, $l = \sum_{j\in B} e^{s_j - m}$ and $a = \sum_{j\in B} e^{s_j - m} v_j$. It holds after the first tile, and multiplying by $\alpha$ re-bases the old sums from $m$ to $m'$, which preserves it. Hence $a/l = \sum_j \mathrm{softmax}(s)_j\, v_j$ exactly in exact arithmetic; in floating point the online and one-pass forms differ by rounding only, measured below 1e-12 against the dense masked reference (§6.5). Two edge cases are handled explicitly: the first tile uses $\alpha = 0$ ($e^{-\infty}$), and a tile lying entirely in the causal future is skipped, since folding it in would evaluate $e^{-\infty-(-\infty)} = \text{NaN}$.

**Implementation.** `attention.rs` → `fn sparse_attention` (one-pass form, zeros for an empty row); `scheduled.rs` → `fn scheduled_attention` (online form; an empty row is rejected at schedule validation with `EmptyRow`). **Evidence.** `large_logits_do_not_overflow_the_softmax`, `large_logits_stay_finite_across_scheduled_blocks`, `an_all_masked_row_returns_zeros_rather_than_nan`, `an_empty_row_is_rejected_rather_than_producing_nan`.

### 3.19 The scheduled-attention backward pass

For query row $i$ with scheduled causal columns $C_i$, let $P_{ij} = \mathrm{softmax}_{j\in C_i}(s_{ij})$, $s_{ij} = q_i\cdot k_j/\sqrt{d_h}$ and $O_i = \sum_j P_{ij} v_j$. Given the upstream gradient $G_i = \partial L/\partial O_i$:

$$
\frac{\partial L}{\partial v_j} = \sum_{i} P_{ij}\, G_i,
\qquad
\Delta_i = \sum_{j\in C_i} P_{ij}\,(G_i\cdot v_j),
\qquad
\frac{\partial L}{\partial s_{ij}} = P_{ij}\bigl(G_i\cdot v_j - \Delta_i\bigr),
$$

$$
\frac{\partial L}{\partial q_i} = \frac{1}{\sqrt{d_h}}\sum_{j} \frac{\partial L}{\partial s_{ij}}\, k_j,
\qquad
\frac{\partial L}{\partial k_j} = \frac{1}{\sqrt{d_h}}\sum_{i} \frac{\partial L}{\partial s_{ij}}\, q_i .
$$

*Derivation.* The softmax Jacobian is $\partial P_j/\partial s_k = P_j(\delta_{jk} - P_k)$, so $\partial L/\partial s_k = \sum_j (G\cdot v_j)\,P_j(\delta_{jk} - P_k) = P_k(G\cdot v_k - \Delta)$: a rank-one correction rather than a full matrix. $\square$

The schedule is held constant. It is combinatorial, derived from the keys by a selection with no useful derivative, so a gradient can teach a model to use the blocks it was given but not to schedule differently. Scores are recomputed per row rather than stored — the trade flash attention makes in reverse — so the working set stays proportional to one row.

The same Jacobian–vector product, $\partial L/\partial z_i = p_i\bigl(g_i - \langle p, g\rangle\bigr)$, is what `ml::neural::DenseLayer::backward` applies for a softmax output layer. Until it did, the layer multiplied by an elementwise derivative of zeros and a network with a softmax output did not train at all while its loss stayed finite.

**Implementation.** `scheduled.rs` → `fn scheduled_attention_backward`. **Evidence.** `gradients_match_finite_differences_on_a_dense_schedule` and `…_on_a_sparse_schedule` compare every analytic gradient against a central difference of the forward kernel (truncation error $O(h^2)$), the only check that shares no assumption with the code under test; `keys_outside_the_schedule_receive_no_gradient`; `the_value_gradient_reproduces_the_attention_weights`; `the_softmax_layer_gradient_matches_finite_differences`.

### 3.20 Block salience, recovered mass and the oracle

**Salience.** Partition a sequence of length $s$ into $N = s/B$ blocks of size $B$ (a power of two dividing $s$) with centroids $c_b = \frac1B\sum_{t\in b} k_t$. `block_salience` runs the Kruskal scan of §3.8 on the centroids; at each merge the smaller component is absorbed (on equal sizes, the component of the lower-indexed edge endpoint), and **every** member of the absorbed component has its salience overwritten with the merge distance. Because merges arrive in increasing height, the final value is

$$
\mathrm{sal}(b) \;=\; \max\bigl\lbrace \, h_e \;:\; e \text{ is a merge at which the component containing } b \text{ was absorbed} \,\bigr\rbrace ,
\qquad \mathrm{sal}(b) = 0 \text{ if never absorbed.}
$$

Three properties follow, and one widely repeated property does not.

1. Every non-zero salience is a finite $H_0$ death of the centroid cloud, since it is a Kruskal merge height (§3.8). `block_salience_is_the_elder_rule_over_centroids` asserts this membership.
2. Exactly one block scores 0 when the centroids are distinct. *Proof:* every component always contains exactly one never-written block — true of singletons, and preserved by each merge, which writes the absorbed side's unwritten block and leaves the absorbing side's alone — so one block remains unwritten at the end. $\square$
3. The per-block assignment depends on block order, through the tie-break.
4. **The multiset of saliences is not in general the $H_0$ barcode, and not in general invariant to block order.** When a component with more than one member is absorbed, the earlier deaths written to its members are overwritten. Running the shipped function on one-dimensional centroids $0, 1, 10, 12$ (block size 1) returns saliences $(9, 9, 2, 0)$ against finite $H_0$ deaths $\lbrace 1, 2, 9\rbrace $; reversing the order returns $(9, 9, 1, 0)$. The multiset equals the barcode when every absorbed component is a singleton at the moment of absorption. `the_salience_multiset_is_invariant_to_block_order` passes on a fixture that does not exercise the case (§8.9).

**Schedule.** With `sink_blocks` $= \mathsf{s}$, `local_radius_blocks` $= \mathsf{r}$ and $T$ the `topk_topology_blocks` $= \mathsf{k}$ highest-salience blocks (ties by index), query block $q$ may see

$$
S_q \;=\; \Bigl(\lbrace 0,\dots,\mathsf{s}-1\rbrace  \,\cup\, \lbrace q-\mathsf{r},\dots,q\rbrace  \,\cup\, T\Bigr) \cap \lbrace 0,\dots,q\rbrace ,
\qquad
\lvert S_q\rvert \le \min(q+1,\ \mathsf{s}+\mathsf{r}+1+\mathsf{k}).
$$

The local window contains $q$, so no row is empty. The dense causal schedule has $N(N+1)/2$ blocks. At $N = 16$ with $\mathsf{s}=1,\ \mathsf{r}=1,\ \mathsf{k}=2$ the bound gives at most $1+2+3+4+12\cdot5 = 70$ of 136 blocks, a reduction of at least 48.5%; the measured schedule visits 56, a 58.8% reduction (§6.5).

**Recovered mass.** With $P_{ij}$ the causal softmax of row $i$ over all its legal keys,

$$
M(S) \;=\; \frac{1}{N}\sum_{q}\sum_{b\in S_q} T_{qb},
\qquad
T_{qb} \;=\; \frac{1}{B}\sum_{i\in q}\ \sum_{j\in b,\ j\le i} P_{ij},
$$

so $M = 1$ for the dense causal schedule and $0 < M \le 1$ otherwise.

**Proposition (the oracle is exact).** For per-row budgets $\kappa_q$, the schedule taking the $\kappa_q$ largest entries of row $q$ of $T$ maximises $M$. *Proof:* $M$ is a sum of independent per-row terms, each a sum of $\kappa_q$ entries of a fixed row, and the largest $\kappa_q$ entries of a list maximise such a sum (exchange argument). $\square$ The oracle is therefore the ceiling itself, not an approximation to it, and `no_schedule_recovers_more_mass_than_the_oracle` is an assertion rather than an aspiration.

**Implementation.** `scheduled.rs` → `fn block_salience`, `fn topology_block_schedule`, `fn inverted_topology_block_schedule` (lowest salience first, the zero sentinel excluded), `fn block_mass_recovered`, `fn oracle_block_schedule`, `fn random_block_schedule`. The random baseline is a partial Fisher–Yates shuffle driven by the linear congruential generator $x \leftarrow 6364136223846793005\,x + 1442695040888963407 \pmod{2^{64}}$, drawing the top 31 bits, `x >> 33`.

### 3.21 The placement statistic

Not textbook mathematics, but the instrument every attention ablation here is reported against. For a selector $S$, a random baseline $R$ and a budget-matched oracle $O$, with $m(\cdot)$ the recovered attention mass at **equal budget**,

$$
\mathrm{pl}(S) \;=\; \frac{m(S) - m(R)}{m(O) - m(R)},
\qquad
m(S) \;=\; \frac{1}{s}\sum_{i=1}^{s}\ \sum_{j\in S_i} \mathrm{softmax}_i(j).
$$

Placement 0 is indistinguishable from choosing keys uniformly at random; placement 1 matches a selector that computed every score before choosing; a negative value is worse than random. The normalisation makes numbers comparable across input distributions, since $m(S)$ alone drifts with the key distribution and would let a selector look better by being tested on easier data.

**Two ways the statistic lies, both encountered here.** Equal budget is load-bearing: a selector that declines to select — 1.00 keys per row where its baselines take 5.53 — posts catastrophic placement without losing on mechanism (§8.3). And the denominator can vanish or invert: with a dense fallback, $m(S)$ exceeds $m(O)$, because dense recovers all the mass while the oracle is budget-limited, and the ratio blew up to a printed **+7.614** that would read as a 700% win over an oracle it never competed with (§8.7). The two regimes are now scored on different scales deliberately.

**Cost is reported beside quality.** Dense causal attention performs $(s+1)/2$ dot products per row on average; `selection_dot_cost` reports what a selector performs to make its choice. A quality metric with no paired cost metric eventually rewards a method for doing more work (§8.4).

**Implementation.** `attention.rs` → `fn attention_mass_recovered`, `fn selection_dot_cost`, `fn dense_dot_cost`; the block form is `scheduled.rs` → `fn block_mass_recovered`.

### 3.22 The routing gap ratio

**Why Euclidean proximity fails as a proxy for attention mass.** For every query and key,

$$
\lVert q - k\rVert^2 \;=\; \lVert q\rVert^2 + \lVert k\rVert^2 - 2\,q\cdot k ,
$$

so ranking keys by distance from $q$ equals ranking them by dot product exactly when $\lVert k\rVert$ is constant across keys. Vary the norms and the rankings decouple: a large-norm key is simultaneously far away and high-scoring. This is the identity behind §8.2.

**Routing on directions.** The routed selector clusters the unit-normalised keys $\hat k_t = k_t/\lVert k_t\rVert$ (a zero vector stays at the origin rather than having a direction invented for it). The clustering is invariant under per-key rescaling $k_t \mapsto \lambda_t k_t$ with $\lambda_t > 0$, because $\hat k_t$ is. Within the candidate clusters the exact dot product ranks, restoring the norm sensitivity the geometry discarded.

**The gap ratio.** Let $h_1 \le \dots \le h_{n-1}$ be the single-linkage merge heights of $\lbrace \hat k_t\rbrace $ and $c$ the number of clusters left after the cut, so $n - c$ merges are applied. Then

$$
\gamma \;=\; \frac{h_{\,n-c+1}}{h_{\,n-c}},
$$

the first merge above the cut over the last merge below it, with $\gamma = 1$ when either side is missing or $h_{n-c} = 0$. A cloud with genuine cluster structure has a large jump at the cut: components stay separate until well past the within-cluster scale. A cloud with no structure chains, single-linkage absorbing one point at a time at nearly equal heights, and $\gamma \approx 1$. Measured separation, 6 trials each: **structured minimum 2.70, chained maximum 1.04**, no overlap.

**What decides routing.** `routing_plan` sets `worth_routing` by thresholding the predicted cost ratio, `cost_ratio < 0.6` (`ROUTING_COST_THRESHOLD`), where the ratio is the routed selector's dot products over dense. The gap ratio is reported beside that decision, not used in it: it is the barcode-only statistic that separates the regimes and could be cached so that a runtime skips the clustering entirely. The thesis this supports is conditional — not "topology makes attention faster", which is false on uniform keys, but that **the $H_0$ barcode separates the distributions on which a topological method pays from those on which it does not**, and is cheap enough to consult before committing.

**Implementation.** `attention.rs` → `fn routing_plan`, `fn single_linkage_clusters`, `Selector::TopologicalRouted`, `Selector::Adaptive` (routes when the plan says so, otherwise runs dense). **Evidence.** `the_h0_barcode_alone_separates_the_two_regimes`, `the_plan_predicts_the_cost_it_will_actually_incur`, `the_plan_declines_to_route_exactly_when_routing_would_not_pay`, `routing_clusters_are_invariant_to_per_key_rescaling`.

### 3.23 Admissible bounds for hierarchical pruning

For a block with centroid $c$ and radius $\rho = \max_{x} \lVert x - c\rVert$, and any query $q$,

$$
q\cdot x \;\le\; \lVert q\rVert\,\lVert x\rVert \;\le\; \lVert q\rVert\bigl(\lVert c\rVert + \rho\bigr) \;=:\; U(q)
\qquad \text{for every } x \text{ in the block,}
$$

by Cauchy–Schwarz and then $\lVert x\rVert \le \lVert c\rVert + \lVert x - c\rVert$. A block is pruned when $U(q) < \theta$. Since $U$ over-estimates every score in the block, a pruned block contains no $x$ with $q\cdot x \ge \theta$: pruning changes cost, never the answer — the argument branch-and-bound rests on, which is why the bound must be an over-estimate rather than an estimate.

**Aggregation preserves admissibility and monotonicity.** A parent over children $(c_i, \rho_i, n_i)$ stores the count-weighted centroid $C = \sum_i n_i c_i / \sum_i n_i$ and the radius $R_P = \max_i\bigl(\lVert c_i - C\rVert + \rho_i\bigr)$. Every point of child $i$ satisfies $\lVert x - C\rVert \le \rho_i + \lVert c_i - C\rVert \le R_P$, so $R_P$ is a valid radius, and

$$
\lVert C\rVert + R_P \;\ge\; \lVert C\rVert + \lVert c_i - C\rVert + \rho_i \;\ge\; \lVert c_i\rVert + \rho_i ,
$$

so a parent's bound dominates each child's. Consequently `hierarchical_query` returns exactly the set $\lbrace \,b : U_b(q) \ge \theta\,\rbrace $ that testing every fine block would: descending the tree only skips subtrees in which every block would be pruned anyway. The tree has three levels (64-token blocks, 256-token clusters of 4, 1024-token super-clusters of 16) over at most 128 fine blocks. `pruning_ratio` reports the fraction actually skipped, so a hierarchical query that prunes nothing is visible as such.

**Other block statistics.** `from_points` also records $\mathrm{Var}\bigl(\lVert x - c\rVert\bigr) = \mathbb{E}\lVert x-c\rVert^2 - \bigl(\mathbb{E}\lVert x-c\rVert\bigr)^2$ and the mean cosine between each point and $c$. `select_compression` maps variance below 0.1 to centroid-delta coding, otherwise concentration above 0.9 to 4-bit quantisation, otherwise full precision; `estimate_compression_ratio` returns the constants 4.0, 4.0 and 1.0 for the three cases. These are estimates by name and by construction; no measured compression figure appears in this document.

**Implementation.** `aether.rs` → `BlockMetadata::{from_points, upper_bound_score, can_prune}`, `HierarchicalBlockTree::{build_from_blocks, hierarchical_query, pruning_ratio}`.

### 3.24 Drift as a second difference

`DriftDetector` predicts each centroid by constant velocity, $\hat c_{t+1} = c_t + v_t$ with $v_t = c_t - c_{t-1}$, and reports the prediction error. Substituting,

$$
\mathrm{drift}_t \;=\; \lVert \hat c_t - c_t\rVert \;=\; \lVert c_t - 2c_{t-1} + c_{t-2}\rVert \qquad (t \ge 3),
$$

the norm of the discrete second difference — an acceleration. It is zero on any constant-velocity trajectory, $\lVert c_2 - c_1\rVert$ at $t = 2$ (the velocity starts at zero) and 0 at $t = 1$. `is_drifting` thresholds $\lVert v_t\rVert$, the first difference, not the drift score. **Evidence.** `test_drift_detector`: centroids $0, 1, 2$ then $5$ give drift $\lVert 5 - 2\cdot2 + 1\rVert = 2$. **Implementation.** `aether.rs` → `DriftDetector::{update, is_drifting}`.

### 3.25 The Chebyshev guard

For any distribution with finite mean $\mu$ and variance $\sigma^2$ — no normality assumption — Chebyshev's two-sided inequality and Cantelli's one-sided inequality give

$$
\Pr\bigl(\lvert X-\mu\rvert \ge k\sigma\bigr) \le \frac{1}{k^2},
\qquad
\Pr\bigl(X - \mu \le -k\sigma\bigr) \le \frac{1}{1+k^2}.
$$

Both hold for the empirical distribution of any finite sample, with $\mu$ and $\sigma$ its population mean and standard deviation. That is the setting here: the manifold heap cannot assume a distribution of object lifetimes, because that would be an assumption about user programs, and these inequalities hold regardless. They are loose, which is the correct trade for a guard.

**The rule as implemented.** Each occupied slot carries a liveness score $\ell \in (0, 10]$: 1.0 at allocation, $+0.5$ on `touch`, $+1.0$ on `get_mut`, $+2.0$ on `mark`, capped at 10. A regulation pass computes $\mu$ and $\sigma$ over all $n$ occupied slots (not per block), then for each slot multiplies $\ell$ by 0.95 and reclaims it if it is unmarked and $0.95\,\ell \le \mu - k\sigma$ with $k = 2$. Only the lower tail is tested, so the relevant inequality is Cantelli's: were the comparison made on the undecayed score, at most $n/(1+k^2) = n/5$ objects could be reclaimed per pass.

**The decay shifts the effective $k$.** Comparing the decayed score against pre-decay statistics is the same as comparing $\ell$ against $\mu - k_{\mathrm{eff}}\sigma$ with

$$
k_{\mathrm{eff}} \;=\; \frac{20k\sigma - \mu}{19\,\sigma},
\qquad
\#\lbrace \text{reclaimed}\rbrace  \;\le\; \frac{n}{1 + k_{\mathrm{eff}}^{2}} \quad \text{when } k_{\mathrm{eff}} > 0 .
$$

For $k = 2$ the bound exists only while $\mu < 40\sigma$, and it weakens as $\mu/\sigma$ grows. When $\sigma = 0$ the boundary collapses to $\mu$ and every unmarked object is reclaimed.

**What guarantees safety.** The mark, not the inequality: a marked object is never reclaimed. The guard only limits how many *unmarked* objects are reclaimed per pass, so a false "safe" retains garbage — a leak — and never frees a traced object.

**Implementation.** `memory.rs` → `ChebyshevGuard::{calculate, is_safe}`, `ManifoldHeap::regulate_entropy`. No test exercises the guard.

### 3.26 The governor control law

The kernel wakes when the state deviation $\Delta_t = \lVert \mu(t) - \mu(t_{\mathrm{last}})\rVert_2$ reaches a threshold $\varepsilon_t$, and the governor adapts $\varepsilon$ to hold the effective wake rate $R_t = \Delta_t/\varepsilon_t$ near $R^\star$ = `TARGET_TICK_RATE` = 1000:

$$
e_t \;=\; \operatorname{clamp}\!\left(1 - \frac{R_t}{R^\star},\ -1,\ 1\right),
\qquad
\ln\varepsilon_{t+1} \;=\; \ln\varepsilon_t - \alpha\, e_t - \beta\,(e_t - e_{t-1}),
$$

followed by clamping $\varepsilon$ to $[\varepsilon_{\min}, \varepsilon_{\max}] = [0.001, 10]$, with $\alpha = 0.25$, $\beta = 0.05$ and $e_0 = 0$. A high observed rate ($e < 0$) raises $\varepsilon$; a low one lowers it.

**Fixed point.** For a steady $\Delta$, $e = 0$ exactly when $\varepsilon = \varepsilon^\star = \Delta/R^\star$, since $R/R^\star = \varepsilon^\star/\varepsilon$. If $\varepsilon^\star$ lies outside the clamp band — $\Delta < 1$ or $\Delta > 10^4$ — the governor settles on the nearer bound.

**Proposition (local stability).** Near $\varepsilon^\star$ the unclamped law is locally asymptotically stable if and only if

$$
0 < \alpha < 2 - 2\beta \qquad (\beta \ge 0),
$$

independently of $\Delta$ and of the time step.

*Proof.* Write $u = \ln(\varepsilon/\varepsilon^\star)$. Then $e = 1 - e^{-u} = u + O(u^2)$, and the law linearises to $u_{t+1} = (1-\alpha-\beta)\,u_t + \beta\,u_{t-1}$, with characteristic polynomial $z^2 - (1-\alpha-\beta)z - \beta$. The Jury conditions for $z^2 + pz + q$ are $1 + p + q > 0$, $1 - p + q > 0$ and $\lvert q\rvert < 1$, which here read $\alpha > 0$, $2 - \alpha - 2\beta > 0$ and $\beta < 1$; the second implies the third when $\alpha > 0$. $\square$

With the shipped gains the roots are $z = \tfrac12\bigl(0.7 \pm \sqrt{0.69}\bigr) \approx 0.765,\ -0.065$, so the deviation $u$ contracts by about a factor 0.765 per step in the linear regime. Far from equilibrium the error clamp bounds each step: $\lvert e_t\rvert \le 1$ and $\lvert e_t - e_{t-1}\rvert \le 2$ give $\lvert\ln\varepsilon_{t+1} - \ln\varepsilon_t\rvert \le \alpha + 2\beta = 0.35$, so one step multiplies $\varepsilon$ by a factor in $[e^{-0.35}, e^{0.35}] \approx [0.705, 1.419]$.

**History.** The previous law, $\varepsilon \mathrel{-}= \alpha(R^\star - R) + \beta\,\dot e$ with $\alpha = 0.01$ and $\dot e$ divided by $dt \approx 10^{-3}$ s, mixed an error of order $10^3$ Hz with a threshold of order $10^{-3}$ to 10; one step crossed the whole band, $\varepsilon$ alternated between the two clamps, and the scheduler stopped waking once it sat at 10. The derivative term now uses the per-step change in error; `dt` is still accepted and feeds only an unused integral accumulator, so the controller is PD, not PID, whatever its doc comment's heading says.

**Implementation.** `governor.rs` → `GeometricGovernor::adapt`, `should_trigger` (the wake test $\Delta \ge \varepsilon$); caller `aether-kernel/src/scheduler.rs` → `SparseScheduler::{should_wake, handle_event}`. **Evidence.** `test_steady_deviation_converges_to_equilibrium` pins convergence to $\varepsilon^\star$ within relative 1e-6 for $\Delta \in \lbrace 5, 50, 2000\rbrace $ over 200 steps, and that a further step does not move it; `test_epsilon_clamped_min`/`max` pin the band. The stability proposition itself is derived, not tested.

### 3.27 Ring gossip consensus

Each node on a ring of $n$ nodes replaces its estimate by the average of its own and its predecessor's, from a synchronous snapshot:

$$
x_i^{(t+1)} \;=\; \tfrac12\, x_i^{(t)} + \tfrac12\, x_{i-1}^{(t)} \quad (\text{indices mod } n),
\qquad
x^{(t+1)} = W x^{(t)},\quad W = \tfrac12(I + P),
$$

with $P$ the cyclic shift. $W$ is circulant and doubly stochastic, so the mean $\bar x$ of the estimates is preserved exactly. Its eigenvalues are $\lambda_k = \tfrac12(1 + \omega^k)$ with $\omega = e^{2\pi i/n}$, of modulus $\lvert\cos(\pi k/n)\rvert$. $W$ is normal, so on the complement of the consensus direction

$$
\bigl\lVert x^{(t)} - \bar x\,\mathbf 1\bigr\rVert_2 \;\le\; \cos^{t}\!\left(\frac{\pi}{n}\right)\bigl\lVert x^{(0)} - \bar x\,\mathbf 1\bigr\rVert_2, \qquad n \ge 2 .
$$

For even $n$ the alternating mode ($k = n/2$) vanishes after one step. At the ring's capacity of 16 nodes, $\cos(\pi/16) \approx 0.981$, so reducing the disagreement by a factor $10^6$ takes $\lceil \ln 10^{-6} / \ln\cos(\pi/16)\rceil = 713$ ticks.

**Implementation.** `ml/gossip.rs` → `GossipNode::update_consensus` (mixing weight 0.5), `GossipRing::tick` (snapshot, hence synchronous), `GossipRing::converge` (stops when the estimates' variance falls below a tolerance). Estimates have `MAX_DIM = 3` components.

### 3.28 Learning primitives

The `ml` subtree implements standard methods from scratch against `libm`. Where the constants differ from the textbook the difference is stated.

**Initialisation.** `Tensor::kaiming_uniform` draws from $\mathcal U\bigl(-\sqrt{3/n_{\mathrm{in}}},\ \sqrt{3/n_{\mathrm{in}}}\bigr)$, whose variance is $1/n_{\mathrm{in}}$ — LeCun's variance, half of He et al.'s $2/n_{\mathrm{in}}$ for ReLU networks, which corresponds to the bound $\sqrt{6/n_{\mathrm{in}}}$. `DenseLayer::new` draws from $\mathcal U(-a, a)$ with $a = \sqrt{2/(n_{\mathrm{in}} + n_{\mathrm{out}})}$, variance $2/\bigl(3(n_{\mathrm{in}}+n_{\mathrm{out}})\bigr)$, one third of Glorot's $2/(n_{\mathrm{in}}+n_{\mathrm{out}})$. Both use the generator $x \leftarrow 6364136223846793005\,x + 1 \pmod{2^{64}}$, mapped to $[-1, 1]$ by $2x/(2^{64}-1) - 1$.

**Optimisers.** SGD with momentum: $v \leftarrow \mu v - \eta g$, $w \leftarrow w + v$ (default $\mu = 0.9$). Adam (Kingma & Ba 2015):

$$
m \leftarrow \beta_1 m + (1-\beta_1)\,g,\quad
v \leftarrow \beta_2 v + (1-\beta_2)\,g^{2},\quad
w \leftarrow w - \eta\,\frac{m/(1-\beta_1^{t})}{\sqrt{v/(1-\beta_2^{t})} + \epsilon},
$$

with defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

**Activations.** $\sigma'(x) = \sigma(x)\bigl(1-\sigma(x)\bigr)$ and $\tanh'(x) = 1 - \tanh^2 x$, with arguments clamped to $[-500, 500]$ before exponentiation; $\mathrm{ReLU}'(0) = 0$, matching the GPU kernel; LeakyReLU has slope 0.01 below zero, including at zero. Softmax is handled by the Jacobian–vector product of §3.19 rather than by an elementwise derivative.

**Classification.** Logistic regression minimises binary cross-entropy with clipped probabilities. Gaussian naive Bayes predicts $\arg\max_c\ \ln\pi_c - \tfrac12\sum_j\bigl[\ln(2\pi\sigma_{cj}^2) + (x_j - \mu_{cj})^2/\sigma_{cj}^2\bigr]$. AdaBoost over decision stumps (Freund & Schapire 1997) uses $\alpha_t = \tfrac12\ln\bigl((1-\epsilon_t)/\epsilon_t\bigr)$ with $\epsilon_t$ clamped to $[10^{-10}, 1-10^{-10}]$, reweights $w_i \leftarrow w_i\, e^{-\alpha_t y_i h_t(x_i)}$ with renormalisation, predicts $\operatorname{sign}\sum_t \alpha_t h_t(x)$, and keeps at most 32 stumps. The `u32`/`i32` split in return types is deliberate: `u32` is a class index, `i32` a $\pm1$ margin label.

**Regression.** `ManifoldRegressor` fits on the first embedding coordinate. The linear model is ordinary least squares, $b = \bigl(n\sum xy - \sum x\sum y\bigr)/\bigl(n\sum x^2 - (\sum x)^2\bigr)$, $a = (\sum y - b\sum x)/n$. Higher polynomial degrees are **not** a least-squares solve: each coefficient $d \ge 2$ is set to $c_d = \sum_i r_i x_i^{d} / \sum_i x_i^{2d}$ from the residuals $r_i$ of the current fit — one greedy pass of coordinate descent, equal to the least-squares solution only when the monomials are orthogonal on the sample. The RBF model is the Nadaraya–Watson estimator $\hat y(x) = \sum_i K(x,x_i)\,y_i / \sum_i K(x,x_i)$ with $K = e^{-\gamma\lVert x - x_i\rVert^2}$; the "Gaussian process" model is the same estimator with $\gamma = 1/(2\ell^2)$; the "geodesic regression" model is the degree-4 greedy polynomial fit.

**Streaming statistics.** `GeometricConcentrator` maintains per-coordinate running means and sums of squared deviations by Welford's recurrence, $\mu_n = \mu_{n-1} + (x - \mu_{n-1})/n$ and $M_n = M_{n-1} + (x - \mu_{n-1})(x - \mu_n)$, over at most the first 8 coordinates. Its "principal dimension" is the coordinate of largest $M$ — an axis-aligned choice, not principal component analysis — and `concentration_ratio` is $M_{\max}/\sum M$.

### 3.29 The convergence predicates

The language premise is that loops can stop on topology. Two built-in convergence predicates exist in the tree, and neither consults the persistence engine; the generic `seal until stable(expr)` condition of §4.6 is how a program stops on a Betti vector instead.

**(a) The language's `regress` statement** — `aether-lang` interpreter, `EscalatingRegressor::is_converged`. At epoch $t$, with residuals $r_i = y_i - \hat y_i$, the interpreter computes

$$
\mathbf b_t \;=\; \Bigl(\,\bigl\lfloor \mathrm{sc}_t/2 \bigr\rfloor + 1,\ \ \bigl\lfloor \mathrm{osc}_t/4 \bigr\rfloor\,\Bigr),
$$

where $\mathrm{sc}_t$ counts indices $i \ge 1$ with $[r_i \ge 0] \ne [r_{i-1} \ge 0]$ (sign changes) and $\mathrm{osc}_t$ counts indices $i \ge 2$ with $[r_i - r_{i-1} > 0] \ne [r_{i-1} > 0]$, and stops when

$$
e_t < \varepsilon \quad\lor\quad \bigl(t \ge 3 \;\wedge\; \mathbf b_{t-2} = \mathbf b_{t-1} = \mathbf b_t\bigr),
$$

with $e_t$ the mean squared error. The pair $\mathbf b_t$ is a sign-pattern count on the residual sequence, not persistent homology. $\varepsilon$ is always $10^{-6}$ in practice (§4.7). Models escalate by epoch — linear; polynomial of degree 2, 3, 4; RBF with $\gamma = 0.1\,t$ for epochs 4–6; then RBF with $\gamma = 1$ — for at most 100 epochs with `escalate: true`, otherwise 10.

**(b) The library detector** — `ml/convergence.rs`, `ConvergenceDetector::is_converged`, with window $W' = \max(W, 3)$ and at least $W'$ epochs recorded:

$$
e_{\mathrm{last}} < \varepsilon \quad\lor\quad \Bigl(\ \mathbf b \text{ constant over the last } W' \text{ epochs} \ \wedge\ \forall i:\ d_i \le 1.5\, d_{i-1} \ \wedge\ d_{\mathrm{last}} < 0.01\ \Bigr),
$$

where $d$ is the recorded drift and the middle condition ranges over consecutive entries of the window. The first disjunct means an epoch whose error falls below $\varepsilon$ converges regardless of topology. `convergence_score` is $0.4\,(1 - v/W') + 0.3\,e^{-10 d_{\mathrm{last}}} + 0.3\min(1, \varepsilon/e_{\mathrm{last}})$, with $v$ the number of Betti changes in the window. `BettiNumbers::distance` is the $L_1$ distance $\lvert\Delta\beta_0\rvert + \lvert\Delta\beta_1\rvert$ — integer-valued, which is the property the premise depends on. `ResidualAnalyzer::compute_betti` returns the sign-pattern pair $\bigl(1 + \lceil \mathrm{sc}/2\rceil,\ \lfloor \mathrm{dc}/4\rfloor\bigr)$, $\mathrm{dc}$ counting changes of monotonic direction. The interpreter does not call this detector.

**Consequences for the claims.** Topological convergence here does not eliminate tuning; it replaces a continuous threshold with a discrete window (3 epochs in the interpreter, $W'$ in the library). Neither predicate is a conjunction of topology and tolerance; both are disjunctions, so a loop may stop on the scalar alone. And the stopping signal of `regress` is a residual sign pattern rather than a Betti number of a filtration — a connection to the persistence engine that the design implies and that statement has not made. A seal loop can make it explicitly with `seal until stable(...)` over a `topology.betti` result (§4.6), with a one-pass window.

### 3.30 References

- Adamaszek, M. & Adams, H. (2017). The Vietoris–Rips complexes of a circle. *Pacific Journal of Mathematics* 290(1).
- Adams, H. et al. (2017). Persistence images: a stable vector representation of persistent homology. *JMLR* 18.
- Bauer, U. (2021). Ripser: efficient computation of Vietoris–Rips persistence barcodes. *Journal of Applied and Computational Topology* 5.
- Bubenik, P. (2015). Statistical topological data analysis using persistence landscapes. *JMLR* 16.
- Chazal, F., de Silva, V. & Oudot, S. (2014). Persistence stability for geometric complexes. *Geometriae Dedicata* 173.
- Chen, C. & Kerber, M. (2011). Persistent homology computation with a twist. *EuroCG 2011*.
- Chintakunta, H., Gentimis, T., González-Díaz, R., Jiménez, M.-J. & Krim, H. (2015). An entropy-based persistence barcode. *Pattern Recognition* 48(2).
- Cohen-Steiner, D., Edelsbrunner, H. & Harer, J. (2007). Stability of persistence diagrams. *Discrete & Computational Geometry* 37.
- Dao, T. et al. (2022). FlashAttention: fast and memory-efficient exact attention with IO-awareness. *NeurIPS 2022*.
- de Silva, V. & Carlsson, G. (2004). Topological estimation using witness complexes. *Eurographics Symposium on Point-Based Graphics*.
- de Silva, V., Morozov, D. & Vejdemo-Johansson, M. (2011). Dualities in persistent (co)homology. *Inverse Problems* 27.
- Edelsbrunner, H. & Harer, J. (2010). *Computational Topology: An Introduction.* AMS.
- Freund, Y. & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *JCSS* 55.
- Gower, J. C. & Ross, G. J. S. (1969). Minimum spanning trees and single linkage cluster analysis. *Applied Statistics* 18.
- Kingma, D. P. & Ba, J. (2015). Adam: a method for stochastic optimization. *ICLR 2015*.
- Milakov, M. & Gimelshein, N. (2018). Online normalizer calculation for softmax. arXiv:1805.02867.
- Sauer, T., Yorke, J. A. & Casdagli, M. (1991). Embedology. *Journal of Statistical Physics* 65.
- Takens, F. (1981). Detecting strange attractors in turbulence. *Lecture Notes in Mathematics* 898.
- Zomorodian, A. & Carlsson, G. (2005). Computing persistent homology. *Discrete & Computational Geometry* 33.

The derivations in this section are also rendered with figures on the [documentation site](https://teerthsharma.github.io/Aether-Lang/), whose [`docs/theory.md`](docs/theory.md) collects the theory in one page.

## 4. The language

### 4.1 Lexical conventions

Statements terminate with `~` rather than `;`. There is no technical reason; the choice is aesthetic, it makes the source recognisable at a glance, and it costs a new reader a few seconds.

The keyword `seal` has a second spelling, the four-byte codepoint `🦭` (U+1F9AD). The lexer maps both to the same token, so `🦭 until` and `seal until` are interchangeable; the ASCII spelling exists for terminals and tools that handle the codepoint badly.

```aether
let x = 10~
let xs = [1.0, 2.0, 3.0]~

fn dist(a, b) {
    return (a - b) * (a - b)~
}

if x > 5 { print("big")~ } else { print("small")~ }

for i in xs { print(i)~ }
```

Keywords the lexer recognises — a stricter list than "the language supports":

`let` `fn` `return` `if` `else` `for` `while` `in` `break` `continue` `class` `new` `self` `import` `from` `as` `true` `false` `seal` `until` `convergence` `escalate` `manifold` `embed` `block` `cluster` `regress` `render` `project` `dim` `tau` `model` `center` `spread` `color` `axis` `format` `output`

### 4.2 Statement grammar

Every variant the parser produces, from `StmtKind`:

| Statement | Form | Notes |
|---|---|---|
| `Var` | `let x = expr~` | |
| `Assign` | `x = expr~` | assigning an undefined name is a runtime error |
| `Fn` | `fn name(a, b) { ... }` | |
| `Return` | `return expr~` | |
| `If` | `if cond { ... } else { ... }` | |
| `While` | `while cond { ... }` | |
| `For` | `for i in iterable { ... }` | |
| `Loop` | `seal until expr { ... }` | the seal loop, §4.6 |
| `Break` / `Continue` | `break~` / `continue~` | |
| `Class` | `class Name { ... }` | with `new` and `self` |
| `Import` | `import topology~`, `from x import y~` | |
| `Manifold` | `manifold M = embed(data, dim=3, tau=5)~` | first-class declaration |
| `Block` | `block { ... }` | |
| `Regress` | `regress { model: "polynomial", escalate: true }~` | §4.7 |
| `Render` | `render { ... }~` | ASCII / WebGL export |
| `Expr` | any expression as a statement | |
| `Empty` | `~` | |

`Manifold`, `Block`, `Regress` and `Render` being **statement kinds rather than function calls** is the concrete meaning of "topology is a language primitive": they have parser rules and AST nodes, not entries in a builtin table. `Regress` is a statement, not an expression — `let r = regress { ... }~` is a parse error.

### 4.3 Expression grammar

From `ExprKind`: `Literal` · `Ident` · `BinaryOp` · `UnaryOp` · `FieldAccess` · `Call` · `MethodCall` · `Index` · `Config` · `New` · `Range` · `List` · `Member` · `Element`. `Member` and `Element` are postfix `.field` and `[i]` on any expression's value, added with the module bindings of §4.9; `Index` remains the manifold slice `M[a:b]`.

Binary operators `+ - * / % == != < > <= >= && ||` parse to `Add Sub Mul Div Mod Eq Neq Lt Gt Le Ge And Or`; unary operators are `Neg` and `Not`.

**`Config(Vec<ConfigPair>)`** is a first-class brace-delimited configuration literal — `{ model: "polynomial", escalate: true }` — which is what lets `regress` read as a declaration rather than a call with six positional arguments.

**`CallArg::Named { name, value }`** puts named arguments in the grammar rather than simulating them with a configuration object: `topology.ph(M, max_dim=1, mode="vr", max_points=16)` parses natively, and `max_dim` is distinguishable from `max_points` at the call site without consulting a signature. Call arguments may also be separated by whitespace alone, a leniency with a consequence described in §4.6.

The AST carries **source positions**, so `aether check` reports `line`, `column` and `message` rather than a panic backtrace.

### 4.4 Numeric literals

The lexer reads a decimal literal as an integer part and at most six fractional digits, right-padded to six, producing a fixed-point value

$$
\texttt{Float}(i, f) \;\longmapsto\; i + f\cdot 10^{-6}, \qquad 0 \le f < 10^{6}.
$$

Two consequences follow directly from `lexer.rs` → `fn read_number`, and both were confirmed by running the CLI. **Exponent notation is exact or refused**: `1e-6`, `2.5e3`, `1.5E-2` and `3e+2` scale the six-decimal fixed-point literal in integer micro-units (`fn read_exponent`), and a value the representation cannot hold — `1e-7`, or an overflow — is a lexer error, because rounding `1e-7` to zero would turn a tolerance into a loop that never stops. Until this revision there was no exponent notation at all and `1e-6` lexed as `1`, `e`, `-`, `6`. A **seventh fractional digit** is not consumed and becomes a separate integer token, so `let y = 0.1234567~` fails to parse. The smallest positive literal is `0.000001`.

The AST's `Number` type keeps the integer and fractional parts (`Number::Float { int_part, frac_part }`) and is used for configuration values, `for`-range bounds and `ConvergenceCond::Epsilon`; there a literal such as `0.1` is preserved exactly as written until `Number::as_f64` converts it. Ordinary expression literals are converted to `f64` at parse time (`Literal::Num(f64)`).

### 4.5 The topology module

```aether
import topology~

manifold M = embed(data, dim=3, tau=5)~

let diagram   = topology.ph(M, max_dim=2, mode="vr", max_points=16)~
let b         = topology.betti(diagram, radius=0.4)~
let intervals = topology.intervals(diagram)~
```

`topology.ph` calls the bounded persistence engine of §5.2 directly. The interpreter starts from `PersistenceConfig::low_load()` and overrides it from named arguments:

| Argument | Default | Meaning |
|---|---|---|
| `max_dim` | 2 | highest homology dimension, $K$ |
| `radius` | $\infty$ | filtration cap $R$ |
| `max_points` | 24 | point (or landmark) cap |
| `max_simplices` | 4,096 | simplex cap |
| `mode` | lazy witness, 24 landmarks | `"vr"`, `"rips"`, `"vietoris_rips"`, or `"witness"`, `"landmark"` |
| `landmarks` | `max_points` | landmark count in witness mode |

The caps are a **budget, not a correctness limit**: exceeding one returns `TooManyPoints` or `TooManySimplices` rather than quietly degrading or exhausting memory. With every default in place the budget binds early. By §3.2, the full 3-skeleton on $n$ points at infinite radius has $n + \binom n2 + \binom n3 + \binom n4$ simplices, which is 4,047 at $n = 18$ and $19 + 171 + 969 + 3{,}876 = 5{,}035$ at $n = 19$. `topology.ph(M)` with no arguments therefore fails with `TooManySimplices { max: 4096 }` on any cloud of 19 or more points, and a caller must pass `radius`, `max_dim=1` or `max_simplices` to go further. Both sides of that boundary were confirmed through the CLI.

`topology.betti(diagram, radius=r)` returns $[\beta_0, \beta_1, \beta_2]$ at $r$ by §3.7. `topology.intervals(diagram)` returns `[dimension, birth, death]` triples, with $-1$ as the death of an essential class.

### 4.6 Seal loops

The seal loop is `seal until <cond> { body }`, where the parser decides the kind of condition (`LoopStmt { until: Option<LoopCond>, body }`, with `LoopCond::{Expr, Stable, Convergence}`): `until stable(e)` and `until convergence(ε)` are fixed at parse time, and any other condition is an ordinary expression. The operational semantics of the expression form, from `interpreter.rs` → `fn execute_seal`:

$$
\mathtt{seal\ until}\ c\ \lbrace B\rbrace \ :\quad
\text{for at most } 1000 \text{ iterations: evaluate } c;\ \text{stop if } c = \mathtt{true};\ \text{error if } c \text{ is not boolean};\ \text{otherwise run } B,
$$

with `break`, `continue` and `return` propagating as in `while`. Nine of the Lean theorems in §7.4 exercise the corresponding rules of the Lean semantics on concrete programs.

```aether
let n = 0~
🦭 until n >= 3 {
    n = n + 1~
}
print(n)~
```

prints `3`.

**Tolerance seal loops.** `seal until convergence(ε)` has its own semantics, as `stable` does. Writing $v_i$ for the value of the body's last statement after pass $i$, the loop stops after the first pass $i \ge 2$ with

$$
d(v_i, v_{i-1}) \le \varepsilon,
$$

where $d$ is the max norm: $\lvert x - y\rvert$ on numbers, the maximum over elements on lists and records of equal shape ($+\infty$ when the shape changes), and the difference of `final_error` on regression results. A body whose value has no distance is refused by name rather than read as zero, and so is a negative $\varepsilon$. `convergence` is a keyword, so no program function can shadow it; the 1,000-pass ceiling still applies. Newton's iteration for $\sqrt2$,

```aether
let x = 1~
let n = 0~
🦭 until convergence(1e-6) {
    n = n + 1~
    x = (x + 2 / x) / 2~
    x~
}
print([n, x])~
```

prints `[5, 1.414213562373095]`: the fourth pass moves $x$ by $2.1\times10^{-6}$ and the fifth by $1.6\times10^{-12}$. With $\varepsilon = 0$ the same loop waits for an exact repeat, which floating point reaches at 1.414213562373095, one ulp below the correctly rounded $\sqrt2$. `crates/aether-lang/tests/seal_convergence.rs` pins both, the max-norm rule on lists, both refusals, the literals and the grouping.

Until this revision the spelling did not run: the example the README previously opened with, `🦭 until convergence(1e-6) { regress { model: "polynomial", escalate: true }~ }`, passed `aether check` and failed at runtime with `condition must be boolean`, because `convergence` was a keyword with no definition and `1e-6` lexed as `1`, `e`, `-`, `6`. Three changes make the loop run as written: the tolerance semantics above, exponent literals (§4.4), and parenthesised grouping, which the expression grammar also lacked — `(x + 2 / x) / 2` was a parse error. That example now reaches its body and stops with `Runtime error: no manifold for regression`, because `regress` needs a manifold in scope. A tolerance loop is a scalar stopping rule; a seal loop terminates on topology only when its condition computes something topological, which is what `stable` over a Betti vector does.

**Stable-invariant seal loops.** `seal until stable(expr)` is decided by the parser, so a program function named `stable` does not change its meaning; before the Titan reshape it was a run-time name check that such a function overrode. Writing $v_i$ for the value of `expr` evaluated before iteration $i$, the loop stops at the first $i \ge 1$ with $v_i = v_{i-1}$ — the first pass of the body that left the watched value unchanged — still within 1,000 iterations. Equality is exact and structural over numbers, booleans, strings, lists and records; values of other kinds are refused. `seal until stable(topology.betti(topology.ph(M), radius=r))` therefore stops on the Betti vector of a filtration, which is the construction the language's premise describes; §4.9 shows the same form over a certified face count. Note that exact equality makes $\beta$-stability a one-pass window: a vector that repeats once is accepted, the objection §2.1 raises against topological stopping without a window.

### 4.7 The regress statement and ConvergenceCond

The AST declares a three-way convergence condition:

```rust
pub enum ConvergenceCond {
    Epsilon(Number),              // scalar tolerance
    BettiStable { epochs: u32 },  // topological
    Custom(Expr),                 // arbitrary predicate
}
```

It appears only in the `until:` field of a `regress` configuration — never in a seal loop — and the parser always produces `Custom` there. `Epsilon` and `BettiStable` are never constructed. The interpreter reads a tolerance only from `Epsilon`, so every `regress` statement runs with $\varepsilon = 10^{-6}$ whatever its `until:` says, and the topological variant `BettiStable { epochs }` has no effect anywhere.

What `regress` does run is the interpreter's own `EscalatingRegressor`, whose stopping predicate is §3.29 (a): a disjunction of a scalar tolerance and a three-epoch repeat of a residual sign-pattern pair. It does not call `aether_core::ml::regressor::ManifoldRegressor` or `aether_core::ml::convergence::ConvergenceDetector`. `escalate: true` raises the epoch limit from 10 to 100; the model itself escalates on a fixed schedule by epoch.

### 4.8 Two execution engines

The tree-walking interpreter (`interpreter.rs`) is the reference: where the engines disagree, the interpreter's behaviour is the language's. `TitanVM` (`vm.rs`) compiles the AST to bytecode and fails closed. `Compiler::compile` returns `Result<Chunk, CompileError>`, and a construct it cannot compile is refused as `titan cannot compile CONSTRUCT at line L, column C`. `aether run --mode titan` prints that message after `titan: ` on stderr and exits 1. It never falls back to the interpreter, because a fallback would make every Titan measurement a measurement of the interpreter reported under Titan's name. Both CLIs run both engines through one `run_program`, so final-value printing (`Display`, with `()` hidden) and the footer are shared, and any difference in output is a difference between the engines. Titan's parity row is Partial while the reshape described here is in progress.

**Construct coverage.** The Titan column is filled from the parity report below.

| Construct | Interpreter | Titan |
|---|---|---|
| number, boolean and string literals | runs | see parity report |
| arithmetic, comparison and boolean operators | runs; `x / 0` is `inf` | see parity report |
| `let` and assignment | runs; assigning an unbound name is refused | see parity report |
| `if`, `while` | runs; a numeric condition is refused | see parity report |
| `for` over a range | runs; bounds truncated to integers, step ±1, iterator left bound | see parity report |
| `break`, `continue`, `return` | runs | see parity report |
| `fn` declaration and call | runs; late binding, lexical scope (below) | see parity report |
| named arguments | runs on natives; a user function refuses them | see parity report |
| `print` | runs; each argument evaluated, then printed | see parity report |
| lists, indexing and element access | runs | see parity report |
| records and field access | runs | see parity report |
| methods on values | runs | see parity report |
| `import` and module calls, including §4.9 | runs | see parity report |
| seal loop, `until expr` or no `until` | runs; at most 1,000 passes | see parity report |
| seal loop, `until stable(e)` | runs; §4.6 | see parity report |
| seal loop, `until convergence(eps)` | runs; §4.6 | see parity report |
| `manifold` and `block` declarations | runs | see parity report |
| `regress` | runs; §4.7 | see parity report |
| `render` | no-op | see parity report |
| `class` and `new` | partial | see parity report |

**The language change.** Holding two engines to one behaviour required writing the behaviour down, and three rules change what programs do. They are changes to the language, not engine details, and both engines implement them.

| Rule | Program | Before | After |
|---|---|---|---|
| lexical scope | `fn g() { y~ } fn f(y) { g()~ } f(3)~` | value `3`: `g` reads `f`'s `y` | value `()`: `y` is unbound in `g`, and an unbound read gives `()` |
| unbound call | `print(nosuch(1))~` | prints `()` and continues | `Runtime error: undefined function 'nosuch'`, exit 1 |
| loop form | `fn stable(v) { return v >= 2~ }`, then `seal until stable(n)` over a body that raises `n` to 3 | calls the program's `stable`; stops at `n = 2` | the parser's `Stable(n)`; stops when a pass leaves `n` unchanged, at `n = 3` |

Before, every call cloned the caller's whole variable map (`interpreter.rs` → `call_user_fn`). A function therefore read whatever its caller had bound, which is dynamic scope, at a cost proportional to the number of bound names; a bare `import` binds every export of a module. After, a function sees the globals read-only, plus its own parameters and locals. Its writes, including a write to a global's name, stay local and are discarded on return, and no call copies the environment. For a call made at top level the two rules agree, because there the caller's variables are the globals. They differ only when a function reads a name bound in its caller's frame. A write to a global inside a function was already discarded on return, so that is unchanged. The seal loop's condition is decided at parse time (`LoopCond::{Expr, Stable, Convergence}`), so a program function named `stable` no longer changes a loop's meaning at run time. Functions are still bound when their `fn` statement executes, reading an unbound variable still gives `()`, and a program's value is still the value of its last statement.

**Parity method.** `crates/aether-cli/tests/engine_goldens/` pins the reference. For every program in `examples/`, and for one small program per construct written from `interpreter.rs`, a golden records the interpreter CLI's stdout, stderr error line and exit code. The interpreter must reproduce every golden. The same corpus then runs under `--mode titan`, and each program is *matched* (all three equal), *refused* (a compile error) or *diverged* (it ran and something differs). A refusal is never counted as parity. Refused and diverged are separate ratchets that may only shrink, and the parity row moves to Active only when both are zero on the full corpus.

| Corpus | Programs | Matched | Refused | Diverged |
|---|---:|---:|---:|---:|
| `engine_goldens` | see parity report | see parity report | see parity report | see parity report |

**The benchmark gate.** Titan is kept only if it is at least 3× faster, by median, than the interpreter without its per-call clone, on both `fib(25)` and a $10^6$-iteration numeric loop. Each program also runs with `import monodromy~` above it, which exposes the old clone's dependence on scope size. The control is that fixed interpreter, not the one above, so the comparison credits Titan only with what an engine change buys. If the gate fails, `vm.rs` and `--mode titan` are deleted, and this section records the measurement that deleted them. The 3× threshold is a judgement, not a derivation: below it, a second engine doubles the cost of every language feature for too little gain.

<!-- numbers: pending benchmark -->

| File | Lines | Role |
|---|---:|---|
| `interpreter.rs` | 1,971 | tree-walking evaluator; topology builtins, seal loops, `regress`, manifold primitives |
| `parser.rs` | 1,269 | recursive descent, positioned AST |
| `vm.rs` | 911 | `TitanVM` bytecode VM |
| `lexer.rs` | 507 | tokeniser, including the four-byte `🦭` codepoint |
| `ast.rs` | 408 | node definitions above |
| `ascii_render.rs` | 179 | terminal rendering for `render` |
| `webgl_export.rs` | 117 | WebGL export path for `render` |
| `python.rs` | 81 | `pyo3` surface; the bindings package is empty |

### 4.9 The certified library

Ten modules extend the language with mathematics that answers only when it can prove the answer. Their definitions, theorems and derivations are in the [documentation](https://teerthsharma.github.io/Aether-Lang/integrated/); here they are language features. `import <module>~` binds its functions, `from linking import writhe~` binds one, and `coupling.coupling_fixed_point(...)` calls through the module name. Results are numbers, lists, strings, booleans or records read with `r.field`; reading a field a record does not have is an error, never 0. Verdicts are strings a program branches on (`r.verdict == "linked"`). Every typed refusal of the core reaches the program as a runtime error of the form `<module> refused: <Variant { data }>` — for example `arrangement refused: EdgesCross { first: 0, second: 1, count: 1 }` — never as a number standing in for the answer. Points are lists of numbers, zero-padded to the dimension a routine works in, and matrices are lists of rows.

| `import` | Functions | Returns |
|---|---|---|
| `linking` | `linking_number(a, b)`, `writhe(c)`, `knot_determinant(c)` | record `verdict` (`"linked"`, `"zero_linking"`, `"undetermined"`), `lk`, `value`, `error_bound`; a number; a number |
| `certify` | `certified_argmin(scores, radii)`, `certified_topk(scores, radii, k, largest=, ordered=)`, `certified_threshold(scores, radii, t)` | an index; a list of indices; a list of `"below"`, `"undetermined"`, `"above"` |
| `arrangement` | `euler(segments, grid=)` | record `pieces`, `faces`, `chi`, `vertices`, `edges`, `radius` |
| `resolvent` | `resolvent_attend(q, k, v, gates, switch)`, switch `"softmax"`, `"kernel"` or `"path"` | the readout, one value per row |
| `orbit` | `orbit_partition(values, gold=)` | record `n`, `m`, `error_floor`, `precision_floor`, `recovery_ceiling`, and with `gold` also `m_star`, `recall_floor`; an undefined bound is omitted |
| `monodromy` | `collision_certificate(f, sampler, sizes=, repeats=)`, `symmetry_group(points, step=)`, `ph_dimension(sampler, alpha=, sizes=, repeats=)`, `lyapunov(jacobians, dt=)` | records with `verdict` and the witness `p`, `q`; `order`, `dihedral`; `dimension`; a list of exponents. `f` and `sampler` are program functions the core calls back |
| `track` | `track(frames, gate, divide=true)` | record `nodes`, `edges`, `divisions`, `dividing`, `certified`, `cost` |
| `coupling` | `coupling_rollout(T, c, z0, steps)`, `coupling_fixed_point(T, c)`, `rips_islands(points, r)` | a list of states; record `point`, `converged`, `rho`, `contractive`; an island count |
| `kvwitness` | `witness_topk(learned, topk, context_len, segments)`, `segment_coverage(row, context_len, segments)`, `attention_mass_recall(row, attention)` | the merged row; numbers |
| `planner` | `plan_memory([[size, first, last], ...])`, `transitive_reduction(n, edges)`, `islands(n, incidences)` | record `offsets`, `arena`, `peak`; a list of edges; record `count`, `labels` (−1 for a node in no incidence) |

Two grammar additions make these quantities usable in conditions: postfix `.field` and `[i]` on any expression (`ExprKind::Member`, `ExprKind::Element`, §4.3), and the stable-invariant seal loop `seal until stable(expr)` of §4.6. `examples/integrated_mathematics.aegis` chains five modules — cells tracked across frames, their lineage forest counted as a planar arrangement, the same lineages recovered as planner islands, the cell-to-lineage map bounded by `orbit`, and a rounding-certified decision on the result — under a seal loop whose condition is a certified division count:

```aether
// Integrated mathematics: one pipeline across five certified modules.
//
//   track        cells over frames, divisions by min-cost circulation
//   arrangement  the lineage forest drawn in the (x, frame) plane: a forest
//                encloses no face, so faces = 0 and chi = number of lineages
//   planner      the same lineages as islands of the link graph
//   orbit        cell -> lineage as a many-to-one map, with its proved bounds
//   certify      a rounding-certified decision on the result
//
// and a seal loop whose termination condition is a certified quantity.

import track~
import arrangement~
import planner~
import orbit~
import certify~

// A cell at the origin divides; a second cell at x = 10 drifts right.
let frames = [
    [[0, 0, 0], [10, 0, 0]],
    [[2, 0, 0], [-2, 0, 0], [10.5, 0, 0]],
    [[2, 1, 0], [-2, 1, 0], [11, 0, 0]]
]~

// Seal loop: admit one frame per pass until a pass leaves the certified
// division count unchanged. Divisions go 0, 1, 1: sealed with all 3 frames.
let seen = [frames[0]]~
let k = 1~
🦭 until stable(track(seen, 12).divisions) {
    if k < 3 {
        seen.push(frames[k])~
        k = k + 1~
    }
}
let t = track(seen, 12)~
print(["frames admitted, divisions, certified", k, t.divisions, t.certified])~

// Every link [f, i, f', i'] becomes a planar segment (x, f) -> (x', f') and an
// incidence between node ids 3f + i and 3f' + i'.
let edges = t.edges~
let drawing = []~
let links = []~
let e = 0~
while e < edges.len() {
    let ed = edges[e]~
    let p = frames[ed[0]][ed[1]]~
    let c = frames[ed[2]][ed[3]]~
    drawing.push([[p[0], ed[0]], [c[0], ed[2]]])~
    links.push([ed[0] * 3 + ed[1], ed[2] * 3 + ed[3]])~
    e = e + 1~
}

let forest = euler(drawing)~
print(["forest: pieces, faces, chi", forest.pieces, forest.faces, forest.chi])~

let isl = islands(9, links)~
print(["lineages as islands", isl.count])~

// cell -> lineage over the 8 detected cells (node 2 is an unused id).
let lineage = []~
let v = 0~
while v < 9 {
    let label = isl.labels[v]~
    if label != -1 {
        lineage.push(label)~
    }
    v = v + 1~
}
let o = orbit_partition(lineage)~
print(["cells, lineages, identification error floor, recovery ceiling", o.n, o.m, o.error_floor, o.recovery_ceiling])~

// Three independent counts of the same thing must agree before we certify.
if forest.pieces == isl.count && isl.count == o.m && forest.faces == 0 {
    print("agreement: arrangement pieces = planner islands = orbit count")~
}

// Certified: which lineage holds the most cells? The counts are exact, so a
// radius of 0.25 still separates them and the argmax is certified.
let size0 = 0~
let size1 = 0~
let w = 0~
while w < lineage.len() {
    if lineage[w] == 0 {
        size0 = size0 + 1~
    } else {
        size1 = size1 + 1~
    }
    w = w + 1~
}
print(["lineage sizes, largest (certified argmax)", size0, size1, certified_topk([size0, size1], 0.25, 1, largest=true)[0]])~
```

`cargo run -p aether-cli -- run examples/integrated_mathematics.aegis`:

```
═══════════════════════════════════════════════════════════════
  🛡️ AEGIS - Running: examples/integrated_mathematics.aegis
  Mode: interpreter
═══════════════════════════════════════════════════════════════
[frames admitted, divisions, certified, 3, 1, true]
[forest: pieces, faces, chi, 2, 0, 2]
[lineages as islands, 2]
[cells, lineages, identification error floor, recovery ceiling, 8, 2, 6, 0.2]
agreement: arrangement pieces = planner islands = orbit count
[lineage sizes, largest (certified argmax), 5, 3, 0]

Execution complete. 🦭
```

The three independent counts agree — two arrangement pieces, two planner islands, two orbits. The lineage forest encloses no face, so its Euler characteristic equals its number of lineages ([`arrangement` docs](https://teerthsharma.github.io/Aether-Lang/integrated/arrangement/)), and the identification error floor is $n - m = 8 - 2 = 6$ ([`orbit` docs](https://teerthsharma.github.io/Aether-Lang/integrated/orbit/), Bound 1). The recovery ceiling 0.2 is $1/5$, the pooling bound for the larger lineage of five cells.

One program per module sits beside it. Each output below was produced by `cargo run -p aether-cli -- run examples/<file>`:

| File | Module | Output |
|---|---|---|
| `hopf_link.aegis` | `linking` | Hopf link `linked`, $\mathrm{Lk} = 1$, error bound 7.07e-14; a split pair `zero_linking`; writhe of a planar square 0; knot determinants 3 (trefoil) and 1 (square) |
| `certified_argmin.aegis` | `certify` | argmin 1; top-2 $[1, 2]$; argmax $[3]$; against 2.05: above, below, undetermined, above; `1.0 < 1.5` certified at radius 0.25 |
| `planar_faces.aegis` | `arrangement` | square: 1 piece, 1 face, $\chi = 0$; with half-diagonals 1, 4, $-3$ at snap radius 1.35e-6; a `stable` seal loop sealed at 4 spokes and 4 faces |
| `resolvent_switches.aegis` | `resolvent` | softmax $[1, 1, 1]$; kernel $[1, 2, 3]$; path $[0, 1, 1.5]$; one gate closed $[1, 1, 2]$ |
| `orbit_bounds.aegis` | `orbit` | $n = 6$, $m = 3$, error floor 3; precision floor 0.6; recovery ceiling $1/3$; $m^\ast = 3$, recall floor 0.5; an injective map has error floor 0 |
| `jacobian_free.aegis` | `monodromy` | $x^2$: collision exhibited, witness $0.146$ and $-0.146$; $2x + 1$: none at this sampling, $\lambda = 2$; square outline: order 4, dihedral; PH dimension 1.005 (segment) and 2.165 (square, the small-$n$ upward bias of §9); exponents $\pm\ln 2$ for $\mathrm{diag}(2, 1/2)$; logistic map at $r = 4$: 0.69314 |
| `cell_tracking.aegis` | `track` | 5 nodes, 1 division at node $[0, 0]$, certified; one-to-one mode: 3 links, 0 divisions |
| `coupled_rollout.aegis` | `coupling` | rollout $[1,1], [1.5,1.5], [1.75,1.75]$; fixed point $[2, 2]$, $\rho = 0.5$, contractive; a `stable` seal loop sealed at $[2, 2]$ after 55 steps; 2 islands at $r = 1.5$ |
| `kv_witness.aegis` | `kvwitness` | merged row $[13, 14, 0, 4, 8, -1]$; coverage 0.25 → 1; recall 0.25 → 0.3125 |
| `memory_planner.aegis` | `planner` | offsets $[0, 100, 0]$, arena 180 = peak 180; reduction $[[0,1],[1,2],[2,3]]$; 2 islands, labels $[0, 0, 0, -1, 1, 1]$ |

`crates/aether-lang/tests/integrated_modules.rs` holds 23 tests: one closed-form case and one refusal per module, the stable-invariant seal loop, single-symbol and module-method imports, and manifold slicing beside element indexing. Named arguments a function does not know are ignored rather than rejected (§9).

---

## 5. Implementation

### 5.1 Workspace

The workspace has seven members.

| Crate | Role | Status |
|---|---|---|
| `aether-core` | the mathematics: persistence, diagrams, attention, runtime substrate, ML; `no_std` | real |
| `aether-lang` | lexer, parser, AST, interpreter, Titan VM | real |
| `aether-kernel` | `no_std` x86_64 microkernel | real; compiles, not asserted to boot |
| `aether-cli` | `repl` / `run` / `check` | real |
| `aether-gpu` | wgpu compute backend, f32 | real; used by nothing outside itself |
| `aegis-core` | duplicate from a rename | queued for deletion (§5.12) |
| `aegis-cli` | duplicate from a rename | queued for deletion (§5.12) |

`aether-core` by module:

| Module | What it computes | Notable decision |
|---|---|---|
| `persistence` | $\mathbb{F}_2$ reduction over Rips and lazy witness filtrations, $H_0$–$H_2$ (§3.1–§3.9) | face lookup is a `BTreeMap` on the zero-padded vertex array; it was a linear scan (§6.2) |
| `diagram` | bottleneck, Wasserstein, landscapes, images, entropy (§3.11–§3.14) | exact, not approximate; the cubic Hungarian solver says so in a `ponytail:` comment |
| `attention` | sparse-attention reference kernel, seven selectors, routing plan, cost model (§3.18, §3.21, §3.22) | a CPU reference; no SIMD, no threading |
| `scheduled` | CSR block-scheduled attention and its backward pass (Triton port; §3.18–§3.20) | working set is one score tile, never `seq × seq` |
| `manifold` | delay embedding, streaming $\varepsilon$-graph, pipeline (§3.16, §3.17) | `embed` returns `Option`, encoding the warm-up in the type |
| `topology` | byte-level shape signature (§5.7) | shares vocabulary with `persistence` and none of its guarantees |
| `aether` | block summaries, hierarchical pruning, drift (§3.23, §3.24) | the pruning bound is admissible, so pruning is exact |
| `governor` | adaptive wake threshold (§3.26) | a law on $\ln\varepsilon$ with a proved local stability condition |
| `memory` | manifold heap, generational handles, Chebyshev guard (§3.25) | safety comes from marking, not from the inequality |
| `state` | `SystemState<D>` and three deviation metrics | $L_2$, $L_\infty$ and $L_1$ offered, not chosen |
| `ml` | tensors, layers, optimisers, clustering, classification, regression, convergence (§3.27–§3.29) | written from scratch, `no_std` |
| `linking`, `certify`, `arrangement`, `resolvent`, `orbit`, `monodromy`, `track`, `coupling`, `kvwitness`, `planner` | the certified library: certified linking numbers, rounding certificates, arrangement invariants, a three-corner attention operator, partition bounds, sampled-geometry decisions, tracking, coupling operators, segment witnesses, runtime planning | [§4.9](#49-the-certified-library) and the [documentation](https://teerthsharma.github.io/Aether-Lang/integrated/); each separates what it certifies from what it estimates |

The `no_std` claim is checked on a real target:

```bash
cargo build -p aether-core --no-default-features --features no_std \
  -Z build-std=core,alloc --target thumbv7m-none-eabi
```

That is a Cortex-M3, and it builds.

### 5.2 The persistence engine

`crates/aether-core/src/persistence.rs`, 1,043 lines, `no_std`.

**Simplex representation.** A simplex is a zero-padded fixed vertex array plus a vertex count, `([usize; 4], len)`, with `SIMPLEX_VERTICES = 4` because the engine enumerates up to tetrahedra — what $H_2$ requires and no more. A triangle $\lbrace 3, 7, 11\rbrace $ is stored as `([3, 7, 11, 0], 3)`. The padding is what makes the array a **canonical key**: `simplex()` zero-fills the unused slots and `boundary_indices` builds faces the same way, so two routes to the same simplex produce identical keys, and combinations are generated once each, so keys are unique. This single decision bought the 26× of §6.2; the previous code scanned `simplices[..before]` linearly for every face of every simplex, $O(m^2)$ in the simplex count, and put a ceiling of roughly 32 points on the engine.

**Configuration is a budget, not a limit.**

```rust
pub struct PersistenceConfig {
    pub max_homology_dim: usize,
    pub max_points: usize,
    pub max_simplices: usize,
    pub max_radius: f64,
    pub complex_kind: ComplexKind,
}
```

Four presets ship, their point caps derived from the measured timings of §6.3 rather than guessed:

| Preset | `max_homology_dim` | `max_points` | `max_simplices` | Complex | Rationale |
|---|---:|---:|---:|---|---|
| `h2_default()` | 2 | 48 | 1,000,000 | Vietoris–Rips | tetrahedra are the $O(n^4)$ term |
| `h1_dense()` | 1 | 128 | 1,000,000 | Vietoris–Rips | dropping $H_2$ buys a much larger point budget |
| `h0_only()` | 0 | 512 | 1,000,000 | Vietoris–Rips | components only; the cheapest useful configuration |
| `low_load()` | 1 | 24 | 4,096 | witness, 24 landmarks | the embedded and kernel profile; `const` so it can initialise a static |

The doc comment on `h2_default` states the policy: *the caps are a fail-fast budget, not a statement about correctness; raise them explicitly when the workload justifies the wait.*

**Failing fast.** `PersistenceError` has five variants, all refusals before or during construction: `InvalidDimension` ($K > 2$, or a malformed distance-matrix call), `InvalidRadius` (negative or NaN $R$, or an invalid distance matrix), `TooManyPoints { actual, max }`, `TooManySimplices { max }` and `EmptyInput`. Exceeding a cap never subsamples, never degrades to a coarser complex and never proceeds until the allocator gives up. `simplex_cap_still_fails_fast_rather_than_exhausting_memory` and `point_cap_is_still_enforced_when_configured` pin the behaviour, because the failure being prevented is the one that costs a machine, and an untested cap is one that quietly stopped working three refactors ago.

**Entry points.**

```rust
pub fn persistent_homology<const D: usize>(points: &[ManifoldPoint<D>], config: PersistenceConfig)
    -> Result<PersistenceDiagram, PersistenceError>;
pub fn time_delay_persistence<const D: usize>(samples: &[f64], tau: usize, config: PersistenceConfig)
    -> Result<PersistenceDiagram, PersistenceError>;
pub fn persistent_homology_from_distances(distances: &[f64], n: usize, config: PersistenceConfig)
    -> Result<PersistenceDiagram, PersistenceError>;
```

`time_delay_persistence` is the composition the language uses — delay-embed a scalar series, then compute persistence — as one function so the embedder and the engine cannot disagree about $\tau$. `persistent_homology_from_distances` builds the Rips filtration from any validated matrix (§3.1), which is how a filtration can come from a geodesic, a correlation distance, or distances computed elsewhere, such as on a GPU; witness mode is not supported there, since it needs landmark-to-witness distances from a larger set.

```rust
pub struct PersistencePair { pub dimension: usize, pub birth: f64, pub death: Option<f64> }
pub struct PersistenceDiagram { pub pairs: Vec<PersistencePair> }
impl PersistenceDiagram { pub fn betti_at(&self, radius: f64) -> BettiNumbers3; }
```

A diagram is a multiset, so comparisons go through §3.11 rather than element-wise equality. `death: None` marks an essential class, which is why it is an `Option` rather than `f64::MAX`.

### 5.3 The diagram module

`crates/aether-core/src/diagram.rs`, 475 lines. Public surface: `bottleneck_distance`, `wasserstein_distance`, `persistence_landscape` with `LandscapeConfig` (defaults: 3 levels, 64 samples on $[0, 1]$), `persistence_image` with `ImageConfig` (defaults: $32 \times 32$, $\sigma = 0.1$, window $[0,1]\times[0,1]$), `total_persistence`, `persistent_entropy`, `landscape_norm`. Every function first extracts and birth-sorts the finite bars of one dimension, an $O(b\log b)$ step. The mathematics is §3.11–§3.14.

### 5.4 The attention subsystem

`crates/aether-core/src/attention.rs`, 801 lines. A **CPU reference kernel**: no GPU, no SIMD, no threading.

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
| `Dense` | upper bound on quality and on cost | what every sparse method must beat on cost without losing quality |
| `Local { window }` | the trivial baseline | a sliding window; locality with no cleverness |
| `Random { budget, seed }` | **the floor** | a selector that cannot beat random at equal budget has no mechanism; deterministic per (seed, row) via SplitMix64 |
| `OracleTopK { budget }` | **the ceiling** | computes every score, then takes the true top-k; not deployable — a ruler |
| `Topological { budget, radius_scale }` | Euclidean-proximity selection within a radius relative to the row's median query–key distance | the claim that [measured negative](#8-negative-results-what-we-got-wrong) (§8.2) |
| `TopologicalRouted { budget, clusters }` | $H_0$-cluster routing on key directions, exact dot product within | the claim that survived, conditionally (§8.4, §3.22) |
| `Adaptive { budget, clusters }` | routed when the plan says it pays, dense when it does not | the shipping default; never worse than dense in cost or quality |

**Cost accounting, added after the fact.** `selection_dot_cost` and `dense_dot_cost` exist because every test in the original suite measured how good a selection was and none measured what it cost. That omission let a selector post +0.94 placement while examining 0.999× the dense dot-product count — dense attention wearing a clustering hat (§8.4). The general lesson: *a quality metric with no paired cost metric will eventually reward a method for doing more work.*

**The routing plan as a runtime check.** `routing_plan` computes the $H_0$ clustering once per key tensor — amortised over every query, head and layer that reuses it — and returns `cost_ratio`, `largest_cluster_share`, `gap_ratio`, `threshold` and `worth_routing`. It converts a conditional claim into a decision the code makes at run time rather than an assumption made at design time (§3.22).

**`single_linkage_clusters` and the theorem it is checked against.** Returns `(assignment, merge_heights)`. Labels are canonicalised to first-occurrence order, because otherwise they depend on union-find internals and two runs on the same data are not comparable — which is what makes `repeated_runs_are_bitwise_identical` meaningful. Ties break by index. The merge heights are the finite $H_0$ deaths of the same cloud (§3.8), and the suite asserts one implementation against the other. `normalize` projects points to the unit sphere first, making the clustering invariant to per-point rescaling — the property the routed selector needs and the nearest-neighbour selector lacks.

**Every selector materialises a dense `[seq, seq]` boolean mask.** No selector here is sub-quadratic in *memory*, however few keys it picks: the savings are real in dot products and absent in allocation. This is a property of a reference built for checkability, and it is the recorded trigger on the clustering routine's `ponytail:` comment — a neighbour graph saves nothing until the mask stops being dense.

### 5.5 Scheduled attention

`crates/aether-core/src/scheduled.rs`, 1,008 lines. Topology-derived sparse attention, which the author first wrote as a Triton kernel ([`triton-lang/kernels#22`](https://github.com/triton-lang/kernels/pull/22), merged), rebuilt in Rust. The Triton version requires CUDA; this compiles wherever `aether-core` does, including `no_std`.

```rust
pub struct BlockSchedule { pub offsets: Vec<usize>, pub indices: Vec<usize> }   // CSR
pub fn dense_causal_block_schedule(num_blocks: usize) -> BlockSchedule;
pub fn block_salience(keys: &[f64], seq: usize, dim: usize, block_size: usize) -> Result<Vec<f64>, ScheduleError>;
pub fn topology_block_schedule(/* keys, seq, dim, TopologyScheduleConfig */) -> Result<BlockSchedule, ScheduleError>;
pub fn inverted_topology_block_schedule(/* same */) -> Result<BlockSchedule, ScheduleError>;
pub fn scheduled_attention(/* q, k, v, seq, head_dim, &schedule, block_size */) -> Result<Vec<f64>, ScheduleError>;
pub fn dense_masked_attention(/* same */) -> Result<Vec<f64>, ScheduleError>;
pub fn scheduled_attention_backward(/* same, d_out */) -> Result<AttentionGradients, ScheduleError>;
pub fn block_mass_recovered(/* ... */) -> Result<f64, ScheduleError>;
pub fn random_block_schedule(budget: &[usize], seed: u64) -> Result<BlockSchedule, ScheduleError>;
pub fn oracle_block_schedule(/* ... */) -> Result<BlockSchedule, ScheduleError>;
```

**Why the split is the point.** The port preserves the original's separation of a combinatorial half from a numeric half, and that separation is what makes each half checkable alone:

| Half | Nature | Checkable against | Result |
|---|---|---|---|
| CSR block schedule | combinatorial — which blocks | the Python builder, **exactly** | `[0,1,3,6,10]` / `[0,0,1,0,1,2,0,1,2,3]` for 4 blocks |
| Kernel | numeric — what the blocks compute | dense masked attention | max abs diff **< 1e-12**, 4 schedules × 4 geometries |

A monolithic port would be checkable only end to end, where a schedule bug and a numeric bug are indistinguishable from a wrong output. Split, the schedule is compared by exact integer equality with the upstream builder — no tolerance and no judgement call.

`BlockSchedule::from_rows` validates every invariant at construction — non-empty, causal, strictly sorted, in range — so the kernel needs no per-row guards; `ScheduleError` names nine malformations, from `BlockSizeNotPowerOfTwo` to `ShapeMismatch`. `dense_masked_attention` lives in the shipping crate rather than the test file on purpose: it is the reference the sparse kernel is checked against, and a reference that lives only in tests tends to drift from the thing it references. The working-set claim — one score tile live at a time, never a `seq × seq` matrix — is what distinguishes block-scheduled attention from masked dense attention with extra steps, and `the_working_set_does_not_grow_with_the_sequence` measures it rather than asserting it in a comment.

**Reproduced:** the CSR schedule (exactly), the numeric output (1e-12), the **58.8% block reduction** at 16 blocks with `local_radius=1 sink=1 topk=2`, salience against the persistence engine's $H_0$ deaths (1e-9), and exactly one block scoring zero. **Not reproduced:** the upstream wall-clock figures (56.6% block reduction at seq 1024, 80.9% at seq 4096, 1.04×–3.48× sparse-vs-dense-CSR), measured on an RTX 4060. This port is a scalar CPU kernel; it reproduces the answer and the block reduction, not the speed.

### 5.6 The ML subsystem

`crates/aether-core/src/ml/`, 12 modules plus `mod.rs`, 5,075 lines, all `no_std`, all written from scratch against `libm` — 45% of the crate's source before the port. One caveat applies to all of it: **these are correct-and-small implementations, not competitive ones.** They exist so the language and the kernel have learning primitives that compile with no operating system underneath; each is outperformed by its `scikit-learn` equivalent by margins nobody has measured here. The mathematics is §3.27–§3.29.

**`tensor`** (304 lines). `Tensor { data, shape }` with shared storage and strided access. `from_vec` takes ownership, `new` copies, `zeros`/`ones`, `kaiming_uniform` (variance $1/n_{\mathrm{in}}$, §3.28), multi-index `get`/`set`, a naive triple-loop `matmul`, elementwise `add`/`sub`/`mul`, `scale`, `transpose`, `flatten`, `sum`, and `map` over `Fn(f64) -> f64`. `set` takes `&self` rather than `&mut self` — interior mutability, flagged here so a reader does not discover it from a type error. A comment at `ml/tensor.rs:6` reads *"future hooks for wgpu"*; `aether-gpu` exists, but nothing connects the two.

**`neural`** (685 lines). `Activation` (ReLU, Sigmoid, Tanh, Linear, LeakyReLU, Softmax) with `apply`, `apply_scalar` and `derivative` as peers, so each activation's gradient is testable independently — which is where activation bugs live, since a wrong derivative does not crash but trains to a worse optimum while the loss curve looks fine. `OptimizerConfig` (SGD with momentum, Adam) is split from `OptimizerState`, so configuration is cheap to share while per-parameter moment buffers live with the layer that owns the parameters. `DenseLayer`, `MLP`, `TrainingResult`.

**`autograd`** (363 lines). A tape-based (Wengert list) reverse mode over `Gc<Tensor>` handles stored in the manifold heap, with `add`, `mul`, `matmul`, `relu` and `backward`. **There is no finite-difference check for this module**, which is stated here rather than left for a reader to notice.

**`convolution`** (190 lines). `Conv2D`, bounded by compile-time constants so it needs no allocator: `MAX_KERNEL_SIZE = 5`, `MAX_CHANNELS_IN = 3`, `MAX_CHANNELS_OUT = 8`, `MAX_IMG_DIM = 32`. **Forward only**: a feature extractor, not a trainable layer.

**`regressor`** (383 lines). `ManifoldRegressor<D>` with `fit`, `predict` and `upgrade_model`, which promotes along linear → polynomial degrees 2–5 → RBF ($\gamma = 0.5$, doubling to 2) → Gaussian-process form (length scale halving from 1.0 while it exceeds 0.1) → geodesic, ordered by `ModelType::complexity`. Capped at `MAX_DEGREE = 8` and `MAX_POINTS = 256`; the degree cap is a cheaper defence against ill-conditioned high-degree fits than a condition-number check. The fit methods are §3.28. Despite the name, this is **not** what the language's `regress` statement calls (§4.7).

**`clustering`** (736 lines). `KMeans<D>` with `with_max_iter`, `with_tol` and `with_seed` — the seed matters, since an unseeded k-means makes every downstream test flaky in a way that gets diagnosed as a tolerance problem for a week; `DBSCAN<D>`; `AgglomerativeClustering<D>` with `Linkage` and `cut_tree(result, k)`; `auto_k_selection`, which returns $\beta_0$ of the $\varepsilon$-graph (§3.17). `cut_tree` performs the same operation as the $H_0$ cut in `single_linkage_clusters` from the classical direction — two implementations of overlapping mathematics that, unlike the attention one, are not cross-checked against each other or against the persistence engine.

**`classification`** (780 lines). Bounded by `MAX_CLASSES = 16`, `MAX_POINTS = 256`, `MAX_FEATURES = 32`.

| Type | `fit` returns | Predicts |
|---|---|---|
| `LogisticRegression` | `f64` (final loss) | `predict_proba` → `f64`, `predict` → `u32` |
| `KNNClassifier<D>` | — | `u32` |
| `Perceptron` | — | `i32` (sign convention) |
| `GaussianNB` | — | `u32` |
| `DecisionStump` | — | `i32`; AdaBoost's weak learner |
| `AdaBoost` | — | `i32` |

**`convergence`** (406 lines). `BettiNumbers { beta_0, beta_1 }` with `is_singular` ($\beta_0 = 1$, $\beta_1 = 0$) and the integer `distance`; `ConvergenceDetector::{new, record_epoch, is_converged, convergence_score}`; `ResidualAnalyzer<D>::{set_residuals, compute_betti, compute_drift, is_collapsed}`; `Answer::from_detector`, which returns `Option` so that "no answer yet" is representable rather than encoded as a sentinel. The predicates are §3.29 (b); the module is exercised by its own tests and is not called by the interpreter.

| Module | Lines | Surface |
|---|---:|---|
| `gossip` | 243 | ring averaging, `MAX_DIM = 3`, at most 16 nodes (§3.27) |
| `dataloader` | 334 | `DataLoader`, `BatchIterator<'a>`, batching and shuffle |
| `linalg` | 296 | scalar reductions, distance primitives, `LossConfig` |
| `benchmark` | 293 | `EscalatingBenchmark<D>`, `BenchmarkConfig`, `TestFunction`, `generate_test_function`, `MAX_EPOCHS = 1000` |

`benchmark` is an **internal harness for the escalation policy**, not a performance benchmark, and no number in this document comes from it; the name is flagged rather than defended.

### 5.7 The runtime substrate

The parts of `aether-core` that are not mathematics in the TDA sense: how the runtime allocates, adapts, prunes and observes. They are what justify the phrase "a runtime that also owns the scheduler and the allocator".

**`memory`** — the manifold heap (553 lines). A `no_std` allocator organising objects spatially rather than by free-list order: `Gc<T>` (an index plus a generation counter), `ObjectHeader`, `HeapSlot<T>`, `SpatialBlock<T>` (eight slots, 64-byte aligned, contiguous liveness array), `SpatialNode`, `ManifoldHeap<T>`, `MemoryMode`. `Gc<T>` is a **generational handle**, not a pointer: a stale handle whose slot has been reused fails the generation check and returns `None` rather than aliasing whatever now lives there. `Copy` and `Clone` are implemented by hand so no `T: Copy` bound leaks onto the handle. Liveness lives in the block rather than the slot so a sweep reads one packed array; the source comment attributes this to SIMD, but there is no SIMD in the crate, so the layout permits vectorisation rather than performing it. `alloc`, `get`, `get_mut`, `touch`, `mark`, `active_count`, `capacity`; `touch` records access without reading the value. The collector and its guard are §3.25.

**`governor`** (323 lines). The adaptive wake threshold of §3.26: `new`, `with_epsilon`, `with_gains` (the calibration knob a real system needs, since a clock drifts, a sensor reads off, and a workload is not the one the gains were picked against), `adapt`, `should_trigger`, `reset`. Shipped constants:

| Constant | Value | Role |
|---|---:|---|
| `TARGET_TICK_RATE` | 1000.0 Hz | $R^\star$, the wake rate the governor holds |
| `ALPHA` ($\alpha$) | 0.25 | proportional gain on $\ln\varepsilon$ |
| `BETA` ($\beta$) | 0.05 | derivative gain on the per-step change in error |
| `EPSILON_MIN` | 0.001 | floor; prevents runaway sensitivity |
| `EPSILON_MAX` | 10.0 | ceiling; prevents sleeping through events |
| `EPSILON_INITIAL` | 0.1 | starting threshold |

**`aether`** — hierarchical block pruning (544 lines). `BlockMetadata<D>` (centroid, radius, distance variance, concentration, count), `HierarchicalBlockTree<D>` over `BLOCK_LEVELS = [64, 256, 1024]` with at most `MAX_BLOCKS = 128` fine blocks, `CompressionStrategy`, `select_compression`, `estimate_compression_ratio` and `DriftDetector<D>` over a 32-entry history. The bound and its proof are §3.23; drift is §3.24.

**`manifold`** — embedding and the streaming pipeline (728 lines). `ManifoldPoint<D>`, `TimeDelayEmbedder<D>` (§3.16), `SparseAttentionGraph<D>` (§3.17), `GeometricConcentrator<D>` (Welford statistics, §3.28) and `TopologicalPipeline<D>`, whose `push` is the whole streaming path in one call: drop samples with $\lvert x\rvert < 10^{-9}$, embed, add the point to the graph, read $(\beta_0, \beta_1)$ of the graph, and project the point either onto the highest-variance axis (when $\beta_1 = 0$) or by distance to the centroid of its depth-3 graph neighbourhood (otherwise), returning `Option<(β₀, β₁, u64)>` with a hash of the point and projection. That is what makes `manifold M = embed(data, tau=1)` a streaming construct rather than a batch one.

**`topology`** — the byte-level module, and a naming hazard (422 lines). It shares vocabulary with the persistence engine and operates on entirely different input:

```rust
pub fn compute_betti_0(data: &[u8]) -> u32;
pub fn compute_betti_1(data: &[u8]) -> u32;
pub fn compute_shape(data: &[u8]) -> TopologicalShape;
```

For bytes $x_1,\dots,x_N$, call position $i$ a *gap* when $\lvert x_{i+1} - x_i\rvert > 15$ (`CLUSTER_THRESHOLD`). Then `compute_betti_0` is the number of maximal runs of consecutive gaps, and `compute_betti_1` is the number of 4-windows $(a, b, c, d)$ with $\lvert a - d\rvert \le 5$ and $\max(\lvert a-b\rvert, \lvert a-c\rvert) > 5$ — a return-to-start count. `TopologicalShape { betti_0, betti_1, density }` has $\mathrm{density} = \beta_0/N$, and `distance` is the Euclidean norm of the difference of the triples. `verify_shape` passes when $0.1 \le \mathrm{density} \le 0.6$ (`DENSITY_MIN`, `DENSITY_MAX`) and $\beta_1 \le 10$ (`MAX_BETTI_1`). `MAX_BETTI_1` is a **rejection threshold** on an unnormalised count, not a saturating cap: `compute_betti_1` reports the raw count and `verify_shape` fails with `ExcessiveLoops` above 10. `verify_sliding_window` applies the same thresholds to every window of `WINDOW_SIZE = 64` bytes with an $O(N)$ incremental update, returning `Err(offset)` at the first failure — the position rather than a bare `false`.

| | `topology::compute_betti_0` | `persistence::persistent_homology` |
|---|---|---|
| Input | `&[u8]` byte slice | `&[ManifoldPoint<D>]` |
| Output | `u32` | full `PersistenceDiagram` |
| Method | run count of byte gaps | exact $\mathbb{F}_2$ column reduction |
| Exact? | a heuristic, not homology | exact, invariant-tested |
| Cost | $O(N)$ | §5.13 |

Three functions in the crate are named some variant of `compute_betti_0` — this one, `SparseAttentionGraph::compute_betti_0`, and the engine's $H_0$ — and only the last is persistent homology. **None of the 12 persistence invariants apply to this module.**

**`state`** (224 lines). `SystemState<D> { vector, timestamp }` with `deviation` ($L_2$), `max_deviation` ($L_\infty$), `manhattan_deviation` ($L_1$), `magnitude` and `elapsed_since`. The three metrics answer different questions — total displacement, worst single component, summed change — and the scheduler's trigger $\Delta \ge \varepsilon$ wakes on different events under each: $L_\infty$ fires when one component spikes, $L_1$ on diffuse drift that $L_\infty$ would sleep through. Exposing all three rather than picking one is right for a substrate whose workload is unknown at design time. The kernel scheduler currently uses $L_2$. Carrying `timestamp` inside the state lets the scheduler compute the elapsed time it passes to the governor without a separate clock source, relevant in a kernel where "what time is it" is not a free question.

### 5.8 aether-lang

Lexer → AST → recursive-descent parser → tree-walking interpreter, plus the `TitanVM` bytecode VM, with the file sizes in §4.8. The interpreter holds the topology builtins, seal-loop execution, the `regress` escalation loop and the manifold primitives, and imports from `aether-core` the persistence engine, delay embedding, block metadata, drift detection, `Conv2D`, `Tensor`, `KMeans` and `MLP`. The VM uses the manifold heap. Positioned AST nodes are why `aether check` reports `Parse error at line L, column C: message`, a small thing that is disproportionately visible to anyone who uses the language. `python.rs` exposes a `pyo3` surface; `pyproject.toml` builds it through `maturin`, and the Python bindings package is empty.

`aether-lang` no longer pulls `wgpu`, `reqwest`, `safetensors`, `pollster` or `bytemuck`. All five sat in its **default** feature set with zero call sites, so every default build compiled an HTTP client and a 2023-era GPU stack that nothing referenced; all five are deleted, and a comment in its `Cargo.toml` records why.

### 5.9 aether-kernel

A `no_std` x86_64 microkernel that links `aether-core` and uses it to make scheduling decisions. It **compiles** for `x86_64-unknown-none`:

```bash
cargo build -p aether-kernel -Z build-std=core,alloc --target x86_64-unknown-none
```

It is **not asserted to boot**. That distinction is maintained throughout this document because it is the one most often blurred: compiling proves the code type-checks and links against a bare-metal target; booting requires QEMU logs and a hardware matrix, and neither exists here.

**`allocator` — a bump allocator on a static heap.** `HEAP_SIZE = 64 * 1024`, a `static mut` byte array, a spin-locked `BumpAllocator`, and `#[global_allocator] static GLOBAL_ALLOCATOR: AegisAllocator`. It hands out sequential pointers and **never frees** — right for a kernel that boots, sets up and runs a fixed workload, wrong for anything that allocates in a loop. It is a different allocator from the manifold heap of §5.7: that one has generational handles and entropy-regulated collection, this one is 64 KB of bump, and the naming does not make the difference obvious.

**`interrupts` — the state vector lives in interrupt context.** `IDT` (a `spin::Lazy<InterruptDescriptorTable>`), `CURRENT_STATE: Mutex<SystemState<STATE_DIMENSION>>`, `IRQ_COUNTER` and `TIMESTAMP` behind spin mutexes, with `get_current_state` and `update_state_component`. The state the governor and scheduler read is maintained by the interrupt handlers themselves, not by a userspace abstraction observing the kernel — the load-bearing piece of the claim that topology makes execution decisions. It requires `#![feature(abi_x86_interrupt)]`, which pins the toolchain to nightly independently of `-Z build-std`.

**`scheduler` — sparse and deviation-triggered.** `SparseScheduler<D>` wakes when $\Delta = \lVert \mu(t) - \mu(t_{\mathrm{last}})\rVert_2 \ge \varepsilon$, with $\varepsilon$ supplied by the governor (§3.26), and issues `WFI` between events. A timer-tick scheduler does fixed work per unit time whether or not anything happened; a deviation-triggered one does work in proportion to how far the state moved, which is the whole power argument. On each event it passes the governor the elapsed time in seconds from the state timestamps, falling back to `DEFAULT_DT = 0.001`. `ENTROPY_MULTIPLIER = 6364136223846793005` is the multiplier of Knuth's MMIX linear congruential generator; naming it matters, because a reader who does not recognise it cannot tell a well-chosen multiplier from a typed number.

**Seven tests in this crate never execute**: four in `scheduler.rs`, two in `loader.rs`, one in `interrupts.rs`. `aether-kernel` is excluded from the host test job because it cannot link for a host target (§7.5), and nothing else runs them. Tests that cannot run are documentation with a misleading syntax highlight.

**`loader` — ELF checks and the byte-topology gate.** `ELF_MAGIC`, `LoadError`, `ElfInfo`, `verify_elf` (ordinary header validation) and `verify_binary_topology`, which is `topology::is_shape_valid` applied to the whole image (§5.7). Two properties follow from the definitions. Because $\beta_1$ there is an unnormalised count compared against an absolute cap of 10, the whole-image check rejects any input with more than ten qualifying 4-byte windows, regardless of its length. And it is **not a security mechanism**: an adversary who knows the check exists can pad a payload to match a byte-distribution signature. It is a structural plausibility heuristic that would catch some truncated or corrupted images; cryptographic signing is what authenticates code, and the word "authentication" in the source is an overclaim that this section corrects rather than repeats.

**`boot` — BIOS handoff.** `BootInfo`, `MemoryRegion`, `MemoryRegionKind`, `Framebuffer`, `HardwareTopology`, `IoCaps`. This is where a latent bug was found once the crate was made to compile again. `BootInfo::config_root` returned a pointer to the eight-byte **RSDP signature string** where the ACPI root table address was intended; the RSDP begins with the literal bytes `"RSD PTR "`, so any ACPI enumeration built on it would have parsed ASCII as a table header. The defect had been unreachable for the life of the repository because CI triggered on `main` and `develop` while the branch is `master`, so no run had ever executed and the crate had stopped compiling. Fixing the workflow surfaced four compile defects — including `multiboot2::load`, removed in 0.24 — and this fifth, genuine one beneath them. A compile error is not a bug's hiding place; it is its roof.

**`serial` — the only output channel.** `COM1 = 0x3F8` behind a spin-locked `SerialPort`. It is the channel on which the missing QEMU logs would arrive, so the path from "compiles" to "boots" runs through this file.

### 5.10 aether-cli

`repl`, `run` and `check`, with `rustyline` for line editing and `clap` for arguments. It accepts `.aether`, `.ae` and the pre-rename `.aegis` silently and warns on any other extension, including the `.ag` of three examples — extension debt in plain sight. The execution banner still prints `AEGIS` and `Mode: bio`; the mode is a label and does nothing biological.

```bash
cargo run -p aether-cli -- run program.aether   # execute
cargo run -p aether-cli -- check program.aether # parse only, no execution
cargo run -p aether-cli -- repl                  # interactive
```

### 5.11 aether-gpu

A wgpu compute backend in f32 (WGSL has no f64): 20 WGSL kernels — matmul, tiled matmul, pairwise distance, softmax, fused gradients, Adam, SGD and the scheduled-attention forward and backward — resident tensors, batched submission, 107 tests, measured on an RTX 4060 Laptop GPU over Vulkan. **Nothing outside the crate calls it.** Within it, `scheduled_attention_or_cpu` and its backward counterpart route between the GPU and CPU implementations by capability.

It is a separate crate rather than a feature of `aether-core` deliberately: `aether-core` is `no_std` and builds for `thumbv7m-none-eabi`, while `wgpu` needs `std` and a driver stack. A feature flag would leave a `no_std` crate whose dependency graph resolves only on hosted targets. The hardware tests are `#[ignore]`d unless the `gpu` feature is on (§1.5).

Both candidate integrations into `aether-core` are measured, and neither is made:

- **`Tensor::matmul`** crosses over at $n = 128$ with f64↔f32 conversion counted. *How much* it pays above that is not measurable on this machine: the same ratio has been observed anywhere from 10× to 63× at $n = 512$, and printing the individual timings showed why — across six runs the CPU term moved 1.6× while the GPU term moved 5.2×, so the ratio inherits a swing that repetition and interleaving do not remove. An earlier version of this text quoted 38× as though it were a property of the code. The crossover is the durable claim, because a threshold tolerates noise that a magnitude does not. Precision is 5e-7 relative — adequate for training, not for assertions at 1e-9 — so wiring it in means deciding whether `ml::Tensor` may drop to f32, a semantic change no benchmark authorises.
- **`pairwise_sqdist`** never pays: the persistence reduction is CPU-side and sequential, so the distance matrix has to come back.

The full ledger — measurements, negative results and corrections, including the selector ablation and the recall-training experiment of §6.5 — is [`crates/aether-gpu/FEATURES.md`](crates/aether-gpu/FEATURES.md), whose headline counts are bound to the tree by `crates/aether-gpu/tests/features_doc.rs`.

### 5.12 Duplicate crates

The project was renamed from AEGIS to AETHER **by copying directories rather than moving them**, and the consequences are still visible:

- `aegis-core` and `aegis-cli` are near-copies of their `aether` counterparts. `aegis-cli` declares `aegis-core` in its manifest and never uses it in source; it is built on `aether-lang` under the alias `aegis_lang`, and its `main.rs` differs from `aether-cli`'s in naming and in the positioned parse-error reporting added to `aether-cli` since.
- `crates/aegis-core/src/ml/autograd.rs` is 298 lines that Cargo has never compiled, since no `mod` reaches it; `tests/orphaned_sources.rs` exists because of it.
- Half the doc comments across the tree still say AEGIS: `ml/tensor.rs` opens with *"AEGIS Tensor Engine"*, the kernel's global allocator is `AegisAllocator`, and the CLI banner prints `AEGIS`.
- Repository examples use `.aegis` and `.ag`.
- On one point the duplicate is better configured than the original: `aegis-core` declares `nalgebra` as `optional = true`, while `aether-core` declares it unconditionally with zero call sites (§8.6).

The state of the repository is part of what a reader is evaluating, and a reviewer who discovers duplicate crates reasonably wonders what else is unstated. Deletion is queued; CI meanwhile builds `aegis-core` for `thumbv7m-none-eabi` so that its `no_std` configuration is at least compiled.

### 5.13 Complexity reference

$n$ is the point count, $m$ the simplex count, $s$ the sequence length, $d$ the dimension, $b$ the bar count, $\lvert L\rvert$ the landmark count, $B$ the block size and $N_b$ the number of fine blocks.

**Persistence**

| Operation | Cost | Note |
|---|---|---|
| Rips enumeration, $K$ = `max_homology_dim` | $O(n^{K+2})$ | tetrahedra are the $O(n^4)$ term — the reason `h2_default` caps at 48 points (§3.2) |
| Pairwise distances | $O(n^2 d)$ | |
| Face lookup (`BTreeMap`) | $O(\log m)$ per face | was an $O(m)$ linear scan (§6.2) |
| $\mathbb{F}_2$ column reduction | $O(m^3)$ worst case | measured scaling is close to $m^{1.5}$ (§6.3) |
| `betti_at(radius)` | $O(b)$ | a query on a computed diagram |
| Witness filtration | $O(n\lvert L\rvert^2 + n\lvert L\rvert^{K+2})$ | maxmin selection, then one witness scan per candidate simplex (§3.9) |

**Diagram metrics**

| Operation | Cost | Note |
|---|---|---|
| Bottleneck distance | $O(b^3\log b)$ | binary search over candidate costs × Kuhn's matching |
| $p$-Wasserstein | $O(b^3)$ | Hungarian, exact; `ponytail:` ceiling, trigger at `max_points` > 2048 |
| Landscape, $r$ samples | $O(r\, b\log b)$ | per-sample descending sort of the positive tents |
| Persistence image, $p$ pixels | $O(b\,p)$ | every bar deposits on every pixel |
| Total persistence, entropy | $O(b\log b)$ | the birth sort dominates the $O(b)$ sum |

**Attention**

| Operation | Cost | Note |
|---|---|---|
| Dense attention | $O(s^2 d)$ | the reference everything else is measured against |
| `single_linkage_clusters` | $O(s^2 d + s^2\log s)$ | edge enumeration + sort; `ponytail:`-marked, not the binding term while the mask is $\Theta(s^2)$ |
| `select_mask`, any selector | $\Theta(s^2)$ memory | **the dense `[s, s]` bool mask dominates every selector's asymptotics** |
| `OracleTopK` | $O(s^2 d)$ | computes every score; a ruler, not a method |
| `routing_plan` | $O(s^2 d + s^2\log s)$ | the clustering, run for the plan and again for the cost model |
| Scheduled attention, `nnz` scheduled blocks | $O(\mathrm{nnz}\cdot B^2 d)$ | working set is one tile |

**Runtime substrate**

| Operation | Cost | Note |
|---|---|---|
| `ManifoldHeap::alloc` / `get` | $O(1)$ | free list; index plus generation check |
| `ChebyshevGuard::calculate` | $O(\text{capacity})$ | one pass over occupied slots |
| `hierarchical_query` | $O(N_b)$ iterations, $O(d)$ per bound | bounds evaluated only under unpruned parents |
| `GeometricGovernor::adapt` | $O(1)$ | one exponential |
| `TimeDelayEmbedder::push` / `embed` | $O(1)$ / $O(D)$ | ring buffer |
| `SparseAttentionGraph::add_point` | $O(n)$ | bitset row per point, first 64 points only |
| `SparseAttentionGraph::compute_betti_0` | $O(n\cdot\min(n, 64))$ | depth-first search over bitsets, 64-entry stack; not union-find |
| `estimate_betti_1` | $O(n)$ plus the $\beta_0$ search | exact cycle rank of the graph, **not** persistent homology (§3.17) |

Asymptotics are not the measurement. The measured ceiling (§6.3) is $H_0$ at $n = 4{,}000$ in **335 s**, $H_1$ at $n = 300$ in **131 s** and $H_2$ at $n = 70$ in **15.3 s**, all single-threaded and `--release`. Nothing in the engine is parallelised or vectorised; the presets exist to keep a caller inside those numbers rather than discovering them.

### 5.14 Design decisions

**The tilde.** `~` terminates statements. Not `;`, because `;` is what everything else uses and the source was meant to look different at a glance. An aesthetic argument, not a technical one, stated as such.

**The seal codepoint.** `🦭 until ...`: a four-byte codepoint as a control-flow keyword. The lexer handles it; `grep` and some terminals may not, which is why `seal until` exists.

**Not special-casing $H_0$ to union-find.** Union-find would make $H_0$ near-linear instead of the 335 s that $n = 4{,}000$ takes. It is deliberately not done: the invariant suite tests the engine's $H_0$ *against* an independent union-find, and making the engine use union-find would turn that test tautological. The check is worth more than the speed.

**Denying only `correctness` and `suspicious` in clippy, plus every rustc warning.** Style, complexity and performance lints are allowed in the gate and printed by a separate advisory step that never fails. A `-D warnings` gate over every clippy group fails on first contact and gets switched off within a week, and then the correctness lints stop being enforced too: strict where lints find bugs, quiet where they find taste. The advisory step prints the current list rather than a count, because a count in a comment is exactly the kind of claim that goes stale.

**Keeping a guard that provably never fires.** `boundary_indices` checks that each face index precedes its coface's, although `every_face_is_present_and_precedes_its_coface` proves it always does. The guard is free, and a malformed complex degrades instead of lying.

**Errors instead of truncation.** Every cap in the persistence engine returns an error rather than a partial answer (§3.2), because a partial complex computes a diagram that looks valid and describes a shape nobody asked about.

**11,637 lines of Lean.** `Aether/` holds a Lean 4 formalization with 48 theorems and no `sorry`. It is not built by CI, and the 8,281 lines of `Lexer`, `Parser`, `Pipeline`, `Static` and `VM` hold exactly one theorem between them (§7.4). Either gate it or cut it; the ledger lists it as ungated, which is the honest interim state.

---

## 6. Evaluation

### 6.1 Measurement substrate

Everything measured in this document ran on a **single machine**: Windows 11, Rust nightly, single core, `--release` where noted. **No confidence intervals, single runs, no core pinning, no turbo control.** These are engineering measurements taken to size caps and catch regressions, not a study. The 26× of §6.2 is robust to all of that, because it is a 26× difference on identical assertions on the same machine within one session; the absolute timings of §6.3 are not, and should be read as orders of magnitude.

### 6.2 The face-index refactor

`find_simplex` scanned `simplices[..before]` linearly for every face of every simplex, making face lookup $O(m)$ and the reduction $O(m^2)$ in lookups alone. It is now a `BTreeMap` keyed on the zero-padded vertex array (§5.2), $O(\log m)$ per face.

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

The invariant suite (11 tests at the time; 12 now) existing *before* the refactor is the reason it was safe. That is what property tests buy: not bug-finding, but permission to change things.

### 6.3 Measured scale ceiling

`cargo run -p aether-core --example scale_probe --release`, regular circle, single core, $R = \infty$.

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

**The pairs column is predicted exactly.** Every entry equals the closed form of §3.6 — $n$ for dimension 0, $\binom n2 + 1$ for dimension 1, $\binom n3 + n$ for dimension 2 — because at $R = \infty$ the number of pairs depends only on $n$ and $K$.

**Empirical scaling.** Between consecutive rows the local exponent $\hat p = \ln(t_2/t_1)/\ln(n_2/n_1)$ is 2.96 and 2.93 in dimension 0, 4.23 to 4.55 in dimension 1, and 5.72 to 6.27 in dimension 2. Measured instead against the simplex count $m = \sum_{k\le K+1}\binom{n}{k+1}$, every one of the seven intervals gives an exponent between 1.41 and 1.56: running time is close to $m^{1.5}$ in all three dimensions, well inside the $O(m^3)$ worst case of §3.5. Both are derived from the single-run timings above and inherit their caveats.

Presets are sized to these timings: `h2_default` 48 points, `h1_dense` 128, `h0_only` 512. **The caps are a time budget, not a correctness limit.** Raise them explicitly and wait longer. For context, ripser routinely handles clouds of tens of thousands of points; this engine does not. The trade purchased is `no_std` with a minimal dependency surface and the ability to run inside a kernel.

### 6.4 Exactness against closed form

| Property | Assertion | Result |
|---|---|---|
| Circle $H_1$ death | $2r\sin(\pi\lceil n/3\rceil/n)$, 8 values of $n$ (§3.15) | exact to **1e-12** |
| $H_0$ deaths | equal an independent union-find MST (§3.8) | exact to **1e-9** |
| Permutation invariance | shuffled rows give an identical diagram | **1e-9**, 5 seeds |
| Isometry invariance | rotate 0.9128 rad + translate | bottleneck **< 1e-9** |
| Scale equivariance | $c \in \lbrace 0.125, 0.5, 2, 37\rbrace $ | exact |
| Stability | $\lVert\delta\rVert \le \varepsilon \Rightarrow d_B \le 2\varepsilon$ (§3.10) | 12 seed × $\varepsilon$ cases pass |
| Negative control | Gaussian blob | **0** long $H_1$ bars, 4 seeds |
| $\partial\circ\partial = 0$ | every simplex, 5 complexes (§3.3) | exact |
| Diagram size | closed form of §3.6 | 10 of 10 probe rows (not a test) |

The Gaussian blob is worth as much as any positive case: a pipeline that finds structure in noise finds it everywhere, and every downstream claim built on it is unfalsifiable.

### 6.5 Scheduled attention

**Reproduction.** The port keeps the original's split (§5.5): the CSR schedule matches the Python builder exactly — `[0,1,3,6,10]` / `[0,0,1,0,1,2,0,1,2,3]` for 4 blocks — and the kernel matches dense masked attention to **1e-12** over 4 schedule configurations × 4 block geometries.

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

136 is $16 \cdot 17/2$, the dense causal count, and §3.20 bounds any schedule with this configuration to at most 70 blocks. The upstream PR measured 56.6% at seq 1024 and 80.9% at seq 4096 on an RTX 4060, with 1.04×–3.48× sparse-vs-dense-CSR wall clock. **Those timings are not reproduced here and are not claimed here.** This port is a scalar CPU kernel with no SIMD, no threading and no GPU.

**Whether the selection is any good — measured, and it is not.** Everything above is a statement about *cost*. A 58.8% block reduction says the schedule visits fewer blocks; it says nothing about whether the blocks it visits are the right ones. A schedule choosing blocks uniformly at random is also 58.8% cheaper, and a model trained on either still converges. Cost is evidence about sparsity, not about topology. The question that separates them is how much of the true attention mass each schedule keeps at an identical per-row budget, bracketed by random selection below and the exact oracle of §3.20 above. Recovered mass, budget matched per row, `cargo run -p aether-gpu --example selector_ablation --release`:

| seq | density | random | topological | oracle | placement |
|---:|---:|---:|---:|---:|---:|
| 64 | 72.2% | 0.8151 | 0.8475 | 0.9483 | 24.3% |
| 128 | 36.8% | 0.4976 | 0.5674 | 0.6799 | 38.3% |
| 256 | 20.8% | 0.3208 | 0.3711 | 0.4728 | 33.1% |
| 512 | 12.0% | 0.2281 | 0.2337 | 0.3272 | **5.6%** |

Placement (§3.21) is the share of the achievable gain the selector captures, and it collapses as the sequence lengthens — the regime sparse attention exists for. Holding seq at 512 and raising only `topk_topology_blocks` drives it **negative**, to −109% at top-k 32, where the selector recovers 0.7446 against random's 0.8643. Spending more budget on topology makes the schedule worse, and the same holds on i.i.d. keys, the control for a fixture that merely rewards locality.

**The signal is real and its sign is reversed.** `block_salience` scores a block by an $H_0$ merge height (§3.20), which measures how *isolated* it is. Attention mass concentrates where a key resembles the query, and a block unlike everything else is unlike the typical query too, so the two rankings are anti-correlated by construction. Selecting the *lowest*-salience blocks at an identical budget beats random by a margin that grows with budget — 26.9% at top-k 2, rising to 44.7% at 16 — which is what an informative signal looks like. `topology_block_schedule` is left as it is: flipping the ranking changes what the method is, and that belongs to whoever is making the claim rather than to the ablation that found it; `inverted_topology_block_schedule` sits beside it so the choice is explicit.

**Training does not rescue it.** With a query projection learned *through* the attention kernel — the concern being that a model could reshape its queries to suit whatever schedule it is given — dense reaches 86% on an associative-recall task while every sparse arm plateaus between 56% and 61%, and the three sparse arms stay indistinguishable. A model learns a great deal from blocks it can see and cannot learn its way to blocks it never sees. Full tables, controls and the failures behind them are in [`crates/aether-gpu/FEATURES.md`](crates/aether-gpu/FEATURES.md).

---

## 7. Verification

### 7.1 Test inventory

A count is not evidence, so the suites that carry the correctness argument are listed with what they assert. `aether-core` runs 330 tests (56 unit, 274 integration), `aether-lang` 64, and `aether-gpu` 107 integration tests of which 80 are hardware-gated. The per-suite figures below are bound to the files by `readme_claims.rs`: the test count exactly, the line count to 5%.

| Suite | Tests | Lines | Asserts |
|---|---:|---:|---|
| `persistence_invariants.rs` | 12 | 740 | algebraic and metric invariants of the engine |
| `diagram_distance.rs` | 17 | 422 | metrics and vectorisations |
| `attention_contracts.rs` | 29 | 1,255 | selector contracts, causality, cost, routing |
| `scheduled_attention.rs` | 16 | 643 | CSR schedule, salience, kernel parity |
| `persistence_scale.rs` | 7 | 231 | scale, closed form, caps |
| `ablation_baselines.rs` | 9 | 436 | random and oracle schedules |
| `attention_backward.rs` | 5 | 332 | finite-difference gradients |
| `activation_contracts.rs` | 7 | 382 | derivatives at kinks, softmax-layer gradient |
| `ci_gate.rs` | 2 | 141 | the two clippy steps lint the same packages |
| `readme_claims.rs` | 2 | 304 | this document's suite counts; no CPU crate depends on the GPU backend |
| `doc_ratchet.rs` | 1 | 92 | the count of modules enforcing `missing_docs` never falls |
| `orphaned_sources.rs` | 1 | 121 | every source file is reachable from a module tree |
| `linking.rs` | 18 | 627 | Hopf, torus and Whitehead ground truth; rounding rule; refusals ([`linking` docs](https://teerthsharma.github.io/Aether-Lang/integrated/linking/)) |
| `certify.rs` | 17 | 761 | enclosures and certificates against exact integer arithmetic ([`certify` docs](https://teerthsharma.github.io/Aether-Lang/integrated/certify/)) |
| `arrangement.rs` | 18 | 655 | closed-form counts, snap windows, invariances, refusals ([`arrangement` docs](https://teerthsharma.github.io/Aether-Lang/integrated/arrangement/)) |
| `resolvent.rs` | 17 | 856 | the three corners and the mirrored Lean identities ([`resolvent` docs](https://teerthsharma.github.io/Aether-Lang/integrated/resolvent/)) |
| `orbit.rs` | 15 | 805 | the five bounds on seeded and exhaustive instances ([`orbit` docs](https://teerthsharma.github.io/Aether-Lang/integrated/orbit/)) |
| `monodromy.rs` | 18 | 766 | closed-form injectivity, symmetry, dimension and spectrum cases ([`monodromy` docs](https://teerthsharma.github.io/Aether-Lang/integrated/monodromy/)) |
| `track.rs` | 15 | 762 | lineage-forest invariants and the division certificate ([`track` docs](https://teerthsharma.github.io/Aether-Lang/integrated/track/)) |
| `coupling.rs` | 18 | 1,292 | lifted product, contraction and error bounds, islands ([`coupling` docs](https://teerthsharma.github.io/Aether-Lang/integrated/coupling/)) |
| `kvwitness.rs` | 15 | 511 | witness invariants, budget contract, coverage ([`kvwitness` docs](https://teerthsharma.github.io/Aether-Lang/integrated/kvwitness/)) |
| `planner.rs` | 15 | 487 | byte-disjointness, reduction minimality, canonical islands ([`planner` docs](https://teerthsharma.github.io/Aether-Lang/integrated/planner/)) |

The last ten rows are the suites of the certified library; the mathematics each pins is in the [documentation](https://teerthsharma.github.io/Aether-Lang/integrated/), and each is also listed below in the form `readme_claims.rs` binds.

#### `persistence_invariants.rs` — 12 tests, 740 lines

The suite that makes the engine believable. Every entry is a **property over generated inputs**, not one hand-picked cloud with one expected number.

| Test | What it pins |
|---|---|
| `diagram_is_invariant_under_input_permutation` | row order cannot change the diagram; 5 seeds, 1e-9 |
| `diagram_is_invariant_under_rotation_and_translation` | isometry invariance — rotate 0.9128 rad, translate; bottleneck < 1e-9 |
| `diagram_scales_linearly_with_the_point_cloud` | scale equivariance for $c \in \lbrace 0.125, 0.5, 2, 37\rbrace $; exact |
| `a_scaled_radius_cap_selects_the_same_complex` | the cap must scale with the data, catching absolute-threshold bugs |
| `an_edge_just_beyond_the_radius_cap_is_excluded` | one distance on the cap, one $5\times10^{-4}$ above it; the comparison carries no slack |
| `bottleneck_distance_respects_the_stability_bound` | **CSEH stability**, $d_B \le 2\varepsilon$, 12 seed × $\varepsilon$ combinations (§3.10) |
| `circle_has_exactly_one_long_h1_bar_dying_at_sqrt3_times_the_radius` | the canonical positive control |
| `separated_clusters_produce_one_h0_bar_each_until_the_gap_closes` | $\beta_0$ tracks the actual component count |
| `gaussian_noise_produces_no_long_h1_bar` | **the negative control**: 4 seeds, zero long $H_1$ bars |
| `h0_matches_an_independent_union_find` | cross-validation against separate code; 1e-9 |
| `a_single_point_has_one_essential_component_and_nothing_else` | degenerate input |
| `duplicate_points_do_not_break_the_reduction` | zero-distance pairs, the classic reduction crasher |

`gaussian_noise_produces_no_long_h1_bar` is worth as much as every positive case combined. `h0_matches_an_independent_union_find` is why the engine deliberately does not special-case $H_0$ (§5.14). In the module itself, `boundary_of_boundary_is_zero_over_z2` and `every_face_is_present_and_precedes_its_coface` run across five complexes.

#### `diagram_distance.rs` — 17 tests, 422 lines

| Group | Tests |
|---|---|
| Bottleneck as a metric | `bottleneck_of_a_diagram_with_itself_is_zero`, `bottleneck_is_symmetric`, `bottleneck_satisfies_the_triangle_inequality` |
| Bottleneck correctness | `bottleneck_matches_a_hand_computed_pairing`, `bottleneck_projects_an_unmatched_bar_to_the_diagonal` |
| Wasserstein | `wasserstein_sums_where_bottleneck_takes_a_maximum`, `wasserstein_is_at_least_bottleneck` |
| Cross-check | `distances_respect_the_stability_theorem_on_real_diagrams` |
| Landscapes | `a_single_bar_gives_the_expected_tent_function`, `landscape_levels_are_ordered_pointwise`, `landscape_takes_the_kth_largest_tent_where_bars_cross`, `landscape_is_one_lipschitz_in_the_bottleneck_distance`, `an_empty_diagram_gives_a_zero_landscape` |
| Images | `persistence_image_has_the_requested_shape_and_is_nonnegative`, `persistence_image_weights_long_bars_more_than_short_ones`, `sigma_controls_the_kernel_width`, `persistence_image_is_translation_equivariant_in_birth` |

`wasserstein_sums_where_bottleneck_takes_a_maximum` and `landscape_takes_the_kth_largest_tent_where_bars_cross` exist because a mutant survived (§7.3). `sigma_controls_the_kernel_width` exists because **no test referenced $\sigma$ at all**, so every image property held for any fixed kernel width.

#### `attention_contracts.rs` — 29 tests, 1,255 lines

The largest file in the repository, and its shape tells the story: nine tests establish contracts every selector must satisfy, and the remaining twenty record a claim being repaired three times.

**Contracts, for every selector:** `a_full_mask_reproduces_dense_attention_exactly` · `attention_output_is_a_convex_combination_of_values` · `a_uniform_query_averages_the_values` · `a_masked_key_contributes_exactly_nothing` · `the_realized_pattern_equals_the_requested_pattern` · `a_budgeted_selector_respects_its_budget` · `a_single_position_attends_to_itself` · `shapes_around_the_block_boundary_are_handled` · `the_topological_selector_is_scale_equivariant`

**Causality — the class of bug that silently destroys a language model:** `no_output_position_depends_on_a_later_position` (behavioural: perturb a future position, the output must not move) · `causal_selectors_never_select_a_future_key` (structural: inspect the mask). A kernel can pass one and fail the other.

**Numerical safety:** `an_all_masked_row_returns_zeros_rather_than_nan` · `large_logits_do_not_overflow_the_softmax` · `repeated_runs_are_bitwise_identical`. The all-masked row is the sharp edge: softmax over an empty set is 0/0, and a decision that is not tested gets refactored into a NaN that surfaces four layers downstream.

**Measurement instruments:** `oracle_top_k_upper_bounds_every_other_selector_at_the_same_budget` · `the_oracle_is_priced_as_the_diagnostic_it_is` · `the_topological_selector_is_placed_on_the_random_to_oracle_axis`

**Negative results, pinned so they cannot drift back:** `the_topological_advantage_collapses_when_key_norms_vary` · `routing_is_sparse_only_when_the_keys_have_h0_structure` · `routing_buys_its_quality_at_a_real_discount_on_structured_keys` · `the_routed_selector_survives_the_key_norm_spread_that_broke_the_old_one`

**The routing decision:** `the_plan_predicts_the_cost_it_will_actually_incur` · `the_plan_declines_to_route_exactly_when_routing_would_not_pay` · `the_h0_barcode_alone_separates_the_two_regimes` · `the_adaptive_selector_never_costs_more_than_dense` · `the_adaptive_selector_keeps_the_quality_of_whichever_path_it_picks`

**Cross-validation against the persistence engine:** `single_linkage_merge_heights_equal_the_h0_persistence_deaths` · `routing_clusters_are_invariant_to_per_key_rescaling` · `the_routed_selector_obeys_every_contract_the_others_do` — the last guarding against a selector added to an enum, tested only on its new behaviour, and never re-run against the contracts the others satisfy.

#### `scheduled_attention.rs` — 16 tests, 643 lines

| Group | Tests |
|---|---|
| Schedule structure | `the_dense_causal_schedule_is_lower_triangular_csr`, `a_csr_schedule_is_well_formed_at_every_size`, `the_topology_schedule_contains_sink_local_and_salient_blocks`, `the_topology_schedule_visits_fewer_blocks_than_the_dense_one` |
| Salience | `block_salience_is_the_elder_rule_over_centroids`, `the_salience_multiset_is_invariant_to_block_order`, `the_schedule_depends_on_block_order` |
| Numeric parity | `a_dense_schedule_reproduces_full_causal_attention`, `a_sparse_schedule_matches_its_own_dense_masked_reference`, `scheduling_a_block_actually_changes_what_the_row_sees` |
| Safety | `the_kernel_is_deterministic`, `large_logits_stay_finite_across_scheduled_blocks`, `an_empty_row_is_rejected_rather_than_producing_nan` |
| Working set | `the_working_set_does_not_grow_with_the_sequence` |
| Input validation | `the_builder_rejects_malformed_inputs`, `the_kernel_rejects_a_schedule_that_does_not_match_the_sequence` |

`scheduling_a_block_actually_changes_what_the_row_sees` is the anti-tautology test: without it, a kernel that ignored the schedule and computed dense attention would pass every parity check in the file. `block_salience_is_the_elder_rule_over_centroids` asserts that each non-zero salience is an $H_0$ death — membership, which holds. `the_salience_multiset_is_invariant_to_block_order` asserts multiset equality under block reversal; it holds on the suite's fixture and fails in general (§3.20, §8.9).

#### `persistence_scale.rs` — 7 tests, 231 lines, `--release`

`h0_handles_five_hundred_points` · `h1_handles_one_hundred_points` · `circle_h1_dies_at_the_exact_regular_polygon_chord` · `circle_h1_death_converges_to_sqrt3_from_above` · `dense_sampling_does_not_manufacture_extra_loops` · `simplex_cap_still_fails_fast_rather_than_exhausting_memory` · `point_cap_is_still_enforced_when_configured`. This is the file that went from 29.07 s to 1.10 s across the `BTreeMap` refactor. `dense_sampling_does_not_manufacture_extra_loops` is a second negative control: sampling a circle more densely must not invent $H_1$ classes.

#### `ablation_baselines.rs` — 9 tests, 436 lines

Pins the brackets the selection-quality measurement of §6.5 depends on: `the_dense_schedule_recovers_all_of_the_mass`, `no_schedule_recovers_more_mass_than_the_oracle` (the oracle theorem of §3.20, the most valuable assertion in the file), `the_inverted_selector_ignores_the_block_with_no_finite_death`, `the_reported_budget_is_what_the_schedule_spends`, `the_baselines_spend_exactly_the_budget_they_are_given`, `the_random_baseline_is_reproducible_and_not_constant`, `baselines_produce_schedules_the_kernel_accepts`, `recovered_mass_is_a_fraction`, `a_larger_budget_never_recovers_less`.

#### `attention_backward.rs` — 5 tests, 332 lines

`gradients_match_finite_differences_on_a_dense_schedule` and `gradients_match_finite_differences_on_a_sparse_schedule` compare every analytic gradient of §3.19 with a central difference of the forward kernel; `keys_outside_the_schedule_receive_no_gradient`; `the_value_gradient_reproduces_the_attention_weights`; `a_mismatched_cotangent_is_rejected`. A wrong backward pass is the least visible defect in the codebase — the forward stays correct, the loss still falls, and the model converges to a plausible worse optimum — which is why finite differences, sharing no assumption with the code under test, are the load-bearing check.

#### `activation_contracts.rs` — 7 tests, 382 lines

`relu_takes_the_lower_branch_at_exactly_zero`, `leaky_relu_leaks_at_exactly_zero`, `derivatives_match_central_differences_away_from_kinks`, `softmax_returns_a_zero_derivative_that_nothing_consumes`, `a_softmax_output_layer_learns`, `the_softmax_layer_gradient_matches_finite_differences`, `one_optimiser_step_scales_with_the_learning_rate`. Written because a doc comment asserted ReLU's derivative is zero at exactly zero, matching the GPU kernel, and nothing in `aether-core` checked it: a derivative at one point is measure-zero for random inputs and decisive for the inputs that arrive in practice, since a dead ReLU sits exactly there.

#### `ci_gate.rs` — 2 tests, 141 lines

`both_clippy_steps_lint_the_same_packages` and `the_advisory_clippy_step_cannot_fail_the_build`. The gate and the advisory step exist as a pair because `-D warnings` and a printed style lint cannot come from one invocation, and a crate added to one and not the other would be silently unlinted for style.

#### `readme_claims.rs` — 2 tests, 304 lines

`the_readme_suite_counts_match_the_suites` binds every `` `name.rs` — N tests, M lines `` heading and every file-tree row in this document to the suite it names, and requires each documented suite to appear in both forms. `no_cpu_crate_depends_on_the_gpu_backend` reads every workspace manifest's dependency sections and fails if a CPU crate gains an `aether-gpu` edge, because this document states that none has one. It was written after running the documented commands found every line count stale and one test count wrong.

#### `linking.rs` — 18 tests, 627 lines

Ground truth for [`linking` docs](https://teerthsharma.github.io/Aether-Lang/integrated/linking/): the Hopf link and the $(2, 2k)$ torus family certify their linking numbers with one sign, the Whitehead link has $\mathrm{Lk} = 0$ and is never certified separable, the sign of $\omega_{ij}$ agrees with an independent midpoint quadrature, the bound covers the measured distance to the integer, and rounding happens only when the bound proves it. The linking number is invariant under rotation, translation, cyclic shift of the start vertex and swapping the curves, bitwise invariant under power-of-two scaling, and negated by reversing an orientation or reflecting. Writhe is odd under reflection and refuses self-intersection; knot determinants match the classical table and are projection-invariant, mirror-blind and incomplete; every refusal path is taken.

#### `certify.rs` — 17 tests, 761 lines

Soundness of [`certify` docs](https://teerthsharma.github.io/Aether-Lang/integrated/certify/) against exact `i128` arithmetic: Gram and direct enclosures contain the exact score, the cheap radius dominates the tight one, no certified top-$k$ set contradicts exact arithmetic on adversarial near-ties, the threshold trit never places a score on the wrong side, the certified set survives the worst corner of its box, and the two soundness findings of §8.10 are pinned.

#### `arrangement.rs` — 18 tests, 655 lines

[`arrangement` docs](https://teerthsharma.github.io/Aether-Lang/integrated/arrangement/)'s integers on figures with closed-form counts, the snap-window boundaries, invariance under rigid motion, scaling, permutation and segment reversal, and each refusal path.

#### `resolvent.rs` — 17 tests, 856 lines

The three corners of [`resolvent` docs](https://teerthsharma.github.io/Aether-Lang/integrated/resolvent/) — softmax against the reference to 1e-14, the kernel and path-product corners bitwise — each Lean statement mirrored numerically, causality, and finiteness at logits beyond $10^4$.

#### `orbit.rs` — 15 tests, 805 lines

Each of the five bounds of [`orbit` docs](https://teerthsharma.github.io/Aether-Lang/integrated/orbit/) on seeded random and exhaustive small instances, attainment where the proof says it is attained, and `None` on impossible counts.

#### `monodromy.rs` — 18 tests, 766 lines

Every assertion scored against a closed-form answer rather than a recorded output ([`monodromy` docs](https://teerthsharma.github.io/Aether-Lang/integrated/monodromy/)): folds and the wrapped exponential collide, injective controls are cleared, the critical point of $x^3$ reads as a collision with the witness saying which, polygons recover $C_n$ and their mirrors, integer dimensions are recovered within stated tolerances, and the logistic, period-2 and Hénon spectra match their exact or published values.

#### `track.rs` — 15 tests, 762 lines

[`track` docs](https://teerthsharma.github.io/Aether-Lang/integrated/track/)'s guarantees: constant-velocity particles tracked at their closed-form cost, exactly one split on the correct parent, a split taken only when it beats a new lineage, a dropout ending the track without renumbering, the gate priced inside the assignment rather than applied after it, the gate applied in micrometres rather than voxels, the forest invariants on seeded scenes, a barely calibrated division price still certifying, refusal exactly at the calibration bound, and invariance under rigid motion and detection permutation.

#### `coupling.rs` — 18 tests, 1,292 lines

[`coupling` docs](https://teerthsharma.github.io/Aether-Lang/integrated/coupling/)'s operator bit-exact against the naive lifted product, the ridge fit, the contraction and rollout bounds on seeded states, the spectral ceiling, the fixed point, and the island partition against flood fill and the $H_0$ barcode.

#### `kvwitness.rs` — 15 tests, 511 lines

[`kvwitness` docs](https://teerthsharma.github.io/Aether-Lang/integrated/kvwitness/)'s witness invariants — disjointness, distinctness, coverage of $K \cup W(K)$ — the budget and fallback contracts, the lost-coverage case of §9, and the random and locality nulls.

#### `planner.rs` — 15 tests, 487 lines

[`planner` docs](https://teerthsharma.github.io/Aether-Lang/integrated/planner/)'s offset planner never lets live tensors share a byte and never beats the peak-live bound, reuses the leading gap, and prints its ratio to that bound over 500 instances; the transitive reduction preserves reachability, is minimal, removes length-three bypasses in every edge order, and rejects an edge against the topological numbering; island labels are canonical.

The two remaining suites have one test each: `doc_ratchet.rs` fails if the number of modules enforcing `missing_docs` falls, so finished documentation cannot regress; `orphaned_sources.rs` fails if any source file is unreachable from a module tree, since the compiler cannot warn about a file it is never given.

### 7.2 What the suite does not cover

- **No external parity.** Nothing compares against ripser, GUDHI, giotto-tda or Dionysus. Every test above is internal consistency, which a self-consistently wrong implementation can satisfy.
- **No finite-difference check for `ml/autograd.rs`.** The scheduled-attention backward pass and the softmax dense layer have one; the general reverse-mode engine does not.
- **Thin `ml/` coverage.** Beyond the activation contracts, the clustering, classification and regression modules have far less coverage than the topology core. K-means is not tested for initialisation sensitivity, and the two single-linkage implementations are not checked against each other.
- **The certified library is outside the mutation harnesses.** The 166 tests of its ten suites run in CI, but no injected defect has measured what they would catch.
- **Untested formulas.** `persistent_entropy`, `total_persistence`, `landscape_norm` and the Chebyshev guard have no dedicated tests; the governor's stability condition is derived, not asserted; the salience-multiset test uses a fixture that cannot expose the overwrite of §3.20.
- **Seven kernel tests never execute** — four in `scheduler.rs`, two in `loader.rs`, one in `interrupts.rs`.
- **The Lean tree is not built.** No `lake build` in CI.
- **No fuzzing, no `proptest`, no Miri.** The property tests are hand-rolled loops over seeds, not shrinking generators.

### 7.3 Mutation testing

A passing suite says nothing about what it would catch. So known defects are injected, one at a time, and the catches are counted.

**Whole-tree result, both harnesses:** 52 defects injected, **none escape**.

| Harness | Defects | Escaping | Suites run separately | Command |
|---|---:|---:|---|---|
| `aether-core` | 26 | **0** | `ablation_baselines`, `scheduled_attention`, `attention_backward`, `persistence_invariants`, `diagram_distance` | `./crates/aether-core/mutants.sh` |
| `aether-gpu` | 26 | **0** | `gpu_parity`, `gradcheck`, `attention_parity` | `./crates/aether-gpu/mutants.sh` |

The GPU harness needs an adapter and refuses to run without one, because every hardware test is `#[ignore]`d by default and a skipped test is a passing test — a GPU-less run would report every mutant surviving and call that a coverage result. It was measured on an RTX 4060 over Vulkan, and its result is recorded in `FEATURES.md`. The core harness needs no hardware.

Both harnesses run their suites separately rather than together, and the reason is measured rather than stylistic: when the GPU harness held 24 mutants, 19 of them were caught by exactly one of its three suites, so a combined pass/fail would hide which suite does the work, and dropping any one suite would let those defects through.

Neither harness yet injects defects into the ten modules of the certified library (§4.9).

The records below are in discovery order and give the counts as they stood when each was written, so their denominators are smaller than the table above.

**Persistence invariants** — 3 defects:

| Injected defect | New suite | Prior suite |
|---|---|---|
| Triangle filtration drops one of three edges | **4 / 11** | 0 / 6 |
| Hardcoded `+0.001` absolute epsilon in the filtration | **4 / 11** | 0 / 6 |
| Column reduction terminates after one operation | **7 / 11** | 1 / 6 |

The prior six example tests missed two of the three defects **entirely** — the difference between "one hand-picked cloud with one expected number" and "a property that holds for every input".

That table was a record, not a command: no harness in the tree injected those defects, so its counts could not be checked, and in a document whose first rule is that every number is measured, a number with no command behind it is the one to distrust. `bash crates/aether-core/mutants.sh` now injects all three, alongside the others, and reports how many tests in each suite fail rather than whether one does. Re-measured, with the suite since grown to twelve:

| Injected defect | Claimed | Measured now |
|---|---|---|
| Triangle filtration drops one of three edges | 4 / 11 | **4 / 12** |
| Column reduction terminates after one operation | 7 / 11 | **7 / 12** |
| Hardcoded `+0.001` absolute epsilon in the filtration | 4 / 11 | **1 / 12** |

The third had stopped being caught entirely: before the test named below it survived **every suite in the workspace**. The cause was not a weakened assertion. Ten of the twelve tests build their configuration from a shared helper that sets `max_radius: f64::INFINITY`, and `INFINITY + 0.001 == INFINITY`, so for those ten the mutant is not undetected — it is *not a mutation at all*. Only two tests set a finite cap, and the older one exercises it at 1.2 and 1.2 × 6, where no pairwise distance on a twelve-point circle falls in the thousandth of a unit above either. A defect four tests once caught became unreachable for ten of them without any assertion changing; the default configuration drifted underneath them. `an_edge_just_beyond_the_radius_cap_is_excluded` places one distance exactly on the cap and one $5\times10^{-4}$ above it, so the boundary is the only thing the answer depends on, with a control that raises the cap to confirm the merge does happen. With it, **0 of 22 mutants escape** in the harness as it then stood. The lesson, pointed at this section itself: *test fixtures chosen for convenience rather than for discrimination produce suites that pass and prove nothing.*

**Diagram metrics** — 5 defects:

| Injected defect | Caught by |
|---|---|
| Landscape skips the per-sample descending sort | 2 / 17 |
| Image drops the linear persistence weight | 1 / 17 |
| Wasserstein returns the max instead of the sum | 1 / 17 |
| Image hardcodes the Gaussian width, ignoring σ | 1 / 17 |
| Bottleneck forbids diagonal projection | not run — infinite costs diverge the matching search |

All four runnable defects are in the harness and every count reproduces exactly: 2/17, 1/17, 1/17, 1/17. The fifth stays out rather than being quietly dropped; a mutant that hangs the run would trade one unverified claim for an unusable one. Across the whole section that is **six of seven reproducible claims landing on their original figures**, and one that had rotted — the filtration epsilon, now 1/12 against a claimed 4/11.

Three of the four are caught by a single test of seventeen, the state the epsilon reached before it reached zero, so the harness also reports *which* tests catch each defect: three defects at 1/17 could be three independent tests or one test carrying all three, and only the second is alarming. They are independent:

| Defect | Caught by |
|---|---|
| Wasserstein returns the max | `wasserstein_sums_where_bottleneck_takes_a_maximum` |
| Image ignores σ | `sigma_controls_the_kernel_width` |
| Image drops the persistence weight | `persistence_image_weights_long_bars_more_than_short_ones` |

Those are the tests written *because* a mutant survived, which is now a measurement rather than a memory. Across all 26 core mutants no single test catches more than two, so nothing in the suite is load-bearing in the way the epsilon turned out to be. Nothing was added to pad the fractions — a second test written to move a number tests the fixture, not the code. The harness is the guard: the epsilon slipped because no command re-ran it, and now one does.

**Two of the five survived the first version of the suite.** The level-ordering test used *nested* bars, whose tent values already arrive in descending order, so an implementation skipping the sort passed. And no test referenced σ, so every image property — shape, non-negativity, weighting, translation equivariance — held for any fixed kernel width. Both were rewritten, with crossing bars and a concentration measurement, until the mutants died.

> Test fixtures chosen for convenience rather than for discrimination produce suites that pass and prove nothing. A suite that has not been mutated is a suite of unknown strength.

### 7.4 The Lean formalization

`Aether/`, 11,637 lines of Lean 4, toolchain `leanprover/lean4:4.31.0`. **Not built by CI**, which is the first and most important thing to say about it.

| File | Lines | Theorems | `example` blocks | `sorry` | What it is |
|---|---:|---:|---:|---:|---|
| `Core.lean` | 3,356 | **47** | 174 | 0 | syntax, values, big-step relations, executable evaluator |
| `VM.lean` | 2,526 | 1 | 144 | 0 | a VM and checked frame compiler |
| `Static.lean` | 1,891 | 0 | 120 | 0 | a static checker |
| `Parser.lean` | 1,569 | 0 | 84 | 0 | a parser |
| `Pipeline.lean` | 1,442 | 0 | 152 | 0 | a source pipeline with diagnostics |
| `Lexer.lean` | 853 | 0 | 33 | 0 | a lexer |
| **Total** | **11,637** | **48** | **707** | **0** | |

**What the theorems state.** `Core.lean` defines inductive big-step relations — `StepStmt`, `StepBlock`, `EvalExprWithFnsRel`, `StepStmtWithFns`, `StepBlockWithFns` — alongside fuel-bounded executable functions `evalExprWithFns` and `execStmtWithFns`. Two of its 47 theorems are universally quantified lemmas about environments: `lookup_bind_same` ($\mathrm{lookup}(\mathrm{bind}(\rho, x, v), x) = v$) and `eval_bound_var`. The other 45 are **closed instances**: each fixes one concrete program — a literal, a binary addition, an `if`, a `for`, nine seal-loop programs — and states that if the relation holds for it, the executable evaluator computes the expected result, discharged by `native_decide`. They are therefore checks that the executor agrees with the semantics on specific programs, not soundness theorems over all programs, and `native_decide` extends the trusted base to Lean's compiler. The one theorem in `VM.lean`, `compileCheckedFrameProgram_static_ok`, is general: a successful checked compilation implies the static checker accepted the program.

**The shape of the table is the summary.** 47 of 48 theorems live in one file. The 8,281 lines of the other five files hold one theorem between them and 533 `example` blocks: they are a **second implementation** of the language — lexer, parser, static checker, pipeline, VM — written in Lean, exercised by examples, not proofs about the Rust implementation. Zero `sorry` is real and worth having.

**What it does not do.** It does not verify the Rust: there is no extraction, refinement proof or correspondence argument connecting `Core.lean` to `crates/aether-core` or `crates/aether-lang`. And it is not checked to still compile; given that this repository's CI had never executed at all until recently, an ungated tree should be assumed stale until a build says otherwise. The formal core is described further in [`docs/FORMAL_CORE.md`](docs/FORMAL_CORE.md).

**The disposition.** Gate it or cut it. Gating means adding `lake build` to `ci.yml` and accepting the toolchain-pinning cost. Cutting means acknowledging that 8,281 lines of largely theorem-free re-implementation are a liability. The present state — present, impressive in a line count, unverified — is the one state worse than either decision, and it is listed as Ungated rather than counted as a feature.

### 7.5 Continuous integration

Three workflows. [`ci.yml`](.github/workflows/ci.yml) is the gate, and its header records why it exists in its current form: it *triggers on the branch this repository actually uses; the previous config listed `main`/`develop`, neither of which exists here, so no run had ever executed and the workspace had drifted out of compiling.*

| Job | Runner | What it proves |
|---|---|---|
| `test` | ubuntu · windows · macos, `fail-fast: false` | fmt (ubuntu only), clippy gate, advisory clippy, `cargo test --workspace --exclude aether-kernel`, and a compile-only build of the GPU tests with `--features gpu` |
| `invariants` | ubuntu | the topology suites, **named separately so a failure names itself**: `persistence_invariants`, `persistence_scale --release`, `diagram_distance`, `attention_contracts`, `scheduled_attention`, and `--lib persistence` |
| `no_std_check` | ubuntu | builds `aether-core`, and `aegis-core` without `std`, for `thumbv7m-none-eabi` |
| `kernel` | ubuntu | builds `aether-kernel` for `x86_64-unknown-none` |
| `docker` | ubuntu | builds the image and runs `docker run --rm aether:test --help` |
| `release` | ubuntu | tag-gated, `needs:` all five above |

**`invariants` is a separate job on purpose.** It re-runs suites the `test` job already covers, and the redundancy buys a **named check** in the pull-request list: a failure reads `Persistence Invariants` rather than one line inside a long test log. The workflow comment states the reasoning — *a benchmark measured on a wrong implementation is not a result* — which is rule 6 of §1.4 expressed as CI structure.

**`fail-fast: false` on the OS matrix.** A Windows-only failure and a macOS-only failure are different bugs, and fail-fast would hide the second behind the first.

**The docker job runs the binary.** `docker run --rm aether:test --help` is the difference between an image that builds and an image that works. The Dockerfile had been copying crate directories from paths that never existed.

**The toolchain is unpinned nightly**, so a release that promotes a lint into `correctness` or `suspicious` can turn the gate red with no commit touching the code. The CI comment records this standing exposure.

[`docs.yml`](.github/workflows/docs.yml) builds the MkDocs site with `mkdocs build --strict` and deploys it to GitHub Pages; `--strict` turns broken internal links into build failures, the same argument applied to documentation. [`publish.yml`](.github/workflows/publish.yml) handles releases.

**What CI does not run:** `lake build` for the Lean tree, a QEMU boot, external TDA parity, the seven kernel tests, and the hardware-gated GPU tests.

`aether-kernel` is excluded from the host test job because it is a `no_std` bare-metal binary with no host-target global allocator or panic handler and **cannot link for a host target**; its own job builds it for `x86_64-unknown-none`. Running `cargo test --workspace` without the exclusion fails, and that failure is a property of the target, not a bug.

---

## 8. Negative results: what we got wrong

A first-class section, with the numbers that killed each claim. It is the part of a README a reader should want to see first in someone else's project, and it is where this one keeps the evidence that the evidence policy of §1.4 is applied to the author as well as to the code.

The audit that started it found that the CI workflow triggered on `main` and `develop` while the repository's branch is `master`, so **no CI run had ever executed** in the history of the project. Consequently `aether-kernel` had silently stopped compiling, the `Dockerfile` copied crate directories from paths that had never existed, and `pyproject.toml` was publishing a description no benchmark supported. All of it was fixed in [PR #177](https://github.com/teerthsharma/Aether-Lang/pull/177). The general lesson: a green checkmark that has never been seen is not a green checkmark.

### 8.1 "Current World's Fastest Agentic AI Language"

That string was the `description` field in `pyproject.toml`, **published to PyPI**. No benchmark in this repository supports it, and it sat three files away from a README section titled *Evidence Policy* that forbids unverified speedup claims. Removed.

### 8.2 Euclidean proximity as an attention-mass proxy: negative

The claim: keys geometrically near a query carry most of the attention mass, so they can be selected without computing scores. Measured against random and oracle top-k at the same budget — seq 32, head_dim 8, budget 6, 8 trials per row, placement as in §3.21:

| key-norm spread | random | nearest-neighbour | oracle | placement |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.4902 | 0.5667 | 0.5769 | **+0.884** |
| 1.0 | 0.4899 | 0.5613 | 0.6242 | +0.533 |
| 2.0 | 0.4892 | 0.5265 | 0.6723 | +0.202 |
| 4.0 | 0.4873 | 0.4577 | 0.7616 | **−0.109** |
| 8.0 | 0.4848 | 0.3725 | 0.8841 | **−0.285** |

Past spread 4 the selector is **worse than choosing keys uniformly at random**. The reason is the identity of §3.22, $\lVert q - k\rVert^2 = \lVert q\rVert^2 + \lVert k\rVert^2 - 2\,q\cdot k$: with equal key norms, ranking by distance *is* ranking by dot product, so the +0.884 at spread 0 is close to tautological; vary the norms and the rankings decouple. The +0.884 is **not reported as a result**. `the_topological_advantage_collapses_when_key_norms_vary` pins both ends so it cannot drift back.

### 8.3 The first fix was also wrong: a measurement artifact

The first repair used an **absolute** radius of 0.6 against a median query–key distance of 2.395. It selected **1.00 keys per row** where its same-budget baselines selected **5.53**, and posted placements of **−3.573 to −4.177**. It did not lose on mechanism; it lost on budget, by declining to select. It is the same bug class as the hardcoded epsilon the scale-equivariance test exists to catch — an absolute threshold pretending to be relative. The radius is now a multiple of the row's median distance, which makes the rule scale-equivariant, and the ablation asserts equal mean budget before comparing anything.

### 8.4 Topological routing on unstructured keys: not sparse at all

The repaired selector clusters key *directions* and ranks by exact dot product. Placement held flat at +0.87 across the whole spread curve — apparently a clean win. Then the cost was measured, `cargo run -p aether-core --example routing_cost --release`:

| key distribution | $H_0$ component sizes | cost vs dense | placement |
| --- | --- | ---: | ---: |
| uniform random | `[61, 1, 1, 1]` | **0.999** | +0.942 |
| 4 real clusters | `[16, 16, 16, 16]` | **0.449** | +0.990 |
| 8 real clusters | `[8, 8, 8, 8]` | 0.528 | +0.995 |
| 16 real clusters | `[4, 4, 4, 4]` | 0.733 | +0.989 |

On uniform keys it examines **0.999× the dense dot-product count**: dense attention with clustering overhead, its +0.94 placement bought by looking at every key. Single linkage **chains** on a cloud with no density gaps — 61 of 64 keys in one component. That is not a clustering bug; it is $H_0$ correctly reporting that uniform data has no structure to route on. The persistence diagram was right; the claim was wrong.

**Every test in the suite measured how good a selection was. None measured what it cost.** `selection_dot_cost` and `dense_dot_cost` exist because of this, and the claim is now conditional:

> Topological routing is a real sparsity win exactly when the key distribution has $H_0$ structure, and no win at all when it does not.

`routing_plan` makes that a runtime check. Its gap ratio, derived from the $H_0$ barcode alone, separates the regimes without overlap — **structured minimum 2.70 against chained maximum 1.04**, 6 trials each (§3.22).

### 8.5 The cheap fallback: indistinguishable from random

When routing does not pay, `Selector::Adaptive` must do something else. The first fallback was a budget-6 sliding window, which measured placement **+0.014** on unstructured keys — random, to three decimals. Not a fallback bug: when the keys have no structure there is **no cheap-and-good option**, because finding the top-k without computing the scores is exactly what the structure was supposed to make possible. The fallback is now dense, and the guarantee is *never worse than dense, in cost or in quality*, the only one safe to enable by default.

### 8.6 `nalgebra`: a phantom dependency

Found while rewriting an earlier revision of this document, by reading `Cargo.toml` instead of the sentence beside it. `aether-core` — the crate this document holds up as having a minimal dependency surface — declares

```toml
nalgebra = { version = "0.32", default-features = false, features = ["libm"] }
```

with **zero call sites**: `grep -rn nalgebra crates/ --include=*.rs` returns nothing. The README asserted in four places that the core had "exactly one dependency, `libm`" and that `libm` plus `heapless` was "the whole mathematical surface"; all four were false and were corrected. It is the same defect class as the `wgpu`, `pollster` and `bytemuck` entries that sat in `aether-lang`'s default features with zero call sites, now deleted (§5.8), with an aggravating difference: this one sits in the crate whose minimality is a headline claim.

**The `no_std` claim survives; the minimality claim does not.** `default-features = false` with the `libm` feature is why `thumbv7m-none-eabi` still builds, so the Cortex-M3 result stands; the "one dependency" line was decoration on a real result, and it was the decoration that was false. **The duplicate crate was better configured than the real one**: `aegis-core` declares the same dependency as `optional = true`. **A dependency count is a claim, and claims need commands**: it rotted because it read like a fact about a file rather than a measurement, and it now carries one in [Reproducing every number](#reproducing-every-number). Queued for removal.

### 8.7 A number that lied

With the dense fallback, the unstructured placement printed **+7.614**. Meaningless: dense recovers all the attention mass, which sits *above* the budget-limited oracle, so the ratio of §3.21 blows up. Left in a table it would read as a 700% win over an oracle it never competed with. The two regimes are now scored on different scales deliberately.

### 8.8 Tests that were wrong, twice

- A scale test asserted that the circle $H_1$ death *converges* to $\sqrt3 r$ monotonically. It failed at $n = 24$, because the engine returns $\sqrt3 r$ **exactly** whenever $3 \mid n$. Writing the test found the sharper theorem of §3.15.
- A port test asserted that per-block salience is permutation-equivariant. It failed at block 2 (`0` against `1.724`): under component-size ties the absorbed component depends on index order. The test was wrong, not the code — and the replacement, which asserts multiset invariance, is itself too strong in general (§8.9).

### 8.9 Corrections made while re-deriving the mathematics

This revision re-derived every formula in §3 from the source and ran the documented programs through the CLI. Each statement below was carried by an earlier revision of the README and is contradicted by the code; each is corrected in the section named.

| Earlier statement | What the code does | Where |
|---|---|---|
| `seal until convergence(1e-6)` exits when the Betti numbers of the residual manifold stop changing | parsed, then failed at runtime with `condition must be boolean`: `convergence` was undefined and `1e-6` was not a literal. Fixed in this revision as a scalar max-norm tolerance, not a Betti rule | §4.4, §4.6 |
| `convergence(1e-6)` parses to `ConvergenceCond::Epsilon`; the topological `BettiStable` variant is parsed and implemented | the parser only ever produces `Custom`; `Epsilon` and `BettiStable` are never constructed; `regress` always runs with $\varepsilon = 10^{-6}$ | §4.7 |
| The seal loop exits when the topology has stabilised **and** the scalar tolerance is met | both convergence predicates are disjunctions; the error alone suffices | §3.29 |
| The stopping signal is the Betti numbers of the residual manifold | a sign-change and oscillation count on the residual sequence; the persistence engine is not consulted | §3.29 |
| `regress { ... }` calls `ml::regressor::ManifoldRegressor` | it calls the interpreter's own `EscalatingRegressor` | §4.7 |
| The multiset of block saliences is invariant to block order, being the $H_0$ barcode | salience overwrites earlier deaths when a multi-member component is absorbed; centroids $0, 1, 10, 12$ give $(9, 9, 2, 0)$ against deaths $\lbrace 1, 2, 9\rbrace $, and $(9, 9, 1, 0)$ reversed | §3.20 |
| `routing_plan` uses the gap ratio to decide whether to route | the decision thresholds the predicted cost ratio at 0.6; the gap ratio is reported beside it | §3.22 |
| `MAX_BETTI_1 = 10` is a saturating cap | it is a rejection threshold; the count is unsaturated | §5.7 |
| The byte-level $\beta_0$ is windowed density clustering | it is the number of maximal runs of byte gaps above 15 | §5.7 |
| `SparseAttentionGraph::compute_betti_0` is an exact union-find, $O(n\,\alpha(n))$ | a depth-first search over `u64` bitsets with a 64-entry stack, exact only for graphs of at most 64 points | §3.17, §5.13 |
| `estimate_betti_1` is a cycle-rank heuristic | it is the exact cycle rank of the graph, and an upper bound on $\beta_1$ of the Rips complex | §3.17 |
| `kaiming_uniform` is He initialisation | its variance is $1/n_{\mathrm{in}}$, half of He's; `DenseLayer`'s "Xavier" variance is one third of Glorot's | §3.28 |
| The persistence image integrates the Gaussian over a pixel grid | it evaluates the density at pixel centres; the error bound is derived | §3.14 |
| The Chebyshev guard measures liveness across spatial blocks | across occupied slots, one-sided, compared after a 0.95 decay that shifts the effective $k$ | §3.25 |
| The witness complex costs $O(\ell^{k+2} + n\ell)$ | $O(n\ell^2 + n\ell^{K+2})$; every candidate simplex scans every witness, and the formula is evaluated on each simplex rather than by flag completion | §3.9 |
| The delay embedder returns `None` for the first $(D-1)\tau$ samples | for the first $D\tau - 1$ samples | §3.16 |
| Floats are carried as integer and fractional parts at the AST level | only in the `Number` type used by configuration values and ranges; expression literals are `f64`, fixed-point with six fractional digits | §4.4 |
| The governor update is $\varepsilon(t+1) = \varepsilon(t) + \alpha e(t) + \beta\,de/dt$, asymptotically stable | the sign was reversed against the code of the time, and that law alternated between its clamps; the current law acts on $\ln\varepsilon$ and its stability condition is proved | §3.26 |
| The license is MIT | `LICENSE` is the Aether-Lang Custom Attribution License | [License](#license) |

Stale counts corrected in the same pass: 11 persistence invariants (12), 102, 93 and 60 GPU tests (107), 76 hardware-gated tests (80), 50, 24, 25 and 10 GPU or whole-tree mutants (26 GPU, 52 total), 215 passing tests in the claims table (the badge's figure, re-measured), six workspace crates (seven), "no gradcheck anywhere" (the scheduled backward pass and the softmax layer have finite-difference checks; `ml/autograd.rs` does not), the four-kernel-test count (seven tests never execute), and every source line count in §4.8, §5 and [Repository layout](#repository-layout).

## 9. Limitations

Longer than most projects' feature lists, deliberately. Every limitation below is also stated where the relevant mechanism is described; this section collects them.

**No external parity.** The persistence engine has never been compared against ripser, GUDHI, giotto-tda or Dionysus on shared fixtures. The invariant suite is not parity: a self-consistently wrong implementation can satisfy every internal property it tests. This is the largest correctness debt in the repository.

**The core claim is unmeasured.** Whether stopping on topological stability beats stopping on a tuned scalar residual, on real problems, has not been tested. The language can now express the rule — `seal until stable(topology.betti(...))` watches a Betti vector of a filtration (§4.6) — but only with a one-pass window, and the built-in `regress` statement still stops on a sign pattern of its residuals (§3.29); `BettiStable` is never constructed (§4.7), and the headline `seal until convergence(ε)` spelling fails at runtime (§4.6). A seal loop is topological only when its author writes a topological condition.

**Scale.** $H_0$ at $n = 4{,}000$ takes 335 s and $H_1$ at $n = 300$ takes 131 s; running time grows roughly as $m^{1.5}$ in the simplex count, single-threaded, without the clearing, cohomology or apparent-pair optimisations. Production TDA libraries handle clouds orders of magnitude larger. The language's default `topology.ph` budget refuses clouds of 19 or more points unless the caller raises it (§4.5).

**The witness complex is unbounded as an approximation.** No theorem relating its diagram to the Rips diagram of the full cloud is implemented or tested, and the code's per-simplex evaluation differs from the textbook flag completion (§3.9).

**Block salience does not compute the barcode.** Overwriting deaths on absorption makes the salience multiset depend on block order and differ from the $H_0$ barcode whenever a multi-member component is absorbed, and the test asserting otherwise uses a fixture that cannot expose it (§3.20). Per-block salience is also order-dependent through the tie-break; a fix needs a tie-break on centroid content and a single write per death.

**The block schedule's cost reduction is real and its selection is not.** At an identical per-row budget the topological ranking recovers less attention mass than random selection, the deficit widens with budget, and inverting the ranking beats random instead. Two synthetic fixtures, one machine; the shipped `topology_block_schedule` is unchanged (§6.5).

**Attention results are synthetic.** Every ablation is measured on synthetic keys. Whether real attention key distributions carry $H_0$ structure is unmeasured, and it decides whether the routing result transfers at all.

**Every selector allocates a dense `[seq, seq]` mask.** No selector in `attention` is sub-quadratic in memory, however few keys it picks. `aether_core::attention` has no backward pass; the scheduled port has one, checked by finite differences.

**The Triton port reproduces answers, not timings.** Scalar CPU, no SIMD, no threading. The upstream 1.04×–3.48× figures were measured on an RTX 4060 this workspace's CPU kernel cannot reach.

**The GPU backend is used by nothing outside its own crate.** `aether-gpu` is real — 20 WGSL kernels, resident tensors, 107 tests, finite-difference and mutation checks — but no line of `aether-core` or `aether-lang` calls it, and none can: `aether-core` is `no_std` and `wgpu` is not. `Tensor::matmul` crosses over at $n = 128$ with conversion counted, and its magnitude above that is not measurable here, having been recorded anywhere from 10× to 63× because the GPU term swings 5.2× between runs while the CPU term moves 1.6×; only the crossover is quoted. `pairwise_sqdist` never pays, because the reduction is CPU-side and the matrix must come back (§5.11).

**One phantom dependency remains.** `nalgebra` is a non-optional dependency of `aether-core` with zero call sites (§8.6).

**Streaming-path Betti numbers are small-graph quantities.** `SparseAttentionGraph` records edges only among the first 64 points, aliases reverse edges for later points onto earlier bits, and counts components with a fixed 64-entry stack (§3.17).

**The `ml` subtree has thin coverage.** 5,075 lines of clustering, classification, regression, neural layers, autograd and convolution are covered mainly by in-module tests and the activation contracts. K-means is not tested for initialisation sensitivity, the two single-linkage implementations are not checked against each other, `ml/autograd.rs` has no finite-difference check, `ml/convolution.rs` is forward-only, higher-degree polynomial "fits" are one greedy pass rather than least squares, and `ml/benchmark.rs` is an escalation harness, not a benchmark, as are the four `examples/` files with `benchmark` in their names.

**`topology.rs` shares vocabulary with the persistence engine and none of its guarantees.** It computes run and window counts over bytes; none of the 12 persistence invariants apply; three functions in the crate are named some variant of `compute_betti_0` and only one of them is persistent homology. `verify_binary_topology` is not a security mechanism, and its whole-image check rejects any input with more than ten qualifying windows regardless of length (§5.9).

**The governor's stability is local.** The proposition of §3.26 is a linearisation about $\varepsilon^\star$; large-signal behaviour is bounded by the error clamp but not otherwise analysed, and a $\Delta$ outside $[1, 10^4]$ settles on a clamp. It is PD, not PID.

**Untested formulas.** Persistent entropy, total persistence, the landscape norm and the Chebyshev guard have no dedicated tests.

**The Titan VM has no parity suite against the interpreter.** Two execution engines, no differential test.

**The kernel compiles but is not asserted to boot.** No QEMU logs, no hardware matrix, and seven tests in the crate never execute.

**The Lean tree is ungated and does not verify the Rust.** 11,637 lines, 48 theorems, no `sorry`; 45 of the 47 theorems in `Core.lean` are closed instances discharged by `native_decide`; 8,281 lines hold one theorem between them; no `lake build` in CI; no correspondence to the Rust implementation (§7.4).

**Duplicate crates and extension drift.** `aegis-core` and `aegis-cli` are near-copies left by a rename done by copying; the examples use `.aegis` and `.ag`, and the CLI warns on the latter.

**Example and argument hygiene.** `examples/seal_loop_demo.aegis`, the fullest seal-loop example, does not parse under the current grammar: it uses a `seal for` form the parser does not have, and `aether check` stops at line 33 with `expected {, found for`. The breakage predates the port and occurs identically on the pre-port tree. Named arguments a function does not recognise are ignored rather than rejected, so a misspelt option silently takes its default.

**`monodromy` decides little and estimates with known bias ([`monodromy` docs](https://teerthsharma.github.io/Aether-Lang/integrated/monodromy/)).** Only the injectivity decision is Jacobian-free; the Lyapunov estimator consumes $DF$. An injective map with a critical point is read as a collision — $x \mapsto x^3$ at the origin — and the witness, not the verdict, tells the two situations apart. $\rho_{\mathrm{free}}$ falls with $n$, so its threshold is valid only from 400 points; it was set from eight 2-D maps and checked on three 3-D maps, and the source missed a 3-D non-injective map reading 2.050e-2 at $n = 200$. A collision in a region the sampler never reaches is invisible to any bounded sample. The persistent-homology dimension is biased downward in high dimension (8.660 at $n \le 400$ and 8.531 at $n \le 1600$ on the 10-cube, in the source's measurement), its interval models sampling noise rather than bias, and at small $n$ a segment reads fractal against $d_{\mathrm{top}} = 1$ — a failure the suite pins deliberately. Symmetry recovery is sound, not complete: on jittered regular polygons the source recovered 6/6, 5/6, 4/6, 2/6 and 2/6 at 0, 2, 5, 8 and 10% jitter. Oseledets' hypotheses are not checked, and the Kaplan–Yorke dimension equals the information dimension only under the Kaplan–Yorke conjecture.

**`kvwitness` is correct, bounded and unevaluated ([`kvwitness` docs](https://teerthsharma.github.io/Aether-Lang/integrated/kvwitness/)).** Full segment coverage holds for the learned set plus its witnesses but not always for the merged row: occupancy is read before the tail is overwritten, so a segment whose only learned token sits in an overwritten slot is lost. `overwriting_a_sole_tail_occupant_uncovers_its_segment` pins the case: the row $[0, 1, 2, 3, 12, -1, -1, -1]$ with $L = 16$, $S = 4$ merges to $[0, 1, 2, 3, 4, 8, -1, -1]$, coverage 0.75. The uncovered-run bound is $2\lceil L/S_{\mathrm{eff}}\rceil - 2$, looser than the upstream description's $L/S$. No model-level quality evaluation exists here or upstream; the kernel cost figures are the pull request's, on an RTX 4060 Laptop GPU.

**The certified library carries fixed calibrations and first-order bounds.** Thresholds in `monodromy` and `arrangement` are policy constants, not re-fitted; the `linking` certificate is first order in the unit roundoff, not interval arithmetic; the `coupling` error bound inherits an in-sample residual level as its hypothesis; `track` has no gap closing; and none of the ten suites is in the mutation harnesses of §7.3.

**All timings are single-machine, single-run** — Windows 11, nightly, no confidence intervals, no turbo control, no core pinning (§6.1).

---

## Quick start

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
  Mode: interpreter
═══════════════════════════════════════════════════════════════
[4, 0, 0]

Execution complete. 🦭
```

$\beta_0 = 4$, $\beta_1 = 0$, $\beta_2 = 0$ at radius 0.5. `print` renders lists in brackets and records as `{k: v}`.

**Every number in that output is predicted by §3.** The interpreter embeds with $D = 3$, so 8 samples give $8 - 3 + 1 = 6$ points (§3.16). The signal has period 4, so those six points take four distinct values, $(3,2,1)$, $(2,3,2)$, $(1,2,3)$ and $(2,1,2)$, whose pairwise distances are $\sqrt3$, 2 and $\sqrt8$. Below $\sqrt3$ the four locations are separate components, so $\beta_0 = 4$ at any radius under 1.732. With $R = \infty$ and $K = 1$ the diagram has $\binom62 + 1 = 16$ pairs (§3.6), which `topology.intervals(diagram)` confirms: two $[0,0)$ bars for the duplicated points, three $H_0$ deaths at $\sqrt3$, one essential class, and one $H_1$ bar $[\sqrt3, 2)$ — the 4-cycle through the four distinct points, filled when the diagonal of length 2 enters — plus zero-length $H_1$ pairs. `topology.betti(diagram, radius=1.8)` returns $[1, 1, 0]$.

Other commands:

```bash
cargo run -p aether-cli -- repl              # interactive
cargo run -p aether-cli -- check f.aether    # parse only, no execution
cargo run -p aether-cli -- run examples/hopf_link.aegis   # a certified linking number (§4.9)
```

A seal loop, with a condition that does run:

```aether
let n = 0~
🦭 until n >= 3 {
    n = n + 1~
}
print(n)~
```

### Building and testing

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

# reproduce the tables in this document
cargo run -p aether-core --example scale_probe   --release
cargo run -p aether-core --example routing_cost  --release
```

Aliases in `.cargo/config.toml`: `cargo gate`, `cargo invariants`, `cargo kernel`, `cargo embedded`, `cargo cli`, `cargo lang`, `cargo core`.

---

## Requirements

**Toolchain.** Rust **nightly**, named in `rust-toolchain.toml` without a pinned date. Nightly is required for `-Z build-std` (kernel and embedded targets) and for `#![feature(abi_x86_interrupt)]` in the kernel. The CLI itself uses stable features only. Edition 2021.

**Components.** `rust-src` and `llvm-tools-preview` for bare-metal builds; `clippy` and `rustfmt` for the gate.

**Targets.**

| Target | Purpose | Extra flags |
|---|---|---|
| host (`x86_64-pc-windows-msvc`, Linux, macOS) | CLI, tests | — |
| `x86_64-unknown-none` | kernel | `-Z build-std=core,alloc` |
| `thumbv7m-none-eabi` | `no_std` verification | `-Z build-std=core,alloc` |

**Dependencies of `aether-core`.** Three declared:

```toml
libm     = "0.2"
heapless = "0.8"
nalgebra = { version = "0.32", default-features = false, features = ["libm"] }
```

`libm` and `heapless` are used; `nalgebra` has zero call sites and is compiled by every build of the crate (§8.6). Its `default-features = false` with the `libm` feature is why the `thumbv7m-none-eabi` build still succeeds.

**Feature flags** (`aether-core`):

| Feature | Implies | Meaning |
|---|---|---|
| `std` *(default)* | `alloc` | standard library; CLI and tests |
| `alloc` | — | `no_std` with an allocator |
| `no_std` | `alloc` | bare-metal mode |

The CLI adds `clap` and `rustyline`. `aether-gpu` adds `wgpu` and needs a Vulkan, Metal or DX12 adapter for its hardware tests, which are enabled only with `--features gpu` (§5.11).

**Optional Python.** `pyproject.toml` builds a `pyo3` extension through `maturin`; the bindings package is currently empty.

**Optional Lean.** `lean-toolchain` names `leanprover/lean4:4.31.0`; `lake build` builds `Aether/`, and nothing runs it in CI.

---

## Reproducing every number

The evidence policy in one rule: **a number without a reproduction command does not go in a table.** This section discharges it — every quantitative claim above, mapped to the command that produces it.

**(guarded)** marks a claim a test fails on when it drifts, so the command is a way to *see* the number rather than the only thing keeping it true. The guards are `crates/aether-core/tests/readme_claims.rs` (suite test and line counts; the absence of a GPU dependency edge) and `crates/aether-gpu/tests/features_doc.rs` (the ignored count, the WGSL kernel count, the Rust and Lean line counts). Everything else is a snapshot: correct when it was run, and nothing notices if it stops being.

| Claim | Where | Command |
|---|---|---|
| 428 passed, 80 ignored (guarded: ignored) | [Status](#15-status) | `cargo test --workspace --exclude aether-kernel` |
| Per-suite test and line counts (guarded) | [§7.1](#71-test-inventory) | `cargo test -p aether-core --test readme_claims` |
| 12 persistence invariants | [§7.1](#persistence_invariantsrs--12-tests-740-lines) | `cargo test -p aether-core --test persistence_invariants` |
| 17 diagram-metric tests | [§7.1](#diagram_distancers--17-tests-422-lines) | `cargo test -p aether-core --test diagram_distance` |
| 29 attention contracts | [§7.1](#attention_contractsrs--29-tests-1255-lines) | `cargo test -p aether-core --test attention_contracts -- --nocapture` |
| 16 scheduled-attention tests | [§7.1](#scheduled_attentionrs--16-tests-643-lines) | `cargo test -p aether-core --test scheduled_attention -- --nocapture` |
| 7 scale tests, 1.10 s | [§6.2](#62-the-face-index-refactor) | `cargo test -p aether-core --test persistence_scale --release` |
| Scale ceiling table; pairs equal §3.6 | [§6.3](#63-measured-scale-ceiling) | `cargo run -p aether-core --example scale_probe --release` |
| Routing cost table | [§8.4](#84-topological-routing-on-unstructured-keys-not-sparse-at-all) | `cargo run -p aether-core --example routing_cost --release` |
| Placement / spread table | [§8.2](#82-euclidean-proximity-as-an-attention-mass-proxy-negative) | `cargo test -p aether-core --test attention_contracts -- --nocapture` |
| 58.8% block reduction | [§6.5](#65-scheduled-attention) | `cargo test -p aether-core --test scheduled_attention -- --nocapture` |
| Selection-quality table, −109% at top-k 32, 26.9% / 44.7% inverted | [§6.5](#65-scheduled-attention) | `cargo run -p aether-gpu --example selector_ablation --release` |
| Gap ratio 2.70 vs 1.04 | [§3.22](#322-the-routing-gap-ratio) | `cargo test -p aether-core --test attention_contracts -- --nocapture` |
| Salience counterexample $(9, 9, 2, 0)$ | [§3.20](#320-block-salience-recovered-mass-and-the-oracle) | call `block_salience(&[0.0, 1.0, 10.0, 12.0], 4, 1, 1)`, then the reversed input |
| Newton's $\sqrt2$ seals on pass 5 under `convergence(1e-6)` | [§4.6](#46-seal-loops) | `cargo test -p aether-lang --test seal_convergence`, or `cargo run -p aether-cli -- run` on the program in §4.6 |
| Default `topology.ph` refuses 19 points | [§4.5](#45-the-topology-module) | `cargo run -p aether-cli -- run` on a 21-sample series with `topology.ph(M)` |
| Governor settles at $\Delta/R^\star$ | [§3.26](#326-the-governor-control-law) | `cargo test -p aether-core --lib governor` |
| Kernel compiles bare metal | [Status](#15-status) | `cargo build -p aether-kernel -Z build-std=core,alloc --target x86_64-unknown-none` |
| `no_std` on Cortex-M3 | [Status](#15-status) | `cargo build -p aether-core --no-default-features --features no_std -Z build-std=core,alloc --target thumbv7m-none-eabi` |
| Formatting clean | [Status](#15-status) | `cargo fmt --all -- --check` |
| Clippy clean | [Status](#15-status) | `cargo clippy --workspace --exclude aether-kernel --all-targets -- -D warnings -D clippy::correctness -D clippy::suspicious -A clippy::style -A clippy::complexity -A clippy::perf` |
| 53,228 Rust lines, 105 files (guarded: lines) | [§1.3](#13-scope-of-the-claims) | `(Get-ChildItem crates -Recurse -Filter *.rs \| Get-Content).Count`, or `find crates -name '*.rs' \| xargs cat \| wc -l` |
| `nalgebra` has zero call sites | [§8.6](#86-nalgebra-a-phantom-dependency) | `grep -rn nalgebra crates/ --include=*.rs` |
| 107 GPU tests, 80 hardware-gated | [§5.11](#511-aether-gpu) | `cargo test -p aether-gpu --features gpu --release` |
| 20 WGSL kernels (guarded) | [§5.11](#511-aether-gpu) | `grep -c '^@compute' crates/aether-gpu/src/shaders.wgsl` |
| 0 of 26 GPU mutants escape, 0 of 26 in core | [§7.3](#73-mutation-testing) | `./crates/aether-gpu/mutants.sh`, `./crates/aether-core/mutants.sh` |
| matmul crossover $n = 128$ (magnitude not reproducible; 10×–63× observed) | [§5.11](#511-aether-gpu) | `cargo run -p aether-gpu --example tensor_crossover --release`, and `-- --samples` for the raw timings |
| 11,637 Lean lines (guarded), 48 theorems, 0 `sorry` | [§7.4](#74-the-lean-formalization) | `(Get-ChildItem Aether -Recurse -Filter *.lean \| Get-Content).Count`, then `Select-String "^\s*(theorem\|lemma)\s"` and `Select-String "\bsorry\b"` |
| Certified-library test counts (guarded) and the 500-instance arena ratio 1.0283 / 1.2826 | [`planner` docs](https://teerthsharma.github.io/Aether-Lang/integrated/planner/) | `cargo test -p aether-core --test planner -- --nocapture`; each suite by `--test <module>` |
| 707 Lean `example` blocks; 45 of 47 `Core.lean` theorems by `native_decide` | [§7.4](#74-the-lean-formalization) | `grep -cE '^\s*example' Aether/*.lean`; inspect the proof of each `theorem` in `Core.lean` |

### Numbers that are not reproducible from this repository

Listed separately rather than mixed into the table above, because the distinction is the point.

| Claim | Why it cannot be reproduced here | Where it came from |
|---|---|---|
| 56.6% / 80.9% block reduction at seq 1024 / 4096 | requires CUDA | upstream [`triton-lang/kernels#22`](https://github.com/triton-lang/kernels/pull/22) |
| 1.04×–3.48× sparse-vs-dense wall clock | measured on an RTX 4060 | same |
| 29.07 s before the `BTreeMap` refactor | requires checking out the parent of `27d70fa` | historical, same machine |

The upstream GPU figures are **cited, not claimed**.

---

## Repository layout

```
Aether-Lang/
├── crates/
│   ├── aether-core/          the mathematics, no_std, libm + heapless
│   │   ├── src/              see the module tree below
│   │   ├── tests/            the correctness suites, see below
│   │   └── examples/         scale_probe, routing_cost — reproduce the tables
│   ├── aether-gpu/           wgpu compute backend, 20 WGSL kernels, f32
│   │   ├── src/shaders.wgsl      matmul, tiled matmul, pairwise distance, softmax,
│   │   │                         fused gradients, Adam, SGD, scheduled attention
│   │   ├── tests/                107 tests: parity, gradcheck, f32 topology, doc guards
│   │   ├── mutants.sh            mutation harness, 26 mutants
│   │   └── FEATURES.md           measurements, negative results, corrections
│   ├── aether-lang/          lexer, parser, AST, interpreter, Titan VM
│   ├── aether-kernel/        no_std x86_64 microkernel
│   ├── aether-cli/           repl / run / check
│   ├── aegis-core/           duplicate, queued for deletion
│   └── aegis-cli/            duplicate, queued for deletion
├── Aether/                   Lean 4 formalization (ungated)
├── docs/                     MkDocs site, docs/theory.md, the claim ledger
├── examples/                 .aegis / .ag programs (extension debt)
├── CITATION.cff              citation metadata and DOI
└── .github/workflows/        ci.yml, docs.yml, publish.yml
```

### aether-core by module

```
crates/aether-core/src/                                    lines
├── lib.rs                                                   69   module exports, feature gates
├── monodromy.rs                                          1,409   injectivity, symmetry, PH dimension, Lyapunov (I.6)
├── persistence.rs                                        1,043   F2 reduction, Rips + witness, H0-H2
├── scheduled.rs                                          1,008   CSR block schedule, kernel, backward, baselines
├── arrangement.rs                                          889   planar arrangement invariants (I.3)
├── coupling.rs                                             856   coupling operator, fixed point, islands (I.8)
├── attention.rs                                            801   7 selectors, routing plan, cost model
├── manifold.rs                                             728   Takens embedding, streaming graph, pipeline
├── certify.rs                                              671   rounding certificates (I.2)
├── track.rs                                                661   assignment and circulation tracking (I.7)
├── linking.rs                                              588   Gauss linking, writhe, knot determinant (I.1)
├── memory.rs                                               553   manifold heap, Chebyshev guard
├── aether.rs                                               544   hierarchical block tree, drift
├── diagram.rs                                              475   bottleneck, Wasserstein, landscapes, images
├── orbit.rs                                                442   orbit partitions, five bounds (I.5)
├── topology.rs                                             422   byte-level shape signature, not the engine
├── kvwitness.rs                                            399   segment witnesses for sparse top-k (I.9)
├── planner.rs                                              399   offsets, transitive reduction, islands (I.10)
├── resolvent.rs                                            392   three attention corners of one head (I.4)
├── governor.rs                                             323   control law on ln(epsilon)
├── state.rs                                                224   SystemState, three deviation metrics
└── ml/                                                   5,075
    ├── classification.rs                                   780   6 classifiers
    ├── clustering.rs                                       736   KMeans, DBSCAN, agglomerative, auto-k
    ├── neural.rs                                           685   activations, optimisers, dense layers
    ├── convergence.rs                                      406   BettiNumbers, detector, residuals
    ├── regressor.rs                                        383   model escalation
    ├── autograd.rs                                         363   reverse mode, no finite-difference check
    ├── dataloader.rs                                       334   batching, shuffle
    ├── tensor.rs                                           304   N-d array
    ├── linalg.rs                                           296   reductions, distances, losses
    ├── benchmark.rs                                        293   escalation harness, not a benchmark
    ├── gossip.rs                                           243   ring averaging
    ├── convolution.rs                                      190   Conv2D, forward only
    └── mod.rs                                               62
```

```
crates/aether-core/tests/                                  lines
├── coupling.rs                                           1,292   18 tests
├── attention_contracts.rs                                1,255   29 tests
├── resolvent.rs                                            856   17 tests
├── orbit.rs                                                805   15 tests
├── monodromy.rs                                            766   18 tests
├── track.rs                                                762   15 tests
├── certify.rs                                              761   17 tests
├── persistence_invariants.rs                               740   12 tests
├── arrangement.rs                                          655   18 tests
├── scheduled_attention.rs                                  643   16 tests
├── linking.rs                                              627   18 tests
├── kvwitness.rs                                            511   15 tests
├── planner.rs                                              487   15 tests
├── ablation_baselines.rs                                   436    9 tests
├── diagram_distance.rs                                     422   17 tests
├── activation_contracts.rs                                 382    7 tests
├── attention_backward.rs                                   332    5 tests
├── readme_claims.rs                                        304    2 tests
├── persistence_scale.rs                                    231    7 tests  (--release)
├── ci_gate.rs                                              141    2 tests
├── orphaned_sources.rs                                     121   one test: every source file is in a module tree
└── doc_ratchet.rs                                           92   one test: the missing_docs ratchet
                                                          ─────
                                                         12,621
```

**12,621 lines of tests against 12,896 lines of non-`ml` source.** The topology core and the certified library are well covered; the 5,075-line `ml` subtree is covered mainly by in-module tests and one contract suite, and that asymmetry is the shape of this repository's assurance.

### aether-lang and aether-kernel

```
crates/aether-lang/src/                                    lines
├── interpreter.rs                                         1,971   the reference implementation
├── parser.rs                                              1,269   recursive descent, positioned AST
├── vm.rs                                                    911   TitanVM, no parity suite vs interpreter
├── lexer.rs                                                 507   includes the 4-byte seal codepoint
├── ast.rs                                                   408   node definitions
├── ascii_render.rs                                          179   terminal render
├── webgl_export.rs                                          117   WebGL export
├── python.rs                                                 81   pyo3 surface, bindings package empty
├── lib.rs                                                    71
└── mod.rs                                                    46

crates/aether-kernel/src/
├── main.rs                     kernel_main, panic handler
├── lib.rs                      re-exports aether-core
├── interrupts.rs               IDT, CURRENT_STATE behind spin mutexes
├── scheduler.rs                SparseScheduler, never-executed tests
├── allocator.rs                64 KB bump, #[global_allocator]
├── loader.rs                   ELF verification + byte-topology check
├── serial.rs                   COM1, the only output channel
└── boot/
    ├── bios.rs                 BootInfo, MemoryRegion, Framebuffer
    └── topology.rs             HardwareTopology, IoCaps
```

### Example programs

`examples/` holds the 17 programs below, one program per module of the certified library plus a pipeline (listed in §4.9), and `tour.aegis`, the program at the top of this document. All use the pre-rename `.aegis` and `.ag` extensions; the CLI accepts `.aegis` and warns on `.ag`.

| File | Bytes | What it demonstrates |
|---|---:|---|
| `simple.aegis` | 32 | the smallest program that runs |
| `seal_demo.aegis` | 549 | a seal loop, minimal |
| `titan_bench.ag` | 615 | Titan VM |
| `llm_benchmark.aegis` | 993 | LLM-shaped workload |
| `llm_demo.ag` | 1,263 | |
| `llm_benchmark.py` | 1,347 | |
| `hello_manifold.aegis` | 1,407 | embedding a series into a manifold |
| `ml_test.ag` | 1,426 | ML module surface |
| `regression_demo.aegis` | 1,602 | `regress` with model escalation |
| `3d_cluster.aegis` | 1,660 | clustering in 3D |
| `benchmark_compare.py` | 2,252 | Python-side comparison harness |
| `grand_benchmark.aegis` | 2,724 | |
| `neural_topology.aegis` | 2,895 | neural network and topology together |
| `benchmark_seal_vs_linear.aegis` | 4,638 | seal loop against a linear baseline |
| `seal_loop_demo.aegis` | 5,119 | the fullest seal-loop example |
| `visualization_demo.aegis` | 6,191 | `render` to ASCII and WebGL |
| `benchmark_suite.aegis` | 8,163 | |

Four of these have `benchmark` in the name and one exists to compare against something. **No number from any of them appears in this document, and none is run by CI.** They are demonstration programs and exploratory harnesses, not evidence; the core convergence claim is unmeasured (§9) whatever a directory listing suggests. Run one with `cargo run -p aether-cli -- run examples/hello_manifold.aegis`.

The two Rust examples under `crates/aether-core/examples/` differ in kind: they are the probes that produce numbers this document quotes. `scale_probe.rs` opens with a comment stating its role — *"Not a test: a probe that prints the numbers the docs are allowed to quote."*

---

## FAQ

**Is this production-ready?**
No. It is a research language, and its persistence engine takes 335 seconds to compute $H_0$ on 4,000 points. For production TDA, use ripser.

**Why is it called both AETHER and AEGIS?**
It was renamed, and the rename was done by copying directories. Half the doc comments say AEGIS, the examples use `.aegis`, and two duplicate crates remain in the workspace (§5.12).

**Do the Betti numbers actually help convergence?**
Unmeasured. The persistence engine is correct by every internal test, but no controlled experiment compares stopping on $\beta$-stability with stopping on a tuned scalar residual, and the built-in `regress` statement stops on a residual sign pattern or on its error (§3.29); a program can stop on a Betti vector with `seal until stable(...)` (§4.6). It is the most important missing number in this repository.

**Is `seal until convergence(1e-6)` topological?**
No. It stops when a pass changes the body's value by at most $10^{-6}$ in the max norm, which is a scalar tolerance (§4.6). A seal loop stops on whatever boolean condition it is given, on a tolerance with `convergence(ε)`, or, with `seal until stable(expr)`, when a watched value — a Betti vector, a certified face count — stops changing. Only the last is topological, and only when the watched value is.

**Is the persistent homology correct?**
It satisfies 12 invariants including the stability theorem, reproduces a closed-form ground truth to 1e-12, agrees with an independent union-find on $H_0$, produces exactly the pair counts §3.6 predicts, and survives mutation testing. It has **never been compared against ripser or GUDHI**. Those are different levels of assurance, and this document does not blur them.

**Why `no_std`?**
So the same persistence code that runs in the CLI runs in `aether-kernel` on bare metal. The mathematical surface is `libm` and `heapless`, plus a declared-but-unused `nalgebra` logged as a defect (§8.6).

**Does the kernel boot?**
It **compiles** for `x86_64-unknown-none`. Booting is not tested.

**Is there GPU acceleration?**
There is a GPU **backend** — `aether-gpu`, 20 WGSL kernels, resident tensors, 107 tests, an RTX 4060 over Vulkan — and no GPU **acceleration of the language or the engine**, because nothing in `aether-core` or `aether-lang` calls it. `Tensor::matmul` pays above $n = 128$ by a magnitude that is not measurable on this machine; `pairwise_sqdist` never pays (§5.11). This answer once read "No", and referred to `wgpu`, `pollster` and `bytemuck` entries with zero call sites that have since been deleted; the current backend is a separate crate on a current wgpu.

**Does topological convergence eliminate hyperparameter tuning?**
No. It replaces a continuous threshold $\varepsilon$ with a discrete stability window — still a hyperparameter. The argument for it is that an integer count is easier to reason about and does not interact with the scale of the loss.

**Why is `estimate_betti_1` an estimate when there is a full persistence engine?**
It is the exact cycle rank of the streaming $\varepsilon$-graph and an upper bound on $\beta_1$ of the corresponding Rips complex (§3.17) — cheap enough to run per sample, where the engine is not.

**There are two single-linkage implementations. Is that a bug?**
Duplication, not a bug. `attention::single_linkage_clusters` is cross-checked against the persistence engine; `ml::clustering::AgglomerativeClustering` is not, and the two are not checked against each other (§7.2).

**Why is the kernel's allocator 64 KB of bump when `aether-core` has a sophisticated heap?**
Different layers. `AegisAllocator` is the kernel's `#[global_allocator]`, correct for a kernel that boots and runs a fixed workload; the manifold heap is a data structure inside `aether-core`.

**Is `verify_binary_topology` a security feature?**
No. It is a structural plausibility heuristic that an adversary defeats by padding (§5.9).

**The governor's doc comment says PID. Is it?**
It is PD on $\ln\varepsilon$; an integral term is accumulated and unused. Its local stability condition is $0 < \alpha < 2 - 2\beta$, satisfied by the shipped gains (§3.26).

**Why `-A clippy::style` when the project is this particular about correctness?**
A gate that fails on first contact gets switched off, and the correctness lints go with it. Strict where lints find bugs, quiet where they find taste; the advisory step prints the rest (§5.14).

**Why not make $H_0$ use union-find?**
Because `h0_matches_an_independent_union_find` compares the engine's $H_0$ against a union-find, and making the engine use one would make the test tautological (§5.14).

**Is the Lean formalization proving things about the Rust?**
No. 45 of its 48 theorems check the Lean executor against the Lean semantics on specific programs; there is no correspondence to the Rust, and no `lake build` in CI (§7.4).

**How fast is this compared to ripser?**
Unmeasured, and deliberately absent from the prior-art table. The inference from the measured ceiling is that it is substantially slower; that is an inference, not a benchmark.

**Why is the CHANGELOG almost empty?**
It is a 72-line log from the AEGIS era that has not tracked recent history; commit messages and the claim ledger carry it.

**What would change the assessment of the whole approach?**
A controlled experiment showing that $\beta$-stability stopping, wired to the persistence engine, does no better than a tuned scalar residual on real problems. A negative result there would be worth more to this project than another green suite.

---

## Contributing

The claim ledger at [`docs/reference/status.md`](docs/reference/status.md) is the authority on what is gated, what is ungated and which command produces each piece of evidence. Ranked by value, highest first:

1. **The external parity harness.** Compare the engine against a pinned ripser or GUDHI on shared fixtures. Every correctness claim currently rests on internal invariants, which a self-consistently wrong implementation can satisfy. Nothing else on this list is close.
2. **The controlled convergence experiment.** Using `seal until stable(...)` over a Betti vector, ideally with a window longer than one pass, test whether stopping on $\beta$-stability beats stopping on a tuned scalar residual on real problems. This is the core premise of the language and it is unmeasured.
3. **QEMU boot logs.** Move the kernel from "compiles" to "boots" with evidence.
4. **Gate the Lean tree.** Add `lake build` to CI, or cut the tree.
5. **Delete `aegis-core` and `aegis-cli`.** Mechanical, uncontroversial, and removes the most embarrassing thing in the tree.
6. **Wire `Tensor::matmul` to `aether-gpu`, or decide not to.** Cost and precision are measured (§5.11); what remains is whether `ml::Tensor` may drop to f32.
7. **A finite-difference check for `ml/autograd.rs`.**
8. **Differential tests between the interpreter and `TitanVM`.**
9. **Make the seal loop's headline spelling run**, or remove `convergence` from the keyword list, and either construct `ConvergenceCond::BettiStable` or delete it.
10. **Decide what block salience computes.** Write each death once, per the elder rule, or change the claim and the multiset test.

The house rule for any contribution that adds a number to this document: **it comes with the command that reproduces it.** A number without a reproduction command does not go in a table, and a benchmark on an implementation whose correctness is unestablished is not a result.

---

## Glossary

**Seal loop** — `seal until expr { body }`, also spelled `🦭 until`: a loop whose termination condition is an arbitrary boolean expression, evaluated before each iteration, for at most 1,000 iterations; `seal until stable(expr)` stops once a pass of the body leaves `expr` unchanged. It is topological when its condition is.

**Positive / negative simplex** — in the column reduction, a simplex whose column reduces to zero creates a homology class (positive); one whose column keeps a lowest entry destroys the class created by that entry (negative) (§3.5).

**Essential class** — a class that never dies within the filtration; its bar runs to infinity (`death: None`). A connected cloud has exactly one essential $H_0$ class.

**Elder rule** — when two components merge, the younger dies. In a Rips filtration all vertices are born at 0, so only the death values are canonical (§3.8).

**Witness complex** — the landmark-based construction of §3.9 that makes persistence viable at 24 landmarks on a Cortex-M3. An approximation, labelled as one.

**Fail-fast budget** — the persistence engine's caps. Exceeding one returns `TooManyPoints` or `TooManySimplices` rather than subsampling or exhausting memory; a time budget, not a correctness limit.

**Placement** — $(m(S) - m(R))/(m(O) - m(R))$ at equal budget: 0 is random, 1 is an oracle that read every score (§3.21).

**Gap ratio** — the ratio of the first single-linkage merge height above the cut to the last one below it, computed from the $H_0$ barcode alone. Structured clouds measure at least 2.70, chained ones at most 1.04 (§3.22).

**Chaining** — the single-linkage behaviour on a cloud with no density gaps: points are absorbed one at a time into one giant component, `[61, 1, 1, 1]` out of 64 keys. Not a bug; $H_0$ reporting that there is no structure to route on.

**Sink block** — in the scheduled-attention configuration, a leading block every query attends to regardless of salience. Attention sinks are an empirical phenomenon in transformers; the scheduler reserves them explicitly.

**Budget** — the number of keys, or blocks, a selector may use per row. Every ablation holds it equal across selectors, because a selector that declines to select posts catastrophic numbers without losing on mechanism (§8.3).

**Manifold heap** — the `no_std` allocator in `aether_core::memory`, organising objects in spatial blocks and reclaiming cold, unmarked ones under the guard of §3.25.

**Generational handle** — `Gc<T>`: an index plus a generation counter, so a stale handle whose slot was reused returns `None` instead of aliasing.

**Geometric concentrator** — the streaming component in `manifold.rs` that tracks per-coordinate variance and projects onto the highest-variance axis (axis-aligned, not PCA).

**Titan VM** — the bytecode VM in `aether_lang::vm`, behind the interpreter on coverage.

**Bio mode** — the label the CLI prints as its execution mode. It changes nothing.

**`ponytail:` comment** — the repository's marker for a deliberate shortcut, naming its ceiling and the trigger that should force revisiting it. The topology core has two, in `attention.rs` and `diagram.rs`; the certified library adds one each in `certify.rs` and `track.rs`; all four name concrete triggers.

**Active / Ungated / Hardware-gated** — the status vocabulary of §1.5: evidence produced by a CI command; evidence that exists but that no CI command produces; evidence that needs an adapter no CI runner has.

---

## License

Aether-Lang is distributed under the **Aether-Lang Custom Attribution License, Version 1.0** — see [LICENSE](LICENSE). In summary: personal use is unlimited with attribution; educational use is permitted with attribution in course materials; open-source use is permitted only if the project prominently credits Teerth Sharma and includes the license in full; commercial use requires prior written permission. Forks and derivatives that are distributed must carry the same license and attribution requirement. The summary is not the license; the file is.

Copyright © 2026 Teerth Sharma. The Lean formalization, the persistence engine, the language, and every mistake catalogued above are original work. The scheduled-attention module rebuilds in Rust the kernel the same author contributed to [`triton-lang/kernels#22`](https://github.com/triton-lang/kernels/pull/22) under that repository's license.

To cite this work, use [`CITATION.cff`](CITATION.cff) (DOI [10.5281/zenodo.21997728](https://doi.org/10.5281/zenodo.21997728)).

---

<p align="center">
  <strong>Invented by <a href="https://teerthsharma.vercel.app/">Teerth Sharma</a></strong><br>
  <a href="https://github.com/teerthsharma/Aether-Lang">github.com/teerthsharma/Aether-Lang</a> · <code>teerthsharma@outlook.com</code>
</p>

<p align="center">
  <em>Every number above was measured. Every claim names its control.</em>
</p>
