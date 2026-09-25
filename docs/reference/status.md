# Status Matrix

Every row names the command that produces its evidence. A row whose command does
not run in `.github/workflows/ci.yml` is not Active, however many tests its file
contains.

Run the whole ledger:

```bash
cargo test --workspace --exclude aether-kernel
```

## Active

| Surface | Evidence | Command |
| --- | --- | --- |
| Lexer tokens for current syntax | 4 tests in `lexer.rs` | `cargo test -p aether-lang` |
| Parser for statements and operators | 7 tests in `parser.rs` | `cargo test -p aether-lang` |
| Interpreter assignments, loops, functions | 11 tests in `interpreter.rs` | `cargo test -p aether-lang` |
| Numeric-list manifold embedding | `manifold_embed_uses_user_numeric_list` | `cargo test -p aether-lang` |
| `topology.ph` and `topology.betti` | `topology_betti_uses_persistent_homology_engine` | `cargo test -p aether-lang` |
| Bounded persistent homology H0/H1/H2 | 9 tests in `persistence.rs` | `cargo test -p aether-core --lib persistence` |
| Lazy witness mode | `witness_mode_uses_landmarks_without_rejecting_full_signal_size` | `cargo test -p aether-core --lib persistence` |
| **Persistence invariants** | **12 tests in `tests/persistence_invariants.rs`** | **`cargo invariants`** |
| Block metadata and compression selection | 4 tests in `aether.rs` | `cargo test -p aether-core` |
| Drift detector | `aether.rs` test | `cargo test -p aether-core` |
| Sparse graph and pipeline | 7 tests in `manifold.rs` | `cargo test -p aether-core` |
| Geometric governor | 5 tests in `governor.rs` | `cargo test -p aether-core` |
| CLI parse-error formatting | `aether-cli` test | `cargo test -p aether-cli` |
| `no_std` core on a real embedded target | Builds for `thumbv7m-none-eabi` | `cargo embedded` |
| Kernel compiles for bare metal | Builds for `x86_64-unknown-none` | `cargo kernel` |
| **Diagram metrics** (bottleneck, p-Wasserstein) | **17 tests in `tests/diagram_distance.rs`** | `cargo test -p aether-core --test diagram_distance` |
| **Vectorizations** (landscapes, images, entropy) | same suite | `cargo test -p aether-core --test diagram_distance` |
| **Attention correctness contracts** | **29 tests in `tests/attention_contracts.rs`** | `cargo test -p aether-core --test attention_contracts` |
| **Topology-derived scheduled attention** | **16 tests in `tests/scheduled_attention.rs`** | `cargo test -p aether-core --test scheduled_attention` |
| **Scale past 32 points** | **7 tests in `tests/persistence_scale.rs`** | `cargo test -p aether-core --test persistence_scale --release` |

<div class="ts-viz" data-viz="k-ledger" data-title="Test counts behind the Active rows" data-caption="Only rows that state a count are shown. The scheduler row comes from Partial Or Gated: its 4 tests never execute."></div>

### What the invariant suite asserts

Each row is a theorem stated as an executable assertion. [Theory →](../theory.md#ev-invariants)

| Property | Assertion | Constant |
| --- | --- | --- |
| Permutation invariance | Shuffled rows give an identical diagram | tolerance 1e-9 |
| Isometry invariance | Rotation + translation move the diagram by 0 | tolerance 1e-9 |
| Scale equivariance | Scaling by `c` scales all births/deaths by `c` | `c` ∈ {0.125, 0.5, 2, 37} |
| Stability (Cohen-Steiner–Edelsbrunner–Harer) | ‖perturbation‖ ≤ ε ⇒ bottleneck ≤ 2ε | **2ε**, the Rips constant |
| Circle ground truth | Exactly one long H₁ bar, dying at **2r·sin(π·⌈n/3⌉/n)** — the exact regular-polygon chord, which equals √3·r when 3 divides n and tends to it otherwise | exact to 1e-12 for n ∈ {9,10,11,12,13,17,24,48} |
| Cluster ground truth | `k` separated clusters give β₀ = `k` | k = 3, gap 4.0 |
| Negative control | A Gaussian blob yields **zero** long H₁ bars | 4 seeds |
| Elder rule | H₀ deaths equal an independent union-find MST | exact |
| ∂∘∂ = 0 over 𝔽₂ | Every simplex in 5 complexes | exact |
| Filtration monotonicity | Every face present, and preceding its coface | exact |

<div class="ts-viz" data-viz="k-circle" data-title="Circle ground truth: 2r·sin(π·⌈n/3⌉/n)" data-caption="The long H1 bar's death for n points on a unit circle. The blue points are the n values the suite asserts to 1e-12."></div>

| Injected defect | Invariant tests catching it (of 11) | Pre-existing example tests (of 6) |
| --- | --- | --- |
| Dropped edge in the triangle filtration | 4 | 0 |
| Hardcoded `+0.001` absolute epsilon | 4 | 0 |
| Reduction stops after one column operation | 7 | 1 |

<div class="ts-viz" data-viz="k-mutation" data-title="Mutants killed, per injected defect" data-caption="Each count is copied from the mutation paragraphs. The weak pre-existing tests and the mutant that was never run appear in orange."></div>

- Bottleneck: exact (binary search over costs, Kuhn matching, diagonal projection).
- Essential-class counts must match exactly.

### What the diagram suite asserts

`tests/diagram_distance.rs`, 17 tests over `aether_core::diagram`.

| Property | Assertion |
| --- | --- |
| Metric axioms | `d(a,a) = 0`, symmetry, triangle inequality |
| Hand-computed pairing | Two bars at L∞ distance 0.2 give exactly 0.2 |
| Diagonal projection | An unmatched bar of persistence 0.4 costs 0.2 |
| Bottleneck vs Wasserstein | Two bars displaced 0.2 each: bottleneck 0.2, 1-Wasserstein 0.4 |
| Ordering | `W_p ≥ bottleneck` for p ∈ {1, 2, 4} |
| Stability on real diagrams | Library metric reproduces the 2ε bound end to end |
| Landscape tent shape | Bar [0,2] gives samples 0, 0.5, 1, 0.5, 0 |
| Landscape level ordering | λ₁ ≥ λ₂ ≥ … pointwise, on **crossing** bars |
| Landscape k-th largest | λ₁(1.0) = 0.6 from the later bar, λ₂(1.0) = 0.2 from the earlier |
| Landscape stability | sup-norm ≤ bottleneck (1-Lipschitz) |
| Image weighting | A long bar deposits > 5x the mass of a near-diagonal one |
| Image translation equivariance | Shifting births and the window gives an identical raster |
| Kernel width | σ = 0.05 concentrates > 3x more than σ = 0.6 |

<div class="ts-viz" data-viz="k-diagram" data-title="Diagram fixtures: bottleneck, Wasserstein, landscapes" data-caption="The bars come from tests/diagram_distance.rs. Distances are recomputed here by exhaustive matching, with projection onto the diagonal."></div>

Mutation-tested, five injected defects:

| Injected defect | Caught by |
| --- | --- |
| Landscape skips the per-sample descending sort | 2 of 17 |
| Image drops the linear persistence weight | 1 of 17 |
| Wasserstein returns the max instead of the sum | 1 of 17 |
| Image hardcodes the Gaussian width, ignoring σ | 1 of 17 |
| Bottleneck forbids diagonal projection | (not run — the ∞ costs make the matching search diverge) |

The sort and σ mutants first survived; both tests were rewritten until they died. [Theory →](../theory.md#ev-diagram-mutation)

### What the attention suite asserts

`tests/attention_contracts.rs`, 29 tests over `aether_core::attention`, ordered by bug caught per line of test.

| Contract | Assertion |
| --- | --- |
| **Selection cost** | **Dot products per row measured against dense; oracle priced at full dense, random at 0** |
| **Plan/cost agreement** | **`routing_plan.cost_ratio` equals the selector's measured cost to 1e-12, over 6 key distributions** |
| **Routing decision** | **`worth_routing` agrees with the measured cost against the threshold on 27 cases, and is not constant** |
| **Adaptive bound** | **Never exceeds dense cost on any key distribution; declined routing still recovers 1.000 of mass** |
| **Barcode separation** | **`gap_ratio` structured-min 2.70 > chained-max 1.04, no overlap** |
| Cluster/H0 agreement | Single-linkage merge heights equal the engine's H0 deaths to 1e-12 |
| Per-key rescale invariance | Multiplying individual keys by gains in [1, 7] leaves cluster assignment identical |
| Dense parity | Full mask reproduces dense SDPA **bitwise**, over 5 shapes |
| Convex combination | Every output coordinate lies inside the value range |
| Closed form | Zero queries give exactly the mean of `v` |
| Mask fidelity | Perturbing a masked-out value row leaves the output bitwise identical |
| Pattern equality | Realized pattern equals requested, element-wise, for all 5 selectors |
| Budget | No row exceeds its budget |
| Causality | Perturb position `j`; every output at `i < j` is bitwise unchanged |
| Causal selection | No selector picks a future key |
| All-masked row | Returns zeros, not NaN, and does not contaminate other rows |
| Overflow | Logits scaled 400x stay finite and saturate onto a single value row |
| Determinism | 5 repeated runs bitwise identical, selector included |
| Shape edges | `seq ∈ {1, 7, 8, 9, 17}` × `head_dim ∈ {1, 3, 6, 8}` |
| Scale equivariance | Scaling q and k by `c ∈ {0.01, 0.5, 2, 100}` gives an identical mask |
| Oracle bound | No same-budget selector recovers more mass than oracle top-k |
| Fair ablation | Every selector spends the same mean budget before mass is compared |

**No gradient check here:** `aether_core::attention` has no backward pass. `aether_core::scheduled::scheduled_attention_backward` has one, finite-difference gradchecked in `tests/attention_backward.rs`.

### The routed selector — the fix, and what it actually costs

`Selector::TopologicalRouted`: H0 clustering of unit-normalised keys picks candidates, the exact dot product ranks them. [Theory →](../theory.md#ev-routed-selector)

| key-norm spread | nearest-neighbour | routed |
|---|---|---|
| 0.0 | +0.884 | **+0.898** |
| 2.0 | +0.202 | **+0.885** |
| 4.0 | −0.109 | **+0.874** |
| 8.0 | −0.285 | **+0.866** |

Routed stays flat where nearest-neighbour goes negative.

<div class="ts-viz" data-viz="k-spread" data-title="Placement vs key-norm spread, reproduced live" data-caption="The lines plot the page tables. The rings re-run the test fixture in the browser (seeds 67 to 81, seq 32, budget 6, bit-exact RNG) and land on the same numbers."></div>

Placement without cost is not a result. Dot products per row vs dense, 64 keys, budget 8:

```bash
cargo run -p aether-core --example routing_cost --release
```

| key distribution | H0 component sizes | cost vs dense | placement |
|---|---|---|---|
| uniform random | `[61, 1, 1, 1]` | **0.999** | +0.942 |
| 4 real clusters | `[16, 16, 16, 16]` | **0.449** | +0.990 |
| 8 real clusters | `[8, 8, 8, 8]` | 0.528 | +0.995 |
| 16 real clusters | `[4, 4, 4, 4]` | 0.733 | +0.989 |

- Uniform keys: router examines every key; the +0.94 placement is worth nothing.
- Structured keys: **+0.92 to +0.99 of oracle at 0.449x dense**, holding at key-norm spread 8.
- Claim: a sparsity win exactly when keys have H0 structure, none otherwise — `routing_is_sparse_only_when_the_keys_have_h0_structure`.

<div class="ts-viz" data-viz="k-routing-cost" data-title="What routing costs, and when it declines" data-caption="The rows are copied from the routing_cost, Adaptive and gap_ratio tables. Anything that failed the 0.6 threshold or was rejected is shown in orange."></div>

<div class="ts-viz" data-viz="k-router" data-title="select_mask on 64 keys, live" data-caption="Blue cells are the keys the selector picked, and shading is dense softmax weight. The cost ratio, gap ratio and worth_routing flag come from the ported routing_plan."></div>

### Deciding whether to route, at runtime

`routing_plan(k, seq, head_dim, clusters, budget, causal)` clusters once per key tensor and reports the cost **before any query runs**. [Theory →](../theory.md#ev-routing-plan)

| field | meaning |
| --- | --- |
| `cost_ratio` | dot products per row as a fraction of dense; asserted to equal what the selector actually spends, to 1e-12 |
| `largest_cluster_share` | near 1 means chaining |
| `gap_ratio` | from the H0 barcode alone: first merge height above the cut ÷ last below it |
| `worth_routing` | `cost_ratio < 0.6` |

The barcode alone separates the regimes (6 trials each, seq 48):

| key distribution | `gap_ratio` |
| --- | --- |
| 6 real clusters | **min 2.70** |
| uniform random | **max 1.04** |

No overlap.

`Selector::Adaptive { budget, clusters }` acts on the plan:

| key distribution | decision | cost vs dense | quality |
| --- | --- | --- | --- |
| structured | route | 0.449 | placement **+0.980** |
| unstructured | decline | 1.000 | recovers **1.000** of attention mass |

- Fallback is dense: never worse than dense in cost or quality. A budget-6 window fallback scored **+0.014**. `Local` is opt-in.
- Unstructured is scored as recovered mass, not placement (placement vs dense once read **+7.6**).

### The nearest-neighbour ablation — negative

Same-budget oracle top-k ablation. Seq 32, head_dim 8, budget 6, uniform random q/k:

```bash
cargo test -p aether-core --test attention_contracts -- --nocapture
```

| seed | random | topological | oracle | placement |
|---|---|---|---|---|
| 67 | 0.4875 | 0.5789 | 0.5879 | +0.910 |
| 71 | 0.4813 | 0.5648 | 0.5743 | +0.898 |
| 73 | 0.4882 | 0.5619 | 0.5742 | +0.857 |
| 79 | 0.4947 | 0.5795 | 0.5850 | +0.939 |

<div class="ts-viz" data-viz="k-seeds" data-title="Four seeds of the tautological placement" data-caption="With equal key norms, nearest-neighbour ranks keys like the dot product does. These rows must not be quoted as a result."></div>

That placement is **largely tautological and must not be quoted.** Vary key norms and it decouples (8 trials each). [Theory →](../theory.md#ev-nn-ablation)

| key-norm spread | random | topological | oracle | placement |
|---|---|---|---|---|
| 0.0 | 0.4902 | 0.5667 | 0.5769 | **+0.884** |
| 0.5 | 0.4901 | 0.5705 | 0.6002 | +0.732 |
| 1.0 | 0.4899 | 0.5613 | 0.6242 | +0.533 |
| 2.0 | 0.4892 | 0.5265 | 0.6723 | +0.202 |
| 4.0 | 0.4873 | 0.4577 | 0.7616 | **−0.109** |
| 8.0 | 0.4848 | 0.3725 | 0.8841 | **−0.285** |

- **At high spread the selector is worse than uniform random.** Pinned by `the_topological_advantage_collapses_when_key_norms_vary`.
- First run: −3.6 to −4.2, from an absolute `epsilon` 0.6 vs median distance 2.4 (1.0 keys/row vs 5.5). Radius is now relative; equal mean budget is asserted.

### Scheduled attention — the Rust port of triton-lang/kernels#22

`aether_core::scheduled` ports [triton-lang/kernels#22](https://github.com/triton-lang/kernels/pull/22) from CUDA to anywhere `aether-core` runs, including `no_std`. [Theory →](../theory.md#ev-scheduled)

| Half | Nature | Checked against |
| --- | --- | --- |
| CSR block schedule | combinatorial | exact set equality; the lower-triangular CSR the Python builder emits, `[0, 1, 3, 6, 10]` / `[0, 0, 1, 0, 1, 2, 0, 1, 2, 3]` for 4 blocks |
| Kernel | numeric | dense masked attention, to 1e-12 |

Schedule sources, all clamped causally:

- **sink blocks**;
- a **local window**, which always contains the query block, so no row is empty;
- **top-k 0D-persistence salient blocks**.

<div class="ts-viz" data-viz="k-csr" data-title="CSR block schedule, built by the ported topology_block_schedule" data-caption="The default is the unit-test fixture (16 blocks, seed 13), which yields 56 / 136. The outlined row is the query tile walking its CSR row. Mass uses the ported block_mass_recovered."></div>

| Source | Setting | Block reduction |
| --- | --- | --- |
| This port (measured) | 16 blocks, `local_radius=1, sink=1, topk=2` | **56 / 136, 58.8%** |
| Triton PR (external, RTX 4060) | seq 1024 | 56.6% |
| Triton PR (external, RTX 4060) | seq 4096 | 80.9% |

<div class="ts-viz" data-viz="k-reduction" data-title="Block reduction: this port vs the Triton PR" data-caption="Only 58.8% is measured here. The Triton figures are external and shown in grey."></div>

| Test | Asserts |
| --- | --- |
| `block_salience_is_the_elder_rule_over_centroids` | Every non-zero salience equals an H0 death from this crate's engine; **exactly one** block scores 0 |
| `the_schedule_depends_on_block_order` | Per-block salience is *not* permutation-equivariant (ties break on index); the multiset is invariant |

<div class="ts-viz" data-viz="k-elder" data-title="Elder-rule salience, and why block order matters" data-caption="Shuffling block order moves the zero-salience block and changes per-block scores, while the sorted multiset (the H0 barcode) stays identical."></div>

## Partial Or Gated

| Surface | Gate |
| --- | --- |
| Titan VM language parity | VM tests per construct |
| Full static type checking | Static checker and diagnostics |
| Complete class/object semantics | Interpreter tests and docs |
| Render as a user-facing graphics command | CLI artifact or exported file test |
| `Seal.train` semantic contract | Interpreter test and training artifact |
| ML model quality | Deterministic datasets and baseline metrics |
| Cohomology, Mapper, multiparameter persistence | Not implemented. |
| Sparse-attention **speedup** | No attention kernel runs on a GPU (`aether-gpu` exists but nothing calls it), so there is nothing to measure a speedup against. `aether_core::attention` is a CPU reference for correctness and ablation only. |
| Attention **backward pass** | Not implemented for `aether_core::attention`. The scheduled port's backward exists and is gradchecked (`tests/attention_backward.rs`). |
| Scheduled-attention **wall-clock** speedup | The Triton original measured 1.04x-3.48x sparse-vs-dense-CSR on an RTX 4060. This port is a scalar CPU kernel with no SIMD, no threading and no GPU; it reproduces the *answer* and the *block reduction*, not the timing. |
| Batched / multi-head scheduling | The Triton kernel shares one CSR schedule across batch and head lanes. This port handles a single `[seq, head_dim]` lane; batching is a loop the caller writes. |
| Topological **routing** speedup in wall-clock | Cost is counted in dot products, not seconds. A wall-clock claim needs a GPU path wired into routing, which does not exist. |
| Topological routing on real activations | Measured only on synthetic keys. Real attention key distributions are heavy-tailed and may have different H0 structure. |
| Topological *nearest-neighbour* key selection as an attention-mass proxy | Holds only for roughly homogeneous key norms; **negative at high spread** (see the ablation above). Needs either key normalisation or a dot-product ranking over a topology-derived candidate set, plus a re-run of the ablation. |
| External TDA parity | Bottleneck ≈ 0 against a pinned `ripser`/`gudhi` on shared fixtures. The invariant suite above is **not** parity: a self-consistently wrong implementation can satisfy every internal property. |
| Sparse scheduler | 4 tests exist in `scheduler.rs`, but `aether-kernel` is a `no_std` binary with no test harness, so **none of them execute in CI or locally**. Needs a host-testable extraction or a QEMU harness. |
| Bare-metal boot | The kernel compiles for `x86_64-unknown-none` (Active, above). Booting it is not tested: needs QEMU boot logs and a hardware matrix. |
| Security detection claim | Threat model, corpus, metrics |
| GPU acceleration | A separate `aether-gpu` crate (wgpu compute backend, hardware-gated tests) exists, but nothing in `aether-core` or `aether-lang` calls it. The old `wgpu`/`pollster`/`bytemuck` entries in `aether-lang` were deleted. |

### Measured scale

Single core, Windows 11. The point cap is a time budget, not a correctness limit.

```bash
cargo run -p aether-core --example scale_probe --release
```

| dim | n    | pairs | seconds |
|-----|------|-------|---------|
| 0   | 200  | 200   | 0.049   |
| 0   | 1000 | 1000  | 5.781   |
| 0   | 4000 | 4000  | 335.049 |
| 1   | 60   | 1771  | 0.117   |
| 1   | 120  | 7141  | 2.202   |
| 1   | 200  | 19901 | 20.728  |
| 1   | 300  | 44851 | 131.343 |
| 2   | 30   | 4090  | 0.100   |
| 2   | 50   | 19650 | 1.859   |
| 2   | 70   | 54810 | 15.338  |

`tests/persistence_scale.rs` (release): **29.07 s → 1.10 s**, 26x, after indexing the face lookup. [Theory →](../theory.md#ev-scale)

<div class="ts-viz" data-viz="k-scale" data-title="scale_probe timings, log-log" data-caption="All ten rows from the scale_probe table, single core, Windows 11."></div>

Defaults now: `h2_default` 48 points, `h1_dense` 128, `h0_only` 512.

## Removed From Active Claims

- Unverified speedup factors.
- Placeholder benchmark rows.
- Broad "verified full trust chain" language for this repository.
- Production security guarantees.
- Hardware acceleration claims.
- "Current World's Fastest Agentic AI Language" — was the published PyPI
  description in `pyproject.toml`. No benchmark in this repository supports it,
  and it contradicts the Evidence Policy in `README.md`.

## Known-broken, now fixed

Recorded so the ledger shows what the absence of a working CI had hidden:

| Was | Now |
| --- | --- |
| CI triggered on `main`/`develop`; this repo's branch is `master`, so **no CI run had ever executed** | Triggers on `master` |
| `aether-kernel` did not compile: missing `alloc::vec::Vec` and `alloc::string::ToString` imports under `no_std`, `multiboot2::load` removed in 0.24, `framebuffer_tag()` return type changed | Compiles for `x86_64-unknown-none` |
| `BootInfo::config_root` returned a pointer to the RSDP *signature string*, not the ACPI root table address | Returns XSDT (ACPI 2.0+) or RSDT address |
| `Dockerfile` copied `aegis-core/`, `aegis-lang/`, `aegis-kernel/` from the repo root; the crates live under `crates/` | Copies `crates/`, builds `aether-cli` |
| `.dockerignore` excluded `Cargo.lock`, so images could not build from the locked dependency set | `Cargo.lock` retained |
| `cargo kernel` and `cargo lang` aliases named crates that do not exist | Renamed to the real crates |
| `cargo check -p aether-core --no-default-features` (in `README.md`) fails: `no_std` needs `alloc` and a bare-metal target | `cargo embedded` |
| `aether_core::os` (304 lines) had zero callers and two traits with zero implementors | Deleted |
