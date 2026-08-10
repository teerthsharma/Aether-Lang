# aether-gpu feature ledger

State carried between iterations of the running build loop. Each row is a
capability that exists in the tree and has a test or a measurement behind it.
A row is only added once the thing works; planned work lives in **Next**.

Measured on NVIDIA GeForce RTX 4060 Laptop GPU, Vulkan, `DiscreteGpu`, driver
595.79, Windows 11, f32 throughout (WGSL has no f64).

## What is true now

Everything below this section is in discovery order and includes conclusions
that were later retracted. That is deliberate — the reasoning that produced a
wrong answer is worth keeping — but it means the current state has to be
reconstructed from a sequence of corrections. This section is the state.

**The backend works and nothing uses it.** 13 WGSL kernels, resident tensors,
batched submission, 70 tests, 0 of 10 mutants escaping. No line of `aether-core`
or `aether-lang` calls it.

**Both integrations are measured; neither is made.**

| candidate | verdict | why |
|---|---|---|
| `Tensor::matmul` | **done** — `tensor_matmul` | **36× at n=512**, measured on the shipped bridge with conversion and stride gather counted. Crossover n=128. Opt-in per call site rather than a change to `Tensor`, so the f32 question is asked by the caller who knows what the result feeds |
| `pairwise_sqdist` | **not worth doing at any size** | 90–100% of its time is transfer, and the persistence reduction is CPU-side so the matrix must come back |

**One rule predicts both**, and is the single most portable thing here: with a
CPU consumer, an operation pays according to its arithmetic per byte returned.
Matmul is O(n³) over O(n²) bytes, so the ratio grows with n and must cross over.
Pairwise distance is O(n²d) over O(n²) bytes, so the ratio is fixed at d and
never does. See [the rule](#the-rule-arithmetic-per-byte-returned).

**Precision is settled for both.** f32 costs ~5e-7 relative on matmul — fine for
training and thresholding, not for anything asserting at 1e-9. For topology, an
end-to-end diagram built from real kernel output has Betti numbers identical to
f64, though the *guarantee* of identical combinatorics is unavailable past n≈32.

**Open defects:** an exit-time crash in wgpu's multi-backend instance teardown,
worked around by instantiating one backend
([details](#a-crash-now-reproducible-at-1-in-5)).

### Which numbers here are checked, and which are snapshots

Two counts in this file are bound to the repository and fail the build when they
drift. Everything else is a measurement taken once, and the distinction matters
more than it looks: a reader has no way to tell a self-checking figure from one
that was true in August unless the document says which is which.

| figure | status | how to re-derive |
|---|---|---|
| test count | **checked** — `the_documented_test_count_matches_the_tests_that_exist` | `cargo test -p aether-gpu --test features_doc` |
| mutant count | **checked** — `the_documented_mutant_count_matches_the_harness` | same |
| f32 matmul error ~5e-7 | **bounded** — a test asserts it stays within 1e-9…1e-4 | `cargo test -p aether-gpu --features gpu f32_matmul` |
| gradient agreement, 7,732 entries | **bounded** — tolerances asserted per fixture | `cargo test -p aether-gpu --features gpu --test gradcheck` |
| Betti numbers unchanged under f32 | **checked** — asserted, not reported | `cargo test -p aether-gpu --features gpu --test f32_topology` |
| every timing, every ratio | **snapshot** | `cargo run -p aether-gpu --example gpu_bench --release` |
| crossover n=128, 36× at n=512 | **snapshot** | `cargo run -p aether-gpu --example tensor_crossover --release` |
| crash rates (8/60, 0/180) | **snapshot** | `crates/aether-gpu/examples/teardown_repro.rs`, 30 runs a variant |

The snapshots cannot be bound and it is not a gap in the tooling. A timing
depends on the adapter, the driver, the thermal state and what else the machine
was doing; asserting one would produce a test that fails for reasons unrelated
to the code, which is worse than a number carrying a date. What they can carry
is the command that reproduces them, and every row above has one.

The bounded rows are the middle case: the *value* moves between runs, but the
*claim* — that it stays inside a stated range — is asserted. That is the
strongest form available for a measured quantity, and it is why those tests
assert both an upper and a lower bound rather than only the direction that
would embarrass the code.

### Conclusions this document reached and then withdrew

Listed here so a reader is not misled by encountering them mid-file:

| claim | status |
|---|---|
| "The cross-validated accuracy numbers were measuring interpolation" | **withdrawn** — interleaved CV and independent draws agree; the distance ratio does not detect leakage |
| "Buffer churn from unread dispatches causes the crash" | **withdrawn** — a pattern with zero dispatches crashes at the highest rate |
| "Allocation churn causes the crash" | **withdrawn** — it is the multi-backend instance |
| "f32 is viable for the filtration" | **narrowed** — true in practice, not guaranteed past n≈32 |
| "The GPU kernel's output needs symmetrising" | **withdrawn** — it is bitwise symmetric by construction |
| "The suite never reports success for work that did not happen" | **withdrawn** — it did exactly that for twenty-odd commits |

## Shipped

### Device and context

| # | Capability | Evidence |
|--:|---|---|
| 1 | `GpuContext` — adapter enumeration, device, queue, pipeline cache | `the_gpu_reports_which_adapter_it_is_using` |
| 2 | `AdapterInfo` with `is_hardware()` — rejects software rasterizers | `the_selected_adapter_is_real_hardware_not_a_software_rasterizer` |
| 3 | `GpuError` — `NoAdapter` / `NoDevice` / `ShapeMismatch` / `Readback` | `mismatched_shapes_are_rejected_rather_than_dispatched` |
| 4 | Graceful no-GPU degradation — returns `Err`, never panics | tests skip with printed reason |

### Kernels (WGSL)

| # | Kernel | Evidence |
|--:|---|---|
| 5 | `matmul` — naive, one invocation per output element | `gpu_matmul_matches_the_cpu_reference` |
| 6 | `matmul_tiled` — 16×16 workgroup-memory tiles | `the_tiled_kernel_matches_the_cpu_reference` |
| 7 | `pairwise_sqdist` — the O(n²d) term under Vietoris–Rips | `the_distance_matrix_matches_the_cpu_reference` |
| 8 | `add_broadcast_row` — bias broadcast across rows | `bias_is_broadcast_across_rows_not_down_columns` |
| 9 | `relu` | `relu_clamps_negatives_and_leaves_positives_alone` |
| 10 | `relu_backward` — zero at exactly zero, matching forward | `relu_backward_is_zero_at_exactly_zero` |
| 11 | `transpose` | used by every backward step in `train_resident` |
| 12 | `column_sums` — bias gradient | same |
| 13 | `sgd_update` — parameter update on device | same |
| 14 | `sigmoid` — branch form, overflow-safe past ±88 | `train_resident` inference path |
| 15 | `sigmoid_bce_grad` — fused, avoids `p*(1-p)` underflow | same |

### Resident tensors

| # | Capability | Evidence |
|--:|---|---|
| 16 | `GpuTensor` — buffer persists across operations | `a_resident_chain_equals_the_same_chain_with_readbacks` |
| 17 | `upload` / `read` — the only bus crossings | same |
| 18 | `matmul_resident` | `the_tiled_kernel_matches_the_cpu_reference` |
| 19 | `pairwise_sqdist_resident` | `the_distance_matrix_is_symmetric_with_a_zero_diagonal` |
| 20 | `add_bias_resident` | `train_resident` |
| 21 | `relu_resident` | same |
| 22 | `relu_backward_resident` | same |
| 23 | `transpose_resident` | same |
| 24 | `column_sums_resident` | same |
| 25 | `sgd_update_resident` | same |
| 26 | `sigmoid_resident` | same |
| 27 | `sigmoid_bce_grad_resident` | same |

### CPU references

| # | Capability | Purpose |
|--:|---|---|
| 28 | `cpu_matmul` | parity baseline, naive on purpose |
| 29 | `cpu_pairwise_sqdist` | parity baseline |

### Executables

| # | Binary | What it produces |
|--:|---|---|
| 30 | `train_mlp_cv` | round-tripping baseline, 7.76 s |
| 31 | `gpu_bench` | naive vs tiled vs resident, separates kernel from bus |
| 32 | `train_resident` | fully-resident training, 0.48 s |

## Measurements

Fully-resident MLP, two spirals, 5-fold CV, 100 epochs/fold:

```
fold              1      2      3      4      5
test accuracy  0.7900 0.8100 0.8700 0.8100 0.8300

CV test accuracy    0.8220 +/- 0.0271
majority-class      0.5000   <- control
separation          +0.3220
wall clock          0.48 s   (round-trip path: 7.76 s, 16.03x)
```

Accuracy is **identical** between the round-tripping and resident paths to four
decimals on every fold. Same seeds, same data, different execution path, same
answer — which is the parity evidence for the port.

Chain of 8 matmuls, 256×256, median of 10:

```
round-trip every step    17.763 ms
resident, one readback    2.811 ms   6.32x
```

Square matmul, median of 20:

| size | CPU ms | rt-naive | rt-tiled | resident |
|---:|---:|---:|---:|---:|
| 64 | 0.179 | 0.817 | 0.771 | 0.756 |
| 128 | 1.024 | 0.904 | 0.926 | 0.766 |
| 256 | 8.523 | 1.220 | 0.778 | 0.746 |
| 512 | 71.191 | 3.286 | 2.496 | 1.929 |
| 1024 | 3714.298 | 20.234 | 15.401 | 11.031 |

## Multi-class

Softmax and categorical cross-entropy, with reporting that can detect the
failure accuracy hides: a three-class model that abandons one class and splits
the rest still posts a plausible accuracy. The confusion matrix and per-class
recall are what show whether it did.

Three interleaved spirals, 900 points, stratified 5-fold CV, 100 epochs/fold,
network 2 → 64 → 64 → 3:

```
confusion matrix (rows = true, cols = predicted)
             pred 0  pred 1  pred 2
  true 0        251       9      40
  true 1         67     218      15
  true 2         22      21     257

class     precision   recall       F1
0            0.7382   0.8367   0.7844
1            0.8790   0.7267   0.7956
2            0.8237   0.8567   0.8399

CV accuracy       0.8067 +/- 0.0379
macro F1          0.8066
majority-class    0.3333   <- control
separation        +0.4733
wall clock        0.68 s
```

Macro F1 tracking accuracy to four decimals is the evidence that no class was
abandoned. Class 1 is the weakest at 0.7267 recall, losing 67 points to class 0.

Folds are stratified rather than contiguous slices of a shuffled array. On three
classes the balance drift from contiguous slicing moves accuracy by more than
the effects being measured.

The data is generated, not observed. It is used because three interleaved
spirals are not linearly separable, so the majority-class control is a real
floor rather than a formality — but nothing measured on it transfers to real
data, and this row is not evidence about real datasets.

## Correction: the split investigation overstated its conclusion

> **Read the section after this one first.** This correction precedes the
> investigation it corrects, because both are in the order they happened rather
> than the order they make sense in. The short version: the finding below was
> wrong, and the reason it was wrong is more useful than the finding was.

The section below concluded that every accuracy figure in this crate was
measuring interpolation between near-duplicates, on the evidence that held-out
points sat 0.99× the cloud's own spacing from their nearest training point.

**That inference was wrong.** Two genuinely independent draws from the same
generator measure **1.39×** — dense i.i.d. sampling naturally places points near
each other, and a close training neighbour is what "same distribution" means.
The ratio cannot separate a leaky split from honest i.i.d. data, so 0.99× was
never evidence of a leak.

Checked directly, by re-running both examples against independently drawn test
sets:

| | interleaved CV | independent draws |
|---|---:|---:|
| Adam, `train_optimizers` | 1.0000 | 1.0000 |
| SGD, `train_multiclass` | 0.8067 | 0.8020 ± 0.0337 |

**They agree.** The interleaved split was not inflating anything on this data,
and the numbers it produced were about right.

What survives:

- The **blocked** split really is broken, and the ratio detects it: 9.72×, with
  both optimisers falling below the majority-class control. It measures
  extrapolation into arcs never seen.
- Independent draws remain the correct construction, on the structural argument
  that a partition of one deterministic sweep is not a sample. That argument
  never depended on the ratio, which is why it survives the ratio turning out
  not to mean what it was taken to mean.
- `SplitDiagnostic::is_separated` has been replaced by `is_extrapolating`, and
  `report_split` no longer issues a leak verdict it cannot support. A unit test
  pins the correction: a low ratio does not distinguish the two cases.

The original section is kept below rather than rewritten, because the reasoning
that produced a wrong conclusion from a real measurement is the part worth
keeping.

## The accuracy numbers above were measuring the wrong thing

Adam reached a cross-validated **1.0000 with standard deviation 0.0000** across
all five folds. A perfect held-out score is the most suspicious result a
harness can produce, so it was checked rather than reported.

```
median distance, held-out point to nearest training point   0.01425
median nearest-neighbour distance in the full cloud         0.01439
ratio                                                       0.99x
```

**The split separates nothing.** The spirals are generated by walking an arc in
order, and round-robin stratification deals consecutive points into different
folds — so every held-out point has a training point essentially on top of it.
The number measures interpolation between near-duplicates.

This applies to every accuracy figure this crate has reported: 0.8220 for
`train_resident`, 0.8067 for `train_multiclass`, and the 1.0000 above. The
kernels are unaffected — gradcheck and the mutation matrix are what verify
those — but the accuracy figures were not measuring generalisation.

### Blocking the split is not the fix

Holding out contiguous arc segments instead sends both optimisers to **at or
below the majority-class control**:

| | SGD | Adam |
|---|---:|---:|
| interleaved CV | 0.8544 | 1.0000 |
| blocked CV | 0.1522 | 0.3578 |
| **independent-sample holdout** | **0.8178** | **1.0000** |
| majority-class control | 0.3333 | 0.3333 |

Blocking asks the model to extrapolate into arc regions it never saw, which on
interleaved spirals is not learnable from the other arcs. One split is trivially
easy and the other is impossible.

Neither is fixable by choosing a better partition. The data is a
one-dimensional manifold swept in order, and **no partition of a single ordered
sweep is an i.i.d. split**. Drawing the test set independently from the same
generator is, and that row is the one to read.

The reconciliation: Adam's 1.0000 was the right answer for the wrong reason.
The task is genuinely learnable and Adam genuinely solves it — the independent
holdout confirms it — but the cross-validated evidence for that was worthless,
and would have stayed worthless if the number had been less surprising.

## Mutation testing

A suite nobody has mutated is a suite of unknown strength. Two defects injected
into `shaders.wgsl`, one at a time, rebuilt, and run against both suites.

| Injected defect | `gradcheck` | `gpu_parity` |
|---|---|---|
| `sigmoid_bce_grad` drops the `1/batch` scaling | **caught** — `dw1[0]` relative error 4.0 | survives |
| `relu_backward` boundary `> 0.0` becomes `>= 0.0` | survives | **caught** — `relu_backward_is_zero_at_exactly_zero` |
| `softmax_xent_grad` sign flipped, `(y - p)` for `(p - y)` | **caught** — `db2[0]` relative error 2.0 | survives |

A sign flip produces a relative error of exactly 2.0 and a dropped `1/batch`
produces exactly `batch - 1` at batch 5. Both are the signature of the defect
rather than a generic mismatch, which is worth reading off the failure message:
a relative error near 2 means the direction is wrong, near `batch - 1` means the
scale is.

### The remaining kernels

Seven more defects, one per previously-unmutated kernel. Reproduce with:

```bash
./crates/aether-gpu/mutants.sh
```

The harness patches the shader, forces the modification time forward, runs each
suite separately, restores from git, and exits with the number of defects that
escaped everything — so it is usable as a gate rather than only as a report.

It refuses to run without real GPU hardware. Every kernel test skips when no
adapter is present, and a skip is a pass, so a GPU-less run would report all
seven mutants surviving and announce a total coverage failure that actually
describes the machine. That is the same mistake the suites themselves guard
against, one level up.

| Injected defect | `gpu_parity` | `gradcheck` |
|---|---|---|
| `transpose` output index swapped | survives | **caught** |
| `column_sums` skips the first row | survives | **caught** |
| `matmul` reads A transposed | **caught** | survives |
| `matmul_tiled` second barrier removed | **caught** | survives |
| `pairwise_sqdist` distance not squared | **caught** | survives |
| `sgd_update` ascends instead of descending | **caught** | survives |
| `softmax_rows` max subtraction removed | **caught** | survives |
| `relu_backward` gates on the gradient, not the pre-activation | **caught** | **caught** |
| `adam_update` bias correction dropped | **caught** | survives |
| `adam_update` epsilon inside the square root | **caught** | survives |

The epsilon-placement mutant escaped every suite on its first run. At ordinary
gradient magnitudes `sqrt(vhat) + eps` and `sqrt(vhat + eps)` differ by about
5e-5 relative, inside any tolerance the other Adam tests can justify. It is only
visible where `vhat` is comparable to epsilon: at a gradient of 1e-6 the second
form gives a denominator a hundred times larger and a step a hundred times
smaller. That is the general shape of an epsilon bug — invisible in the regime
the code normally runs in, decisive in the regime epsilon exists for — so the
test that catches it uses a deliberately tiny gradient.

### The harness deleted uncommitted work

The harness restores by `git checkout`, which discards uncommitted changes to
the shader. Running it while the Adam kernels were written but not committed
deleted them, and the output never said so: it reported that every Adam pattern
failed to match, which reads as a harness bug, then that the clean tree failed,
because the Rust side still referenced pipelines the shader no longer defined.

It now refuses to run against a dirty shader. A destructive restore is fine; a
destructive restore that reports the damage as a pattern-matching problem is
not.

**Every defect is now caught by at least one suite. Two were not, before this
run added tests for them.**

**`sgd_update` ascending escaped everything.** No test asserted the direction of
a parameter update, and both gradient checks stop at the gradients without ever
applying one. The optimizer could have been performing gradient ascent and
thirty tests would have reported success; only the training examples would have
diverged, and nothing under `cargo test` runs those. Five tests now cover it:
the exact arithmetic, a direction property over 64 random pairs, `lr = 0` as a
no-op, linear scaling in the rate, and length mismatch rejection.

**`matmul_tiled` without its second barrier escaped at 64x64x64 and fails at
128x512x128.** The size is the whole finding. At 32 tile iterations across 64
concurrent workgroups the race surfaces; below that the suite reported a clean
pass on a kernel with a data race in it. A suite whose largest case is small
does not test less, it reports the wrong answer. Catching it at all is a
property of this adapter's scheduling — a missing barrier stays undefined
behaviour whether or not a given GPU exposes it.

**Neither suite alone is sufficient**, and the reason is structural rather than
accidental. Gradcheck samples random parameters, so a pre-activation never lands
exactly on zero and the ReLU kink is never probed; the boundary case has to be
asserted directly. Conversely the boundary test uses three hand-picked values
and cannot detect a scaling error that multiplies every gradient uniformly.

### A methodology trap worth recording

The first attempt at the second mutant reported "survives" in both suites, and
then a test failed *after the mutant was reverted* — which is not a coherent
result and was the signal that the measurement, not the code, was wrong.

Cause: the revert was `Copy-Item` from a backup taken before the mutation, which
restores the backup's original modification time. That timestamp was older than
the compiled artifact, so Cargo saw no reason to rebuild and the test binary
still contained the mutant. The shader is pulled in with `include_str!`, so the
staleness was invisible in `git diff`, which reported the working tree clean.

Any mutation run on a file consumed through `include_str!` has to force the
modification time forward, or verify the mutant is present in the built
artifact rather than only in the source.

## Submission batching, and the regression it caused first

Every resident operation used to build its own encoder and submit it, so a
training step cost one queue submission per operation. Work now accumulates
into a single encoder, submitted at `flush()` or at the next `read()`.

Batching alone made things **worse**:

| variant | wall clock, 5 folds x 100 epochs |
|---|---:|
| one submission per operation | 0.48 s |
| batched, flush once per fold | **0.65 s** |
| batched, flush once per step | **0.27 s** |

Flushing per fold records the entire run into one encoder. The GPU idles while
the CPU builds roughly two thousand dispatches, every intermediate buffer from
every epoch stays alive until the end, and the CPU/GPU overlap that the
unbatched version got for free is gone. The win only appears with a flush
boundary at the training step: batch within a step, submit per step.

Recorded because the intuition "fewer submissions is faster" is what produced
the 0.65 s version, and the measurement is the only thing that contradicted it.

Final: **0.27 s against the 7.76 s round-tripping baseline, 28.55x**, with the
CV accuracy unchanged at 0.8220 +/- 0.0271.

## The rule: arithmetic per byte returned

Every performance result in this file follows from one ratio, and it is worth
stating before the individual measurements because it predicts them.

With a CPU consumer, an operation pays on the GPU according to how much
arithmetic it does per byte that has to come back:

| operation | work | bytes returned | ratio |
|---|---|---|---|
| matmul | O(n³) | O(n²) | **grows as n** |
| pairwise distance | O(n²d) | O(n²) | **fixed at d** |

So matmul must cross over at some size and stay crossed, while pairwise distance
at d=3 must approach a constant and — if that constant is below one — never
cross, no matter how good the kernel is. Both halves measured, round-trip GPU
against the CPU reference:

| n | matmul CPU | matmul GPU | ratio | dist CPU | dist GPU | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.014 | 1.907 | 0.01× | 0.001 | 1.869 | 0.00× |
| 64 | 0.105 | 1.893 | 0.06× | 0.005 | 1.960 | 0.00× |
| 96 | 0.351 | 1.905 | 0.18× | 0.010 | 2.135 | 0.00× |
| 128 | 0.992 | 1.948 | 0.51× | 0.018 | 1.898 | 0.01× |
| 192 | 3.537 | 2.128 | **1.66×** | 0.040 | 2.043 | 0.02× |
| 256 | 10.068 | 2.231 | **4.51×** | 0.070 | 2.062 | 0.03× |

Matmul crosses between n=128 and n=192 and keeps climbing — 21.85× at 1024 with
residency. Distance climbs far more slowly, reaching only 0.81× at n=1024 and
0.74× at 2048 in the table above, and never crosses.

The GPU column is nearly flat at about 2 ms for both operations at these sizes,
which is the same statement from the other direction: that is fixed overhead,
and the only question is whether the CPU's work grows fast enough to exceed it.

**This is why the two integration recommendations differ**, and neither is a
judgement about kernel quality:

- `Tensor::matmul` is worth routing to the GPU **above n=128**, and the resident
  API makes it much better than that when calls chain.
- `pairwise_sqdist` is not worth routing at any size, because the persistence
  reduction is CPU-side and sequential, so the matrix must come back. Fixing
  that needs the *consumer* on the GPU, not a better kernel.

### Checked against the real code, with conversion counted

The crossover above is measured against this crate's naive f32 reference, which
is a stand-in. The code an integration would replace is
`aether_core::ml::Tensor::matmul`: f64, strided indexing, a `RefCell` borrow per
access. And `Tensor` being f64 while WGSL is f32 means a real call converts both
operands down and the result back up — three O(n²) passes that no measurement so
far included.

| n | `Tensor::matmul` | GPU raw | GPU + conversion | ratio |
|---:|---:|---:|---:|---:|
| 64 | 0.168 | 0.695 | 0.650 | 0.26× |
| **128** | 2.012 | 0.997 | 0.827 | **2.43×** |
| 192 | 6.869 | 0.828 | 0.830 | 8.27× |
| 256 | 18.028 | 1.130 | 1.300 | 13.86× |
| 384 | 61.803 | 2.368 | 3.057 | 20.22× |
| 512 | 158.570 | 3.280 | 4.153 | **38.18×** |

The n≈150 estimate was conservative in the right direction: the real crossover
is **n=128 with conversion included**, and the advantage at 512 is 38× rather
than the 4.5× the stand-in suggested at 256.

The reason is that `Tensor::matmul` is substantially slower than the stand-in —
18.0 ms against 10.1 ms at n=256 — because it carries f64, stride arithmetic and
interior-mutability borrows the reference does not. Conversion costs real time
but not decisive time: about 0.87 ms at n=512, roughly a quarter of the raw
figure.

At n=128 and n=192 the conversion column reads at or below the raw column, which
is impossible and is measurement noise at the sub-millisecond scale these
medians sit at. It is left as measured rather than smoothed, since the honest
reading is that the two are indistinguishable there, not that conversion is
free.

### And what f32 costs the answer

Cost was measured before acceptability, which is backwards, so: routing `Tensor`
through an f32 kernel changes every consumer's numbers, and the size of that
change decides who can use it.

| n | worst relative error |
|---:|---:|
| 16 | 1.222e-7 |
| 64 | 2.287e-7 |
| 256 | 6.735e-7 |

The growth is √n, as f32 accumulation over a k-term dot product predicts — 1.87×
from n=16 to 64 against a predicted 2×. That is f32 behaving like f32, which is
the point of asserting the *growth rate* rather than a single tolerance: a
linear or quadratic blow-up would mean a defect, and a flat line would mean the
comparison was not measuring anything.

At roughly **5e-7 relative**, an f32 `Tensor` path is:

- **fine** for neural network training — gradients are noisier than this by
  orders of magnitude, and this crate already trains to accuracy identical to
  the f64 CPU path, fold for fold;
- **fine** for clustering and classification, which threshold and argmax;
- **not fine** for anything asserting at 1e-9 or tighter, which includes the
  persistence engine's own invariant suite.

So the recommendation is **per consumer, not per crate**. Both bounds are
asserted in `f32_matmul_precision_is_stated_as_a_number_not_an_adjective`, so
the test fails if the kernel becomes either much worse or much better than the
recommendation assumes — a silent improvement would invalidate the "unsuitable
for 1e-9" half just as surely as a regression invalidates the other.

The precision argument for distances does **not** transfer here. Different
operation, different error growth, and carrying a conclusion across would be the
exact move these tests exist to prevent.

## Why the distance kernel loses: it is the bus, not the kernel

The kernel measures 0.52× the CPU reference at n=512, which reads as a slow
kernel. It is not. Compute is isolated by timing one dispatch against ten — the
transfers are identical in both, so the difference over nine is the marginal
cost of a dispatch:

| n | total ms | compute ms | transfer ms | transfer % |
|---:|---:|---:|---:|---:|
| 256 | 2.512 | −0.017 | 2.529 | 100.7% |
| 512 | 2.703 | 0.131 | 2.572 | 95.2% |
| 1024 | 3.442 | 0.335 | 3.107 | 90.3% |
| 2048 | 8.395 | 0.651 | 7.744 | 92.2% |

**90–100% of the time is transfer.** At n=512 the kernel computes in 0.131 ms
against the CPU reference's 1.192 ms — roughly **9× faster** — and loses overall
because moving the answer back costs 2.5 ms.

The n=256 compute figure is negative, which is the honest reading of a
measurement below its own noise floor rather than a number to round up to zero.

This is an architectural limit, not a kernel problem, and no amount of tiling or
occupancy work touches it. The persistence reduction runs on the CPU and is
inherently sequential, so an n×n matrix has to cross the bus. GPU distances can
only pay when the *consumer* is also on the GPU.

So the conclusion from the precision work stands and now has a cause: routing
`aether-core` through this kernel does not pay at the sizes the engine admits,
and would not pay at larger ones either until the reduction moves too.

## A crash, now reproducible at 1 in 5

`STATUS_ACCESS_VIOLATION` (`0xc0000005`) at process exit. First seen once in
`train_resident` after a rebuild, then not again in twelve runs, and left
recorded as undiagnosed rather than fixed.

It has reappeared in `gpu_bench`, and there it reproduces: **1 of 5 runs**. All
output completes first, so the fault is at teardown, not during work.

### The bisect, and what it ruled out

`gpu_bench` takes about fifteen seconds a run, which makes thirty-run statistics
slow, so `examples/teardown_repro.rs` strips it to four candidate patterns at
2048×2048 (16 MB per buffer), 30 runs each:

| pattern | dispatches | results read | crashes |
|---|---|---|---:|
| `drop` | 100 | none | 2 / 30 |
| `read` | 100 | all | 1 / 30 |
| `flush` | 100 | none, but flushed | 0 / 30 |
| `alloc` | **0** | — | **6 / 30** |

**The suspected cause was wrong.** `alloc` has the highest rate and issues no
dispatches at all — it only uploads large buffers and drops them. Whatever the
fault is, it is not recorded-but-unread work referencing freed buffers, which
was the hypothesis the previous entry recorded.

The rate differences between the first three patterns are **not** statistically
resolved at 30 runs — 2/30 against 0/30 is nowhere near significant, and reading
a ranking into them would be over-interpreting. Only the qualitative result is
solid, and it does not depend on the rates: a pattern with zero dispatches
crashes, so dispatch handling is not the mechanism.

### It is the multi-backend instance

The next two questions were backend and size dependence, both unmeasured.
Answering them needed `WGPU_BACKEND` support in `GpuContext` first:
`InstanceDescriptor::default()` does not read that variable, so setting it
without this change measures whatever backend would have been chosen anyway.

| backend | n=512 | n=1024 | n=2048 |
|---|---:|---:|---:|
| Vulkan pinned | 0/30 | 0/30 | 0/30 |
| DX12 pinned | 0/30 | 0/30 | 0/30 |

Zero everywhere — including the cell that had produced 6/30. A crash that
disappears as soon as it is measured is not a result, so the original condition
was re-run as a control: same binary, `WGPU_BACKEND` unset, n=2048, **2/30**.

Pooling every observation by whether a backend was pinned:

| instance | crashes |
|---|---:|
| all backends (`Backends::default()`, no env var) | **8 / 60** |
| a single backend pinned, either one | **0 / 180** |

8/60 against 0/180 is significant where the individual cells were not. The fault
tracks **instantiating every backend at once**. Not buffer size — nothing at
512, 1024 or 2048 with a backend pinned. Not allocation volume. Not the choice
between Vulkan and DX12, since either alone is clean.

### Fixed by construction

The workaround was found and then left unapplied for one commit, which is the
wrong place to stop: a known-crashing default is not made acceptable by being
documented. `GpuContext::new` now instantiates exactly one backend, trying
Vulkan, DX12, Metal and GL in order and stopping at the first that yields an
adapter. `WGPU_BACKEND` still overrides.

| configuration | crashes |
|---|---:|
| before, all backends | 8 / 60 |
| after, single backend by default | **0 / 60** |

The underlying mechanism is still in wgpu's multi-backend instance teardown,
below anything this crate controls, so `teardown_repro` remains the reproduction
to attach to an upstream report. What changed is that reaching it now takes
deliberately setting `WGPU_BACKEND` to a multi-backend value, rather than using
the crate as written.

The `Drop` impl earns its place regardless of this: without it, work recorded
and never flushed is discarded silently, so a caller that updates parameters and
never reads them back loses the update with no error.

## Is f32 good enough for the topology?

The question standing between this crate and the thing the repository is about.
`pairwise_sqdist` exists because the O(n²d) distance computation dominates every
Vietoris–Rips filtration in `aether-core`, but the engine computes in f64 and
WGSL has no f64. Routing distances through the GPU means computing the
filtration from f32 values.

That is a topological question, and the engine can be asked it directly.
Coordinates are rounded through f32 and back — exactly what a GPU path does to
its inputs — and the two diagrams compared. The stability theorem the engine's
own invariant suite already asserts gives the bound in advance:

$$d_B(\mathrm{Dgm}(X), \mathrm{Dgm}(X')) \le 2\varepsilon, \quad \varepsilon = \max \lVert x - x' \rVert$$

| fixture | ε | bottleneck | bound 2ε |
|---|---:|---:|---:|
| random cloud, seed 1 | 2.318e-8 | 2.081e-8 | 4.636e-8 |
| random cloud, seed 2 | 2.000e-8 | **2.618e-8** | 3.999e-8 |
| random cloud, seed 3 | 1.951e-8 | **2.225e-8** | 3.902e-8 |
| random cloud, seed 5 | 1.809e-8 | **2.570e-8** | 3.618e-8 |
| random cloud, seed 8 | 1.948e-8 | **2.062e-8** | 3.896e-8 |
| circle n=12, H₁ | 1.554e-8 | 7.772e-9 | 3.109e-8 |
| circle n=18, H₁ | 2.617e-8 | 1.342e-8 | 5.234e-8 |
| circle n=24, H₁ | 1.711e-8 | **2.126e-8** | 3.423e-8 |

**Betti numbers are identical** across 39 filtration radii × 6 seeds, and the
count of long H₁ bars on a circle is unchanged at every n tested.

So f32 is viable for this engine. The bar endpoints move by about 2e-8, and the
discrete invariants — which is what the language's seal loop actually terminates
on — do not move at all. A convergence rule comparing integers cannot see a
1e-8 shift in a birth time.

The bolded rows are worth noting: in five of eight fixtures the bottleneck
distance **exceeds ε** while staying under 2ε. The factor of 2 in the theorem is
load-bearing here, not slack — a test written against ε rather than 2ε would
fail on real data and look like a precision bug.

### The accumulation half, which is worse

Everything above rounds the *coordinates* and then computes in f64. A real
kernel also rounds every subtraction, square and partial sum, and that error is
not bounded by the coordinate displacement. Running the actual kernel:

```
kernel vs f32 CPU reference   2.384e-7 (squared units)
kernel vs exact f64           3.671e-7 relative
```

A Vietoris–Rips filtration is determined by the **order** simplices enter, not
their values. If every distance moves by less than half the smallest gap between
two distinct distances, no pair can swap and the combinatorics are identical by
construction. Measured:

| n | distance error | smallest distinct gap | ratio | ordering |
|---:|---:|---:|---:|---|
| 16 | 5.013e-8 | 3.014e-6 | 0.017 | guaranteed |
| 32 | 6.690e-8 | 3.857e-6 | 0.017 | guaranteed |
| 64 | 1.099e-7 | 9.380e-9 | **11.7** | **not guaranteed** |
| 96 | 8.323e-8 | 7.318e-8 | **1.14** | **not guaranteed** |

**The guarantee fails at n=64** — an eighth of the 512 points `h0_only` admits.
The kernel's error is roughly constant at f32 epsilon, but n points carry
n(n−1)/2 distances inside a bounded range, so the smallest gap shrinks
quadratically. The two curves cross between 32 and 64 and never come back.

So the iteration-14 conclusion needs narrowing. An f32 distance path **cannot
promise identical combinatorics at the sizes this engine is configured for**.
What survives is the weaker bound: filtration values move by at most the
distance error, so bars shift by about 1e-7. Whether a bar appears or disappears
is no longer settled by construction, and would have to be measured per cloud.

### So it was measured

"Not guaranteed" is a worst-case statement: two distances *can* swap. It says
nothing about whether a swap changes homology, and most do not — exchanging the
entry order of two simplices that are not both pivotal leaves the barcode alone
apart from endpoint shifts.

Points displaced by the kernel's own error magnitude, Betti numbers compared
across the full filtration:

| n | homology | clouds changed |
|---:|---|---|
| 32 | H₁ | 0 / 50 |
| 64 | H₁ | 0 / 50 |
| 128 | H₀ | 0 / 50 |
| 256 | H₀ | 0 / 30 |

**0 of 180.** By the rule of three that puts the 95% upper bound on the rate at
1.7% — not "never", but below that.

Sweeping the displacement upward to find where it does break, n=64 H₁:

| displacement | clouds changed |
|---|---|
| 1e-6 | 0 / 20 |
| 1e-5 | 0 / 20 |
| **1e-4** | **4 / 20** |
| 1e-3 | 19 / 20 |
| 1e-2 | 20 / 20 |

**Largest clean displacement 1e-5 against a kernel error of 4e-7: a 25× margin.**

The worst-case and average-case answers genuinely differ and both are true. The
ordering guarantee is unavailable past n≈32, and the topology is nonetheless
unchanged in every cloud tested, with two orders of magnitude of room. An f32
distance path is viable here in practice; it just cannot be *proved* viable by
the ordering argument, and a caller who needs a guarantee rather than evidence
does not have one.

### End to end, through the engine

Every result above was a proxy. The coordinate test rounded inputs and computed
in f64; the perturbation study displaced points to *model* the kernel's error.
Neither ran a diagram through actual kernel output, because the engine took
points rather than a distance matrix.

`aether_core::persistence::persistent_homology_from_distances` closes that. It
builds a Rips filtration from a supplied `[n, n]` matrix, sharing the simplex
enumeration with the point-based path rather than duplicating it, and validates
that the matrix is square, symmetric, zero-diagonal, non-negative and finite —
because an asymmetric matrix still produces a barcode, silently, and that
barcode answers no question. The triangle inequality is deliberately not
checked: Rips is defined for any symmetric non-negative dissimilarity.

Diagrams from real `pairwise_sqdist` output against exact f64 distances:

| n | homology | distance error | bottleneck | Betti |
|---:|---|---:|---:|---|
| 32 | H₁ | 9.021e-8 | 2.463e-8 | identical |
| 64 | H₁ | 9.435e-8 | 3.389e-8 | identical |
| 128 | H₀ | 1.127e-7 | 2.384e-8 | identical |

The kernel output goes in unmodified — square root, then straight to the engine.
An earlier version averaged `d(i,j)` with `d(j,i)` on the stated assumption that
f32 rounding made them differ in the last bit. **That was wrong.** IEEE-754
subtraction is exactly antisymmetric, so `a − b` and `b − a` differ only in sign
bit, their squares are bitwise identical, and the kernel accumulates both orders
over the same range. Symmetry is a property of the arithmetic, not an
approximation.

The averaging was harmless and still worth deleting: it hid the property rather
than relying on it, and would have masked a genuine indexing bug. The symmetry
test now asserts bitwise equality rather than agreement to 1e-5 — a tolerance
that loose would have accepted a transposition producing slightly different
entries.

The bottleneck distance comes in *below* the distance error every time, which is
what the bound predicts: perturbing every filtration value by at most δ moves
the diagram by at most δ.

The matrix path is also asserted to reproduce the point path bar for bar on
identical input, to 1e-15. An entry point that generalises another one owes that
first — otherwise every result obtained through it describes a different engine.

None of which makes the kernel worth using: it still measures 0.52× the CPU
reference at n=512. The precision objection is now answered end to end, at every
level from coordinates to barcode. The performance one is untouched.

## Negative results

**The distance kernel does not currently help the persistence engine.**

| n | CPU ms | GPU ms | ratio |
|---:|---:|---:|---:|
| 256 | 0.240 | 2.041 | 0.12x |
| 512 | 1.216 | 2.319 | 0.52x |
| 1024 | 4.564 | 4.679 | 0.98x |
| 2048 | 29.431 | 22.516 | 1.31x |

`PersistenceConfig`'s widest preset, `h0_only`, caps at 512 points, where the
GPU path runs at **half** the speed of the CPU reference. Break-even is near
n=1024. The kernel is correct and kept, but a caller must raise the point caps
past 1024 before dispatching to it is worth the transfer, and nothing routes
through it today.

**Small matmuls lose.** At 64×64 the GPU is 4.2× slower than the CPU reference.
Dispatch and allocation dominate below roughly 128×128.

## Limits

- All timings are medians from a single run of one binary on one machine, no
  core pinning, no turbo control.
- CPU columns are the naive single-threaded reference this crate ships as its
  parity baseline, not a tuned BLAS. Ratios compare this crate's own two paths.
- f32 only. `aether-core` computes in f64 on the CPU path, and nothing in
  `aether-core` routes through this crate yet.
- Gradcheck runs a depth × head × size matrix against one generic reference.
  Six configurations, 7,732 gradient entries, worst relative error 1.985e-4:

  | configuration | stage 1 | stage 2 entries | worst rel. error |
  |---|---:|---:|---:|
  | 1-layer sigmoid 5×3×4 | 21 / 21 | 21 | 3.179e-7 |
  | 2-layer sigmoid 5×3×4×4 | 41 / 41 | 41 | 4.954e-7 |
  | 1-layer softmax 5×3×4×3 | 31 / 31 | 31 | 3.594e-5 |
  | 1-layer sigmoid 33×17×48 | 32 / 913 | 913 | 9.127e-5 |
  | 2-layer sigmoid 33×17×48×48 | 32 / 3,265 | 3,265 | 1.985e-4 |
  | 2-layer softmax 33×17×48×48×5 | 32 / 3,461 | 3,461 | 1.692e-4 |

  The tile-crossing dimensions are `2*16+1`, `16+1`, `3*16`, so every gradient
  matmul spans several of the kernel's 16-wide tiles and leaves a partial tail.
  Stage 1 sweeps every parameter on the small fixtures and samples on a fixed
  stride on the large ones, where a full sweep is affordable in release and not
  in debug.
- 0.8220 is what 100 full-batch epochs at lr=0.5 buys on this task. It is a
  fixed budget, not a tuned result, and not the architecture's ceiling.

## Next

Ordered by value, highest first.

1. **Gradcheck** against finite differences. The backward path is the least
   verified thing here.
2. **Batched dispatch** — one submission per training step instead of one per
   op, to cut the remaining per-dispatch overhead.
3. **Wire `aether_core::ml::Tensor::matmul`** to dispatch. Needs a decision:
   the core is f64 and the GPU is f32.
4. **Softmax and cross-entropy** for multi-class, unlocking real datasets.
5. **Adam** alongside SGD.
6. **f16 storage** with f32 accumulation.
7. Raise `PersistenceConfig` caps past 1024 so the distance kernel earns its
   dispatch, then route Vietoris–Rips through it.
