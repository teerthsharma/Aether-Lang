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

**The backend works and nothing uses it.** 20 WGSL kernels, resident tensors,
batched submission, 98 tests, 0 of 24 mutants escaping. No line of
`aether-core` or `aether-lang` calls it.

`scheduled_attention_resident` returns a `GpuTensor` so attention output can feed
another kernel without a round trip. It was the last kernel here without a
resident path, which meant every use downloaded a full `[seq, head_dim]` result
even when the next operation was a matmul on the device — the transfer pattern
this file already records as dominating `pairwise_sqdist`. The read-back call is
now that path plus a download rather than a second implementation, and a test
asserts the two agree bitwise so a later change cannot quietly give them separate
dispatches.

The newest kernel is the exception worth naming: `scheduled_attention` is the
GPU port of this repository's headline mechanism, which until now ran only on
the CPU while the GPU did generic MLP work. It is still not called from
`aether-core` — the sentence above holds — but it is the first kernel here that
implements the thing the project is about rather than a primitive it needs.

### The selector recovers less attention mass than random selection

The first same-budget ablation of the topological schedule, and it does not
support the mechanism. Recovered attention mass, per-row budget matched exactly,
`cargo run -p aether-gpu --example selector_ablation --release`:

| seq | density | random | topological | oracle | position |
|---:|---:|---:|---:|---:|---:|
| 64 | 72.2% | 0.8151 | 0.8475 | 0.9483 | 24.3% |
| 128 | 36.8% | 0.4976 | 0.5674 | 0.6799 | 38.3% |
| 256 | 20.8% | 0.3208 | 0.3711 | 0.4728 | 33.1% |
| 512 | 12.0% | 0.2281 | 0.2337 | 0.3272 | 5.6% |

Position is `(topological - random) / (oracle - random)`: the share of the
achievable gain the selector captures. It collapses as the sequence lengthens,
which is the regime sparse attention exists for.

Two explanations fitted, and they separate cleanly. Holding `seq` at 512 and
raising only `topk_topology_blocks` does not recover the position — it drives it
**negative**, to −109% at top-k 32, where the selector recovers 0.7446 against
random's 0.8643. More budget spent on topology makes the schedule worse. On iid
keys, the control for the drifting fixture rewarding locality, position is
negative at every allowance.

**The signal is real and its sign is reversed.** `block_salience` scores a block
by H0 death time under single-linkage merging, which measures how *isolated* it
is. Attention mass concentrates where a key resembles the query, and a block
unlike everything else is unlike the typical query too. Selecting the
*lowest*-salience blocks instead, at an identical budget:

| top-k | random | highest | lowest | oracle | lowest position |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.2281 | 0.2337 | 0.2547 | 0.3272 | 26.9% |
| 4 | 0.3089 | 0.2932 | 0.3521 | 0.4383 | 33.4% |
| 8 | 0.4404 | 0.3882 | 0.5053 | 0.6097 | 38.3% |
| 16 | 0.6337 | 0.5308 | 0.7177 | 0.8218 | 44.7% |

Inverted, the selector beats random by a margin that *grows* with budget, which
is what an informative signal looks like. `topology_block_schedule` is left
unchanged: flipping the ranking changes what the method is, and that is a
decision to take deliberately rather than as a side effect of an ablation.

Limits: two synthetic fixtures on one machine, `head_dim` 32 and `block_size` 8
throughout. The iid position percentages divide by a headroom of about 0.008 and
are unstable as ratios — the absolute deficit there is 0.0148, small but
consistently negative across every allowance. No trained model is involved; this
measures the schedule against the attention it approximates, not downstream task
performance, and a selector can in principle lose mass and still serve a model
well.

### The first place the backend falls back to the CPU

`GpuError::Unsupported` was split out of `ShapeMismatch` because the two ask
different things of a caller: mismatched shapes are a bug at the call site, while
a `head_dim` past the kernel's private scratch is a limit of this backend on a
launch `aether-core` computes perfectly well. That distinction was decorative
until something acted on it.

`scheduled_attention_or_cpu` and `scheduled_attention_backward_or_cpu` act on it. A supported launch runs on the GPU, one
past a ceiling runs on `aether_core::scheduled::scheduled_attention`, and a
malformed launch still fails — falling back on a caller's bug would turn it into
a silently slower correct-looking answer.

**It returns which path ran, and that is the point of the signature.** The two
routes do not agree to the same tolerance: WGSL has no f64, so the GPU answer is
f32 widened on the way out while the CPU answer is f64 throughout. A helper that
hid the switch would silently change a result's precision with the size of its
input, which surfaces much later as a number nobody can reproduce. `AttentionPath`
makes it something a caller can ignore deliberately but not by accident.

The test covers all three routes rather than the interesting one, and checks the
CPU route's *values* against the reference at 1e-12 — asserting only the path
would pass on a fallback that returned zeros.

Both halves of the operation route, and the test asserts they route the *same
way* rather than that each works. A fallback on the forward and none on the
backward is not a smaller version of having both: a caller whose forward survives
a ceiling and whose backward does not must implement the fallback anyway, and then
maintains two routing policies that have to agree. Asking both for the same launch
and requiring the same verdict is what makes that a contract instead of a
coincidence.

This is also the first thing in the crate that answers the standing sentence at
the top of this file. The backend still is not called from `aether-core`, and
cannot be: `aether-core` is `no_std` and `wgpu` is not. But a caller who wants the
GPU where it helps and correctness everywhere else now has one function per
direction to call instead of a policy to implement.

### A crashed test binary is not a caught mutant

The mutation harnesses classified any non-`ok` cargo run as evidence about the
mutant. A test binary that *dies* is a third outcome, and the guard added to
refuse unclassifiable output found it on its first real encounter: a
`STATUS_ACCESS_VIOLATION` during `attention_parity`, which produces no test
result and no compile error.

That is the intermittent this file already documents — a fault in wgpu's instance
teardown, worked around by pinning a single backend and never eliminated. Two
consecutive runs put it on different mutants in different suites
(`softmax scale dropped` in `attention_parity`, then `causal mask never fires` in
`gpu_parity`), which is what confirms it is environmental rather than caused by
the mutant it lands on.

A crash is therefore scored as **not caught**. Its two possible causes — the
mutant taking the process down, or the teardown fault — are indistinguishable
from the output, and crediting a catch would assign coverage to whichever it was.
Reporting less coverage than exists is the safe direction for a number whose
entire purpose is to say how much the suite would notice.

The consequence is visible and worth stating: when the crash lands on the only
suite that catches a mutant, that run reports an escape. `softmax scale dropped`
is caught 6/17 by `attention_parity` alone, so the run it crashed reported 1 of 20
escaping and the next reported 0. That is the harness saying *this cell could not
be measured*, which is different from both "caught" and "escaped" and is the only
honest thing it can say.

A crashed cell is now retried once before being scored. What identifies this
fault as environmental — that consecutive runs put it on different mutants in
different suites — is exactly what makes a single repeat clear it, so scoring the
first crash discards a cell the retry would have measured. Only crashes retry: a
test that *fails* is evidence, and re-running it to see whether it fails again is
sampling until the answer is convenient.

The retries are counted and reported, because this fault is worked around and not
fixed. A silent retry hides how often it fires, so a rising rate — a driver
update, a change that makes teardown more likely — would stay invisible until
cells began failing twice and the harness started reporting escapes that were not
there.

The first run under it reported `1 cell(s) crashed and were retried` alongside 0
of 20 escaping, which is the mechanism working on real data rather than on a
description of it: that same crash, one change earlier, produced a spurious
escape. **One crash in sixty cells is the current rate.** A number that climbs
between runs is the workaround wearing out.

Which cells crashed is recorded too, because the environmental explanation rests
on the crash moving and a count alone cannot distinguish that from a fault
settling on one kernel — which would mean the mutant causes it and the retry is
hiding a catch. Four crashes observed so far:

| mutant | suite |
|---|---|
| `matmul: reads A transposed` | `attention_parity` |
| `scheduled_attention: softmax scale dropped` | `attention_parity` |
| `scheduled_attention: causal mask never fires` | `gpu_parity` |
| `adam_update: bias correction dropped` | `attention_parity` |

Four different mutants, touching matmul, softmax scaling, the causal mask and
Adam's bias correction — no shared code between them, and the last has nothing to
do with attention at all while crashing in the attention suite. That is what the
environmental reading predicts and a mutant-caused one does not.

Three of the four land in `attention_parity`, which reads like concentration
until the mechanism is considered: it is the largest suite and creates the most
GPU contexts, and the fault is in context teardown. More teardowns, more chances.
That explains the skew without weakening the reading — but it is the number to
watch, because a fault that stopped scattering across *mutants* would be the
signal this table exists to provide.

### Reverse mode, CPU reference and GPU port

`recall_training` freezes attention and trains only a head, and records that as
its narrowest assumption: a model trained end to end could reshape its queries to
suit whatever schedule it was given, which would make the comparison a different
experiment. It was frozen because there was no backward pass.

`aether_core::scheduled::scheduled_attention_backward` is that pass, in f64, with
the schedule held fixed. Holding it fixed is not a simplification — a block is
selected or it is not, so the schedule has no useful derivative — and it means
the gradient can teach a model to use the blocks it was given better, never to
choose different ones.

Verified by central differences against the forward kernel, which is the only
reference that shares no assumption with the code under test. Worst disagreement
6.535e-12 on a dense schedule and 3.859e-12 on a sparse one, against a 1e-7
tolerance. Three structural tests sit alongside: keys the schedule excludes
receive exactly zero gradient rather than something small, a one-hot cotangent
recovers the forward's attention weights in `dv` and they sum to 1, and a
mismatched cotangent is rejected.

Five backward mutants are in `crates/aether-core/mutants.sh` — rank-one softmax
correction dropped, delta unweighted, `dq` accumulating `q` instead of `k`, `dv`
unweighted, scale omitted. All five die, and only against `attention_backward`;
the selection and mechanism suites report them surviving. **0 of 19 escape**
across the three suites.

Four WGSL kernels now port it. `attention_row_stats` computes the per-row
maximum, log-sum-exp and delta once; `attention_dq`, `attention_dk` and
`attention_dv` read them. Without that sharing, each (row, column) pair would
cost a full sweep of the row's scheduled blocks and the kernels would be
quadratic in the sequence where the reference is linear in the scheduled work.

**No atomics anywhere**, which is a constraint rather than an optimisation. `dq`
accumulates over the columns a query row sees, so one invocation per query row
owns its result; `dk` and `dv` accumulate over the query rows that see a column,
so those run one invocation per *key* row. Both directions keep accumulation
thread-local, which makes them deterministic for the same reason `matmul` is. The
shorter alternative — one thread per query row writing `dk` through atomics —
would make the result depend on scheduling order, and this file already records
that a non-deterministic kernel invalidates every A/B comparison made with it.

Checked against the f64 reference on dense and sparse schedules, with the three
gradients asserted separately: `dv` is linear in the values and never reads the
delta term, so a mistake there leaves it exactly right while corrupting the other
two, and a joint assertion would report one failure where the split reports which
half is broken. A third test pins that the kernel zeroes what the reference
zeroes — `dk` and `dv` walk the schedule in the opposite direction from every
other kernel here, testing block membership per query block, and a test that
answered yes too often would write gradient where there should be none.

### End-to-end training does not rescue a sparse schedule

`recall_training` freezes attention and records its narrowest assumption: a model
trained end to end could reshape its queries to suit whatever schedule it was
handed, so a frozen comparison might understate what a schedule is worth. With a
verified backward pass that assumption is testable, and `recall_end_to_end` tests
it by learning a query projection `Wq` through the attention kernel with keys,
values and schedule fixed. Learning only the queries is not a simplification of
the question — reshaping queries was the question.

| schedule | identity | trained | change | control: scrambled → trained |
|---|---:|---:|---:|---|
| dense | 51.0% | **86.0%** | +35.0% | 40.0% → 55.0% |
| topological | 56.0% | 61.0% | +5.0% | 48.0% → 54.0% |
| inverted | 52.0% | 56.0% | +4.0% | 46.0% → 55.0% |
| random | 53.0% | 59.0% | +6.0% | 50.0% → 53.0% |

**Training does not close the gap.** Dense reaches 86% while every sparse arm
plateaus between 56% and 61%, and the three remain indistinguishable from each
other. A model can learn a great deal from blocks it can see and cannot learn its
way to blocks it never sees. The frozen result stands, and its stated limit is
discharged rather than merely acknowledged.

The first run of this said the opposite by saying nothing: every arm moved by ±1
to 3% and the gap survived, which reads as the same conclusion. The positive
control refused it. Starting from a scrambled `Wq`, training recovered **42.0% to
42.0%** — not a small improvement, exactly none — so the null described an
optimiser that could not move rather than a schedule that could not be
compensated for.

The cause is in the design. `recall_training` plants the query at `MATCH = 30` so
the target dominates the softmax and the frozen features carry the label cleanly.
A softmax collapsed onto one column has a vanishing Jacobian: `ds = p (dp -
delta)` cancels when a single `p` approaches 1, so almost no gradient reaches
`Wq`. Sharp retrieval is what makes the task learnable and what makes it
untrainable, and the two requirements pull against each other. At `MATCH = 5` the
softmax is still informative and still differentiable, the control recovers in
every arm, and the numbers above mean something.

#### The f32 backward drives the same training outcome

The table above was produced with the f64 CPU backward. `--gpu-backward` runs the
identical experiment through the four WGSL kernels instead, and every one of the
sixteen figures is unchanged — all four arms, both columns, and all four control
pairs.

That is a check the parity tests cannot make. They pin the gradient against the
reference at a point; this accumulates f32 gradients across 60 epochs of 300
sequences, which is where a small per-step error would compound into a different
model. It does not.

What the agreement does *not* establish: accuracy on 100 held-out sequences moves
in 1% steps, so identical accuracy means the training *outcome* is the same, not
that the trajectories or the learned weights are. A weaker claim than bitwise
agreement and the one the measurement supports.

The GPU backward took 623 s against 216 s for the CPU one, on the same tree and
back to back. Unlike the ratios elsewhere in this file that figure is not fragile
— each run aggregates roughly 36,000 backward calls, so per-call variance averages
out.

Most of the gap was a round trip in the middle of the backward pass. The
statistics kernel wrote its output to its own tensor, that tensor was downloaded,
concatenated onto the operands and re-uploaded, once per call, on the reasoning
that the shared four-binding layout had no free binding to leave it in. The
premise was true and the conclusion did not follow: a `read_write` binding is
*readable*, so the kernel can take the packed operand buffer as its output, read
q, k, v and dOut out of it, and fill a reserved tail in place. The gradient
kernels then bind that same buffer as `a` and find everything already there. No
second layout, no concatenation kernel, and the `ponytail:` marker is gone.

**623 s to 493 s, with all sixteen figures unchanged.**

The remaining gap to the CPU's 216 s was attributed to "the per-call operand
upload and four dispatches". That sentence was written without measuring, and
`attention_cost` measures it. The two candidates predict different numbers — a
backward that is dispatch-bound costs about 4× a forward, one that is
transfer-bound about 1.3× — so one run separates them:

| seq | forward | backward | ratio | upload | upload share |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.894 ms | 1.620 ms | 1.81× | 0.002 ms | 0.1% |
| 128 | 1.098 ms | 2.535 ms | 2.31× | 0.003 ms | 0.1% |
| 256 | 1.416 ms | 3.587 ms | 2.53× | 0.006 ms | 0.2% |
| 512 | 2.098 ms | 6.287 ms | 3.00× | 0.018 ms | 0.3% |

**Half of that attribution was wrong, and it was the half named first.** The
upload is 0.1–0.3% of the call. Allocating and filling a buffer the size of the
backward's packed operands is free at these sizes, so buffer reuse — the obvious
next optimisation, and one with real aliasing hazards against the pending
encoder — would have bought 0.3% at the top end.

The dispatch count is the cost, and the ratio says so by climbing toward it: 1.81×
at seq 64 where fixed per-call overhead dilutes everything, 3.00× at 512 where
kernel work dominates. It stays under 4 because the four backward kernels are not
each as expensive as the forward.

What this leaves is that the four dispatches are what the API shape costs, which
is what the original sentence claimed — arrived at by measurement rather than by
naming two plausible causes and trusting the reasoning.

The mutation harness caught its own staleness on this change. Moving the write to
`c[s_base + row * 3u + 2u]` left one mutant's pattern matching nothing, and the
run reported `SKIPPED <- pattern did not match` and counted it as an escape. That
is the behaviour the script documents and the reason it counts that way: a stale
pattern is not coverage, and scoring it as a pass would have reported a mutant
that never ran as a mutant that died. Corrected, 0 of 20 escape.

#### f32 gradient error grows about linearly with the sequence

The agreement above was measured at 64 positions, which left open whether f32
holds at the 512 the ablation uses — every softmax there sums four times as many
terms, and both `attention_row_stats` and the `dk`/`dv` accumulations grow
linearly in the sequence.

Worst `|gpu - cpu|` across `dq`, `dk` and `dv`, dense schedule, `head_dim` 16:

| seq | worst error | ratio to previous |
|---:|---:|---:|
| 64 | 1.159e-7 | — |
| 128 | 1.635e-7 | 1.41× |
| 256 | 4.500e-7 | 2.75× |
| 512 | 1.251e-6 | 2.78× |

Eight times the sequence costs 10.8× the error, which is growth of roughly
`length^1.14` — essentially linear, and nowhere near the quadratic the test
permits. At 512 the disagreement is 160× below the tolerance the other parity
tests assert, so f32 is not the limiting factor anywhere this repository
currently measures, and on this trend would not become one until the sequence is
two orders of magnitude longer.

The test pins the growth rate rather than the magnitude. An absolute bound would
pass on a kernel whose error was about to explode one size later; the ratio
between consecutive rows is the quantity that says whether f32 remains viable at
lengths not yet run.

Limits: 400 sequences with 100 held out, so a 5-point difference is inside the
sampling error and only the 25-point dense gap is resolved. `Wq` is 8×8 with
plain gradient descent, no momentum and one learning rate for every arm. The
control establishes that training moves accuracy, not that it reaches any
optimum — a better optimiser might extract more from a sparse schedule than this
one does. One run per backend, not a distribution. The error table is one seed
per size and a dense schedule, which is the worst case for accumulation and the
best case for cancellation; a sparse schedule sums fewer terms and a different
seed would move the last digit.

### The function that makes the ablation fair had no test

The selection code in `aether-core` produces every number in the two sections
below, and none of those numbers is checked against an external reference — they
are computed. A mutation harness at `crates/aether-core/mutants.sh` measures
whether the eight contract tests could tell if the computation were wrong. Six of
seven mutants died. One survived:

```
schedule_budget: reports one block too many          survives  <- ESCAPED
```

`schedule_budget` reads the per-row block counts off the topological schedule,
and every baseline is built from its output. Inflating it hands random and oracle
selection blocks the topological schedule never spent, so the comparison stops
being same-budget while still being labelled one. The bias has a direction: the
baselines get stronger and the selector looks worse, which is the direction of
the conclusion already drawn here. A defect that pushes a result the way the
result already went is the one least likely to be questioned, and it was the one
with no coverage.

The shipped code is correct — the mutant is a deliberate injection, not a bug
found — so the tables below stand. What was missing was any test that would have
noticed otherwise. `the_reported_budget_is_what_the_schedule_spends` now checks
the reported counts against the schedule's own CSR rows, against the flat index
total, and against the closed form `q + 1` for the dense schedule.

The harness was then extended to `block_salience` and `topology_block_schedule`,
which is where the *explanation* lives rather than the measurement. This document
attributes the selector's deficit to salience ranking blocks by isolation, and
that reading assumes `block_salience` computes the H0 death times it claims. If
it did not, the anti-correlation would be an artefact and the explanation a story
told about a bug. Seven more mutants cover it — elder rule inverted, death
recorded for the surviving component, merges in decreasing distance, centroids
summed rather than averaged, squared distance left unrooted, local window
narrowed to the diagonal, sink blocks dropped. **0 of 14 escape.**

The two suites turn out to cover disjoint ground: every selection mutant is
caught only by `ablation_baselines`, every mechanism mutant only by
`scheduled_attention`, with no overlap in either direction.

Two of the mechanism mutants are rank-preserving and worth naming. Summing
centroids instead of averaging, and leaving the squared distance unrooted, both
scale every block identically, so the merge order and therefore every schedule
the selector produces are unchanged. `ablation_baselines` reports `survives` for
both, correctly — the ablation depends on the *ranking*, which those mutants do
not touch. They die only against the one test that checks salience magnitudes.
The distinction matters for reading this file: the ranking is what every result
here rests on, and it is covered from two independent directions.

### A trained model finds no advantage either, and the first run of it was noise

The mass measurement above compares a schedule against the attention it
approximates. It assumes losing mass costs a model something, which nothing had
checked — a selector could drop most of the mass and keep whatever a task needs.

`recall_training` closes that. Associative recall on 2,400 sequences: the final
query is planted onto a content-addressed earlier key, the label is the sign of a
component of the value stored there, and a schedule that misses the block holding
it leaves features carrying no information about the label. Attention and the
head both run on GPU with resident tensors and Adam; no gradient reaches the
schedule, so this measures retained signal rather than the model's ability to
compensate.

| schedule | density | mass | accuracy | fold range |
|---|---:|---:|---:|---|
| dense | 100.0% | 1.0000 | 92.4% | 90.8 – 94.4 |
| topological | 50.1% | 0.5916 | 60.5% | 56.9 – 62.5 |
| inverted | 51.9% | 0.6097 | 59.0% | 55.8 – 66.7 |
| random | 50.1% | 0.6110 | 61.0% | 59.6 – 62.7 |
| local+sink only | 33.1% | 0.4512 | 52.0% | 49.4 – 54.2 |

**At equal budget the three content-selecting schedules are indistinguishable.**
Topological, inverted and random all land near 60% with overlapping fold ranges,
against 92.4% for dense and a 50% chance floor. Whatever the salience ranking
contributes on this task, it is smaller than the resolution of a 2,400-sample
five-fold comparison.

The comparison is decided by McNemar's test on paired predictions rather than by
reading the accuracy column, since every arm is evaluated on the same sequences
and the discordant counts carry the signal that two independent intervals throw
away:

| pair | A>B | B>A | chi² | verdict |
|---|---:|---:|---:|---|
| dense vs topological | 854 | 88 | 621.26 | dense better |
| topological vs inverted | 564 | 528 | 1.12 | not resolved |
| topological vs random | 539 | 550 | 0.09 | not resolved |
| inverted vs random | 538 | 585 | 1.88 | not resolved |
| topological vs local+sink | 646 | 441 | 38.29 | topological better |
| random vs local+sink | 727 | 511 | 37.34 | random better |

The same experiment at 600 samples reported topological 58.5% against random
52.2% and would have been written up as the selector winning. Four times the data
moved random to 61.0%, the highest of the three.

The instrument was then tested on the run that produced that gap, and **it
failed**. McNemar at an uncorrected 5% returns `topological better` there, with
chi² 4.72 against a 3.841 threshold — a nominally significant verdict the larger
run refutes at chi² 0.09. The test was behaving correctly; the use of it was not.
Ten pairs are compared at 5% each and the interesting one is chosen after seeing
the table, which manufactures roughly one spurious verdict every two runs.

Bonferroni at 0.05/10 puts the bar at chi² 7.88 and declines it. Re-run at 600
samples under the correction, every comparison among the sparse arms reports `not
resolved`, which is the right answer for that much data. The correction also
costs real power there: topological against local+sink scores 6.90 at 600 and
38.29 at 2,400, so it is declined at the smaller size despite being a genuine
effect. Losing a true positive is the price of not publishing a false one.

Holm's step-down replaced Bonferroni to recover some of that cost. It holds the
same family-wise rate while relaxing the bar as comparisons are consumed — 7.88
on the strongest row down to 5.73 by the eighth — and is never stricter, so it
dominates Bonferroni by construction.

**On this data it changes nothing.** At 2,400 samples both procedures resolve the
same seven comparisons and leave the same three unresolved. At 600 the effect
Bonferroni gave away is `topological vs local+sink` at chi² 6.90, and Holm judges
it at 6.96 — short by 0.06, so it is declined under both. The false positive that
motivated the correction, `topological vs random` at 4.72, faces a relaxed bar of
6.24 under Holm and is still declined.

That the two corrections agree everywhere is worth more than the power Holm was
adopted to recover: the finding does not depend on which multiple-comparison
procedure is used, which was an open question and is now a measured one.

The first configuration was worse than uninformative. With the planted query
scaled at 6, the target key drew about 5% of the softmax against 127 distractors,
dense scored 56.7%, and every arm sat near chance for a reason having nothing to
do with the schedules. The example now refuses to print the comparison unless
dense clears 80%: a control that fails should stop the report rather than appear
as one row of it.

Limits: one synthetic task, one sequence length, features taken from a single row
of a single head. Frozen attention is the point of the design and also its
narrowest assumption — a model trained end to end could reshape queries to suit
whatever schedule it was given, and that is a different and much larger
experiment. `local+sink only` runs at a lower density than the other three, so
its deficit is not a same-budget result and cannot separate budget from
selection.

**Both integrations are measured; neither is made.**

| candidate | verdict | why |
|---|---|---|
| `Tensor::matmul` | **done** — `tensor_matmul` | **crossover n=128**, which is stable across runs. Magnitude is tens of times at n=512, and no tighter — the ratio's run-to-run spread is 96%. Opt-in per call site rather than a change to `Tensor`, so the f32 question is asked by the caller who knows what the result feeds |
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
| crossover n=128 (stable); magnitude tens of × (spread 96%) | **snapshot** | `cargo run -p aether-gpu --example tensor_crossover --release` |
| crash rates (8/60, 0/180) | **snapshot** | `crates/aether-gpu/examples/teardown_repro.rs`, 30 runs a variant |

The snapshots cannot be bound and it is not a gap in the tooling. A timing
depends on the adapter, the driver, the thermal state and what else the machine
was doing; asserting one would produce a test that fails for reasons unrelated
to the code, which is worse than a number carrying a date. What they can carry
is the command that reproduces them, and every row above has one.

### How unstable the snapshots actually are

Reported to three decimals throughout this file, which overstates them. Measured
by re-running an unchanged binary:

| quantity at n=512 | observed range | spread |
|---|---|---:|
| bridge time | 5.034 – 5.696 ms | 12% |
| `Tensor::matmul` time | 197.9 – 321.8 ms | 43% |
| **their ratio** | **18.5× – 61.0×** | **96%** |

The bridge row understates the term badly and is left as recorded because it is
what that sample showed. A later measurement that printed individual timings
instead of medians found the same quantity ranging 4.9 – 25.7 ms — a 5.2× swing
against the 12% here. See *Which side the variance is on* below.

**The ratio is less stable than either term, not more.** An earlier note in this
file claimed the opposite — that both columns move together so the comparison
survives what the absolutes do not — and that was asserted rather than measured.
It is wrong: the two terms drift independently, so dividing them compounds the
error instead of cancelling it.

Three attempts to stabilise it, none of which worked:

| attempt | rationale | ratio spread |
|---|---|---:|
| baseline, 3 CPU reps | — | 96% |
| 9 CPU reps | a median of three is a poor estimator | worse |
| paired, interleaved | each pair sees the same thermal state, so dividing within a pair cancels drift | **130%** |

#### Which side the variance is on

Every attempt above changed *how* the ratio was aggregated, and none asked the
prior question: which of the two factors moves. `--samples` on the crossover
example answers it by printing individual timings instead of a median. Six
consecutive runs, 24 alternating pairs each, n=512:

| factor | range across runs | swing |
|---|---|---:|
| `Tensor::matmul` | 167 – 262 ms | 1.6× |
| bridge | 4.9 – 25.7 ms | **5.2×** |

**The variance is GPU-side.** The CPU term is comparatively steady; the GPU term
moves by a factor of five, and the ratio inherits it almost entirely.

That is also the explanation for the paired result, which until now was recorded
as an unexplained negative. Pairing cancels *common-mode* noise — drift that
moves both terms together. This noise is not common-mode. It sits almost
entirely on one side, so there is nothing for pairing to cancel, and it actively
hurts: it puts the GPU's full variance into every individual ratio sample rather
than leaving it to be averaged down by a median over a block.

An earlier revision of this section concluded from the same negative result that
the variance must live at a shorter timescale than a measurement pair. That was
an inference from pairing failing rather than an observation, and the raw samples
do not support it: within a single run the timings ramp and then flatten, and the
settled portion is far tighter than the run-to-run spread.

What produces the GPU-side swing is not identified. It is not per-sample scatter
and not monotone drift within a process; it changed between runs, stayed changed
across four consecutive runs, and then reverted. Naming a cause on that evidence
would repeat the error this section is correcting.

The interleaved measurement is kept anyway. It is the right design, and the
reason to hold it is that it is correct, not that it is faster; reverting to a
method known to be worse because a better one did not help here would be
optimising the number rather than the measurement.

The practical conclusion is unchanged and now has a mechanism behind it: this
machine cannot produce a trustworthy magnitude, because the quantity that moves
is the one being measured, and no aggregation inside the benchmark reaches it.

What this means for the figures quoted here:

- **The crossover, n=128, is trustworthy.** It is a threshold — whether a ratio
  sits above or below 1 — and it came out at 128 in every run.
- **The magnitude is not.** "36× at n=512" is one draw from a distribution
  spanning at least 18× to 61×. The defensible claim is *tens of times faster at
  n=512 on this hardware*, and any tighter figure in this document should be
  read as illustrative of a single run.

The correct fix is measuring on an idle machine with pinned clocks, which is not
what produced any number here.

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
| "Ratios are more stable than the timings they divide" | **withdrawn** — measured at 96% spread against 43% and 12% for the terms |
| "The benchmark noise is short-timescale" | **withdrawn** — inferred from pairing failing; raw samples show the variance is GPU-side and between runs |
| "Four mechanical mutants escape every suite" | **withdrawn** — site 13 is caught 2 of 3 runs, and the other three differ only within f32 rounding of a maximum that cancels; no hole was found |

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

### The kernels that had no mutant of their own

An audit of the harness against the shader found four kernels carrying no
injected defect. Two of them, `adam_moments` and `add_broadcast_row`, appear in
no test by name at all: nothing calls them directly, and they are reached only
through `adam_update_resident` and `add_bias_resident`. Whether an indirect path
like that notices a defect in what it calls is an assumption every time it is
not measured, and it had not been measured here.

Five mutants close it — the two above, `sigmoid_bce_grad`, and a second
`attention_dk` defect that makes its block-membership test always succeed rather
than corrupting its arithmetic, since a selection bug and a numerical bug fail
differently.

| Injected defect | `gpu_parity` | `gradcheck` | `attention_parity` |
|---|---|---|---|
| `adam_moments` second moment decays with the first beta | **caught** 1/43 | survives | survives |
| `adam_moments` second moment accumulates the gradient unsquared | **caught** 3/43 | survives | survives |
| `add_broadcast_row` broadcasts down the wrong axis | **caught** 1/43 | **caught** 6/14 | survives |
| `sigmoid_bce_grad` batch averaging dropped | survives | **caught** 4/14 | survives |
| `attention_dk` membership test always succeeds | survives | survives | **caught** 1/21 |

**The indirect path does notice.** Both untested-by-name kernels are caught, and
`add_broadcast_row` is caught independently by two suites. The assumption held,
which is worth stating plainly because it is the outcome that would have been
assumed anyway — the reason to run it was that an assumption and a measurement
are the same shape until one of them fails.

Whole-harness result on this tree, RTX 4060 over Vulkan: **0 of 24 escape**, no
pattern unmatched. Nineteen of the twenty-four are caught by exactly one suite,
so dropping any single suite would let nineteen defects through — which is why
the harness runs them separately and reports per-suite rather than combining
them into one pass or fail.

### Mutants nobody chose

`mutants.sh` injects defects someone picked, which bounds the suite from below
and says nothing about defects nobody imagined. The curated set and the tests
that score it were written by the same person with the same idea of what breaks,
so a blind spot shared between them is invisible to both.

`mutants-mechanical.sh` removes the choosing. It enumerates every comparison
operator in the shader and flips each one — `<` to `<=`, `>=` to `>`, `==` to
`!=` — so a survivor is a defect class nobody selected for.

The enumeration requires whitespace on both sides of the operator, and that is
the only reason the result means anything. WGSL spells generics with angle
brackets, so a pattern matching bare `<` and `>` finds 174 sites of which only 92
are comparisons; the other 82 are `vec3<u32>` and friends. Flipping one produces
a file that does not parse, and the harness scores a compile error as **caught** —
correctly, for a real defect the type system rejects, and disastrously here. It
would have reported 82 phantom catches and a coverage figure mostly measuring the
WGSL grammar.

| | count |
|---|---:|
| comparison sites | 92 |
| inside `//` comments, skipped | 2 |
| flips run | 90 |
| **surviving every suite** | **52** |

Against 0 of 24 for the curated set. **That number is not a coverage figure**, and
reading it as one would repeat the mistake this file spends most of its length
correcting.

A flip that does not change what the program computes is an *equivalent mutant*.
No test can catch it, because there is nothing to catch. Deciding which survivors
are equivalent needs evidence the suites cannot supply: a suite reports pass or
fail against a tolerance, so "survives" means the outputs agreed within that
tolerance, not that they were identical. `examples/equivalence_probe.rs` closes
that gap by checksumming the raw bits of each kernel's output, so a mutant can be
run against a clean baseline and compared exactly.

Three survivors, one per class, measured this way:

| site | flip | combined checksum | verdict |
|---|---|---|---|
| 1 | `matmul` guard, `row >= m` → `row > m` | `0x88cb576d1452b4a3` | identical — equivalent |
| 3 | `matmul` loop, `i < k` → `i <= k` | `0x88cb576d1452b4a3` | identical — equivalent |
| 46 | `scheduled_attention` max, `>` → `>=` | `0x88cb576d1452b4a3` | identical — equivalent |

Baseline `0x88cb576d1452b4a3`. All three agree with it bit for bit.

The mechanism differs per class and the outcome does not. wgpu bounds-checks
every access, so the guard flip's extra thread writes out of range and the write
is discarded, and the loop flip's extra iteration reads out of range and gets
zero, which contributes nothing to a sum of products. The max flip changes only
which of two equal candidates is selected, and the maximum is the same value
either way.

**So a boundary mutant in a bounds-checked shading language is equivalent by
construction far more often than in a language where the same defect corrupts
memory.** A naive mutation score for WGSL understates coverage badly, and the
figure to quote is the curated 0 of 24 with the mechanical sweep as context, not
52 of 90 as though it were a hole count.

#### All of them, and the four that are real

The probe now dispatches all twenty kernels rather than five, and
`assert_all_kernels_covered` fails the run if the shader gains one it does not
reach — without that, the gap returns silently the first time a kernel is added
and every later verdict is quietly narrower than it looks.
`classify-survivors.sh` runs every survivor against the baseline.

| of 54 survivors | |
|---|---:|
| bit-identical output — nothing to catch | 50 |
| caught on re-run; the first sample was unlucky | 1 |
| differ only within f32 rounding | 3 |
| **genuine coverage holes** | **0** |

**No coverage hole was found.** That conclusion took three corrections to reach,
and each one is recorded below rather than folded away, because the sequence is
the useful part: a bitwise probe reports "changed" for a mutation that is
mathematically inert, and "changed while the suites pass" reads like a hole every
time.

| site | kernel | flip | what it does |
|---|---|---|---|
| ~~13~~ | `pairwise_sqdist` | `j >= dims.m` → `j >` | **not a hole — see below** |
| 25 | `softmax_xent_grad` | `j < dims.n` → `<=` | reads one column past the row, into the next row |
| 43 | `scheduled_attention` | `n < block_size` → `<=` | one extra key per block |
| 57 | `attention_row_stats` | `d < head_dim` → `<=` | one extra component per row |

**Site 13 was reported as a hole and is not one.** Re-running each mutant three
times per suite instead of once:

| site | `gpu_parity` | `gradcheck` | `attention_parity` |
|---|---|---|---|
| 13 | **caught 2 of 3** | 0/3 | 0/3 |
| 25 | 0/3 | 0/3 | 0/3 |
| 43 | 0/3 | 0/3 | 0/3 |
| 57 | 0/3 | 0/3 | 0/3 |

`the_distance_matrix_matches_the_cpu_reference` does catch site 13, most of the
time. The flip lets thread `j = m` write `c[i*m + m]`, which is the cell thread
`(i+1, 0)` writes legitimately, so the two race and which value survives depends
on scheduling. Run on its own the mutant passed **8 of 8**; run inside the full
suite, where the test harness executes in parallel and the device is under load,
it failed **2 of 3**. The sweep took one sample and drew the wrong conclusion
from it.

So the count went from four to **three**, and the harness now confirms a survival
with a second pass before reporting it. Only survivors are re-run: a catch is
positive evidence that repetition cannot overturn, while a survival is an absence
of evidence and is exactly what an unlucky schedule manufactures. Two
observations do not make it sound, only less wrong — a mutant caught one run in
ten still reads as a survivor most of the time.

##### And then three went to zero

A checksum answers "did anything change" and cannot answer "by how much", which
is the question that decides whether a difference is a defect.
`equivalence_probe --values` prints every output instead, so the remaining three
can be measured rather than assumed:

| site | kernel | worst relative difference | against f32 epsilon 1.19e-07 |
|---|---|---:|---|
| 25 | `softmax_xent_grad` | 2.276e-07 | 1.9× |
| 43 | `scheduled_attention` | 8.333e-07 | 7× |
| 57 | attention backward (`dq`, `dk`, `dv`) | 3.255e-05 | 273× |

All three flips land on a loop computing a **running maximum subtracted purely
for numerical stability**, and every one of those cancels:
`exp(a - mx) / Σ exp(a - mx)` is the same softmax for any `mx`, and the backward
pass divides `weighted` by `denom`, which carries `exp(-mx)` in both. Perturbing
the maximum therefore cannot change what the kernel computes. It changes only
where the exponentials sit in the floating-point range, which is why site 57
moves 273 epsilons while remaining exactly as correct as before.

Calling these holes would demand tests asserting bit-exact intermediate rounding —
tests of the arithmetic's last place rather than of the arithmetic. The suites are
right not to fail on them.

A sparse schedule was added to the probe on the theory that site 43's extra
column is masked by `col > row` under a dense causal schedule and would not be
under a sparse one. It measured **8.333e-07 either way**, identical to three
decimal places, because the flip is on the max pass rather than the accumulation
pass. The hypothesis was wrong; the fixture is kept because attention against a
schedule with blocks genuinely omitted is worth covering regardless of what it
proved here.

**So the mechanical sweep found no coverage hole.** For this operator class, on
these fixtures, the curated mutants and the suites scoring them are not missing
anything — which is a stronger statement about the suites than the 0 of 24 that
prompted the sweep, because nobody chose these sites.

Every one reads or writes into **adjacent, in-bounds** memory. That is exactly
what separates them from the fifty: a flip that runs off the end of a buffer is
bounds-checked into a discarded write or a zero read and changes nothing, while a
flip that lands one element further along the *same* allocation silently
corrupts real data. The bounds checking that makes most boundary mutants harmless
is no help at all here.

The same defect is caught elsewhere, which is what makes these holes rather than
quirks: flipping `matmul`'s column guard (site 2) writes into the next row too and
`gpu_parity` reports 6 of 43 failing. `pairwise_sqdist` has the identical defect
and nothing notices.

Localisation was measured, not assumed. Applying site 13 changes the
`pairwise_sqdist` checksum and no other; site 25 changes `softmax_xent_grad` and
no other. Each defect stays inside its own kernel, so the probe is reporting the
kernel at fault rather than a downstream consequence.

##### The line between rounding and defect, written down

Judging site 57's 273 epsilons against f32 epsilon alone was the weakest step in
the chain: epsilon says what the format can represent, not what a caller needs.
The requirement was never missing, only scattered — every suite asserts one as a
literal, each with a reason beside it.

| what | requirement | asserted in | why that number |
|---|---:|---|---|
| matmul, elementwise | `8 · ε · √k · max\|reference\|` | `gpu_parity.rs:33` | f32 accumulation over `k` terms grows as `√k` and with the magnitude of the terms; constant is twenty times the worst observed ratio, derived below |
| attention forward | `2e-4` | `attention_parity.rs:46` | `exp` is its own derivative, so score error passes into weights undamped; still far tighter than the O(1) moves the tests exist to catch |
| attention backward gradients | `2e-4` (`TOL`) | `attention_parity.rs:926` | same constant as the forward pass, applied to the worst gradient disagreement |
| resident output chained into another kernel | `1e-5` | `attention_parity.rs:553` | a chained product, tighter than `TOL` because it involves no exponential |
| training drift, per step | `1e-4` | `attention_parity.rs:1020` | checked at every step, not only the last, so a divergence that appears at step three and damps by step ten cannot pass |
| distance matrix vs f64 | `1e-6` absolute, `1e-5` relative | `f32_topology.rs:392` | |
| Betti numbers under f32 | **exact** | `f32_topology.rs` | the only assertion here with no tolerance at all |

Against that budget the three measured differences are not marginal:

##### One of the six was not doing anything

Transcribing the budget invited the obvious question of whether each number is
justified, and one is not. Measured against what the suite observes:

| shape | worst absolute difference | in units of `ε·√k` |
|---|---:|---:|
| 8×16×8 | 1.192e-07 | 0.25 |
| 32×32×32 | 2.384e-07 | 0.35 |
| 17×5×23 | 2.980e-08 | 0.11 |

The error stays under `0.35·ε·√k` while `1e-5·√k` permitted `84·ε·√k` — about
240× between what was allowed and what happens. This repository has already
caught itself at 145× once, on a gradient tolerance, and the failure is the same:
a bound that loose does not constrain the kernel.

Tightened to `8·ε·√k`, roughly twenty times the worst observed ratio, and stated
in epsilons because that is the unit the error is in. All 43 tests still pass.

Fitting a constant on three shapes with `k ≤ 32` and applying it to reductions
sixteen times deeper is a guess about the shape of the growth, so the guess was
checked. The ratio of observed error to `ε·√k`, swept over a 32× range:

| k | 16 | 64 | 256 | 512 |
|---|---:|---:|---:|---:|
| ratio | 0.257 | 0.240 | 0.354 | 0.329 |

Flat, not climbing, so the bound keeps its ~23× margin at the top of the range as
well as the bottom. `k = 512` is now part of that test; the f64 reference is
O(n³) and costs about two seconds, which is affordable once. `k = 1024` is not
covered.

One corroboration was already in the tree.
`f32_matmul_error_grows_like_the_square_root_of_the_reduction_depth` had used
`8.0 * 1.19e-7 * sqrt(n)` as its own bound since long before this, so two
independent attempts to say how much error f32 matmul is allowed both landed on
eight epsilons. Neither was derived from the other, which is the only reason the
agreement is evidence of anything.

##### The bound had no magnitude in it

The two were not identical, and the difference was the defect. The sqrt-growth
test divides by the largest exact value before comparing; `tolerance` did not, so
it was an absolute allowance carrying an unstated assumption about how large the
operands are. Every fixture here draws from `fill`, which produces values in
`[-0.5, 0.5]`, and the assumption held invisibly for that reason alone.

Scaling the operands at k=32 and leaving the kernel untouched:

| operand scale | worst absolute error | against a fixed `8·ε·√k` |
|---:|---:|---:|
| 1 | 2.384e-07 | 0.04× — passes |
| 10 | 1.907e-05 | 3.5× — **fails** |
| 100 | 1.953e-03 | 362× — **fails** |
| 1000 | 2.500e-01 | 46341× — **fails** |

The error grows with the square of the operand scale, as the product of two
scaled operands must. A fixed absolute bound therefore rejects a **correct**
kernel for any input much above unit magnitude. That is not a wrong answer
escaping — it is the suite failing on correct code the first time someone writes
a fixture with larger numbers in it, and the kernel getting the blame.

`tolerance` now scales by the largest reference value, which is relative in the
only sense that survives cancellation: an entry near zero has unbounded relative
error and says nothing, while the largest entry sets the scale the accumulation
error is proportional to. Re-measured, the ratio is 0.03, 0.02, 0.02, 0.02 across
those same four scales — flat, where it had spanned six orders of magnitude.

The change loosens the bound at unit magnitude by the reference scale, so the
teeth were re-checked rather than assumed: the injected 1e-5 error still fails
the same seven tests, the three `tolerance()` callers among them.

##### Trying to break it with cancellation, and failing

Scaling by the largest reference value should fail where a result is small
*because* it cancelled: the terms are large, the error is proportional to them,
and the scale collapses. Three fixtures were built to produce that, and none did.

| fixture | worst error | verdict |
|---|---:|---|
| `±1000` alternating against ones | 0 | exactly representable, nothing rounds |
| `1000 + δ` alternating, inexact cancellation | 0 | Sterbenz: subtracting nearby values is exact |
| entries spanning `10⁻³` to `10³`, random signs | 6.250e-02 | passes a 5.235e0 bound, 84× headroom |

The first two are the fixture's fault and the third is the point. Two structural
facts protect the bound, and neither was designed in.

`tolerance` compares a GPU f32 accumulation against `cpu_matmul`, which is **also
f32**. It bounds the disagreement between two orderings of the same products, not
the departure from the exact result — so it is bounded by reordering effects
rather than by absolute rounding, and the catastrophic-cancellation bound from
numerical analysis is answering a question this assertion does not ask. The
untiled kernel and the CPU reference sum in the same order and frequently agree
bit for bit, which is why the first two fixtures measured exactly zero even
through the tiled kernel.

The scale is the maximum over the **whole result matrix**, not per entry. One
entry cancelling to nothing leaves every other entry at full size, so the scale
stays representative of the term magnitudes that the error is proportional to. A
per-entry relative bound would have the weakness this section went looking for;
the matrix-wide maximum does not.

Recorded as a negative result: the risk named in the previous revision is real in
general and not reachable on this kernel with these assertions. What would reach
it is a reference computed in f64, which would make the comparison one of absolute
accuracy rather than of ordering — a different and stricter test than the one
these fixtures perform, and not one this crate currently makes.

The tightening has teeth, demonstrated rather than asserted. Injecting a 1e-5
relative error into `matmul` and running the suite under both bounds:

| | tests failing |
|---|---|
| caught by **both** | `a_rectangular_product_is_not_transposed`, `f32_matmul_error_grows_like_the_square_root_of_the_reduction_depth`, `tensor_matmul_matches_the_cpu_path`, `the_tensor_bridge_reads_through_strides_not_the_flat_buffer` |
| caught **only** by `8·ε·√k` | `gpu_matmul_matches_the_cpu_reference`, `shapes_around_the_workgroup_boundary_are_handled`, `a_resident_chain_equals_the_same_chain_with_readbacks` |

The three in the second row are precisely the tests that call `tolerance()`. Under
the old bound they passed a defect four of their neighbours caught, which is the
sharpest way to put it: they were present, green, and contributing nothing.

| site | measured | governing requirement | headroom |
|---|---:|---:|---:|
| 25 | 2.276e-07 | 8·ε·√k | ~5× inside at k=5 |
| 43 | 8.333e-07 | 2e-4 | 240× inside |
| 57 | 3.255e-05 | 2e-4 forward, 1e-4 per-step drift | 3–6× inside |

Site 57 is the tightest and still sits several times inside the requirement its
own suite asserts, so the verdict rests on a stated tolerance with a documented
rationale rather than on f32 epsilon. It is also the one to re-examine first if a
caller ever needs attention tighter than 2e-4, because 3–6× is headroom that a
change to the accumulation order could consume.

Limits: only comparison operators were swept, so arithmetic and index expressions
are untouched, and zero holes in this class says nothing about those. The two
comment sites were found by reading the survivor list rather than by anticipating
them, so other classes that are equivalent for unnoticed reasons may remain. Every
verdict is against these fixtures — an equivalent verdict means "changed nothing
measurable on this input", not "cannot change anything", and the sparse-schedule
fixture was added precisely because the first one could not have told the
difference for site 43. The budget above is a transcription of what the suites
already enforce, not an independent derivation: if a tolerance was chosen too
loosely in the first place, this table inherits that and makes it look
authoritative.

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


