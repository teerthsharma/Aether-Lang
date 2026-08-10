# aether-gpu feature ledger

State carried between iterations of the running build loop. Each row is a
capability that exists in the tree and has a test or a measurement behind it.
A row is only added once the thing works; planned work lives in **Next**.

Measured on NVIDIA GeForce RTX 4060 Laptop GPU, Vulkan, `DiscreteGpu`, driver
595.79, Windows 11, f32 throughout (WGSL has no f64).

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

## Mutation testing

A suite nobody has mutated is a suite of unknown strength. Two defects injected
into `shaders.wgsl`, one at a time, rebuilt, and run against both suites.

| Injected defect | `gradcheck` | `gpu_parity` |
|---|---|---|
| `sigmoid_bce_grad` drops the `1/batch` scaling | **caught** — `dw1[0]` relative error 4.0 | survives |
| `relu_backward` boundary `> 0.0` becomes `>= 0.0` | survives | **caught** — `relu_backward_is_zero_at_exactly_zero` |

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

## An unreproduced crash

One `STATUS_ACCESS_VIOLATION` (`0xc0000005`) at process exit, on the first run
of `train_resident` after a rebuild. It did not reproduce in twelve subsequent
runs of the same binary, and has not reappeared since.

Not diagnosed. `GpuContext` now implements `Drop` to submit any recorded work
and block until the device is idle, which closes a plausible mechanism --
tearing down a device with queue work in flight -- but no evidence links that
mechanism to the observed fault. It is hardening, not a fix, and it is listed
here rather than in **Shipped** for that reason.

The `Drop` impl earns its place independently: without it, work recorded and
never flushed is discarded silently, so a caller that updates parameters and
never reads them back would lose the update with no error.

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
- No gradcheck. The backward kernels are checked against forward/backward
  agreement at the ReLU boundary and by end-to-end training convergence, not by
  finite differences.
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
