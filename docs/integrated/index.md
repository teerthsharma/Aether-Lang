# Integrated Mathematics

Ten modules in `aether-core` port mathematics from the author's research repositories and upstream pull requests. Each module takes a geometric, topological or numerical object and returns one of two things: an invariant or decision together with the condition under which it is proved, or a typed refusal naming the condition it could not establish. None returns a plausible number where the evidence does not support one.

Each module lives in `crates/aether-core/src/<module>.rs`, with its tests in `crates/aether-core/tests/<module>.rs`. The ten suites hold 166 `#[test]` functions, counted from the files. CI runs all of them through `cargo test --workspace --exclude aether-kernel`.

## The ten modules

| Module | Object | Invariant or decision | Certificate or refusal | Tests | Source |
| --- | --- | --- | --- | --- | --- |
| [`linking`](linking.md) | Two closed polygons in \(\mathbb{R}^3\) | Gauss linking number, writhe, knot determinant \(\lvert\Delta(-1)\rvert\) | Rounds to \(n\) only when \(\lvert\widehat{\operatorname{Lk}} - n\rvert + B < \tfrac12\). `Linked`, `ZeroLinking` (certifies nothing), `Undetermined`. No "unlinked" verdict | 18 | [nerve](https://github.com/teerthsharma/nerve), [tangle](https://github.com/teerthsharma/tangle) |
| [`certify`](certify.md) | Scores with forward-error radii | Top-k set, argmin, threshold side | Max-in/min-out over \(\gamma_{d+2}\) enclosures. A refusal names the frontier pair and every straddling index | 17 | [separatrix](https://github.com/teerthsharma/separatrix) |
| [`arrangement`](arrangement.md) | Planar segment arrangement | Pieces, bounded faces \(E - V + C\), \(\chi = V - E\) | A snap window of ratio at least \(\rho\) (default 10), with P1–P3 checked. Otherwise a typed refusal of kind Geometry, Budget or Input | 18 | [planimeter](https://github.com/teerthsharma/planimeter) |
| [`resolvent`](resolvent.md) | One causal attention head with switches \((\beta, \mathrm{qk}, g)\) | Softmax, unnormalized kernel and path product as corners of one operator | 18 named Lean 4 theorems, mirrored as tests. Non-finite input refused | 17 | [resolvent](https://github.com/teerthsharma/resolvent) `8c19735` |
| [`orbit`](orbit.md) | Fibres of a many-to-one map | Error floor, pooling and join recovery ceilings, precision floor, recall floor | One-sided bounds under injective \(R\). `None` on counts no partition can produce | 15 | [caustic](https://github.com/teerthsharma/caustic), branchcut (private) |
| [`monodromy`](monodromy.md) | Sampled maps and point clouds | Collision, rotation and dihedral order, PH dimension, Lyapunov spectrum | `CollisionExhibited` or `NoCollisionAtThisSampling`, never "injective". Fractality is decided on an interval. Only the injectivity decision is Jacobian-free | 18 | [monodromy](https://github.com/teerthsharma/monodromy) |
| [`track`](track.md) | Detection centroids per frame | Lineage forest with divisions | Zero certificate violations under \(c_{\text{div}} \ge c_{\text{app}} + c_{\text{det}}\), otherwise `Uncalibrated` | 15 | cleave (private) |
| [`coupling`](coupling.md) | State \(z\), action \(a\) | Banach fixed point, rollout error bound, island count \(\beta_0\) | Contractive iff \(\rho < 1\), for the autonomous map. The bound holds under an \(\varepsilon\) hypothesis. The `LyapunovGain` claim is declined | 18 | [sigmoid](https://github.com/teerthsharma/sigmoid) |
| [`kvwitness`](kvwitness.md) | A learned sparse top-k row over context \(L\) | Segment witnesses \(e_s = \lceil sL/S_{\text{eff}}\rceil\), coverage, mass recall | Budget and prefix contracts. Same-budget random and locality nulls. Full coverage after merge is not guaranteed, and a test pins the counterexample | 15 | [vllm#47942](https://github.com/vllm-project/vllm/pull/47942) (open), `a41354cb34` |
| [`planner`](planner.md) | Tensor lifetimes, a DAG, an incidence graph | Arena offsets, exact transitive reduction, canonical islands | Plans are always sound (greedy, not optimal). Arena at least the peak of live bytes. Precondition violations panic | 15 | [XNNPACK#10801](https://github.com/google/XNNPACK/pull/10801), [tensorflow#124410](https://github.com/tensorflow/tensorflow/pull/124410), [mujoco#3396](https://github.com/google-deepmind/mujoco/pull/3396), [mujoco_warp#1541](https://github.com/google-deepmind/mujoco_warp/pull/1541) |

The upstream pull requests behind `planner`, `kvwitness` and the earlier `scheduled` port are listed with their state and algorithmic idea under [Upstream Contributions](upstream.md).

## Negative results carried by the ports

Several ports found that their source was wrong. Each finding is recorded on its module page and, where it can be, pinned by a test:

- **certify.** The source's direct-kernel radius used \(\gamma_{d+1}\), but the sound constant is \(\gamma_{d+2}\). In the source's own Python, 645 of 199,998 random binary32 pairs escaped the \(\gamma_{d+1}\) radius, and none escaped \(\gamma_{d+2}\). The naive rule that compares only the k-th and (k+1)-th enclosures is also unsound. The counterexample is scores \([0,1,2,10]\), radii \([12,0,0,0]\), \(k = 2\).
- **linking.** nerve rounded the linking number on a measured deviation. This port carries a first-order error bound instead, and the zero verdict certifies nothing: the Whitehead link has \(\operatorname{Lk} = 0\) and is not split.
- **monodromy.** An injective map with a critical point reads as a collision. At small \(n\), a uniform segment is called fractal against \(d_{\text{top}} = 1\). Both are pinned as tests.
- **coupling.** sigmoid's `LyapunovGain` descent condition is false in general. At \(T_0 = 10I\) with the default gains, the closed loop has a root \((3 + \sqrt{17})/2 \approx 3.56\).
- **kvwitness.** The pull request's bound of \(L/S\) on the uncovered run is loose. Segment coverage implies \(2\lceil L/S_{\text{eff}}\rceil - 2\), and coverage can drop after the merge.

## The shared discipline

Every module follows the same three rules. [Theory →](../theory.md#th-claim-boundaries)

1. **State the object and the theorem.** Each page gives the definitions in display mathematics with every symbol defined, and names the proof: Lean, a counting argument, Higham's error analysis, or the Banach fixed-point theorem.
2. **Separate the certified from the estimated.** A certificate is conditional on stated hypotheses (injective truth, a first-order error model, a calibration inequality). An estimate is reported with its bias and failure regime.
3. **Refuse rather than guess.** Every refusal is a typed value that names the condition. None is a NaN, a zero or a default.

## Reproduction

```bash
cargo test -p aether-core --test linking --test certify --test arrangement --test resolvent \
  --test orbit --test monodromy --test track --test coupling --test kvwitness --test planner
```

At commit `c4aff0b`, all 166 tests pass: debug build, rustc 1.99.0-nightly (2026-07-30), Windows 11.

The [linking page](linking.md) recomputes one certificate live in the browser: the linking number of a twisted band, with its error bound and verdict.
