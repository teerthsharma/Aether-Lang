# Benefit Emergence

A benefit appears only when carried structure makes a downstream decision local, bounded, or auditable.
[Theory →](../theory.md#th-benefit-emergence)

## Pattern

```text
raw object -> structural representation -> low-cost invariant -> gate
```

Examples:

- source text becomes tokens, AST nodes, spans, and runtime values;
- scalar samples become delay-coordinate points;
- point clouds become persistence diagrams or Betti counts;
- point batches become block centroids, radii, variances, and concentrations;
- kernel state becomes a vector with a deviation threshold;
- binary data becomes a shape heuristic with rejection reasons.

The system benefit emerges from the gate:

- a parser can stop at the span that violates grammar;
- a topology call can fail before unbounded simplex expansion;
- a block query can prune a block when its upper bound is below threshold;
- a scheduler can skip work when state deviation is below epsilon;
- a benchmark policy can reject claims without artifacts.

<div class="ts-viz" data-viz="ml-sparse-gate" data-title="A deviation gate skipping work" data-caption="should_wake: work runs only when Δ = ‖μ(t) − μ(t_last)‖₂ ≥ ε, and each wake resets the reference state. ε is held fixed here; the governor that adapts it is not simulated."></div>


## Non-Claim

Not currently proven: general model-quality improvement, general security detection, hardware acceleration, asymptotic speedup over external libraries.
Each needs benchmark artifacts, baselines, correctness metrics and environment records.

## Engineering Rule

Aether docs should describe a benefit only through the mechanism that produces
it:

```text
representation + invariant + gate = claimed behavior
```

If one of those three parts is missing, the claim belongs in roadmap text.
