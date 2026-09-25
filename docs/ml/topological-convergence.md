# Topological Convergence

Topological convergence in Aether means that model or residual behavior is
observed through shape signals, not only scalar loss.

## Current Internal Signals

The convergence modules use:

- scalar error;
- Betti-number history;
- centroid drift;
- residual sign-change and oscillation heuristics;
- fixed windows and thresholds.

<div class="ts-viz" data-viz="ml-seal-detector" data-title="A seal loop that stops when the residual shape stops changing" data-caption="Gradient descent on a cubic; each epoch feeds ResidualAnalyzer β and drift plus RMSE into ConvergenceDetector (ε = 1e-3), and the loop halts on the first epoch where is_converged() returns true."></div>


## Internal Convergence Shape

Sign changes and oscillation counts of \(r_i = y_i - \hat{y_i}\): a lightweight heuristic, not persistent homology.
[Theory →](../theory.md#th-residual-shape)

<div class="ts-viz" data-viz="ml-residual-shape" data-title="The interpreter residual heuristic, epoch by epoch" data-caption="EscalatingRegressor::run_escalating on a demo series: β = (sign_changes/2 + 1, oscillations/4) per model, stopping once the last three β readings match."></div>


The persistent-homology path is separate:

```aether
let diagram = topology.ph(M, max_dim=2)~
let b = topology.betti(diagram, radius=0.5)~
```

## Claim Boundary

[Theory →](../theory.md#th-claim-boundaries)

It is accurate to say:

- Aether exposes topology and residual-shape signals for convergence logic.
- Some tests verify parser and interpreter paths for `seal until` and topology
  calls.
- The core crate contains convergence and residual-analysis structures.

It is not yet accurate to say:

- every training loop terminates by persistent homology;
- topology improves model quality on external datasets;
- topological convergence replaces validation metrics;
- convergence behavior is benchmarked across model classes.
