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

## Internal Convergence Shape

For a residual sequence \(r_i = y_i - \hat{y_i}\), the interpreter-level
escalating regressor estimates shape using sign changes and oscillation counts.
That is a lightweight residual heuristic, not persistent homology.

The persistent-homology path is separate:

```aether
let diagram = topology.ph(M, max_dim=2)~
let b = topology.betti(diagram, radius=0.5)~
```

## Claim Boundary

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
