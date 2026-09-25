# Sparse Events

The kernel and core scheduler model use a state-deviation gate.

## State

Implementation: `crates/aether-core/src/state.rs`.

A state is a vector:

\[
\mu(t) \in \mathbb{R}^D
\]

The scheduler stores the last handled state \(\mu(t_{last})\).

## Wake Condition

Implementation: `crates/aether-kernel/src/scheduler.rs`.

\[
\Delta(t) = \|\mu(t)-\mu(t_{last})\|_2
\]

\[
\Delta(t) \ge \epsilon(t)
\]

When the condition is true, the scheduler handles the event, adapts the
governor, records the current state, and increments the event count. When it is
false, the scheduler increments skip count and mixes entropy.

<div class="ts-viz" data-viz="k-wake" data-title="Sparse-event scheduler, running live" data-caption="A 2-D state takes one interrupt per 1 ms. It wakes only when the L2 deviation reaches epsilon, using the SparseScheduler and GeometricGovernor code ported with the same constants."></div>

## Governor

Implementation: `crates/aether-core/src/governor.rs`.

The governor adapts epsilon from observed deviation and elapsed time. It clamps
epsilon to a bounded interval so the threshold cannot collapse to zero or grow
without bound.

<div class="ts-viz" data-viz="k-governor" data-title="GeometricGovernor.adapt() under a constant deviation" data-caption="epsilon -= 0.01*e + 0.05*de/dt with e = 1000 - delta/epsilon, clamped to [0.001, 10]. With the shipped gains it jumps between the two clamps instead of settling."></div>

## Active Evidence

Unit tests cover:

- no wake on unchanged state;
- wake on large deviation;
- entropy accumulation;
- event ratio computation;
- governor epsilon initialization and clamp behavior;
- threshold trigger behavior.

## Claim Boundary

The sparse-event model is active code with tests. System-wide power reduction,
latency guarantees, and bare-metal production behavior require hardware
artifacts and should be labeled as roadmap until measured.
