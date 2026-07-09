# Derivations

This page records the formulas used across the Aether docs. It is intentionally
mechanical: each formula names the object, the implementation surface, and the
claim boundary.

## Time-Delay Embedding

Implementation: `TimeDelayEmbedder<D>`.

For scalar samples \(x(t)\), delay \(\tau\), and dimension \(D\):

\[
\Phi(t) = [x(t), x(t-\tau), x(t-2\tau), \ldots, x(t-(D-1)\tau)]
\]

Current interpreter boundary:

- the DSL workspace uses \(D=3\);
- `tau=0` is normalized to `1`;
- an embedded point is emitted only after enough samples exist.

## Euclidean Distance

Implementation: `ManifoldPoint<D>::distance`.

\[
d(p,q) = \sqrt{\sum_{i=1}^{D}(p_i-q_i)^2}
\]

This distance is used by manifold neighborhoods, Vietoris-Rips construction,
lazy witness construction, and block metadata.

## Block Centroid

Implementation: `BlockMetadata<D>::from_points`.

For a block \(B = \{x_1,\ldots,x_n\}\):

\[
\mu_B = \frac{1}{n}\sum_{i=1}^{n} x_i
\]

## Block Radius

\[
r_B = \max_i d(x_i,\mu_B)
\]

## Distance Variance

Let:

\[
\bar{d} = \frac{1}{n}\sum_{i=1}^{n} d(x_i,\mu_B)
\]

Then:

\[
\sigma_B^2 =
  \frac{1}{n}\sum_{i=1}^{n} d(x_i,\mu_B)^2 - \bar{d}^2
\]

## Concentration

\[
c_B =
\frac{1}{n}\sum_{i=1}^{n}
\frac{x_i \cdot \mu_B}{\|x_i\|\|\mu_B\|}
\]

Zero-norm terms are skipped by implementation guards.

## Cauchy-Schwarz Upper Bound

Implementation: `BlockMetadata<D>::upper_bound_score`.

For query \(q\):

\[
score(q,B) \le \|q\|(\|\mu_B\| + r_B)
\]

If this bound is below a threshold, the block can be pruned without inspecting
every point in the block.

## Sparse Event Trigger

Implementation: `SparseScheduler<D>::should_wake`.

For system state \(\mu(t)\), last handled state \(\mu(t_{last})\), and adaptive
threshold \(\epsilon(t)\):

\[
\Delta(t) = \|\mu(t)-\mu(t_{last})\|_2
\]

\[
\text{wake} \iff \Delta(t) \ge \epsilon(t)
\]

## Governor Update

Implementation: `GeometricGovernor::adapt`.

The observed rate is:

\[
R_{actual} = \frac{\Delta(t)}{\epsilon(t)}
\]

The error is:

\[
e(t) = R_{target} - R_{actual}
\]

The derivative term is:

\[
\frac{de}{dt} = \frac{e(t)-e(t-1)}{dt}
\]

The implementation applies a proportional-derivative adjustment and clamps
\(\epsilon\) into a fixed interval:

\[
\epsilon(t+1) = clamp(\epsilon(t) - \alpha e(t) - \beta \frac{de}{dt})
\]

The sign follows the current code path: high observed rate raises epsilon after
the update dynamics settle.

## Binary Shape Heuristic

Implementation: `crates/aether-core/src/topology.rs`.

The binary shape gate computes:

\[
density = \frac{\beta_0}{|B|}
\]

and compares density and approximate loop count against fixed thresholds.

Claim boundary: this is a heuristic gate with tests. It is not documented as a
production malware detector or a formally complete authentication system.
