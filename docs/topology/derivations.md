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

<div class="ts-viz" data-viz="topo-delay" data-title="Time-delay embedding Φ(t), D = 3" data-caption="A two-tone signal embedded as [x(t), x(t−τ), x(t−2τ)]; drag the 3D view to rotate, move τ and t."></div>


## Euclidean Distance

Implementation: `ManifoldPoint<D>::distance`.

\[
d(p,q) = \sqrt{\sum_{i=1}^{D}(p_i-q_i)^2}
\]

This distance is used by manifold neighborhoods, Vietoris-Rips construction,
lazy witness construction, and block metadata.

<div class="ts-viz" data-viz="topo-distance" data-title="Euclidean distance d(p,q)" data-caption="The hypotenuse of the coordinate differences, shown for D = 2; drag p and q."></div>


## Block Centroid

Implementation: `BlockMetadata<D>::from_points`.

For a block \(B = \{x_1,\ldots,x_n\}\):

\[
\mu_B = \frac{1}{n}\sum_{i=1}^{n} x_i
\]

<div class="ts-viz" data-viz="topo-block" data-focus="centroid" data-title="Block centroid μ_B" data-caption="The mean of seven draggable points (the cross)."></div>


## Block Radius

\[
r_B = \max_i d(x_i,\mu_B)
\]

<div class="ts-viz" data-viz="topo-block" data-focus="radius" data-title="Block radius r_B" data-caption="The farthest point from μ_B sets the dashed enclosing circle."></div>


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

<div class="ts-viz" data-viz="topo-block" data-focus="variance" data-title="Distance variance σ²_B" data-caption="Spread of the point-to-centroid distances: green circle is d̄, dashed circle is r_B."></div>


## Concentration

\[
c_B =
\frac{1}{n}\sum_{i=1}^{n}
\frac{x_i \cdot \mu_B}{\|x_i\|\|\mu_B\|}
\]

Zero-norm terms are skipped by implementation guards.

<div class="ts-viz" data-viz="topo-block" data-focus="concentration" data-title="Concentration c_B" data-caption="Mean cosine between each point's direction from the origin (orange) and the centroid's direction (blue)."></div>


## Cauchy-Schwarz Upper Bound

Implementation: `BlockMetadata<D>::upper_bound_score`.

For query \(q\):

\[
score(q,B) \le \|q\|(\|\mu_B\| + r_B)
\]

If this bound is below a threshold, the block can be pruned without inspecting
every point in the block.

<div class="ts-viz" data-viz="topo-bound" data-title="Cauchy–Schwarz pruning bound" data-caption="Every q·xᵢ sits under ‖q‖(‖μ_B‖ + r_B); when that bound falls below the threshold the block is skipped. Drag q."></div>


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

<div class="ts-viz" data-viz="topo-wake" data-title="Sparse event trigger" data-caption="A drifting state μ(t) with a burst mid-run; a wake fires whenever it leaves the ε-ball around the last handled state."></div>


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

<div class="ts-viz" data-viz="topo-governor" data-title="Governor update ε(t+1)" data-caption="The proportional-derivative update with governor.rs constants, iterated from ε₀ = 0.1 for a constant Δ; orange steps hit a clamp."></div>


## Binary Shape Heuristic

Implementation: `crates/aether-core/src/topology.rs`.

The binary shape gate computes:

\[
density = \frac{\beta_0}{|B|}
\]

and compares density and approximate loop count against fixed thresholds.

<div class="ts-viz" data-viz="topo-shape" data-title="Binary shape heuristic" data-caption="β₀, β₁ and density = β₀/|B| computed byte-for-byte as in topology.rs, with the gate's verdict; pick a preset or edit the hex."></div>


Claim boundary: this is a heuristic gate with tests. It is not documented as a
production malware detector or a formally complete authentication system.
