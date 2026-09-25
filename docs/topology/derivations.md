# Derivations

Each formula names its implementation surface; the viz beside it computes it.
[Theory →](../theory.md#th-derivations)

## Time-Delay Embedding

Implementation: `TimeDelayEmbedder<D>`.

\[
\Phi(t) = [x(t), x(t-\tau), x(t-2\tau), \ldots, x(t-(D-1)\tau)]
\]

- DSL workspace: \(D=3\); `tau=0` is normalized to `1`.
- A point is emitted only after enough samples exist.

<div class="ts-viz" data-viz="topo-delay" data-title="Time-delay embedding Φ(t), D = 3" data-caption="A two-tone signal embedded as [x(t), x(t−τ), x(t−2τ)]; drag the 3D view to rotate, move τ and t."></div>


## Euclidean Distance

Implementation: `ManifoldPoint<D>::distance`. Used by manifold neighborhoods, Vietoris-Rips, lazy witness and block metadata.

\[
d(p,q) = \sqrt{\sum_{i=1}^{D}(p_i-q_i)^2}
\]

<div class="ts-viz" data-viz="topo-distance" data-title="Euclidean distance d(p,q)" data-caption="The hypotenuse of the coordinate differences, shown for D = 2; drag p and q."></div>


## Block Centroid

Implementation: `BlockMetadata<D>::from_points`, for \(B = \{x_1,\ldots,x_n\}\).

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

With \(\bar{d}\) the mean point-to-centroid distance. [Theory →](../theory.md#th-block-statistics)

\[
\sigma_B^2 = \frac{1}{n}\sum_{i=1}^{n} d(x_i,\mu_B)^2 - \bar{d}^2
\]

<div class="ts-viz" data-viz="topo-block" data-focus="variance" data-title="Distance variance σ²_B" data-caption="Spread of the point-to-centroid distances: green circle is d̄, dashed circle is r_B."></div>


## Concentration

Zero-norm terms are skipped by implementation guards.

\[
c_B = \frac{1}{n}\sum_{i=1}^{n} \frac{x_i \cdot \mu_B}{\|x_i\|\|\mu_B\|}
\]

<div class="ts-viz" data-viz="topo-block" data-focus="concentration" data-title="Concentration c_B" data-caption="Mean cosine between each point's direction from the origin (orange) and the centroid's direction (blue)."></div>


## Cauchy-Schwarz Upper Bound

Implementation: `BlockMetadata<D>::upper_bound_score`. Bound below threshold → block pruned without inspecting its points. [Theory →](../theory.md#th-cauchy-schwarz)

\[
score(q,B) \le \|q\|(\|\mu_B\| + r_B)
\]

<div class="ts-viz" data-viz="topo-bound" data-title="Cauchy–Schwarz pruning bound" data-caption="Every q·xᵢ sits under ‖q‖(‖μ_B‖ + r_B); when that bound falls below the threshold the block is skipped. Drag q."></div>


## Sparse Event Trigger

Implementation: `SparseScheduler<D>::should_wake`, with \(\Delta(t) = \|\mu(t)-\mu(t_{last})\|_2\) and adaptive threshold \(\epsilon(t)\).

\[
\text{wake} \iff \Delta(t) \ge \epsilon(t)
\]

<div class="ts-viz" data-viz="topo-wake" data-title="Sparse event trigger" data-caption="A drifting state μ(t) with a burst mid-run; a wake fires whenever it leaves the ε-ball around the last handled state."></div>


## Governor Update

Implementation: `GeometricGovernor::adapt`, a clamped proportional-derivative step on \(\ln\epsilon\) with \(e(t) = \mathrm{clamp}(1 - R_{actual}/R_{target}, \pm 1)\), \(R_{actual} = \Delta(t)/\epsilon(t)\). A steady \(\Delta\) settles at \(\epsilon^* = \Delta/R_{target}\). [Theory →](../theory.md#th-governor)

\[
\ln\epsilon(t+1) = \ln\epsilon(t) - \alpha e(t) - \beta\,(e(t) - e(t-1))
\]

<div class="ts-viz" data-viz="topo-governor" data-title="Governor update ε(t+1)" data-caption="The proportional-derivative update with governor.rs constants, iterated from ε₀ = 0.1 for a constant Δ; orange steps hit a clamp."></div>


## Binary Shape Heuristic

Implementation: `crates/aether-core/src/topology.rs`. Density and approximate loop count are compared against fixed thresholds.

\[
density = \frac{\beta_0}{|B|}
\]

<div class="ts-viz" data-viz="topo-shape" data-title="Binary shape heuristic" data-caption="β₀, β₁ and density = β₀/|B| computed byte-for-byte as in topology.rs, with the gate's verdict; pick a preset or edit the hex."></div>


Heuristic gate with tests; not a production malware detector or complete authentication system. [Theory →](../theory.md#th-shape-heuristic)
