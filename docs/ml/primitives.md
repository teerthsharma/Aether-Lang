# ML Primitives

Aether's ML code lives under `crates/aether-core/src/ml`. The documentation
describes the module inventory and claim boundaries rather than presenting it as
a benchmarked replacement for external ML frameworks.

## Module Inventory

| Module | Active surface |
| --- | --- |
| `tensor` | Owned tensor data, shapes, indexing, map, add, sub, mul, scale, transpose, matmul, reductions |
| `linalg` | Loss functions, distances, RBF kernel, numerical gradients |
| `regressor` | Linear, polynomial, RBF-style, Gaussian-process-labeled, and geodesic-labeled model enum paths |
| `convergence` | Betti records, drift/error windows, residual analysis |
| `benchmark` | Escalating benchmark runner over internal test functions |
| `clustering` | KMeans, DBSCAN, agglomerative clustering, auto-k helper |
| `classification` | Logistic regression, KNN, perceptron, Gaussian naive Bayes, decision stump, AdaBoost, nearest centroid |
| `neural` | Dense layers, activations, optimizer config, MLP training loop |
| `autograd` | Tape and variable scaffolding for differentiable tensor operations |
| `convolution` | Conv2D forward path |
| `dataloader` | Batch iteration over tensors |
| `gossip` | Local centroid and consensus propagation |

<div class="ts-viz" data-viz="ml-kmeans" data-title="KMeans: assign, update, stop" data-caption="KMeans::fit with k-means++ seeding from seed 42; stops when no label changes or inertia moves less than 1e-4."></div>
<div class="ts-viz" data-viz="ml-classify" data-title="Logistic regression and perceptron boundaries" data-caption="LogisticRegression::fit (lr 0.1, tol 1e-4, 100 iterations) and Perceptron::fit (lr 1, stop at zero errors) trained live on two Gaussian classes."></div>
<div class="ts-viz" data-viz="ml-mlp-xor" data-title="A 2-8-1 MLP learning XOR" data-caption="The test_mlp_xor setup: tanh hidden layer, sigmoid output, MSE loss, SGD with momentum, crate LCG initialisation with seeds 42 and 43."></div>
<div class="ts-viz" data-viz="ml-conv2d" data-title="Conv2D forward pass" data-caption="One 3×3 kernel slides over a 12×12 input with zero padding and stride, and ReLU gives each output cell."></div>
<div class="ts-viz" data-viz="ml-gossip" data-title="Gossip consensus on a ring" data-caption="GossipRing::tick: each node averages with its ring predecessor (α = 0.5); converge stops when every estimate is within tolerance of the mean."></div>


## Language Boundary

The DSL exposes a narrower surface than the Rust crate inventory. Constructors
such as `Ml.MLP`, `Ml.KMeans`, and `Ml.Conv2D` are available through native
function dispatch. Individual methods should be documented as active only when
the interpreter path is implemented and tested.

## Claim Boundary

[Theory →](../theory.md#th-claim-boundaries)

The ML module can be described as internal Rust ML primitives. It should not be
documented as:

- faster than PyTorch, TensorFlow, sklearn, GUDHI, or ripser;
- production-ready for all model families;
- equivalent to external framework semantics;
- hardware accelerated.

Those claims require benchmark artifacts and parity tests.
