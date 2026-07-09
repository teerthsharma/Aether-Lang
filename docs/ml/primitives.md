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

## Language Boundary

The DSL exposes a narrower surface than the Rust crate inventory. Constructors
such as `Ml.MLP`, `Ml.KMeans`, and `Ml.Conv2D` are available through native
function dispatch. Individual methods should be documented as active only when
the interpreter path is implemented and tested.

## Claim Boundary

The ML module can be described as internal Rust ML primitives. It should not be
documented as:

- faster than PyTorch, TensorFlow, sklearn, GUDHI, or ripser;
- production-ready for all model families;
- equivalent to external framework semantics;
- hardware accelerated.

Those claims require benchmark artifacts and parity tests.
