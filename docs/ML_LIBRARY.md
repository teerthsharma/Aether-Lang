# AEGIS ML Library Reference

> **Complete Machine Learning from Scratch in a Topologically-Complete Language**

AEGIS provides a mathematically rigorous ML library built on topological foundations. Every model terminates via **convergence detection**, not arbitrary epoch limits.

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Linear Algebra Primitives](#linear-algebra-primitives)
3. [Regression Models](#regression-models)
4. [Topological Convergence](#topological-convergence)
5. [Manifold Operations](#manifold-operations)
6. [AETHER Geometric Primitives](#aether-geometric-primitives)

---

## Core Concepts

### The AEGIS ML Philosophy

| Traditional ML | AEGIS ML |
|---------------|----------|
| Fixed epochs (guess) | **Topological convergence** (proven) |
| O(n) iteration | **O(log n)** via seal loops |
| Dense attention O(n²) | **Sparse attention** via geometry |
| Gradient magnitude stopping | **Betti number stability** |

### The `~` Statement Terminator

Every AEGIS statement ends with `~` (the "seal"):

```aegis
let x = 5~
let y = compute_something()~
regress { model: "linear" }~
```

### The `🦭` Seal Loop

AEGIS's revolutionary loop construct that terminates when **topology stabilizes**:

```aegis
🦭 until convergence(1e-6) {
    train_step()~
}
// Terminates when Betti numbers stabilize!
```

---

## Linear Algebra Primitives

### ManifoldPoint<D>

A point in D-dimensional manifold space.

```rust
// Rust API
let p1 = ManifoldPoint::<3>::new([1.0, 2.0, 3.0]);
let p2 = ManifoldPoint::<3>::new([4.0, 5.0, 6.0]);

let dist = p1.distance(&p2);        // Euclidean distance
let is_near = p1.is_neighbor(&p2, 0.5);  // Within ε neighborhood
```

```aegis
// AEGIS syntax
let p1 = point(1.0, 2.0, 3.0)~
let p2 = point(4.0, 5.0, 6.0)~
let dist = distance(p1, p2)~
```

**Methods:**
| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(coords: [f64; D]) -> Self` | Create point |
| `zero` | `const fn zero() -> Self` | Origin point |
| `distance` | `fn distance(&self, other: &Self) -> f64` | Euclidean L2 |
| `is_neighbor` | `fn is_neighbor(&self, other: &Self, ε: f64) -> bool` | Locality check |

### Vector Operations

```aegis
// Built-in vector operations
fn dot(a, b) {
    let sum = 0.0~
    for i in 0..len(a) {
        sum = sum + a[i] * b[i]~
    }
    return sum~
}

fn norm(v) {
    return sqrt(dot(v, v))~
}

fn normalize(v) {
    let n = norm(v)~
    let result = zeros(len(v))~
    for i in 0..len(v) {
        result[i] = v[i] / n~
    }
    return result~
}
```

---

## Regression Models

### ManifoldRegressor<D>

The core regression engine operating on D-dimensional manifold-embedded data.

```rust
// Rust API
use aegis_core::ml::regressor::{ManifoldRegressor, ModelType};

let mut regressor: ManifoldRegressor<3> = ManifoldRegressor::new(ModelType::Linear);
regressor.add_point([0.0, 0.5, 0.25], 1.0);
regressor.add_point([0.1, 0.55, 0.28], 1.1);

let error = regressor.fit();
let prediction = regressor.predict(&[0.05, 0.52, 0.26]);
```

```aegis
// AEGIS syntax
manifold M = embed(data, dim=3, tau=5)~

regress {
    model: "linear",
    escalate: true,
    until: convergence(1e-6)
}~

let pred = predict(new_point)~
```

### Model Types

```rust
pub enum ModelType {
    Linear,                           // y = a + bx
    Polynomial(u8),                   // y = Σ aᵢxⁱ
    Rbf { gamma: f64 },              // Radial Basis Function
    GaussianProcess { length_scale: f64 },  // Approximate GP
    GeodesicRegression,              // Manifold-aware
}
```

**Complexity Ladder (for auto-escalation):**
| Model | Complexity | Best For |
|-------|------------|----------|
| `Linear` | 1 | Quick baseline |
| `Polynomial(2)` | 3 | Curved relationships |
| `Polynomial(3)` | 4 | Cubic patterns |
| `Rbf { gamma: 1.0 }` | 5 | Non-linear clusters |
| `GaussianProcess` | 7 | Uncertainty quantification |
| `GeodesicRegression` | 9 | Manifold-curved data |

### Escalating Regression

AEGIS automatically upgrades model complexity when simpler models fail:

```aegis
manifold M = embed(sensor_data, dim=3, tau=7)~

// Start simple, escalate until convergence
regress {
    model: "linear",
    escalate: true,
    until: convergence(1e-6)
}~

// Progression: Linear → Poly(2) → Poly(3) → RBF → GP → Converged!
```

### Methods Reference

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(model: ModelType) -> Self` | Create regressor |
| `add_point` | `fn add_point(&mut self, point: [f64; D], target: f64)` | Add training data |
| `fit` | `fn fit(&mut self) -> f64` | Fit model, returns MSE |
| `predict` | `fn predict(&self, point: &[f64; D]) -> f64` | Predict target |
| `upgrade_model` | `fn upgrade_model(&mut self)` | Escalate complexity |
| `error` | `fn error(&self) -> f64` | Current MSE |
| `coefficients` | `fn coefficients(&self) -> &Coefficients` | Fitted params |

---

## Topological Convergence

### BettiNumbers

The fundamental shape signature: (β₀, β₁)
- **β₀**: Connected components (clusters)
- **β₁**: Loops/cycles (holes)

```rust
use aegis_core::ml::convergence::BettiNumbers;

let betti = BettiNumbers::new(1, 0);
assert!(betti.is_singular());  // β₀=1, β₁=0 → perfect convergence

let dist = betti.distance(&BettiNumbers::new(2, 1));  // L1 distance = 2
```

**Interpretation:**
| Shape | β₀ | β₁ | Meaning |
|-------|----|----|---------|
| Single blob | 1 | 0 | Converged! |
| Two clusters | 2 | 0 | Under-fitting |
| Ring/torus | 1 | 1 | Cyclic pattern |
| Scattered | >3 | ? | Not converged |

### ConvergenceDetector

Detects when training should stop based on topological stability:

```rust
use aegis_core::ml::convergence::ConvergenceDetector;

let mut detector = ConvergenceDetector::new(1e-6, 5);  // ε=1e-6, window=5

// Each epoch
detector.record_epoch(
    BettiNumbers::new(2, 1),  // Current topology
    0.1,                       // Centroid drift
    0.05                       // Error
);

if detector.is_converged() {
    println!("Sealed! Score: {}", detector.convergence_score());
}
```

**Convergence Criteria:**
1. **Error < ε**: Loss below threshold
2. **Betti stable**: Same shape for `window` epochs
3. **Drift stable**: Centroid movement near zero

### ResidualAnalyzer

Compute Betti numbers of regression residuals:

```rust
use aegis_core::ml::convergence::ResidualAnalyzer;

let mut analyzer: ResidualAnalyzer<3> = ResidualAnalyzer::new(0.5);
analyzer.set_residuals(&residual_values);

let betti = analyzer.compute_betti();
// β₀ = sign-change clusters
// β₁ = oscillation cycles
```

---

## Manifold Operations

### TimeDelayEmbedder<D>

Implements **Takens' Theorem**: transform 1D time series into D-dimensional manifold.

```
Φ(t) = [x(t), x(t-τ), x(t-2τ), ..., x(t-(D-1)τ)]
```

```rust
use aegis_core::manifold::TimeDelayEmbedder;

let mut embedder: TimeDelayEmbedder<3> = TimeDelayEmbedder::new(5);  // τ=5

for value in time_series {
    embedder.push(value);
}

if let Some(point) = embedder.embed() {
    // 3D manifold point capturing system dynamics
}
```

```aegis
// AEGIS syntax
manifold M = embed(time_series, dim=3, tau=5)~
```

### SparseAttentionGraph<D>

O(n) attention instead of O(n²) via geometric locality:

```
A(i,j) = 1  iff  d(pᵢ, pⱼ) < ε
```

```rust
use aegis_core::manifold::SparseAttentionGraph;

let mut graph: SparseAttentionGraph<3> = SparseAttentionGraph::new(0.5);

graph.add_point(ManifoldPoint::new([0.0, 0.0, 0.0]));
graph.add_point(ManifoldPoint::new([0.1, 0.1, 0.1]));

let (beta_0, beta_1) = graph.shape();  // Topological signature
let degree = graph.degree(0);          // Neighbor count
```

### GeometricConcentrator<D>

Streaming PCA for dimension reduction:

```rust
use aegis_core::manifold::GeometricConcentrator;

let mut concentrator: GeometricConcentrator<3> = GeometricConcentrator::new();

for point in manifold_points {
    concentrator.update(&point);
}

let principal_dim = concentrator.principal_dimension();
let ratio = concentrator.concentration_ratio();  // Variance explained
```

### TopologicalPipeline<D>

Complete: Stream → Embed → Sparse Attention → Shape

```rust
use aegis_core::manifold::TopologicalPipeline;

let mut pipeline: TopologicalPipeline<3> = TopologicalPipeline::new(5, 0.5);

for value in data_stream {
    if let Some((beta_0, beta_1)) = pipeline.push(value) {
        println!("Shape: β₀={}, β₁={}", beta_0, beta_1);
    }
}
```

---

## AETHER Geometric Primitives

### BlockMetadata<D>

Geometric summary of a data block:

```rust
use aegis_core::aether::BlockMetadata;

let points = vec![
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
];

let block = BlockMetadata::<3>::from_points(&points);
println!("Centroid: {:?}", block.centroid);
println!("Radius: {}", block.radius);
println!("Variance: {}", block.variance);
println!("Concentration: {}", block.concentration);
```

| Field | Type | Description |
|-------|------|-------------|
| `centroid` | `[f64; D]` | Block center (mean) |
| `radius` | `f64` | Max deviation from centroid |
| `variance` | `f64` | Distance variance |
| `concentration` | `f64` | Angular alignment (cosine) |
| `count` | `usize` | Point count |

### HierarchicalBlockTree<D>

Multi-scale tree for O(log n) queries:

```
Level 2: [||||] 1024-token super-clusters
Level 1: [||] [||] 256-token clusters  
Level 0: [|] [|] [|] [|] 64-token blocks (finest)
```

```rust
use aegis_core::aether::HierarchicalBlockTree;

let mut tree: HierarchicalBlockTree<3> = HierarchicalBlockTree::new();
tree.build_from_blocks(&blocks);

// Hierarchical query with early pruning
let active_mask = tree.hierarchical_query(&query, threshold);
let pruning = tree.pruning_ratio(&active_mask);
// Often 50-90% of blocks pruned!
```

### DriftDetector<D>

Track semantic drift for online learning:

```rust
use aegis_core::aether::DriftDetector;

let mut detector: DriftDetector<3> = DriftDetector::new();

for batch in batches {
    let centroid = compute_centroid(&batch);
    let drift = detector.update(&centroid);
    
    if detector.is_drifting(0.1) {
        trigger_retraining();
    }
}
```

---

## Quick Reference

### AEGIS ML Cheat Sheet

```aegis
// === DATA LOADING ===
let data = [1.0, 2.1, 3.5, 4.2, 5.1]~

// === MANIFOLD EMBEDDING ===
manifold M = embed(data, dim=3, tau=5)~

// === BLOCK EXTRACTION ===
block B = M[0:64]~
centroid C = B.center~
radius R = B.spread~

// === REGRESSION ===
regress {
    model: "polynomial",
    degree: 3,
    escalate: true,
    until: convergence(1e-6)
}~

// === SEAL LOOP ===
🦭 until convergence(1e-6) {
    train_step()~
}

// === PREDICTION ===
let pred = predict(new_point)~

// === VISUALIZATION ===
render M {
    color: by_density,
    trajectory: on
}~
```

---

## See Also

- [ML Tasks Encyclopedia](ML_TASKS.md) - Every ML task with AEGIS implementations
- [ML From Scratch](ML_FROM_SCRATCH.md) - Building ML algorithms from zero
- [AETHER Integration](ML_AETHER.md) - Deep dive into geometric intelligence
- [Examples](EXAMPLES.md) - Complete working examples
- [Tutorial](TUTORIAL.md) - Step-by-step guide
