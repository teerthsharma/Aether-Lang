# AEGIS API Reference

Complete Rust API documentation for the AEGIS kernel.

---

## Table of Contents

1. [Language Module (`lang`)](#language-module)
2. [ML Module (`ml`)](#ml-module)
3. [Manifold Module (`manifold`)](#manifold-module)
4. [AETHER Module (`aether`)](#aether-module)
5. [Topology Module (`topology`)](#topology-module)

---

## Language Module

### `lang::Lexer`

Tokenizes AEGIS source code.

```rust
use aegis::lang::Lexer;

let mut lexer = Lexer::new("manifold M = embed(data, dim=3)");
let tokens = lexer.tokenize();
```

**Methods:**
| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(source: &str) -> Lexer` | Create lexer from source |
| `next_token` | `fn next_token(&mut self) -> Token` | Get next token |
| `tokenize` | `fn tokenize(&mut self) -> Vec<Token>` | Tokenize entire source |

### `lang::TokenKind`

Token types in AEGIS:

```rust
pub enum TokenKind {
    // Keywords
    Manifold, Block, Regress, Render, Embed, Until, Escalate, Convergence,
    
    // Literals
    Identifier(String), Number(i64), Float(i64, i64), StringLit(String),
    
    // Punctuation
    Equals, Colon, Comma, Dot, LBrace, RBrace, LBracket, RBracket, LParen, RParen,
    
    // Special
    Newline, Eof, Error(String),
}
```

### `lang::Parser`

Parses token stream into AST.

```rust
use aegis::lang::Parser;

let mut parser = Parser::new(source);
let program = parser.parse()?;
```

**Methods:**
| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(source: &str) -> Parser` | Create parser |
| `parse` | `fn parse(&mut self) -> Result<Program, ParseError>` | Parse program |

### `lang::Interpreter`

Executes AEGIS programs.

```rust
use aegis::lang::{Parser, Interpreter};

let mut parser = Parser::new(source);
let program = parser.parse()?;

let mut interpreter = Interpreter::new();
let result = interpreter.execute(&program)?;
```

**Methods:**
| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new() -> Interpreter` | Create interpreter |
| `execute` | `fn execute(&mut self, prog: &Program) -> Result<Value, Error>` | Execute program |

---

## ML Module

### `ml::ManifoldRegressor<D>`

Non-linear regression on D-dimensional manifolds.

```rust
use aegis::ml::regressor::{ManifoldRegressor, ModelType};

let mut regressor: ManifoldRegressor<3> = ManifoldRegressor::new(ModelType::Linear);
regressor.add_point([0.0, 0.5, 0.25], 1.0);
regressor.add_point([0.1, 0.55, 0.28], 1.1);

let error = regressor.fit();
let prediction = regressor.predict(&[0.05, 0.52, 0.26]);
```

**Methods:**
| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(model: ModelType) -> Self` | Create regressor |
| `add_point` | `fn add_point(&mut self, point: [f64; D], target: f64)` | Add training data |
| `fit` | `fn fit(&mut self) -> f64` | Fit model, returns error |
| `predict` | `fn predict(&self, point: &[f64; D]) -> f64` | Predict value |
| `upgrade_model` | `fn upgrade_model(&mut self)` | Escalate to next complexity |
| `error` | `fn error(&self) -> f64` | Get current error |
| `coefficients` | `fn coefficients(&self) -> &Coefficients` | Get fitted coefficients |

### `ml::ModelType`

Available regression models:

```rust
pub enum ModelType {
    Linear,
    Polynomial(u8),           // degree
    Rbf { gamma: f64 },
    GaussianProcess { length_scale: f64 },
    GeodesicRegression,
}
```

**Complexity levels:**
| Model | Complexity |
|-------|------------|
| `Linear` | 1 |
| `Polynomial(d)` | 1 + d |
| `Rbf` | 5 |
| `GaussianProcess` | 7 |
| `GeodesicRegression` | 9 |

### `ml::ConvergenceDetector`

Detects topological convergence.

```rust
use aegis::ml::convergence::{ConvergenceDetector, BettiNumbers};

let mut detector = ConvergenceDetector::new(1e-6, 5);

// Record epoch metrics
detector.record_epoch(BettiNumbers::new(2, 1), 0.1, 0.05);
detector.record_epoch(BettiNumbers::new(1, 0), 0.01, 0.01);

if detector.is_converged() {
    println!("Converged! Score: {}", detector.convergence_score());
}
```

**Methods:**
| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(epsilon: f64, window: usize) -> Self` | Create detector |
| `record_epoch` | `fn record_epoch(&mut self, betti: BettiNumbers, drift: f64, error: f64)` | Record metrics |
| `is_converged` | `fn is_converged(&self) -> bool` | Check convergence |
| `convergence_score` | `fn convergence_score(&self) -> f64` | Score 0-1 |
| `reset` | `fn reset(&mut self)` | Reset detector |

### `ml::BettiNumbers`

Topological shape signature.

```rust
use aegis::ml::convergence::BettiNumbers;

let betti = BettiNumbers::new(1, 0);
assert!(betti.is_singular());  // Single component, no loops
```

**Methods:**
| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(beta_0: u32, beta_1: u32) -> Self` | Create Betti numbers |
| `is_singular` | `fn is_singular(&self) -> bool` | β₀=1, β₁=0 |
| `distance` | `fn distance(&self, other: &Self) -> u32` | L1 distance |

### `ml::EscalatingBenchmark<D>`

Auto-escalating benchmark runner.

```rust
use aegis::ml::benchmark::{EscalatingBenchmark, BenchmarkConfig};

let config = BenchmarkConfig {
    epsilon: 1e-6,
    max_epochs: 100,
    escalation_patience: 10,
    stability_window: 5,
    auto_escalate: true,
};

let mut benchmark: EscalatingBenchmark<3> = EscalatingBenchmark::new(config);
benchmark.add_data([0.0, 0.5, 0.25], 1.0);

let result = benchmark.run();
println!("Converged: {}, Epochs: {}", result.converged, result.epochs);
```

---

## Manifold Module

### `manifold::TimeDelayEmbedder<D>`

Implements Takens' theorem for time-delay embedding.

```rust
use aegis::manifold::TimeDelayEmbedder;

let mut embedder: TimeDelayEmbedder<3> = TimeDelayEmbedder::new(5);  // tau=5

embedder.push(1.0);
embedder.push(2.0);
// ... push more values

if let Some(point) = embedder.embed() {
    println!("Embedded point: {:?}", point.coords);
}
```

**Methods:**
| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(tau: usize) -> Self` | Create with time delay |
| `push` | `fn push(&mut self, value: f64)` | Add sample |
| `embed` | `fn embed(&self) -> Option<ManifoldPoint<D>>` | Get embedded point |
| `reset` | `fn reset(&mut self)` | Clear buffer |

### `manifold::ManifoldPoint<D>`

A point in D-dimensional manifold space.

```rust
use aegis::manifold::ManifoldPoint;

let p1 = ManifoldPoint::<3>::new([1.0, 2.0, 3.0]);
let p2 = ManifoldPoint::<3>::new([1.5, 2.5, 3.5]);

let dist = p1.distance(&p2);
let is_close = p1.is_neighbor(&p2, 1.0);
```

### `manifold::SparseAttentionGraph<D>`

Sparse attention using geometric locality.

```rust
use aegis::manifold::SparseAttentionGraph;

let mut graph: SparseAttentionGraph<3> = SparseAttentionGraph::new(0.5);

graph.add_point(ManifoldPoint::new([0.0, 0.0, 0.0]));
graph.add_point(ManifoldPoint::new([0.1, 0.1, 0.1]));

let (beta_0, beta_1) = graph.shape();
```

---

## AETHER Module

### `aether::BlockMetadata<D>`

Geometric metadata for a block of points.

```rust
use aegis::aether::BlockMetadata;

let points = vec![
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
];

let block = BlockMetadata::<3>::from_points(&points);
println!("Centroid: {:?}", block.centroid);
println!("Radius: {}", block.radius);
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `centroid` | `[f64; D]` | Block center |
| `radius` | `f64` | Max deviation |
| `variance` | `f64` | Point variance |
| `concentration` | `f64` | Angular concentration |
| `count` | `usize` | Number of points |

### `aether::HierarchicalBlockTree<D>`

Hierarchical tree for multi-scale analysis.

```rust
use aegis::aether::HierarchicalBlockTree;

let mut tree: HierarchicalBlockTree<3> = HierarchicalBlockTree::new();
tree.build_from_blocks(&blocks);

let active_mask = tree.hierarchical_query(&query, threshold);
let pruning = tree.pruning_ratio(&active_mask);
```

### `aether::DriftDetector<D>`

Track centroid drift for convergence detection.

```rust
use aegis::aether::DriftDetector;

let mut detector: DriftDetector<3> = DriftDetector::new();

let drift = detector.update(&[0.0, 0.0, 0.0]);
let drift = detector.update(&[0.01, 0.01, 0.01]);

if detector.is_drifting(0.1) {
    println!("Still drifting!");
}
```

---

## Topology Module

### `topology::BinaryTopology`

Compute topology of binary data.

```rust
use aegis::topology::BinaryTopology;

let data = [0u8; 64];
let topo = BinaryTopology::analyze(&data);

println!("β₀ = {}, β₁ = {}", topo.beta_0, topo.beta_1);
```

### `topology::TopologicalLoader`

Verify binary code via topological signature.

```rust
use aegis::loader::TopologicalLoader;

let loader = TopologicalLoader::new();
let result = loader.verify(binary_data, reference_shape);
```

---

## Error Types

### `lang::ParseError`

Parser error with location.

```rust
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub column: usize,
}
```

---

## Constants

```rust
// State dimension for kernel
pub const STATE_DIMENSION: usize = 4;

// Initial adaptive threshold
pub const INITIAL_EPSILON: f64 = 0.1;

// Maximum blocks
pub const MAX_BLOCKS: usize = 64;

// Maximum tree depth
pub const MAX_DEPTH: usize = 16;
```

---

## See Also

- [Language Reference](LANGUAGE.md)
- [Tutorial](TUTORIAL.md)
- [Examples](EXAMPLES.md)
- [Architecture](ARCHITECTURE.md)
