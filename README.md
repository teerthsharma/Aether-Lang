# AEGIS 🛡️

**A 3D ML Language Kernel for Manifold-Native Machine Learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Rust](https://img.shields.io/badge/rust-nightly-orange.svg)](https://www.rust-lang.org/)

<p align="center">
  <img src="docs/aegis-logo.svg" alt="AEGIS Logo" width="200"/>
</p>

> *A domain-specific language where code exists in 3D geometric space, regression benchmarks escalate until convergence, and answers emerge from topological structure.*

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **3D Manifold Primitives** | Code in 3D geometric space using `manifold`, `block`, `regress` |
| **Escalating Benchmarks** | Regression automatically increases complexity until perfect |
| **Topological Convergence** | "Answers come" via Betti number stability |
| **Docker Ready** | Full REPL and CLI in containerized environment |

---

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/aegis.git
cd aegis

# Start the REPL
docker-compose up aegis

# Run a script
docker-compose run aegis run examples/hello_manifold.aegis

# Run benchmarks
docker-compose run aegis benchmark
```

### Using Cargo

```bash
# Install Rust nightly
rustup install nightly
rustup default nightly

# Build
cargo build --release

# Run tests
cargo test --lib
```

---

## 📝 AEGIS Language

### Example Script

```aegis
// Create a 3D manifold from time-series data
manifold M = embed(data, dim=3, tau=5)

// Extract geometric blocks
block B = M.cluster(0:64)
centroid C = B.center

// Run escalating regression
regress {
    model: "polynomial",
    degree: 3,
    escalate: true,
    until: convergence(1e-6)
}

// Visualize in 3D
render M {
    color: by_density,
    highlight: B
}
```

### Core Constructs

| Construct | Syntax | Description |
|-----------|--------|-------------|
| `manifold` | `manifold M = embed(data, dim=3)` | Create 3D embedded space |
| `block` | `block B = M.cluster(0:64)` | Extract geometric region |
| `regress` | `regress { model: "rbf", escalate: true }` | Run ML regression |
| `render` | `render M { color: gradient }` | Visualize manifold |

---

## 🧠 How "Answers Come"

AEGIS uses **topological convergence** instead of arbitrary loss thresholds:

```
Epoch 1:  Linear          → Error: 0.15, β = (3, 1)
Epoch 5:  Polynomial(3)   → Error: 0.03, β = (2, 1)  ↑ escalate
Epoch 12: RBF             → Error: 0.008, β = (1, 0) ↑ escalate
Epoch 15: Converged!      → β stable, drift → 0 ✓
```

**Convergence is detected when:**
1. Betti numbers (β₀, β₁) stabilize
2. Centroid drift approaches zero
3. Residual manifold collapses to a point

---

## 📐 Mathematical Foundation

### Time-Delay Embedding (Takens' Theorem)

```
Φ(t) = [x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ)]
```

### Topological Shape Signature

```
Shape(B) = (β₀, β₁)  where β₀ = connected components, β₁ = loops
```

### Escalating Models

| Level | Model | Complexity |
|-------|-------|------------|
| 1 | Linear | O(1) |
| 2 | Polynomial(d) | O(d) |
| 3 | RBF Kernel | O(n) |
| 4 | Gaussian Process | O(n²) |
| 5 | Geodesic Regression | O(n² log n) |

---

## 🐳 Docker Commands

```bash
# Build image
docker build -t aegis .

# Interactive REPL
docker run -it aegis repl

# Run script
docker run -v $(pwd)/examples:/aegis/examples aegis run /aegis/examples/hello_manifold.aegis

# Run benchmarks
docker run aegis benchmark

# Development shell
docker-compose run dev
```

---

## 📁 Project Structure

```
aegis/
├── src/
│   ├── lang/           # AEGIS DSL
│   │   ├── lexer.rs    # Tokenizer
│   │   ├── parser.rs   # AST generator
│   │   └── interpreter.rs
│   ├── ml/             # ML Engine
│   │   ├── regressor.rs
│   │   ├── convergence.rs
│   │   └── benchmark.rs
│   └── [kernel modules]
├── examples/           # .aegis scripts
├── docker/             # Docker CLI
├── docs/               # Documentation
├── Dockerfile
└── docker-compose.yml
```

---

## 🔬 Research References

- **Takens' Theorem**: Time-delay embedding for attractor reconstruction
- **Persistent Homology**: Topological data analysis for shape detection
- **AETHER**: DOI: 10.13141/RG.2.2.14811.27684

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
docker-compose run dev
cargo test
cargo fmt
cargo clippy
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE)

---

<p align="center">
  <b>Making data 3D for everyone through topological geometry</b> 🌐
</p>
