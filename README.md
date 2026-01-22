<div align="center">

# 🛡️ AEGIS

### **The 3D ML Language Kernel**

*Manifold-Native Machine Learning Where Code Exists in Geometric Space*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/teerthsharma/aegis)
[![Rust](https://img.shields.io/badge/rust-nightly-orange.svg)](https://www.rust-lang.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[**Quick Start**](#-quick-start) • 
[**Documentation**](#-documentation) • 
[**Examples**](#-examples) • 
[**Contributing**](#-contributing)

---

</div>

## 🌟 What is AEGIS?

AEGIS is a **domain-specific language** designed for machine learning on geometric manifolds. Unlike traditional ML frameworks that treat data as flat tensors, AEGIS embeds data into **3D geometric space** where patterns become visible shapes.

```aegis
// Create a 3D manifold from time-series data
manifold M = embed(sensor_data, dim=3, tau=5)

// Run escalating regression until convergence
regress {
    model: "polynomial",
    escalate: true,
    until: convergence(1e-6)
}

// The answer comes through topology
render M { color: by_density }
```

### 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **3D Manifold Primitives** | `manifold`, `block`, `regress`, `render` - code in geometric space |
| **Escalating Benchmarks** | Regression automatically increases complexity until perfect |
| **Topological Convergence** | "Answers come" via Betti number stability, not arbitrary thresholds |
| **Docker Ready** | Full REPL and CLI in containerized environment |
| **Bare-Metal Capable** | Runs as a microkernel on x86_64 hardware |

---

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Pull and run AEGIS REPL
docker run -it teerthsharma/aegis repl

# Run a script
docker run -v $(pwd):/scripts teerthsharma/aegis run /scripts/my_script.aegis

# Run escalating benchmarks
docker run teerthsharma/aegis benchmark
```

### Using Docker Compose

```bash
git clone https://github.com/teerthsharma/aegis.git
cd aegis

# Start interactive REPL
docker-compose up aegis

# Run benchmarks
docker-compose run benchmark
```

### Building from Source

```bash
# Install Rust nightly
rustup install nightly
rustup default nightly
rustup component add rust-src llvm-tools-preview

# Clone and build
git clone https://github.com/teerthsharma/aegis.git
cd aegis
cargo build --release

# Run tests
cargo test --lib
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [**Getting Started**](docs/GETTING_STARTED.md) | First steps with AEGIS |
| [**Language Reference**](docs/LANGUAGE.md) | Complete syntax and grammar |
| [**API Reference**](docs/API.md) | Rust API documentation |
| [**Tutorial**](docs/TUTORIAL.md) | Step-by-step tutorial |
| [**Examples**](docs/EXAMPLES.md) | Annotated code examples |
| [**Mathematical Foundations**](docs/MATHEMATICS.md) | The math behind AEGIS |
| [**Architecture**](docs/ARCHITECTURE.md) | System design and internals |
| [**FAQ**](docs/FAQ.md) | Frequently asked questions |
| [**Changelog**](CHANGELOG.md) | Version history |

---

## 📐 The AEGIS Philosophy

### "The Answer Comes Through Topology"

Traditional ML uses arbitrary loss thresholds to decide when to stop training. AEGIS uses **topological convergence**:

```
Epoch 1:  Linear          → Error: 0.15, β = (3, 1)
Epoch 5:  Polynomial(3)   → Error: 0.03, β = (2, 1)  ↑ escalate
Epoch 12: RBF             → Error: 0.008, β = (1, 0) ↑ escalate  
Epoch 15: Converged!      → β stable, drift → 0 ✓
```

**Convergence is detected when:**
1. 🔄 Betti numbers (β₀, β₁) stabilize
2. 📉 Centroid drift approaches zero
3. 🎯 Residual manifold collapses to a point

### Why 3D?

Takens' theorem proves that for many dynamical systems, embedding in 3D is sufficient to reconstruct the attractor. This means:

- **Non-linear relationships become visible** as geometric shapes
- **Clustering is natural** - nearby points in manifold space are related
- **Anomalies stand out** - they're geometrically distant

---

## 💡 Examples

### Hello Manifold

```aegis
// Embed time-series into 3D manifold
manifold M = embed(data, dim=3, tau=5)

// Extract a geometric block
block B = M.cluster(0:64)

// Get block properties
centroid C = B.center
radius R = B.spread

// Visualize
render M {
    color: by_density,
    highlight: B
}
```

### Escalating Regression

```aegis
manifold M = embed(sensor_data, dim=3, tau=7)

// Run regression that automatically escalates
// Linear → Polynomial → RBF → GP → Geodesic
regress {
    model: "polynomial",
    degree: 2,
    escalate: true,
    until: convergence(1e-8)
}
```

### Cluster Analysis

```aegis
manifold M = embed(data, dim=3, tau=5)

// Extract multiple blocks
block A = M[0:50]
block B = M[50:100]
block C = M[100:150]

// Compare centroids
dist_AB = distance(A.center, B.center)
dist_BC = distance(B.center, C.center)

render M { color: by_cluster }
```

📖 See [more examples →](docs/EXAMPLES.md)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AEGIS Language Kernel                    │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: AEGIS DSL                                         │
│  ├── Lexer (tokenize .aegis files)                          │
│  ├── Parser (recursive descent, AST generation)             │
│  └── Interpreter (manifold workspace, regression engine)    │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: ML Engine                                         │
│  ├── ManifoldRegressor (Linear, Poly, RBF, GP, Geodesic)    │
│  ├── EscalatingBenchmark (auto complexity increase)         │
│  └── TopologicalConvergence (Betti stability detection)     │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Geometric Primitives                              │
│  ├── TimeDelayEmbedder (Takens' theorem)                    │
│  ├── BlockMetadata (centroid, radius, variance)             │
│  └── HierarchicalBlockTree (AETHER extensions)              │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Topological Analysis                              │
│  ├── BettiNumbers (β₀ connected components, β₁ loops)       │
│  ├── DriftDetector (centroid trajectory tracking)           │
│  └── ResidualAnalyzer (topology of residuals)               │
├─────────────────────────────────────────────────────────────┤
│  Layer 0: Sparse-Event Microkernel                          │
│  ├── GeometricGovernor (PID-on-manifold)                    │
│  ├── SparseScheduler (execute only when Δ ≥ ε)              │
│  └── TopologyLoader (ELF verification via Betti)            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmarks

AEGIS includes a rigorous benchmark suite:

```bash
# Run all benchmarks
cargo test --lib benchmarks

# Run performance benchmarks (slower)
cargo test benchmark_performance -- --ignored --nocapture
```

| Benchmark | Description | Status |
|-----------|-------------|--------|
| Linear Regression | Basic linear fit | ✅ |
| Polynomial Regression | Polynomial fit accuracy | ✅ |
| Escalating Convergence | Auto-escalation test | ✅ |
| Betti Stability | Topology convergence | ✅ |
| All Test Functions | Sine, Poly, Exp, Mixture | ✅ |
| Model Upgrade Sequence | L→P→RBF→GP→Geo | ✅ |
| Large Dataset Performance | 256 point stress test | ✅ |

---

## 🐳 Docker

### Available Commands

```bash
# Interactive REPL
docker run -it teerthsharma/aegis repl

# Run a script
docker run -v $(pwd):/data teerthsharma/aegis run /data/script.aegis

# Benchmark suite
docker run teerthsharma/aegis benchmark

# Version info
docker run teerthsharma/aegis version

# Help
docker run teerthsharma/aegis --help
```

### Docker Compose Services

| Service | Purpose | Command |
|---------|---------|---------|
| `aegis` | Interactive REPL | `docker-compose up aegis` |
| `benchmark` | Full benchmark suite | `docker-compose run benchmark` |
| `dev` | Development shell | `docker-compose run dev` |

---

## 🔬 Research Background

AEGIS is built on solid mathematical foundations:

- **Takens' Theorem (1981)** - Time-delay embedding for attractor reconstruction
- **Persistent Homology** - Topological data analysis for shape detection
- **AETHER Framework** - Geometric sparse attention (DOI: 10.13141/RG.2.2.14811.27684)
- **PID Control Theory** - Adaptive threshold control on manifolds

### Publications

1. Takens, F. (1981). *Detecting strange attractors in turbulence*
2. Edelsbrunner, H. & Harer, J. (2010). *Computational Topology*
3. AETHER Geometric Extensions. DOI: 10.13141/RG.2.2.14811.27684

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/teerthsharma/aegis.git
cd aegis
docker-compose run dev

# Run tests
cargo test --lib

# Format code
cargo fmt

# Lint
cargo clippy
```

### Good First Issues

- [ ] Add more test functions to benchmark suite
- [ ] Implement ASCII visualization for `render`
- [ ] Add WebGL export for 3D visualization
- [ ] Write more example scripts

---

## 📄 License

MIT License - See [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- The Rust community for excellent no_std support
- TDA researchers for persistent homology foundations
- AETHER project for geometric sparse attention

---

<div align="center">

**Making data 3D for everyone through topological geometry** 🌐

[Report Bug](https://github.com/teerthsharma/aegis/issues) • 
[Request Feature](https://github.com/teerthsharma/aegis/issues) • 
[Discussions](https://github.com/teerthsharma/aegis/discussions)

</div>
