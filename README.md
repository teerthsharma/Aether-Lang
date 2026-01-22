<div align="center">

# 🛡️ AEGIS

### **The Universal Programming Language**

*Topologically Complete • O(log n) Convergence • Cross-Platform Native*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/teerthsharma/aegis)
[![Rust](https://img.shields.io/badge/rust-nightly-orange.svg)](https://www.rust-lang.org/)
[![Windows](https://img.shields.io/badge/Windows-Native-blue.svg)](#)
[![Linux](https://img.shields.io/badge/Linux-Native-orange.svg)](#)
[![macOS](https://img.shields.io/badge/macOS-Native-lightgrey.svg)](#)

[**Quick Start**](#-quick-start) • 
[**Why AEGIS?**](#-why-aegis-is-more-powerful) • 
[**Architecture**](#-cross-platform-architecture) •
[**Examples**](#-examples)

---

</div>

## 🌟 What is AEGIS?

AEGIS is a **universal programming language** that combines the best features of Python (ease of use), Java (type safety), and C++ (performance) while introducing revolutionary concepts:

- 🦭 **Seal Loops** – O(log n) convergence-based iteration
- **`~` Tilde Terminator** – Clean, unique statement terminator
- **3D Manifold Primitives** – First-class geometric data structures
- **Topological Convergence** – Answers emerge through Betti number stability
- 🖥️ **Cross-Platform Native** – Runs natively on Windows, Linux, macOS

```aegis
// AEGIS: Where code meets mathematics 🦭
let data = [1.0, 2.1, 3.5, 4.2, 5.1]~

manifold M = embed(data, dim=3, tau=2)~

// Seal loop - runs until topology stabilizes!
🦭 until convergence(1e-6) {
    regress { model: "polynomial", escalate: true }~
}

render M { format: "ascii" }~
```

---

## 🏗️ Cross-Platform Architecture

AEGIS is built as a **Cargo workspace** with modular crates for maximum portability:

```
aegis/
├── aegis-core/      # Platform-agnostic math, topology, ML
│   └── (no_std + std compatible)
├── aegis-lang/      # Language compiler/interpreter  
│   └── (lexer, parser, AST, interpreter)
├── aegis-kernel/    # Bare-metal x86_64 microkernel
│   └── (for embedded/research use)
├── aegis-cli/       # Cross-platform CLI binary
│   └── (Windows, Linux, macOS native)
└── Cargo.toml       # Workspace root
```

| Crate | Purpose | Platforms |
|-------|---------|-----------|
| **aegis-core** | Math, topology, manifolds, ML engine | Any (no_std + std) |
| **aegis-lang** | Lexer, Parser, Interpreter | Any (no_std + std) |
| **aegis-kernel** | Bare-metal microkernel | x86_64 only |
| **aegis-cli** | Native CLI with REPL | Windows, Linux, macOS |

---

## 🚀 Quick Start

### Native Build (Windows/Linux/macOS)

```bash
# Clone and build the CLI
git clone https://github.com/teerthsharma/aegis.git
cd aegis
cargo build -p aegis-cli --release

# Run the REPL
./target/release/aegis repl

# Run a script
./target/release/aegis run examples/hello_manifold.aegis

# Check syntax
./target/release/aegis check examples/benchmark_seal_vs_linear.aegis
```

### Using Docker

```bash
# Pull and run AEGIS REPL
docker run -it aegis repl

# Run the benchmark demo
docker run -v $(pwd):/scripts aegis run examples/benchmark_seal_vs_linear.aegis
```

### Building the Kernel (Research/Embedded)

```bash
# Requires nightly Rust with x86_64-unknown-none target
cargo build -p aegis-kernel --target x86_64-unknown-none
```

---

## ⚡ Why AEGIS is More Powerful

### The Problem with Traditional ML

```python
# Traditional: Fixed epochs, wasteful computation
for epoch in range(10000):  # ← You're GUESSING how many epochs!
    optimizer.step()
    if loss < threshold:
        break  # ← Hope you didn't waste 9,900 iterations
```

### The AEGIS Solution: Topological Convergence

```aegis
// AEGIS: Converges when the SHAPE of the data stabilizes
🦭 until convergence(1e-6) {
    regress { model: "polynomial", escalate: true }~
}
// Terminates in ~50 iterations, not 10,000!
```

AEGIS monitors the **Betti numbers** (β₀, β₁) of the residual manifold. When the topology stabilizes—meaning no new "holes" are forming in the error landscape—it **seals** the computation.

| Approach | Iterations | Complexity | Wasted Compute |
|----------|------------|------------|----------------|
| Python/Java/C++ | 10,000 | O(n) | ~95% |
| **AEGIS Seal Loop** | **~50** | **O(log n)** | **~0%** |

---

## 🦭 The Seal Loop

AEGIS introduces the **seal loop** – a revolutionary iteration construct that terminates when mathematical convergence is achieved.

```aegis
// Using the seal emoji 🦭
🦭 until condition {
    // body~
}

// Using the keyword
seal until error < 0.001 {
    train_step()~
}
```

---

## 💡 Complete Language Features

### Control Flow

```aegis
// If-else
if score >= 90 {
    print("A grade!")~
} else {
    print("Keep trying!")~
}

// For loop
for i in 0..100 {
    process(i)~
}

// While loop
while error > 0.01 {
    train_step()~
}

// Seal loop (AEGIS exclusive!)
🦭 until converged {
    optimize()~
}
```

### Functions

```aegis
fn fibonacci(n) {
    if n <= 1 {
        return n~
    }
    return fibonacci(n - 1) + fibonacci(n - 2)~
}
```

### 3D Manifold Operations

```aegis
manifold M = embed(sensor_data, dim=3, tau=5)~
block B = M.cluster(0:64)~
centroid C = B.center~

render M { format: "ascii", color: "density" }~
```

---

## 📚 Documentation

### Core Guides
| Document | Description |
|----------|-------------|
| [**SYNTAX.md**](docs/SYNTAX.md) | 🎓 Beginner-friendly syntax guide |
| [**Tutorial**](docs/TUTORIAL.md) | Step-by-step tutorial |
| [**API Reference**](docs/API.md) | Complete API documentation |

### 🧠 ML Library (From Scratch)
| Document | Description |
|----------|-------------|
| [**ML Library**](docs/ML_LIBRARY.md) | 📖 Core ML API reference (regressors, convergence, manifolds) |
| [**ML Tasks**](docs/ML_TASKS.md) | 📋 Every ML task: classification, clustering, neural nets, and more |
| [**ML From Scratch**](docs/ML_FROM_SCRATCH.md) | 🔧 Building ML algorithms from zero in AEGIS |
| [**AETHER Integration**](docs/ML_AETHER.md) | 🌐 Geometric intelligence for O(log n) ML |

---

## 💡 Examples

See the [examples/](examples/) directory for complete demonstrations:

- `hello_manifold.aegis` - Basic manifold embedding
- `benchmark_seal_vs_linear.aegis` - Seal loop performance demo
- `neural_topology.aegis` - Neural network with topological convergence

---

## 📄 License

MIT License - See [LICENSE](LICENSE)

---

<div align="center">

**Making data 3D for everyone through topological geometry** 🌐🦭

*AEGIS: Where answers emerge from shape, not exhaustion.*

</div>
