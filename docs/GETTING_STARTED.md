# Getting Started with AEGIS

Welcome to AEGIS, the 3D ML Language Kernel! This guide will help you get up and running quickly.

## Table of Contents

1. [Installation](#installation)
2. [Your First Script](#your-first-script)
3. [Understanding Manifolds](#understanding-manifolds)
4. [Running Regression](#running-regression)
5. [Next Steps](#next-steps)

---

## Installation

### Option 1: Docker (Recommended)

The easiest way to get started is with Docker:

```bash
# Pull the AEGIS image
docker pull teerthsharma/aegis

# Start the REPL
docker run -it teerthsharma/aegis repl
```

### Option 2: From Source

```bash
# Install Rust nightly
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup install nightly
rustup default nightly
rustup component add rust-src llvm-tools-preview

# Clone AEGIS
git clone https://github.com/teerthsharma/aegis.git
cd aegis

# Build
cargo build --release
```

---

## Your First Script

Create a file called `hello.aegis`:

```aegis
// hello.aegis - Your first AEGIS program

// Step 1: Create a 3D manifold from data
// dim=3 means 3D space
// tau=5 is the time delay for embedding
manifold M = embed(data, dim=3, tau=5)

// Step 2: Extract a geometric block (first 64 points)
block B = M.cluster(0:64)

// Step 3: Get the block's centroid (center point)
centroid C = B.center

// Step 4: Render the manifold
render M {
    color: by_density
}
```

Run it:

```bash
# With Docker
docker run -v $(pwd):/scripts teerthsharma/aegis run /scripts/hello.aegis

# Or in the REPL
aegis> load hello.aegis
```

**Expected output:**
```
[Lexing] Tokenizing script...
[Parsing] Building AST...
[Interpreting] Executing statements...

Execution Summary:
  Manifolds created: 1
  Blocks extracted:  1
  Regressions run:   0
  Renders:           1

✓ Completed successfully
```

---

## Understanding Manifolds

### What is a Manifold?

In AEGIS, a **manifold** is a 3D geometric space where your data lives. Instead of treating data as flat numbers, we embed it into 3D space where:

- **Similar points are close together**
- **Patterns become visible shapes**
- **Anomalies are geometrically distant**

### Time-Delay Embedding

AEGIS uses **Takens' theorem** to create manifolds from 1D time-series:

```
Original data:    [x₁, x₂, x₃, x₄, x₅, x₆, ...]

With tau=2:
Point 1: (x₁, x₃, x₅)
Point 2: (x₂, x₄, x₆)
...
```

This transforms temporal patterns into geometric shapes!

### Creating a Manifold

```aegis
// Basic manifold (3D, tau=1)
manifold M = embed(data, dim=3)

// Custom time delay
manifold M = embed(data, dim=3, tau=5)

// From specific data source
manifold M = embed(sensor_readings, dim=3, tau=7)
```

---

## Running Regression

### Basic Regression

```aegis
manifold M = embed(data, dim=3, tau=5)

// Simple polynomial regression
regress {
    model: "polynomial",
    degree: 3
}
```

### Escalating Regression

The magic of AEGIS is **escalating regression** - automatically increasing model complexity until convergence:

```aegis
manifold M = embed(data, dim=3, tau=5)

regress {
    model: "polynomial",
    degree: 2,
    escalate: true,              // Enable auto-escalation
    until: convergence(1e-6)     // Stop when error < 1e-6
}
```

**What happens:**
1. Starts with polynomial degree 2
2. If error doesn't converge, escalates to degree 3
3. Continues escalating: Poly → RBF → Gaussian Process
4. Stops when **topological convergence** is detected

### Model Types

| Model | Description | Complexity |
|-------|-------------|------------|
| `"linear"` | y = a + bx | O(1) |
| `"polynomial"` | y = Σaᵢxⁱ | O(d) |
| `"rbf"` | Radial Basis Function | O(n) |
| `"gp"` | Gaussian Process | O(n²) |
| `"geodesic"` | Manifold regression | O(n² log n) |

---

## Working with Blocks

### What is a Block?

A **block** is a geometric region of the manifold - a cluster of nearby points.

```aegis
manifold M = embed(data, dim=3, tau=5)

// Extract block by index range
block B = M[0:64]

// Or use cluster method
block B = M.cluster(0:64)
```

### Block Properties

```aegis
block B = M[0:64]

// Centroid (center point)
centroid C = B.center

// Radius (max distance from center)
radius R = B.spread

// Use in computations
render M {
    highlight: B  // Highlight this block
}
```

---

## Understanding Convergence

### What is Topological Convergence?

Instead of arbitrary loss thresholds, AEGIS detects convergence through **topology**:

1. **Betti Numbers (β₀, β₁)** - Count connected components and loops
2. **Centroid Drift** - How much the solution is moving
3. **Residual Collapse** - Whether errors are shrinking to a point

### Convergence Conditions

```aegis
// Error threshold
until: convergence(1e-6)

// Betti stability (advanced)
until: betti_stable(10)  // Stable for 10 epochs
```

### Reading Convergence Output

```
Epoch 1:  Linear          → Error: 0.15, β = (3, 1)
Epoch 5:  Polynomial(3)   → Error: 0.03, β = (2, 1)  ↑ escalate
Epoch 12: RBF             → Error: 0.008, β = (1, 0) ↑ escalate
Epoch 15: Converged!      → β stable, drift → 0 ✓

The Answer Has Come ✓
  Coefficients: [0.9987, -0.0234, 0.0012, ...]
```

---

## Next Steps

Now that you understand the basics:

1. **📖 Read the [Language Reference](LANGUAGE.md)** - Complete syntax guide
2. **💡 Explore [Examples](EXAMPLES.md)** - More complex scripts
3. **🔬 Study [Mathematics](MATHEMATICS.md)** - The theory behind AEGIS
4. **🏗️ Learn [Architecture](ARCHITECTURE.md)** - How AEGIS works internally
5. **🤝 [Contribute](../CONTRIBUTING.md)** - Help improve AEGIS!

---

## Common Issues

### "Command not found: aegis"

Make sure you're using Docker:
```bash
docker run -it teerthsharma/aegis repl
```

### "File not found"

Mount your directory when running scripts:
```bash
docker run -v $(pwd):/scripts teerthsharma/aegis run /scripts/yourfile.aegis
```

### "Convergence not reached"

Try increasing max epochs or using a more flexible model:
```aegis
regress {
    model: "rbf",
    escalate: true,
    until: convergence(1e-4)  // Less strict threshold
}
```

---

## Getting Help

- 📖 [Full Documentation](../README.md#-documentation)
- ❓ [FAQ](FAQ.md)
- 🐛 [Report Issues](https://github.com/teerthsharma/aegis/issues)
- 💬 [Discussions](https://github.com/teerthsharma/aegis/discussions)
