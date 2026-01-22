# AEGIS Tutorial

A step-by-step guide to mastering AEGIS, the 3D ML Language Kernel.

---

## Table of Contents

1. [Part 1: Basic Concepts](#part-1-basic-concepts)
2. [Part 2: Working with Data](#part-2-working-with-data)
3. [Part 3: Regression Techniques](#part-3-regression-techniques)
4. [Part 4: Advanced Topics](#part-4-advanced-topics)
5. [Part 5: Real-World Applications](#part-5-real-world-applications)

---

## Part 1: Basic Concepts

### 1.1 The Manifold Mental Model

Think of a manifold as a 3D sculpture of your data. Every data point becomes a location in 3D space, and patterns become geometric shapes you can see and manipulate.

```
Traditional View:        AEGIS View:
                        
  data = [1,2,3,4,5]        ●
                           ●  ●
                          ●    ●
                         (3D shape!)
```

### 1.2 Creating Your First Manifold

```aegis
// The embed() function transforms data into 3D space
manifold M = embed(data, dim=3, tau=5)
```

**Parameters:**
- `data`: Your input data source
- `dim`: Number of dimensions (usually 3)
- `tau`: Time delay for embedding (experiment with this!)

### 1.3 Understanding tau (τ)

The `tau` parameter controls how the embedding unfolds in time:

| tau | Effect |
|-----|--------|
| 1 | Adjacent points, fine detail |
| 5 | Medium spread, balanced |
| 10+ | Wide spread, global patterns |

**Pro tip:** Start with tau = 5 and adjust based on results.

---

## Part 2: Working with Data

### 2.1 Blocks as Regions

A block is a region of your manifold - think of it as selecting a piece of the 3D sculpture:

```aegis
manifold M = embed(data, dim=3, tau=5)

// Extract points 0-63
block early = M[0:64]

// Extract points 64-127
block middle = M[64:128]

// Extract points 128-191
block late = M[128:192]
```

### 2.2 Block Properties

Every block has geometric properties:

```aegis
block B = M[0:64]

// Center point of the block
centroid C = B.center

// How spread out the points are
radius R = B.spread
```

### 2.3 Comparing Blocks

```aegis
manifold M = embed(data, dim=3, tau=5)

block A = M[0:50]
block B = M[50:100]

// Centroids reveal cluster positions
centroid_A = A.center
centroid_B = B.center

// Spreads reveal cluster tightness
spread_A = A.spread
spread_B = B.spread

// Visualize both
render M {
    color: by_cluster,
    highlight: A
}
```

---

## Part 3: Regression Techniques

### 3.1 Simple Regression

Start with basic polynomial regression:

```aegis
manifold M = embed(data, dim=3, tau=5)

regress {
    model: "polynomial",
    degree: 3
}
```

### 3.2 Escalating Regression

Let AEGIS automatically find the right model complexity:

```aegis
manifold M = embed(data, dim=3, tau=5)

regress {
    model: "polynomial",
    degree: 2,
    escalate: true,
    until: convergence(1e-6)
}
```

**The escalation sequence:**
```
Linear → Poly(2) → Poly(3) → Poly(4) → RBF → GP → Geodesic
```

### 3.3 Choosing Models

| Model | Use When |
|-------|----------|
| `"linear"` | Data is roughly linear |
| `"polynomial"` | Curved but smooth |
| `"rbf"` | Complex local patterns |
| `"gp"` | Uncertainty quantification needed |
| `"geodesic"` | True manifold structure matters |

### 3.4 Convergence Strategies

```aegis
// Strict convergence
until: convergence(1e-8)

// Relaxed convergence (faster)
until: convergence(1e-4)

// Topology-based (most robust)
until: betti_stable(5)
```

---

## Part 4: Advanced Topics

### 4.1 Understanding Betti Numbers

Betti numbers describe the "shape" of data:

| Betti | Meaning | Example |
|-------|---------|---------|
| β₀ = 1 | One connected cluster | ● |
| β₀ = 3 | Three separate clusters | ●  ●  ● |
| β₁ = 1 | One loop/cycle | ○ |
| β₁ = 0 | No loops | ● |

**Perfect convergence:** β₀ = 1, β₁ = 0 (single point, no loops)

### 4.2 Monitoring Convergence

The convergence process shows:

```
Epoch  Model          Error    Betti    Action
─────────────────────────────────────────────
1      Linear         0.150    (3, 1)   
5      Polynomial(2)  0.080    (2, 1)   ↑ escalate
10     Polynomial(3)  0.030    (2, 0)   ↑ escalate
15     RBF            0.008    (1, 0)   
18     RBF            0.006    (1, 0)   β stable
20     Converged!     0.005    (1, 0)   ✓ done
```

### 4.3 Hierarchical Blocks

Create hierarchies for multi-scale analysis:

```aegis
manifold M = embed(data, dim=3, tau=5)

// Fine level (64 points)
block fine = M[0:64]

// Medium level (256 points)
block medium = M[0:256]

// Coarse level (1024 points)
block coarse = M[0:1024]

// Compare centroids at different scales
c_fine = fine.center
c_medium = medium.center
c_coarse = coarse.center
```

---

## Part 5: Real-World Applications

### 5.1 Anomaly Detection

Anomalies are geometrically distant from cluster centroids:

```aegis
manifold M = embed(sensor_data, dim=3, tau=10)

// Normal behavior block
block normal = M[0:1000]
normal_center = normal.center
normal_spread = normal.spread

// Check new data
block test = M[1000:1100]
test_center = test.center

// Anomaly if test_center is far from normal_center
// (distance > 2 * normal_spread suggests anomaly)
```

### 5.2 Time Series Forecasting

```aegis
manifold M = embed(historical_data, dim=3, tau=7)

regress {
    model: "gp",
    target: M.project(axis=0),
    escalate: true,
    until: convergence(1e-5)
}

// Coefficients can be used for prediction
```

### 5.3 Pattern Recognition

```aegis
manifold M = embed(signal_data, dim=3, tau=5)

// Extract known patterns
block pattern_A = M[0:100]
block pattern_B = M[100:200]

// Compare to new signal
block unknown = M[200:300]

// Match based on centroid distance and spread similarity
```

### 5.4 Dimensionality Reduction

```aegis
// Embed high-dimensional data into 3D for visualization
manifold M = embed(high_dim_data, dim=3, tau=3)

render M {
    color: by_density,
    trajectory: on
}

// The 3D manifold reveals structure invisible in high dimensions
```

---

## Exercises

### Exercise 1: Hello Manifold
Create a manifold, extract a block, and render it.

### Exercise 2: Compare Clusters
Extract 3 blocks and compare their centroids and spreads.

### Exercise 3: Escalate to Convergence
Run escalating regression on sine wave data until convergence.

### Exercise 4: Anomaly Detection
Create a normal block and detect an anomalous test block.

---

## Next Steps

- 📖 [Language Reference](LANGUAGE.md) - Complete syntax
- 📊 [Examples](EXAMPLES.md) - More code samples
- 🔬 [Mathematics](MATHEMATICS.md) - Theory deep-dive
- 🏗️ [Architecture](ARCHITECTURE.md) - Internals
