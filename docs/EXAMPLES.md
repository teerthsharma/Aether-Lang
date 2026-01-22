# AEGIS Examples

A collection of annotated AEGIS scripts for common use cases.

---

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Regression Examples](#regression-examples)
3. [Visualization Examples](#visualization-examples)
4. [Advanced Examples](#advanced-examples)
5. [Complete Applications](#complete-applications)

---

## Basic Examples

### Example 1: Hello World

The simplest AEGIS program:

```aegis
// hello_world.aegis
// Create a manifold and render it

manifold M = embed(data, dim=3, tau=5)
render M
```

### Example 2: Block Extraction

Extract and analyze a geometric block:

```aegis
// block_basics.aegis

manifold M = embed(data, dim=3, tau=5)

// Extract first 64 points as a block
block B = M.cluster(0:64)

// Get geometric properties
centroid C = B.center    // Center point
radius R = B.spread      // Max distance from center

// Render with block highlighted
render M {
    highlight: B
}
```

### Example 3: Multiple Blocks

Compare different regions of the manifold:

```aegis
// multiple_blocks.aegis

manifold M = embed(time_series, dim=3, tau=7)

// Extract three time periods
block morning = M[0:100]
block afternoon = M[100:200]
block evening = M[200:300]

// Get centers
c1 = morning.center
c2 = afternoon.center
c3 = evening.center

render M {
    color: by_cluster
}
```

---

## Regression Examples

### Example 4: Simple Polynomial

Fit a polynomial to data:

```aegis
// polynomial_fit.aegis

manifold M = embed(measurements, dim=3, tau=5)

regress {
    model: "polynomial",
    degree: 3
}
```

### Example 5: Escalating Regression

Automatically find the right model:

```aegis
// escalating.aegis

manifold M = embed(sensor_data, dim=3, tau=7)

// Start simple, escalate until convergence
regress {
    model: "linear",       // Start with linear
    escalate: true,        // Auto-escalate
    until: convergence(1e-6)
}

// Output will show:
// Linear → Poly(2) → Poly(3) → RBF → Converged!
```

### Example 6: RBF Kernel Regression

For complex non-linear patterns:

```aegis
// rbf_regression.aegis

manifold M = embed(complex_data, dim=3, tau=10)

regress {
    model: "rbf",
    escalate: true,
    until: convergence(1e-5)
}
```

### Example 7: Gaussian Process

With uncertainty quantification:

```aegis
// gp_regression.aegis

manifold M = embed(noisy_data, dim=3, tau=5)

regress {
    model: "gp",
    until: convergence(1e-4)
}
```

---

## Visualization Examples

### Example 8: Density Coloring

Color by point density:

```aegis
// density_viz.aegis

manifold M = embed(data, dim=3, tau=5)

render M {
    color: by_density
}
```

### Example 9: Cluster Coloring

Color by cluster assignment:

```aegis
// cluster_viz.aegis

manifold M = embed(data, dim=3, tau=5)

// Define clusters
block A = M[0:100]
block B = M[100:200]

render M {
    color: by_cluster,
    highlight: A
}
```

### Example 10: Trajectory Visualization

Show time evolution:

```aegis
// trajectory_viz.aegis

manifold M = embed(time_series, dim=3, tau=5)

render M {
    color: gradient,
    trajectory: on    // Show temporal path
}
```

### Example 11: Axis Projection

Project to 2D for specific views:

```aegis
// projection_viz.aegis

manifold M = embed(data, dim=3, tau=5)

// X-Y projection (axis=2 for Z projection)
render M {
    color: by_density,
    axis: 2
}
```

---

## Advanced Examples

### Example 12: Anomaly Detection

Detect outliers geometrically:

```aegis
// anomaly_detection.aegis

manifold M = embed(sensor_readings, dim=3, tau=10)

// Normal operating range
block normal = M[0:500]
normal_center = normal.center
normal_radius = normal.spread

// Check recent data
block recent = M[500:550]
recent_center = recent.center

// If distance(recent_center, normal_center) > 2*normal_radius
// then we have an anomaly

render M {
    highlight: recent,
    color: by_density
}
```

### Example 13: Change Point Detection

Find where patterns change:

```aegis
// change_detection.aegis

manifold M = embed(process_data, dim=3, tau=5)

// Sliding window analysis
block w1 = M[0:50]
block w2 = M[50:100]
block w3 = M[100:150]
block w4 = M[150:200]

// Compute centroids
c1 = w1.center
c2 = w2.center
c3 = w3.center
c4 = w4.center

// Large centroid jumps indicate change points
render M {
    trajectory: on
}
```

### Example 14: Multi-Scale Analysis

Analyze at multiple granularities:

```aegis
// multiscale.aegis

manifold M = embed(data, dim=3, tau=5)

// Fine scale (16 point blocks)
block fine_1 = M[0:16]
block fine_2 = M[16:32]
block fine_3 = M[32:48]
block fine_4 = M[48:64]

// Medium scale (64 point blocks)
block medium = M[0:64]

// Coarse scale (256 point blocks)
block coarse = M[0:256]

// Compare hierarchical centroids
```

### Example 15: Streaming Analysis

Process data in windows:

```aegis
// streaming.aegis

// Initialize with historical data
manifold M = embed(historical, dim=3, tau=5)

// Reference block
block reference = M[0:100]
ref_center = reference.center

// As new data arrives, compare to reference
// (In practice, this would be in a loop)
block window = M[100:150]
window_center = window.center
```

---

## Complete Applications

### Application 1: Sensor Fusion

Fuse multiple sensor streams:

```aegis
// sensor_fusion.aegis
// Combine temperature, pressure, and humidity sensors

manifold M = embed(fused_sensors, dim=3, tau=7)

// Normal operating envelope
block normal_ops = M[0:200]

// Current state
block current = M[200:210]

// Check if current is within normal envelope
regress {
    model: "rbf",
    escalate: true,
    until: convergence(1e-4)
}

render M {
    color: by_density,
    highlight: current
}
```

### Application 2: Predictive Maintenance

Predict equipment failures:

```aegis
// predictive_maintenance.aegis

manifold M = embed(vibration_data, dim=3, tau=10)

// Known good state
block healthy = M[0:500]
healthy_center = healthy.center
healthy_spread = healthy.spread

// Known failure precursor
block pre_failure = M[500:600]

// Current readings
block current = M[600:650]
current_center = current.center

// If current is closer to pre_failure than healthy
// schedule maintenance!

render M {
    color: by_cluster,
    trajectory: on
}
```

### Application 3: Financial Analysis

Detect market regime changes:

```aegis
// market_regimes.aegis

manifold M = embed(price_returns, dim=3, tau=5)

// Bull market period
block bull = M[0:250]

// Bear market period
block bear = M[250:500]

// Current market
block now = M[500:520]

// Classify current regime based on proximity
regress {
    model: "gp",
    escalate: true,
    until: convergence(1e-5)
}

render M {
    color: by_cluster
}
```

### Application 4: Signal Processing

Denoise and analyze signals:

```aegis
// signal_processing.aegis

manifold M = embed(noisy_signal, dim=3, tau=7)

// Regression smooths the manifold
regress {
    model: "polynomial",
    degree: 5,
    escalate: true,
    until: convergence(1e-6)
}

// The fitted coefficients represent the denoised signal

render M {
    color: gradient,
    trajectory: on
}
```

---

## Running Examples

### With Docker

```bash
# Run any example
docker run -v $(pwd)/examples:/scripts teerthsharma/aegis run /scripts/hello_world.aegis
```

### With REPL

```bash
docker run -it teerthsharma/aegis repl
aegis> load examples/escalating.aegis
```

---

## Contributing Examples

Have a cool AEGIS script? Contribute it!

1. Fork the repo
2. Add your script to `examples/`
3. Add documentation here
4. Submit a PR

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
