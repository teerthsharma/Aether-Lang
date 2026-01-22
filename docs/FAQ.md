# Frequently Asked Questions (FAQ)

Common questions about AEGIS, the 3D ML Language Kernel.

---

## General Questions

### What is AEGIS?

AEGIS is a **domain-specific language** for machine learning on geometric manifolds. It embeds data into 3D space where patterns become visible shapes, and uses topological methods to detect when models have truly converged.

### Why "AEGIS"?

AEGIS means "shield" in Greek mythology. The name reflects the project's origins as a security-focused microkernel with topological code authentication. It has evolved into a full ML language while retaining its geometric foundations.

### Is AEGIS a replacement for Python/TensorFlow/PyTorch?

No. AEGIS is complementary - it excels at:
- Non-linear pattern detection
- Anomaly detection via geometry
- When you need to "see" your data in 3D
- Topological convergence (not arbitrary loss thresholds)

Use traditional ML frameworks for standard supervised/unsupervised learning.

### What's "topological convergence"?

Instead of stopping when `loss < threshold`, AEGIS stops when the **shape** of the residuals stabilizes. This is measured via Betti numbers (β₀ = connected components, β₁ = loops). When β stabilizes and drift → 0, we've truly converged.

---

## Installation Questions

### How do I install AEGIS?

**Docker (recommended):**
```bash
docker pull teerthsharma/aegis
docker run -it teerthsharma/aegis repl
```

**From source:**
```bash
rustup install nightly
git clone https://github.com/teerthsharma/aegis.git
cd aegis && cargo build --release
```

### Do I need Rust installed?

Only if building from source. Docker users don't need Rust.

### What platforms are supported?

- **Docker**: Linux, macOS, Windows (via Docker Desktop)
- **Native**: Linux, macOS, Windows (with nightly Rust)

### Why nightly Rust?

AEGIS uses unstable features:
- `abi_x86_interrupt` for interrupt handlers
- `no_std` / `no_main` for bare-metal operation
- `build-std` for custom target compilation

---

## Language Questions

### What's the file extension for AEGIS scripts?

`.aegis` - for example: `my_script.aegis`

### How do I run an AEGIS script?

```bash
# Docker
docker run -v $(pwd):/scripts teerthsharma/aegis run /scripts/script.aegis

# REPL
aegis> load script.aegis
```

### What does `tau` mean?

`tau` (τ) is the **time delay** for Takens embedding. It controls how far apart samples are in the embedding:

```aegis
// tau=1: adjacent samples
manifold M = embed(data, dim=3, tau=1)

// tau=5: samples 5 apart (smoother patterns)
manifold M = embed(data, dim=3, tau=5)
```

**Rule of thumb:** Start with tau=5, increase for smoother patterns.

### What does `dim=3` mean?

The embedding dimension. AEGIS typically uses 3D because:
1. Takens' theorem shows 3D is sufficient for many systems
2. 3D is visualizable (humans can understand it)
3. Higher dimensions add computation without much benefit

### What's a "block"?

A block is a geometric region of the manifold - a subset of points. Think of it as selecting a piece of a 3D point cloud:

```aegis
block B = M[0:64]     // Points 0-63
block B = M.cluster(0:64)  // Same thing
```

---

## Regression Questions

### What regression models are available?

| Model | Syntax | Best For |
|-------|--------|----------|
| Linear | `"linear"` | Simple trends |
| Polynomial | `"polynomial"` + `degree` | Smooth curves |
| RBF | `"rbf"` | Complex local patterns |
| Gaussian Process | `"gp"` | Uncertainty estimation |
| Geodesic | `"geodesic"` | True manifold structure |

### What does "escalate" do?

When `escalate: true`, AEGIS automatically increases model complexity if the current model doesn't converge:

```
Linear → Poly(2) → Poly(3) → ... → RBF → GP → Geodesic
```

### How do I know when regression converged?

Look for:
```
Converged! → β stable, drift → 0 ✓
```

Or check the output:
- Betti numbers stable (e.g., `β = (1, 0)` for 3+ epochs)
- Error below epsilon
- "The Answer Has Come"

### My regression isn't converging. What do I do?

1. **Increase max epochs**: Default may be too low
2. **Relax epsilon**: `convergence(1e-4)` instead of `1e-6`
3. **Check tau**: Try different values (3, 5, 7, 10)
4. **Use `escalate: true`**: Let AEGIS find the right model

---

## Visualization Questions

### How do I visualize my manifold?

```aegis
render M {
    color: by_density
}
```

### What color modes are available?

| Mode | Effect |
|------|--------|
| `by_density` | Color by local point density |
| `by_cluster` | Color by cluster assignment |
| `gradient` | Gradient along trajectory |

### Can I export to 3D formats?

Currently, AEGIS outputs ASCII visualization. WebGL and OBJ export are planned features.

### How do I highlight specific blocks?

```aegis
block B = M[0:64]
render M {
    color: by_density,
    highlight: B
}
```

---

## Performance Questions

### How fast is AEGIS?

Benchmarks on standard hardware:
- 64 points: < 1ms
- 256 points: < 10ms
- 1024 points: < 100ms
- Regression convergence: varies (typically 10-100 epochs)

### How much memory does AEGIS use?

Minimal - AEGIS is designed for bare-metal operation:
- Fixed-size allocations (no heap in kernel mode)
- `heapless` collections with compile-time bounds
- Typical: < 1MB for most scripts

### Can I run on embedded systems?

Yes! AEGIS is `no_std` compatible and can run on:
- x86_64 bare metal
- ARM (with modifications)
- Any system with a Rust nightly target

---

## Docker Questions

### How do I mount local files in Docker?

```bash
docker run -v $(pwd):/scripts teerthsharma/aegis run /scripts/your_file.aegis
```

### What Docker commands are available?

- `docker run teerthsharma/aegis repl` - Interactive REPL
- `docker run teerthsharma/aegis run <file>` - Execute script
- `docker run teerthsharma/aegis benchmark` - Run benchmarks
- `docker run teerthsharma/aegis --help` - Show help

### How do I save output from Docker?

```bash
docker run -v $(pwd)/output:/output teerthsharma/aegis run script.aegis > /output/results.txt
```

---

## Troubleshooting

### "Command not found: aegis"

You're not using Docker. Run:
```bash
docker run -it teerthsharma/aegis repl
```

### "File not found"

Mount your directory:
```bash
docker run -v $(pwd):/scripts teerthsharma/aegis run /scripts/file.aegis
```

### "Parse error: unexpected token"

Check your syntax. Common issues:
- Missing colons in config blocks: `model: "polynomial"` ✓
- Missing commas between config items
- Unclosed braces `{}`

### "Build failed: serde_core errors"

This is a nightly toolchain issue, not your code. Solutions:
1. Update nightly: `rustup update nightly`
2. Pin to a specific nightly in `rust-toolchain.toml`
3. Use Docker instead

### "Convergence not reached"

See [regression troubleshooting](#my-regression-isnt-converging-what-do-i-do).

---

## Contributing Questions

### How do I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md). Quick start:
```bash
git clone https://github.com/teerthsharma/aegis.git
docker-compose run dev
cargo test --lib
```

### What should I work on?

Check [GitHub Issues](https://github.com/teerthsharma/aegis/issues) for:
- `good first issue` - Beginner friendly
- `help wanted` - Community input needed
- `enhancement` - New features

### How do I report bugs?

Open a [GitHub Issue](https://github.com/teerthsharma/aegis/issues) with:
- AEGIS version
- Rust version (if building from source)
- Minimal reproducible script
- Expected vs actual behavior

---

## Still Have Questions?

- 📖 [Full Documentation](../README.md#-documentation)
- 💬 [GitHub Discussions](https://github.com/teerthsharma/aegis/discussions)
- 🐛 [Report Issue](https://github.com/teerthsharma/aegis/issues)
