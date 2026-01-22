# Changelog

All notable changes to AEGIS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- WebGL 3D visualization export
- ASCII art rendering in terminal
- Additional regression models (kernel ridge, neural manifold)
- WASM compilation target
- Language server protocol (LSP) for IDE support

---

## [0.1.0-alpha] - 2026-01-22

### Added

#### AEGIS Language (DSL)
- **Lexer** (`src/lang/lexer.rs`) - Tokenizes `.aegis` scripts
  - 3D manifold keywords: `manifold`, `block`, `regress`, `render`, `embed`
  - Literals: integers, floats, strings, booleans
  - Operators and punctuation
- **Parser** (`src/lang/parser.rs`) - Recursive descent parser
  - Full AST generation
  - Config block parsing for `regress` and `render`
  - Error recovery with line/column reporting
- **AST** (`src/lang/ast.rs`) - Abstract syntax tree definitions
  - `ManifoldDecl`, `BlockDecl`, `RegressStmt`, `RenderStmt`
  - Expression types: Num, Bool, Str, Ident, FieldAccess, Call, MethodCall
- **Interpreter** (`src/lang/interpreter.rs`) - Runtime execution
  - Manifold workspace management
  - Block extraction and geometry
  - Escalating regression integration

#### ML Engine
- **ManifoldRegressor** (`src/ml/regressor.rs`)
  - Linear regression
  - Polynomial regression (arbitrary degree)
  - RBF kernel regression
  - Gaussian Process (approximate)
  - Geodesic manifold regression
  - Model upgrade sequence for escalation
- **ConvergenceDetector** (`src/ml/convergence.rs`)
  - Betti number tracking
  - Centroid drift detection
  - Residual manifold analysis
  - Topological convergence scoring
- **EscalatingBenchmark** (`src/ml/benchmark.rs`)
  - Auto-escalating model complexity
  - Configurable patience and stability window
  - Test function generators

#### Docker Support
- **Dockerfile** - Multi-stage build
  - Builder stage with nightly Rust
  - Runtime stage with CLI
- **docker-compose.yml** - Service definitions
  - `aegis` - Interactive REPL
  - `benchmark` - Benchmark runner
  - `dev` - Development shell
- **aegis-cli.sh** - Full CLI implementation
  - `aegis repl` - Interactive mode
  - `aegis run <file>` - Script execution
  - `aegis benchmark` - Benchmark suite

#### Documentation
- Comprehensive README with badges
- Getting Started guide
- Language Reference (EBNF grammar)
- API Reference
- Tutorial (5 parts)
- Examples (15+ annotated scripts)
- FAQ
- Contributing guidelines
- Architecture documentation
- Mathematical foundations

#### CI/CD
- GitHub Actions workflow
  - Build and test
  - Docker build
  - Script validation
  - Documentation generation

### Changed
- `lib.rs` - Added `cfg_attr` for test compatibility
- `Cargo.toml` - Added `heapless` dependency

### Fixed
- N/A (initial release)

---

## [0.0.1] - 2026-01-15

### Added
- Initial AEGIS sparse-event microkernel
- Geometric governor (PID-on-manifold)
- Sparse scheduler
- Topological code loader
- Time-delay embedder
- AETHER block trees
- Basic topology analysis

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0-alpha | 2026-01-22 | Full AEGIS language, ML engine, Docker |
| 0.0.1 | 2026-01-15 | Initial microkernel |

---

## Migration Guides

### 0.0.1 → 0.1.0-alpha

No breaking changes. New language features are additive.

To use the new AEGIS language:
```aegis
manifold M = embed(data, dim=3, tau=5)
regress { model: "polynomial", escalate: true }
```

---

## Contributors

Thanks to all contributors who made this release possible!

---

## Links

- [GitHub Releases](https://github.com/teerthsharma/aegis/releases)
- [Issue Tracker](https://github.com/teerthsharma/aegis/issues)
- [Documentation](https://github.com/teerthsharma/aegis#-documentation)
