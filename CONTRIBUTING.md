# Contributing to AEGIS

Thank you for your interest in contributing to AEGIS! This document provides guidelines for contributing to the project.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/aegis.git
cd aegis

# Start development environment
docker-compose run dev

# Run tests
cargo test --lib

# Format code
cargo fmt

# Run linter
cargo clippy
```

## 📋 Development Process

### 1. Fork & Clone

1. Fork the repository on GitHub
2. Clone your fork locally
3. Add upstream remote: `git remote add upstream https://github.com/ORIGINAL/aegis.git`

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Make Changes

- Follow the code style (run `cargo fmt`)
- Add tests for new features
- Update documentation as needed

### 4. Test

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with verbose output
cargo test -- --nocapture
```

### 5. Submit PR

1. Push to your fork
2. Create Pull Request against `main`
3. Fill out the PR template
4. Wait for review

## 📝 Code Style

### Rust

- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

### AEGIS Scripts (.aegis)

```aegis
// Use comments to explain intent
manifold M = embed(data, dim=3, tau=5)

// Group related statements
block B = M.cluster(0:64)
centroid C = B.center

// Use descriptive names
regress {
    model: "polynomial",
    escalate: true
}
```

## 🏗️ Architecture

### Adding a New ML Model

1. Add model type to `src/ml/regressor.rs`:

```rust
pub enum ModelType {
    // ... existing models
    YourNewModel { param: f64 },
}
```

2. Implement fitting in `ManifoldRegressor`:

```rust
fn fit_your_model(&mut self, param: f64) -> f64 {
    // Implementation
}
```

3. Add to escalation order in `upgrade_model()`

### Adding a New Language Construct

1. Add token to `src/lang/lexer.rs`
2. Add AST node to `src/lang/ast.rs`
3. Add parsing in `src/lang/parser.rs`
4. Add execution in `src/lang/interpreter.rs`

## 📊 Testing

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_your_feature() {
        // Arrange
        let input = ...;
        
        // Act
        let result = your_function(input);
        
        // Assert
        assert_eq!(result, expected);
    }
}
```

### Integration Tests

Place in `tests/` directory:

```rust
// tests/integration_test.rs
use aegis::lang::Parser;

#[test]
fn test_full_script_execution() {
    let source = r#"
        manifold M = embed(data, dim=3)
        regress { escalate: true }
    "#;
    // ...
}
```

## 📖 Documentation

- Use `///` for public API docs
- Use `//!` for module-level docs
- Include examples in doc comments

```rust
/// Compute topological convergence score.
///
/// # Arguments
/// * `betti` - Current Betti numbers
/// * `drift` - Centroid drift value
///
/// # Returns
/// Convergence score in range [0, 1]
///
/// # Example
/// ```
/// let score = convergence_score(BettiNumbers::new(1, 0), 0.001);
/// assert!(score > 0.9);
/// ```
pub fn convergence_score(betti: BettiNumbers, drift: f64) -> f64 {
    // ...
}
```

## 🐛 Bug Reports

Please include:
- AEGIS version
- Rust version (`rustc --version`)
- OS and version
- Minimal reproducible example
- Expected vs actual behavior

## 💡 Feature Requests

Open an issue with:
- Clear description of the feature
- Use case / motivation
- Proposed API (if applicable)
- Willingness to implement

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
