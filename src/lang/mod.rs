//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Language Core
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! A domain-specific language for 3D manifold-native machine learning.
//!
//! Key Features:
//! - Manifold primitives (embed, block, cluster)
//! - Non-linear regression with escalating benchmarks
//! - Topological convergence detection
//! - 3D visualization directives
//!
//! Example `.aegis` script:
//! ```aegis
//! manifold M = embed(data, dim=3, tau=5)
//! regress { model: "polynomial", escalate: true, until: convergence(1e-6) }
//! render M { color: by_density }
//! ```
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

pub mod lexer;
pub mod ast;
pub mod parser;
pub mod interpreter;

// Re-exports for convenience
pub use lexer::{Lexer, Token, TokenKind};
pub use ast::*;
pub use parser::Parser;
pub use interpreter::Interpreter;
