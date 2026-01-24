//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Language Core
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! A full-fledged programming language for 3D manifold-native machine learning.
//!
//! Key Features:
//!   - Seal loops (🦭) with topological convergence
//!   - Tilde (~) statement terminator
//!   - Control flow (if, for, while, fn)
//!   - Manifold primitives (embed, block, cluster)
//!   - ASCII and WebGL visualization
//!
//! Example `.aegis` script:
//! ```aegis
//! let data = [1.0, 2.0, 3.0]~
//! manifold M = embed(data, dim=3, tau=5)~
//! 🦭 until convergence(1e-6) {
//!     regress { model: "polynomial", escalate: true }~
//! }
//! render M { format: "ascii" }~
//! ```
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
#[macro_use]
extern crate alloc;



// ═══════════════════════════════════════════════════════════════════════════════
// Module Declarations
// ═══════════════════════════════════════════════════════════════════════════════

pub mod ascii_render;
pub mod ast;
pub mod interpreter;
pub mod lexer;
pub mod parser;
pub mod webgl_export;

// Re-exports for convenience
pub use ast::*;
pub use interpreter::Interpreter;
pub use lexer::{Lexer, Token, TokenKind};
pub use parser::Parser;

// Re-export core types that the interpreter uses
pub use aegis_core::{
    BlockMetadata, DriftDetector, HierarchicalBlockTree, ManifoldPoint, SparseAttentionGraph,
    TimeDelayEmbedder,
};
