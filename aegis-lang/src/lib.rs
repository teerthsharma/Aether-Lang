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
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec, boxed::Box, format, vec};

#[cfg(feature = "std")]
use std::{string::String, vec::Vec, boxed::Box, format, vec};

// ═══════════════════════════════════════════════════════════════════════════════
// Module Declarations
// ═══════════════════════════════════════════════════════════════════════════════

pub mod lexer;
pub mod ast;
pub mod parser;
pub mod interpreter;
pub mod ascii_render;
pub mod webgl_export;

// Re-exports for convenience
pub use lexer::{Lexer, Token, TokenKind};
pub use ast::*;
pub use parser::Parser;
pub use interpreter::Interpreter;

// Re-export core types that the interpreter uses
pub use aegis_core::{
    ManifoldPoint, TimeDelayEmbedder, SparseAttentionGraph,
    BlockMetadata, HierarchicalBlockTree, DriftDetector,
};
