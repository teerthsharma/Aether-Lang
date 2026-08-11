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

// ═══════════════════════════════════════════════════════════════════════════════
// Aether-Lang — invented by Teerth Sharma
// https://github.com/teerthsharma/Aether-Lang
// Copyright (c) 2026 Teerth Sharma. All Rights Reserved.
// ═══════════════════════════════════════════════════════════════════════════════
//

#![cfg_attr(not(feature = "std"), no_std)]

// Looks unused under `std` and is required without it.
//
// This crate's imports from the alloc crate sit behind
// `cfg(not(feature = "std"))`, so a host build compiles none of them and
// `-W unused` reports this line as dead. It is not: deleting it breaks every
// no_std build. `unused_extern_crates` reports, in one configuration, a fact
// that is only true in that configuration, which is why this workspace denies
// `unused_imports` and not the whole `unused` group.
#[cfg(feature = "alloc")]
extern crate alloc;

// ═══════════════════════════════════════════════════════════════════════════════
// Module Declarations
// ═══════════════════════════════════════════════════════════════════════════════

pub mod ascii_render;
pub mod ast;
pub mod interpreter;
pub mod lexer;
pub mod parser;
pub mod vm;
pub mod webgl_export;

#[cfg(feature = "python")]
pub mod python;

// Re-exports for convenience
pub use ast::*;
pub use interpreter::Interpreter;
pub use lexer::{Lexer, Token, TokenKind};
pub use parser::Parser;

// Re-export core types that the interpreter uses
pub use aether_core::{
    BlockMetadata, DriftDetector, HierarchicalBlockTree, ManifoldPoint, SparseAttentionGraph,
    TimeDelayEmbedder,
};
