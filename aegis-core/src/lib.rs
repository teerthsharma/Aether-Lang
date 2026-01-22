//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Core Library
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Platform-agnostic mathematical foundation for AEGIS.
//! Works on both `no_std` (bare-metal kernel) and `std` (CLI/apps).
//!
//! Core Modules:
//!   - topology: TDA, Betti numbers, shape verification
//!   - manifold: Time-delay embedding, sparse attention graphs
//!   - aether: AETHER geometric primitives, hierarchical blocks
//!   - ml: Regression engine, convergence detection
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(feature = "alloc", not(feature = "std")))]
extern crate alloc;

// ═══════════════════════════════════════════════════════════════════════════════
// Module Exports
// ═══════════════════════════════════════════════════════════════════════════════

pub mod topology;
pub mod manifold;
pub mod aether;
pub mod ml;
pub mod state;
pub mod governor;

// Re-export key types for convenience
pub use topology::{TopologicalShape, VerifyResult, compute_betti_0, compute_betti_1, compute_shape, verify_shape};
pub use manifold::{ManifoldPoint, TimeDelayEmbedder, SparseAttentionGraph, TopologicalPipeline};
pub use aether::{BlockMetadata, HierarchicalBlockTree, DriftDetector};
