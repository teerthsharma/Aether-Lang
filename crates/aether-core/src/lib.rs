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

// ═══════════════════════════════════════════════════════════════════════════════
// Aether-Lang — invented by Teerth Sharma
// https://github.com/teerthsharma/Aether-Lang
// Copyright (c) 2026 Teerth Sharma. All Rights Reserved.
// ═══════════════════════════════════════════════════════════════════════════════
//

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

// ═══════════════════════════════════════════════════════════════════════════════
// Module Exports
// ═══════════════════════════════════════════════════════════════════════════════

pub mod aether;
pub mod attention;
pub mod diagram;
pub mod governor;
pub mod manifold;
pub mod memory;
pub mod ml;
pub mod persistence;
pub mod scheduled;
pub mod state;
pub mod topology;

// Re-export key types for convenience
pub use aether::{BlockMetadata, DriftDetector, HierarchicalBlockTree};
pub use diagram::{
    bottleneck_distance, landscape_norm, persistence_image, persistence_landscape,
    persistent_entropy, total_persistence, wasserstein_distance, ImageConfig, LandscapeConfig,
};
pub use manifold::{ManifoldPoint, SparseAttentionGraph, TimeDelayEmbedder, TopologicalPipeline};
pub use persistence::{
    persistent_homology, time_delay_persistence, BettiNumbers3, ComplexKind, PersistenceConfig,
    PersistenceDiagram, PersistenceError, PersistencePair,
};
pub use topology::{
    compute_betti_0, compute_betti_1, compute_shape, verify_shape, TopologicalShape, VerifyResult,
};
