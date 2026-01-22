//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS ML Engine
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Machine learning on 3D manifolds with:
//! - Escalating regression benchmarks
//! - Topological convergence detection
//! - Non-linear optimization on manifold space
//!
//! "Run regression benchmarks infinitely harder each until perfect,
//!  and those answers come" - through topology.
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

pub mod regressor;
pub mod convergence;
pub mod benchmark;

pub use regressor::*;
pub use convergence::*;
pub use benchmark::*;
