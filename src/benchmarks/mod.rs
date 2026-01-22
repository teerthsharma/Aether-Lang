//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Benchmark Suite
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Rigorous benchmarks for all AEGIS mathematical components:
//! - Governor: PID convergence and stability
//! - Topology: Shape verification accuracy
//! - AETHER: Hierarchical pruning efficiency
//! - Manifold: Embedding quality and sparse attention
//! - ML: Escalating regression and convergence detection
//!
//! Run with: cargo test benchmarks --lib --target x86_64-pc-windows-msvc -- --nocapture
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

pub mod governor_bench;
pub mod topology_bench;
pub mod aether_bench;
pub mod manifold_bench;
pub mod ml_bench;
pub mod runner;

pub use runner::*;
