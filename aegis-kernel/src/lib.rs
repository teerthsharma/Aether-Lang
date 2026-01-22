//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Kernel Library
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Bare-metal x86_64 microkernel with sparse event loop.
//! This is the original AEGIS kernel for research and embedded use.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]

extern crate alloc;

// ═══════════════════════════════════════════════════════════════════════════════
// Module Declarations
// ═══════════════════════════════════════════════════════════════════════════════

pub mod allocator;
pub mod serial;
pub mod interrupts;
pub mod scheduler;
pub mod loader;

// Re-export core types
pub use aegis_core::*;
pub use aegis_lang::*;

/// Kernel state dimension
pub const STATE_DIMENSION: usize = 4;

/// Initial adaptive threshold
pub const INITIAL_EPSILON: f64 = 0.1;
