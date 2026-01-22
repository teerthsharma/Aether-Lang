//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS: Geometric Sparse-Event Microkernel
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! A formally verified, event-driven microkernel that:
//!   1. Executes only upon significant state deviation (Δ ≥ ε)
//!   2. Authenticates code via topological signature (Hₖ)
//!
//! Core Discipline: Topological Systems Engineering
//!
//! Mathematical Foundation:
//!   - State Vector: μ(t) ∈ ℝ^d
//!   - Deviation Metric: Δ(t) = ||μ(t) - μ(t_last)||₂
//!   - Adaptive Threshold: ε(t+1) = ε(t) + α·e(t) + β·de/dt
//!   - Topology Gate: Shape(B) = (β₀, β₁) via Persistent Homology
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]

extern crate alloc;

// ═══════════════════════════════════════════════════════════════════════════════
// Module Declarations
// ═══════════════════════════════════════════════════════════════════════════════

pub mod state;
pub mod governor;
pub mod topology;
pub mod manifold;
pub mod aether;
pub mod scheduler;
pub mod interrupts;
pub mod allocator;
pub mod serial;
pub mod loader;
pub mod lang;  // AEGIS 3D ML Language DSL
pub mod ml;    // ML regression engine

#[cfg(test)]
mod tests;

use core::panic::PanicInfo;
use bootloader::{entry_point, BootInfo};

use crate::scheduler::SparseScheduler;
use crate::state::SystemState;

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Dimensionality of our state manifold
/// d = 4: [Memory Pressure, IRQ Rate, Queue Depth, Entropy]
pub const STATE_DIMENSION: usize = 4;

/// Initial adaptive threshold ε₀
pub const INITIAL_EPSILON: f64 = 0.1;

// ═══════════════════════════════════════════════════════════════════════════════
// Entry Point
// ═══════════════════════════════════════════════════════════════════════════════

entry_point!(kernel_main);

/// The kernel entry point
/// 
/// This is where we transition from bootloader to AEGIS control.
/// The kernel operates as a Dynamic System on a Manifold, only waking
/// when the state trajectory deviates significantly from equilibrium.
fn kernel_main(boot_info: &'static BootInfo) -> ! {
    // Initialize serial output for debugging
    serial::init();
    serial_println!("═══════════════════════════════════════════════════════════════");
    serial_println!("  AEGIS v0.1.0-alpha");
    serial_println!("  Geometric Sparse-Event Microkernel");
    serial_println!("  Topological Systems Engineering");
    serial_println!("═══════════════════════════════════════════════════════════════");
    
    // Initialize heap allocator
    allocator::init_heap(boot_info);
    serial_println!("[INIT] Heap allocator initialized");
    
    // Initialize interrupt descriptor table
    interrupts::init_idt();
    serial_println!("[INIT] IDT configured");
    
    // Initialize the sparse scheduler with initial state
    let initial_state = SystemState::<STATE_DIMENSION>::zero();
    let mut scheduler = SparseScheduler::new(initial_state);
    
    serial_println!("[INIT] Sparse scheduler initialized");
    serial_println!("[INIT] ε₀ = {:.4}", scheduler.governor().epsilon());
    serial_println!("");
    serial_println!("[AEGIS] Entering sparse event loop...");
    serial_println!("[AEGIS] CPU will halt until Δ(t) ≥ ε(t)");
    
    // ═══════════════════════════════════════════════════════════════════════════
    // THE SPARSE EVENT LOOP
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // This is the heart of AEGIS. Unlike traditional kernels that tick at
    // fixed intervals (100Hz-1000Hz), we only wake when:
    //
    //   ||μ(t) - μ(t_last)||₂ ≥ ε(t)
    //
    // Between events, the CPU is halted in WFI (Wait For Interrupt) state,
    // consuming near-zero power.
    //
    loop {
        // Get current system state (updated by interrupt handlers)
        let current_state = interrupts::get_current_state();
        
        // The Sparse Trigger: Check if deviation exceeds threshold
        if scheduler.should_wake(&current_state) {
            // State has deviated significantly - process the event
            scheduler.handle_event(current_state);
            
            // Log for research purposes
            #[cfg(feature = "debug_topology")]
            serial_println!("[EVENT] Δ={:.4}, ε={:.4}", 
                scheduler.last_deviation(), 
                scheduler.governor().epsilon()
            );
        } else {
            // Deviation below threshold - accumulate entropy and halt
            scheduler.accumulate_entropy();
        }
        
        // Enter low-power state until next interrupt
        // This is the "Sleeping" in "Sleeping Kernel"
        x86_64::instructions::hlt();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Panic Handler
// ═══════════════════════════════════════════════════════════════════════════════

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("");
    serial_println!("═══════════════════════════════════════════════════════════════");
    serial_println!("  AEGIS KERNEL PANIC");
    serial_println!("═══════════════════════════════════════════════════════════════");
    serial_println!("{}", info);
    serial_println!("═══════════════════════════════════════════════════════════════");
    
    // Halt forever
    loop {
        x86_64::instructions::hlt();
    }
}
