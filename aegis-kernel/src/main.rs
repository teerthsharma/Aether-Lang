//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Kernel Entry Point
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Bare-metal x86_64 microkernel with sparse event loop.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]

extern crate alloc;

use core::panic::PanicInfo;
use bootloader::{entry_point, BootInfo};

use aegis_kernel::{
    serial, serial_println,
    allocator, interrupts,
    scheduler::SparseScheduler,
    STATE_DIMENSION,
};
use aegis_core::state::SystemState;

// ═══════════════════════════════════════════════════════════════════════════════
// Entry Point
// ═══════════════════════════════════════════════════════════════════════════════

entry_point!(kernel_main);

/// The kernel entry point
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
    loop {
        // Get current system state (updated by interrupt handlers)
        let current_state = interrupts::get_current_state();
        
        // The Sparse Trigger: Check if deviation exceeds threshold
        if scheduler.should_wake(&current_state) {
            // State has deviated significantly - process the event
            scheduler.handle_event(current_state);
            
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
    
    loop {
        x86_64::instructions::hlt();
    }
}
