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

use aether_core::state::SystemState;
use aether_kernel::{
    allocator, interrupts, scheduler::SparseScheduler, serial, serial_println, STATE_DIMENSION,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Entry Point
// ═══════════════════════════════════════════════════════════════════════════════

/// The kernel entry point (called by bootloader)
#[no_mangle]
pub extern "C" fn _start(boot_info_addr: u64) -> ! {
    kernel_main(boot_info_addr)
}

/// The kernel main function
fn kernel_main(boot_info_addr: u64) -> ! {
    // Initialize serial output for debugging
    serial::init();
    serial_println!("═══════════════════════════════════════════════════════════════");
    serial_println!("  AEGIS v0.1.0-alpha");
    serial_println!("  Bio-Kernel: Geometric Sparse-Event Microkernel");
    serial_println!("  Topological Systems Engineering");
    serial_println!("═══════════════════════════════════════════════════════════════");

    // ═══════════════════════════════════════════════════════════════════════════
    // PHASE 1: BIO-SCAN (Hardware Discovery)
    // ═══════════════════════════════════════════════════════════════════════════
    serial_println!("[BIO-SCAN] Scanning organism topology...");
    
    // Construct BootInfo from the passed address (Multiboot2)
    // Safety: We assume the bootloader passed a valid address in rdi/first arg.
    let boot_info = unsafe { aether_kernel::boot::bios::BootInfo::new(boot_info_addr) };
    
    // Perform Bio-Scan
    let topology = aether_kernel::boot::topology::HardwareTopology::bio_scan(&boot_info);
    
    serial_println!("[BIO-SCAN] Neural Clusters (Cores): {}", topology.cpu_cores);
    serial_println!("[BIO-SCAN] Synaptic Space  (RAM)  : {} MB", topology.total_memory / 1024 / 1024);
    serial_println!("[BIO-SCAN] Sensory Organs  (I/O)  : {:?}", topology.io_capabilities);

    // ═══════════════════════════════════════════════════════════════════════════
    // PHASE 2: ADAPTIVE INITIALIZATION
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Initialize heap allocator
    allocator::init_heap();
    serial_println!("[INIT] Heap allocator initialized");

    // Initialize interrupt descriptor table
    interrupts::init_idt();
    serial_println!("[INIT] IDT configured");

    // Initialize the sparse scheduler with initial state
    let initial_state = SystemState::<STATE_DIMENSION>::zero();
    let mut scheduler = SparseScheduler::new(initial_state);

    serial_println!("[INIT] Sparse scheduler initialized");
    serial_println!("[INIT] ε₀ = {:.4}", scheduler.governor().epsilon());
    
    // Adaptive Logic based on Topology
    if topology.total_memory > 32 * 1024 * 1024 * 1024 {
         serial_println!("[ADAPT] High Memory detected > 32GB: Enabling Deep Manifold History");
         // Enable deep history (placeholder)
    } else if topology.total_memory < 1 * 1024 * 1024 * 1024 {
         serial_println!("[ADAPT] Low Memory detected < 1GB: Switching to Sparse Mode");
         // Enable sparse mode (placeholder)
    }

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
            serial_println!(
                "[EVENT] Δ={:.4}, ε={:.4}",
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
