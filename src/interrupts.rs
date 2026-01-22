//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Interrupt Handling
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Configures the IDT and handles hardware interrupts.
//! Each interrupt updates the global state vector μ(t).
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame};
use spin::Mutex;
use crate::state::SystemState;
use crate::STATE_DIMENSION;
use crate::serial_println;

// ═══════════════════════════════════════════════════════════════════════════════
// Global State
// ═══════════════════════════════════════════════════════════════════════════════

/// Global IDT (interrupt descriptor table)
static IDT: spin::Lazy<InterruptDescriptorTable> = spin::Lazy::new(|| {
    let mut idt = InterruptDescriptorTable::new();
    
    idt.breakpoint.set_handler_fn(breakpoint_handler);
    idt.double_fault.set_handler_fn(double_fault_handler);
    
    idt
});

/// Global system state (updated by interrupt handlers)
static CURRENT_STATE: Mutex<SystemState<STATE_DIMENSION>> = 
    Mutex::new(SystemState { vector: [0.0; STATE_DIMENSION], timestamp: 0 });

/// IRQ counter (for rate calculation)
static IRQ_COUNTER: spin::Mutex<u64> = spin::Mutex::new(0);

/// Timestamp counter (microseconds since boot)
static TIMESTAMP: spin::Mutex<u64> = spin::Mutex::new(0);

// ═══════════════════════════════════════════════════════════════════════════════
// IDT Initialization
// ═══════════════════════════════════════════════════════════════════════════════

/// Initialize the Interrupt Descriptor Table
pub fn init_idt() {
    IDT.load();
}

/// Get current system state
pub fn get_current_state() -> SystemState<STATE_DIMENSION> {
    *CURRENT_STATE.lock()
}

/// Update state component
pub fn update_state_component(index: usize, value: f64) {
    if index < STATE_DIMENSION {
        let mut state = CURRENT_STATE.lock();
        state.vector[index] = value;
        state.timestamp = *TIMESTAMP.lock();
    }
}

/// Increment IRQ counter and update state
fn record_irq() {
    let mut counter = IRQ_COUNTER.lock();
    *counter += 1;
    
    // Update IRQ rate in state vector (dimension 1)
    let mut state = CURRENT_STATE.lock();
    state.vector[1] = *counter as f64 / 1000.0; // Normalize
    
    let mut ts = TIMESTAMP.lock();
    *ts += 1; // Simplified timestamp increment
    state.timestamp = *ts;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Exception Handlers
// ═══════════════════════════════════════════════════════════════════════════════

extern "x86-interrupt" fn breakpoint_handler(stack_frame: InterruptStackFrame) {
    serial_println!("[INT] BREAKPOINT");
    serial_println!("{:#?}", stack_frame);
    record_irq();
}

extern "x86-interrupt" fn double_fault_handler(
    stack_frame: InterruptStackFrame, 
    _error_code: u64
) -> ! {
    serial_println!("[FATAL] DOUBLE FAULT");
    serial_println!("{:#?}", stack_frame);
    
    loop {
        x86_64::instructions::hlt();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_state_update() {
        update_state_component(0, 0.5);
        let state = get_current_state();
        assert!((state.vector[0] - 0.5).abs() < 1e-10);
    }
}
