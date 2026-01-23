# OS Development with AEGIS

This guide explains how to use the AEGIS core primitives to build an operating system kernel.

## 1. Bootstrapping

To use AEGIS in a bare-metal environment, you must disable the standard library.
In your kernel's `Cargo.toml`:

```toml
[dependencies]
aegis-core = { version = "0.1", default-features = false, features = ["no_std"] }
```

### Entry Point
Your assembly entry point (e.g., `boot.S`) should call a Rust function marked `#[no_mangle] extern "C"`.

```rust
#![no_std]
#![no_main]

use aegis_core::os::CpuContext;

#[no_mangle]
pub extern "C" fn kmain() -> ! {
    // Initialize serial port
    // Initialize GDT/IDT
    loop {}
}
```

## 2. Interrupt Handling

AEGIS provides the `ExceptionFrame` struct to map raw stack data from interrupts.

```rust
use aegis_core::os::ExceptionFrame;

#[no_mangle]
pub extern "x86-interrupt" fn general_protection_fault_handler(
    stack_frame: ExceptionFrame, 
    _error_code: u64
) {
    panic!("GP Fault at IP: {:#x}", stack_frame.rip);
}
```

## 3. Context Switching

Use `CpuContext` to save and restore task state.

```rust
use aegis_core::os::CpuContext;

pub struct Task {
    pub context: CpuContext,
    pub id: u64,
}

impl Task {
    pub fn new() -> Self {
        Self {
            context: CpuContext::empty(),
            id: 0,
        }
    }
}
```

## 4. Geometric Scheduling

AEGIS is designed for **Sparse-Event** kernels. Instead of a simple round-robin scheduler, use the **Geometric Governor** to decide when to switch tasks.

```rust
use aegis_core::governor::GeometricGovernor;

// Calculate system deviation
let deviation = current_state.deviation(&last_state);

// Only switch context if deviation exceeds adaptive threshold
if governor.should_intervene(deviation) {
    switch_context();
}
```

## 5. Paging

Use `PageTableEntry` to manipulate hardware page tables safely.

```rust
use aegis_core::os::PageTableEntry;
 
let mut pte = PageTableEntry::new(0);
pte.set_addr(0x1000); 
// Flags are preserved
```
