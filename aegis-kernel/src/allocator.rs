//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Heap Allocator
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! A simple bump allocator for Phase 1. Will be replaced with a proper
//! slab allocator in later phases for verified disjointness.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use core::alloc::{GlobalAlloc, Layout};
use core::ptr::null_mut;
use spin::Mutex;
use bootloader::BootInfo;

/// Heap size: 64 KiB for Phase 1
const HEAP_SIZE: usize = 64 * 1024;

/// Static heap buffer
static mut HEAP: [u8; HEAP_SIZE] = [0; HEAP_SIZE];

/// Bump allocator
struct BumpAllocator {
    next: usize,
    allocations: usize,
}

impl BumpAllocator {
    const fn new() -> Self {
        Self { next: 0, allocations: 0 }
    }
    
    fn alloc(&mut self, layout: Layout) -> *mut u8 {
        // Align up
        let alloc_start = (self.next + layout.align() - 1) & !(layout.align() - 1);
        let alloc_end = alloc_start + layout.size();
        
        if alloc_end > HEAP_SIZE {
            return null_mut();
        }
        
        self.next = alloc_end;
        self.allocations += 1;
        
        unsafe { HEAP.as_mut_ptr().add(alloc_start) }
    }
}

/// Global allocator instance
static ALLOCATOR: Mutex<BumpAllocator> = Mutex::new(BumpAllocator::new());

/// Global allocator wrapper
struct AegisAllocator;

unsafe impl GlobalAlloc for AegisAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATOR.lock().alloc(layout)
    }
    
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // Bump allocator doesn't deallocate
        // Phase 2 will implement proper slab allocator
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: AegisAllocator = AegisAllocator;

/// Initialize heap (placeholder for bootloader memory map integration)
pub fn init_heap(_boot_info: &'static BootInfo) {
    // Phase 1: Use static buffer
    // Phase 2: Will use boot_info memory map for proper heap region
}
