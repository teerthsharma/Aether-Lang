# AEGIS Bio-Kernel: BIOS & Bootloader PRD

## 1. Executive Summary
This document defines the Product Requirements for the AEGIS "Bio-Kernel" boot process. The goal is to ensure the OS kernel is **adaptive** and **architecturally aware** from the moment of boot. Unlike traditional kernels that rely on hardcoded expectations, the Bio-Kernel must dynamically ingest the hardware topology (via ACPI, Device Tree, or UEFI) and configure its internal "biological" state (Geometric Governor) accordingly.

## 2. Core Philosophy: The Living Hardware
The hardware is not just a platform; it is the **substrate** of the organism.
- **CPU Cores** = Neural Clusters (Compute capability)
- **RAM** = Synaptic Space (State capacity)
- **PCIe/Peripherals** = Sensory/Motor Organs (I/O capability)

The BIOS/Bootloader must act as the "DNA transcription" layer, translating raw silicon capability into a structured `HardwareTopology` that the AEGIS kernel can understand and optimize for.

## 3. Technical Requirements

### 3.1 Boot Protocol Support
The system must support modern, standard boot protocols to ensure broad compatibility.
- **Primary:** Multiboot2 (x86_64 standard)
- **Secondary:** Limine (Modern, flexible)
- **Legacy:** BIOS via shim (not prioritized, focus on UEFI/BIOS via Multiboot2)

### 3.2 The Handoff Structure (`BootInfo`)
The bootloader must construct and pass a robust `BootInfo` structure containing:
1.  **Memory Map:** A sanitized, non-overlapping map of usable RAM, reserved areas, and ACPI tables.
2.  **Physical Topology:**
    *   ACPI RSDP pointer (x86)
    *   Device Tree Blob (DTB) pointer (ARM/RISC-V)
3.  **Framebuffer Info:**
    *   Physical address, size, stride, format (RGB888/BGR888).
    *   *Requirement:* Resolution must be high enough for visual manifold debugging (min 1024x768).

### 3.3 Hardware Discovery (The "Bio-Scan")
Upon initialization (`kernel_main`), the OS must perform a "Bio-Scan":
1.  **ACPI Parsing (x86):**
    *   Parse MADT (Multiple APIC Description Table) to find all Local APICS (Cores).
    *   Parse SRAT (System Resource Affinity Table) to understand NUMA nodes (Memory locality).
2.  **Topology Construction:**
    *   Build a `HardwareTopology` graph where nodes are Cores/Memory and edges are latency/bandwidth.
    *   This graph informs the `Governor` where to place latency-sensitive threads (High-frequency signal processing vs. Background ML tasks).

### 3.4 Adaptive Initialization
*   **Single Core:** If only 1 core detected, disable `Manifold` parallelism, run strictly serial.
*   **Multi-Core:** Detect L3 cache sharing. Group threads effectively.
*   **Low Memory:** If RAM < 1GB, switch ML engine to "Sparse Mode" (lighter matrices).
*   **High Memory:** If RAM > 32GB, enable "Deep Manifold History" (longer time horizons).

## 4. Interface Definition (Rust)

The kernel expects the following traits to be implemented by the architecture-specific layer:

```rust
pub struct HardwareTopology {
    pub cpu_cores: usize,
    pub numa_nodes: usize,
    pub total_memory: u64,
    pub io_capabilities: IoCaps,
}

pub trait BiosInterface {
    /// Get the raw memory map iterator
    fn memory_map(&self) -> impl Iterator<Item = MemoryRegion>;
    
    /// Get the sensory window (Framebuffer)
    fn framebuffer(&self) -> Option<Framebuffer>;
    
    /// Get the root configuration pointer (RSDP or DTB)
    fn config_root(&self) -> PhysAddr;
}
```

## 5. Success Metrics
1.  **Universal Boot:** Kernel compiles once and boots on both QEMU (ACPI) and Bare Metal (UEFI/BIOS) without code changes, just adapting to the `BootInfo`.
2.  **Topology Awareness:** Kernel logs correct core count and memory layout on startup.
3.  **Graceful Degeneracy:** Kernel boots even if ACPI is malformed, falling back to "Safe Mode" (1 core, assumes contiguous RAM).
