// ═══════════════════════════════════════════════════════════════════════════════
// Aether-Lang — invented by Teerth Sharma
// https://github.com/teerthsharma/Aether-Lang
// Copyright (c) 2026 Teerth Sharma. All Rights Reserved.
// ═══════════════════════════════════════════════════════════════════════════════
//
//! Multiboot2 boot information, decoded into the kernel's own vocabulary.
//!
//! The kernel owns these types rather than borrowing `aether_core::os`, because
//! `aether_core` is the platform-agnostic math layer and must not grow a
//! dependency on a boot protocol.

use multiboot2::{BootInformation, BootInformationHeader};

/// A physical address. Bare `u64` because every multiboot2 accessor hands back
/// one and the kernel does raw pointer arithmetic on it.
pub type PhysAddr = u64;

/// A region of physical memory.
#[derive(Debug, Clone, Copy)]
pub struct MemoryRegion {
    pub start: PhysAddr,
    pub end: PhysAddr,
    pub kind: MemoryRegionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRegionKind {
    Usable,
    Reserved,
    Acpi,
    Kernel,
    Bootloader,
    Unknown,
}

/// Framebuffer information (Sensory Window).
#[derive(Debug, Clone, Copy)]
pub struct Framebuffer {
    pub address: PhysAddr,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub bpp: u8,
}

/// BootInfo passed from the bootloader.
pub struct BootInfo {
    multiboot_start: PhysAddr,
}

impl BootInfo {
    /// Create a new BootInfo from the physical address of the multiboot2 structure.
    ///
    /// # Safety
    /// The caller must ensure that `multiboot_start` points to a valid Multiboot2
    /// information structure, 8-byte aligned as the spec requires, and that it
    /// stays mapped for the lifetime of the kernel.
    pub unsafe fn new(multiboot_start: PhysAddr) -> Self {
        Self { multiboot_start }
    }

    /// Access the raw multiboot information.
    ///
    /// The `'static` lifetime is sound only because `new` requires the structure
    /// to remain mapped forever, which is true for the bootloader-provided MBI.
    fn raw(&self) -> Option<BootInformation<'static>> {
        unsafe { BootInformation::load(self.multiboot_start as *const BootInformationHeader).ok() }
    }

    /// Iterate over the memory map using a callback.
    /// This avoids returning complex iterators with lifetimes.
    pub fn walk_memory_map<F>(&self, mut f: F)
    where
        F: FnMut(MemoryRegion),
    {
        let Some(info) = self.raw() else {
            return;
        };
        let Some(tag) = info.memory_map_tag() else {
            return;
        };

        for area in tag.memory_areas() {
            f(MemoryRegion {
                start: area.start_address(),
                end: area.end_address(),
                kind: match multiboot2::MemoryAreaType::from(area.typ()) {
                    multiboot2::MemoryAreaType::Available => MemoryRegionKind::Usable,
                    multiboot2::MemoryAreaType::Reserved => MemoryRegionKind::Reserved,
                    multiboot2::MemoryAreaType::AcpiAvailable => MemoryRegionKind::Acpi,
                    multiboot2::MemoryAreaType::ReservedHibernate => MemoryRegionKind::Reserved,
                    _ => MemoryRegionKind::Unknown,
                },
            });
        }
    }

    pub fn framebuffer(&self) -> Option<Framebuffer> {
        let info = self.raw()?;
        // `framebuffer_tag` is Option<Result<..>>: absent tag vs. present-but-
        // unrecognised type are different failures. Both mean "no framebuffer".
        let tag = info.framebuffer_tag()?.ok()?;

        Some(Framebuffer {
            address: tag.address(),
            width: tag.width(),
            height: tag.height(),
            pitch: tag.pitch(),
            bpp: tag.bpp(),
        })
    }

    /// Physical address of the ACPI root table: XSDT when the bootloader supplied
    /// an ACPI 2.0+ RSDP, otherwise the 32-bit RSDT.
    pub fn config_root(&self) -> Option<PhysAddr> {
        let info = self.raw()?;

        if let Some(tag) = info.rsdp_v2_tag() {
            return Some(tag.xsdt_address() as PhysAddr);
        }
        if let Some(tag) = info.rsdp_v1_tag() {
            return Some(tag.rsdt_address() as PhysAddr);
        }
        None
    }
}
