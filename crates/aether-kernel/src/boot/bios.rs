use core::iter::Iterator;
use aether_core::PhysAddr;
use multiboot2::{BootInformation, BootInformationHeader};

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

/// The interface that the BIOS/Bootloader must satisfy.
/// Acts as the "DNA transcription" layer.
pub trait BiosInterface {
    /// Get the raw memory map iterator
    // We use a simplified return type here as returning `impl Iterator` in traits is tricky in no_std without GATs/TAITs fully stabilized or boxing.
    // Ideally we'd return a custom iterator struct.
    // For simplicity, we'll let the caller get the raw iter via a method or just handle it here.
    // Changed: simplified to a function that processes regions or returns a specific iterator type.
    // BUT since we can't return `impl Iterator` easily in stable traits, let's use a callback or just return a concrete iterator if possible, or just panic if not implemented.
    // Wait, we can define the Iterator type in the impl.
    // Let's refine the trait to be more practical for this step.
    // We will just expose a method to get the info we need.
    
    // For this step, I'll modify the trait to be simpler or implement it directly on the struct.
}

/// BootInfo passed from the bootloader.
pub struct BootInfo {
    multiboot_start: PhysAddr,
}

impl BootInfo {
    /// Create a new BootInfo from the physical address of the multiboot2 structure.
    ///
    /// # Safety
    /// The caller must ensure that `multiboot_start` points to a valid Multiboot2 information structure.
    pub unsafe fn new(multiboot_start: PhysAddr) -> Self {
        Self { multiboot_start }
    }

    /// Access the raw multiboot information.
    fn raw(&self) -> Option<BootInformation> {
        unsafe { multiboot2::load(self.multiboot_start as usize).ok() }
    }
    
    /// Iterate over the memory map using a callback.
    /// This avoids returning complex iterators with lifetimes.
    pub fn walk_memory_map<F>(&self, mut f: F) 
    where F: FnMut(MemoryRegion)
    {
         if let Some(info) = self.raw() {
            if let Some(tag) = info.memory_map_tag() {
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
                        }
                    });
                }
            }
         }
    }

    pub fn framebuffer(&self) -> Option<Framebuffer> {
        let info = self.raw()?;
        let tag = info.framebuffer_tag().ok()?; // Unwrap result
        
        Some(Framebuffer {
            address: tag.address(),
            width: tag.width(),
            height: tag.height(),
            pitch: tag.pitch(),
            bpp: tag.bpp(),
        })
    }

    pub fn config_root(&self) -> Option<PhysAddr> {
        let info = self.raw()?;
        
        // Try RSDP (new ACPI)
        if let Some(tag) = info.rsdp_v2_tag() {
            // Fix: signature() returns Result<&str, ...>, need to unwrap or handle
             return Some(tag.signature().ok()?.as_ptr() as u64);
        }
        
        // Try RSDP (old ACPI)
        if let Some(tag) = info.rsdp_v1_tag() {
            // Similarly.
            return None; // Placeholder
        }
        
        // DTB not standard in multiboot2 usually (it's MBI), but we can look for it.
        None
    }
}
