//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS ELF Loader with Topological Verification
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Parses ELF binaries and verifies .text sections using TDA.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::topology::{verify_sliding_window, is_shape_valid};

/// ELF magic bytes
const ELF_MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];

/// Loader error types
#[derive(Debug, Clone)]
pub enum LoadError {
    /// Not a valid ELF file
    InvalidMagic,
    /// File too small
    TooSmall,
    /// Topological anomaly detected
    InvalidGeometry { offset: usize },
    /// Unsupported architecture
    UnsupportedArch,
}

/// Minimal ELF header fields we need
#[derive(Debug, Clone, Copy)]
pub struct ElfInfo {
    pub is_64bit: bool,
    pub is_little_endian: bool,
    pub entry_point: u64,
    pub ph_offset: u64,
    pub ph_count: u16,
}

/// Verify and load ELF binary
pub fn verify_elf(data: &[u8]) -> Result<ElfInfo, LoadError> {
    // Check minimum size
    if data.len() < 64 {
        return Err(LoadError::TooSmall);
    }
    
    // Check magic
    if data[0..4] != ELF_MAGIC {
        return Err(LoadError::InvalidMagic);
    }
    
    // Parse header
    let is_64bit = data[4] == 2;
    let is_little_endian = data[5] == 1;
    
    if !is_64bit {
        return Err(LoadError::UnsupportedArch);
    }
    
    // Extract entry point (offset 24 for 64-bit)
    let entry_point = u64::from_le_bytes([
        data[24], data[25], data[26], data[27],
        data[28], data[29], data[30], data[31],
    ]);
    
    // Program header offset (offset 32)
    let ph_offset = u64::from_le_bytes([
        data[32], data[33], data[34], data[35],
        data[36], data[37], data[38], data[39],
    ]);
    
    // Program header count (offset 56)
    let ph_count = u16::from_le_bytes([data[56], data[57]]);
    
    // ═══════════════════════════════════════════════════════════════════════════
    // TOPOLOGICAL VERIFICATION
    // ═══════════════════════════════════════════════════════════════════════════
    // Slide window over executable sections and verify shape
    
    // For Phase 1, verify the entire file (simplified)
    // Phase 3 will parse sections properly
    if data.len() >= 64 {
        match verify_sliding_window(&data[64..], 64) {
            Ok(()) => {},
            Err(offset) => return Err(LoadError::InvalidGeometry { offset: offset + 64 }),
        }
    }
    
    Ok(ElfInfo {
        is_64bit,
        is_little_endian,
        entry_point,
        ph_offset,
        ph_count,
    })
}

/// Quick topology check without full ELF parsing
pub fn verify_binary_topology(data: &[u8]) -> bool {
    is_shape_valid(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_invalid_magic() {
        let data = [0x00u8; 64];
        assert!(matches!(verify_elf(&data), Err(LoadError::InvalidMagic)));
    }
    
    #[test]
    fn test_too_small() {
        let data = [0x7F, b'E', b'L', b'F'];
        assert!(matches!(verify_elf(&data), Err(LoadError::TooSmall)));
    }
}
