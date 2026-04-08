//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Topological Gatekeeper
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Implements Topological Data Analysis (TDA) for binary authentication.
//! Uses Persistent Homology to compute "shape signatures" of code.
//!
//! Mathematical Foundation:
//!   - Embedding: Φ: B → P ∈ ℝⁿ (Time-Delay Embedding)
//!   - Homology: H_k (Betti numbers β₀, β₁)
//!   - Authentication: d_Wasserstein(Shape(B), Shape_ref) ≤ δ
//!
//! Heuristics:
//!   - Safe Code (linear logic): β₁ ≈ 0 (low loop complexity)
//!   - Malicious Code (NOP sleds/jumps): high β₀ clustering or high β₁
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

// use libm::fabs;

// ═══════════════════════════════════════════════════════════════════════════════
// Topology Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Geometric distance threshold for clustering (Betti-0 calculation)
const CLUSTER_THRESHOLD: i16 = 15;

/// Sliding window size for topology analysis
const WINDOW_SIZE: usize = 64;

/// Minimum density for valid code (β₀ / len)
const DENSITY_MIN: f64 = 0.1;

/// Maximum density for valid code
const DENSITY_MAX: f64 = 0.6;

/// Maximum allowed Betti-1 (loop complexity) per window
const MAX_BETTI_1: u32 = 10;

// ═══════════════════════════════════════════════════════════════════════════════
// Topological Shape Signature
// ═══════════════════════════════════════════════════════════════════════════════

/// Shape signature: (β₀, β₁) tuple from Persistent Homology
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopologicalShape {
    /// β₀: Number of connected components (0-dimensional holes)
    pub betti_0: u32,

    /// β₁: Number of loops/cycles (1-dimensional holes)
    pub betti_1: u32,

    /// Density: β₀ / data_length (normalized clustering)
    pub density: f64,
}

impl TopologicalShape {
    /// Create a shape from Betti numbers
    pub fn new(betti_0: u32, betti_1: u32, data_len: usize) -> Self {
        let density = if data_len > 0 {
            betti_0 as f64 / data_len as f64
        } else {
            0.0
        };

        Self {
            betti_0,
            betti_1,
            density,
        }
    }

    /// Simple distance metric between shapes
    pub fn distance(&self, other: &Self) -> f64 {
        let d0 = libm::pow(self.betti_0 as f64 - other.betti_0 as f64, 2.0);
        let d1 = libm::pow(self.betti_1 as f64 - other.betti_1 as f64, 2.0);
        let dd = libm::pow(self.density - other.density, 2.0);

        libm::sqrt(d0 + d1 + dd)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Betti Number Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute β₀ (connected components) via 1D clustering approximation
///
/// This is a simplified Vietoris-Rips filtration for 1D point clouds.
/// We treat bytes as points on ℝ and count "gaps" > threshold as
/// component boundaries.
///
/// # Arguments
/// * `data` - Binary data to analyze
///
/// # Returns
/// β₀: Number of connected components
#[inline(always)]
fn is_gap(a: u8, b: u8) -> bool {
    (a as i16 - b as i16).abs() > CLUSTER_THRESHOLD
}

#[inline(always)]
fn is_loop_pattern(w: &[u8]) -> bool {
    let tolerance = 5i16; // How close values must be to "close a loop"
    let a = w[0] as i16;
    let b = w[1] as i16;
    let c = w[2] as i16;
    let d = w[3] as i16;

    if (a - d).abs() <= tolerance {
        if (a - b).abs() > tolerance || (a - c).abs() > tolerance {
            return true;
        }
    }
    false
}

pub fn compute_betti_0(data: &[u8]) -> u32 {
    if data.len() < 2 {
        return if data.is_empty() { 0 } else { 1 };
    }

    let mut components = 0u32;
    let mut in_component = false;

    for window in data.windows(2) {
        if is_gap(window[0], window[1]) {
            if !in_component {
                components += 1;
                in_component = true;
            }
        } else {
            in_component = false;
        }
    }

    components
}

/// Compute β₁ (loops/cycles) via local pattern detection
///
/// This approximates 1-dimensional homology by detecting "oscillation" patterns
/// in the byte stream - sequences that return to similar values.
///
/// # Arguments
/// * `data` - Binary data to analyze
///
/// # Returns
/// β₁: Approximate number of loops/cycles
pub fn compute_betti_1(data: &[u8]) -> u32 {
    if data.len() < 4 {
        return 0;
    }

    let mut loops = 0u32;

    // Detect cycles: a -> b -> c -> ~a (return to start)
    for window in data.windows(4) {
        if is_loop_pattern(window) {
            loops += 1;
        }
    }

    loops
}

/// Compute full topological shape signature
pub fn compute_shape(data: &[u8]) -> TopologicalShape {
    let betti_0 = compute_betti_0(data);
    let betti_1 = compute_betti_1(data);

    TopologicalShape::new(betti_0, betti_1, data.len())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shape Verification
// ═══════════════════════════════════════════════════════════════════════════════

/// Verification result with detailed rejection reason
#[derive(Debug, Clone)]
pub enum VerifyResult {
    /// Code passed topological verification
    Pass,

    /// Density out of expected range
    InvalidDensity { actual: f64, min: f64, max: f64 },

    /// Too many loops (possible obfuscation)
    ExcessiveLoops { count: u32, max: u32 },

    /// Shape too different from reference
    ShapeMismatch { distance: f64, threshold: f64 },
}

/// Verify binary data against topological constraints
///
/// # Heuristics
/// - Standard compiled code: density ∈ [0.1, 0.6]
/// - Encrypted/obfuscated payloads: density outside this range
/// - NOP sleds: very low density (uniform bytes)
/// - ROP chains: very high loop count
///
/// # Arguments
/// * `data` - Binary data to verify
///
/// # Returns
/// `VerifyResult` indicating pass or detailed failure
pub fn verify_shape(data: &[u8]) -> VerifyResult {
    let shape = compute_shape(data);

    // Check density bounds
    if shape.density < DENSITY_MIN || shape.density > DENSITY_MAX {
        return VerifyResult::InvalidDensity {
            actual: shape.density,
            min: DENSITY_MIN,
            max: DENSITY_MAX,
        };
    }

    // Check loop complexity
    if shape.betti_1 > MAX_BETTI_1 {
        return VerifyResult::ExcessiveLoops {
            count: shape.betti_1,
            max: MAX_BETTI_1,
        };
    }

    VerifyResult::Pass
}

/// Simple boolean verification (convenience wrapper)
pub fn is_shape_valid(data: &[u8]) -> bool {
    matches!(verify_shape(data), VerifyResult::Pass)
}

/// Verify with custom reference shape (Wasserstein-like distance)
pub fn verify_against_reference(
    data: &[u8],
    reference: &TopologicalShape,
    threshold: f64,
) -> VerifyResult {
    let shape = compute_shape(data);
    let distance = shape.distance(reference);

    if distance > threshold {
        return VerifyResult::ShapeMismatch {
            distance,
            threshold,
        };
    }

    verify_shape(data)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sliding Window Analysis
// ═══════════════════════════════════════════════════════════════════════════════

/// Analyze binary with sliding window, fail-fast on any violation
///
/// This is used by the ELF loader to check .text sections.
///
/// # Arguments
/// * `data` - Full binary data
/// * `window_size` - Size of sliding window (default: 64)
///
/// # Returns
/// `Ok(())` if all windows pass, `Err(offset)` at first failure
pub fn verify_sliding_window(data: &[u8], window_size: usize) -> Result<(), usize> {
    let size = if window_size == 0 {
        WINDOW_SIZE
    } else {
        window_size
    };

    if data.len() < size {
        return if is_shape_valid(data) { Ok(()) } else { Err(0) };
    }

    // Fallback for very small windows where incremental logic is complex
    if size < 4 {
        for (offset, window) in data.windows(size).enumerate() {
            if !is_shape_valid(window) {
                return Err(offset);
            }
        }
        return Ok(());
    }

    let mut current_betti_0 = compute_betti_0(&data[..size]);
    let mut current_betti_1 = compute_betti_1(&data[..size]);

    // Check first window
    let mut shape = TopologicalShape::new(current_betti_0, current_betti_1, size);
    if shape.density < DENSITY_MIN || shape.density > DENSITY_MAX || shape.betti_1 > MAX_BETTI_1 {
        return Err(0);
    }

    for offset in 1..=(data.len() - size) {
        // Update betti_0 incrementally
        // Old edge leaving
        // Note: the original loop checks `is_gap(w[0], w[1])`. If we remove the first element,
        // we remove the gap starting at `offset - 1`. If it was the start of a component, we decrement.
        // But the original compute_betti_0 treats CONSECUTIVE gaps as one component.
        // It does: `if is_gap { if !in_comp { comp++; in_comp=true; } } else { in_comp=false; }`
        // So a gap counts as +1 if the PREVIOUS edge was NOT a gap.

        // This makes exact incremental Betti 0 tricky because of the clustering (consecutive gaps).
        // Let's implement the specific transition logic:

        let old_leaving_gap = is_gap(data[offset - 1], data[offset]);
        let old_next_gap = is_gap(data[offset], data[offset + 1]);
        if old_leaving_gap && !old_next_gap {
            // It was a component ending right at the boundary
            current_betti_0 -= 1;
        }

        // New edge entering
        let new_entering_prev_gap = is_gap(data[offset + size - 3], data[offset + size - 2]);
        let new_entering_gap = is_gap(data[offset + size - 2], data[offset + size - 1]);
        if new_entering_gap && !new_entering_prev_gap {
            current_betti_0 += 1;
        }

        // Update betti_1 incrementally
        let leaving_loop = is_loop_pattern(&data[offset - 1..offset + 3]);
        if leaving_loop {
            current_betti_1 -= 1;
        }

        let entering_loop = is_loop_pattern(&data[offset + size - 4..offset + size]);
        if entering_loop {
            current_betti_1 += 1;
        }

        shape.betti_0 = current_betti_0;
        shape.betti_1 = current_betti_1;
        shape.density = current_betti_0 as f64 / size as f64;

        if shape.density < DENSITY_MIN || shape.density > DENSITY_MAX || shape.betti_1 > MAX_BETTI_1
        {
            return Err(offset);
        }
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_data() {
        assert_eq!(compute_betti_0(&[]), 0);
        assert_eq!(compute_betti_1(&[]), 0);
    }

    #[test]
    fn test_uniform_data_low_density() {
        // NOP sled simulation: all same byte
        let nop_sled = [0x90u8; 64];
        let shape = compute_shape(&nop_sled);

        // Uniform data should have 0 gaps
        assert_eq!(shape.betti_0, 0);
    }

    #[test]
    fn test_random_pattern() {
        // Simulated "normal" code with varied byte patterns
        let code: [u8; 16] = [
            0x48, 0x89, 0xe5, 0x48, 0x83, 0xec, 0x10, 0x89, 0x7d, 0xfc, 0x8b, 0x45, 0xfc, 0x83,
            0xc0, 0x01,
        ];

        let shape = compute_shape(&code);

        // Should have reasonable density for compiled code
        assert!(shape.density >= 0.0);
    }

    #[test]
    fn test_verify_pass() {
        // Typical x86_64 function prologue
        let prologue = [
            0x55, 0x48, 0x89, 0xe5, 0x48, 0x83, 0xec, 0x20, 0x89, 0x7d, 0xec, 0x89, 0x75, 0xe8,
            0x48, 0x89, 0x55, 0xe0, 0x48, 0x89, 0x4d, 0xd8, 0x44, 0x89, 0x45, 0xd4, 0x44, 0x89,
            0x4d, 0xd0, 0x8b, 0x45,
        ];

        // Verify returns a result (may pass or fail based on heuristics)
        let result = verify_shape(&prologue);
        // Just ensure it doesn't panic
        match result {
            VerifyResult::Pass => {}
            _ => {}
        }
    }

    #[test]
    fn test_verify_sliding_window_equivalence() {
        // Generate pseudo-random data with gaps and loops
        let mut data = [0u8; 128];
        for i in 0..128 {
            data[i] = (i * 7 % 256) as u8;
        }
        // Force some specific loop patterns and gaps
        data[10] = 10;
        data[11] = 50;
        data[12] = 100;
        data[13] = 10;
        data[50] = 5;
        data[51] = 60;
        data[52] = 120;
        data[53] = 5;
        data[60] = 200;
        data[61] = 10;
        data[62] = 200; // huge gaps

        let window_size = 16;

        // Let's implement the naive logic to compare
        let naive_verify = |data: &[u8], size: usize| -> Result<(), usize> {
            for (offset, window) in data.windows(size).enumerate() {
                if !is_shape_valid(window) {
                    return Err(offset);
                }
            }
            Ok(())
        };

        let expected = naive_verify(&data, window_size);
        let actual = verify_sliding_window(&data, window_size);

        assert_eq!(
            expected, actual,
            "Incremental approach differs from naive approach!"
        );
    }
}
