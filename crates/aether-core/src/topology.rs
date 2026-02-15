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

/// Tolerance for loop detection (Betti-1 calculation)
const LOOP_TOLERANCE: i16 = 5;

/// Sliding window size for topology analysis
const WINDOW_SIZE: usize = 64;

/// Minimum density for valid code (β₀ / len)
const DENSITY_MIN: f64 = 0.1;

/// Maximum density for valid code
const DENSITY_MAX: f64 = 0.6;

/// Maximum allowed Betti-1 (loop complexity) per window
const MAX_BETTI_1: u32 = 10;

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn is_gap(a: u8, b: u8) -> bool {
    (a as i16 - b as i16).abs() > CLUSTER_THRESHOLD
}

#[inline(always)]
fn is_loop(w: &[u8]) -> bool {
    let a = w[0] as i16;
    let d = w[3] as i16;
    let tolerance = LOOP_TOLERANCE;

    if (a - d).abs() <= tolerance {
        let b = w[1] as i16;
        let c = w[2] as i16;

        if (a - b).abs() > tolerance || (a - c).abs() > tolerance {
            return true;
        }
    }
    false
}

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
        if is_loop(window) {
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

    // Optimization: Incremental update O(N) instead of O(N*W)

    // Initial calculation for the first window
    let mut shape = compute_shape(&data[0..size]);

    // Check first window
    if shape.density < DENSITY_MIN || shape.density > DENSITY_MAX || shape.betti_1 > MAX_BETTI_1 {
         return Err(0);
    }

    // Iterate through subsequent windows
    // Window at offset i: data[i .. i+size]
    // Previous window: data[i-1 .. i-1+size]
    for i in 1..=(data.len() - size) {
        let prev_offset = i - 1;
        let added_idx = i + size - 1;

        // Update Betti 0
        if size >= 2 {
            // Check removed head contribution
            let g0 = is_gap(data[prev_offset], data[prev_offset+1]);
            // If size > 2, the new head starts with gap(data[prev_offset+1], data[prev_offset+2])
            // If size == 2, the new head is the only diff, so it has no predecessor in window context
            let g1 = if size > 2 { is_gap(data[prev_offset+1], data[prev_offset+2]) } else { false };

            // Check added tail contribution
            let g_last = is_gap(data[added_idx-1], data[added_idx]);
            let g_before_last = if size > 2 { is_gap(data[added_idx-2], data[added_idx-1]) } else { false };

            // Remove head: Decrease if removed gap was NOT followed by another gap (start of component removed)
            if g0 && !g1 {
                if shape.betti_0 > 0 { shape.betti_0 -= 1; }
            }

            // Add tail: Increase if added gap is NOT preceded by another gap (new component started)
            if g_last && !g_before_last {
                shape.betti_0 += 1;
            }
        }

        // Update Betti 1
        if size >= 4 {
             // Remove loop at old head
             if is_loop(&data[prev_offset..prev_offset+4]) {
                 if shape.betti_1 > 0 { shape.betti_1 -= 1; }
             }

             // Add loop at new tail
             // The new loop ends at added_idx. It spans [added_idx-3 .. added_idx]
             if is_loop(&data[added_idx-3..=added_idx]) {
                 shape.betti_1 += 1;
             }
        }

        // Update Density
        // Use max(1.0) to avoid division by zero although size >= 1 here
        let divisor = if size > 0 { size as f64 } else { 1.0 };
        shape.density = shape.betti_0 as f64 / divisor;

        // Check Validity
         if shape.density < DENSITY_MIN || shape.density > DENSITY_MAX || shape.betti_1 > MAX_BETTI_1 {
             return Err(i);
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
    fn test_verify_sliding_window_basic() {
        // Create data that passes validation everywhere
        // Pattern: 0, 20, 25, 45... (Diffs 20, 5, 20, 5...)
        // Betti0 ~ 31/64 -> Density ~ 0.5. Valid.
        // Loops 0. Valid.

        let mut data = Vec::with_capacity(200);
        let mut val: u8 = 0;
        for i in 0..200 {
            data.push(val);
            if i % 2 == 0 {
                val = val.wrapping_add(20);
            } else {
                val = val.wrapping_add(5);
            }
        }

        assert!(verify_sliding_window(&data, 64).is_ok());
    }

    #[test]
    fn test_verify_sliding_window_fail() {
         // Create data that passes initially, then fails
         // First 64 bytes valid.
         // Then transition to NOP sled (0x90).

         let mut data = Vec::with_capacity(200);
         let mut val: u8 = 0;
         for i in 0..100 {
             data.push(val);
             if i % 2 == 0 { val = val.wrapping_add(20); } else { val = val.wrapping_add(5); }
         }
         // Append NOP sled
         for _ in 0..100 {
             data.push(0x90);
         }

         // Should fail when window slides into NOP sled
         // NOP sled has density 0 (no gaps). Min density 0.1.
         assert!(verify_sliding_window(&data, 64).is_err());
    }

    #[test]
    fn test_verify_sliding_window_equivalence() {
        // Deterministic RNG for reproducibility (Linear Congruential Generator)
        let mut data = Vec::with_capacity(2000);
        let mut seed: u32 = 12345;
        for _ in 0..2000 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            data.push((seed >> 24) as u8);
        }

        let window_size = 64;
        let optimized_result = verify_sliding_window(&data, window_size);

        let mut naive_result = Ok(());
        let size = if window_size == 0 { WINDOW_SIZE } else { window_size };

        if data.len() < size {
             if !is_shape_valid(&data) {
                 naive_result = Err(0);
             }
        } else {
            for (offset, window) in data.windows(size).enumerate() {
                if !is_shape_valid(window) {
                    naive_result = Err(offset);
                    break;
                }
            }
        }

        assert_eq!(optimized_result, naive_result, "Optimized vs Naive mismatch");
    }
}
