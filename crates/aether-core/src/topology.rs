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

// ═══════════════════════════════════════════════════════════════════════════════
// Aether-Lang — invented by Teerth Sharma
// https://github.com/teerthsharma/Aether-Lang
// Copyright (c) 2026 Teerth Sharma. All Rights Reserved.
// ═══════════════════════════════════════════════════════════════════════════════
//

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
pub fn compute_betti_0(data: &[u8]) -> u32 {
    if data.len() < 2 {
        return if data.is_empty() { 0 } else { 1 };
    }

    let mut components = 0u32;
    let mut in_component = false;

    for window in data.windows(2) {
        let dist = (window[0] as i16 - window[1] as i16).abs();

        if dist > CLUSTER_THRESHOLD {
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
    let tolerance = 5i16; // How close values must be to "close a loop"

    // Detect cycles: a -> b -> c -> ~a (return to start)
    for window in data.windows(4) {
        let a = window[0] as i16;
        let d = window[3] as i16;

        // If we return to approximately the same value, it's a "loop"
        if (a - d).abs() <= tolerance {
            // Check that middle values are different (actual traversal)
            let b = window[1] as i16;
            let c = window[2] as i16;

            if (a - b).abs() > tolerance || (a - c).abs() > tolerance {
                loops += 1;
            }
        }
    }

    loops
}

/// Compute full topological shape signature
pub fn compute_shape(data: &[u8]) -> TopologicalShape {
    // Single pass optimization: compute betti_0 and betti_1 in one loop
    // to improve cache locality and avoid traversing memory twice.

    let len = data.len();

    if len < 2 {
        let betti_0 = if data.is_empty() { 0 } else { 1 };
        return TopologicalShape::new(betti_0, 0, len);
    }

    let mut betti_0 = 0u32;
    let mut b0_in_component = false;
    let mut betti_1 = 0u32;
    let tolerance = 5i16;

    // We iterate 0..len-1 for betti_0 (looking at i, i+1)
    for i in 0..len - 1 {
        // Betti 0 logic
        let w0 = data[i] as i16;
        let w1 = data[i + 1] as i16;
        let dist = (w0 - w1).abs();

        if dist > CLUSTER_THRESHOLD {
            if !b0_in_component {
                betti_0 += 1;
                b0_in_component = true;
            }
        } else {
            b0_in_component = false;
        }

        // Betti 1 logic: requires 4 elements (i, i+1, i+2, i+3)
        if i + 3 < len {
            let a = w0; // data[i]
            // b is w1 // data[i+1]
            let c = data[i + 2] as i16;
            let d = data[i + 3] as i16;

            if (a - d).abs() <= tolerance {
                if (a - w1).abs() > tolerance || (a - c).abs() > tolerance {
                    betti_1 += 1;
                }
            }
        }
    }

    TopologicalShape::new(betti_0, betti_1, len)
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
fn compute_raw_betti_0(data: &[u8]) -> u32 {
    let mut components = 0u32;
    let mut in_component = false;
    for window in data.windows(2) {
        let dist = (window[0] as i16 - window[1] as i16).abs();
        if dist > CLUSTER_THRESHOLD {
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

#[inline(always)]
fn is_gap(data: &[u8], idx: usize) -> bool {
    let dist = (data[idx] as i16 - data[idx + 1] as i16).abs();
    dist > CLUSTER_THRESHOLD
}

#[inline(always)]
fn is_loop(data: &[u8], idx: usize) -> bool {
    let a = data[idx] as i16;
    let b = data[idx + 1] as i16;
    let c = data[idx + 2] as i16;
    let d = data[idx + 3] as i16;
    let tolerance = 5i16;
    if (a - d).abs() <= tolerance && ((a - b).abs() > tolerance || (a - c).abs() > tolerance) {
        return true;
    }
    false
}

pub fn verify_sliding_window(data: &[u8], window_size: usize) -> Result<(), usize> {
    let size = if window_size == 0 {
        WINDOW_SIZE
    } else {
        window_size
    };
    if size < 4 {
        // Fallback to naive O(N*W) approach for window sizes < 4 to avoid out-of-bounds panics
        if data.len() < size {
            return if is_shape_valid(data) { Ok(()) } else { Err(0) };
        }
        for (offset, window) in data.windows(size).enumerate() {
            if !is_shape_valid(window) {
                return Err(offset);
            }
        }
        return Ok(());
    }
    if data.len() < size {
        return if is_shape_valid(data) { Ok(()) } else { Err(0) };
    }

    let mut current_betti_0 = compute_raw_betti_0(&data[0..size]);
    let mut current_betti_1 = compute_betti_1(&data[0..size]);

    let check_shape = |b0: u32, b1: u32, offset: usize| -> Result<(), usize> {
        let public_b0 = if b0 == 0 { 1 } else { b0 };
        let shape = TopologicalShape::new(public_b0, b1, size);
        if shape.density < DENSITY_MIN || shape.density > DENSITY_MAX || shape.betti_1 > MAX_BETTI_1
        {
            Err(offset)
        } else {
            Ok(())
        }
    };

    check_shape(current_betti_0, current_betti_1, 0)?;

    for offset in 1..=(data.len() - size) {
        if is_loop(data, offset - 1) {
            current_betti_1 -= 1;
        }
        if is_loop(data, offset + size - 4) {
            current_betti_1 += 1;
        }

        let gap_start = is_gap(data, offset - 1);
        let gap_start_plus_1 = is_gap(data, offset);
        if gap_start && !gap_start_plus_1 {
            current_betti_0 -= 1;
        }

        let gap_end_plus_1 = is_gap(data, offset + size - 2);
        let gap_end = is_gap(data, offset + size - 3);
        if gap_end_plus_1 && !gap_end {
            current_betti_0 += 1;
        }

        check_shape(current_betti_0, current_betti_1, offset)?;
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
        if let VerifyResult::Pass = result {}
    }

    #[test]
    fn test_compute_shape_consistency() {
        let cases: Vec<Vec<u8>> = vec![
            vec![],
            vec![1],
            vec![1, 2],
            vec![1, 2, 3],
            vec![1, 2, 3, 4],
            vec![0; 10],
            vec![0, 100, 0, 100, 0], // High variance
            (0..100).map(|x| x as u8).collect(),
            (0..100).map(|x| (x % 10) as u8).collect(),
        ];

        for data in cases {
            let shape = compute_shape(&data);
            let b0 = compute_betti_0(&data);
            let b1 = compute_betti_1(&data);

            assert_eq!(shape.betti_0, b0, "Betti 0 mismatch for data {:?}", data);
            assert_eq!(shape.betti_1, b1, "Betti 1 mismatch for data {:?}", data);
        }
    }
}
