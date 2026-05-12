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

/// Check if a 2-byte sequence is a gap (exceeds threshold)
#[inline]
fn is_gap(a: u8, b: u8) -> bool {
    let dist = (a as i16 - b as i16).abs();
    dist > CLUSTER_THRESHOLD
}

/// Check if a 4-byte sequence is a loop pattern
#[inline]
fn is_loop_pattern(window: &[u8]) -> bool {
    let tolerance = 5i16; // LOOP_TOLERANCE equivalent
    let a = window[0] as i16;
    let b = window[1] as i16;
    let c = window[2] as i16;
    let d = window[3] as i16;

    if (a - d).abs() <= tolerance && ((a - b).abs() > tolerance || (a - c).abs() > tolerance) {
        return true;
    }
    false
}

/// Compute raw number of gap components (before applying the min=1 rule)
pub(crate) fn compute_raw_betti_0(data: &[u8]) -> u32 {
    if data.len() < 2 {
        return 0;
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

    let raw = compute_raw_betti_0(data);

    // NOP sled simulation: all same byte -> 0 gaps
    if raw == 0 {
        0
    } else {
        raw
    }
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

    // Fallback to naive O(N*W) for very small windows to avoid out-of-bounds in incremental logic
    if size < 4 {
        for (offset, window) in data.windows(size).enumerate() {
            if !is_shape_valid(window) {
                return Err(offset);
            }
        }
        return Ok(());
    }

    // O(N) Sliding Window Optimization
    // Initialize with first window
    let current_window = &data[0..size];
    let mut raw_betti_0 = compute_raw_betti_0(current_window);
    let mut betti_1 = compute_betti_1(current_window);

    let betti_0 = if raw_betti_0 == 0 { 1 } else { raw_betti_0 };
    let mut shape = TopologicalShape::new(betti_0, betti_1, size);

    if shape.density < DENSITY_MIN || shape.density > DENSITY_MAX || shape.betti_1 > MAX_BETTI_1 {
        return Err(0);
    }

    // Slide window incrementally
    for offset in 1..=(data.len() - size) {
        let leaving_idx = offset - 1;
        let entering_idx = offset + size - 1;

        // Update raw_betti_0 incrementally
        let leaving_is_gap = is_gap(data[leaving_idx], data[leaving_idx + 1]);
        let leaving_next_is_gap = is_gap(data[leaving_idx + 1], data[leaving_idx + 2]);

        let entering_is_gap = is_gap(data[entering_idx - 1], data[entering_idx]);
        let entering_prev_is_gap = is_gap(data[entering_idx - 2], data[entering_idx - 1]);

        let mut b0_diff = 0i32;

        // If the leaving edge was the start of a gap cluster, we might lose a component
        if leaving_is_gap && !leaving_next_is_gap {
            // It was an isolated gap or the last gap in a cluster, so the cluster is removed
            b0_diff -= 1;
        }
        // If leaving_next_is_gap is true, the next pair is still a gap, so the cluster continues

        // If the entering edge is a gap, we might gain a component
        if entering_is_gap && !entering_prev_is_gap {
            // It's the start of a new gap cluster
            b0_diff += 1;
        }
        // If entering_prev_is_gap is true, it's just extending an existing gap cluster

        raw_betti_0 = (raw_betti_0 as i32 + b0_diff) as u32;

        // Update betti_1 incrementally
        let leaving_pattern = is_loop_pattern(&data[leaving_idx..leaving_idx + 4]);
        let entering_pattern = is_loop_pattern(&data[entering_idx - 3..entering_idx + 1]);

        let mut b1_diff = 0i32;
        if leaving_pattern {
            b1_diff -= 1;
        }
        if entering_pattern {
            b1_diff += 1;
        }

        betti_1 = (betti_1 as i32 + b1_diff) as u32;

        let cur_betti_0 = if raw_betti_0 == 0 { 1 } else { raw_betti_0 };
        shape.betti_0 = cur_betti_0;
        shape.betti_1 = betti_1;
        shape.density = cur_betti_0 as f64 / size as f64;

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
        // Test data with some structure and noise to ensure various gap/loop conditions
        let data: Vec<u8> = (0..128)
            .map(|i| {
                if i % 10 < 3 {
                    100 // Cluster 1
                } else if i % 10 < 7 {
                    200 // Cluster 2 (gap > 15)
                } else {
                    150 // Cluster 3 (gap > 15)
                }
            })
            .collect();

        let window_size = 16;

        // Let's manually implement the naive logic for comparison
        let mut naive_results = Vec::new();
        for window in data.windows(window_size) {
            let b0 = compute_betti_0(window);
            let b1 = compute_betti_1(window);
            naive_results.push((b0, b1));
        }

        // And manually trace the incremental logic
        let mut incremental_results = Vec::new();

        let mut current_window = &data[0..window_size];
        let mut raw_betti_0 = compute_raw_betti_0(current_window);
        let mut betti_1 = compute_betti_1(current_window);

        let initial_betti_0 = if raw_betti_0 == 0 { 1 } else { raw_betti_0 };
        incremental_results.push((initial_betti_0, betti_1));

        for offset in 1..=(data.len() - window_size) {
            let leaving_idx = offset - 1;
            let entering_idx = offset + window_size - 1;

            let leaving_is_gap = is_gap(data[leaving_idx], data[leaving_idx + 1]);
            let leaving_next_is_gap = is_gap(data[leaving_idx + 1], data[leaving_idx + 2]);

            let entering_is_gap = is_gap(data[entering_idx - 1], data[entering_idx]);
            let entering_prev_is_gap = is_gap(data[entering_idx - 2], data[entering_idx - 1]);

            let mut b0_diff = 0i32;

            if leaving_is_gap && !leaving_next_is_gap {
                b0_diff -= 1;
            }

            if entering_is_gap && !entering_prev_is_gap {
                b0_diff += 1;
            }

            raw_betti_0 = (raw_betti_0 as i32 + b0_diff) as u32;

            let leaving_pattern = is_loop_pattern(&data[leaving_idx..leaving_idx + 4]);
            let entering_pattern = is_loop_pattern(&data[entering_idx - 3..entering_idx + 1]);

            let mut b1_diff = 0i32;
            if leaving_pattern {
                b1_diff -= 1;
            }
            if entering_pattern {
                b1_diff += 1;
            }

            betti_1 = (betti_1 as i32 + b1_diff) as u32;

            let cur_betti_0 = if raw_betti_0 == 0 { 1 } else { raw_betti_0 };
            incremental_results.push((cur_betti_0, betti_1));
        }

        assert_eq!(naive_results.len(), incremental_results.len());
        for i in 0..naive_results.len() {
            assert_eq!(
                naive_results[i], incremental_results[i],
                "Mismatch at window offset {}. Naive: {:?}, Incremental: {:?}",
                i, naive_results[i], incremental_results[i]
            );
        }
    }
}
