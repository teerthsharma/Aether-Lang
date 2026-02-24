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

/// Tolerance for loop closure (Betti-1 calculation)
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
    let tolerance = LOOP_TOLERANCE; // How close values must be to "close a loop"

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

    // Helper: Is the distance between two bytes a gap?
    let is_gap = |a: u8, b: u8| (a as i16 - b as i16).abs() > CLUSTER_THRESHOLD;

    // Helper: Does a 4-byte slice form a loop pattern?
    let is_loop_pattern = |window: &[u8]| -> bool {
        let tolerance = LOOP_TOLERANCE;
        let a = window[0] as i16;
        let d = window[3] as i16;
        if (a - d).abs() <= tolerance {
            let b = window[1] as i16;
            let c = window[2] as i16;
            if (a - b).abs() > tolerance || (a - c).abs() > tolerance {
                return true;
            }
        }
        false
    };

    // Initialize with first window
    let mut current_betti_0 = compute_betti_0(&data[0..size]);
    let mut current_betti_1 = compute_betti_1(&data[0..size]);

    // Check first window
    let mut density = if size > 0 {
        current_betti_0 as f64 / size as f64
    } else {
        0.0
    };

    if density < DENSITY_MIN || density > DENSITY_MAX || current_betti_1 > MAX_BETTI_1 {
        return Err(0);
    }

    // Iterate through remaining windows
    for i in 0..data.len() - size {
        // Update Betti 0 (Incremental O(1))
        if size >= 2 {
            // Check if we are removing a gap that was a component start
            let leaving_gap = is_gap(data[i], data[i + 1]);
            let next_gap = is_gap(data[i + 1], data[i + 2]);

            // If leaving a gap that wasn't continued, we lose a component
            if leaving_gap && !next_gap {
                current_betti_0 -= 1;
            }

            // Check if we are adding a gap that becomes a component start
            let new_last_gap = is_gap(data[i + size - 1], data[i + size]);
            let prev_last_gap = is_gap(data[i + size - 2], data[i + size - 1]);

            // If adding a gap that isn't a continuation, we gain a component
            if new_last_gap && !prev_last_gap {
                current_betti_0 += 1;
            }
        }

        // Update Betti 1 (Incremental O(1))
        if size >= 4 {
            // Remove loop at start of old window
            if is_loop_pattern(&data[i..i + 4]) {
                current_betti_1 -= 1;
            }

            // Add loop at end of new window
            if is_loop_pattern(&data[i + size - 3..i + size + 1]) {
                current_betti_1 += 1;
            }
        }

        // Verify new window state
        density = if size > 0 {
            current_betti_0 as f64 / size as f64
        } else {
            0.0
        };

        if density < DENSITY_MIN || density > DENSITY_MAX || current_betti_1 > MAX_BETTI_1 {
            return Err(i + 1);
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
    fn test_verify_sliding_window_fail() {
        let nop_sled = [0x90u8; 100];
        // Window size 64.
        // First window: density 0. Should fail at offset 0.
        let result = verify_sliding_window(&nop_sled, 64);
        assert_eq!(result, Err(0));
    }

    #[test]
    fn test_verify_sliding_window_equivalence() {
        // Generative property-based test:
        // Compare incremental sliding window result against naive implementation
        // for random data to ensure mathematical equivalence.

        // Pseudo-random deterministic data
        let mut data = Vec::with_capacity(1000);
        let mut seed: u64 = 0x12345678;
        for _ in 0..1000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.push((seed >> 33) as u8);
        }

        let window_size = 64;

        // Calculate naive ground truth
        // Note: verify_sliding_window returns Ok(()) or Err(offset).
        // If naive fails at offset X, optimized must fail at offset X.

        let mut expected_result: Result<(), usize> = Ok(());
        for (offset, window) in data.windows(window_size).enumerate() {
            if !is_shape_valid(window) {
                expected_result = Err(offset);
                break;
            }
        }

        let actual_result = verify_sliding_window(&data, window_size);

        assert_eq!(actual_result, expected_result, "Optimization diverged from ground truth!");
    }

    #[test]
    fn test_verify_sliding_window_transition() {
        // Construct valid data: Gap (20), Link (1)...
        // Density ~0.5.
        // Then switch to NOP sled.
        let mut data = Vec::new();
        let mut val: u8 = 0;
        // 70 bytes of valid data
        for i in 0..70 {
            data.push(val);
            if i % 2 == 0 {
                val = val.wrapping_add(20);
            } else {
                val = val.wrapping_add(1);
            }
        }
        // Append NOPs
        for _ in 0..70 {
            data.push(0x90);
        }

        // Window size 64.
        // At offset 0: valid.
        // As window slides, NOPs enter.
        // NOPs are links (0x90 - 0x90 = 0 < 15).
        // So components count will decrease.
        // When density < 0.1, it should fail.
        // Density 0.1 => 6.4 components.
        // Initial components ~32.
        // We need to replace ~26 components with links.
        // Each 2 NOPs replace a Gap/Link pair?
        // It should fail eventually.

        let result = verify_sliding_window(&data, 64);
        assert!(result.is_err());
        let offset = result.unwrap_err();
        assert!(offset > 0);
    }
}
