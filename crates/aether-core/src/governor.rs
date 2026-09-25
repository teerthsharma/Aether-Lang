//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Geometric Governor
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Implements the Adaptive Threshold Controller using Nonlinear Control Theory
//! (PID-on-Manifold).
//!
//! Mathematical Foundation:
//!   Error Signal: e(t) = clamp(1 - (Δ(t)/ε(t)) / R_target, -1, 1)
//!   Update Law: ln ε(t+1) = ln ε(t) - α·e(t) - β·(e(t) - e(t-1))
//!   Fixed point: ε* = Δ / R_target, reached geometrically for a steady Δ
//!
//! Intuition:
//!   - If kernel wakes too often (e < 0): raise ε (decrease sensitivity)
//!   - If kernel is sluggish (e > 0): lower ε (increase sensitivity)
//!
//! This ensures the kernel doesn't:
//!   - "Stutter" (thrash) during high load
//!   - "Sleep" during critical transients
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
// Governor Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Target "Frame Rate" - ideal kernel tick rate in Hz
/// The governor tries to maintain this balance between responsiveness and efficiency
const TARGET_TICK_RATE: f64 = 1000.0;

/// Proportional gain (α)
/// Controls response to instantaneous error (dimensionless: e is a fraction of
/// the target rate, and ln ε moves by at most α per step)
const ALPHA: f64 = 0.25;

/// Derivative gain (β)
/// Controls response to rate of change of error
/// Helps dampen oscillations
const BETA: f64 = 0.05;

/// Minimum allowed epsilon (prevents runaway sensitivity)
const EPSILON_MIN: f64 = 0.001;

/// Maximum allowed epsilon (prevents system from sleeping too long)
const EPSILON_MAX: f64 = 10.0;

/// Default initial epsilon
const EPSILON_INITIAL: f64 = 0.1;

// ═══════════════════════════════════════════════════════════════════════════════
// Geometric Governor
// ═══════════════════════════════════════════════════════════════════════════════

/// The Geometric Governor: Adaptive Threshold Controller
///
/// This is the "How" of AEGIS - it dynamically adjusts the sensitivity
/// threshold ε(t) based on system behavior, using classical nonlinear
/// control theory (PID controller on the state manifold).
///
/// # Control Law
/// ```text
/// ln ε(t+1) = ln ε(t) - α·e(t) - β·(e(t) - e(t-1))
///
/// where:
///   e(t) = clamp(1 - R_actual / R_target, -1, 1)
///   R_actual = Δ/ε (effective "frame rate")
/// ```
///
/// # Stability Properties
/// - Bounded: ε ∈ [EPSILON_MIN, EPSILON_MAX]
/// - Asymptotically stable around R_target
/// - Damped oscillation via derivative term
#[derive(Debug, Clone)]
pub struct GeometricGovernor {
    /// Current adaptive threshold ε(t)
    epsilon: f64,

    /// Previous error (for derivative calculation)
    last_error: f64,

    /// Accumulated integral error (for potential PID extension)
    integral_error: f64,

    /// Number of adjustments made (for statistics)
    adjustment_count: u64,

    /// Custom gains (optional override)
    alpha: f64,
    beta: f64,
}

impl GeometricGovernor {
    /// Create a new governor with default parameters
    pub fn new() -> Self {
        Self {
            epsilon: EPSILON_INITIAL,
            last_error: 0.0,
            integral_error: 0.0,
            adjustment_count: 0,
            alpha: ALPHA,
            beta: BETA,
        }
    }

    /// Create a governor with custom initial epsilon
    pub fn with_epsilon(epsilon: f64) -> Self {
        let mut gov = Self::new();
        gov.epsilon = epsilon.clamp(EPSILON_MIN, EPSILON_MAX);
        gov
    }

    /// Create a governor with custom gains
    pub fn with_gains(alpha: f64, beta: f64) -> Self {
        Self {
            epsilon: EPSILON_INITIAL,
            last_error: 0.0,
            integral_error: 0.0,
            adjustment_count: 0,
            alpha,
            beta,
        }
    }

    /// Get current epsilon value
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Get adjustment statistics
    pub fn adjustment_count(&self) -> u64 {
        self.adjustment_count
    }

    /// Adapt epsilon based on observed deviation
    ///
    /// Implements the control law on ln ε:
    /// ```text
    /// ln ε(t+1) = ln ε(t) - α·e(t) - β·(e(t) - e(t-1))
    /// ```
    ///
    /// # Arguments
    /// * `deviation_delta` - The observed deviation Δ(t)
    /// * `dt` - Time delta since last adaptation (in seconds)
    ///
    /// # Returns
    /// The new epsilon value
    pub fn adapt(&mut self, deviation_delta: f64, dt: f64) -> f64 {
        // Prevent division by zero
        if dt <= 0.0 || self.epsilon <= 0.0 {
            return self.epsilon;
        }

        // ═══════════════════════════════════════════════════════════════════
        // Step 1: Calculate the "Effective Rate" we are seeing
        // ═══════════════════════════════════════════════════════════════════
        //
        // The effective rate is how often the kernel WOULD wake up
        // given the current deviation and threshold.
        //
        // Rate = Δ / ε
        //
        // If Δ is high relative to ε, we're waking up often.
        // If Δ is low relative to ε, we're barely waking up.

        let current_rate = deviation_delta / self.epsilon;

        // ═══════════════════════════════════════════════════════════════════
        // Step 2: Calculate Control Error
        // ═══════════════════════════════════════════════════════════════════
        //
        // Positive error: We're too slow (need to lower ε, increase sensitivity)
        // Negative error: We're too fast (need to raise ε, decrease sensitivity)

        // Error is the rate miss as a fraction of the target, clamped to [-1, 1]:
        //   e = 1 - R_actual / R_target
        // The raw miss (R_target - R_actual) is in Hz, around 1000, while epsilon
        // lives in [0.001, 10], so alpha * e moved epsilon across its whole range
        // in one step and it alternated between the two clamps.
        let error = (1.0 - current_rate / TARGET_TICK_RATE).clamp(-1.0, 1.0);

        // Per-step change in error. Dividing by dt (about 0.001 s) multiplied the
        // derivative kick by 1000; the step count, not wall time, is what the
        // discrete update integrates over.
        let d_error = error - self.last_error;

        // Update ln(epsilon), not epsilon. Near the fixed point
        // epsilon* = delta / R_target, e ~ ln(epsilon / epsilon*), so
        //   u(k+1) = (1 - alpha - beta) u(k) + beta u(k-1),  u = ln(epsilon / epsilon*)
        // whose roots lie inside the unit circle for the default gains.
        let adjustment = (self.alpha * error) + (self.beta * d_error);
        self.epsilon *= libm::exp(-adjustment);
        self.last_error = error;
        self.adjustment_count += 1;
        self.integral_error += error * dt;

        // Safety clamps: epsilon* outside the band settles on the nearer bound.
        self.epsilon = self.epsilon.clamp(EPSILON_MIN, EPSILON_MAX);

        self.epsilon
    }

    /// Check if a deviation exceeds the current threshold
    ///
    /// This is the core decision function: Δ(t) ≥ ε(t)?
    pub fn should_trigger(&self, deviation: f64) -> bool {
        deviation >= self.epsilon
    }

    /// Reset the governor to initial state
    pub fn reset(&mut self) {
        self.epsilon = EPSILON_INITIAL;
        self.last_error = 0.0;
        self.integral_error = 0.0;
        self.adjustment_count = 0;
    }

    /// Get the current error (for diagnostics)
    pub fn last_error(&self) -> f64 {
        self.last_error
    }
}

impl Default for GeometricGovernor {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_epsilon() {
        let gov = GeometricGovernor::new();
        assert!((gov.epsilon() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_epsilon_clamped_min() {
        let mut gov = GeometricGovernor::new();

        // Drive epsilon down with high deviation (system waking too often)
        for _ in 0..10000 {
            gov.adapt(1000.0, 0.001);
        }

        assert!(gov.epsilon() >= EPSILON_MIN);
    }

    #[test]
    fn test_epsilon_clamped_max() {
        let mut gov = GeometricGovernor::new();

        // Drive epsilon up with low deviation (system sleeping too much)
        for _ in 0..10000 {
            gov.adapt(0.0001, 0.001);
        }

        assert!(gov.epsilon() <= EPSILON_MAX);
    }

    #[test]
    fn test_high_load_raises_epsilon() {
        let mut gov = GeometricGovernor::new();
        let initial = gov.epsilon();

        // High deviation = waking too often = raise epsilon
        gov.adapt(10000.0, 0.001);

        // Epsilon should increase (after initial transient)
        // Note: May need multiple iterations due to derivative term
        for _ in 0..10 {
            gov.adapt(10000.0, 0.001);
        }

        assert!(gov.epsilon() > initial);
    }

    #[test]
    fn test_trigger_threshold() {
        let gov = GeometricGovernor::with_epsilon(0.5);

        assert!(!gov.should_trigger(0.4));
        assert!(gov.should_trigger(0.5));
        assert!(gov.should_trigger(0.6));
    }
    #[test]
    fn test_steady_deviation_converges_to_equilibrium() {
        // The fixed point of e = 0 is epsilon* = delta / TARGET_TICK_RATE. A steady
        // delta must settle there, not alternate between the two clamps.
        for delta in [5.0, 50.0, 2000.0] {
            let target = delta / TARGET_TICK_RATE;
            let mut gov = GeometricGovernor::new();
            for _ in 0..200 {
                gov.adapt(delta, 0.001);
            }
            let settled = gov.epsilon();
            assert!(
                ((settled - target) / target).abs() < 1e-6,
                "delta {delta}: epsilon {settled}, expected {target}"
            );
            // Further steps must not move it: a clamp-to-clamp oscillation would.
            gov.adapt(delta, 0.001);
            assert!(((gov.epsilon() - settled) / target).abs() < 1e-6);
        }
    }
}
