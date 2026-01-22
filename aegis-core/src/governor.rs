//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Geometric Governor
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Implements the Adaptive Threshold Controller using Nonlinear Control Theory
//! (PID-on-Manifold).
//!
//! Mathematical Foundation:
//!   Error Signal: e(t) = R_target - Δ(t)/ε(t)
//!   Update Law: ε(t+1) = ε(t) + α·e(t) + β·de/dt
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

#![allow(dead_code)]

// use libm::fabs;

// ═══════════════════════════════════════════════════════════════════════════════
// Governor Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Target "Frame Rate" - ideal kernel tick rate in Hz
/// The governor tries to maintain this balance between responsiveness and efficiency
const TARGET_TICK_RATE: f64 = 1000.0;

/// Proportional gain (α)
/// Controls response to instantaneous error
const ALPHA: f64 = 0.01;

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
/// ε(t+1) = ε(t) + α·e(t) + β·de/dt
/// 
/// where:
///   e(t) = R_target - R_actual
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
    /// Implements the PID-on-Manifold control law:
    /// ```text
    /// ε(t+1) = ε(t) + α·e(t) + β·de/dt
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
        // Error = Target - Actual
        //
        // Positive error: We're too slow (need to lower ε, increase sensitivity)
        // Negative error: We're too fast (need to raise ε, decrease sensitivity)
        
        let error = TARGET_TICK_RATE - current_rate;
        
        // ═══════════════════════════════════════════════════════════════════
        // Step 3: Calculate Derivative of Error
        // ═══════════════════════════════════════════════════════════════════
        //
        // de/dt = (e(t) - e(t-1)) / dt
        //
        // This term helps dampen oscillations and provides predictive control.
        
        let d_error = (error - self.last_error) / dt;
        
        // ═══════════════════════════════════════════════════════════════════
        // Step 4: Apply Control Law (PD Controller)
        // ═══════════════════════════════════════════════════════════════════
        //
        // Δε = α·e + β·de/dt
        //
        // Note: We could add an integral term (γ·∫e·dt) for PID,
        // but PD is sufficient for our stability requirements.
        
        let adjustment = (self.alpha * error) + (self.beta * d_error);
        
        // ═══════════════════════════════════════════════════════════════════
        // Step 5: Update State
        // ═══════════════════════════════════════════════════════════════════
        
        self.epsilon += adjustment;
        self.last_error = error;
        self.adjustment_count += 1;
        
        // Update integral for potential future use
        self.integral_error += error * dt;
        
        // ═══════════════════════════════════════════════════════════════════
        // Step 6: Safety Clamps
        // ═══════════════════════════════════════════════════════════════════
        //
        // Prevent epsilon from:
        // - Vanishing (→ system never sleeps, 100% CPU)
        // - Exploding (→ system never wakes, misses events)
        
        if self.epsilon < EPSILON_MIN {
            self.epsilon = EPSILON_MIN;
        }
        if self.epsilon > EPSILON_MAX {
            self.epsilon = EPSILON_MAX;
        }
        
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
}
