//! ═══════════════════════════════════════════════════════════════════════════════
//! Governor Benchmark: PID-on-Manifold Stability Analysis
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Benchmarks for the GeometricGovernor adaptive threshold controller.
//! Validates:
//!   - Convergence under varying loads
//!   - Epsilon stability bounds
//!   - Lyapunov energy decay
//!   - Response dynamics at different target rates
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use libm::{sqrt, fabs};
use crate::governor::GeometricGovernor;

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for governor benchmarks
#[derive(Debug, Clone)]
pub struct GovernorBenchConfig {
    /// Number of iterations per test
    pub iterations: u32,
    /// Target rates to test (Hz)
    pub target_rates: [f64; 4],
    /// Time step (seconds)
    pub dt: f64,
    /// Epsilon tolerance for convergence
    pub convergence_epsilon: f64,
}

impl Default for GovernorBenchConfig {
    fn default() -> Self {
        Self {
            iterations: 10000,
            target_rates: [10.0, 100.0, 1000.0, 10000.0],
            dt: 0.001,
            convergence_epsilon: 0.01,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark Results
// ═══════════════════════════════════════════════════════════════════════════════

/// Results from governor stability benchmarks
#[derive(Debug, Clone)]
pub struct GovernorBenchResults {
    /// Whether epsilon stayed bounded [0.001, 10.0]
    pub epsilon_bounded: bool,
    /// Final epsilon value
    pub final_epsilon: f64,
    /// Iterations to reach steady state (-1 if never)
    pub convergence_iterations: i32,
    /// Peak overshoot ratio
    pub peak_overshoot: f64,
    /// Final Lyapunov energy V = e²/2
    pub final_lyapunov_energy: f64,
    /// Mean squared error at end
    pub mean_error_squared: f64,
    /// Epsilon trajectory samples (every 100 iterations)
    pub epsilon_trajectory: [f64; 100],
    /// Number of trajectory samples
    pub trajectory_len: usize,
}

impl GovernorBenchResults {
    fn new() -> Self {
        Self {
            epsilon_bounded: true,
            final_epsilon: 0.0,
            convergence_iterations: -1,
            peak_overshoot: 0.0,
            final_lyapunov_energy: 0.0,
            mean_error_squared: 0.0,
            epsilon_trajectory: [0.0; 100],
            trajectory_len: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Governor Benchmark Implementation
// ═══════════════════════════════════════════════════════════════════════════════

/// Run comprehensive governor stability benchmarks
pub struct GovernorBenchmark {
    config: GovernorBenchConfig,
}

impl GovernorBenchmark {
    pub fn new(config: GovernorBenchConfig) -> Self {
        Self { config }
    }

    /// Run all governor benchmarks
    pub fn run_all(&self) -> [GovernorBenchResults; 4] {
        [
            self.run_convergence_test(self.config.target_rates[0]),
            self.run_convergence_test(self.config.target_rates[1]),
            self.run_convergence_test(self.config.target_rates[2]),
            self.run_convergence_test(self.config.target_rates[3]),
        ]
    }

    /// Run convergence test at specific target rate
    pub fn run_convergence_test(&self, target_rate: f64) -> GovernorBenchResults {
        let mut gov = GeometricGovernor::new();
        let mut results = GovernorBenchResults::new();
        
        let initial_epsilon = gov.epsilon();
        let mut sum_error_sq = 0.0;
        let mut converged = false;
        let mut prev_error = 0.0;
        
        // Track epsilon bounds
        const EPSILON_MIN: f64 = 0.001;
        const EPSILON_MAX: f64 = 10.0;
        
        for i in 0..self.config.iterations {
            // Simulate deviation based on target rate
            // deviation = epsilon * target_rate (to achieve target rate)
            let simulated_deviation = gov.epsilon() * target_rate;
            
            // Adapt governor
            let new_epsilon = gov.adapt(simulated_deviation, self.config.dt);
            
            // Check bounds
            if new_epsilon < EPSILON_MIN || new_epsilon > EPSILON_MAX {
                results.epsilon_bounded = false;
            }
            
            // Track overshoot
            let overshoot = (new_epsilon - initial_epsilon) / initial_epsilon;
            if overshoot > results.peak_overshoot {
                results.peak_overshoot = overshoot;
            }
            
            // Compute error and Lyapunov energy
            let error = gov.last_error();
            let lyapunov_energy = error * error / 2.0;
            sum_error_sq += error * error;
            
            // Check convergence (error stable and small)
            if !converged && i > 100 {
                let error_change = fabs(error - prev_error);
                if fabs(error) < self.config.convergence_epsilon && 
                   error_change < self.config.convergence_epsilon * 0.1 {
                    results.convergence_iterations = i as i32;
                    converged = true;
                }
            }
            prev_error = error;
            
            // Sample trajectory
            if i % 100 == 0 && results.trajectory_len < 100 {
                results.epsilon_trajectory[results.trajectory_len] = new_epsilon;
                results.trajectory_len += 1;
            }
            
            // Save final values
            if i == self.config.iterations - 1 {
                results.final_epsilon = new_epsilon;
                results.final_lyapunov_energy = lyapunov_energy;
                results.mean_error_squared = sum_error_sq / (i + 1) as f64;
            }
        }
        
        results
    }

    /// Run Lyapunov stability analysis
    pub fn run_lyapunov_analysis(&self) -> LyapunovResults {
        let mut gov = GeometricGovernor::new();
        let mut results = LyapunovResults::new();
        
        let mut prev_energy = f64::MAX;
        let mut energy_decreasing_count = 0u32;
        
        for i in 0..self.config.iterations {
            // Random-ish deviation to stress test
            let deviation = 0.5 + 0.4 * ((i as f64 * 0.1).sin());
            
            gov.adapt(deviation, self.config.dt);
            
            let error = gov.last_error();
            let energy = error * error / 2.0;
            
            // Track energy decrease
            if energy < prev_energy {
                energy_decreasing_count += 1;
            }
            prev_energy = energy;
            
            // Record peak energy
            if energy > results.peak_energy {
                results.peak_energy = energy;
            }
            
            results.final_energy = energy;
        }
        
        results.energy_decrease_ratio = energy_decreasing_count as f64 / self.config.iterations as f64;
        results.is_stable = results.energy_decrease_ratio > 0.6; // >60% decreasing = stable
        
        results
    }

    /// Run stress test with extreme deviations
    pub fn run_stress_test(&self) -> StressTestResults {
        let mut gov = GeometricGovernor::new();
        let mut results = StressTestResults::new();
        
        // Phase 1: Very low load
        for _ in 0..1000 {
            gov.adapt(0.001, self.config.dt);
        }
        results.epsilon_after_low_load = gov.epsilon();
        
        // Phase 2: Sudden spike
        for _ in 0..100 {
            gov.adapt(1000.0, self.config.dt);
        }
        results.epsilon_after_spike = gov.epsilon();
        
        // Phase 3: Recovery
        for _ in 0..1000 {
            gov.adapt(1.0, self.config.dt);
        }
        results.epsilon_after_recovery = gov.epsilon();
        
        // Check all values are bounded
        results.all_bounded = 
            results.epsilon_after_low_load >= 0.001 && results.epsilon_after_low_load <= 10.0 &&
            results.epsilon_after_spike >= 0.001 && results.epsilon_after_spike <= 10.0 &&
            results.epsilon_after_recovery >= 0.001 && results.epsilon_after_recovery <= 10.0;
        
        results
    }
}

/// Results from Lyapunov stability analysis
#[derive(Debug, Clone)]
pub struct LyapunovResults {
    pub peak_energy: f64,
    pub final_energy: f64,
    pub energy_decrease_ratio: f64,
    pub is_stable: bool,
}

impl LyapunovResults {
    fn new() -> Self {
        Self {
            peak_energy: 0.0,
            final_energy: 0.0,
            energy_decrease_ratio: 0.0,
            is_stable: false,
        }
    }
}

/// Results from stress testing
#[derive(Debug, Clone)]
pub struct StressTestResults {
    pub epsilon_after_low_load: f64,
    pub epsilon_after_spike: f64,
    pub epsilon_after_recovery: f64,
    pub all_bounded: bool,
}

impl StressTestResults {
    fn new() -> Self {
        Self {
            epsilon_after_low_load: 0.0,
            epsilon_after_spike: 0.0,
            epsilon_after_recovery: 0.0,
            all_bounded: false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_governor_convergence_10hz() {
        let bench = GovernorBenchmark::new(GovernorBenchConfig::default());
        let result = bench.run_convergence_test(10.0);
        
        assert!(result.epsilon_bounded, "Epsilon should stay bounded");
        println!("10Hz: final_epsilon={:.6}, convergence_iter={}", 
            result.final_epsilon, result.convergence_iterations);
    }

    #[test]
    fn test_governor_convergence_1000hz() {
        let bench = GovernorBenchmark::new(GovernorBenchConfig::default());
        let result = bench.run_convergence_test(1000.0);
        
        assert!(result.epsilon_bounded, "Epsilon should stay bounded");
        println!("1000Hz: final_epsilon={:.6}, convergence_iter={}", 
            result.final_epsilon, result.convergence_iterations);
    }

    #[test]
    fn test_lyapunov_stability() {
        let bench = GovernorBenchmark::new(GovernorBenchConfig::default());
        let result = bench.run_lyapunov_analysis();
        
        assert!(result.is_stable, "System should be Lyapunov stable");
        println!("Lyapunov: energy_decrease_ratio={:.2}%, final_energy={:.6}", 
            result.energy_decrease_ratio * 100.0, result.final_energy);
    }

    #[test]
    fn test_stress_resilience() {
        let bench = GovernorBenchmark::new(GovernorBenchConfig::default());
        let result = bench.run_stress_test();
        
        assert!(result.all_bounded, "All epsilon values should be bounded");
        println!("Stress: low={:.4}, spike={:.4}, recovery={:.4}", 
            result.epsilon_after_low_load, 
            result.epsilon_after_spike, 
            result.epsilon_after_recovery);
    }
}
