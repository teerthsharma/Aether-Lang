//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Sparse-Event Scheduler
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! The core scheduler that only wakes the CPU on significant state deviation.
//!
//! Flow:
//!   1. On IRQ: Update State μ(t)
//!   2. Check: Is ||μ(t) - μ_last||₂ ≥ ε?
//!   3. Yes: Wake CPU, process, update ε, set μ_last = μ(t)
//!   4. No: Increment entropy_pool, return to WFI
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::state::SystemState;
use crate::governor::GeometricGovernor;

// ═══════════════════════════════════════════════════════════════════════════════
// Scheduler Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Default time delta for governor adaptation (1ms = 0.001s)
const DEFAULT_DT: f64 = 0.001;

/// Entropy multiplier for RNG seeding
const ENTROPY_MULTIPLIER: u64 = 6364136223846793005;

// ═══════════════════════════════════════════════════════════════════════════════
// Sparse Scheduler
// ═══════════════════════════════════════════════════════════════════════════════

/// The Sparse-Event Scheduler
/// 
/// This is the heart of AEGIS. Unlike traditional schedulers that tick at
/// fixed intervals, we only wake when the state trajectory deviates
/// significantly from equilibrium.
/// 
/// # Key Invariant
/// CPU halts unless: ||μ(t) - μ(t_last)||₂ ≥ ε(t)
#[derive(Debug)]
pub struct SparseScheduler<const D: usize> {
    /// The Geometric Governor (adaptive threshold controller)
    governor: GeometricGovernor,
    
    /// Last "handled" state μ(t_last)
    last_state: SystemState<D>,
    
    /// Entropy pool (accumulated noise for RNG)
    entropy_pool: u64,
    
    /// Last measured deviation (for diagnostics)
    last_deviation: f64,
    
    /// Event counter (how many times we've woken)
    event_count: u64,
    
    /// Skip counter (how many times we stayed asleep)
    skip_count: u64,
}

impl<const D: usize> SparseScheduler<D> {
    /// Create a new scheduler with initial state
    pub fn new(initial_state: SystemState<D>) -> Self {
        Self {
            governor: GeometricGovernor::new(),
            last_state: initial_state,
            entropy_pool: 0,
            last_deviation: 0.0,
            event_count: 0,
            skip_count: 0,
        }
    }
    
    /// Create a scheduler with custom governor
    pub fn with_governor(initial_state: SystemState<D>, governor: GeometricGovernor) -> Self {
        Self {
            governor,
            last_state: initial_state,
            entropy_pool: 0,
            last_deviation: 0.0,
            event_count: 0,
            skip_count: 0,
        }
    }
    
    /// Get reference to the governor
    pub fn governor(&self) -> &GeometricGovernor {
        &self.governor
    }
    
    /// Get mutable reference to the governor
    pub fn governor_mut(&mut self) -> &mut GeometricGovernor {
        &mut self.governor
    }
    
    /// Get the last deviation value
    pub fn last_deviation(&self) -> f64 {
        self.last_deviation
    }
    
    /// Get event count
    pub fn event_count(&self) -> u64 {
        self.event_count
    }
    
    /// Get skip count
    pub fn skip_count(&self) -> u64 {
        self.skip_count
    }
    
    /// Get current entropy pool value
    pub fn entropy_pool(&self) -> u64 {
        self.entropy_pool
    }
    
    /// The Sparse Trigger: Check if we should wake
    /// 
    /// Returns true if: Δ(t) = ||μ(t) - μ(t_last)||₂ ≥ ε(t)
    pub fn should_wake(&mut self, current: &SystemState<D>) -> bool {
        let deviation = current.deviation(&self.last_state);
        self.last_deviation = deviation;
        
        self.governor.should_trigger(deviation)
    }
    
    /// Handle a significant event (state deviation exceeded threshold)
    /// 
    /// This:
    /// 1. Updates the governor (adapts ε)
    /// 2. Stores current state as new reference
    /// 3. Increments event counter
    pub fn handle_event(&mut self, current: SystemState<D>) {
        // Calculate time delta for PID controller
        let dt = if current.timestamp > self.last_state.timestamp {
            (current.timestamp - self.last_state.timestamp) as f64 / 1_000_000.0 // μs to seconds
        } else {
            DEFAULT_DT
        };
        
        // Adapt the threshold based on observed behavior
        self.governor.adapt(self.last_deviation, dt);
        
        // Update reference state
        self.last_state = current;
        
        // Increment event counter
        self.event_count += 1;
    }
    
    /// Accumulate entropy when we don't wake
    /// 
    /// The noise from interrupts that don't trigger events is useful
    /// for seeding the random number generator.
    pub fn accumulate_entropy(&mut self) {
        // Use LCG-style mixing for entropy accumulation
        self.entropy_pool = self.entropy_pool
            .wrapping_mul(ENTROPY_MULTIPLIER)
            .wrapping_add(1);
        
        self.skip_count += 1;
    }
    
    /// Get entropy bytes (consumes entropy)
    pub fn consume_entropy(&mut self, bytes: &mut [u8]) {
        for byte in bytes.iter_mut() {
            self.entropy_pool = self.entropy_pool
                .wrapping_mul(ENTROPY_MULTIPLIER)
                .wrapping_add(1);
            *byte = (self.entropy_pool >> 32) as u8;
        }
    }
    
    /// Get ratio of events to total interrupts
    pub fn event_ratio(&self) -> f64 {
        let total = self.event_count + self.skip_count;
        if total == 0 {
            0.0
        } else {
            self.event_count as f64 / total as f64
        }
    }
    
    /// Reset scheduler state (for testing)
    pub fn reset(&mut self) {
        self.governor.reset();
        self.last_state = SystemState::zero();
        self.entropy_pool = 0;
        self.last_deviation = 0.0;
        self.event_count = 0;
        self.skip_count = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_no_deviation_no_wake() {
        let state = SystemState::<4>::zero();
        let mut scheduler = SparseScheduler::new(state);
        
        // Same state should not trigger wake
        let same_state = SystemState::<4>::zero();
        assert!(!scheduler.should_wake(&same_state));
    }
    
    #[test]
    fn test_large_deviation_wakes() {
        let state = SystemState::<4>::zero();
        let mut scheduler = SparseScheduler::new(state);
        
        // Large deviation should trigger wake (default ε = 0.1)
        let different_state = SystemState::new([1.0, 1.0, 1.0, 1.0], 1000);
        assert!(scheduler.should_wake(&different_state));
    }
    
    #[test]
    fn test_entropy_accumulation() {
        let state = SystemState::<4>::zero();
        let mut scheduler = SparseScheduler::new(state);
        
        let initial = scheduler.entropy_pool();
        scheduler.accumulate_entropy();
        
        assert_ne!(scheduler.entropy_pool(), initial);
    }
    
    #[test]
    fn test_event_ratio() {
        let state = SystemState::<4>::zero();
        let mut scheduler = SparseScheduler::new(state);
        
        // Simulate some events and skips
        scheduler.event_count = 25;
        scheduler.skip_count = 75;
        
        let ratio = scheduler.event_ratio();
        assert!((ratio - 0.25).abs() < 1e-10);
    }
}
