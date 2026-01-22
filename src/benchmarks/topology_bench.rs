//! ═══════════════════════════════════════════════════════════════════════════════
//! Topology Benchmark: Shape Verification Accuracy
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Benchmarks for the Topological Gatekeeper:
//!   - NOP sled detection (uniform bytes)
//!   - ROP chain detection (high loop patterns)
//!   - Encrypted payload detection (high entropy)
//!   - False positive rate on legitimate code
//!   - Sliding window throughput
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::topology::{compute_betti_0, compute_betti_1, verify_shape, VerifyResult};

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for topology benchmarks
#[derive(Debug, Clone)]
pub struct TopologyBenchConfig {
    /// Number of test samples per category
    pub samples_per_category: usize,
    /// Window size for sliding window tests
    pub window_size: usize,
}

impl Default for TopologyBenchConfig {
    fn default() -> Self {
        Self {
            samples_per_category: 100,
            window_size: 64,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark Results
// ═══════════════════════════════════════════════════════════════════════════════

/// Results from topology benchmarks
#[derive(Debug, Clone)]
pub struct TopologyBenchResults {
    /// True positive rate for NOP sled detection
    pub nop_sled_tpr: f64,
    /// True positive rate for ROP chain detection  
    pub rop_chain_tpr: f64,
    /// True positive rate for encrypted payload detection
    pub encrypted_tpr: f64,
    /// False positive rate on legitimate code
    pub legitimate_fpr: f64,
    /// Mean Betti-0 for each category
    pub mean_betti0: CategoryStats,
    /// Mean Betti-1 for each category
    pub mean_betti1: CategoryStats,
    /// Total samples tested
    pub total_samples: usize,
}

/// Statistics per category
#[derive(Debug, Clone, Default)]
pub struct CategoryStats {
    pub nop_sled: f64,
    pub rop_chain: f64,
    pub encrypted: f64,
    pub legitimate: f64,
}

impl TopologyBenchResults {
    fn new() -> Self {
        Self {
            nop_sled_tpr: 0.0,
            rop_chain_tpr: 0.0,
            encrypted_tpr: 0.0,
            legitimate_fpr: 0.0,
            mean_betti0: CategoryStats::default(),
            mean_betti1: CategoryStats::default(),
            total_samples: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test Data Generators
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate NOP sled pattern (uniform 0x90 bytes)
fn generate_nop_sled(size: usize, seed: u32) -> [u8; 256] {
    let mut data = [0x90u8; 256];
    // Add slight variation based on seed
    for i in 0..size.min(256) {
        if (seed.wrapping_add(i as u32)) % 20 == 0 {
            data[i] = 0x91; // Occasional variant
        }
    }
    data
}

/// Generate ROP chain pattern (repeating address-like sequences)
fn generate_rop_chain(size: usize, seed: u32) -> [u8; 256] {
    let mut data = [0u8; 256];
    // ROP chains have repeating patterns with address-like structures
    for i in 0..size.min(256) {
        let offset = seed.wrapping_add(i as u32);
        data[i] = match i % 8 {
            0..=3 => ((0x40 + (offset % 0x40)) & 0xFF) as u8, // Address bytes
            4..=5 => 0x00, // NULL bytes
            _ => ((offset % 0x10) + 0x10) as u8, // Gadget offsets
        };
    }
    data
}

/// Generate encrypted/random payload (high entropy)
fn generate_encrypted(size: usize, seed: u32) -> [u8; 256] {
    let mut data = [0u8; 256];
    // Pseudo-random using simple LCG
    let mut state = seed;
    for i in 0..size.min(256) {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        data[i] = ((state >> 16) & 0xFF) as u8;
    }
    data
}

/// Generate legitimate code pattern (mixed instruction-like bytes)
fn generate_legitimate_code(size: usize, seed: u32) -> [u8; 256] {
    let mut data = [0u8; 256];
    // Realistic code has moderate variation but structured patterns
    let opcodes: [u8; 16] = [
        0x48, 0x89, 0x8B, 0x55, 0x41, 0x50, 0x51, 0x52,
        0xC3, 0xE8, 0xFF, 0x83, 0x74, 0x75, 0x0F, 0x31
    ];
    
    for i in 0..size.min(256) {
        let idx = (seed.wrapping_add(i as u32) as usize) % opcodes.len();
        data[i] = opcodes[idx];
        // Add operands occasionally
        if i > 0 && data[i - 1] == 0x89 {
            data[i] = 0xE5; // mov rbp, rsp pattern
        }
    }
    data
}

// ═══════════════════════════════════════════════════════════════════════════════
// Topology Benchmark Implementation
// ═══════════════════════════════════════════════════════════════════════════════

/// Run comprehensive topology benchmarks
pub struct TopologyBenchmark {
    config: TopologyBenchConfig,
}

impl TopologyBenchmark {
    pub fn new(config: TopologyBenchConfig) -> Self {
        Self { config }
    }

    /// Run all topology benchmarks
    pub fn run_all(&self) -> TopologyBenchResults {
        let mut results = TopologyBenchResults::new();
        
        let mut nop_detected = 0u32;
        let mut rop_detected = 0u32;
        let mut encrypted_detected = 0u32;
        let mut legitimate_rejected = 0u32;
        
        let mut nop_b0_sum = 0.0f64;
        let mut nop_b1_sum = 0.0f64;
        let mut rop_b0_sum = 0.0f64;
        let mut rop_b1_sum = 0.0f64;
        let mut enc_b0_sum = 0.0f64;
        let mut enc_b1_sum = 0.0f64;
        let mut leg_b0_sum = 0.0f64;
        let mut leg_b1_sum = 0.0f64;
        
        for i in 0..self.config.samples_per_category {
            let seed = i as u32;
            
            // Test NOP sled
            let nop_data = generate_nop_sled(128, seed);
            let nop_slice = &nop_data[..128];
            nop_b0_sum += compute_betti_0(nop_slice) as f64;
            nop_b1_sum += compute_betti_1(nop_slice) as f64;
            if !matches!(verify_shape(nop_slice), VerifyResult::Pass) {
                nop_detected += 1;
            }
            
            // Test ROP chain
            let rop_data = generate_rop_chain(128, seed);
            let rop_slice = &rop_data[..128];
            rop_b0_sum += compute_betti_0(rop_slice) as f64;
            rop_b1_sum += compute_betti_1(rop_slice) as f64;
            if !matches!(verify_shape(rop_slice), VerifyResult::Pass) {
                rop_detected += 1;
            }
            
            // Test encrypted payload
            let enc_data = generate_encrypted(128, seed);
            let enc_slice = &enc_data[..128];
            enc_b0_sum += compute_betti_0(enc_slice) as f64;
            enc_b1_sum += compute_betti_1(enc_slice) as f64;
            if !matches!(verify_shape(enc_slice), VerifyResult::Pass) {
                encrypted_detected += 1;
            }
            
            // Test legitimate code
            let leg_data = generate_legitimate_code(128, seed);
            let leg_slice = &leg_data[..128];
            leg_b0_sum += compute_betti_0(leg_slice) as f64;
            leg_b1_sum += compute_betti_1(leg_slice) as f64;
            if !matches!(verify_shape(leg_slice), VerifyResult::Pass) {
                legitimate_rejected += 1;
            }
        }
        
        let n = self.config.samples_per_category as f64;
        
        results.nop_sled_tpr = nop_detected as f64 / n;
        results.rop_chain_tpr = rop_detected as f64 / n;
        results.encrypted_tpr = encrypted_detected as f64 / n;
        results.legitimate_fpr = legitimate_rejected as f64 / n;
        
        results.mean_betti0.nop_sled = nop_b0_sum / n;
        results.mean_betti0.rop_chain = rop_b0_sum / n;
        results.mean_betti0.encrypted = enc_b0_sum / n;
        results.mean_betti0.legitimate = leg_b0_sum / n;
        
        results.mean_betti1.nop_sled = nop_b1_sum / n;
        results.mean_betti1.rop_chain = rop_b1_sum / n;
        results.mean_betti1.encrypted = enc_b1_sum / n;
        results.mean_betti1.legitimate = leg_b1_sum / n;
        
        results.total_samples = self.config.samples_per_category * 4;
        
        results
    }

    /// Run individual pattern test
    pub fn test_single_pattern(&self, pattern: PatternType) -> SinglePatternResult {
        let data = match pattern {
            PatternType::NopSled => generate_nop_sled(128, 0),
            PatternType::RopChain => generate_rop_chain(128, 0),
            PatternType::Encrypted => generate_encrypted(128, 0),
            PatternType::Legitimate => generate_legitimate_code(128, 0),
        };
        
        let slice = &data[..128];
        SinglePatternResult {
            betti_0: compute_betti_0(slice),
            betti_1: compute_betti_1(slice),
            verify_result: verify_shape(slice),
        }
    }
}

/// Pattern types for testing
#[derive(Debug, Clone, Copy)]
pub enum PatternType {
    NopSled,
    RopChain,
    Encrypted,
    Legitimate,
}

/// Result for single pattern test
#[derive(Debug)]
pub struct SinglePatternResult {
    pub betti_0: u32,
    pub betti_1: u32,
    pub verify_result: VerifyResult,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nop_sled_detection() {
        let bench = TopologyBenchmark::new(TopologyBenchConfig::default());
        let results = bench.run_all();
        
        assert!(results.nop_sled_tpr >= 0.90, 
            "NOP sled TPR should be >= 90%, got {:.1}%", results.nop_sled_tpr * 100.0);
        println!("NOP Sled: TPR={:.1}%, mean_β₀={:.2}, mean_β₁={:.2}",
            results.nop_sled_tpr * 100.0,
            results.mean_betti0.nop_sled,
            results.mean_betti1.nop_sled);
    }

    #[test]
    fn test_rop_chain_detection() {
        let bench = TopologyBenchmark::new(TopologyBenchConfig::default());
        let results = bench.run_all();
        
        assert!(results.rop_chain_tpr >= 0.80, 
            "ROP chain TPR should be >= 80%, got {:.1}%", results.rop_chain_tpr * 100.0);
        println!("ROP Chain: TPR={:.1}%, mean_β₀={:.2}, mean_β₁={:.2}",
            results.rop_chain_tpr * 100.0,
            results.mean_betti0.rop_chain,
            results.mean_betti1.rop_chain);
    }

    #[test]
    fn test_legitimate_code_acceptance() {
        let bench = TopologyBenchmark::new(TopologyBenchConfig::default());
        let results = bench.run_all();
        
        assert!(results.legitimate_fpr <= 0.20, 
            "Legitimate FPR should be <= 20%, got {:.1}%", results.legitimate_fpr * 100.0);
        println!("Legitimate: FPR={:.1}%, mean_β₀={:.2}, mean_β₁={:.2}",
            results.legitimate_fpr * 100.0,
            results.mean_betti0.legitimate,
            results.mean_betti1.legitimate);
    }

    #[test]
    fn test_full_benchmark_suite() {
        let bench = TopologyBenchmark::new(TopologyBenchConfig {
            samples_per_category: 50,
            window_size: 64,
        });
        let results = bench.run_all();
        
        println!("\n═══ Topology Benchmark Results ═══");
        println!("NOP Sled TPR:    {:.1}%", results.nop_sled_tpr * 100.0);
        println!("ROP Chain TPR:   {:.1}%", results.rop_chain_tpr * 100.0);
        println!("Encrypted TPR:   {:.1}%", results.encrypted_tpr * 100.0);
        println!("Legitimate FPR:  {:.1}%", results.legitimate_fpr * 100.0);
        println!("Total samples:   {}", results.total_samples);
    }
}
