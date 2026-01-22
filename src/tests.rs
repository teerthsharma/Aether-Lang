//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Test Harness
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Unit tests for all mathematical components.
//! Run with: cargo test --lib --target x86_64-pc-windows-msvc
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod integration_tests {
    use crate::state::SystemState;
    use crate::governor::GeometricGovernor;
    use crate::scheduler::SparseScheduler;
    use crate::topology::{compute_betti_0, compute_betti_1, verify_shape, VerifyResult};
    use crate::manifold::{TimeDelayEmbedder, SparseAttentionGraph, ManifoldPoint};
    use crate::aether::{BlockMetadata, HierarchicalBlockTree, DriftDetector};

    // ═══════════════════════════════════════════════════════════════════════════
    // STATE VECTOR TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_deviation_euclidean_4d() {
        let s1 = SystemState::new([3.0, 0.0, 0.0, 4.0], 0);
        let s2 = SystemState::new([0.0, 0.0, 0.0, 0.0], 0);
        
        // √(3² + 4²) = 5
        let dev = s1.deviation(&s2);
        assert!((dev - 5.0).abs() < 1e-10, "Expected 5.0, got {}", dev);
    }

    #[test]
    fn test_deviation_symmetry() {
        let s1 = SystemState::new([1.0, 2.0, 3.0, 4.0], 0);
        let s2 = SystemState::new([5.0, 6.0, 7.0, 8.0], 0);
        
        assert!((s1.deviation(&s2) - s2.deviation(&s1)).abs() < 1e-10);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GOVERNOR PID TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_governor_stability_bounds() {
        let mut gov = GeometricGovernor::new();
        
        // Stress test: many iterations shouldn't cause overflow
        for _ in 0..10000 {
            gov.adapt(100.0, 0.001);
        }
        
        assert!(gov.epsilon() >= 0.001);
        assert!(gov.epsilon() <= 10.0);
    }

    #[test]
    fn test_governor_adapts_to_load() {
        let mut gov = GeometricGovernor::with_epsilon(0.5);
        
        // High load (high deviation) should push epsilon UP
        for _ in 0..100 {
            gov.adapt(5000.0, 0.001);
        }
        
        assert!(gov.epsilon() > 0.5, "Epsilon should increase under load");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SPARSE SCHEDULER TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_scheduler_sparse_triggering() {
        let initial = SystemState::<4>::zero();
        let mut sched = SparseScheduler::new(initial);
        
        // Small deviation - should NOT wake
        let small_change = SystemState::new([0.01, 0.0, 0.0, 0.0], 100);
        assert!(!sched.should_wake(&small_change));
        
        // Large deviation - SHOULD wake
        let big_change = SystemState::new([1.0, 1.0, 1.0, 1.0], 200);
        assert!(sched.should_wake(&big_change));
    }

    #[test]
    fn test_scheduler_entropy_growth() {
        let initial = SystemState::<4>::zero();
        let mut sched = SparseScheduler::new(initial);
        
        let e0 = sched.entropy_pool();
        sched.accumulate_entropy();
        sched.accumulate_entropy();
        
        assert_ne!(sched.entropy_pool(), e0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TOPOLOGICAL GATEKEEPER TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_betti_0_uniform_data() {
        // All zeros = uniform = 0 gaps
        let data = [0u8; 64];
        assert_eq!(compute_betti_0(&data), 0);
    }

    #[test]
    fn test_betti_0_alternating() {
        // Alternating 0 and 255 = many gaps
        let mut data = [0u8; 64];
        for i in 0..64 {
            data[i] = if i % 2 == 0 { 0 } else { 255 };
        }
        
        let b0 = compute_betti_0(&data);
        assert!(b0 > 10, "Alternating pattern should have many components");
    }

    #[test]
    fn test_topology_rejects_uniform() {
        // NOP sled (uniform bytes) should be suspicious
        let nop_sled = [0x90u8; 100];
        
        match verify_shape(&nop_sled) {
            VerifyResult::Pass => panic!("NOP sled should be rejected"),
            VerifyResult::InvalidDensity { .. } => {}, // Expected
            _ => {},
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MANIFOLD & EMBEDDING TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_time_delay_embedding() {
        let mut emb = TimeDelayEmbedder::<3>::new(1);
        
        // Feed 1, 2, 3, 4, 5
        for v in 1..=5 {
            emb.push(v as f64);
        }
        
        let point = emb.embed().unwrap();
        // τ=1, D=3: should be [5, 4, 3]
        assert!((point.coords[0] - 5.0).abs() < 1e-10);
        assert!((point.coords[1] - 4.0).abs() < 1e-10);
        assert!((point.coords[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_attention_locality() {
        let mut graph = SparseAttentionGraph::<3>::new(0.5);
        
        // Points within epsilon are neighbors
        graph.add_point(ManifoldPoint::new([0.0, 0.0, 0.0]));
        graph.add_point(ManifoldPoint::new([0.3, 0.0, 0.0])); // distance 0.3 < 0.5
        
        assert!(graph.are_neighbors(0, 1));
    }

    #[test]
    fn test_sparse_attention_not_neighbors() {
        let mut graph = SparseAttentionGraph::<3>::new(0.5);
        
        graph.add_point(ManifoldPoint::new([0.0, 0.0, 0.0]));
        graph.add_point(ManifoldPoint::new([1.0, 0.0, 0.0])); // distance 1.0 > 0.5
        
        assert!(!graph.are_neighbors(0, 1));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // AETHER HIERARCHICAL TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_block_metadata_creation() {
        let points: [[f64; 3]; 3] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ];
        
        let meta = BlockMetadata::from_points(&points);
        
        assert_eq!(meta.count, 3);
        // Centroid should be (0.5, 0.166..., 0)
        assert!((meta.centroid[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hierarchical_tree_building() {
        let blocks: [BlockMetadata<3>; 4] = [
            BlockMetadata::from_points(&[[0.0, 0.0, 0.0]]),
            BlockMetadata::from_points(&[[1.0, 0.0, 0.0]]),
            BlockMetadata::from_points(&[[2.0, 0.0, 0.0]]),
            BlockMetadata::from_points(&[[3.0, 0.0, 0.0]]),
        ];
        
        let mut tree = HierarchicalBlockTree::<3>::new();
        tree.build_from_blocks(&blocks);
        
        // Level 0 should have 4 blocks
        // Level 1 should have 1 block (aggregating all 4)
    }

    #[test]
    fn test_drift_detection() {
        let mut detector = DriftDetector::<3>::new();
        
        // Linear trajectory
        detector.update(&[0.0, 0.0, 0.0]);
        detector.update(&[1.0, 0.0, 0.0]);
        let drift = detector.update(&[2.0, 0.0, 0.0]);
        
        // No drift expected for linear motion
        assert!(drift < 0.1, "Linear trajectory should not drift");
        
        // Now sudden change
        let big_drift = detector.update(&[10.0, 5.0, 0.0]);
        assert!(big_drift > 1.0, "Sudden change should cause drift");
    }
}
