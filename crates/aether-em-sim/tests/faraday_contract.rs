use aether_em_sim::{
    simulate_scenario, CouplingKind, FaradayTensor, ScenarioConfig, Vec3, LIGHT_SPEED_NORMALIZED,
};

#[test]
fn faraday_tensor_is_antisymmetric_and_recovers_fields() {
    let electric = Vec3::new(0.8, -0.2, 0.4);
    let magnetic = Vec3::new(-0.3, 0.7, 0.1);
    let tensor = FaradayTensor::from_fields(electric, magnetic);

    for mu in 0..4 {
        assert_eq!(tensor.component(mu, mu), 0.0);
        for nu in 0..4 {
            let antisymmetric_sum = tensor.component(mu, nu) + tensor.component(nu, mu);
            assert!(antisymmetric_sum.abs() < 1.0e-12);
        }
    }

    let recovered = tensor.fields();
    assert!((recovered.electric.x - electric.x).abs() < 1.0e-12);
    assert!((recovered.electric.y - electric.y).abs() < 1.0e-12);
    assert!((recovered.electric.z - electric.z).abs() < 1.0e-12);
    assert!((recovered.magnetic.x - magnetic.x).abs() < 1.0e-12);
    assert!((recovered.magnetic.y - magnetic.y).abs() < 1.0e-12);
    assert!((recovered.magnetic.z - magnetic.z).abs() < 1.0e-12);
}

#[test]
fn lorentz_invariants_match_field_definition() {
    let electric = Vec3::new(0.5, 0.25, -0.75);
    let magnetic = Vec3::new(0.2, -0.4, 0.6);
    let tensor = FaradayTensor::from_fields(electric, magnetic);
    let invariants = tensor.invariants();

    let expected_i1 = magnetic.norm_squared()
        - electric.norm_squared() / (LIGHT_SPEED_NORMALIZED * LIGHT_SPEED_NORMALIZED);
    let expected_i2 = electric.dot(magnetic);

    assert!((invariants.magnetic_minus_electric - expected_i1).abs() < 1.0e-12);
    assert!((invariants.pseudoscalar - expected_i2).abs() < 1.0e-12);
}

#[test]
fn toroidal_coupling_exposes_cycle_topology_and_small_divergence_residual() {
    let config = ScenarioConfig {
        kind: CouplingKind::ToroidalPulse,
        grid: 24,
        steps: 4,
        coupling: 0.72,
        phase: 0.35,
        separation: 0.55,
    };
    let data = simulate_scenario(config);

    assert_eq!(data.frames.len(), 4);
    assert!(data.summary.maxwell_residual.divergence_e.abs() < 0.25);
    assert!(data.summary.topology.betti_0 >= 1);
    assert!(data.summary.topology.betti_1 >= 1);
    assert!(data.summary.energy_total > 0.0);
    assert!(data.summary.coupling_index > 0.0);
}
