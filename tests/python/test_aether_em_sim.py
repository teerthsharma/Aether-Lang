import math

from aether_em_sim import (
    FaradayTensor,
    ScenarioConfig,
    Vec3,
    simulate_scenario,
)


def test_python_faraday_tensor_is_antisymmetric_and_round_trips_fields():
    electric = Vec3(0.7, -0.1, 0.2)
    magnetic = Vec3(-0.4, 0.3, 0.9)
    tensor = FaradayTensor.from_fields(electric, magnetic)

    for mu in range(4):
        assert tensor.matrix[mu][mu] == 0.0
        for nu in range(4):
            assert math.isclose(
                tensor.matrix[mu][nu] + tensor.matrix[nu][mu],
                0.0,
                abs_tol=1.0e-12,
            )

    recovered_e, recovered_b = tensor.fields()
    assert recovered_e == electric
    assert recovered_b == magnetic


def test_python_scenario_exports_topological_summary():
    dataset = simulate_scenario(
        ScenarioConfig(
            kind="braided_pair",
            grid=18,
            steps=3,
            coupling=0.64,
            phase=0.2,
            separation=0.5,
        )
    )

    assert len(dataset["frames"]) == 3
    assert dataset["summary"]["topology"]["betti_0"] >= 1
    assert dataset["summary"]["topology"]["betti_1"] >= 1
    assert dataset["summary"]["energy_total"] > 0.0
    assert dataset["summary"]["maxwell_residual"]["divergence_e"] < 0.35
