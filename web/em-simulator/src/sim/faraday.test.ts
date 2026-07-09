import { describe, expect, it } from 'vitest';
import {
  faradayTensor,
  lorentzInvariants,
  simulateScenario,
  vector,
} from './faraday';

describe('Faraday tensor simulator', () => {
  it('builds an antisymmetric Faraday tensor from E and B fields', () => {
    const tensor = faradayTensor(vector(0.4, -0.2, 0.9), vector(-0.1, 0.8, 0.3));

    for (let mu = 0; mu < 4; mu += 1) {
      expect(tensor[mu][mu]).toBe(0);
      for (let nu = 0; nu < 4; nu += 1) {
        expect(tensor[mu][nu] + tensor[nu][mu]).toBeCloseTo(0, 12);
      }
    }
  });

  it('computes Lorentz invariants from electric and magnetic vectors', () => {
    const electric = vector(0.5, 0.25, -0.75);
    const magnetic = vector(0.2, -0.4, 0.6);
    const invariants = lorentzInvariants(electric, magnetic);

    expect(invariants.magneticMinusElectric).toBeCloseTo(0.56 - 0.875, 12);
    expect(invariants.pseudoscalar).toBeCloseTo(-0.45, 12);
  });

  it('generates a topological cycle for toroidal pulse scenarios', () => {
    const result = simulateScenario({
      kind: 'toroidal_pulse',
      grid: 20,
      steps: 4,
      coupling: 0.7,
      phase: 0.3,
      separation: 0.5,
    });

    expect(result.frames).toHaveLength(4);
    expect(result.summary.topology.betti0).toBeGreaterThanOrEqual(1);
    expect(result.summary.topology.betti1).toBeGreaterThanOrEqual(1);
    expect(result.summary.energyTotal).toBeGreaterThan(0);
  });
});
