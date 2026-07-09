export type CouplingKind = 'toroidal_pulse' | 'braided_pair' | 'boundary_sheaf';

export type Vector3 = {
  x: number;
  y: number;
  z: number;
};

export type ScenarioConfig = {
  kind: CouplingKind;
  grid: number;
  steps: number;
  coupling: number;
  phase: number;
  separation: number;
};

export type FieldSample = {
  x: number;
  y: number;
  electric: Vector3;
  magnetic: Vector3;
  energyDensity: number;
  invariant: {
    magneticMinusElectric: number;
    pseudoscalar: number;
  };
  poynting: Vector3;
};

export type SimulationDataset = {
  frames: Array<{ t: number; samples: FieldSample[] }>;
  summary: {
    energyTotal: number;
    couplingIndex: number;
    maxwellResidual: {
      divergenceE: number;
      divergenceB: number;
      faradayCurl: number;
    };
    topology: {
      activeVertices: number;
      activeEdges: number;
      betti0: number;
      betti1: number;
      threshold: number;
    };
  };
};

export const vector = (x: number, y: number, z: number): Vector3 => ({ x, y, z });

export const dot = (a: Vector3, b: Vector3) => a.x * b.x + a.y * b.y + a.z * b.z;

export const cross = (a: Vector3, b: Vector3): Vector3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});

export const normSquared = (a: Vector3) => dot(a, a);

export const scale = (a: Vector3, factor: number): Vector3 => ({
  x: a.x * factor,
  y: a.y * factor,
  z: a.z * factor,
});

export const add = (a: Vector3, b: Vector3): Vector3 => ({
  x: a.x + b.x,
  y: a.y + b.y,
  z: a.z + b.z,
});

export const faradayTensor = (electric: Vector3, magnetic: Vector3): number[][] => [
  [0, electric.x, electric.y, electric.z],
  [-electric.x, 0, -magnetic.z, magnetic.y],
  [-electric.y, magnetic.z, 0, -magnetic.x],
  [-electric.z, -magnetic.y, magnetic.x, 0],
];

export const lorentzInvariants = (electric: Vector3, magnetic: Vector3) => ({
  magneticMinusElectric: normSquared(magnetic) - normSquared(electric),
  pseudoscalar: dot(electric, magnetic),
});

export function simulateScenario(input: ScenarioConfig): SimulationDataset {
  const config = normalizeConfig(input);
  const frames = Array.from({ length: config.steps }, (_, step) =>
    sampleFrame(config, step / Math.max(1, config.steps - 1)),
  );
  const flat = frames.flatMap((frame) => frame.samples);
  const energyTotal = flat.reduce((sum, sample) => sum + sample.energyDensity, 0) / flat.length;
  const couplingIndex =
    flat.reduce(
      (sum, sample) =>
        sum + Math.abs(sample.invariant.pseudoscalar) + Math.sqrt(normSquared(sample.poynting)) * 0.05,
      0,
    ) / flat.length;
  const lastSamples = frames.length > 0 ? frames[frames.length - 1].samples : [];
  return {
    frames,
    summary: {
      energyTotal,
      couplingIndex,
      maxwellResidual: residualSummary(lastSamples, config.grid),
      topology: topologySummary(lastSamples, config.grid),
    },
  };
}

function normalizeConfig(config: ScenarioConfig): ScenarioConfig {
  return {
    ...config,
    grid: clamp(Math.round(config.grid), 8, 96),
    steps: clamp(Math.round(config.steps), 1, 128),
    coupling: clamp(config.coupling, 0, 1),
    separation: clamp(config.separation, 0.05, 0.9),
  };
}

function sampleFrame(config: ScenarioConfig, t: number) {
  const samples: FieldSample[] = [];
  for (let row = 0; row < config.grid; row += 1) {
    for (let col = 0; col < config.grid; col += 1) {
      const x = -1 + (2 * col) / (config.grid - 1);
      const y = -1 + (2 * row) / (config.grid - 1);
      const { electric, magnetic } = fieldsFor(config, x, y, t);
      const poynting = cross(electric, magnetic);
      samples.push({
        x,
        y,
        electric,
        magnetic,
        energyDensity: 0.5 * (normSquared(electric) + normSquared(magnetic)),
        invariant: lorentzInvariants(electric, magnetic),
        poynting,
      });
    }
  }
  return { t, samples };
}

function fieldsFor(config: ScenarioConfig, x: number, y: number, t: number) {
  if (config.kind === 'braided_pair') {
    return braidedFields(config, x, y, t);
  }
  if (config.kind === 'boundary_sheaf') {
    return boundarySheafFields(config, x, y, t);
  }
  return toroidalFields(config, x, y, t);
}

function toroidalFields(config: ScenarioConfig, x: number, y: number, t: number) {
  const angle = Math.PI * 2 * (t + config.phase);
  const radius = Math.max(1e-6, Math.hypot(x, y));
  const ringRadius = 0.34 + 0.28 * config.separation;
  const radial = radius - ringRadius;
  const envelope = Math.exp(-(radial * radial) / 0.018) * Math.exp(-0.28 * (x * x + y * y));
  const tangent = vector(-y / radius, x / radius, 0);
  const normal = vector(x / radius, y / radius, 0);
  return {
    electric: add(
      scale(tangent, envelope * (0.72 + 0.28 * Math.sin(angle))),
      vector(0, 0, envelope * config.coupling * Math.cos(angle + radius)),
    ),
    magnetic: add(
      scale(normal, envelope * config.coupling),
      vector(0, 0, envelope * 0.25 * Math.sin(angle - radius)),
    ),
  };
}

function braidedFields(config: ScenarioConfig, x: number, y: number, t: number) {
  const angle = Math.PI * 2 * (t + config.phase);
  const c1 = [config.separation * Math.cos(angle), config.separation * Math.sin(angle)] as const;
  const c2 = [-c1[0], -c1[1]] as const;
  const g1 = gaussian(x - c1[0], y - c1[1], 0.08);
  const g2 = gaussian(x - c2[0], y - c2[1], 0.08);
  return {
    electric: vector(
      (x - c1[0]) * g1 - (x - c2[0]) * g2,
      (y - c1[1]) * g1 - (y - c2[1]) * g2,
      config.coupling * (g1 + g2) * Math.sin(angle),
    ),
    magnetic: vector(
      -(y - c1[1]) * g1 - (y - c2[1]) * g2,
      (x - c1[0]) * g1 + (x - c2[0]) * g2,
      config.coupling * (g1 - g2) * Math.cos(angle),
    ),
  };
}

function boundarySheafFields(config: ScenarioConfig, x: number, y: number, t: number) {
  const phase = Math.PI * 2 * (t + config.phase);
  const left = gaussian(x + config.separation, y, 0.12);
  const right = gaussian(x - config.separation, y, 0.12);
  const boundary = Math.exp(-(y * y) / 0.025);
  return {
    electric: vector(left - right, 0.25 * boundary * Math.sin(phase), config.coupling * boundary * Math.cos(phase + x)),
    magnetic: vector(-0.2 * boundary * Math.cos(phase), config.coupling * (left + right), boundary * Math.sin(phase + y)),
  };
}

function topologySummary(samples: FieldSample[], grid: number) {
  const maxEnergy = Math.max(...samples.map((sample) => sample.energyDensity), 0);
  const threshold = maxEnergy * 0.38;
  const active = samples.map((sample) => sample.energyDensity >= threshold);
  let activeEdges = 0;
  for (let row = 0; row < grid; row += 1) {
    for (let col = 0; col < grid; col += 1) {
      const idx = row * grid + col;
      if (!active[idx]) continue;
      if (col + 1 < grid && active[idx + 1]) activeEdges += 1;
      if (row + 1 < grid && active[idx + grid]) activeEdges += 1;
    }
  }
  const activeVertices = active.filter(Boolean).length;
  const betti0 = countComponents(active, grid);
  return {
    activeVertices,
    activeEdges,
    betti0,
    betti1: Math.max(0, activeEdges + betti0 - activeVertices),
    threshold,
  };
}

function countComponents(active: boolean[], grid: number) {
  const seen = new Array<boolean>(active.length).fill(false);
  let components = 0;
  for (let idx = 0; idx < active.length; idx += 1) {
    if (!active[idx] || seen[idx]) continue;
    components += 1;
    const stack = [idx];
    seen[idx] = true;
    while (stack.length > 0) {
      const current = stack.pop() as number;
      const row = Math.floor(current / grid);
      const col = current % grid;
      const neighbors = [
        row > 0 ? current - grid : -1,
        row + 1 < grid ? current + grid : -1,
        col > 0 ? current - 1 : -1,
        col + 1 < grid ? current + 1 : -1,
      ];
      for (const neighbor of neighbors) {
        if (neighbor >= 0 && active[neighbor] && !seen[neighbor]) {
          seen[neighbor] = true;
          stack.push(neighbor);
        }
      }
    }
  }
  return components;
}

function residualSummary(samples: FieldSample[], grid: number) {
  if (samples.length < grid * grid || grid < 3) {
    return { divergenceE: 0, divergenceB: 0, faradayCurl: 0 };
  }
  const dx = 2 / (grid - 1);
  let divE = 0;
  let divB = 0;
  let curl = 0;
  let count = 0;
  for (let row = 1; row < grid - 1; row += 1) {
    for (let col = 1; col < grid - 1; col += 1) {
      const idx = row * grid + col;
      const left = samples[idx - 1];
      const right = samples[idx + 1];
      const down = samples[idx - grid];
      const up = samples[idx + grid];
      const scaleFactor = 1 + samples[idx].energyDensity;
      divE += Math.abs(((right.electric.x - left.electric.x + up.electric.y - down.electric.y) / (2 * dx)) / scaleFactor);
      divB += Math.abs(((right.magnetic.x - left.magnetic.x + up.magnetic.y - down.magnetic.y) / (2 * dx)) / scaleFactor);
      curl += Math.abs(((right.electric.y - left.electric.y - up.electric.x + down.electric.x) / (2 * dx)) / scaleFactor);
      count += 1;
    }
  }
  return { divergenceE: divE / count, divergenceB: divB / count, faradayCurl: curl / count };
}

function gaussian(x: number, y: number, sigma: number) {
  return Math.exp(-(x * x + y * y) / sigma);
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
