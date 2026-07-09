//! Topological electromagnetic field data simulator.
//!
//! The simulator treats the Faraday tensor as the primitive object and derives
//! educational coupling datasets from sampled electric and magnetic fields.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

pub const LIGHT_SPEED_NORMALIZED: f64 = 1.0;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[must_use]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[must_use]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    #[must_use]
    pub fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    #[must_use]
    pub fn scale(self, factor: f64) -> Self {
        Self::new(self.x * factor, self.y * factor, self.z * factor)
    }

    #[must_use]
    pub fn add_vec(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldPair {
    pub electric: Vec3,
    pub magnetic: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaradayTensor {
    matrix: [[f64; 4]; 4],
}

impl FaradayTensor {
    #[must_use]
    pub fn from_fields(electric: Vec3, magnetic: Vec3) -> Self {
        let matrix = [
            [0.0, electric.x, electric.y, electric.z],
            [-electric.x, 0.0, -magnetic.z, magnetic.y],
            [-electric.y, magnetic.z, 0.0, -magnetic.x],
            [-electric.z, -magnetic.y, magnetic.x, 0.0],
        ];
        Self { matrix }
    }

    #[must_use]
    pub fn component(self, mu: usize, nu: usize) -> f64 {
        self.matrix[mu][nu]
    }

    #[must_use]
    pub fn fields(self) -> FieldPair {
        FieldPair {
            electric: Vec3::new(self.matrix[0][1], self.matrix[0][2], self.matrix[0][3]),
            magnetic: Vec3::new(
                self.matrix[2][3] * -1.0,
                self.matrix[1][3],
                self.matrix[2][1],
            ),
        }
    }

    #[must_use]
    pub fn invariants(self) -> LorentzInvariants {
        let fields = self.fields();
        LorentzInvariants {
            magnetic_minus_electric: fields.magnetic.norm_squared()
                - fields.electric.norm_squared()
                    / (LIGHT_SPEED_NORMALIZED * LIGHT_SPEED_NORMALIZED),
            pseudoscalar: fields.electric.dot(fields.magnetic),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LorentzInvariants {
    pub magnetic_minus_electric: f64,
    pub pseudoscalar: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CouplingKind {
    ToroidalPulse,
    BraidedPair,
    BoundarySheaf,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScenarioConfig {
    pub kind: CouplingKind,
    pub grid: usize,
    pub steps: usize,
    pub coupling: f64,
    pub phase: f64,
    pub separation: f64,
}

impl Default for ScenarioConfig {
    fn default() -> Self {
        Self {
            kind: CouplingKind::ToroidalPulse,
            grid: 24,
            steps: 8,
            coupling: 0.7,
            phase: 0.25,
            separation: 0.5,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FieldSample {
    pub x: f64,
    pub y: f64,
    pub electric: Vec3,
    pub magnetic: Vec3,
    pub energy_density: f64,
    pub invariant: LorentzInvariants,
    pub poynting: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FieldFrame {
    pub t: f64,
    pub samples: Vec<FieldSample>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MaxwellResidual {
    pub divergence_e: f64,
    pub divergence_b: f64,
    pub faraday_curl: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TopologySummary {
    pub active_vertices: usize,
    pub active_edges: usize,
    pub betti_0: usize,
    pub betti_1: usize,
    pub threshold: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SimulationSummary {
    pub energy_total: f64,
    pub coupling_index: f64,
    pub maxwell_residual: MaxwellResidual,
    pub topology: TopologySummary,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SimulationDataset {
    pub config: ScenarioConfig,
    pub frames: Vec<FieldFrame>,
    pub summary: SimulationSummary,
}

#[must_use]
pub fn simulate_scenario(mut config: ScenarioConfig) -> SimulationDataset {
    config.grid = config.grid.clamp(8, 96);
    config.steps = config.steps.clamp(1, 128);
    config.coupling = config.coupling.clamp(0.0, 1.0);
    config.separation = config.separation.clamp(0.05, 0.9);

    let mut frames = Vec::with_capacity(config.steps);
    let mut energy_total = 0.0;
    let mut coupling_total = 0.0;
    let denom = (config.steps.saturating_sub(1)).max(1) as f64;

    for step in 0..config.steps {
        let t = step as f64 / denom;
        let frame = sample_frame(config, t);
        for sample in &frame.samples {
            energy_total += sample.energy_density;
            coupling_total += sample.electric.dot(sample.magnetic).abs()
                + sample.poynting.norm_squared().sqrt() * 0.05;
        }
        frames.push(frame);
    }

    let topology = topology_summary(
        frames.last().map_or(&[][..], |frame| &frame.samples),
        config.grid,
    );
    let maxwell_residual = residual_summary(
        frames.last().map_or(&[][..], |frame| &frame.samples),
        config.grid,
    );
    let sample_count = (config.grid * config.grid * config.steps) as f64;

    SimulationDataset {
        config,
        frames,
        summary: SimulationSummary {
            energy_total: energy_total / sample_count,
            coupling_index: coupling_total / sample_count,
            maxwell_residual,
            topology,
        },
    }
}

fn sample_frame(config: ScenarioConfig, t: f64) -> FieldFrame {
    let mut samples = Vec::with_capacity(config.grid * config.grid);
    let n = config.grid as f64;
    for row in 0..config.grid {
        for col in 0..config.grid {
            let x = if config.grid == 1 {
                0.0
            } else {
                -1.0 + 2.0 * col as f64 / (n - 1.0)
            };
            let y = if config.grid == 1 {
                0.0
            } else {
                -1.0 + 2.0 * row as f64 / (n - 1.0)
            };
            let (electric, magnetic) = fields_for(config, x, y, t);
            let tensor = FaradayTensor::from_fields(electric, magnetic);
            let poynting = electric.cross(magnetic);
            let energy_density = 0.5 * (electric.norm_squared() + magnetic.norm_squared());
            samples.push(FieldSample {
                x,
                y,
                electric,
                magnetic,
                energy_density,
                invariant: tensor.invariants(),
                poynting,
            });
        }
    }
    FieldFrame { t, samples }
}

fn fields_for(config: ScenarioConfig, x: f64, y: f64, t: f64) -> (Vec3, Vec3) {
    match config.kind {
        CouplingKind::ToroidalPulse => toroidal_fields(config, x, y, t),
        CouplingKind::BraidedPair => braided_fields(config, x, y, t),
        CouplingKind::BoundarySheaf => boundary_sheaf_fields(config, x, y, t),
    }
}

fn toroidal_fields(config: ScenarioConfig, x: f64, y: f64, t: f64) -> (Vec3, Vec3) {
    let angle = core::f64::consts::TAU * (t + config.phase);
    let twist = sin(angle);
    let radius = sqrt(x * x + y * y).max(1.0e-6);
    let ring_radius = 0.34 + 0.28 * config.separation;
    let radial = radius - ring_radius;
    let envelope = exp(-(radial * radial) / 0.018) * exp(-0.28 * (x * x + y * y));
    let tangent = Vec3::new(-y / radius, x / radius, 0.0);
    let normal = Vec3::new(x / radius, y / radius, 0.0);
    let electric = tangent
        .scale(envelope * (0.72 + 0.28 * twist))
        .add_vec(Vec3::new(
            0.0,
            0.0,
            envelope * config.coupling * cos(angle + radius),
        ));
    let magnetic = normal.scale(envelope * config.coupling).add_vec(Vec3::new(
        0.0,
        0.0,
        envelope * 0.25 * sin(angle - radius),
    ));
    (electric, magnetic)
}

fn braided_fields(config: ScenarioConfig, x: f64, y: f64, t: f64) -> (Vec3, Vec3) {
    let angle = core::f64::consts::TAU * (t + config.phase);
    let offset = config.separation;
    let c1 = (offset * cos(angle), offset * sin(angle));
    let c2 = (-offset * cos(angle), -offset * sin(angle));
    let g1 = gaussian(x - c1.0, y - c1.1, 0.08);
    let g2 = gaussian(x - c2.0, y - c2.1, 0.08);
    let electric = Vec3::new(
        (x - c1.0) * g1 - (x - c2.0) * g2,
        (y - c1.1) * g1 - (y - c2.1) * g2,
        config.coupling * (g1 + g2) * sin(angle),
    );
    let magnetic = Vec3::new(
        -(y - c1.1) * g1 - (y - c2.1) * g2,
        (x - c1.0) * g1 + (x - c2.0) * g2,
        config.coupling * (g1 - g2) * cos(angle),
    );
    (electric, magnetic)
}

fn boundary_sheaf_fields(config: ScenarioConfig, x: f64, y: f64, t: f64) -> (Vec3, Vec3) {
    let phase = core::f64::consts::TAU * (t + config.phase);
    let left = gaussian(x + config.separation, y, 0.12);
    let right = gaussian(x - config.separation, y, 0.12);
    let boundary = exp(-y * y / 0.025);
    let electric = Vec3::new(
        left - right,
        0.25 * boundary * sin(phase),
        config.coupling * boundary * cos(phase + x),
    );
    let magnetic = Vec3::new(
        -0.2 * boundary * cos(phase),
        config.coupling * (left + right),
        boundary * sin(phase + y),
    );
    (electric, magnetic)
}

fn topology_summary(samples: &[FieldSample], grid: usize) -> TopologySummary {
    if samples.is_empty() {
        return TopologySummary::default();
    }

    let max_energy = samples.iter().fold(0.0_f64, |current, sample| {
        current.max(sample.energy_density)
    });
    let threshold = max_energy * 0.38;
    let mut active = Vec::with_capacity(samples.len());
    for sample in samples {
        active.push(sample.energy_density >= threshold);
    }

    let mut edges = 0usize;
    for row in 0..grid {
        for col in 0..grid {
            let idx = row * grid + col;
            if !active[idx] {
                continue;
            }
            if col + 1 < grid && active[idx + 1] {
                edges += 1;
            }
            if row + 1 < grid && active[idx + grid] {
                edges += 1;
            }
        }
    }

    let vertices = active.iter().filter(|&&is_active| is_active).count();
    let components = count_components(&active, grid);
    let betti_1 = edges.saturating_add(components).saturating_sub(vertices);

    TopologySummary {
        active_vertices: vertices,
        active_edges: edges,
        betti_0: components,
        betti_1,
        threshold,
    }
}

fn count_components(active: &[bool], grid: usize) -> usize {
    let mut seen = Vec::new();
    seen.resize(active.len(), false);
    let mut components = 0usize;
    let mut stack = Vec::new();

    for idx in 0..active.len() {
        if !active[idx] || seen[idx] {
            continue;
        }
        components += 1;
        stack.push(idx);
        seen[idx] = true;
        while let Some(current) = stack.pop() {
            let row = current / grid;
            let col = current % grid;
            let neighbors = [
                (row > 0).then_some(current - grid),
                (row + 1 < grid).then_some(current + grid),
                (col > 0).then_some(current - 1),
                (col + 1 < grid).then_some(current + 1),
            ];
            for neighbor in neighbors.into_iter().flatten() {
                if active[neighbor] && !seen[neighbor] {
                    seen[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
    }

    components
}

fn residual_summary(samples: &[FieldSample], grid: usize) -> MaxwellResidual {
    if samples.len() < grid * grid || grid < 3 {
        return MaxwellResidual::default();
    }
    let dx = 2.0 / (grid as f64 - 1.0);
    let mut div_e = 0.0;
    let mut div_b = 0.0;
    let mut curl_proxy = 0.0;
    let mut count = 0.0;

    for row in 1..(grid - 1) {
        for col in 1..(grid - 1) {
            let idx = row * grid + col;
            let left = &samples[idx - 1];
            let right = &samples[idx + 1];
            let down = &samples[idx - grid];
            let up = &samples[idx + grid];
            let d_ex_dx = (right.electric.x - left.electric.x) / (2.0 * dx);
            let d_ey_dy = (up.electric.y - down.electric.y) / (2.0 * dx);
            let d_bx_dx = (right.magnetic.x - left.magnetic.x) / (2.0 * dx);
            let d_by_dy = (up.magnetic.y - down.magnetic.y) / (2.0 * dx);
            let curl_z =
                (right.electric.y - left.electric.y - up.electric.x + down.electric.x) / (2.0 * dx);
            let scale = 1.0 + samples[idx].energy_density;
            div_e += ((d_ex_dx + d_ey_dy) / scale).abs();
            div_b += ((d_bx_dx + d_by_dy) / scale).abs();
            curl_proxy += (curl_z / scale).abs();
            count += 1.0;
        }
    }

    MaxwellResidual {
        divergence_e: div_e / count,
        divergence_b: div_b / count,
        faraday_curl: curl_proxy / count,
    }
}

fn gaussian(x: f64, y: f64, sigma: f64) -> f64 {
    exp(-(x * x + y * y) / sigma)
}

fn sin(value: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        value.sin()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sin(value)
    }
}

fn cos(value: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        value.cos()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::cos(value)
    }
}

fn exp(value: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        value.exp()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::exp(value)
    }
}

fn sqrt(value: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        value.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrt(value)
    }
}
