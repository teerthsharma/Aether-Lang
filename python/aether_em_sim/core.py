from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LIGHT_SPEED_NORMALIZED = 1.0


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm_squared(self) -> float:
        return self.dot(self)

    def scale(self, factor: float) -> "Vec3":
        return Vec3(self.x * factor, self.y * factor, self.z * factor)

    def add(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass(frozen=True)
class FaradayTensor:
    matrix: tuple[tuple[float, float, float, float], ...]

    @classmethod
    def from_fields(cls, electric: Vec3, magnetic: Vec3) -> "FaradayTensor":
        return cls(
            (
                (0.0, electric.x, electric.y, electric.z),
                (-electric.x, 0.0, -magnetic.z, magnetic.y),
                (-electric.y, magnetic.z, 0.0, -magnetic.x),
                (-electric.z, -magnetic.y, magnetic.x, 0.0),
            )
        )

    def fields(self) -> tuple[Vec3, Vec3]:
        electric = Vec3(self.matrix[0][1], self.matrix[0][2], self.matrix[0][3])
        magnetic = Vec3(-self.matrix[2][3], self.matrix[1][3], self.matrix[2][1])
        return electric, magnetic

    def invariants(self) -> dict[str, float]:
        electric, magnetic = self.fields()
        return {
            "magnetic_minus_electric": magnetic.norm_squared()
            - electric.norm_squared() / (LIGHT_SPEED_NORMALIZED * LIGHT_SPEED_NORMALIZED),
            "pseudoscalar": electric.dot(magnetic),
        }


@dataclass(frozen=True)
class ScenarioConfig:
    kind: str = "toroidal_pulse"
    grid: int = 24
    steps: int = 8
    coupling: float = 0.7
    phase: float = 0.25
    separation: float = 0.5

    def normalized(self) -> "ScenarioConfig":
        return ScenarioConfig(
            kind=self.kind,
            grid=min(96, max(8, int(self.grid))),
            steps=min(128, max(1, int(self.steps))),
            coupling=min(1.0, max(0.0, float(self.coupling))),
            phase=float(self.phase),
            separation=min(0.9, max(0.05, float(self.separation))),
        )


def simulate_scenario(config: ScenarioConfig | None = None) -> dict:
    cfg = (config or ScenarioConfig()).normalized()
    frames = [_sample_frame(cfg, step / max(1, cfg.steps - 1)) for step in range(cfg.steps)]
    flat = [sample for frame in frames for sample in frame["samples"]]
    energy_total = sum(sample["energy_density"] for sample in flat) / len(flat)
    coupling_index = (
        sum(abs(sample["invariant"]["pseudoscalar"]) + 0.05 * _norm(sample["poynting"]) for sample in flat)
        / len(flat)
    )
    last_samples = frames[-1]["samples"] if frames else []
    return {
        "name": "Electromagnetic Field Data Simulator",
        "config": cfg.__dict__,
        "frames": frames,
        "summary": {
            "energy_total": energy_total,
            "coupling_index": coupling_index,
            "maxwell_residual": _residual_summary(last_samples, cfg.grid),
            "topology": _topology_summary(last_samples, cfg.grid),
        },
    }


def _sample_frame(config: ScenarioConfig, t: float) -> dict:
    samples = []
    n = float(config.grid)
    for row in range(config.grid):
        for col in range(config.grid):
            x = -1.0 + 2.0 * col / (n - 1.0)
            y = -1.0 + 2.0 * row / (n - 1.0)
            electric, magnetic = _fields_for(config, x, y, t)
            tensor = FaradayTensor.from_fields(electric, magnetic)
            poynting = electric.cross(magnetic)
            samples.append(
                {
                    "x": x,
                    "y": y,
                    "electric": electric.as_dict(),
                    "magnetic": magnetic.as_dict(),
                    "energy_density": 0.5 * (electric.norm_squared() + magnetic.norm_squared()),
                    "invariant": tensor.invariants(),
                    "poynting": poynting.as_dict(),
                }
            )
    return {"t": t, "samples": samples}


def _fields_for(config: ScenarioConfig, x: float, y: float, t: float) -> tuple[Vec3, Vec3]:
    if config.kind == "braided_pair":
        return _braided_fields(config, x, y, t)
    if config.kind == "boundary_sheaf":
        return _boundary_sheaf_fields(config, x, y, t)
    return _toroidal_fields(config, x, y, t)


def _toroidal_fields(config: ScenarioConfig, x: float, y: float, t: float) -> tuple[Vec3, Vec3]:
    angle = math.tau * (t + config.phase)
    radius = max(1.0e-6, math.sqrt(x * x + y * y))
    ring_radius = 0.34 + 0.28 * config.separation
    radial = radius - ring_radius
    envelope = math.exp(-(radial * radial) / 0.018) * math.exp(-0.28 * (x * x + y * y))
    tangent = Vec3(-y / radius, x / radius, 0.0)
    normal = Vec3(x / radius, y / radius, 0.0)
    electric = tangent.scale(envelope * (0.72 + 0.28 * math.sin(angle))).add(
        Vec3(0.0, 0.0, envelope * config.coupling * math.cos(angle + radius))
    )
    magnetic = normal.scale(envelope * config.coupling).add(
        Vec3(0.0, 0.0, envelope * 0.25 * math.sin(angle - radius))
    )
    return electric, magnetic


def _braided_fields(config: ScenarioConfig, x: float, y: float, t: float) -> tuple[Vec3, Vec3]:
    angle = math.tau * (t + config.phase)
    c1 = (config.separation * math.cos(angle), config.separation * math.sin(angle))
    c2 = (-c1[0], -c1[1])
    g1 = _gaussian(x - c1[0], y - c1[1], 0.08)
    g2 = _gaussian(x - c2[0], y - c2[1], 0.08)
    electric = Vec3(
        (x - c1[0]) * g1 - (x - c2[0]) * g2,
        (y - c1[1]) * g1 - (y - c2[1]) * g2,
        config.coupling * (g1 + g2) * math.sin(angle),
    )
    magnetic = Vec3(
        -(y - c1[1]) * g1 - (y - c2[1]) * g2,
        (x - c1[0]) * g1 + (x - c2[0]) * g2,
        config.coupling * (g1 - g2) * math.cos(angle),
    )
    return electric, magnetic


def _boundary_sheaf_fields(config: ScenarioConfig, x: float, y: float, t: float) -> tuple[Vec3, Vec3]:
    phase = math.tau * (t + config.phase)
    left = _gaussian(x + config.separation, y, 0.12)
    right = _gaussian(x - config.separation, y, 0.12)
    boundary = math.exp(-y * y / 0.025)
    electric = Vec3(left - right, 0.25 * boundary * math.sin(phase), config.coupling * boundary * math.cos(phase + x))
    magnetic = Vec3(-0.2 * boundary * math.cos(phase), config.coupling * (left + right), boundary * math.sin(phase + y))
    return electric, magnetic


def _topology_summary(samples: list[dict], grid: int) -> dict:
    if not samples:
        return {"active_vertices": 0, "active_edges": 0, "betti_0": 0, "betti_1": 0, "threshold": 0.0}
    max_energy = max(sample["energy_density"] for sample in samples)
    threshold = max_energy * 0.38
    active = [sample["energy_density"] >= threshold for sample in samples]
    edges = 0
    for row in range(grid):
        for col in range(grid):
            idx = row * grid + col
            if not active[idx]:
                continue
            if col + 1 < grid and active[idx + 1]:
                edges += 1
            if row + 1 < grid and active[idx + grid]:
                edges += 1
    vertices = sum(active)
    components = _count_components(active, grid)
    return {
        "active_vertices": vertices,
        "active_edges": edges,
        "betti_0": components,
        "betti_1": max(0, edges + components - vertices),
        "threshold": threshold,
    }


def _count_components(active: list[bool], grid: int) -> int:
    seen = [False] * len(active)
    components = 0
    for idx, is_active in enumerate(active):
        if not is_active or seen[idx]:
            continue
        components += 1
        stack = [idx]
        seen[idx] = True
        while stack:
            current = stack.pop()
            row, col = divmod(current, grid)
            neighbors = []
            if row > 0:
                neighbors.append(current - grid)
            if row + 1 < grid:
                neighbors.append(current + grid)
            if col > 0:
                neighbors.append(current - 1)
            if col + 1 < grid:
                neighbors.append(current + 1)
            for neighbor in neighbors:
                if active[neighbor] and not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
    return components


def _residual_summary(samples: list[dict], grid: int) -> dict:
    if len(samples) < grid * grid or grid < 3:
        return {"divergence_e": 0.0, "divergence_b": 0.0, "faraday_curl": 0.0}
    dx = 2.0 / (grid - 1)
    div_e = div_b = curl = count = 0.0
    for row in range(1, grid - 1):
        for col in range(1, grid - 1):
            idx = row * grid + col
            left, right = samples[idx - 1], samples[idx + 1]
            down, up = samples[idx - grid], samples[idx + grid]
            d_ex_dx = (right["electric"]["x"] - left["electric"]["x"]) / (2.0 * dx)
            d_ey_dy = (up["electric"]["y"] - down["electric"]["y"]) / (2.0 * dx)
            d_bx_dx = (right["magnetic"]["x"] - left["magnetic"]["x"]) / (2.0 * dx)
            d_by_dy = (up["magnetic"]["y"] - down["magnetic"]["y"]) / (2.0 * dx)
            curl_z = (
                right["electric"]["y"]
                - left["electric"]["y"]
                - up["electric"]["x"]
                + down["electric"]["x"]
            ) / (2.0 * dx)
            scale = 1.0 + samples[idx]["energy_density"]
            div_e += abs((d_ex_dx + d_ey_dy) / scale)
            div_b += abs((d_bx_dx + d_by_dy) / scale)
            curl += abs(curl_z / scale)
            count += 1.0
    return {"divergence_e": div_e / count, "divergence_b": div_b / count, "faraday_curl": curl / count}


def _gaussian(x: float, y: float, sigma: float) -> float:
    return math.exp(-(x * x + y * y) / sigma)


def _norm(vector: dict[str, float]) -> float:
    return math.sqrt(vector["x"] * vector["x"] + vector["y"] * vector["y"] + vector["z"] * vector["z"])


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Faraday tensor electromagnetic coupling data.")
    parser.add_argument("--kind", choices=["toroidal_pulse", "braided_pair", "boundary_sheaf"], default="toroidal_pulse")
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--coupling", type=float, default=0.7)
    parser.add_argument("--phase", type=float, default=0.25)
    parser.add_argument("--separation", type=float, default=0.5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = simulate_scenario(
        ScenarioConfig(
            kind=args.kind,
            grid=args.grid,
            steps=args.steps,
            coupling=args.coupling,
            phase=args.phase,
            separation=args.separation,
        )
    )
    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0
