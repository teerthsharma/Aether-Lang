# Electromagnetic Field Data Simulator

The Electromagnetic Field Data Simulator is a topological electromagnetic
coupling toolkit inside Aether-Lang. It generates and visualizes educational
Faraday tensor scenarios for institutions, researchers, and students who need an
interactive bridge between field theory, discrete topology, and reproducible
simulation data.

The simulator has three coordinated surfaces:

- Rust reference core: `crates/aether-em-sim`
- Python dataset generator: `python/aether_em_sim`
- React/GitHub Pages app: `web/em-simulator`

The design goal is not to replace full finite-element or finite-difference EM
solvers. It is a fast, inspectable simulator for coupling intuition: Faraday
tensor construction, Lorentz field invariants, Maxwell-style residuals, energy
flow, and topological summaries of active field regions.

## Mathematical Model

The primitive object is the Faraday tensor `F_mu_nu`, represented as an
antisymmetric 4 by 4 tensor built from electric field `E` and magnetic field `B`:

```text
F =
[  0   Ex   Ey   Ez ]
[ -Ex   0  -Bz   By ]
[ -Ey  Bz    0  -Bx ]
[ -Ez -By   Bx    0 ]
```

The implementation tests enforce:

```text
F_mu_nu + F_nu_mu = 0
F_mu_mu = 0
```

From `F`, the simulator derives two Lorentz-style field invariants:

```text
I1 = |B|^2 - |E|^2 / c^2
I2 = E dot B
```

The code uses normalized units with `c = 1` for interactive work. That makes the
fields easy to inspect without unit-conversion noise, while keeping the tensor
and invariant structure explicit.

## Topological Object

The topological object is a sampled field graph on a 2D spacetime slice:

```text
vertices = grid samples whose energy density exceeds an adaptive threshold
edges    = 4-neighborhood adjacency between active vertices
```

The simulator reports:

```text
betti_0 = connected active field components
betti_1 = active_edges + betti_0 - active_vertices
```

This is a graph-cycle Betti proxy, not a full Vietoris-Rips persistence pass.
That choice is deliberate for the hosted app: the browser can recompute the
field and topology instantly as users drag controls. For higher-fidelity
research workflows, the exported JSON can be passed into the existing Aether
persistent-homology stack.

## Scenario Families

### Toroidal Pulse

The toroidal pulse creates a ring-like coupling region. The electric component
flows tangentially around the ring while the magnetic component is radial with a
phase-shifted axial component. This produces a stable active-cycle topology and
a compact example for introducing `F_mu_nu`, Poynting flow, and a beta1 cycle.

Use it for:

- ring resonator intuition
- field-line circulation
- topology of active energy bands
- classroom demonstrations of antisymmetric tensors

### Braided Pair

The braided pair models two rotating field lobes with opposite centers. It is
useful for showing how coupling can move across the grid while preserving
observable structure in the active field graph.

Use it for:

- coupled emitters
- moving interference regions
- time-indexed datasets
- topology-aware feature extraction

### Boundary Sheaf

The boundary sheaf scenario places activity along a narrow interface with
opposing boundary sources. It is named as a sheaf-inspired diagnostic: local
patches have field summaries, and the interface exposes whether neighboring
patch summaries remain compatible.

Use it for:

- boundary effects
- local-to-global consistency intuition
- interface coupling
- qualitative sheaf/cohomology discussions

## Maxwell Residual Diagnostics

The simulator reports bounded diagnostics over the final frame:

```text
divergence_e
divergence_b
faraday_curl
```

These are finite-difference educational residuals. They are not a substitute for
a production Maxwell solver with boundary conditions, material models, and
stability proofs. They are meant to make numerical behavior visible when a user
changes grid size, separation, coupling, or phase.

## Rust API

The Rust crate is the reference numerical implementation.

```rust
use aether_em_sim::{
    simulate_scenario, CouplingKind, FaradayTensor, ScenarioConfig, Vec3,
};

let tensor = FaradayTensor::from_fields(
    Vec3::new(0.5, 0.1, 0.0),
    Vec3::new(0.0, 0.3, 0.2),
);

let invariants = tensor.invariants();

let dataset = simulate_scenario(ScenarioConfig {
    kind: CouplingKind::ToroidalPulse,
    grid: 32,
    steps: 12,
    coupling: 0.72,
    phase: 0.25,
    separation: 0.55,
});
```

Run the focused Rust tests:

```bash
cargo test -p aether-em-sim
```

The crate supports the same feature shape as the rest of Aether core:

```text
default = ["std"]
std     = ["alloc"]
alloc   = []
no_std  = ["alloc"]
```

## Python Dataset Generator

The Python package mirrors the reference equations and provides a JSON export
path for reproducible teaching datasets.

```bash
PYTHONPATH=python python -m aether_em_sim \
  --kind toroidal_pulse \
  --grid 32 \
  --steps 8 \
  --coupling 0.72 \
  --phase 0.25 \
  --separation 0.55 \
  --out web/em-simulator/public/data/sample-scenario.json
```

Python API:

```python
from aether_em_sim import ScenarioConfig, simulate_scenario

dataset = simulate_scenario(
    ScenarioConfig(
        kind="braided_pair",
        grid=24,
        steps=6,
        coupling=0.64,
        phase=0.2,
        separation=0.5,
    )
)
```

Run the focused Python tests:

```bash
PYTHONPATH=python python -m pytest tests/python/test_aether_em_sim.py
```

## React and GitHub Pages App

The app is a Vite React surface built for GitHub Pages. It lets users inspect:

- energy density heatmap
- Poynting direction strokes
- live Faraday tensor at the peak-energy sample
- Lorentz invariants
- active field graph counts
- beta0 and beta1 topology proxy
- Maxwell residual diagnostics
- JSON export of the current interactive dataset

Local development:

```bash
cd web/em-simulator
npm ci
npm test
npm run build
npm run dev
```

The Pages workflow is in `.github/workflows/pages.yml`. On pushes to `main`, it
installs dependencies, runs the Vitest suite, builds the Vite app, and deploys
`web/em-simulator/dist` with GitHub Pages.

The configured Vite base path is:

```text
/Aether-Lang/
```

That matches the intended repository URL:

```text
https://teerthsharma.github.io/Aether-Lang/
```

## Verification Gates

The simulator follows the topology discovery discipline used elsewhere in the
Aether work:

```text
Object: sampled electromagnetic field graph on a spacetime slice
Invariant: Faraday antisymmetry, Lorentz invariants, beta0/beta1 graph proxy
Rule: active energy regions become a graph for component and cycle summaries
Baseline: direct field energy and finite-difference residuals
Failure gates: unstable topology, high residuals, slow browser interaction,
               excessive generated fixture size, or untested tensor signs
```

Current focused checks:

```bash
cargo test -p aether-em-sim
PYTHONPATH=python python -m pytest tests/python/test_aether_em_sim.py
cd web/em-simulator && npm test && npm run build && npm audit
```

## Repository Integration

This module fills the previous alignment gaps found by the math extraction pass:

- Faraday tensor: implemented and tested in Rust, Python, and TypeScript
- Electromagnetic coupling: three scenario families
- Discretized spacetime surface: grid-sampled field frames
- Topological summary: active field graph with beta0/beta1 proxy
- React visualization: Vite app prepared for GitHub Pages
- Dataset export: Python CLI and browser JSON export
- Documentation: this file plus README entry points

## Limitations

- The topology summary is a graph proxy. Use the Aether persistent-homology
  primitives for full filtered-complex analysis.
- The Maxwell residuals are educational finite-difference diagnostics, not a
  formal solver guarantee.
- The browser simulator prioritizes immediate interaction over high-resolution
  offline runs.
- Material models, anisotropic media, boundary-condition solvers, and GPU kernels
  are future layers, not part of this initial hosted toolkit.

## Next Research Extensions

1. Add a Rust-to-WASM path so the React app can call the reference Rust core
   directly.
2. Export active field graphs into Aether's persistent-homology APIs.
3. Add material presets for dielectric, waveguide, and metamaterial teaching
   scenarios.
4. Add benchmark fixtures comparing topology summaries at multiple grid
   resolutions.
5. Add a notebook that validates invariant stability over phase sweeps.
