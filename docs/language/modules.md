# Module Contracts

The interpreter exposes modules through `Value::Module` and native functions.

<div class="ts-viz" data-viz="pipe-modules" data-title="Module contracts" data-caption="Pick a module: the import path it takes through the interpreter, its active names, and whether its contract is active or gated."></div>


## `math`

Active names:

- `sin`;
- `cos`;
- `sqrt`;
- `exp`;
- `pi`.

## `topology`

Active names:

- `topology.ph(manifold, ...)`;
- `topology.betti(diagram_or_manifold, radius=...)`;
- `topology.intervals(diagram)`;
- `topology.Betti(...)` alias path.

## `Ml`

Active construction and helper surface includes:

- `Ml.MLP(...)`;
- `Ml.KMeans(...)`;
- `Ml.Conv2D(...)`;
- tensor helpers such as matrix multiply, add, ReLU, and softmax through native
  function dispatch.

Individual ML methods must be documented from tests, not from constructor
presence alone. A constructible object is not the same as a complete algorithmic
contract.

## `Seal`

`Seal.train` is exposed as a native function entrypoint. Treat training-quality
or topological-stop behavior as gated unless the exact call path has a test or
benchmark artifact.
