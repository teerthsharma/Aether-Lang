# Aether Lang: Practical Guide

Aether Lang (the `aether-lang` crate) is an early-stage language + runtime in the AEGIS ecosystem. This guide is intentionally **down to earth** and describes what the current parser/interpreter actually supports today.

## Running Aether

```bash
aether repl
aether run path/to/file.aether
aether check path/to/file.aether
```

Notes:
- `.aether` and `.ae` are the preferred extensions. Other extensions will run but print a warning.
- `--mode titan` is available but experimental.

## Program basics

### Comments

```aether
// Single-line comment
```

### Statements and blocks

- Statements are separated by newlines.
- Blocks use braces `{ ... }`.

### Literals

- Numbers: `1`, `3.14`
- Strings: `"hello"`
- Booleans: `true`, `false`

### Variables

```aether
let x = 10
let name = "Aether"
```

Type hints are parsed but not enforced:

```aether
int count = 3
point C = some_expr
```

### Arithmetic

Only numeric `+`, `-`, `*`, `/` are supported today.

### Lists

```aether
let values = [1, 2, 3]
values.push(4)
let count = values.len()
let last = values.pop()
```

Nested lists are supported (used for tensors in the ML helpers).

## Geometry primitives

### Manifolds

```aether
manifold M = embed(data, dim=3, tau=5)
```

Current behavior:
- The runtime uses **built-in sample data** (a small sine wave), not your `data` input yet.
- The embedding dimension is fixed at **3**; `dim` is accepted but ignored.
- `tau` is parsed and used.

### Blocks

```aether
block B = M.cluster(0:64)
block C = M[64:128]
```

Blocks are created and stored, but **block properties** (e.g., `B.center`) are not yet exposed in the interpreter.

### Render

```aether
render M {
    color: by_density,
    trajectory: true
}
```

The `render` statement is parsed but is currently a **no-op** in the runtime.

## Regression

```aether
regress {
    model: "polynomial",
    degree: 3,
    escalate: true,
    until: convergence(1e-6)
}
```

Current behavior:
- Uses the **first** manifold in the program.
- `escalate` only changes the number of training epochs.
- `model`, `degree`, `target`, and `until` are parsed but **not applied** in the interpreter yet.

## Control flow

```aether
if flag {
    print("on")
} else {
    print("off")
}

while running {
    print("tick")
}
```

Notes:
- Conditions must evaluate to **booleans**. Comparison operators (`<`, `>`, `==`) are tokenized but **not parsed** yet.
- `seal { ... }` runs the block a fixed **1000 iterations** (no condition support yet).
- `for`, `fn`, `return`, `break`, and `continue` are parsed but do not have runtime behavior yet.

## Modules and built-ins

### `print`

```aether
print("hello")
```

### `math` module

```aether
import math
from math import sin
```

Available: `pi`, `sin`, `cos`, `sqrt`, `exp`.

### `topology` module

```aether
import topology
```

Exposes `Betti`, which currently returns placeholder values.

### `Ml` module

```aether
import Ml
let net = Ml.MLP(0.01)
net.add_layer(2, 4, "relu")
let loss = net.train([[0, 0], [1, 1]], [[0], [1]], 10)
```

Available today:
- Constructors: `MLP`, `KMeans`, `Conv2D`
- Helpers: `load_weights`, `matmul`, `add`, `relu`, `softmax`

Notes:
- `MLP.add_layer`, `MLP.train`, and `MLP.predict/forward` are wired.
- `KMeans` and `Conv2D` are constructible but have no methods wired yet.
- Several other ML helper names exist but are currently stubs.

### `Seal` module

```aether
import Seal
let result = Seal.train(net, data, targets)
```

`Seal.train` is parsed but **currently returns unit**.

## Summary of current limitations

- No comparison or logical operators in expressions.
- No unary operators.
- `render` does not render yet.
- `embed` ignores input data and uses a built-in sample series.
- `until`/topological convergence is parsed but not applied.
- Class methods are parsed but not executed (fields are stored).

## Next references

- [Syntax Reference](SYNTAX.md)
- [Getting Started](GETTING_STARTED.md)
- [Examples](EXAMPLES.md)
