# Syntax

Aether source is statement-oriented. Newlines and `~` can terminate statements.

## Literals And Variables

```aether
let x = 10~
let y = 3.14~
let ok = true~
let name = "aether"~
let values = [1.0, 2.0, 3.0]~
```

The parser accepts optional type-hint style declarations:

```aether
point C = [1.0, 2.0, 3.0]~
```

Current type hints are parsed as syntax. They are not a full static type system.

## Expressions

Active expression operators:

- arithmetic: `+`, `-`, `*`, `/`, `%`;
- comparison: `<`, `>`, `<=`, `>=`, `==`, `!=`;
- logical: `&&`, `||`, `!`;
- ranges: `0..4` for `for` loops and `0:64` for slices.

## Control Flow

```aether
if count == 0 {
  print("empty")~
} else {
  print("nonempty")~
}

while count < 3 {
  count = count + 1~
}

for i in 0..4 {
  total = total + i~
}
```

`break` and `continue` are active inside loops.

## Seal Loop

```aether
seal until count >= 3 {
  count = count + 1~
}
```

The active implementation evaluates the `until` expression. Topological
convergence syntax can be parsed in regression configuration, but full
language-level convergence semantics should be treated as gated unless a test
covers the exact path.

## Functions

```aether
fn add(a, b) {
  return a + b~
}

let result = add(2, 3)~
```

Functions return explicit `return` values or the last value produced by the
body.

## Manifolds And Blocks

```aether
let data = [1.0, 2.0, 3.0, 4.0]~
manifold M = embed(data, tau=1)~
block B = M.cluster(0:2)~
```

The active interpreter uses a fixed 3D embedding workspace. `tau` is used.
`dim` can be parsed but is not the runtime dimension selector in the current
interpreter path.
