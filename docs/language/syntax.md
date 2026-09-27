# Syntax

Aether source is statement-oriented. Newlines and `~` can terminate statements.

<div class="ts-viz" data-viz="pipe-lexer" data-preset="syntax" data-title="Source to tokens to AST" data-caption="Every snippet on this page, tokenized and parsed by a port of lexer.rs and parser.rs; ~ and newlines both appear as separators."></div>


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

A seal loop has three stopping rules, all capped at 1,000 passes.

| Condition | Stops | Kind |
|---|---|---|
| `until expr` | before a pass, when `expr` is `true` | any boolean |
| `until convergence(ε)` | after pass $i \ge 2$ with $d(v_i, v_{i-1}) \le \varepsilon$ | scalar tolerance |
| `until stable(expr)` | before a pass, when `expr` equals its value one pass earlier | exact invariant |

For `convergence`, $v_i$ is the value of the body's last statement after pass
$i$, and $d$ is the max norm: $\lvert x - y \rvert$ on numbers, the maximum over
elements on lists and records of equal shape ($+\infty$ if the shape changes),
and the difference of `final_error` on regression results. A body value with no
distance, and a negative $\varepsilon$, are refused by name.

```aether
let x = 1~
let n = 0~
🦭 until convergence(1e-6) {
  n = n + 1~
  x = (x + 2 / x) / 2~
  x~
}
print([n, x])~
```

Newton's iteration for $\sqrt2$ prints `[5, 1.414213562373095]`. The fourth
pass moves $x$ by $2.1\times10^{-6}$ and the fifth by $1.6\times10^{-12}$.
`crates/aether-lang/tests/seal_convergence.rs` pins this case.

`stable` is the topological form when the watched value is topological. For
example, `seal until stable(topology.betti(topology.ph(M), radius=r))` stops on
a Betti vector, and `seal until stable(euler(segs).faces)` stops on a certified
face count (see [Integrated Mathematics](../integrated/index.md)). Equality is
exact, so stability is a one-pass window.

Numeric literals are six-decimal fixed point. Exponents (`1e-6`, `2.5e3`)
scale them exactly. A literal the representation cannot hold, such as `1e-7`,
is a lexer error rather than a silent zero. Parentheses group:
`(x + 2 / x) / 2`.

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
