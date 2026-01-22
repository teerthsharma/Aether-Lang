# AEGIS Language Specification

> Version 0.1.0

## 1. Overview

AEGIS is a domain-specific language for manifold-native machine learning, where code operates in 3D geometric space.

## 2. Syntax

### 2.1 Comments

```aegis
// Single-line comment
```

### 2.2 Manifold Declaration

```aegis
manifold <name> = embed(<source>, dim=<int>, tau=<int>)
```

**Parameters:**
- `source`: Input data source (e.g., `data`, file path)
- `dim`: Embedding dimension (1-16, default: 3)
- `tau`: Time delay for Takens embedding (default: 1)

**Example:**
```aegis
manifold M = embed(data, dim=3, tau=5)
```

### 2.3 Block Declaration

```aegis
block <name> = <manifold>.cluster(<start>:<end>)
block <name> = <manifold>[<start>:<end>]
```

**Example:**
```aegis
block B = M.cluster(0:64)
block B2 = M[64:128]
```

### 2.4 Variable Assignment

```aegis
<type_hint> <name> = <expression>
<name> = <expression>
```

**Example:**
```aegis
centroid C = B.center
radius R = B.spread
x = 42
```

### 2.5 Regression Statement

```aegis
regress {
    model: <string>,
    degree: <int>,         // optional
    target: <expression>,  // optional
    escalate: <bool>,
    until: <convergence>
}
```

**Model Types:**
- `"linear"` - Linear regression
- `"polynomial"` - Polynomial regression
- `"rbf"` - Radial Basis Function
- `"gp"` - Gaussian Process
- `"geodesic"` - Manifold geodesic regression

**Convergence Conditions:**
- `convergence(<epsilon>)` - Error threshold
- `betti_stable(<epochs>)` - Betti number stability

**Example:**
```aegis
regress {
    model: "polynomial",
    degree: 3,
    escalate: true,
    until: convergence(1e-6)
}
```

### 2.6 Render Statement

```aegis
render <manifold> {
    color: <mode>,
    highlight: <block>,
    trajectory: <bool>,
    axis: <int>
}
```

**Color Modes:**
- `by_density` - Color by point density
- `by_cluster` - Color by cluster assignment
- `gradient` - Gradient along axis

**Example:**
```aegis
render M {
    color: by_density,
    highlight: B,
    trajectory: on
}
```

## 3. Data Types

| Type | Description | Example |
|------|-------------|---------|
| `int` | Integer | `42`, `-7` |
| `float` | Floating point | `3.14159` |
| `bool` | Boolean | `true`, `false` |
| `string` | Text | `"polynomial"` |
| `manifold` | 3D embedded space | `M` |
| `block` | Geometric region | `B` |
| `point` | D-dimensional point | `C` |

## 4. Built-in Functions

### embed(source, dim, tau)
Embed 1D time-series into D-dimensional manifold using Takens' theorem.

### convergence(epsilon)
Convergence condition based on error threshold.

### betti_stable(epochs)
Convergence when Betti numbers stable for N epochs.

## 5. Properties

### Block Properties

| Property | Type | Description |
|----------|------|-------------|
| `.center` | point | Centroid of block |
| `.spread` | float | Radius (max deviation) |
| `.variance` | float | Variance of points |
| `.count` | int | Number of points |

### Manifold Properties

| Property | Type | Description |
|----------|------|-------------|
| `.center` | point | Global centroid |
| `.betti` | (int, int) | Betti numbers (β₀, β₁) |
| `.dim` | int | Embedding dimension |

## 6. Grammar (EBNF)

```ebnf
program     = { statement } ;
statement   = manifold_decl | block_decl | var_decl | regress_stmt | render_stmt ;

manifold_decl = "manifold" IDENT "=" expr ;
block_decl   = "block" IDENT "=" expr ;
var_decl     = [ IDENT ] IDENT "=" expr ;
regress_stmt = "regress" config_block ;
render_stmt  = "render" IDENT [ config_block ] ;

config_block = "{" { config_pair } "}" ;
config_pair  = IDENT ":" expr [ "," ] ;

expr         = primary { "." IDENT [ call_args ] } ;
primary      = NUMBER | STRING | BOOL | IDENT | call_expr | index_expr ;
call_expr    = IDENT call_args ;
call_args    = "(" [ arg { "," arg } ] ")" ;
arg          = expr | IDENT "=" expr ;
index_expr   = IDENT "[" NUMBER ":" NUMBER "]" ;

IDENT        = letter { letter | digit | "_" } ;
NUMBER       = digit { digit } [ "." digit { digit } ] ;
STRING       = '"' { char } '"' ;
BOOL         = "true" | "false" ;
```

## 7. Examples

### Hello World

```aegis
// Create manifold
manifold M = embed(data, dim=3)

// Render it
render M {
    color: gradient
}
```

### Escalating Regression

```aegis
manifold M = embed(sensor_data, dim=3, tau=7)

regress {
    model: "polynomial",
    escalate: true,
    until: convergence(1e-8)
}
```

### Cluster Analysis

```aegis
manifold M = embed(data, dim=3, tau=5)

block A = M[0:50]
block B = M[50:100]

centroid_a = A.center
centroid_b = B.center

render M {
    color: by_cluster,
    trajectory: on
}
```
