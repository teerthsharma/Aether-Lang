# Status Matrix

## Active

| Surface | Evidence |
| --- | --- |
| Lexer tokens for current syntax | `lexer.rs` tests |
| Parser for statements and operators | `parser.rs` tests |
| Interpreter assignments, loops, functions | `interpreter.rs` tests |
| Numeric-list manifold embedding | `interpreter.rs` test |
| `topology.ph` and `topology.betti` | `interpreter.rs` topology test |
| Bounded persistent homology H0/H1/H2 | `persistence.rs` tests |
| Lazy witness mode | `persistence.rs` test |
| Block metadata and compression selection | `aether.rs` tests |
| Drift detector | `aether.rs` test |
| Sparse graph and pipeline | `manifold.rs` tests |
| Geometric governor | `governor.rs` tests |
| Sparse scheduler | `scheduler.rs` tests |
| CLI parse-error formatting | `aether-cli` test |

## Partial Or Gated

| Surface | Gate |
| --- | --- |
| Titan VM language parity | VM tests per construct |
| Full static type checking | Static checker and diagnostics |
| Complete class/object semantics | Interpreter tests and docs |
| Render as a user-facing graphics command | CLI artifact or exported file test |
| `Seal.train` semantic contract | Interpreter test and training artifact |
| ML model quality | Deterministic datasets and baseline metrics |
| External TDA parity | ripser/GUDHI-style fixtures |
| no_std workspace claim | `cargo check --no-default-features` in CI |
| Bare-metal product claim | Boot logs and hardware matrix |
| Security detection claim | Threat model, corpus, metrics |

## Removed From Active Claims

- Unverified speedup factors.
- Placeholder benchmark rows.
- Broad "verified full trust chain" language for this repository.
- Production security guarantees.
- Hardware acceleration claims.
