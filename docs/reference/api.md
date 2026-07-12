# API Reference

This page is a compact public map. It is not generated Rustdoc.

## `aether-lang`

| Type | Path | Role |
| --- | --- | --- |
| `Lexer` | `lexer.rs` | Converts source text into tokens |
| `TokenKind` | `lexer.rs` | Token vocabulary |
| `Parser` | `parser.rs` | Builds AST from token stream |
| `ParseError` | `parser.rs` | Parse error with line and column |
| `Program` | `ast.rs` | Top-level statement list |
| `ExprKind` | `ast.rs` | Expression node variants |
| `StmtKind` | `ast.rs` | Statement node variants |
| `Interpreter` | `interpreter.rs` | Executes AST programs |
| `Value` | `interpreter.rs` | Runtime value enum |
| `TitanVM` | `vm.rs` | Stack-based VM |
| `Compiler` | `vm.rs` | AST-to-opcode compiler |
| `AsciiRenderer` | `ascii_render.rs` | ASCII point-cloud rendering |
| `WebGLExporter` | `webgl_export.rs` | HTML/WebGL point-cloud export |

## `aether-core`

| Type or function | Path | Role |
| --- | --- | --- |
| `ManifoldPoint<D>` | `manifold.rs` | Point in D-dimensional space |
| `TimeDelayEmbedder<D>` | `manifold.rs` | Delay-coordinate embedding |
| `SparseAttentionGraph<D>` | `manifold.rs` | Epsilon-neighborhood graph |
| `TopologicalPipeline<D>` | `manifold.rs` | Streaming topology pipeline |
| `BlockMetadata<D>` | `aether.rs` | Centroid/radius/variance/concentration |
| `HierarchicalBlockTree<D>` | `aether.rs` | Multi-level block summary tree |
| `DriftDetector<D>` | `aether.rs` | Centroid drift tracking |
| `persistent_homology` | `persistence.rs` | Bounded PH engine |
| `time_delay_persistence` | `persistence.rs` | Samples to PH diagram |
| `PersistenceConfig` | `persistence.rs` | PH bounds and complex selection |
| `ComplexKind` | `persistence.rs` | Vietoris-Rips or witness mode |
| `PersistenceDiagram` | `persistence.rs` | Persistence pair collection |
| `BettiNumbers3` | `persistence.rs` | `beta_0`, `beta_1`, `beta_2` |
| `TopologicalShape` | `topology.rs` | Binary-shape heuristic result |
| `verify_shape` | `topology.rs` | Binary-shape gate |
| `GeometricGovernor` | `governor.rs` | Adaptive epsilon controller |
| `SystemState<D>` | `state.rs` | Scheduler state vector |

## `aether-cli`

| Command | Role |
| --- | --- |
| `aether repl` | Interactive interpreter session |
| `aether run <file>` | Parse and execute a script |
| `aether run <file> --mode titan` | Compile and run with Titan VM |
| `aether check <file>` | Parse-only syntax check |

## `aether-kernel`

| Type or function | Role |
| --- | --- |
| `SparseScheduler<D>` | State-deviation wake gate |
| `verify_elf` | ELF header validation |
| `verify_binary_topology` | Loader-facing topology heuristic |
| `init_heap` | Allocator initialization |
| `HardwareTopology` | Boot-time hardware topology record |
