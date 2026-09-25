# Runtime Surface

Aether is a workspace, not a single binary.

| Component | Path | Active role |
| --- | --- | --- |
| Language crate | `crates/aether-lang` | Lexer, parser, AST, interpreter, Titan VM, exporters |
| Core crate | `crates/aether-core` | Manifolds, topology, ML primitives, governors, state |
| CLI crate | `crates/aether-cli` | REPL, script runner, syntax checker |
| Kernel crate | `crates/aether-kernel` | no_std sparse scheduler, loader, allocator, boot scaffolding |
| Compatibility core | `crates/aegis-core` | Legacy compatibility surface |
| Compatibility CLI | `crates/aegis-cli` | Legacy compatibility binary |

<div class="ts-viz" data-viz="pipe-workspace" data-title="Cargo workspace map" data-caption="Select a crate to see its path dependencies, dependents and external crates, read from each crates/*/Cargo.toml."></div>


## Runtime Objects

The interpreter stores named variables in a map from identifiers to runtime
values. Handles are used for manifolds, blocks, classes, and objects so large
state stays inside interpreter-owned arenas.

```text
identifier -> Value::Manifold(handle) -> manifolds[handle]
identifier -> Value::Block(handle) -> blocks[handle]
identifier -> Value::Persistence(diagram)
identifier -> Value::Tensor(tensor)
```

## Extension Boundary

New language-level features should pass through the same path:

1. token kind;
2. AST node;
3. parser rule;
4. interpreter behavior or VM opcode;
5. tests;
6. documentation status update.

Skipping one stage creates a parsed-only feature or a runtime-only feature.

<div class="ts-viz" data-viz="pipe-extension" data-title="Extension boundary check" data-caption="Untick a stage to see which kind of partial feature it leaves behind."></div>

