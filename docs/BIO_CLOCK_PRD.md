# Bio Clock Status

The Bio Clock material is a design concept for memory and lifecycle behavior.
It is not an active user-facing runtime contract unless backed by code paths and
tests.

Current related surfaces:

- `crates/aegis-core/src/memory.rs`;
- `crates/aether-core/src/memory.rs`;
- [Runtime Surface](concepts/runtime-surface.md);
- [Status Matrix](reference/status.md).

Future documentation should state the allocator or memory structure being used,
the invariant it maintains, and the test or benchmark that verifies it.
