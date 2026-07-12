# Contribution Standard

A small language/runtime repository still needs claim discipline.

## Code

- Public Rust APIs should return typed errors or documented `Result` values
  when failure is expected.
- `unwrap()` in production paths needs a local invariant that is clear from the
  surrounding code.
- Unsafe code needs a `Safety` comment and a test or review note for the
  boundary.
- Optional dependencies must stay optional for default builds unless the crate
  contract changes.
- Parser features need lexer, parser, interpreter or VM coverage as applicable.

## Docs

Every concept page should include:

- the active implementation path;
- the formula or state transition if there is one;
- plain mechanical meaning;
- the active claim;
- the gated or roadmap claim;
- a failure mode.

## Benchmarks

Every speed claim needs:

- raw artifact;
- baseline list;
- hardware/software environment;
- correctness metric;
- seed or determinism note.

## Tone

Use neutral systems language. Prefer "the implementation does X under Y
condition" over broad promotional phrasing. Benefits should emerge from the
mechanism being documented: representation, invariant, and gate.

## Commits

Group related work into coherent commits that can be reviewed independently.
