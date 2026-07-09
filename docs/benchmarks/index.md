# Benchmark Policy

Benchmarks are evidence artifacts, not hand-written speed tables.

## Output Contract

A benchmark artifact should record:

- benchmark id;
- source commit;
- crate or binary under test;
- input generator and seed;
- sample size;
- hardware and operating system;
- build profile;
- correctness metric;
- elapsed time;
- memory measurement if available;
- warning list;
- baseline list.

## Allowed Claims

Allowed without full benchmark study:

- "unit tests cover this behavior";
- "this command builds";
- "this timing is a smoke record";
- "this page describes a roadmap target."

Not allowed without artifacts:

- speedup factors;
- lower memory footprint;
- production readiness;
- security detection rates;
- external framework parity;
- hardware acceleration.

## Local Checks

```powershell
cargo fmt --all -- --check
cargo test -p aether-core
cargo test -p aether-lang
cargo test -p aether-cli
cargo check -p aether-core --no-default-features
python -m mkdocs build --strict
```

## Current Benchmark Status

The repository has many unit tests. It does not currently expose a complete
E2E claim benchmark artifact equivalent to the reference project's
`e2e_claims.py` gate. Until that exists, performance pages should describe
policy and local checks rather than speedup tables.
