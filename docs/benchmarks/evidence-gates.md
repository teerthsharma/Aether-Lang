# Evidence Gates

This page maps claims to required evidence.

| Claim type | Required evidence |
| --- | --- |
| Parser support | Lexer/parser unit test plus accepted example |
| Interpreter behavior | Interpreter unit test checking resulting `Value` |
| Titan VM behavior | VM test for compiled opcode path |
| Topology correctness | Known fixture with expected Betti numbers or intervals |
| Witness-mode behavior | Test showing landmark cap and non-empty diagram |
| ML primitive behavior | Unit test for algorithm output on deterministic input |
| CLI behavior | Command smoke test or captured output |
| no_std compatibility | `cargo check --no-default-features` for target crate |
| Speed claim | Benchmark artifact with baseline and correctness metric |
| Security claim | Corpus, threat model, false-positive and false-negative records |
| Hardware claim | Hardware logs, environment, repeatable command, failure mode |

## Current Gaps

- No committed E2E claim artifact.
- No external TDA parity benchmark page.
- No complete Titan parity matrix.
- No production binary-authentication corpus.
- No hardware power or boot artifact.

These are gaps in evidence, not necessarily gaps in implementation intent.
