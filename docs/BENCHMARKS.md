# Benchmarks

The benchmark policy is maintained at [benchmarks/index.md](benchmarks/index.md).

Legacy speedup tables have been removed from the active docs surface because
they did not include raw artifacts, baselines, environment records, correctness
metrics, or seeds.

Use local checks as smoke evidence:

```powershell
cargo fmt --all -- --check
cargo test -p aether-core
cargo test -p aether-lang
cargo test -p aether-cli
cargo check -p aether-core --no-default-features
python -m mkdocs build --strict
```
