# Aether Lang (AEGIS)

Aether Lang is the language + runtime for the AEGIS ecosystem: a 3D manifold-based ML language that embeds data into geometry and detects **topological convergence** instead of relying on arbitrary loss thresholds.

## Highlights

- **3D manifold embeddings** using Takens-style time-delay reconstruction.
- **Blocks and geometric primitives** for clustering, centroids, and spread analysis.
- **Escalating regression** with topological convergence checks.
- **Interactive REPL** and script runner (`aether` CLI).
- **Rust-first, no_std-friendly core** with kernel experimentation support.

## Quickstart

### Docker (recommended)

```bash
docker pull teerthsharma/aether
docker run -it teerthsharma/aether repl
```

Run a script from your local folder:

```bash
docker run -v $(pwd):/scripts teerthsharma/aether run /scripts/hello.aether
```

### Build from source

```bash
rustup install nightly
rustup component add rustfmt clippy

git clone https://github.com/teerthsharma/aether.git
cd aether
cargo build -p aether-cli --release
./target/release/aether repl
```

## Hello, manifold

```aether
// hello.aether
manifold M = embed(data, dim=3, tau=5)
block B = M.cluster(0:64)
centroid C = B.center

render M {
    color: by_density
}
```

> **Note:** In the REPL, statements are terminated with `~`. Scripts use the `.aether` (or `.ae`) extension.

## CLI usage

```bash
aether repl                 # Interactive REPL
aether run path/to/file.aether
aether check path/to/file.aether
```

## Documentation

- [Getting Started](docs/GETTING_STARTED.md)
- [Language Guide](docs/LANGUAGE.md)
- [Syntax Reference](docs/SYNTAX.md)
- [Examples](docs/EXAMPLES.md)
- [Architecture](docs/ARCHITECTURE.md)
- [FAQ](docs/FAQ.md)

## Repository layout

- `crates/aether-lang` — Lexer, parser, AST, interpreter, VM
- `crates/aether-core` — Manifolds, ML, math, geometry
- `crates/aether-cli` — `aether` command-line interface
- `crates/aegis-*` — Compatibility crates and alternate CLI
- `docs/` — Full documentation and references
- `examples/` — Example Aether scripts

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow and standards.

## License

MIT — see [LICENSE](LICENSE).
