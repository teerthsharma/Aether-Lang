# ═══════════════════════════════════════════════════════════════════════════════
# AETHER: topological ML language — CLI image
#
# Build: docker build -t aether .
# Run:   docker run --rm -it aether repl
# Help:  docker run --rm aether --help
# ═══════════════════════════════════════════════════════════════════════════════

FROM rust:1-bookworm AS builder

# rust-toolchain.toml pins nightly; rustup honours it on first cargo invocation.
#
# `--component` takes one value per flag. Passing two names after a single flag
# made rustup read the second as a toolchain:
#   error: invalid value 'llvm-tools-preview' for '[TOOLCHAIN]...'
RUN rustup toolchain install nightly \
    --component rust-src \
    --component llvm-tools-preview

WORKDIR /build

# Crates live under crates/. The previous Dockerfile copied them from the repo
# root, where they have never been, so every image build failed.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates
COPY examples ./examples

# aether-kernel is bare-metal and cannot link for a host target, so build the
# CLI binary specifically rather than the whole workspace.
RUN cargo build --release -p aether-cli

# ═══════════════════════════════════════════════════════════════════════════════
# Runtime
#
# The Python ML stack the previous image installed (torch, transformers,
# sentencepiece, protobuf — several GB) belongs to examples/*.py, not to the
# CLI. Benchmarks that need it should use their own image.
# ═══════════════════════════════════════════════════════════════════════════════

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /aether

COPY --from=builder /build/target/release/aether /usr/local/bin/aether
COPY --from=builder /build/examples ./examples

# ENTRYPOINT, not CMD: `docker run aether --help` must pass --help to the binary
# rather than replace the command with it.
ENTRYPOINT ["aether"]
CMD ["--help"]
