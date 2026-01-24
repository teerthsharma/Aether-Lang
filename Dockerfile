# ═══════════════════════════════════════════════════════════════════════════════
# AEGIS: 3D ML Language Kernel
# Docker Build Configuration
# ═══════════════════════════════════════════════════════════════════════════════
#
# Build: docker build -t aegis .
# Run:   docker run -it aegis
# REPL:  docker run -it aegis repl
# ═══════════════════════════════════════════════════════════════════════════════

FROM rust:1.75-bookworm AS builder

# Install nightly toolchain
RUN rustup install nightly-2024-01-15 && \
    rustup default nightly-2024-01-15 && \
    rustup component add rust-src llvm-tools-preview --toolchain nightly-2024-01-15

WORKDIR /aegis

# Copy workspace manifests
COPY Cargo.toml Cargo.lock* ./
COPY rust-toolchain.toml ./

# Copy crate manifests
COPY aegis-core/Cargo.toml ./aegis-core/
COPY aegis-lang/Cargo.toml ./aegis-lang/
COPY aegis-cli/Cargo.toml ./aegis-cli/
COPY aegis-kernel/Cargo.toml ./aegis-kernel/

# Create dummy sources for caching
RUN mkdir -p aegis-core/src && echo "fn main() {}" > aegis-core/src/lib.rs
RUN mkdir -p aegis-lang/src && echo "fn main() {}" > aegis-lang/src/lib.rs
RUN mkdir -p aegis-cli/src && echo "fn main() {}" > aegis-cli/src/main.rs
RUN mkdir -p aegis-kernel/src && echo "fn main() {}" > aegis-kernel/src/main.rs

# Build dependencies
RUN cargo build --release || true

# Copy actual source (overwrite dummies)
COPY aegis-core/src ./aegis-core/src
COPY aegis-lang/src ./aegis-lang/src
COPY aegis-cli/src ./aegis-cli/src
COPY aegis-kernel/src ./aegis-kernel/src
COPY examples ./examples

# Build real binaries
RUN cargo build --release --bin aegis

# ═══════════════════════════════════════════════════════════════════════════════
# Runtime Stage - AEGIS CLI
# ═══════════════════════════════════════════════════════════════════════════════

FROM rust:1.75-slim-bookworm AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /aegis

# Copy binary from builder
COPY --from=builder /aegis/target/release/aegis /usr/local/bin/aegis
COPY --from=builder /aegis/examples ./examples

# Determine entrypoint
CMD ["aegis"]
