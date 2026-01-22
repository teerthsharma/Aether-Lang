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

# Copy manifests first for layer caching
COPY Cargo.toml Cargo.lock* ./
COPY rust-toolchain.toml ./

# Create dummy src for dependency caching
RUN mkdir -p src && \
    echo '#![no_std]\n#![no_main]\nfn main() {}' > src/lib.rs

# Build dependencies only (cache layer)
RUN cargo build --release 2>/dev/null || true

# Copy actual source
COPY src ./src
COPY examples ./examples
COPY docs ./docs

# Build the full project
RUN cargo build --release --lib 2>/dev/null || echo "Bare-metal build skipped in Docker"

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

# Copy source for interpretation (not bare-metal binary)
COPY --from=builder /aegis/src ./src
COPY --from=builder /aegis/examples ./examples
COPY --from=builder /aegis/docs ./docs
COPY --from=builder /aegis/Cargo.toml ./

# Create AEGIS CLI directory
RUN mkdir -p /usr/local/bin

# Copy CLI script
COPY docker/aegis-cli.sh /usr/local/bin/aegis
RUN chmod +x /usr/local/bin/aegis

# Set environment
ENV AEGIS_HOME=/aegis
ENV PATH="/usr/local/bin:${PATH}"

# Default command
ENTRYPOINT ["aegis"]
CMD ["--help"]
