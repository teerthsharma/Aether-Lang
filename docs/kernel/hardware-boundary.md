# Hardware Boundary

Aether contains no_std kernel scaffolding, but the public documentation should
distinguish between code that exists and a supported hardware product.

## Active Source Areas

- allocator;
- boot metadata and hardware topology structures;
- interrupts;
- ELF loader and binary-shape checks;
- sparse scheduler;
- serial output.

## Active Claim

The repository contains Rust code and unit tests for kernel-adjacent concepts.

## Gated Claim

Do not claim:

- general boot support on arbitrary machines;
- production binary authentication;
- measured power reduction;
- real-time scheduling guarantees;
- verified hardware isolation.

Each requires hardware configuration, reproducible boot instructions, logs,
test artifacts, and failure-mode documentation.
