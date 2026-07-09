# FAQ

## Is Aether a replacement for Python ML frameworks?

No. The current repository is a Rust language/runtime and systems experiment
with internal ML primitives. Framework replacement, speed, and model-quality
claims require external baselines and artifacts.

## What is active today?

See [Status Matrix](reference/status.md).

## Where are the derivations?

See [Derivations](topology/derivations.md).

## How do I run a script?

```powershell
cargo run -p aether-cli -- run examples/simple.aegis
```

## What does `topology.ph` do?

It calls the bounded persistent-homology engine in `aether-core`. See
[Persistent Homology](topology/persistent-homology.md).

## Are security claims active?

No production security claim is active from documentation alone. The binary
shape code is documented as a heuristic gate. See [Shape Gates](topology/shape-gates.md).
