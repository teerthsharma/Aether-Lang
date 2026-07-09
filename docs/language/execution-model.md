# Execution Model

## REPL

```powershell
cargo run -p aether-cli -- repl
```

The REPL keeps one interpreter instance alive across lines. Each non-empty line
is parsed as a program fragment and executed against the existing variable map.

## Script Runner

```powershell
cargo run -p aether-cli -- run examples/simple.aegis
```

The runner reads the full source file, parses it, and executes it through the
interpreter by default.

## Syntax Checker

```powershell
cargo run -p aether-cli -- check examples/simple.aegis
```

The checker only parses. It does not prove runtime behavior or type safety.

## Titan Mode

```powershell
cargo run -p aether-cli -- run examples/simple.aegis --mode titan
```

Titan mode compiles the AST to VM opcodes and runs the stack VM. Treat Titan
parity as an explicit test requirement. A construct documented as active in the
interpreter is not automatically active in Titan mode.
