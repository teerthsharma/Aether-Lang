# Execution Model

A parsed program runs on one of two engines.

- **The interpreter** (`crates/aether-lang/src/interpreter.rs`) walks the AST.
  It is the reference: where the engines disagree, the interpreter's behaviour
  is the language's.
- **Titan** (`crates/aether-lang/src/vm.rs`) compiles the AST to bytecode and
  runs it on a stack VM. It refuses what it cannot compile. The refusal names
  the construct and its line and column, and the run stops there. Titan never
  hands a program to the interpreter, so a Titan run is either a Titan result
  or a named refusal.

## Running a program

```powershell
cargo run -p aether-cli -- run examples/simple.aegis                # interpreter
cargo run -p aether-cli -- run examples/simple.aegis --mode titan   # Titan
```

`--mode` takes `bio`, the default, which selects the interpreter, or `titan`.
Any other value is an argument error. The `aegis` binary (`aegis-cli`) takes
the same flag and shares the same output path.

Both engines print through one path:

1. a banner whose `Mode:` line names the engine, `interpreter` or `titan`;
2. whatever the program prints;
3. the program's final value with `Display`, unless it is `()`;
4. a blank line and `Execution complete. 🦭`.

A failed `aether` run prints one line to stderr and exits 1:

| stderr line | Cause |
| --- | --- |
| `Parse error at line L, column C: message` | the parser, before either engine starts |
| `Runtime error: message` | either engine, while running |
| `titan: titan cannot compile CONSTRUCT at line L, column C` | Titan refused the program at compile time |

## REPL

```powershell
cargo run -p aether-cli -- repl
```

The REPL keeps one interpreter alive across lines. Each non-empty line is
parsed as a program fragment and executed against the existing variables. The
REPL has no Titan mode.

## Syntax checker

```powershell
cargo run -p aether-cli -- check examples/simple.aegis
```

The checker only parses. It does not prove runtime behavior or type safety.

## Language rules

Both engines implement these rules. Everything not listed follows the
interpreter.

**Last-statement value.** A program's value is the value of its last
statement, and the same holds for a block and for a function body without
`return`. `let x = 3~` as the last line gives the program the value `3`. A
program ending in `print(...)` has the value `()`, so the runner prints
nothing after the program's own output.

**Lexical scope, read-only globals.** A function sees the globals, its own
parameters and its own locals. Every write it makes, a write to a global's
name included, stays local and is discarded when it returns. A function does
not see its caller's locals, and no call copies the environment.

**Late binding.** A function is bound when its `fn` statement executes. A call
resolves when it runs, so a function may call one defined further down the
file, provided that definition has executed by the time of the call.

**Unbound calls are errors.** Calling a name that is not bound stops the run
with `undefined function 'NAME'`. Reading an unbound variable still gives `()`.

**Seal loop forms are decided at parse time.** `until stable(e)` and
`until convergence(eps)` are recognised by the parser, whatever functions the
program defines. Any other `until` is a boolean condition. A user function
named `stable` no longer changes what a loop means. Every seal loop stops after
at most 1,000 passes.

**Reference behaviours kept.** `for` truncates its bounds to integers, steps by
+1 or -1, and leaves the iterator bound after the loop. `x / 0` is `inf`. A
numeric `if` or `while` condition is refused. `render` does nothing.

The scoping rule, unbound calls and the `stable` form are changes to the
language, recorded with before and after outputs in
[PAPER §4.8](https://github.com/teerthsharma/Aether-Lang/blob/master/PAPER.md#48-two-execution-engines).

## Parity

`crates/aether-cli/tests/engine_goldens/` holds the reference outputs. Each
program in the corpus has a golden recording the interpreter CLI's stdout,
its stderr error line and its exit code. The corpus is every program in
`examples/`, plus one small program per construct, written from
`interpreter.rs`. `cargo test -p aether-cli` checks the interpreter against
every golden, then runs the same corpus through Titan.

Each program's Titan run lands in one of three classes:

- **matched**: stdout, error line and exit code equal the golden;
- **refused**: Titan returned a compile error;
- **diverged**: Titan ran, and some part of the output differs.

A refusal never counts as parity. The refused count and the diverged count are
separate ratchets, and each may only shrink. Titan's status row moves from
Partial to Active only when both are zero on the full corpus.

## Performance

Titan is kept only if it runs at least 3x faster, by median, than the
interpreter with its per-call environment clone removed, on both `fib(25)` and
a 10^6-iteration numeric loop. If it does not, `vm.rs` and `--mode titan` are
deleted and the language has one engine.

Measured on an Intel Core i7-14700HX, release builds, median of 7 whole-process
runs (6.5 ms of each is startup):

| Engine | `fib(25)` ms | 10^6 loop ms |
|---|---:|---:|
| interpreter before the reshape | 443.8 | 201.9 |
| interpreter, one frame per call | 237.4 | 205.9 |
| TitanVM, rewritten | **30.7** | 195.7 |

Titan is 7.74x faster on calls and 1.05x on the loop: the gate passes on
`fib(25)` and fails on the loop. The loop's cost is per-statement bookkeeping
(a bound check per assignment, a `Dup` per store, the last-statement register).
If one pass over it does not reach 3x, Titan is deleted.

Parity over 69 programs: 66 matched, 3 refused (`class`/`new`, `regress`,
`block`), 0 diverged.
