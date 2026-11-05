# Titan VM reshape: plan after differential review

## Decision gate first

The first theory scheduled the benchmark after about 1,500 changed lines of `vm.rs`. The room broke that ordering, so the work now opens with a measurement that can cancel it.

The interpreter clones the caller's entire variable `BTreeMap` on every user call (`interpreter.rs:1428`). That cost scales with the number of bound names, and a bare `import` binds every export of a module. A large share of any Titan speedup may therefore be available to the interpreter alone, through a smaller change.

**Gate.** Titan survives only if it is at least **3× faster (median)** than the interpreter with the per-call clone removed, on both fib(25) and a 10⁶-iteration numeric loop. Otherwise `vm.rs` is deleted, `--mode titan` is removed from both CLIs, and the saving lands in the interpreter. One engine is then documented instead of two drifting.

## The plan

### Phase 0: pin the reference and measure (no engine changes)

1. **Goldens.** `crates/aether-cli/tests/engine_goldens/` records the interpreter CLI's full stdout, stderr error line and exit code for:
   - every `examples/*.aegis` and `*.ag`;
   - a construct corpus written from `interpreter.rs`, not from the VM's side.

   The corpus covers these cases, one small program each:
   - descending, fractional and empty `for` ranges (`interpreter.rs:1176-1199`: i64 truncation, ±1 step, iterator left bound);
   - `x / 0` → `inf` (`:1356`);
   - a numeric `if` condition, which the interpreter refuses (`:1143`);
   - `render`, a no-op (`:1122`);
   - `let` as the last statement, whose value is the program's value (`:655-665`, `:1089`);
   - loops returning their last body value;
   - a function reading its caller's locals (dynamic scope);
   - forward calls, and calls to unbound names (currently `()`, `:1391-1398`);
   - a user `fn stable` declared before and after a `stable` seal loop;
   - `print(a, f())` where `f` prints;
   - named arguments.

   A test asserts that the interpreter reproduces every golden. It pins the reference before anything moves.
2. **Benchmark with the gate.** fib(25), a 10⁶-iteration loop, and each of those again with `import monodromy~` above it (scope-size dependence), run under three engines:
   - the interpreter as it is;
   - the interpreter with the clone removed (prototype of Phase 1's scoping rule);
   - Titan, on these programs it already runs correctly.

   Report median of repeated runs, machine, and commit. **Apply the gate.**

### Phase 1: one language, two engines (semantic decisions, both engines, one commit)

- **Scoping rule, written down.** A function sees globals (read-only) plus its parameters and locals, and its writes stay local. For top-level calls this is identical to today's snapshot semantics. It differs only where a function reads its *caller's* locals, and the Phase 0 golden for that case changes deliberately. The interpreter drops the per-call clone.
- **Unbound calls fail in both engines.** Calling an unbound name is a runtime error, not `()`.
- **Loop forms are fixed by the parser.** `until stable(...)` and `until convergence(...)` become `LoopCond::{Stable, Convergence}` at parse time, so a user function named `stable` no longer changes a loop's meaning at run time.
- **One CLI output path.** Both CLIs (`aether-cli`, `aegis-cli`) run programs through one `run(engine)` path that owns final-value printing (Display, `()` hidden) and the footer. Today Titan prints `{:?}` and its own footer (`aether-cli/src/main.rs:212-214`).
- **Parity test.** `crates/aether-cli/tests/engine_parity.rs` runs the built binary under both modes on the golden corpus. It compares stdout, the error line and the exit code.
- **Documentation.** Every golden that changes is listed in the commit and in PAPER §4 as a language change, with the old and new outputs.

### Phase 2: Titan fails closed, on a sound core

- **Fail closed.** `Compiler::compile` returns `Result<Chunk, CompileError { construct, span }>`, and every `_ => {}` arm becomes an error. `aether run --mode titan` prints the construct, line and column, and exits nonzero. There is no silent fallback: a fallback would make the parity and benchmark numbers measure the interpreter while reporting Titan.
- **Print and strings land in this phase**, so the command documented in `docs/language/execution-model.md:32` (`simple.aegis --mode titan`) keeps working. `print` is a per-argument `PRINT` with Display, which preserves the interpreter's evaluate-then-print order.
- **Value discipline.** Each statement writes a `LAST` register, expression statements pop, and a program or function body yields `LAST`. This is the interpreter's last-statement rule, not "the last expression".
- **Reference semantics, copied.**
  - `for` follows the interpreter exactly.
  - `x / 0` gives `inf`.
  - A numeric condition is refused.
  - `render` is a no-op.
- **Late binding.** Functions live in global slots, and `CALL_VALUE` checks that the slot is bound. Forward references then behave as they do in the interpreter: a callee resolves when the call executes.
- **Frames.** Frame size is computed per function at compile time; the Phase 1 scoping rule is what makes that valid. This removes the 256-slot clone per `CALL`.
- **Outcome report.** The parity report counts compiled, matched, refused and diverged programs. *Refused* and *diverged* are two separate ratchets that may only shrink. A refusal is never counted as parity.

### Phase 3: a value-level native convention (both engines)

- **One convention.** `natives::call(name, positional: Vec<Value>, named: Vec<(String, Value)>, host: &mut dyn Host) -> Result<Value, String>`, with `Host::call(&mut self, f: &Value, args: Vec<Value>)`.
- **The interpreter** evaluates arguments left to right and then calls. `print` stays a special form, and the goldens must stay byte-identical.
- **Titan** gets:
  - `CALL_NATIVE(id, argc, names_const)`, so named arguments such as `radius=` and `dt=` survive;
  - function values (`Value::Compiled(idx)`);
  - a re-entrant `run_until_return(depth)`, so a native such as `collision_certificate` calls back into *Titan's* functions. `examples/jacobian_free.aegis` then exercises Titan, not an interpreter embedded inside it.
- **Data opcodes.** Lists and records get `BUILD_LIST`, `INDEX`, `GET_FIELD` and `CALL_METHOD`.
- **Independent checks.** The existing closed-form suites (`integrated_modules.rs` 23, `seal_convergence.rs` 8, `tour.rs` 1) assert mathematical truths, not engine agreement. A bug the two engines share still fails them.

### Phase 4: seal loops in Titan

`SEAL_STABLE` and `SEAL_CONVERGE` opcodes keep the previous value in a hidden slot. They call the `same` and `max_change` helpers, which move to a module both engines import, so each stopping rule has one implementation. The 1,000-pass cap is an explicit counter slot.

### Phase 5: performance (only if the gate passed)

This phase is measured against the Phase 1 interpreter, not the old one. It tries typed f64 arithmetic opcodes, superinstructions for `LOAD; PUSH; ADD; STORE`, and boxed-slice dispatch, each kept only if it moves the median. The "100x" comment in `vm.rs` is deleted in Phase 2 regardless.

### Phase 6: documentation

- **PAPER §4.8** gets the construct-coverage table, the parity counts, and the benchmark table with its control (the Phase 1 interpreter).
- **The docs site** gets an execution-model page for Titan.
- **Status row.** It moves Partial → Active only when *refused = 0 and diverged = 0* on the full corpus.
- **If the gate deleted Titan instead**, the README diagram loses its Titan box and PAPER §4.8 records why.

### Constraints that hold throughout

- **`no_std`.** Titan and the natives stay `no_std`: `aether-kernel` depends on `aether-lang` with `features = ["no_std"]` and re-exports it (`aether-kernel/src/lib.rs:36`). The CI kernel build is the guard.
- **One PR per phase.** Phase 1's language changes are one revertible commit, and the goldens show exactly what it changed.

## Differential: every question, answered

| # | Fellow | Question (short) | Verdict | Where it landed |
|---|---|---|---|---|
| F1 | Foreman | No native table exists; natives take AST args plus the interpreter; Titan has no function values or re-entrant `run` | conceded | Phase 3 value-level convention, `Host`, function values, re-entrant run |
| F2 | Foreman | Copying the interpreter's scoping contradicts compile-time slots; `stable` is a runtime check | conceded | Phase 1 scoping rule and parser-level loop forms |
| F3 | Foreman | Interpreter value is last *statement*; descending ranges count down | conceded | Phase 2 `LAST` register; `for` copied from `:1176-1199` |
| F4 | Foreman | Interpreter timing scales with bound names | conceded | Phase 0 benchmark includes an `import monodromy~` variant; Phase 1 removes the clone |
| F5 | Foreman | Errors not compared; refusing everything empties the allow-list | conceded | parity compares error line and exit code; refused and diverged are separate ratchets; Active needs both zero |
| C1 | Chase | Shared-natives refactor rewrites the oracle; what pins it, and what if both share a bug | conceded / answered | Phase 0 goldens pin the interpreter first; closed-form suites are engine-independent |
| C2 | Chase | Copy the environment per call, or move the oracle | conceded | the oracle moves once, deliberately, in Phase 1, with goldens showing the diff |
| C3 | Chase | Phase 1 breaks the documented `--mode titan` command and `aegis-cli`; rollback | conceded / rejected | print and strings move into Phase 2 and both CLIs are updated there. Rejected: rolling back to silent wrong answers is never the fallback, because refusal is strictly better |
| C4 | Chase | `<` ranges create a divergence; `render` fixed from the wrong side | conceded | reference semantics copied from `interpreter.rs`; corpus written from it |
| C5 | Chase | Benchmark scheduled after the work it could cancel | conceded | Phase 0 gate |
| K1 | Cameron | The oracle breaks the rule Titan is held to | conceded | Phase 1 holds both engines to fail-closed and one written scoping rule |
| K2 | Cameron | Refuse vs run is a forced choice; CLI output differs outside the sink | conceded / rejected | a single CLI path, and the parity test runs the binary. Rejected: auto-fallback, which would falsify which engine ran |
| K3 | Cameron | `CALL_NATIVE` embeds the interpreter; callbacks run on the tree-walker | conceded | Phase 3 re-entrant Titan callbacks |
| K4 | Cameron | Is a second engine the only way to be fast | conceded | Phase 0 gate compares against the fixed interpreter |
| K5 | Cameron | `stable` shadowing differs by engine | conceded | parser-level `LoopCond::Stable` in both engines |

## Still open

- **The 3× gate is a judgement, not a derivation.** Below it, a second engine doubles the cost of every language feature for too little gain. It is the default until a measurement argues otherwise.
- **Lexical scope changes any program that reads a caller's locals.** None of the shipped examples is known to do so; the Phase 0 goldens will confirm or refute that.
- **A swallowed argument error stays.** `get_f64(args).unwrap_or(0.01)` in the `MLP` constructor (`interpreter.rs:1487`) swallows bad arguments. The goldens pin that behaviour, and fixing it is a separate change.
