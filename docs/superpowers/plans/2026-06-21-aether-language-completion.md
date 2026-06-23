# Aether Language Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the Aether/Aegis DSL from a partial parser/runtime toward a complete, testable language implementation.

**Architecture:** Build from the compiler front end outward: lexer correctness, parser grammar, AST/runtime semantics, diagnostics, then compiler/VM lowering. Each slice must add failing tests first and keep `cargo test -p aether-lang` green before moving to the next slice.

**Tech Stack:** Rust workspace, `aether-lang` lexer/parser/AST/interpreter/VM, existing `aether-core` runtime primitives.

## Global Constraints

- Preserve the existing crate layout and public exports.
- Keep changes focused in `crates/aether-lang` unless a task explicitly needs `aether-core`.
- Use TDD: every new language behavior gets a failing test before implementation.
- Maintain `no_std` compatibility where current modules already support it.
- Do not remove existing AEGIS/Aether syntax aliases unless a compatibility task explicitly does so.

---

### Task 1: Implement Documented Expression Surface

**Files:**
- Modify: `crates/aether-lang/src/lexer.rs`
- Modify: `crates/aether-lang/src/parser.rs`
- Modify: `crates/aether-lang/src/ast.rs`
- Modify: `crates/aether-lang/src/interpreter.rs`

**Interfaces:**
- Consumes: `Lexer::tokenize`, `Parser::parse`, `Interpreter::execute`
- Produces: parsed and executable `~`, `..`, `%`, comparisons, logical operators, and unary `!`/`-`

- [x] **Step 1: Write failing tests for doc-backed syntax**

Add tests proving `1..10` tokenizes as `Number`, `DotDot`, `Number`; `~` separates statements; `1 < 2 && !false` parses; and `10 % 4 == 2` executes.

- [x] **Step 2: Run test to verify it fails**

Run: `cargo test -p aether-lang`
Expected: FAIL in lexer/parser/interpreter tests for the missing operators and separators.

- [x] **Step 3: Implement minimal lexer/parser/interpreter support**

Keep `1.5` float handling intact while preserving `1..10` as a range; add precedence layers for `||`, `&&`, equality, comparison, range, additive, multiplicative, unary, and primary expressions.

- [x] **Step 4: Run test to verify it passes**

Run: `cargo test -p aether-lang`
Expected: PASS.

### Task 2: Add Assignment and Mutable Runtime State

**Files:**
- Modify: `crates/aether-lang/src/ast.rs`
- Modify: `crates/aether-lang/src/parser.rs`
- Modify: `crates/aether-lang/src/interpreter.rs`

**Interfaces:**
- Consumes: `StmtKind::Var`, `ExprKind::Ident`
- Produces: `StmtKind::Assign { name, value }` or equivalent, with runtime variable update behavior

- [x] **Step 1: Write failing parser/runtime tests**

```rust
let mut parser = Parser::new("let count = 0~\ncount = count + 1~");
let program = parser.parse().expect("program should parse");
let mut interpreter = Interpreter::new();
interpreter.execute(&program).expect("program should execute");
assert!(matches!(interpreter.variables.get("count"), Some(Value::Num(1.0))));
```

- [x] **Step 2: Implement assignment parsing**

When a statement starts with an identifier followed by `=`, parse it as assignment rather than only declaration.

- [x] **Step 3: Implement assignment execution**

Evaluate the right-hand expression and update `Interpreter::variables`.

- [x] **Step 4: Verify**

Run: `cargo test -p aether-lang`
Expected: PASS.

### Task 3: Make Loops Semantically Useful

**Files:**
- Modify: `crates/aether-lang/src/parser.rs`
- Modify: `crates/aether-lang/src/interpreter.rs`

**Interfaces:**
- Consumes: assignment from Task 2, existing `IfStmt`, `WhileStmt`, `ForStmt`, `LoopStmt`
- Produces: executable `while`, `for i in start..end`, and bounded `seal until condition { ... }`

- [x] **Step 1: Write failing runtime tests**

Cover `while count < 3`, `for i in 0..3`, `break`, and `continue`.

- [x] **Step 2: Add loop control flow result type**

Introduce an internal execution-flow enum such as `RuntimeFlow::Value(Value)`, `Break`, `Continue`, `Return(Value)` so loop control does not collapse to `Unit`.

- [x] **Step 3: Execute `for` ranges**

Bind the iterator name for each integer step from range start to range end, execute the body, and restore/overwrite the variable predictably.

- [x] **Step 4: Add `seal until` parsing and execution**

Support `seal until expr { ... }` and `🦭 until expr { ... }`, preserving the current bare `seal { ... }` fallback.

- [x] **Step 5: Verify**

Run: `cargo test -p aether-lang`
Expected: PASS.

### Task 4: Function Calls and Returns

**Files:**
- Modify: `crates/aether-lang/src/interpreter.rs`

**Interfaces:**
- Consumes: existing `FnDecl`, `ReturnStmt`, `ExprKind::Call`
- Produces: user-defined function storage, local call frames, parameter binding, return propagation

- [x] **Step 1: Write failing tests**

Test `fn add(a, b) { return a + b~ } let result = add(2, 3)~`.

- [x] **Step 2: Store function declarations**

Add a function table to `Interpreter` or store functions in `variables` through a new `Value::Function`.

- [x] **Step 3: Implement call frames**

Bind arguments to parameters in a temporary scope and restore previous bindings after execution.

- [x] **Step 4: Verify**

Run: `cargo test -p aether-lang`
Expected: PASS.

### Task 5: Diagnostics and Check Mode

**Files:**
- Modify: `crates/aether-lang/src/lexer.rs`
- Modify: `crates/aether-lang/src/parser.rs`
- Modify: `crates/aether-cli/src/main.rs`

**Interfaces:**
- Consumes: `Token::line`, `Token::column`, `Span`
- Produces: stable error messages with source location and expected token/context

- [x] **Step 1: Write failing tests**

Assert parser errors include the unexpected token, expected grammar construct, line, and column.

- [x] **Step 2: Preserve lexer errors as parser failures**

Do not let `TokenKind::Error` reach generic unexpected-token paths.

- [x] **Step 3: Surface diagnostics in CLI check/run**

Format errors consistently for `aether check` and `aether run`.

- [x] **Step 4: Verify**

Run: `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 10: Lean Block Flow Semantics

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 9 single-statement relation and statement-flow model
- Produces: checked Lean relation for ordered block execution and control-flow propagation

- [x] **Step 1: Add block execution relation**

Define `StepBlock` for empty blocks, single statements, value-sequencing, and early stop on `return`, `break`, or `continue`.

- [x] **Step 2: Add checked block examples**

Cover `let` followed by expression, assignment followed by expression, early `return`, early `break`, and early `continue`.

- [x] **Step 3: Document block-flow semantics**

Record that final expression values are preserved and control-flow signals stop a block immediately.

- [x] **Step 4: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 161: Proof-Core String `first()` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, argument-aware static method
  compatibility, stack/frame method opcodes, checked source frame compilation,
  and source diagnostic rendering.
- Produces: zero-argument `.first()` support for proof-core strings, returning
  the first character as `str` at runtime when present and rejecting
  non-zero-arity concrete calls statically.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `"open".first()` through `evalExpr`, `checkExpr`,
closed expression bytecode execution, frame expression compilation,
`checkedFrameSourceLocal?`, and `sourceLocal?`. Include an empty-string
runtime failure and a one-argument static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because string method evaluation and static method
compatibility do not yet support `.first()`.

- [x] **Step 3: Implement shared runtime support**

Wire string `.first()` through `evalIndex` so first-character extraction shares
the same runtime behavior as string indexing.

- [x] **Step 4: Implement static method typing**

Extend static method compatibility so zero-argument `str.first()` returns
`str`.

- [x] **Step 5: Document the proof-core method**

Record string `.first()` in the proof-core method surface and checked compiler
coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 162: Proof-Core String `last()` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, argument-aware static method
  compatibility, stack/frame method opcodes, checked source frame compilation,
  and source diagnostic rendering.
- Produces: zero-argument `.last()` support for proof-core strings, returning
  the final character as `str` at runtime when present and rejecting
  non-zero-arity concrete calls statically.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `"open".last()` through `evalExpr`, `checkExpr`,
closed expression bytecode execution, frame expression compilation,
`checkedFrameSourceLocal?`, and `sourceLocal?`. Include an empty-string
runtime failure and a one-argument static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because string method evaluation and static method
compatibility do not yet support `.last()`.

- [x] **Step 3: Implement shared runtime support**

Wire string `.last()` through character-list evaluation so it returns the last
character under the same character semantics as string indexing.

- [x] **Step 4: Implement static method typing**

Extend static method compatibility so zero-argument `str.last()` returns
`str`.

- [x] **Step 5: Document the proof-core method**

Record string `.last()` in the proof-core method surface and checked compiler
coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 163: Proof-Core String `tail()` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, argument-aware static method
  compatibility, stack/frame method opcodes, checked source frame compilation,
  and source diagnostic rendering.
- Produces: zero-argument `.tail()` support for proof-core strings, returning
  the remaining string after the first character at runtime when present and
  rejecting non-zero-arity concrete calls statically.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `"open".tail()` through `evalExpr`, `checkExpr`,
closed expression bytecode execution, frame expression compilation,
`checkedFrameSourceLocal?`, and `sourceLocal?`. Include an empty-string
runtime failure and a one-argument static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because string method evaluation and static method
compatibility do not yet support `.tail()`.

- [x] **Step 3: Implement shared runtime support**

Wire string `.tail()` through character-list evaluation so it returns the
remaining characters under the same character semantics as string indexing.

- [x] **Step 4: Implement static method typing**

Extend static method compatibility so zero-argument `str.tail()` returns
`str`.

- [x] **Step 5: Document the proof-core method**

Record string `.tail()` in the proof-core method surface and checked compiler
coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 164: Proof-Core String `take(count)` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, numeric argument static compatibility,
  stack/frame method opcodes, checked source frame compilation, and source
  diagnostic rendering.
- Produces: `.take(count)` support for proof-core strings, returning the
  prefix string for non-negative numeric counts and rejecting non-numeric
  concrete count arguments statically.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `"open".take(2)` through `evalExpr`, `checkExpr`,
closed expression bytecode execution, frame expression compilation,
`checkedFrameSourceLocal?`, and `sourceLocal?`. Include negative-count
runtime failure and a boolean-count static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because string method evaluation and static method
compatibility do not yet support `.take(count)`.

- [x] **Step 3: Implement shared runtime support**

Add character-list prefix evaluation and wire string `.take(count)` through it,
rejecting negative runtime counts like list `.take(count)`.

- [x] **Step 4: Implement static method typing**

Extend static method compatibility so `str.take(count)` returns `str` only
when the count argument checks as numeric.

- [x] **Step 5: Document the proof-core method**

Record string `.take(count)` in the proof-core method surface and checked
compiler coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 165: Proof-Core String `drop(count)` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, numeric argument static compatibility,
  stack/frame method opcodes, checked source frame compilation, and source
  diagnostic rendering.
- Produces: `.drop(count)` support for proof-core strings, returning the
  suffix string after dropping a non-negative numeric count and rejecting
  non-numeric concrete count arguments statically.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `"open".drop(2)` through `evalExpr`, `checkExpr`,
closed expression bytecode execution, frame expression compilation,
`checkedFrameSourceLocal?`, and `sourceLocal?`. Include negative-count
runtime failure and a boolean-count static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because string method evaluation and static method
compatibility do not yet support `.drop(count)`.

- [x] **Step 3: Implement shared runtime support**

Add character-list suffix evaluation and wire string `.drop(count)` through it,
rejecting negative runtime counts like list `.drop(count)`.

- [x] **Step 4: Implement static method typing**

Extend static method compatibility so `str.drop(count)` returns `str` only
when the count argument checks as numeric.

- [x] **Step 5: Document the proof-core method**

Record string `.drop(count)` in the proof-core method surface and checked
compiler coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 166: Proof-Core String `starts_with(prefix)` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, string argument static compatibility,
  stack/frame method opcodes, checked source frame compilation, and source
  diagnostic rendering.
- Produces: `.starts_with(prefix)` support for proof-core strings, returning
  `bool` for prefix membership and rejecting non-string concrete prefix
  arguments statically.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `"open".starts_with("op")` through `evalExpr`,
`checkExpr`, closed expression bytecode execution, frame expression
compilation, `checkedFrameSourceLocal?`, and `sourceLocal?`. Include a false
runtime prefix case and a non-string argument static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because string method evaluation and static method
compatibility do not yet support `.starts_with(prefix)`.

- [x] **Step 3: Implement shared runtime support**

Wire string `.starts_with(prefix)` through the character-list prefix helper.

- [x] **Step 4: Implement static method typing**

Extend static method compatibility so `str.starts_with(prefix)` returns
`bool` only when the prefix argument checks as `str`.

- [x] **Step 5: Document the proof-core method**

Record string `.starts_with(prefix)` in the proof-core method surface and
checked compiler coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 167: Proof-Core String `ends_with(suffix)` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, string argument static compatibility,
  stack/frame method opcodes, checked source frame compilation, and source
  diagnostic rendering.
- Produces: `.ends_with(suffix)` support for proof-core strings, returning
  `bool` for suffix membership and rejecting non-string concrete suffix
  arguments statically.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `"open".ends_with("en")` through `evalExpr`,
`checkExpr`, closed expression bytecode execution, frame expression
compilation, `checkedFrameSourceLocal?`, and `sourceLocal?`. Include a false
runtime suffix case and a non-string argument static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because string method evaluation and static method
compatibility do not yet support `.ends_with(suffix)`.

- [x] **Step 3: Implement shared runtime support**

Wire string `.ends_with(suffix)` through reverse character-list prefix
matching.

- [x] **Step 4: Implement static method typing**

Extend static method compatibility so `str.ends_with(suffix)` returns `bool`
only when the suffix argument checks as `str`.

- [x] **Step 5: Document the proof-core method**

Record string `.ends_with(suffix)` in the proof-core method surface and
checked compiler coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 168: Proof-Core String `reverse()` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, stack/frame method opcodes, checked
  source frame compilation, and source diagnostic rendering.
- Produces: zero-argument `.reverse()` support for proof-core strings,
  returning a character-reversed `str` and rejecting non-zero-arity concrete
  calls statically.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `"open".reverse()` through `evalExpr`, `checkExpr`,
closed expression bytecode execution, frame expression compilation,
`checkedFrameSourceLocal?`, and `sourceLocal?`. Include a one-argument static
diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because string method evaluation and static method
compatibility do not yet support `.reverse()`.

- [x] **Step 3: Implement shared runtime support**

Wire string `.reverse()` through character-list reversal.

- [x] **Step 4: Implement static method typing**

Extend static method compatibility so zero-argument `str.reverse()` returns
`str`.

- [x] **Step 5: Document the proof-core method**

Record string `.reverse()` in the proof-core method surface and checked
compiler coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 169: Proof-Core List `reverse()` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, static method compatibility, stack/frame
  method opcodes, checked source frame compilation, and source diagnostic
  rendering.
- Produces: zero-argument `.reverse()` support for proof-core lists, returning
  a reversed `list[T]` and rejecting non-zero-arity concrete calls statically.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `[7, 9].reverse()` through `evalExpr`, `checkExpr`,
closed expression bytecode execution, frame expression compilation,
`checkedFrameSourceLocal?`, and `sourceLocal?`. Include an empty-list runtime
example and a one-argument static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because list method evaluation and static method compatibility
do not yet support `.reverse()`.

- [x] **Step 3: Implement shared runtime support**

Wire list `.reverse()` through structural list reversal.

- [x] **Step 4: Implement static method typing**

Extend static method compatibility so zero-argument `list[T].reverse()`
returns `list[T]`.

- [x] **Step 5: Document the proof-core method**

Record list `.reverse()` in the proof-core method surface and checked compiler
coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 170: Proof-Core List `append(value)` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, static method compatibility, stack/frame
  method opcodes, checked source frame compilation, and source diagnostic
  rendering.
- Produces: one-argument pure `.append(value)` support for proof-core lists,
  returning a new `list[T]` with the value at the end and rejecting
  incompatible concrete element types statically.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `[7].append(9)` through `evalExpr`, `checkExpr`,
closed expression bytecode execution, frame expression compilation,
`checkedFrameSourceLocal?`, and `sourceLocal?`. Include an empty-list runtime
example and an incompatible-element static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because list method evaluation and static method compatibility
do not yet support `.append(value)`.

- [x] **Step 3: Implement shared runtime support**

Wire list `.append(value)` through structural list append without mutating the
receiver.

- [x] **Step 4: Implement static method typing**

Extend static method compatibility so `list[T].append(value)` returns
`list[T]` when `value` is compatible with `T`.

- [x] **Step 5: Document the proof-core method**

Record list `.append(value)` in the proof-core method surface and checked
compiler coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 171: Proof-Core List `concat(other)` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, static method compatibility, stack/frame
  method opcodes, checked source frame compilation, and source diagnostic
  rendering.
- Produces: one-argument pure `.concat(other)` support for proof-core lists,
  returning a new list with the receiver values followed by the argument list
  values and rejecting incompatible concrete list element types statically.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `[7].concat([9])` through `evalExpr`, `checkExpr`,
closed expression bytecode execution, frame expression compilation,
`checkedFrameSourceLocal?`, and `sourceLocal?`. Include an empty-receiver
runtime example and an incompatible-list static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because list method evaluation and static method compatibility
do not yet support `.concat(other)`.

- [x] **Step 3: Implement shared runtime support**

Wire list `.concat(other)` through structural list concatenation without
mutating the receiver or argument.

- [x] **Step 4: Implement static method typing**

Extend static method compatibility so `list[T].concat(list[U])` returns
`list[T]` when `T` and `U` are compatible.

- [x] **Step 5: Document the proof-core method**

Record list `.concat(other)` in the proof-core method surface and checked
compiler coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 172: Proof-Core List `prepend(value)` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, static method compatibility, stack/frame
  method opcodes, checked source frame compilation, and source diagnostic
  rendering.
- Produces: one-argument pure `.prepend(value)` support for proof-core lists,
  returning a new `list[T]` with the value at the beginning and rejecting
  incompatible concrete element types statically.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `[9].prepend(7)` through `evalExpr`, `checkExpr`,
closed expression bytecode execution, frame expression compilation,
`checkedFrameSourceLocal?`, and `sourceLocal?`. Include an empty-list runtime
example and an incompatible-element static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because list method evaluation and static method compatibility
do not yet support `.prepend(value)`.

- [x] **Step 3: Implement shared runtime support**

Wire list `.prepend(value)` through structural cons/list construction without
mutating the receiver.

- [x] **Step 4: Implement static method typing**

Extend static method compatibility so `list[T].prepend(value)` returns
`list[T]` when `value` is compatible with `T`.

- [x] **Step 5: Document the proof-core method**

Record list `.prepend(value)` in the proof-core method surface and checked
compiler coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 173: Proof-Core List `join(separator)` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, static method compatibility, stack/frame
  method opcodes, checked source frame compilation, and source diagnostic
  rendering.
- Produces: one-argument pure `.join(separator)` support for proof-core string
  lists, returning a `str` with separator text between elements and rejecting
  incompatible concrete element or separator types statically.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `["a", "b"].join(",")` through `evalExpr`,
`checkExpr`, closed expression bytecode execution, frame expression
compilation, `checkedFrameSourceLocal?`, and `sourceLocal?`. Include an empty
list runtime example, a non-string runtime failure, and a concrete non-string
list static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because list method evaluation and static method compatibility
do not yet support `.join(separator)`.

- [x] **Step 3: Implement shared runtime support**

Wire list `.join(separator)` through recursive string concatenation that fails
if any runtime list element is not a string.

- [x] **Step 4: Implement static method typing**

Extend static method compatibility so `list[str].join(str)` returns `str` and
incompatible concrete list or separator types are rejected.

- [x] **Step 5: Document the proof-core method**

Record list `.join(separator)` in the proof-core method surface and checked
compiler coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 174: Lean Semicolon Statement Separators

**Files:**
- Modify: `Aether/Lexer.lean`
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core tokenization, located tokenization, parser terminator
  skipping, source pipeline parsing, and diagnostic rendering.
- Produces: `;` as a proof-core statement separator equivalent to newline and
  `~`, with stable located spans and source execution through the checked
  compiler pipeline.

- [x] **Step 1: Add failing lexer/parser/source examples**

Add checked examples proving `;` tokenizes as its own token, located
tokenization preserves a half-open semicolon span, `parseProgram` accepts
semicolon-separated statements, and `sourceLocal?` executes a semicolon-
separated source program.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the lexer does not yet emit a semicolon separator and
the parser does not yet treat it as a terminator.

- [x] **Step 3: Implement semicolon tokenization**

Add `TokenKind.semicolon`, scan `;` in plain and located lexers, and render it
in diagnostics.

- [x] **Step 4: Implement semicolon terminator handling**

Treat `TokenKind.semicolon` like newline and `~` in parser terminator skipping
and located pipeline terminator skipping.

- [x] **Step 5: Document the lexical surface**

Record semicolon as a proof-core statement separator in the formal-core docs.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 175: Lean Multiline List Literal Separators

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core list literal parsing, newline tokens, comma tokens, and
  source-pipeline execution.
- Produces: list literals that accept newlines between elements and a trailing
  comma before `]`, matching the existing Rust parser surface while preserving
  the same `Expr.list` core.

- [x] **Step 1: Add failing parser/source examples**

Add checked examples proving `parseProgram` accepts a multiline list literal
with a trailing comma and that `sourceLocal?` executes a source program using
that literal.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `parseListLiteral` currently treats newline after comma
and trailing comma before `]` as parse failures.

- [x] **Step 3: Implement list literal separator handling**

Teach `parseListLiteral` to skip newline tokens within list literals and accept
`]` immediately after a comma as a trailing-comma terminator.

- [x] **Step 4: Document the list literal surface**

Record multiline and trailing-comma list literal syntax in the formal-core
syntax and parser coverage docs.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 176: Lean Multiline Call Argument Separators

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core function-call and method-call argument parsing, newline
  tokens, comma tokens, and source-pipeline execution.
- Produces: call and method argument lists that accept newlines between
  arguments and a trailing comma before `)`, preserving positional and named
  `Arg` nodes.

- [x] **Step 1: Add failing parser/source examples**

Add checked examples proving `parseExpr` accepts a multiline function call with
a trailing comma and that `sourceLocal?` executes a method call using the same
argument-list separator rules.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `parseArgList` currently treats newline after comma and
trailing comma before `)` as parse failures.

- [x] **Step 3: Implement argument-list separator handling**

Teach `parseArgList` to skip newline tokens within argument lists and accept
`)` immediately after a comma as a trailing-comma terminator.

- [x] **Step 4: Document the argument-list surface**

Record multiline and trailing-comma call/method argument syntax in the
formal-core grammar and parser coverage docs.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 177: Lean Multiline Function Parameter Separators

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core untyped and typed function parameter parsing, newline
  tokens, comma tokens, and source-pipeline execution.
- Produces: function declarations whose parameter lists accept newlines between
  parameters and a trailing comma before `)`, preserving the same `Stmt.fnDecl`
  and typed function declaration core nodes.

- [x] **Step 1: Add failing parser/source examples**

Add checked examples proving `parseProgram` accepts multiline untyped and typed
function parameter lists with trailing commas, and that `sourceLocal?` executes
a source program using the untyped form.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `parseParamList` and `parseTypedParamList` currently
treat newline after comma and trailing comma before `)` as parse failures.

- [x] **Step 3: Implement parameter-list separator handling**

Teach untyped and typed parameter-list parsers to skip newline tokens inside
parameter lists and accept `)` immediately after a comma as a trailing-comma
terminator.

- [x] **Step 4: Document the parameter-list surface**

Record multiline and trailing-comma function parameter syntax in the formal-core
grammar and parser coverage docs.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 178: Lean Multiline List Type Annotations

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core annotated local declarations, typed parameters, declared
  return types, and nested `list[...]` type annotation parsing.
- Produces: `list[...]` annotations that accept newlines after `[`, before
  nested element types, and before `]`, preserving the same `AnnTy.list` core.

- [x] **Step 1: Add failing parser/source examples**

Add checked examples proving multiline `list[...]` annotations parse in local
declarations, typed parameters, and declared return types, and that
`sourceLocal?` executes a source program using the local annotation.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `parseAnnTy` currently treats newline tokens inside
`list[...]` annotations as parse failures.

- [x] **Step 3: Implement type-annotation newline handling**

Teach `parseAnnTy` to skip newline tokens at annotation boundaries and inside
`list[...]` before parsing the element type and closing bracket.

- [x] **Step 4: Document multiline list type annotations**

Record newline-tolerant `list[...]` annotations in the formal-core grammar and
parser coverage docs.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 179: Lean Line-Broken Block Openings

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core block parsing, statement terminator skipping, structured
  statements, function declarations, and source-pipeline execution.
- Produces: block-bearing forms that may place a statement separator between
  the header and `{`, preserving the same structured statement core nodes.

- [x] **Step 1: Add failing parser/source examples**

Add checked examples proving `parseProgram` accepts a line-broken `if` block
opening and that `sourceLocal?` executes a function declaration whose `{` starts
after a newline.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `parseBlock` currently requires `{` immediately at the
current token.

- [x] **Step 3: Implement block opening terminator skipping**

Teach `parseBlock` to skip statement terminators before matching `{`, so all
block-bearing forms share the same line-broken opening behavior.

- [x] **Step 4: Document block opening separators**

Record that a separator may appear between a block header and opening brace in
the formal-core syntax and parser coverage docs.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 180: Lean Multiline Type Diagnostic Offsets

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: newline-tolerant `list[...]` type annotation parsing, parser
  diagnostic offsets, and source diagnostic rendering.
- Produces: malformed multiline type annotations whose diagnostics point at the
  offending token after skipped newlines rather than at the outer annotation.

- [x] **Step 1: Add failing parser/source diagnostics**

Add checked examples proving `let xs: list[\n = [1]` reports the `=` token as
the type error, with a rendered source span on line 2.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `annTyFailureOffset?` does not yet skip newline tokens
while walking malformed `list[...]` annotations.

- [x] **Step 3: Implement newline-aware type diagnostic offsets**

Teach the type diagnostic offset walker to skip newline tokens consistently at
annotation boundaries and inside `list[...]`.

- [x] **Step 4: Document multiline type diagnostics**

Record that parser diagnostics for malformed multiline list annotations point
at the offending token after skipped newlines.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 181: Lean Multiline Parenthesized Expressions

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core parenthesized expression parsing, newline tokens, and
  source-pipeline execution.
- Produces: parenthesized expressions that accept newlines after `(` and before
  `)`, preserving the grouped expression as the same core `Expr`.

- [x] **Step 1: Add failing parser/source examples**

Add checked examples proving `parseExpr` accepts a line-broken parenthesized
expression and that `sourceLocal?` executes a source program using that grouped
expression.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `parsePrimary` currently passes newline tokens directly
to `parseExpr` after `(` and requires `)` immediately after the expression.

- [x] **Step 3: Implement parenthesis newline handling**

Teach parenthesized-expression parsing to skip newline tokens after `(` and
before `)`.

- [x] **Step 4: Document multiline parenthesized expressions**

Record newline-tolerant parenthesized expressions in the formal-core grammar
and parser coverage docs.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 182: Lean Multiline Postfix Indexing

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core postfix index parsing, newline tokens, and checked
  source-pipeline execution.
- Produces: postfix index expressions that accept newlines after `[` and before
  `]`, preserving the same `Expr.index` core node.

- [x] **Step 1: Add failing parser/source examples**

Add checked examples proving `parseExpr` accepts a multiline postfix index and
that `sourceLocal?` executes a source program using that index form.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `parsePostfixLoop` currently passes newline tokens
directly to `parseExpr` after `[` and requires `]` immediately after the index.

- [x] **Step 3: Implement index newline handling**

Teach postfix index parsing to skip newline tokens after `[` and before `]`.

- [x] **Step 4: Document multiline index expressions**

Record newline-tolerant postfix indexing in the formal-core grammar and parser
coverage docs.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 183: Lean Multiline Postfix Member Access

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core postfix field/method parsing, newline tokens, and checked
  source-pipeline execution.
- Produces: postfix field and method expressions that accept newlines after `.`,
  preserving the same `Expr.field` and `Expr.method` core nodes.

- [x] **Step 1: Add failing parser/source examples**

Add checked examples proving `parseExpr` accepts multiline postfix field and
method access, and that `sourceLocal?` executes source programs using those
member forms.

- [x] **Step 2: Verify red**

Run `lake build`.

Expected: FAIL before implementation on the new multiline member examples.

- [x] **Step 3: Implement post-dot newline skipping**

Add a parser helper that skips newline tokens after `.`, then use it before
matching the member identifier and optional method-call argument list.

- [x] **Step 4: Document grammar and coverage**

Update the formal grammar and parser coverage notes for newline-tolerant
postfix member access.

- [x] **Step 5: Verify green**

Run:
- `lake build`
- `cargo test -p aether-lang -p aether-cli`

Expected: PASS.

### Task 184: Lean Multiline Function Call Opening

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core function-call parsing, newline tokens, and checked
  source-pipeline execution.
- Produces: function-call expressions that accept newlines between a callee name
  and `(` without consuming ordinary statement-separator newlines when no call
  follows.

- [x] **Step 1: Add failing parser/source examples**

Add checked examples proving `parseExpr` accepts a line-broken call opening and
that `sourceLocal?` executes a user-defined function call using that form.

- [x] **Step 2: Verify red**

Run `lake build`.

Expected: FAIL before implementation on the new multiline call-opening example.

- [x] **Step 3: Implement guarded call-opening newline skipping**

Add a parser helper that skips newline tokens only while checking whether an
identifier or keyword-call name is followed by `(`. Fall back to the original
token tail for non-call identifiers so statement separators remain observable.

- [x] **Step 4: Document grammar and coverage**

Update the formal grammar and parser coverage notes for newline-tolerant
function-call openings.

- [x] **Step 5: Verify green**

Run:
- `lake build`
- `cargo test -p aether-lang -p aether-cli`

Expected: PASS.

### Task 185: Lean Multiline Unary Operands

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core unary parsing, newline tokens, and checked
  source-pipeline execution.
- Produces: unary `-` and `!` expressions that accept newlines before their
  operands while preserving the same `Expr.unary` core nodes.

- [x] **Step 1: Add failing parser/source examples**

Add checked examples proving `parseExpr` accepts line-broken unary operands and
that `sourceLocal?` executes source programs using both boolean and numeric
forms.

- [x] **Step 2: Verify red**

Run `lake build`.

Expected: FAIL before implementation on the new multiline unary examples.

- [x] **Step 3: Implement unary operand newline skipping**

Add a parser helper that skips newline tokens after unary `-` and `!`, then use
it before recursively parsing each unary operand.

- [x] **Step 4: Document grammar and coverage**

Update the formal grammar and parser coverage notes for newline-tolerant unary
expressions.

- [x] **Step 5: Verify green**

Run:
- `lake build`
- `cargo test -p aether-lang -p aether-cli`

Expected: PASS.

### Task 186: Lean Multiline Binary Right-Hand Sides

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core binary operator parsing, newline tokens, and checked
  source-pipeline execution.
- Produces: binary expressions that accept newlines after a recognized operator
  before the right-hand operand while leaving newline statement separators
  observable before operators.

- [x] **Step 1: Add failing parser/source examples**

Add checked examples proving `parseExpr` accepts line-broken arithmetic and
logical binary right-hand sides, and that `sourceLocal?` executes a source
program using that form.

- [x] **Step 2: Verify red**

Run `lake build`.

Expected: FAIL before implementation on the new multiline binary examples.

- [x] **Step 3: Implement binary RHS newline skipping**

Add a parser helper that skips newline tokens after a recognized binary
operator, then call it before parsing the right-hand side in left-associative
operator parsing.

- [x] **Step 4: Document grammar and coverage**

Update the formal grammar and parser coverage notes for newline-tolerant binary
right-hand sides.

- [x] **Step 5: Verify green**

Run:
- `lake build`
- `cargo test -p aether-lang -p aether-cli`

Expected: PASS.

### Task 187: Lean Multiline Assignment Right-Hand Sides

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core let declarations, assignment statements, newline tokens,
  and checked source-pipeline execution.
- Produces: statement-level `=` forms that accept newlines before the
  right-hand expression for typed lets, untyped lets, and reassignments.

- [x] **Step 1: Add failing parser/source examples**

Add checked examples proving `parseProgram` accepts line-broken assignment
right-hand sides for untyped lets, typed lets, and reassignments, and that
`sourceLocal?` executes source programs using those forms.

- [x] **Step 2: Verify red**

Run `lake build`.

Expected: FAIL before implementation on the new multiline assignment examples.

- [x] **Step 3: Implement assignment RHS newline skipping**

Add a parser helper that skips newline tokens after statement-level `=`, then
call it before parsing RHS expressions for typed lets, untyped lets, and
reassignments.

- [x] **Step 4: Document grammar and coverage**

Update the formal grammar and parser coverage notes for newline-tolerant
assignment right-hand sides.

- [x] **Step 5: Verify green**

Run:
- `lake build`
- `cargo test -p aether-lang -p aether-cli`

Expected: PASS.

### Task 188: Lean Multiline Control-Flow Conditions

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core `if`, `while`, conditional `seal until`, newline tokens,
  and checked source-pipeline execution.
- Produces: control-flow condition forms that accept newlines after the
  condition introducer before the condition expression.

- [x] **Step 1: Add failing parser/source examples**

Add checked examples proving `parseProgram` accepts line-broken conditions for
`if`, `while`, and `seal until`, and that `sourceLocal?` executes source using
those forms.

- [x] **Step 2: Verify red**

Run `lake build`.

Expected: FAIL before implementation on the new multiline condition examples.

- [x] **Step 3: Implement condition newline skipping**

Add a parser helper that skips newline tokens after condition introducers, then
call it before parsing `if`, `while`, and `seal until` condition expressions.

- [x] **Step 4: Document grammar and coverage**

Update the formal grammar and parser coverage notes for newline-tolerant
control-flow conditions.

- [x] **Step 5: Verify green**

Run:
- `lake build`
- `cargo test -p aether-lang -p aether-cli`

Expected: PASS.

### Task 189: Lean Multiline Function Declaration Opening

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core function declarations, newline tokens, parameter-list
  parsing, and checked source-pipeline execution.
- Produces: function declarations that accept newlines between the function name
  and `(` before the parameter list.

- [x] **Step 1: Add failing parser/source examples**

Add checked examples proving `parseProgram` accepts line-broken function
declaration openings for untyped and typed declarations, and that `sourceLocal?`
executes a source program using that declaration form.

- [x] **Step 2: Verify red**

Run `lake build`.

Expected: FAIL before implementation on the new multiline function declaration
opening examples.

- [x] **Step 3: Implement function declaration opening newline skipping**

Add a parser helper that skips newline tokens after a function declaration name,
then use it before matching the parameter-list opening `(`. Keep parse
diagnostic classification aligned with the same declaration shape.

- [x] **Step 4: Document grammar and coverage**

Update the formal grammar and parser coverage notes for newline-tolerant
function declaration openings.

- [x] **Step 5: Verify green**

Run:
- `lake build`
- `cargo test -p aether-lang -p aether-cli`

Expected: PASS.

### Task 64: Lean List Literal Core Support

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: lexer `[` and `]` tokens and the existing Rust list literal
  surface.
- Produces: first-class proof-core list literals that parse, type-check,
  evaluate, compare structurally, lower to VM list-construction opcodes, and
  run through the source pipeline.

- [x] **Step 1: Add failing checked examples**

Add Lean examples for parsing empty and non-empty list literals, evaluating
lists, type-checking them as `list`, rejecting numeric operators on lists,
compiling dynamic list construction in stack and frame VMs, and running list
source through the pipeline.

- [x] **Step 2: Extend core syntax and values**

Add `Expr.list` and `Value.list`, with structural equality support.

- [x] **Step 3: Extend parser and static checker**

Parse bracketed comma-separated expressions into `Expr.list`, treat list
literals as `Ty.list`, and still check element expressions for undeclared
variables and other errors.

- [x] **Step 4: Extend VM and pipeline rendering**

Add stack and frame VM list-construction opcodes so list elements are evaluated
before constructing `Value.list`; render `Ty.list` diagnostics.

- [x] **Step 5: Document and verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 65: Lean List Indexing Support

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core `Expr.list`/`Value.list` and lexer bracket tokens.
- Produces: postfix `expr[index]` syntax that parses, checks list/numeric
  operands, evaluates and lowers through both VMs, and reports source spans.

- [x] **Step 1: Add failing checked examples**

Add Lean examples for core list indexing, parser postfix syntax, static
success/error paths, stack and frame VM lowering, and source-pipeline runtime
and diagnostic behavior.

- [x] **Step 2: Extend core syntax and evaluation**

Add `Expr.index` and runtime evaluation for dynamic list indexing, returning
`none` for non-list targets, non-numeric indexes, negative indexes, and
out-of-bounds indexes.

- [x] **Step 3: Extend parser and static checker**

Parse bracket postfix indexing after primary expressions and calls. Check that
the target is list-like, the index is num-like, and return `Ty.unknown` because
current lists are heterogeneous.

- [x] **Step 4: Extend VM and pipeline rendering**

Add stack and frame VM index opcodes, lower target/index bytecode in order, and
surface precise static index diagnostics.

- [x] **Step 5: Document and verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 66: Lean Fixed Micro-Precision Float Literals

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: lexer `float int fracMicros` tokens.
- Produces: proof-core float expressions and values using fixed
  micro-precision arithmetic where possible, with parser, evaluator, static
  checker, bytecode, and source pipeline coverage.

- [x] **Step 1: Add failing checked examples**

Add Lean examples for parsing float literals, evaluating float arithmetic and
comparisons, type-checking floats as `num`, lowering float constants to both
VMs, and running float source through the pipeline.

- [x] **Step 2: Extend core syntax and numeric evaluation**

Add `Expr.float` and `Value.float`, preserve existing integer behavior for
integer-only operations, and evaluate mixed numeric operations through
micro-unit conversion.

- [x] **Step 3: Extend parser and static checker**

Parse `Lexer.TokenKind.float` into `Expr.float` and keep static type `Ty.num`
for both integer and fixed micro-precision literals.

- [x] **Step 4: Extend VM and pipeline behavior**

Lower float literals to `PUSH` values in stack and frame VMs and add source
pipeline examples proving float programs run rather than parse-fail.

- [x] **Step 5: Document and verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 67: Lean Postfix Field Access

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: lexer `.` tokens and postfix expression parsing.
- Produces: proof-core `expr.field` syntax with executable `.length` support
  for strings and lists, static rejection for unsupported concrete fields, VM
  lowering, and source diagnostics.

- [x] **Step 1: Add failing checked examples**

Add Lean examples for parsing field access, evaluating list and string
`length`, statically typing supported fields, rejecting unsupported concrete
field access, compiling field bytecode in stack/frame VMs, and running source
field access through the pipeline.

- [x] **Step 2: Extend core syntax and evaluation**

Add `Expr.field` and evaluate `.length` for `Value.list` and `Value.str`,
returning `none` for unsupported fields or target values.

- [x] **Step 3: Extend parser and static checker**

Parse `.` identifier as a postfix expression after primaries, calls, and
indexing. Type supported concrete fields and return `Ty.unknown` for unknown
targets.

- [x] **Step 4: Extend VM and pipeline diagnostics**

Add stack and frame VM field opcodes and render unsupported field diagnostics
with useful source spans.

- [x] **Step 5: Document and verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 68: Lean Postfix Method Calls

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: lexer `.` tokens, parenthesized argument parsing, and postfix
  expression parsing.
- Produces: proof-core `expr.method(args)` syntax with executable pure
  `.len()` support for strings and lists, static rejection for unsupported
  concrete methods, VM lowering, and source diagnostics.

- [x] **Step 1: Add failing checked examples**

Add Lean examples for parsing postfix method calls, evaluating list and string
`.len()`, statically typing supported method calls, rejecting unsupported
concrete method calls, compiling method bytecode in stack/frame VMs, and
running method calls through the source pipeline.

- [x] **Step 2: Extend core syntax and evaluation**

Add `Expr.method` and evaluate `.len()` with zero arguments for `Value.list`
and `Value.str`, returning `none` for unsupported methods or arities.

- [x] **Step 3: Extend parser and static checker**

Parse `.` identifier followed by parentheses as a postfix method call. Check
argument expressions and type supported concrete methods.

- [x] **Step 4: Extend VM and pipeline diagnostics**

Add stack and frame VM method opcodes and render unsupported method diagnostics
with useful source spans.

- [x] **Step 5: Document and verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 69: Lean String Escape Lexing

**Files:**
- Modify: `Aether/Lexer.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: lexer string literal scanning and located-token spans.
- Produces: proof-core string escapes for `\"`, `\\`, `\n`, and `\t`,
  plus deterministic invalid-escape diagnostics.

- [x] **Step 1: Add failing checked examples**

Add Lean examples for escaped quote/backslash/newline/tab tokenization,
source-pipeline execution of escaped strings, and invalid escape diagnostic
rendering.

- [x] **Step 2: Extend `readString`**

Teach the Lean string scanner to consume valid escape sequences into their
runtime characters while preserving source consumption for located spans.

- [x] **Step 3: Add invalid escape diagnostics**

Return a lexer error for unsupported escape sequences and ensure located-token
spans point at the offending escape.

- [x] **Step 4: Document and verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 70: Lean Block Comment Lexing

**Files:**
- Modify: `Aether/Lexer.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: lexer slash/star character scanning and located-token position
  tracking.
- Produces: `/* ... */` block comments in the Lean proof-core lexer, including
  newline-aware located scanning and unterminated-comment diagnostics.

- [x] **Step 1: Add failing checked examples**

Add Lean examples proving block comments are skipped in plain tokenization,
located tokenization keeps following token positions correct across comment
newlines, and unterminated block comments report deterministic lexer errors.

- [x] **Step 2: Implement plain block-comment skipping**

Teach `scanFuel` to recognize `/*`, consume through the next `*/`, and emit an
unterminated block-comment lexer error when EOF arrives first.

- [x] **Step 3: Implement located block-comment skipping**

Teach `scanLocatedFuel` to skip block comments while advancing line/column
positions for all consumed characters and to span unterminated block comments
from the opening slash to EOF.

- [x] **Step 4: Document and verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 71: Lean Static List Element Typing

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: existing Lean `Ty.list`, list literal checking, and index
  expression checking.
- Produces: element-aware list static types for homogeneous proof-core lists,
  precise indexed expression types, and preserved `unknown` element typing for
  mixed or otherwise imprecise lists.

- [x] **Step 1: Add failing checked examples**

Add Lean examples proving numeric and string list literals infer `list[num]`
and `list[str]`, mixed literals remain `list[unknown]`, indexing a homogeneous
list returns the element type, and assigning a homogeneous indexed result to an
incompatible existing variable is rejected.

- [x] **Step 2: Extend static type representation**

Represent list types with an element type while keeping `unknown` available for
mixed lists, empty lists, function results, and imprecise values.

- [x] **Step 3: Infer list and index types**

Thread element type inference through both the `Option` checker and detailed
checker, update compatibility helpers for fields, methods, assignment, and
diagnostic rendering.

- [x] **Step 4: Document and verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 72: Lean Static Function Result Inference

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Lean function declarations, return statements, and static function
  signature collection.
- Produces: function signatures that retain an inferred result type when
  visible from the body, so calls to concrete-return functions no longer always
  type as `unknown`.

- [x] **Step 1: Add failing checked examples**

Add Lean examples proving a function returning a numeric literal makes
`one()` type as `num`, a `let` initialized from that call records `num`, and a
later assignment of `str` to that binding is rejected.

- [x] **Step 2: Extend function signatures**

Add a result type to `FnSig` while preserving arity checking and duplicate
function diagnostics.

- [x] **Step 3: Infer result types during signature collection**

Infer result types from statically visible `return` statements using unknown
parameter bindings, merging multiple concrete returns conservatively to
`unknown` when they disagree.

- [x] **Step 4: Document and verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 73: Lean Static Implicit Function Result Inference

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: existing function result inference and the runtime rule that a
  function body without explicit `return` uses the final expression value.
- Produces: static function signatures that infer result types from final
  expression statements when no explicit return determines the result.

- [x] **Step 1: Add failing checked examples**

Add Lean examples proving a function whose body ends in an expression infers
that expression's type, records concrete types for variables initialized from
such calls, and rejects incompatible assignments after such calls.

- [x] **Step 2: Infer final expression types**

Teach the signature inference pass to use a final `Stmt.expr` as the implicit
function result while preserving explicit `return` inference and conservative
`unknown` merging for disagreement.

- [x] **Step 3: Document and verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 74: Lean Static If-Branch Environment Joins

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: static block checking for `if`/`else` branches and the existing
  variable environment model.
- Produces: conservative static environment joins for variables introduced in
  both branches with compatible types.

- [x] **Step 1: Add failing checked examples**

Add Lean examples proving a variable introduced as the same type in both
branches is available after the `if`, and a variable introduced with
incompatible branch types is not joined and therefore remains unavailable to
later statements.

- [x] **Step 2: Implement branch joins**

Compute variables newly introduced by both branch result environments, merge
compatible types, and return the joined state from `if` checking. Preserve the
existing behavior for `if` without `else`.

- [x] **Step 3: Document and verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 75: Lean Function Result Inference Uses If-Branch Joins

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 74 branch environment joins and Task 73 implicit
  final-expression function result inference.
- Produces: function signature inference that can use variables introduced by
  compatible `if`/`else` branches before a final expression.

- [x] **Step 1: Add failing checked examples**

Add Lean examples proving a function with `if cond { let x = 1 } else { let x =
2 }; x` infers `num`, and incompatible branch bindings keep the final variable
unavailable so the call result remains imprecise.

- [x] **Step 2: Thread branch joins through signature inference**

Teach the signature inference environment pass to compute the same conservative
join for `if`/`else` statements as the main static checker.

- [x] **Step 3: Document and verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 76: Lean Static Assignment Type Refinement

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: static assignment compatibility, `unknown` types from function
  parameters/results, and the variable environment model.
- Produces: assignment checking that refines `unknown` targets to concrete
  assigned types for later statements.

- [x] **Step 1: Add failing checked examples**

Add Lean examples proving an `unknown` variable assigned a numeric value is
tracked as `num`, and a later incompatible assignment is rejected with an
assignment mismatch.

- [x] **Step 2: Implement assignment refinement**

Update the static variable environment on successful assignment, refining
`unknown` targets and nested `list[unknown]` targets while preserving existing
concrete types.

- [x] **Step 3: Document and verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 77: Lean If-Branch Assignment Refinement Joins

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 74 branch joins and Task 76 assignment type refinement.
- Produces: branch joins that can refine existing `unknown` variables when both
  branches assign compatible concrete types.

- [x] **Step 1: Add failing checked examples**

Add Lean examples proving an existing `unknown` variable assigned `num` in both
branches is refined to `num` after the `if`, and incompatible branch assignments
do not refine the variable.

- [x] **Step 2: Extend branch joins**

Update the join helper to refine existing `unknown` variables from compatible
then/else branch environments while keeping incompatible branch updates
imprecise.

- [x] **Step 3: Document and verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 63: Lean String Literal Core Support

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: lexer `stringLit` tokens and the existing Rust string literal
  surface.
- Produces: first-class proof-core string literals that parse, type-check,
  evaluate, compare for equality/inequality, lower to VM push instructions,
  and run through the source pipeline.

- [x] **Step 1: Add failing checked examples**

Add Lean examples for parsing a string literal, type-checking it as `str`,
evaluating it, rejecting numeric operators on strings, compiling it to stack
and frame VM bytecode, and running a source string through the pipeline.

- [x] **Step 2: Extend core syntax and values**

Add `Expr.str` and `Value.str`, with equality support matching the Rust
interpreter surface.

- [x] **Step 3: Extend parser and static checker**

Parse `stringLit` tokens into `Expr.str`, treat string literals as `Ty.str`,
and preserve existing operand validation so only equality/inequality works for
known string operands.

- [x] **Step 4: Extend VM and pipeline rendering**

Lower string expressions to `push (Value.str ...)` in stack and frame compilers
and render `Ty.str` diagnostics.

- [x] **Step 5: Document and verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 62: Lean Expression-Start Parse Diagnostic Spans

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: parser statement failure classification and located token streams.
- Produces: parse diagnostics that point at the offending expression-start
  token for malformed `let`, assignment, return, `if`, `while`, and
  conditional `seal` expressions when the first expression token is invalid.

- [x] **Step 1: Add failing parser examples**

Update checked parser diagnostics so invalid expression starts report the
offending token instead of the statement keyword.

- [x] **Step 2: Add failing source diagnostic examples**

Update rendered source-pipeline examples so malformed expression starts use the
offending token's source range.

- [x] **Step 3: Implement expression-start failure offsets**

Add token-level helpers that identify invalid expression starts and return the
diagnostic token offset while preserving existing statement-start diagnostics
for incomplete expressions whose first token is valid.

- [x] **Step 4: Document diagnostic precision**

Record that the located source pipeline can now point at invalid expression
starts without requiring a fully spanned AST.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 28: Lean Frame Compiler For/Seal Loops

**Files:**
- Modify: `Aether/VM.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 27 frame compiler control flow and proof-core loop syntax
- Produces: frame bytecode support for `forRange` and `seal` inside
  frame-compiled functions

- [x] **Step 1: Add frame forRange lowering**

Lower integer-range `for` loops to iterator initialization, `iterator < end`,
body bytecode, iterator increment, and a backward frame jump.

- [x] **Step 2: Add frame seal lowering**

Lower `seal until condition` to a pre-body exit check using `not` plus
`FrameOp.jmpIfFalse`, and lower bare `seal` to body bytecode followed by a
backward `FrameOp.jmp`.

- [x] **Step 3: Add checked execution examples**

Verify a function using `forRange` accumulation and a function using
conditional `seal` through `runCompiledFrameProgram`.

- [x] **Step 4: Document remaining loop boundaries**

Record that loop-control patching for `break`/`continue`, named/method calls,
and full correspondence proofs remain future work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 27: Lean Frame Compiler Control Flow

**Files:**
- Modify: `Aether/VM.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 26 source-to-frame compiler and structured statement syntax
- Produces: frame bytecode support for `if`/`else` and `while` inside
  frame-compiled functions

- [x] **Step 1: Add frame jump opcodes**

Extend `FrameOp` with `jmp` and `jmpIfFalse`, and execute them in `frameStep`.

- [x] **Step 2: Add frame if/while lowering**

Lower `Stmt.ifThenElse` and `Stmt.while` in `compileFrameStmt` using relative
frame jumps and recursive frame block compilation.

- [x] **Step 3: Add checked execution examples**

Verify a function with `if`/`else` and a function with a bounded `while` loop
through `runCompiledFrameProgram`.

- [x] **Step 4: Document remaining control-flow boundaries**

Record that frame-compiled `for`/`seal`, loop-control patching for
`break`/`continue`, named/method calls, and full correspondence proofs remain
future work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 26: Lean Source-to-Frame Function Compiler

**Files:**
- Modify: `Aether/VM.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 25 frame VM, `Stmt.fnDecl`, `Stmt.ret`, and `Expr.call`
- Produces: source-to-frame bytecode compilation for top-level functions and
  positional calls

- [x] **Step 1: Add frame compiler environments**

Define frame function metadata, function lookup tables, function collection,
parameter-to-slot mapping, and target computation for hoisted function bodies.

- [x] **Step 2: Add frame expression and statement compilation**

Compile literals, variables, unary/binary expressions, positional calls, `let`,
assignment, expression statements, and `return` into `FrameOp` bytecode.

- [x] **Step 3: Add program compilation and execution helpers**

Compile main statements with top-level `fn` declarations skipped, append
`HALT`, append hoisted function bodies, and expose `runCompiledFrameProgram`.

- [x] **Step 4: Add checked compiler examples and document boundaries**

Verify emitted bytecode and execution for explicit-return functions and
parameter shadowing. Record that branches/loops inside frame-compiled functions,
named/method calls, and full correspondence proofs remain future work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 25: Lean VM Call Frames

**Files:**
- Modify: `Aether/VM.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Rust VM call-frame behavior and Lean proof-core VM values/operators
- Produces: direct Lean bytecode VM model for function calls and returns

- [x] **Step 1: Add frame bytecode and state**

Define `FrameOp`, `CallFrame`, `FrameState`, and initial frame-state helpers.

- [x] **Step 2: Add call/return execution**

Implement bounded `frameStep`, `runFrameFuel`, and `runFrame` with argument
passing, fresh callee locals, return IP tracking, caller-local restoration, and
unit return for empty return stacks.

- [x] **Step 3: Add checked bytecode examples**

Verify direct bytecode examples for explicit return values, implicit unit
return, and caller-local restoration after a call.

- [x] **Step 4: Document remaining compiler boundary**

Record that source-to-call-frame compilation remains future work before full
compiler/VM correspondence proofs.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 24: Lean Function Runtime Semantics

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 23 function parser output, `Expr.call`, and `Stmt.fnDecl`
- Produces: bounded executable Lean semantics for user-defined function calls

- [x] **Step 1: Add function environment model**

Define `Function`, `FnEnv`, lookup, binding, and parameter binding helpers.

- [x] **Step 2: Add executable function evaluator**

Add bounded `evalExprWithFns`, `execStmtWithFns`, and `execBlockWithFns` for
function declarations, positional calls, explicit returns, and implicit
last-value returns.

- [x] **Step 3: Add checked semantic examples**

Verify explicit return, implicit final-expression return, arity mismatch, and
parameter shadowing with `native_decide`.

- [x] **Step 4: Document remaining function boundaries**

Record that VM call-frame bytecode, named/method calls, source spans, and
correctness theorems remain future formalization work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 23: Lean Function Parser

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `Aether.Core.Expr.call`, `Aether.Core.Stmt.fnDecl`, and the Lean
  token stream
- Produces: executable Lean parsing for positional function calls and function
  declarations

- [x] **Step 1: Add positional call argument parsing**

Parse `name(expr, ...)` into `Expr.call name args`, including empty and
comma-separated argument lists.

- [x] **Step 2: Add function parameter parsing**

Parse `fn name(param, ...) { ... }` parameter lists into `List Ident`.

- [x] **Step 3: Add function declaration parsing**

Parse function declarations into `Stmt.fnDecl` with a brace-delimited body.

- [x] **Step 4: Add checked examples and document boundaries**

Verify positional call parsing and a function declaration plus call site with
`native_decide`; record that named arguments, method calls, source spans,
function runtime semantics, and correctness theorems remain future work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 22: Lean VM For/Seal Compilation

**Files:**
- Modify: `Aether/VM.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 21 parsed loop syntax, `Aether.Core` loop statements, and
  Lean VM branch/jump instructions
- Produces: checked compiler lowering for integer-range `for` loops and `seal`
  loops

- [x] **Step 1: Add forRange lowering**

Resolve or allocate the iterator slot, initialize it to the start value, compile
the body with the iterator in scope, check `iterator < end`, increment by one,
and jump back to the condition.

- [x] **Step 2: Add seal lowering**

Lower `seal until condition` to a pre-body exit check using `not` plus
`JMP_IF_FALSE`, and lower bare `seal` to body bytecode followed by a backward
`JMP`.

- [x] **Step 3: Add checked bytecode and execution examples**

Verify emitted bytecode and final VM state for `for i in 0..3`, verify emitted
bytecode and final VM state for `seal until x == 3`, and verify emitted bytecode
for bare `seal`.

- [x] **Step 4: Document remaining VM boundaries**

Record that loop-control patching for `break`/`continue`, functions, and call
frames remain future Lean VM proof targets.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 21: Lean For/Seal Parser

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 20 structured parser and `Aether.Core` loop statement syntax
- Produces: executable Lean parsing for integer-range `for` loops and `seal`
  loops

- [x] **Step 1: Add integer range parser**

Parse numeric `start..end` ranges for the Lean proof-core `Stmt.forRange`
surface.

- [x] **Step 2: Add for-loop parsing**

Parse `for name in start..end { ... }` into `Stmt.forRange`.

- [x] **Step 3: Add seal-loop parsing**

Parse `seal until condition { ... }` into `Stmt.seal (some condition)` and
`seal { ... }` into `Stmt.seal none`.

- [x] **Step 4: Add checked parser examples and document boundaries**

Verify `for`, conditional `seal`, and unconditional `seal` parsing with
`native_decide`; record that functions, calls, source spans, and correctness
theorems remain future parser work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 20: Lean Structured Statement Parser

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 19 proof-core parser and `Aether.Core` structured statement syntax
- Produces: executable Lean parsing for brace blocks, `if`/`else`, and `while`

- [x] **Step 1: Add brace block parser**

Parse `{ ... }` blocks with newline and tilde separators while letting `}` serve
as a statement boundary inside blocks.

- [x] **Step 2: Add structured statement parsing**

Parse `if condition { ... }`, optional `else { ... }`, and
`while condition { ... }` into `Stmt.ifThenElse` and `Stmt.while`.

- [x] **Step 3: Add checked parser examples**

Verify block parsing, if/else parsing, and while-body parsing with
`native_decide`.

- [x] **Step 4: Document remaining parser boundaries**

Record that structured `for`/`seal`, functions, calls, source spans, and parser
correctness theorems remain future Lean parser work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 19: Lean Proof-Core Parser

**Files:**
- Add: `Aether/Parser.lean`
- Modify: `Aether.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 18 Lean lexer and `Aether.Core` expression/statement syntax
- Produces: executable Lean parser for proof-core expressions and simple statements

- [x] **Step 1: Add precedence-aware expression parser**

Parse literals, variables, parenthesized expressions, unary negation/not,
multiplicative/additive operators, comparisons, equality, logical `&&`, and
logical `||` into `Aether.Core.Expr`.

- [x] **Step 2: Add simple statement parser**

Parse `let`, assignment, `return`, `break`, `continue`, expression statements,
newline separators, tilde separators, and EOF termination into
`Aether.Core.Stmt`.

- [x] **Step 3: Add checked parser examples**

Verify arithmetic precedence, parenthesized boolean precedence, tilde-separated
programs, and newline-separated control-flow statements with `native_decide`.

- [x] **Step 4: Wire module and document boundaries**

Import `Aether.Parser` from the top-level Lean module and document remaining
block parsing, structured control flow, functions, calls, spans, and parser
correctness theorem work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 18: Lean Lexical Scanner

**Files:**
- Add: `Aether/Lexer.lean`
- Modify: `Aether.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Rust lexer token surface and Lean proof DSL module scaffold
- Produces: executable Lean token-kind scanner with checked lexical examples

- [x] **Step 1: Model token kinds in Lean**

Define keyword, identifier, literal, operator, punctuation, newline, EOF, and
error token kinds corresponding to the Rust lexer surface.

- [x] **Step 2: Implement executable scanning**

Scan identifiers/keywords, integers, fixed micro-precision floats, strings,
comments, statement separators, ranges, operators, delimiters, lexical errors,
and EOF.

- [x] **Step 3: Add checked examples**

Verify `1.5` versus `1..10`, keyword/operator scanning, comment handling,
tilde separators, and string termination errors with `native_decide`.

- [x] **Step 4: Wire module and document boundaries**

Import `Aether.Lexer` from the top-level Lean module and record that source-span
tracking plus parser integration remain future work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 17: Lean While Loop Compilation

**Files:**
- Modify: `Aether/VM.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 16 branch-aware compiler and VM relative jumps
- Produces: checked compiler lowering for bounded `while` statements

- [x] **Step 1: Add while lowering to branch-aware compiler**

Compile the loop condition, emit `JMP_IF_FALSE` over the body and back jump,
compile the body with threaded slots, and emit a negative `JMP` back to the
condition.

- [x] **Step 2: Add checked bytecode example**

Verify emitted bytecode for `while x < 3 { x = x + 1 }`, including the forward
exit offset and backward continuation offset.

- [x] **Step 3: Add checked execution examples**

Verify bounded VM execution for both multi-iteration and zero-iteration loop
paths.

- [x] **Step 4: Document boundaries**

Record that bounded `while` lowering is modeled in Lean, while `for`/`seal`
loops, functions, and call frames remain future Lean compiler formalization
work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 16: Lean If/Branch Compilation

**Files:**
- Modify: `Aether/VM.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 15 block compiler and VM branch opcodes
- Produces: checked branch-aware compiler for `if` statements

- [x] **Step 1: Add branch-aware compiler layer**

Define mutually recursive statement/block compilation that supports previous straight-line forms plus `if`.

- [x] **Step 2: Lower if to VM jumps**

Emit condition bytecode, `JMP_IF_FALSE`, then-branch bytecode, optional `JMP`, and else-branch bytecode with checked offsets.

- [x] **Step 3: Add checked true/false branch examples**

Verify emitted bytecode for an if/else assignment and final VM locals for true and false conditions.

- [x] **Step 4: Document boundaries**

Record that loops, functions, and call frames remain future Lean compiler formalization work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 15: Lean Straight-Line Block Compilation

**Files:**
- Modify: `Aether/VM.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 14 straight-line statement compiler
- Produces: checked block compiler for sequences of supported straight-line statements

- [x] **Step 1: Add block compiler**

Thread `SlotEnv` through a list of statements and concatenate emitted bytecode.

- [x] **Step 2: Add block execution helper**

Run compiled block bytecode with a trailing `halt` against explicit initial locals.

- [x] **Step 3: Add checked block examples**

Verify emitted bytecode and final VM state for `let x = 1; x = x + 2; x`, and verify unsupported block statements fail compilation.

- [x] **Step 4: Document boundaries**

Record that branching, loops, functions, and call frames remain future block compiler formalization work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 14: Lean Straight-Line Statement Compilation

**Files:**
- Modify: `Aether/VM.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 13 slot-aware expression compilation and VM local-slot model
- Produces: checked statement compilation for `let`, assignment, and expression statements

- [x] **Step 1: Add slot resolution**

Define `SlotEnv.resolve` so declarations allocate a new local slot or reuse an existing one.

- [x] **Step 2: Add straight-line statement compiler**

Lower `let` to expression bytecode plus `STORE`, lower assignment to existing-slot `STORE`, and lower expression statements to expression bytecode.

- [x] **Step 3: Add checked statement execution examples**

Verify emitted bytecode and final VM state for declaration and assignment, plus missing-slot assignment failure.

- [x] **Step 4: Document boundaries**

Record that branching, loops, functions, and call frames remain future statement compiler formalization work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 13: Lean Variable Slot Expression Compilation

**Files:**
- Modify: `Aether/VM.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 12 closed-expression compiler and VM locals model
- Produces: checked slot-aware expression compiler for variables

- [x] **Step 1: Add slot environment**

Define `SlotEnv` as an explicit identifier-to-local-slot table with lookup.

- [x] **Step 2: Add slot-aware expression compiler**

Lower variables to `Op.load slot` while preserving literal, unary, and binary expression lowering.

- [x] **Step 3: Add checked variable correspondence examples**

Verify emitted bytecode for `x + 2`, compare compiled execution against direct `evalExpr`, and check missing variables fail compilation.

- [x] **Step 4: Document scope**

Record that slot-aware variable expression compilation is modeled, while function calls and call frames remain future Lean work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 12: Lean Expression Compiler Correspondence Slice

**Files:**
- Modify: `Aether/VM.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `Aether.Core.evalExpr` and `Aether.VM` stack-machine execution
- Produces: checked expression-to-bytecode compiler for closed literal/unary/binary expressions and evaluator/compiler agreement examples

- [x] **Step 1: Add `compileExpr`**

Lower numeric literals, boolean literals, unary expressions, and binary expressions into stack bytecode.

- [x] **Step 2: Add compiled-expression execution helpers**

Run emitted bytecode with a trailing `halt` and expose the top stack value for checked examples.

- [x] **Step 3: Add checked correspondence examples**

Compare compiled execution with direct expression evaluation for multiplication, nested arithmetic with modulo and unary negation, and boolean negation.

- [x] **Step 4: Document scope and remaining boundaries**

Call out that variable-slot compilation, calls, and call frames remain future formalization work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 11: Lean Core VM Semantics

**Files:**
- Create: `Aether/VM.lean`
- Modify: `Aether.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `Aether.Core` values, operators, and truthiness
- Produces: checked Lean stack-machine model for the core Titan VM subset

- [x] **Step 1: Define Lean bytecode and VM state**

Add opcodes for constants, locals, binary/unary operations, unconditional jumps, conditional false jumps, and halt.

- [x] **Step 2: Define VM stepping**

Implement one-step execution plus bounded fuel-based execution over instruction pointer, stack, locals, code, and halted state.

- [x] **Step 3: Add checked VM examples**

Cover arithmetic stack execution, local store/load with unary negation, and false conditional jump behavior.

- [x] **Step 4: Document VM formalization status**

Record the current bytecode subset and call out function call-frame formalization as remaining work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 9: Lean 4 Formalization Scaffold

**Files:**
- Create: `lakefile.lean`
- Create: `lean-toolchain`
- Create: `Aether.lean`
- Create: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-facing core language from `docs/FORMAL_CORE.md`
- Produces: checked Lean 4 definitions for core syntax, values, expression evaluation, environments, and statement flow

- [x] **Step 1: Check local Lean tooling**

Verify `lean --version` and `lake --version` are available.

- [x] **Step 2: Add Lake package scaffold**

Create a root Lake package with `Aether` as the default target and pin the Lean toolchain.

- [x] **Step 3: Encode the initial core**

Define `Expr`, `Stmt`, `Value`, `Flow`, `Env`, expression evaluation, and a small statement-step relation.

- [x] **Step 4: Add initial checked facts**

Prove lookup after bind and variable evaluation after bind; add checked examples for modulo and truthiness-backed boolean evaluation.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 61: Lean Float Literal Parser Rejection

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: lexer float tokens and the integer-valued proof-core parser.
- Produces: explicit rejection of decimal float literals by the current
  proof-core parser, avoiding silent truncation into integer expressions.

- [x] **Step 1: Add failing parser examples**

Add checked examples requiring `parseExpr` and `parseProgramDetailed` to reject
float literals in proof-core expressions.

- [x] **Step 2: Add failing source diagnostic example**

Add a rendered source-pipeline diagnostic for a statement containing a float
literal.

- [x] **Step 3: Remove float-to-int parser conversion**

Remove the parser branch that turns lexer float tokens into integer `Expr.num`
values.

- [x] **Step 4: Document lexer/parser boundary**

Record that decimal float tokens remain lexed for host/runtime compatibility
but are outside the current integer proof-core expression parser.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 8: VM Loop Flow Parity

**Files:**
- Modify: `crates/aether-lang/src/vm.rs`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: parser/interpreter `Break` and `Continue` statements from Task 3
- Produces: compiler-lowered VM jumps for loop exit and loop continuation

- [x] **Step 1: Write failing VM tests for loop flow**

Cover `break` exiting a `while` loop and `continue` skipping the rest of a `while` loop body through source-level VM execution.

- [x] **Step 2: Verify red**

Run: `cargo test -p aether-lang vm::tests::test_vm_ -- --nocapture`
Expected: FAIL only for the new loop-flow VM tests.

- [x] **Step 3: Add compiler loop context patching**

Record pending `break` and `continue` jump sites while compiling loop bodies, then patch them to the loop exit or continuation target after lowering the loop.

- [x] **Step 4: Document VM loop-flow correspondence**

State that VM `break` and `continue` are compiler-lowered `JMP` instructions rather than dedicated runtime opcodes.

- [x] **Step 5: Verify**

Run: `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 6: Compiler/VM Lowering Parity

**Files:**
- Modify: `crates/aether-lang/src/vm.rs`
- Modify: `crates/aether-lang/src/ast.rs`

**Interfaces:**
- Consumes: AST expression and statement variants supported by the interpreter
- Produces: VM bytecode coverage for arithmetic, comparisons, branches, loops, and variables

- [x] **Step 1: Write failing VM tests matching interpreter behavior**

For each operator and control-flow construct supported by the interpreter, add a VM/compiler test.

- [x] **Step 2: Add missing opcodes**

Add explicit comparison, modulo, boolean, branch, and local assignment opcodes instead of encoding them as arithmetic hacks.

- [x] **Step 3: Verify interpreter/VM parity**

Run the same small programs through interpreter and VM where possible and assert matching numeric/boolean results.

- [x] **Step 4: Verify**

Run: `cargo test -p aether-lang`
Expected: PASS.

### Task 7: Remaining VM and Proof-DSL Work

**Files:**
- Modify: `crates/aether-lang/src/vm.rs`
- Modify: `crates/aether-lang/src/ast.rs`
- Create or modify: `docs/` proof-facing language specification files

**Interfaces:**
- Consumes: user-defined functions from Task 4 and VM expression/control-flow support from Task 6
- Produces: bytecode call frames for user functions, explicit return lowering, and a clearer proof-DSL formalization surface

- [x] **Step 1: Add VM tests for function calls and returns**

Cover `fn add(a, b) { return a + b~ }`, implicit last-expression return, and local parameter isolation through VM execution.

- [x] **Step 2: Add VM call-frame opcodes**

Introduce function labels, call/return opcodes, frame setup, parameter binding, and return value propagation.

- [x] **Step 3: Document the proof-facing core language**

Define the stable grammar and small-step or big-step semantics subset intended for Lean 4 formalization.

- [x] **Step 4: Verify**

Run: `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 29: Lean Frame Compiler Loop-Control Patching

**Files:**
- Modify: `Aether/VM.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Lean `Stmt.break` and `Stmt.continue` parsed into the proof core.
- Produces: frame-compiler jump patching for loop exits and loop continuation
  targets inside frame-compiled functions.

- [x] **Step 1: Add pending loop-control compile result**

Introduce a frame compile result that carries generated bytecode, threaded slot
state, pending `break` jump sites, and pending `continue` jump sites.

- [x] **Step 2: Patch loop boundaries**

Patch pending sites at `while`, integer-range `for`, and `seal` boundaries.
`break` targets the loop exit; `continue` targets the condition check for
`while`/`seal` and the increment block for `for`.

- [x] **Step 3: Preserve public compiler API**

Keep `compileFrameStmt` and `compileFrameBlock` returning `(SlotEnv × List
FrameOp)` and reject unconsumed `break`/`continue` outside loops.

- [x] **Step 4: Add checked examples**

Verify a frame-compiled function where `break` exits a `while` loop and another
where `continue` skips the rest of the current loop body.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 30: Lean Static Well-Formedness Gate

**Files:**
- Create: `Aether/Static.lean`
- Modify: `Aether.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Lean `Aether.Core` syntax for expressions, statements, function
  declarations, calls, and loop-control statements.
- Produces: executable well-formedness checks for the unannotated proof-core
  language before compiler and VM correspondence proofs.

- [x] **Step 1: Add static type and signature model**

Define proof-core static types `num`, `bool`, `unit`, and `unknown`, plus
variable and function-signature environments.

- [x] **Step 2: Check expressions**

Validate known arithmetic and comparison operand shapes, resolve variables,
check function call arity, and use `unknown` for unannotated parameters and
call results.

- [x] **Step 3: Check statements and programs**

Validate declaration-before-use, assignment compatibility, valid
`return` placement, valid `break`/`continue` placement, loops, branches, and
function bodies after collecting top-level signatures.

- [x] **Step 4: Add checked examples**

Cover valid and invalid expressions, undeclared assignment rejection,
top-level loop-control/return rejection, valid loop control, valid function
calls, and arity mismatch rejection.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 31: Lean Checked Frame Compilation Bridge

**Files:**
- Modify: `Aether/VM.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `Aether.Static.checkProgram` and the existing frame compiler.
- Produces: a checked source-to-bytecode entrypoint that refuses statically
  invalid proof-core programs before frame bytecode lowering.

- [x] **Step 1: Import the static checker into the VM module**

Keep raw frame compilation available while making static checking usable by
compiler entrypoints.

- [x] **Step 2: Add checked compiler and runner APIs**

Add `compileCheckedFrameProgram`, `runCheckedFrameProgram`, and
`checkedFrameLocal?`.

- [x] **Step 3: Record a static-gate theorem**

Prove that successful checked frame compilation implies
`Static.checkProgram` did not reject the source program.

- [x] **Step 4: Add checked examples**

Verify valid function code still runs through the checked entrypoint, while
numeric/boolean arithmetic misuse and function arity mismatch are rejected.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 32: Lean Source-To-Checked-VM Pipeline

**Files:**
- Modify: `Aether/VM.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `Aether.Parser.parseProgram`, `compileCheckedFrameProgram`, and
  the frame VM runner.
- Produces: source-string entrypoints that tokenize, parse, statically check,
  lower, and run proof-core programs in Lean.

- [x] **Step 1: Import parser access into VM pipeline helpers**

Use the existing parser module without changing the parser API.

- [x] **Step 2: Add source entrypoints**

Add `compileCheckedFrameSource`, `runCheckedFrameSource`, and
`checkedFrameSourceLocal?`.

- [x] **Step 3: Add checked source examples**

Verify a valid function source program executes through the full pipeline.
Verify malformed source, invalid numeric/boolean arithmetic, and arity mismatch
are rejected before bytecode execution.

- [x] **Step 4: Document the end-to-end path**

Record the tokenize/parse/static-check/lower/run path and its current
boundaries in the formal core documentation.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 33: Lean Stage-Aware Source Diagnostics

**Files:**
- Create: `Aether/Pipeline.lean`
- Modify: `Aether.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Lean lexer tokens, parser output, static checker results, checked
  frame compiler output, and frame VM execution.
- Produces: source pipeline entrypoints with explicit failure phases instead
  of undifferentiated `Option.none`.

- [x] **Step 1: Define pipeline error stages**

Add lexical, parse, static, compile, and runtime error constructors.

- [x] **Step 2: Preserve lexer failures before parsing**

Scan token output for the first `TokenKind.error` and return a lexical
diagnostic rather than collapsing it into parse failure.

- [x] **Step 3: Add staged source entrypoints**

Add `parseSource`, `checkSource`, `compileSource`, `runSource`, and
`sourceLocal?`.

- [x] **Step 4: Add checked examples**

Verify successful execution and distinct lexical, parse, static arithmetic,
static arity, and runtime/fuel behavior.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 34: Lean Detailed Static Diagnostics

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core expressions/statements and the stage-aware source
  pipeline.
- Produces: concrete static error reasons propagated through source
  diagnostics.

- [x] **Step 1: Add static diagnostic type**

Define errors for undeclared variables/functions, unary and binary operand
mismatches, assignment mismatches, arity mismatches, function signature
mismatches, and invalid `return`/`break`/`continue` placement.

- [x] **Step 2: Add detailed static checker**

Add `checkExprDetailed`, `checkStmtDetailed`, `checkBlockDetailed`, and
`checkProgramDetailed` alongside the existing `Option` checker.

- [x] **Step 3: Propagate detailed static errors through the pipeline**

Change `Pipeline.Error.static` to carry `Static.CheckError` and make
`checkSource`/`compileSource` use the detailed checker.

- [x] **Step 4: Add checked examples**

Verify detailed diagnostics for operand mismatch, undeclared assignment,
top-level loop control, top-level return, and arity mismatch.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 35: Lean Parser Diagnostic Wrapper

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: existing `Option` parser and stage-aware source pipeline.
- Produces: parse diagnostics carrying a failure context and the first token
  at the failed statement start.

- [x] **Step 1: Add parser diagnostic types**

Define `ParseContext` and `ParseError` for broad expression, statement, block,
range, parameter, terminator, and program-end failures.

- [x] **Step 2: Add detailed parser wrapper**

Add `parseProgramFromTokensDetailed` and `parseProgramDetailed` without
rewriting the existing parser.

- [x] **Step 3: Add checked parser examples**

Verify diagnostic classification for expression, range, and parameter-list
parse failures.

- [x] **Step 4: Propagate parser diagnostics through the pipeline**

Change `Pipeline.Error.parse` to carry `Parser.ParseError`.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 36: Lean Positioned Lexer Diagnostics

**Files:**
- Modify: `Aether/Lexer.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Lean source strings and existing token-kind lexer.
- Produces: location-aware tokenization and lexical diagnostics with line and
  column while preserving the parser's token-kind API.

- [x] **Step 1: Add source-position model**

Define `SourcePos`, `LocatedToken`, and position advancement helpers.

- [x] **Step 2: Add located tokenization**

Add `tokenizeLocated` as a parallel lexer API that emits token start
positions without changing `tokenize`.

- [x] **Step 3: Add checked lexer examples**

Verify ordinary token positions and lexical error positions.

- [x] **Step 4: Propagate lexical positions through the pipeline**

Change `Pipeline.Error.lex` to carry the first lexer error's `SourcePos` and
make `parseSource` use located tokenization.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 37: Lean Positioned Parse Diagnostics

**Files:**
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: parser diagnostics and located lexer tokens.
- Produces: parse diagnostics that include source line and column without
  changing the parser's token-kind API.

- [x] **Step 1: Add located terminator skipping**

Mirror parser terminator skipping on `LocatedToken` lists so leading newlines
and tildes do not hide the failed statement start.

- [x] **Step 2: Attach parse positions in the pipeline**

Change `Pipeline.Error.parse` to carry `Parser.ParseError` and
`Lexer.SourcePos`.

- [x] **Step 3: Add checked examples**

Verify parse position for an expression failure and for a range failure after
leading terminators.

- [x] **Step 4: Document positioned parse diagnostics**

Record that parser diagnostics in the pipeline now carry context plus source
position.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 38: Lean Duplicate Declaration Static Checks

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: top-level function declarations and function parameter lists.
- Produces: detailed static diagnostics for duplicate names that would
  otherwise be hidden by environment shadowing.

- [x] **Step 1: Add duplicate diagnostic constructors**

Add `duplicateFunction` and `duplicateParameter` to `Static.CheckError`.

- [x] **Step 2: Check duplicate parameters**

Add `bindUnknownParamsDetailed` so function body checking rejects duplicate
parameter names.

- [x] **Step 3: Check duplicate top-level functions**

Add `collectFnSigsDetailed` and make `checkProgramDetailed` use it before
checking the program body.

- [x] **Step 4: Add checked examples**

Verify duplicate function names and duplicate parameter names are rejected
directly and through `Pipeline.compileSource`.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 39: Lean Checked Compiler Uses Detailed Static Gate

**Files:**
- Modify: `Aether/VM.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `Static.checkProgramDetailed` and the frame compiler.
- Produces: strict checked compilation aligned with the diagnostic source
  pipeline.

- [x] **Step 1: Swap checked compiler gate**

Make `compileCheckedFrameProgram` use `checkProgramDetailed` rather than the
older `Option` checker.

- [x] **Step 2: Update static-gate theorem**

Record that successful checked frame compilation implies an accepted detailed
static check witness.

- [x] **Step 3: Add checked compiler examples**

Verify duplicate functions and duplicate parameters are rejected through the
checked compiler/source APIs.

- [x] **Step 4: Document strict checked compilation**

Record that checked AST/source compilation uses the detailed static checker.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 40: Lean Pipeline Diagnostic Rendering

**Files:**
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: pipeline lexical, parser, static, compile, and runtime diagnostic
  variants.
- Produces: deterministic string rendering for source-pipeline diagnostics.

- [x] **Step 1: Add rendering helpers**

Add renderers for source positions, operators, static types, token summaries,
parse contexts, parser errors, and static errors.

- [x] **Step 2: Add full pipeline error rendering**

Add `errorString` and `compileSourceErrorString`.

- [x] **Step 3: Add checked string examples**

Verify stable rendered strings for lexical, parse, static operand, and
duplicate-function diagnostics.

- [x] **Step 4: Document rendered diagnostics**

Record diagnostic rendering in the formal core documentation.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 41: Lean Source Diagnostic Helper Coverage

**Files:**
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `Pipeline.errorString` and staged source entrypoints.
- Produces: deterministic rendered-error helpers for parse, check, compile,
  run, and source-local lookup APIs.

- [x] **Step 1: Add shared result renderer**

Add a generic `resultErrorString` helper for `Except Pipeline.Error α`.

- [x] **Step 2: Add source-stage wrappers**

Add `parseSourceErrorString`, `checkSourceErrorString`,
`runSourceErrorString`, and `sourceLocalErrorString` alongside the existing
`compileSourceErrorString`.

- [x] **Step 3: Add checked examples**

Verify stable strings for lexical, parse, static, and runtime/local-access
failures, plus a no-error run case.

- [x] **Step 4: Document helper coverage**

Record that deterministic rendered diagnostics are available across the public
source pipeline entrypoints.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 42: Lean Later-Statement Parse Positions

**Files:**
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: located lexer tokens and the existing token-kind parser API.
- Produces: parse diagnostics whose `SourcePos` follows the failed statement
  boundary even after earlier statements parsed successfully.

- [x] **Step 1: Add located token-kind projection**

Add a helper that projects `LocatedToken` streams into parser token-kind
streams without changing the parser API.

- [x] **Step 2: Replay statement parsing over located tokens**

Add `parseLocatedProgramDetailed` so the pipeline advances through successful
statements and reports parse failures at the current located statement start.

- [x] **Step 3: Use located parsing in `parseSource`**

Replace the old whole-program fallback position with the located program
parser.

- [x] **Step 4: Add checked examples**

Verify parse failures after a valid first statement report the second
statement's line and column.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 43: Lean Token Source Spans

**Files:**
- Modify: `Aether/Lexer.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: located lexer tokenization and source-position advancement.
- Produces: half-open token source ranges for future parser/AST diagnostics
  while preserving the parser's `TokenKind` stream.

- [x] **Step 1: Add span model**

Define `SourceSpan`, extend `LocatedToken` with a `stop` position, and expose
`LocatedToken.span`.

- [x] **Step 2: Emit token end positions**

Update `scanLocatedFuel` to compute each token's end position for single-char,
multi-char, literal, identifier, newline, EOF, and error tokens.

- [x] **Step 3: Preserve pipeline behavior**

Update located-token pattern matches in `Aether.Pipeline` while continuing to
use token starts for current diagnostics.

- [x] **Step 4: Add checked examples**

Verify located-token ranges, newline ranges, unterminated string ranges, and
`1..10` source spans.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 44: Lean Pipeline Span Diagnostics

**Files:**
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `Lexer.SourceSpan`, span-aware `LocatedToken` values, and staged
  source diagnostics.
- Produces: lexical and parse pipeline diagnostics carrying token ranges
  instead of start positions only.

- [x] **Step 1: Change pipeline error payloads**

Update `Pipeline.Error.lex` and `Pipeline.Error.parse` to carry
`Lexer.SourceSpan`.

- [x] **Step 2: Preserve spans while scanning diagnostics**

Make `firstLexError` return the lexer error token's full span and make
`parseLocatedProgramDetailed` return the failed statement token span.

- [x] **Step 3: Render diagnostic ranges**

Add `spanString` and update `errorString` to render lexical and parse errors
with half-open ranges.

- [x] **Step 4: Add checked examples**

Update checked pipeline examples so lexical and parse diagnostics prove the
exact reported source ranges, including later-statement parse failures.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 45: Lean Static Diagnostic Source Spans

**Files:**
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: detailed static checker errors and located lexer tokens.
- Produces: best-effort source ranges for static diagnostics without changing
  the current unspanned parser AST.

- [x] **Step 1: Add token matching helpers**

Add helpers that match located tokens by identifier, binary operator, unary
operator, and control-flow keyword.

- [x] **Step 2: Add span search helpers**

Add first, last, and second matching token span lookup helpers so static errors
can point at the most useful occurrence for ordinary, call-site, and duplicate
name failures.

- [x] **Step 3: Attach spans to static errors**

Change `Pipeline.Error.static` to carry `Option Lexer.SourceSpan` and have
`checkSource`/`compileSource` attach a best-effort span.

- [x] **Step 4: Render and test static spans**

Update `errorString` and checked examples for operator mismatch, arity
mismatch, duplicate names, and invalid top-level `break`.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 46: Lean Seal Emoji Lexer Alias

**Files:**
- Modify: `Aether/Lexer.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Rust lexer behavior where `seal` and `🦭` map to the same token.
- Produces: Lean lexer parity for the proof DSL seal control-flow alias.

- [x] **Step 1: Add plain lexer emoji handling**

Update `scanFuel` so `🦭` emits `TokenKind.seal`.

- [x] **Step 2: Add located lexer emoji handling**

Update `scanLocatedFuel` so the emoji alias emits `TokenKind.seal` with a
checked source range.

- [x] **Step 3: Add checked examples**

Verify ordinary tokenization for `🦭 until ...` and located-token ranges for
the emoji alias.

- [x] **Step 4: Document alias parity**

Record that the Lean lexer recognizes both `seal` and `🦭`, matching the Rust
lexer surface.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 47: Lean Structural Duplicate Diagnostic Spans

**Files:**
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: located token streams and detailed duplicate-name static errors.
- Produces: more precise duplicate function and duplicate parameter spans
  without requiring a spanned AST.

- [x] **Step 1: Add function-name span scanning**

Add helpers that find identifier spans in `fn name` token patterns and return
the duplicate declaration span for repeated functions.

- [x] **Step 2: Add parameter-list span scanning**

Add helpers that search function parameter lists up to `)` and return the
repeated parameter span.

- [x] **Step 3: Use structural duplicate spans**

Route `duplicateFunction` and `duplicateParameter` through the structural
helpers instead of loose second-identifier matching.

- [x] **Step 4: Add regression examples**

Verify duplicate function diagnostics ignore same-name calls in function
bodies and duplicate parameter diagnostics work when the function name matches
the repeated parameter.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 48: Lean Seal Emoji Source Pipeline

**Files:**
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Lean lexer support for `🦭` as `TokenKind.seal`, parser seal
  statements, static checking, and frame VM lowering.
- Produces: checked end-to-end proof-DSL source pipeline coverage for the seal
  emoji alias.

- [x] **Step 1: Add parse pipeline example**

Verify `parseSource` maps `🦭 until x == 3 { ... }` to the same `Stmt.seal`
syntax as the word-form keyword.

- [x] **Step 2: Add execution pipeline example**

Verify a source program using `🦭 until` statically checks, compiles, runs, and
stores the expected local value.

- [x] **Step 3: Document end-to-end alias coverage**

Record that the seal emoji alias is checked beyond tokenization through the
source pipeline.

- [x] **Step 4: Verify**

Run: `lake build`
Expected: PASS.

### Task 49: Lean Conditional Seal Parse Diagnostics

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: existing parser diagnostic wrapper and `seal until` grammar.
- Produces: more accurate parse context for malformed conditional seal loops.

- [x] **Step 1: Refine statement-start classification**

Classify `seal until ...` parse failures as expression-context failures rather
than generic block failures.

- [x] **Step 2: Add parser diagnostic example**

Verify `parseProgramDetailed "seal until { break }"` reports an expected
expression context.

- [x] **Step 3: Add pipeline rendered example**

Verify the source pipeline renders the conditional seal parse error with the
expected expression context and source range.

- [x] **Step 4: Document diagnostic coverage**

Record conditional `seal until` expression failures in the formal-core parser
diagnostic coverage.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 50: Lean If/While Condition Parse Diagnostics

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: existing parser diagnostic wrapper and structured `if`/`while`
  grammar.
- Produces: more accurate parse context for malformed condition-bearing
  structured statements.

- [x] **Step 1: Refine `if` classification**

Classify `if { ... }` as an expression-context failure while preserving
block-context failures for `if condition` without a block.

- [x] **Step 2: Refine `while` classification**

Classify `while { ... }` as an expression-context failure while preserving
block-context failures for `while condition` without a block.

- [x] **Step 3: Add checked diagnostics**

Add parser and rendered pipeline examples for missing `if` and `while`
conditions.

- [x] **Step 4: Document condition diagnostics**

Record missing `if`/`while` condition-expression coverage in the formal-core
parser diagnostic notes.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 51: Lean Function Body Parse Diagnostics

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: parser diagnostic wrapper and function declaration grammar.
- Produces: more accurate parse context for function declarations whose
  parameter list is complete but whose body block is missing.

- [x] **Step 1: Recognize parameter-list prefixes**

Add token-level recognition for complete function parameter-list prefixes after
`fn name(`.

- [x] **Step 2: Refine function classification**

Classify `fn name(params)` failures as block-context failures while preserving
params-context failures for malformed parameter lists.

- [x] **Step 3: Add checked examples**

Add parser and rendered pipeline examples for a missing function body after a
valid parameter list.

- [x] **Step 4: Document diagnostic distinction**

Record that function parser diagnostics distinguish malformed parameters from
missing body blocks.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 52: Lean For-Loop Body Parse Diagnostics

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: parser diagnostic wrapper and integer-range `for` grammar.
- Produces: more accurate parse context for `for` loops whose range is
  complete but whose body block is missing.

- [x] **Step 1: Recognize integer-range prefixes**

Add token-level recognition for `number .. number` range prefixes after
`for name in`.

- [x] **Step 2: Refine `for` classification**

Classify `for name in start..end` failures as block-context failures while
preserving range-context failures for malformed ranges.

- [x] **Step 3: Add checked examples**

Add parser and rendered pipeline examples for a missing loop body after a valid
integer range.

- [x] **Step 4: Document diagnostic distinction**

Record that `for` parser diagnostics distinguish malformed ranges from missing
body blocks.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 53: Lean Stray Else Parse Diagnostics

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: parser diagnostic wrapper and `if`/`else` grammar.
- Produces: explicit parse context for an `else` token that appears without a
  preceding parsed `if` statement.

- [x] **Step 1: Add `if`-statement parse context**

Represent malformed standalone `else` input as an `if`-statement context rather
than a generic statement or expression failure.

- [x] **Step 2: Classify stray `else`**

Teach statement-start classification to recognize `else` as a dependent token
that requires a preceding parsed `if`.

- [x] **Step 3: Add checked examples**

Add parser and rendered pipeline examples for a stray `else` token.

- [x] **Step 4: Document diagnostic coverage**

Record that parser diagnostics explicitly distinguish stray `else` from other
statement-start failures.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 54: Located Lexer Token Stream Consistency

**Files:**
- Modify: `Aether/Lexer.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: plain `tokenize` scanner and located `tokenizeLocated` scanner.
- Produces: checked projection helper showing the located scanner preserves the
  same token-kind stream as the parser-facing scanner for representative core
  inputs.

- [x] **Step 1: Add failing projection examples**

Add Lean examples that express the desired equality between `tokenize source`
and the token-kind projection of `tokenizeLocated source`.

- [x] **Step 2: Implement token-kind projection**

Expose a small helper that maps located tokens back to their `TokenKind`
stream.

- [x] **Step 3: Cover representative lexical surfaces**

Check equality for ordinary programs, comments/newlines, ranges, strings,
lexical errors, and the `🦭` alias.

- [x] **Step 4: Document the invariant**

Record the located/plain scanner consistency invariant in the formal-core
lexer notes.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 55: Lean Signed Integer For-Ranges

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `Aether/VM.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `Stmt.forRange` signed `Int` bounds, lexer minus/number tokens, and
  existing VM lowering for integer range bounds.
- Produces: parser and source pipeline support for negative integer endpoints
  in `for` ranges.

- [x] **Step 1: Add failing parser and pipeline examples**

Add checked examples for `for i in -2..2 { ... }` parsing and source execution.

- [x] **Step 2: Parse signed integer literals in range positions**

Extend range parsing to accept either `number` or `- number` endpoints while
preserving malformed-range diagnostics.

- [x] **Step 3: Update diagnostic range-prefix classification**

Recognize signed complete range prefixes so missing-body errors still report
`expected block` instead of `expected range`.

- [x] **Step 4: Document signed ranges**

Update the formal-core grammar and parser notes to include signed integer
range endpoints.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 56: Lean Boolean Control-Flow Conditions

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: detailed static checker, source diagnostics, and control-flow
  statements with condition expressions.
- Produces: explicit static rejection for concrete non-boolean `if`, `while`,
  and `seal until` conditions.

- [x] **Step 1: Add failing static examples**

Add checked examples requiring numeric control-flow conditions to produce a
condition-specific static error.

- [x] **Step 2: Add failing source diagnostic examples**

Add rendered pipeline examples for numeric `if`, `while`, and `seal until`
conditions.

- [x] **Step 3: Implement condition compatibility**

Accept `bool` and `unknown` condition types while rejecting concrete `num` and
`unit` conditions with a `conditionMismatch` error.

- [x] **Step 4: Document condition checking**

Update the formal-core static checker notes to record boolean condition
requirements.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 57: Lean Logical Operator Operand Checking

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: static expression typing and existing binary operand mismatch
  diagnostics.
- Produces: explicit rejection for concrete non-boolean operands to `&&` and
  `||`, while preserving `unknown` compatibility for unannotated calls.

- [x] **Step 1: Add failing static examples**

Add checked examples requiring `num && bool` and `bool || num` to fail with
`operandMismatch`.

- [x] **Step 2: Add failing source diagnostic examples**

Add rendered source diagnostics for invalid logical operands.

- [x] **Step 3: Implement logical operand compatibility**

Introduce boolean-like operand checking for `&&` and `||`, accepting only
`bool` and `unknown`.

- [x] **Step 4: Document logical operator checking**

Update the formal-core static checker notes to distinguish arithmetic,
comparison, and logical operand validation.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 58: Lean Equality Operand Compatibility

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: static expression typing and existing binary operand mismatch
  diagnostics.
- Produces: explicit rejection for concrete incompatible operands to `==` and
  `!=`, while preserving `unknown` compatibility for unannotated calls.

- [x] **Step 1: Add failing static examples**

Add checked examples requiring `num == bool` and `bool != num` to fail with
`operandMismatch`.

- [x] **Step 2: Add failing source diagnostic examples**

Add rendered source diagnostics for invalid equality operands.

- [x] **Step 3: Implement equality compatibility**

Accept equality when operand types match or either side is `unknown`; reject
known mixed-type equality.

- [x] **Step 4: Document equality checking**

Update the formal-core static checker notes to mention equality compatibility.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 59: Lean Unary Not Operand Checking

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: static expression typing and existing unary mismatch diagnostics.
- Produces: explicit rejection for concrete non-boolean operands to unary `!`,
  while preserving `unknown` compatibility for unannotated calls.

- [x] **Step 1: Add failing static examples**

Add a checked example requiring `!1` to fail with `unaryMismatch`.

- [x] **Step 2: Add failing source diagnostic examples**

Add a rendered source diagnostic for an invalid unary `!` operand.

- [x] **Step 3: Implement unary not compatibility**

Accept unary `!` only for `bool` and `unknown` operand types.

- [x] **Step 4: Document unary checking**

Update the formal-core static checker notes to include unary logical operand
validation.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 60: Lean Nested Function Declaration Diagnostics

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: static statement checking, function signature collection, and
  source diagnostic rendering.
- Produces: explicit static rejection for `fn` declarations nested inside
  blocks, loops, or other functions.

- [x] **Step 1: Add failing static examples**

Add checked examples requiring nested `fn` declarations to produce a
function-specific static error.

- [x] **Step 2: Add failing source diagnostic examples**

Add a rendered source diagnostic for a nested function declaration.

- [x] **Step 3: Track top-level scope**

Extend static checking scope so only direct top-level statements may declare
functions.

- [x] **Step 4: Document top-level function rule**

Update the formal-core static checker notes to record that proof-core function
declarations are top-level only.

- [x] **Step 5: Verify**

Run: `lake build`
Expected: PASS.

### Task 78: Lean If Statement Big-Step Semantics

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core `Stmt.ifThenElse`, `truthy`, and existing block-flow
  relation.
- Produces: checked big-step statement semantics for `if`/`else` branches,
  including no-`else` falsey conditions.

- [x] **Step 1: Add failing checked examples**

Add examples proving a truthy condition steps through the then branch, a falsey
condition with `else` steps through the else branch, and a falsey condition
without `else` produces `unit` with the original environment.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `StepStmt.ifTrue`, `StepStmt.ifFalseSome`, and
`StepStmt.ifFalseNone` do not exist.

- [x] **Step 3: Add mutual statement/block semantics**

Refactor `StepStmt` and `StepBlock` into a mutual inductive relation so an
`if` statement can delegate the selected branch to `StepBlock` while preserving
the existing block sequencing and early-flow rules.

- [x] **Step 4: Document structured statement semantics**

Record that proof-core statement semantics now evaluate selected `if` branches
as blocks and treat missing `else` as `unit`.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 79: Lean Executable If/Else Core Semantics

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: bounded `evalExprWithFns`, `execStmtWithFns`, `execBlockWithFns`,
  and `Stmt.ifThenElse`.
- Produces: executable bounded core support for running selected `if`/`else`
  branches, including no-`else` falsey conditions.

- [x] **Step 1: Add failing executable examples**

Add checked `native_decide` examples proving `execBlockWithFns` runs the then
branch for true conditions, runs the else branch for false conditions, and
returns `unit` with the original environment when false without `else`.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `execStmtWithFns` falls through to `none` for
`Stmt.ifThenElse`.

- [x] **Step 3: Implement bounded if execution**

Evaluate the condition with the remaining fuel, execute the selected branch via
`execBlockWithFns`, and return `unit` without environment changes when no else
branch is present.

- [x] **Step 4: Document executable structured semantics**

Record the bounded executor's `if`/`else` behavior separately from the Prop
big-step relation.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 80: Lean Executable While Core Semantics

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: bounded `evalExprWithFns`, `execStmtWithFns`, `execBlockWithFns`,
  `truthy`, assignments, `break`, and `continue`.
- Produces: executable bounded core support for `Stmt.while` with normal exit,
  repeated state updates, break exit, continue iteration, and preserved return
  flow.

- [x] **Step 1: Add failing executable examples**

Add checked `native_decide` examples proving false conditions leave the
environment unchanged and return `unit`, repeated assignment reaches the loop
bound, `break` exits while skipping later body statements, and `continue`
starts the next iteration.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `execStmtWithFns` falls through to `none` for
`Stmt.while`.

- [x] **Step 3: Implement bounded while execution**

Evaluate the condition with remaining fuel, execute the body as a block when
truthy, recurse with decreased fuel after ordinary body values or `continue`,
return `unit` on false conditions and `break`, and preserve `return` flow.

- [x] **Step 4: Document executable loop semantics**

Record the bounded executor's `while` behavior, including `break`, `continue`,
and `return` handling.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 81: Lean Executable For-Range Core Semantics

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: bounded `execStmtWithFns`, `execBlockWithFns`, integer
  `Stmt.forRange` bounds, assignment, `break`, and `continue`.
- Produces: executable bounded core support for ascending `forRange`
  iteration, iterator rebinding, normal completion, break exit, continue
  iteration, and preserved return flow.

- [x] **Step 1: Add failing executable examples**

Add checked `native_decide` examples proving `0..3` accumulation, empty range
iterator binding, `break` exit with the current iterator value, and
`continue` advancing to the next integer.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `execStmtWithFns` falls through to `none` for
`Stmt.forRange`.

- [x] **Step 3: Implement bounded for-range execution**

Bind the iterator to `start` while `start < stop`, execute the body as a block,
recurse with `start + 1` after ordinary body values or `continue`, return
`unit` on `break`, preserve `return` flow, and bind the iterator to `stop` on
normal completion.

- [x] **Step 4: Document executable for-range semantics**

Record the bounded executor's ascending integer-range behavior, including
iterator rebinding and control-flow handling.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 82: Lean Executable Seal Core Semantics

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: bounded `execStmtWithFns`, `execBlockWithFns`, optional
  `Stmt.seal` conditions, `truthy`, assignment, `break`, and `continue`.
- Produces: executable bounded core support for conditional `seal until` and
  bare `seal`, including pre-check termination, bounded repetition,
  break/continue handling, and preserved return flow.

- [x] **Step 1: Add failing executable examples**

Add checked `native_decide` examples proving a truthy initial `seal until`
condition skips the body, a conditional seal increments until its condition is
truthy, bare seal exits on `break`, and `continue` starts the next iteration.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `execStmtWithFns` falls through to `none` for
`Stmt.seal`.

- [x] **Step 3: Implement bounded seal execution**

For `seal until`, evaluate the condition before each iteration and stop with
`unit` when truthy. For bare `seal`, execute the body until fuel exhaustion or
control flow exits. In both forms, recurse after ordinary body values or
`continue`, return `unit` on `break`, and preserve `return` flow.

- [x] **Step 4: Document executable seal semantics**

Record the bounded executor's conditional and bare seal behavior, including
pre-checking, fuel bounding, and control-flow handling.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 83: Lean While Statement Big-Step Semantics

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core `Stmt.while`, `truthy`, expression evaluation, and the
  existing mutually defined `StepStmt`/`StepBlock` relation.
- Produces: checked big-step statement semantics for false loop exit, ordinary
  value-producing loop bodies, return propagation, break exit, and continue
  recursion.

- [x] **Step 1: Add failing checked examples**

Add examples proving a falsey `while` condition exits with `unit`, a single
ordinary iteration can recurse to a false exit, and a loop body `break` exits
the loop after preserving body state changes.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `StepStmt.whileFalse`, `StepStmt.whileValue`, and
`StepStmt.whileBreak` do not exist.

- [x] **Step 3: Add while constructors**

Extend the mutual `StepStmt`/`StepBlock` relation with constructors for false
exit, ordinary body recursion, return propagation, break exit, and continue
recursion.

- [x] **Step 4: Document big-step while semantics**

Record the Prop-level `while` rules separately from the bounded executable
core.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 84: Lean For-Range Statement Big-Step Semantics

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core `Stmt.forRange`, integer bounds, iterator binding, and
  the existing mutually defined `StepStmt`/`StepBlock` relation.
- Produces: checked big-step statement semantics for completed ranges,
  ordinary iteration recursion, return propagation, break exit, and continue
  recursion.

- [x] **Step 1: Add failing checked examples**

Add examples proving an empty range binds the iterator to the stop value and
returns `unit`, a one-iteration range can recurse to completion, and `break`
exits the range after preserving the current iterator environment.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `StepStmt.forDone`, `StepStmt.forValue`, and
`StepStmt.forBreak` do not exist.

- [x] **Step 3: Add for-range constructors**

Extend the mutual `StepStmt`/`StepBlock` relation with constructors for range
completion, ordinary body recursion, return propagation, break exit, and
continue recursion.

- [x] **Step 4: Document big-step for-range semantics**

Record the Prop-level `forRange` rules separately from the bounded executable
core.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 85: Lean Seal Statement Big-Step Semantics

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core `Stmt.seal`, optional conditions, `truthy`, expression
  evaluation, and the existing mutually defined `StepStmt`/`StepBlock`
  relation.
- Produces: checked big-step statement semantics for conditional stop,
  conditional iteration, bare seal iteration, return propagation, break exit,
  and continue recursion.

- [x] **Step 1: Add failing checked examples**

Add examples proving a truthy initial `seal until` condition skips the body, a
conditional seal can perform one ordinary iteration and recurse to stop, and a
bare seal body `break` exits the loop after preserving body state changes.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `StepStmt.sealUntilDone`,
`StepStmt.sealUntilValue`, and `StepStmt.sealBreak` do not exist.

- [x] **Step 3: Add seal constructors**

Extend the mutual `StepStmt`/`StepBlock` relation with constructors for
conditional stop, conditional value/return/break/continue behavior, and bare
seal value/return/break/continue behavior.

- [x] **Step 4: Document big-step seal semantics**

Record the Prop-level `seal` rules separately from the bounded executable
core.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 86: Lean Function Declaration Env-Only Big-Step Semantics

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core `Stmt.fnDecl` and the existing env-only
  `StepStmt`/`StepBlock` relation.
- Produces: checked big-step statement semantics for function declarations as
  unit-producing statements with no variable-environment effect.

- [x] **Step 1: Add failing checked example**

Add an example proving a single `fnDecl` statement steps as a block to
`Flow.value Value.unit` without changing the variable environment.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `StepStmt.fnDecl` does not exist.

- [x] **Step 3: Add function declaration constructor**

Extend `StepStmt` with a `fnDecl` constructor that preserves the variable
environment and produces `unit`.

- [x] **Step 4: Document the relation boundary**

Record that this is an env-only statement relation rule, while full
function-environment behavior remains modeled by bounded executable `FnEnv`
semantics until a future Prop relation carries function bindings explicitly.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 87: Prop-Level Function Environment Semantics Slice

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `FnEnv`, `Function`, `bindParams`, proof-core expressions and
  statements, and the executable function semantics as the behavioral guide.
- Produces: initial Prop-level relations for function-aware expression
  evaluation, argument evaluation, statement stepping, and block stepping.

- [x] **Step 1: Add failing checked example**

Add an example proving a block that declares `id(x)`, calls it with `3`, and
binds the result to `y` while threading the function environment.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `StepBlockWithFns` and the related Prop relation names
do not exist.

- [x] **Step 3: Add minimal function-aware Prop relations**

Define `EvalExprWithFnsRel`, `EvalArgsWithFnsRel`, `StepStmtWithFns`, and
`StepBlockWithFns` for literals, variables, function calls, `let`, expression
statements, returns, function declarations, and declaration sequencing.

- [x] **Step 4: Document relation coverage**

Record that the new Prop-level `FnEnv` slice covers declaration binding, call
argument evaluation, parameter frame binding, return/value call results, and
basic block sequencing, while structured control flow remains future work for
this relation.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 88: Prop-Level Function Environment If Semantics

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `EvalExprWithFnsRel`, `StepStmtWithFns`,
  `StepBlockWithFns`, `truthy`, and proof-core `Stmt.ifThenElse`.
- Produces: checked function-aware big-step semantics for true branches,
  false branches with `else`, and false branches without `else`.

- [x] **Step 1: Add failing checked examples**

Add examples proving `StepBlockWithFns` behavior for truthy `if`, falsey
`if` with `else`, and falsey `if` without `else`.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `StepStmtWithFns.ifTrue`,
`StepStmtWithFns.ifFalseSome`, and `StepStmtWithFns.ifFalseNone` do not exist.

- [x] **Step 3: Add function-aware if constructors**

Extend `StepStmtWithFns` with constructors that evaluate the condition through
`EvalExprWithFnsRel`, select the correct branch, and thread both variable and
function environments through `StepBlockWithFns`.

- [x] **Step 4: Document coverage**

Record that the Prop-level `FnEnv` relation now covers structured
`if`/`else`, while loops and other structured control forms remain future work
for this relation.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 89: Prop-Level Function Environment While Semantics

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `EvalExprWithFnsRel`, `StepStmtWithFns`,
  `StepBlockWithFns`, `truthy`, `Flow.break`, `Flow.continue`, and proof-core
  `Stmt.while`.
- Produces: checked function-aware big-step semantics for falsey loop
  completion, ordinary loop recursion, return propagation, break exit,
  continue recursion, and block propagation of break/continue.

- [x] **Step 1: Add failing checked examples**

Add examples proving `StepBlockWithFns` behavior for a falsey `while`, a
one-iteration value-producing `while`, and a `while` body that exits through
`break`.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `StepStmtWithFns.whileFalse`,
`StepStmtWithFns.whileValue`, and `StepStmtWithFns.whileBreak` do not exist.

- [x] **Step 3: Add function-aware while and control-flow constructors**

Extend `StepStmtWithFns` with `whileFalse`, `whileValue`, `whileReturn`,
`whileBreak`, `whileContinue`, `break`, and `continue`. Extend
`StepBlockWithFns` with `consBreak` and `consContinue`.

- [x] **Step 4: Document coverage**

Record that the Prop-level `FnEnv` relation now covers `while` and
break/continue block propagation, while `forRange`, `seal`, and assignment
remain future work for this relation.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 90: Prop-Level Function Environment Assignment Semantics

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `EvalExprWithFnsRel`, `StepStmtWithFns`, `Env.assign`, and
  proof-core `Stmt.assign`.
- Produces: checked function-aware big-step semantics for assignment to an
  existing variable binding while preserving the function environment.

- [x] **Step 1: Add failing checked examples**

Add examples proving direct assignment followed by variable lookup, and a
`while` body that mutates an existing condition variable through assignment.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `StepStmtWithFns.assign` does not exist.

- [x] **Step 3: Add function-aware assignment constructor**

Extend `StepStmtWithFns` with an `assign` constructor that evaluates the RHS
through `EvalExprWithFnsRel`, applies `Env.assign`, preserves `FnEnv`, and
returns the assigned value.

- [x] **Step 4: Document coverage**

Record that the Prop-level `FnEnv` relation now covers assignment, while
`forRange` and `seal` remain future work for this relation.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 91: Prop-Level Function Environment For-Range Semantics

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `StepStmtWithFns`, `StepBlockWithFns`, `Env.bind`,
  `Flow.break`, `Flow.continue`, and proof-core `Stmt.forRange`.
- Produces: checked function-aware big-step semantics for completed ranges,
  ordinary iteration recursion, return propagation, break exit, and continue
  recursion.

- [x] **Step 1: Add failing checked examples**

Add examples proving `StepBlockWithFns` behavior for an empty range, a
one-iteration range with assignment from the iterator, and a range body that
exits through `break`.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `StepStmtWithFns.forDone`,
`StepStmtWithFns.forValue`, and `StepStmtWithFns.forBreak` do not exist.

- [x] **Step 3: Add function-aware for-range constructors**

Extend `StepStmtWithFns` with `forDone`, `forValue`, `forReturn`,
`forBreak`, and `forContinue`, threading both variable and function
environments through the body and recursive step.

- [x] **Step 4: Document coverage**

Record that the Prop-level `FnEnv` relation now covers `forRange`, leaving
`seal` as the remaining structured statement gap for this relation.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 92: Prop-Level Function Environment Seal Semantics

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `EvalExprWithFnsRel`, `StepStmtWithFns`,
  `StepBlockWithFns`, `truthy`, `Flow.break`, `Flow.continue`, and
  proof-core `Stmt.seal`.
- Produces: checked function-aware big-step semantics for conditional seal
  stop, conditional iteration, bare seal iteration, return propagation, break
  exit, and continue recursion.

- [x] **Step 1: Add failing checked examples**

Add examples proving `StepBlockWithFns` behavior for a truthy initial
`seal until` condition, a conditional seal that performs one ordinary
iteration and recurses to stop, and a bare seal body that exits through
`break`.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `StepStmtWithFns.sealUntilDone`,
`StepStmtWithFns.sealUntilValue`, and `StepStmtWithFns.sealBreak` do not
exist.

- [x] **Step 3: Add function-aware seal constructors**

Extend `StepStmtWithFns` with conditional seal done/value/return/break/continue
constructors and bare seal value/return/break/continue constructors, threading
both variable and function environments through body and recursive steps.

- [x] **Step 4: Document coverage**

Record that the Prop-level `FnEnv` relation now covers both conditional and
bare `seal` control flow.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 93: Prop-Level Function Environment Unary/Binary Expressions

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `EvalExprWithFnsRel`, `evalUnOp`, `evalBinOp`,
  proof-core `Expr.unary`, and proof-core `Expr.binary`.
- Produces: checked function-aware expression semantics for unary operators
  and binary operators, including arithmetic and comparisons.

- [x] **Step 1: Add failing checked examples**

Add examples proving `StepBlockWithFns` expression statements for numeric
addition, numeric comparison, and boolean negation.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `EvalExprWithFnsRel.binary` and
`EvalExprWithFnsRel.unary` do not exist.

- [x] **Step 3: Add function-aware expression constructors**

Extend `EvalExprWithFnsRel` with `unary` and `binary` constructors that
evaluate subexpressions and use the existing operator evaluator helpers.

- [x] **Step 4: Document coverage**

Record that the Prop-level `FnEnv` expression relation now covers unary and
binary operators through the shared operator evaluators.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 94: Prop-Level Function Environment List/Index Expressions

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `EvalExprWithFnsRel`, `EvalArgsWithFnsRel`, `evalIndex`,
  proof-core `Expr.list`, and proof-core `Expr.index`.
- Produces: checked function-aware expression semantics for list construction
  and successful list/string indexing.

- [x] **Step 1: Add failing checked examples**

Add examples proving `StepBlockWithFns` expression statements for a mixed list
literal and a successful list index.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `EvalExprWithFnsRel.list` and
`EvalExprWithFnsRel.index` do not exist.

- [x] **Step 3: Add function-aware list/index constructors**

Extend `EvalExprWithFnsRel` with `list`, backed by `EvalArgsWithFnsRel`, and
`index`, backed by the existing `evalIndex` helper.

- [x] **Step 4: Document coverage**

Record that the Prop-level `FnEnv` expression relation now covers list
construction and indexing.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 95: Prop-Level Function Environment Field/Method Expressions

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `EvalExprWithFnsRel`, `EvalArgsWithFnsRel`, `evalField`,
  `evalMethod`, proof-core `Expr.field`, and proof-core `Expr.method`.
- Produces: checked function-aware expression semantics for supported field
  access and pure method calls.

- [x] **Step 1: Add failing checked examples**

Add examples proving `StepBlockWithFns` expression statements for list
`.length` field access and string `.len()` method call.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `EvalExprWithFnsRel.field` and
`EvalExprWithFnsRel.method` do not exist.

- [x] **Step 3: Add function-aware field/method constructors**

Extend `EvalExprWithFnsRel` with `field`, backed by the existing `evalField`
helper, and `method`, backed by `EvalArgsWithFnsRel` plus `evalMethod`.

- [x] **Step 4: Document coverage**

Record that the Prop-level `FnEnv` expression relation now covers field access
and pure method calls through the shared evaluator helpers.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 96: Prop-Level Function Environment Implicit Call Result Example

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `EvalExprWithFnsRel.callValue`, `StepBlockWithFns`, function
  declarations, positional calls, and final expression statement semantics.
- Produces: checked proof coverage for function calls whose body returns an
  ordinary final expression value rather than an explicit `return`.

- [x] **Step 1: Add checked example**

Add an example proving a block that declares `one() { 1 }`, calls it, and binds
the implicit final-expression result to `y`.

- [x] **Step 2: Verify**

Run: `lake build`
Expected: PASS, proving the existing `callValue` constructor covers implicit
final-expression results.

- [x] **Step 3: Document coverage**

Record that checked Prop-level `FnEnv` call witnesses cover both explicit
`return` and implicit final-expression call results.

- [x] **Step 4: Full verification**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 97: Base Prop-to-Executable Expression Correspondence Witnesses

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `EvalExprWithFnsRel` base expression constructors and the bounded
  executable `evalExprWithFns`.
- Produces: checked concrete witnesses that selected Prop expression facts for
  numeric literals, booleans, and variables agree with executable evaluation.

- [x] **Step 1: Add failing checked examples**

Add examples using intended correspondence witness names before defining them.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the base correspondence witness names do not exist.

- [x] **Step 3: Add concrete correspondence witnesses**

Define witnesses for numeric literals, booleans, and variables. Keep them
concrete because `evalExprWithFns` is a bounded partial executable evaluator
that reduces through computation rather than ordinary theorem unfolding.

- [x] **Step 4: Document scope**

Record that these are initial concrete correspondence witnesses, while full
inductive correspondence remains future work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 98: Compound Prop-to-Executable Expression Correspondence Witnesses

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `EvalExprWithFnsRel` compound expression constructors and the
  bounded executable `evalExprWithFns`.
- Produces: checked concrete witnesses that selected Prop expression facts for
  binary arithmetic, unary boolean negation, and list construction agree with
  executable evaluation.

- [x] **Step 1: Add failing checked examples**

Add examples using intended correspondence witness names before defining them.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the compound correspondence witness names do not exist.

- [x] **Step 3: Add concrete correspondence witnesses**

Define witnesses for binary addition, unary boolean negation, and list
construction. Keep them concrete for the same reason as the base witnesses:
`evalExprWithFns` is a bounded partial executable evaluator.

- [x] **Step 4: Document scope**

Record that correspondence witness coverage now includes selected compound
expression constructors while full inductive correspondence remains future work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 99: Accessor Prop-to-Executable Expression Correspondence Witnesses

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `EvalExprWithFnsRel.index`, `EvalExprWithFnsRel.field`,
  `EvalExprWithFnsRel.method`, and the bounded executable `evalExprWithFns`.
- Produces: checked concrete witnesses that selected Prop expression facts for
  list indexing, list `length` fields, and string `len()` methods agree with
  executable evaluation.

- [x] **Step 1: Inspect accessor helper semantics**

Confirm that `evalIndex` supports list indexing, `evalField` supports
`length`, and `evalMethod` supports zero-argument `len`.

- [x] **Step 2: Add failing checked examples**

Add examples using intended correspondence witness names before defining them.

- [x] **Step 3: Verify red**

Run: `lake build`
Expected: FAIL because the accessor correspondence witness names do not exist.

- [x] **Step 4: Add concrete correspondence witnesses**

Define witnesses for list indexing, list `length`, and string `len()`.
Accessor examples over list literals use fuel `3` because the accessor
evaluation consumes one step for the accessor and one for the list target
before evaluating the list elements.

- [x] **Step 5: Document scope**

Record that correspondence witness coverage now includes selected accessor and
method-call expression constructors while full inductive correspondence remains
future work.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 100: Function Call Prop-to-Executable Expression Correspondence Witnesses

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `EvalExprWithFnsRel.callReturn`, `EvalExprWithFnsRel.callValue`,
  `StepBlockWithFns`, function environments, and the bounded executable
  `evalExprWithFns`.
- Produces: checked concrete witnesses that selected Prop expression facts for
  explicit-return and implicit-final-expression function calls agree with
  executable evaluation.

- [x] **Step 1: Inspect call semantics**

Confirm the relation and executable both handle explicit `return` results and
ordinary function-body values as call expression results.

- [x] **Step 2: Add failing checked examples**

Add examples using intended call correspondence witness names before defining
them.

- [x] **Step 3: Verify red**

Run: `lake build`
Expected: FAIL because the function-call correspondence witness names do not
exist.

- [x] **Step 4: Add concrete correspondence witnesses**

Define witnesses for `id(x) { return x }` and `one() { 1 }`. Use fuel `3` so
the call, function-body statement, and returned/body expression all have fuel.

- [x] **Step 5: Document scope**

Record that correspondence witness coverage now includes both explicit-return
and implicit-final-expression function calls while full inductive
correspondence remains future work.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 101: Base Statement Prop-to-Executable Correspondence Witnesses

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `StepStmtWithFns`, `execStmtWithFns`, function-aware expression
  witnesses, and function environments.
- Produces: checked concrete witnesses that selected Prop statement facts for
  `let`, `fn` declaration, and `return` statements agree with executable
  statement evaluation through projected observable results.

- [x] **Step 1: Inspect statement executor semantics**

Confirm `execStmtWithFns` evaluates `let` and `return expr` with one lower
expression fuel, while `fnDecl` immediately adds a function binding and returns
`unit`.

- [x] **Step 2: Add failing checked examples**

Add examples using intended statement correspondence witness names before
defining them.

- [x] **Step 3: Verify red**

Run: `lake build`
Expected: FAIL because the statement correspondence witness names do not exist.

- [x] **Step 4: Add concrete statement witnesses**

Define witnesses for `let x = 7`, `fn id(x)`, and `return x`. Compare projected
executable results so the checks cover observable environments and flow without
requiring decidable equality over function body payloads.

- [x] **Step 5: Document scope**

Record that initial statement correspondence witnesses cover selected
function-aware statements and projected executable outputs.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 102: Assignment and Control Statement Prop-to-Executable Witnesses

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `StepStmtWithFns.assign`, `StepStmtWithFns.expr`,
  `StepStmtWithFns.retNone`, `StepStmtWithFns.break`,
  `StepStmtWithFns.continue`, and the bounded executable `execStmtWithFns`.
- Produces: checked concrete witnesses that selected assignment, expression,
  return-none, break, and continue statement facts agree with executable
  statement evaluation through projected observable results.

- [x] **Step 1: Inspect existing projection pattern**

Confirm the Task 101 `(env, flow)` projection shape works for assignment,
expression statements, and direct control-flow statements.

- [x] **Step 2: Add failing checked examples**

Add examples using intended witness names before defining them.

- [x] **Step 3: Verify red**

Run: `lake build`
Expected: FAIL because the assignment/control statement witness names do not
exist.

- [x] **Step 4: Add concrete witnesses**

Define witnesses for assigning `x = 9`, evaluating `true` as an expression
statement, `return none`, `break`, and `continue`.

- [x] **Step 5: Document scope**

Record that statement correspondence witness coverage now includes assignment,
expression statements, and direct control-flow statements.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 103: Base Block Prop-to-Executable Correspondence Witnesses

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `StepBlockWithFns`, `execBlockWithFns`, and the projected
  statement executable witness pattern.
- Produces: checked concrete witnesses that selected Prop block facts for empty
  blocks, single-statement blocks, value sequencing, and early
  `return`/`break`/`continue` propagation agree with executable block
  evaluation through projected observable results.

- [x] **Step 1: Inspect block executor semantics**

Confirm `execBlockWithFns` returns `unit` for empty blocks, delegates singleton
blocks to `execStmtWithFns`, sequences after `Flow.value`, and stops early on
`return`, `break`, or `continue`.

- [x] **Step 2: Add failing checked examples**

Add examples using intended block correspondence witness names before defining
them.

- [x] **Step 3: Verify red**

Run: `lake build`
Expected: FAIL because the block correspondence witness names do not exist.

- [x] **Step 4: Add concrete block witnesses**

Define witnesses for an empty block, a single boolean expression block, `let`
then `var` value sequencing, and early `return`, `break`, and `continue`.

- [x] **Step 5: Document scope**

Record that block correspondence witness coverage now includes selected block
forms and projected executable outputs.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 104: If Statement Prop-to-Executable Correspondence Witnesses

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `StepStmtWithFns.ifTrue`, `StepStmtWithFns.ifFalseSome`,
  `StepStmtWithFns.ifFalseNone`, `StepBlockWithFns`, and the bounded
  executable `execStmtWithFns`.
- Produces: checked concrete witnesses that selected Prop `if` statement facts
  agree with executable statement evaluation through projected observable
  results.

- [x] **Step 1: Inspect if executor semantics**

Confirm `execStmtWithFns` evaluates the condition, executes the selected branch
as a block when present, and returns `unit` for a falsey condition without an
`else`.

- [x] **Step 2: Add failing checked examples**

Add examples using intended `if` witness names before defining them.

- [x] **Step 3: Verify red**

Run: `lake build`
Expected: FAIL because the `if` witness names do not exist.

- [x] **Step 4: Add concrete witnesses**

Define witnesses for true-branch selection, false-branch `else` selection, and
false-without-else behavior.

- [x] **Step 5: Document scope**

Record that structured statement witness coverage now includes selected `if`
branching cases.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 105: Non-Recursive While Prop-to-Executable Correspondence Witnesses

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `StepStmtWithFns.whileFalse`, `StepStmtWithFns.whileReturn`,
  `StepStmtWithFns.whileBreak`, `StepBlockWithFns`, and the bounded executable
  `execStmtWithFns`.
- Produces: checked concrete witnesses that selected non-recursive Prop `while`
  statement facts agree with executable statement evaluation through projected
  observable results.

- [x] **Step 1: Inspect while semantics**

Confirm false conditions exit immediately, body `return` propagates, and body
`break` exits as `unit`; leave recursive value and continue cases for later.

- [x] **Step 2: Add failing checked examples**

Add examples using intended while witness names before defining them.

- [x] **Step 3: Verify red**

Run: `lake build`
Expected: FAIL because the while witness names do not exist.

- [x] **Step 4: Add concrete witnesses**

Define witnesses for while-false, while-return, and while-break behavior.

- [x] **Step 5: Document scope**

Record that loop correspondence witness coverage now includes selected
non-recursive while exits.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 106: Non-Recursive ForRange Prop-to-Executable Correspondence Witnesses

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `StepStmtWithFns.forDone`, `StepStmtWithFns.forReturn`,
  `StepStmtWithFns.forBreak`, `StepBlockWithFns`, and the bounded executable
  `execStmtWithFns`.
- Produces: checked concrete witnesses that selected non-recursive Prop
  `forRange` statement facts agree with executable statement evaluation through
  projected observable results.

- [x] **Step 1: Inspect forRange semantics**

Confirm completed ranges bind the iterator to the stop value, body `return`
propagates, and body `break` exits with `unit`; leave recursive value and
continue cases for later.

- [x] **Step 2: Add failing checked examples**

Add examples using intended forRange witness names before defining them.

- [x] **Step 3: Verify red**

Run: `lake build`
Expected: FAIL because the forRange witness names do not exist.

- [x] **Step 4: Add concrete witnesses**

Define witnesses for for-done, for-return, and for-break behavior.

- [x] **Step 5: Document scope**

Record that loop correspondence witness coverage now includes selected
non-recursive forRange exits.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 107: Recursive ForRange Prop-to-Executable Correspondence Witnesses

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `StepStmtWithFns.forValue`, `StepStmtWithFns.forContinue`,
  `StepStmtWithFns.forDone`, `StepBlockWithFns`, and the bounded executable
  `execStmtWithFns`.
- Produces: checked concrete witnesses that selected recursive Prop `forRange`
  statement facts agree with executable statement evaluation through projected
  observable results.

- [x] **Step 1: Inspect recursive forRange semantics**

Confirm ordinary body values and body `continue` both advance to the next range
value and then recurse through the executable evaluator.

- [x] **Step 2: Add failing checked examples**

Add examples using intended recursive forRange witness names before defining
them.

- [x] **Step 3: Verify red**

Run: `lake build`
Expected: FAIL because the recursive forRange witness names do not exist.

- [x] **Step 4: Add concrete witnesses**

Define witnesses for value-body recursion into completion and continue-body
recursion into completion.

- [x] **Step 5: Document scope**

Record that loop correspondence witness coverage now includes selected
recursive forRange value and continue behavior.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 108: Seal-Until Prop-to-Executable Correspondence Witnesses

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `StepStmtWithFns.sealUntilDone`,
  `StepStmtWithFns.sealUntilValue`, `StepStmtWithFns.sealUntilBreak`,
  `StepBlockWithFns`, and the bounded executable `execStmtWithFns`.
- Produces: checked concrete witnesses that selected Prop `seal until`
  statement facts agree with executable statement evaluation through projected
  observable results.

- [x] **Step 1: Inspect seal semantics**

Confirm satisfied conditions exit immediately, ordinary body values recurse to
the next condition check, and body `break` exits with `unit`.

- [x] **Step 2: Add failing checked examples**

Add examples using intended `seal until` witness names before defining them.

- [x] **Step 3: Verify red**

Run: `lake build`
Expected: FAIL because the `seal until` witness names do not exist.

- [x] **Step 4: Add concrete witnesses**

Define witnesses for already-done, value-body recursion into completion, and
break-body exit.

- [x] **Step 5: Document scope**

Record that loop correspondence witness coverage now includes selected
`seal until` exits and value recursion.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 109: Seal-Until Return and Continue Correspondence Witnesses

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `StepStmtWithFns.sealUntilReturn`,
  `StepStmtWithFns.sealUntilContinue`, `StepBlockWithFns`, and the bounded
  executable `execStmtWithFns`.
- Produces: checked concrete witnesses that selected Prop `seal until`
  `return` and `continue` facts agree with executable statement evaluation
  through projected observable results.

- [x] **Step 1: Inspect remaining conditional seal semantics**

Confirm body `return` propagates immediately and body `continue` rechecks the
condition using the body-updated environment.

- [x] **Step 2: Add failing checked examples**

Add examples using intended `seal until` return and continue witness names
before defining them.

- [x] **Step 3: Verify red**

Run: `lake build`
Expected: FAIL because the `seal until` return and continue witness names do
not exist.

- [x] **Step 4: Add concrete witnesses**

Define witnesses for return propagation and continue-driven condition recheck.

- [x] **Step 5: Document scope**

Record that conditional seal correspondence witness coverage includes selected
return and continue behavior.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 110: Bare Seal Prop-to-Executable Correspondence Witnesses

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `StepStmtWithFns.sealValue`, `StepStmtWithFns.sealReturn`,
  `StepStmtWithFns.sealBreak`, `StepBlockWithFns`, and the bounded executable
  `execStmtWithFns`.
- Produces: checked concrete witnesses that selected Prop bare `seal` statement
  facts agree with executable statement evaluation through projected observable
  results.

- [x] **Step 1: Inspect bare seal semantics**

Confirm bare `seal` repeats after ordinary body values, propagates body
`return`, and exits with `unit` on body `break`.

- [x] **Step 2: Add failing checked examples**

Add examples using intended bare `seal` witness names before defining them.

- [x] **Step 3: Verify red**

Run: `lake build`
Expected: FAIL because the bare `seal` witness names do not exist.

- [x] **Step 4: Add concrete witnesses**

Define witnesses for value-body recursion into a later break, direct return
propagation, and direct break exit.

- [x] **Step 5: Document scope**

Record that bare seal correspondence witness coverage includes selected value,
return, and break behavior.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 111: Bare Seal Continue Correspondence Witness

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `StepStmtWithFns.sealContinue`, `StepStmtWithFns.sealBreak`,
  `StepBlockWithFns`, and the bounded executable `execStmtWithFns`.
- Produces: a checked concrete witness that a selected Prop bare `seal`
  `continue` fact agrees with executable statement evaluation through projected
  observable results.

- [x] **Step 1: Inspect bare seal continue semantics**

Confirm body `continue` recurses into the next bare-seal iteration with the
body-updated environment.

- [x] **Step 2: Add a failing checked example**

Add an example using the intended bare `seal` continue witness name before
defining it.

- [x] **Step 3: Verify red**

Run: `lake build`
Expected: FAIL because the bare `seal` continue witness name does not exist.

- [x] **Step 4: Add concrete witness**

Define the witness for continue-driven recursion into a later break.

- [x] **Step 5: Document scope**

Record that bare seal correspondence witness coverage includes selected
continue behavior.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 112: Lean Parser Self Expression Support

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: lexer `TokenKind.self` and core variable expressions.
- Produces: parser support for `self` as `Expr.var "self"`, including postfix
  field access such as `self.length`.

- [x] **Step 1: Inspect Rust and Lean parser behavior**

Confirm the Rust parser treats `self` as an expression identifier while the Lean
parser tokenizes `self` but does not accept it as a primary expression.

- [x] **Step 2: Add failing checked examples**

Add checked parser examples for `self` and `self.length` before implementing
the parser support.

- [x] **Step 3: Verify red**

Run: `lake build`
Expected: FAIL because `self` is not yet parsed as an expression primary.

- [x] **Step 4: Implement parser support**

Treat `TokenKind.self` as `Expr.var "self"` in `parsePrimary` so existing
postfix parsing handles field and method forms.

- [x] **Step 5: Document scope**

Record that parser coverage includes `self` expression parsing.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 113: Self Static Diagnostic Span Support

**Files:**
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: parser support for `Expr.var "self"`, located lexer
  `TokenKind.self`, and static undeclared-variable diagnostics.
- Produces: positioned pipeline diagnostics for undeclared `self` references.

- [x] **Step 1: Inspect diagnostic span lookup**

Confirm `self` parses as a variable expression but static diagnostic span
lookup only matches identifier tokens.

- [x] **Step 2: Add a failing checked example**

Add a checked `checkSourceErrorString` example requiring undeclared `self` to
include the `self` source span.

- [x] **Step 3: Verify red**

Run: `lake build`
Expected: FAIL because the undeclared `self` diagnostic lacks a source span.

- [x] **Step 4: Implement matcher support**

Treat `TokenKind.self` as matching the variable name `self` in
`tokenMatchesIdent`.

- [x] **Step 5: Document scope**

Record that positioned static diagnostics handle reserved `self` variable
tokens.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 114: Flexible Postfix Field and Method Names

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: lexer keyword tokens for domain names such as `dim` and `cluster`.
- Produces: Lean parser support for reserved domain keyword tokens as postfix
  field and method names, matching the Rust parser's flexible identifier
  behavior.

- [x] **Step 1: Inspect Rust and Lean postfix parsing**

Confirm Rust accepts flexible identifiers after `.`, while Lean only accepts
ordinary identifier tokens for postfix field and method names.

- [x] **Step 2: Add failing checked examples**

Add parser examples for `self.dim` and `self.cluster()` before implementing the
helper.

- [x] **Step 3: Verify red**

Run: `lake build`
Expected: FAIL because reserved domain keyword tokens are not accepted after
`.`.

- [x] **Step 4: Implement flexible postfix helper**

Add `flexibleIdent?` and use it for postfix field and method names.

- [x] **Step 5: Document scope**

Record that parser coverage includes reserved domain keyword field/method names.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 115: Render Keyword Tokens in Parse Diagnostics

**Files:**
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Lean lexer keyword tokens and source-pipeline parse-error
  rendering.
- Produces: deterministic rendered names for domain, object/module, and other
  proof-core keyword tokens in parser diagnostics.

- [x] **Step 1: Add failing rendered-diagnostic examples**

Add `parseSourceErrorString` examples requiring `manifold` and `class` parse
errors to render those keyword names instead of the generic `token` fallback.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because keyword tokens without explicit `tokenString` cases
render as `token`.

- [x] **Step 3: Extend token rendering**

Add explicit `tokenString` cases for the remaining named proof-core lexer
keyword tokens.

- [x] **Step 4: Document diagnostic coverage**

Record that parser diagnostics render proof-core lexer keywords by name.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 116: Self Expression-Start Diagnostics

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: parser support for `TokenKind.self` as `Expr.var "self"` and
  parse diagnostic classification.
- Produces: expression-start classification that treats `self` as a valid
  expression start, matching the executable expression parser.

- [x] **Step 1: Add failing parser diagnostic example**

Add a checked `parseProgramDetailed` example for a malformed expression that
starts with `self`.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `startsExpr` does not classify `TokenKind.self` as a
valid expression start.

- [x] **Step 3: Update expression-start classifier**

Add `TokenKind.self` to `startsExpr`.

- [x] **Step 4: Document diagnostic boundary**

Record that parse diagnostic classification recognizes `self` expression
starts, while recursive failure locations remain future parser work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 117: Float Expression-Start Diagnostics

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: parser support for `TokenKind.float` as `Expr.float` and parse
  diagnostic classification.
- Produces: expression-start classification that treats decimal float literals
  as valid expression starts, matching the executable expression parser.

- [x] **Step 1: Add failing parser diagnostic example**

Add a checked `parseProgramDetailed` example for a malformed expression that
starts with a decimal float literal.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `startsExpr` does not classify `TokenKind.float` as a
valid expression start.

- [x] **Step 3: Update expression-start classifier**

Add `TokenKind.float` to `startsExpr`.

- [x] **Step 4: Document diagnostic boundary**

Record that parse diagnostic classification recognizes float literal
expression starts, while recursive failure locations remain future parser work.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 118: Keyword Function Call Parsing

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Lean lexer keyword tokens `embed` and `convergence`, Rust parser
  behavior for keyword calls, and pipeline static diagnostic span lookup.
- Produces: Lean parser support for `embed(...)` and `convergence(...)` as
  ordinary `Expr.call` nodes, expression-start classification for those
  keyword calls, and source spans for undeclared keyword-call functions.

- [x] **Step 1: Add failing parser and pipeline examples**

Add checked parser examples for `embed(1)` and `convergence(0.1)`, plus a
rendered static diagnostic for undeclared `embed`.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the Lean parser does not yet accept `embed` or
`convergence` keyword tokens as call expressions.

- [x] **Step 3: Implement keyword-call parser support**

Add a keyword-call helper, parse `embed(...)`/`convergence(...)` into
`Expr.call`, and classify those keyword tokens as expression starts.

- [x] **Step 4: Implement keyword-call diagnostic spans**

Match `embed` and `convergence` keyword tokens as function names in pipeline
static diagnostic span lookup.

- [x] **Step 5: Document scope**

Record keyword-call parser support and undeclared keyword-call diagnostic span
coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 119: Context-Sensitive Keyword Call Diagnostics

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 118 keyword-call parsing for `embed(...)` and
  `convergence(...)`, plus parse diagnostic expression-start classification.
- Produces: diagnostics that treat `embed`/`convergence` as valid expression
  starts only when followed by `(`, so bare keyword-call names are reported at
  the offending keyword token.

- [x] **Step 1: Add failing parser diagnostic examples**

Add checked `parseProgramDetailed` examples for `let x = embed` and
`let x = convergence`.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the keyword-call tokens are classified as expression
starts even when they are not followed by `(`.

- [x] **Step 3: Implement context-sensitive classification**

Add `startsExprPrefix` so `embed` and `convergence` only start expressions when
the next token is `(`, while existing token-level expression starts keep their
behavior.

- [x] **Step 4: Document diagnostic boundary**

Record that bare keyword-call names are malformed expression starts, while
keyword calls with `(` remain valid expression starts.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 120: Named Call and Method Arguments

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Rust parser named-argument behavior, Lean call/method expression
  syntax, static argument checking, evaluator argument evaluation, and VM
  argument lowering.
- Produces: proof-core `Arg` nodes that preserve positional and named call or
  method arguments, parser support for `name=value` arguments, and conservative
  source-order evaluation/checking/lowering of argument payloads.

- [x] **Step 1: Add failing parser examples**

Add checked parser examples for `embed(data, dim=3)` and
`self.cluster(axis=2)` that require named argument nodes.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `Arg` and named-argument parsing do not exist yet.

- [x] **Step 3: Extend core AST**

Add `Arg.positional` and `Arg.named`, change `Expr.call` and `Expr.method` to
store `List Arg`, and keep a coercion from `Expr` to positional `Arg` so
existing positional examples remain concise.

- [x] **Step 4: Parse named arguments**

Extend `parseArgList` to preserve `flexibleIdent = expr` as `Arg.named` and
ordinary expressions as `Arg.positional`.

- [x] **Step 5: Update consumers**

Update static checking, executable evaluation, relational witnesses, and stack
and frame VM compilation to traverse named argument payload expressions in
source order while preserving names in the AST.

- [x] **Step 6: Document semantics**

Record that named argument names are preserved, while current runtime binding
uses source-order argument values and arity.

- [x] **Step 7: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 121: Named Function Argument Binding

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 120 `Arg.named` syntax, function parameter lists, executable
  function evaluation, and frame VM call lowering.
- Produces: executable named function-call semantics where named arguments bind
  to matching parameters, plus frame compiler normalization that lowers named
  calls into the VM's positional call ABI.

- [x] **Step 1: Add failing named-binding examples**

Add checked examples for `pick(b=2, a=7)` in both direct function evaluation
and the checked source pipeline.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because named arguments are still bound by source order.

- [x] **Step 3: Implement evaluator binding**

Add `bindCallArgs` so positional calls delegate to the existing positional
binding path, while named calls bind argument values to parameter names and
reject unknown, duplicate, or incomplete bindings.

- [x] **Step 4: Normalize frame compiler calls**

Normalize named function-call arguments to parameter order before frame
bytecode generation so existing `CALL target arity` instructions keep their
positional ABI.

- [x] **Step 5: Document semantics**

Record that named function calls bind by parameter name and that the frame
compiler lowers them by normalizing call arguments to parameter order.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 122: Static Named Argument Diagnostics

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 121 named function-call runtime semantics, `FnSig`
  collection, detailed static checking, and source diagnostic rendering.
- Produces: static rejection for unknown or duplicate named function
  arguments, including deterministic rendered source ranges.

- [x] **Step 1: Add failing static and pipeline examples**

Add checked examples for `pick(c=1, a=2)` and `pick(a=1, a=2)` that expect
static named-argument diagnostics.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the named-argument diagnostic constructors and
validation do not exist.

- [x] **Step 3: Track function parameter names in signatures**

Extend `FnSig` with `params`, preserve parameter names during signature
collection and result inference, and keep existing arity/result behavior.

- [x] **Step 4: Validate named arguments in static calls**

Reject unknown named arguments and duplicate named arguments before checking
argument payload expressions.

- [x] **Step 5: Render source diagnostics**

Add pipeline error strings and best-effort token spans for the offending named
argument.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 123: Basic Typed Function Parameters

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: lexer `:` tokens, proof-core function declarations, static
  function signatures, and checked source diagnostics.
- Produces: typed function declarations with basic `num`, `bool`, `str`, and
  `unit` parameter annotations, plus static argument type validation for calls.

- [x] **Step 1: Add failing parser/static/pipeline examples**

Add checked examples for parsing `fn id(x: num) { return x }` and rejecting
`id(true)` against a `num` parameter.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because typed function declarations and argument mismatch
diagnostics do not exist.

- [x] **Step 3: Add typed declaration AST and parser support**

Add `AnnTy`, `Stmt.fnDeclTyped`, typed parameter parsing, and parser examples.

- [x] **Step 4: Enforce typed parameter arguments statically**

Extend `FnSig` with parameter type vectors, bind typed function bodies with
annotated parameter types, and reject incompatible call argument types.

- [x] **Step 5: Preserve typed declarations through VM lowering**

Collect and normalize typed function declarations as ordinary frame functions
using their parameter names.

- [x] **Step 6: Document scope**

Record basic typed parameters while keeping richer type syntax and declared
return types as future work.

- [x] **Step 7: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 124: Declared Function Return Types

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 123 `AnnTy` and typed function parameters, parser colon
  tokens, static return inference, and source diagnostic rendering.
- Produces: typed function declarations with declared return annotations and
  detailed static rejection when the inferred body return type disagrees.

- [x] **Step 1: Add failing parser/static/pipeline examples**

Add checked examples for parsing `fn id(x: num): num { return x }` and
rejecting `fn bad(x: num): num { return true }`.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because return-typed function declarations and return mismatch
diagnostics do not exist.

- [x] **Step 3: Extend core and parser**

Add `Stmt.fnDeclTypedReturn` and parse `): type` after typed parameter lists.

- [x] **Step 4: Check declared return types**

Use the declared return annotation as the function result type and compare it
against inferred body return type in the detailed checker.

- [x] **Step 5: Preserve through runtime and VM surfaces**

Treat return-typed declarations as ordinary functions at runtime and during
frame VM function collection/normalization.

- [x] **Step 6: Render source diagnostics**

Add `returnMismatch` error strings and source spans pointing at the mismatched
return expression token.

- [x] **Step 7: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 125: List Type Annotations

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 123/124 annotation grammar, existing `Ty.list`, list literal
  inference, typed parameter checking, and source diagnostic rendering.
- Produces: parsed source annotations such as `xs: list[num]`, including nested
  list annotations, lowered into `Ty.list` for static argument validation.

- [x] **Step 1: Add failing parser/static/pipeline examples**

Add checked examples for parsing `fn first(xs: list[num]): num`, rejecting a
`list[bool]` call argument where `list[num]` is declared, and rendering the
source diagnostic on the bad list literal.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `AnnTy.list` does not exist.

- [x] **Step 3: Extend annotation AST and parser**

Add `AnnTy.list` and parse recursive `list[...]` annotation syntax.

- [x] **Step 4: Lower annotations into static types**

Map `AnnTy.list elem` to `Ty.list (annTyToTy elem)` so existing call argument
compatibility checks handle list annotations.

- [x] **Step 5: Render source diagnostics**

Treat `[` as the source token for concrete list type mismatches.

- [x] **Step 6: Document scope**

Record `list[...]` annotation syntax in the formal core grammar and parser
coverage notes.

- [x] **Step 7: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 126: Typed Local Declarations

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: existing `AnnTy`, `annTyToTy`, assignment compatibility, source
  static diagnostics, executable statement semantics, and frame VM local-slot
  compilation.
- Produces: source syntax `let name: type = expr`, static initializer
  validation against the declared type, and runtime/VM behavior equivalent to
  ordinary local binding after static acceptance.

- [x] **Step 1: Add failing parser/static/pipeline examples**

Add checked examples for parsing `let count: num = 1`, rejecting
`let count: num = true`, and rendering the source diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `Stmt.letDeclTyped` does not exist.

- [x] **Step 3: Extend core and parser**

Add `Stmt.letDeclTyped` and parse optional local declaration annotations before
the `=` token.

- [x] **Step 4: Enforce typed local initializers statically**

Lower the annotation with `annTyToTy`, check initializer compatibility, bind the
declared/refined type on success, and report `assignmentMismatch` on failure.

- [x] **Step 5: Preserve through runtime and VM surfaces**

Execute typed locals like ordinary `let` statements and compile them to the
same local-slot store sequence in the frame VM.

- [x] **Step 6: Document scope**

Record typed local syntax and static compatibility behavior in the formal core.

- [x] **Step 7: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 127: Type Annotation Parse Diagnostics

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: typed local declarations, typed function return syntax, existing
  `ParseContext`, `ParseError`, `parseAnnTy`, `parseTypedParamList`, and
  located source pipeline diagnostics.
- Produces: dedicated `expected type` parser diagnostics for malformed type
  annotation positions instead of generic statement/block/parameter failures.

- [x] **Step 1: Add failing parser/pipeline examples**

Add checked examples for `let count: = 1` and
`fn id(x: num): { return x }` that expect `ParseContext.typeAnnotation` and
user-facing `expected type` source errors.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `ParseContext.typeAnnotation` does not exist.

- [x] **Step 3: Add parser type context**

Extend `ParseContext` with `typeAnnotation` and render it as `type` in the
pipeline.

- [x] **Step 4: Classify malformed annotation sites**

Use `parseAnnTy` for `let name: ...` and `parseTypedParamList` plus
`parseAnnTy` for `fn name(params): ...` to classify malformed annotation starts
and place the diagnostic token after the colon.

- [x] **Step 5: Document diagnostics**

Record that malformed type annotations now report a dedicated parser context
and source span.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 128: Typed Parameter Parse Diagnostics

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Task 127 `ParseContext.typeAnnotation`, existing typed parameter
  syntax, `parseAnnTy`, and located parser diagnostics.
- Produces: dedicated `expected type` parser diagnostics when a function
  parameter colon is not followed by a valid type annotation.

- [x] **Step 1: Add failing parser/pipeline examples**

Add checked examples for `fn id(x:) { return x }` expecting
`ParseContext.typeAnnotation` and a source error at the `)` token.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because malformed typed parameter annotations still report a
generic parameter-list parse failure.

- [x] **Step 3: Detect malformed parameter annotation starts**

Add a diagnostic helper that scans function parameter tokens for
`identifier :` followed by a non-type token and returns the offset after the
colon.

- [x] **Step 4: Wire classification and source spans**

Classify those failures as `typeAnnotation` and reuse the returned offset for
both parser and pipeline diagnostics.

- [x] **Step 5: Document coverage**

Record malformed function parameter annotations in the parser diagnostic
coverage notes.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 129: Unit Literal Expression

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: existing `Value.unit`, `Ty.unit`, annotation syntax for `unit`,
  expression parser literals, static expression checking, runtime expression
  evaluation, and stack/frame VM expression compilation.
- Produces: source-level `unit` expression literal that parses, evaluates,
  type-checks as `unit`, and compiles to a pushed unit value.

- [x] **Step 1: Add failing parser/static/runtime/VM/pipeline examples**

Add checked examples for parsing `unit`, checking it as `Ty.unit`, evaluating
it to `Value.unit`, compiling it in stack and frame VM expression compilers,
and parsing `let done: unit = unit` through the source pipeline.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `Expr.unit` does not exist.

- [x] **Step 3: Add core unit expression**

Add `Expr.unit`, evaluate it to `Value.unit`, and add the function-aware
expression relation constructor.

- [x] **Step 4: Parse and check unit**

Parse identifier `unit` as the unit literal before generic identifier fallback
and type it as `Ty.unit` in both static checker paths.

- [x] **Step 5: Compile unit**

Compile unit literals to `Op.push Value.unit` and `FrameOp.push Value.unit` in
all expression compiler variants.

- [x] **Step 6: Document scope**

Record `unit` as a source literal, static type, and bytecode constant in the
formal core.

- [x] **Step 7: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 130: Implicit Unit Return Mismatch Spans

**Files:**
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: declared return type checking, implicit `Ty.unit` for functions
  with no return-producing body, `returnMismatch` diagnostics, and function-name
  span helpers.
- Produces: source-positioned diagnostics for non-unit functions that
  implicitly return unit because no return expression is present.

- [x] **Step 1: Add failing pipeline example**

Add a checked source example for `fn bad(): num { }` expecting a
`returnMismatch` rendered at the function name.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `returnMismatch` currently tries to locate a concrete
actual-value token, and implicit unit has no source token.

- [x] **Step 3: Add function-name fallback**

For `returnMismatch`, keep the concrete actual-token span when available and
fall back to `fnNameSpanWhere` when no token matches the actual type.

- [x] **Step 4: Document diagnostic behavior**

Record that implicit-unit declared return mismatches point at the function
name.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 131: Explicit Unit Return Mismatch Spans

**Files:**
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: source-level `unit` literal, declared return mismatch diagnostics,
  `tokenMatchesTy`, and the implicit-unit fallback added in Task 130.
- Produces: precise source spans for explicit `return unit` mismatches before
  falling back to the function-name span for implicit unit returns.

- [x] **Step 1: Add failing pipeline example**

Add a checked source example for `fn bad(): num { return unit }` expecting the
diagnostic range to cover the `unit` literal.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `Ty.unit` does not currently match any source token.

- [x] **Step 3: Match unit source tokens**

Teach `tokenMatchesTy` to recognize `Static.Ty.unit` as
`Lexer.TokenKind.identifier "unit"`.

- [x] **Step 4: Document diagnostic behavior**

Record that explicit unit return mismatches point at the `unit` token while
implicit unit mismatches still fall back to the function name.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 132: Option Checker Declared Return Contract

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `checkProgramDetailed`, `CheckError.returnMismatch`, typed return
  declarations, and the option-returning `checkProgram` API.
- Produces: consistent declared-return enforcement for both checker APIs while
  preserving `checkProgram : List Stmt -> Option CheckState`.

- [x] **Step 1: Add failing static example**

Add a checked example showing that plain `checkProgram` rejects a typed
function declared as `num` when the body returns `bool`.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `checkProgram` currently checks the body but ignores the
declared return type.

- [x] **Step 3: Delegate option checker to detailed checker**

Move the `checkProgram` wrapper after `checkProgramDetailed` and implement it
by converting `Except.ok state` to `some state` and any detailed error to
`none`.

- [x] **Step 4: Document checker consistency**

Record that `checkProgram` erases detailed errors rather than maintaining a
separate weaker checker path.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 133: Branch Declared Return Path Checking

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: declared return annotations, `inferBlockReturnTy?`, branch return
  inference, `CheckError.returnMismatch`, and typed function body checking.
- Produces: concrete declared-return mismatch detection for explicit `return`
  paths and final-expression returns inside branches and loops, without
  removing the existing gradual `unknown` compatibility model.

- [x] **Step 1: Add failing static example**

Add a checked example for a `num` function whose `if` branches return `num` and
`bool`, expecting a `returnMismatch` for the `bool` branch.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the old checker merges the branch return types to
`unknown`, then treats `unknown` as compatible with the declared return type.

- [x] **Step 3: Check concrete return paths**

Add a detailed declared-return path walk that checks explicit `return`
statements and final-expression returns against the declared type while
threading local variable bindings through the body.

- [x] **Step 4: Document the invariant**

Record that declared return checking now catches concrete branch mismatches
before relying on the merged return summary.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 134: Partial Branch Implicit Unit Return Checking

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: declared return path checking, `Stmt.ifThenElse`, implicit
  `Ty.unit`, and `CheckError.returnMismatch`.
- Produces: rejection for non-unit declared functions where an `if` branch can
  fall through because no `else` is present.

- [x] **Step 1: Add failing static example**

Add a checked example for `fn maybe(b: bool): num { if b { return 1 } }`,
expecting a `returnMismatch` against implicit `unit`.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the previous return-path checker only inspected the
present `then` branch.

- [x] **Step 3: Treat missing else as implicit unit**

When declared-return checking sees an `if` without an `else`, require the
declared return type to accept `unit`; otherwise report `returnMismatch`.

- [x] **Step 4: Document the invariant**

Record that non-unit functions must cover both branches explicitly.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 135: While Fallthrough Declared Return Checking

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: declared return path checking, `Stmt.while`, implicit `Ty.unit`,
  and `CheckError.returnMismatch`.
- Produces: rejection for non-unit declared functions whose only return path is
  inside a `while` body that may execute zero times.

- [x] **Step 1: Add failing static example**

Add a checked example for `fn loopReturn(b: bool): num { while b { return 1 } }`,
expecting a `returnMismatch` against implicit `unit`.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the previous return-path checker inspected the loop body
but did not account for zero-iteration fallthrough.

- [x] **Step 3: Treat while as fallthrough-capable**

After checking the loop body return paths, require the declared return type to
accept `unit`; otherwise report `returnMismatch`.

- [x] **Step 4: Document the invariant**

Record that non-unit functions cannot rely on a `while` body as their only
return source.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 136: For-Range Fallthrough Declared Return Checking

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: declared return path checking, `Stmt.forRange`, implicit
  `Ty.unit`, and `CheckError.returnMismatch`.
- Produces: rejection for non-unit declared functions whose only return path is
  inside an integer range loop that may execute zero times.

- [x] **Step 1: Add failing static example**

Add a checked example for `fn rangeReturn(): num { for i in 0..0 { return 1 } }`,
expecting a `returnMismatch` against implicit `unit`.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the previous return-path checker inspected the range
loop body but did not account for empty-range fallthrough.

- [x] **Step 3: Treat for-range as fallthrough-capable**

After checking body return paths with the iterator bound as `num`, require the
declared return type to accept `unit`; otherwise report `returnMismatch`.

- [x] **Step 4: Document the invariant**

Record that non-unit functions cannot rely on a `for` range body as their only
return source.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 137: Conditional Seal Fallthrough Declared Return Checking

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: declared return path checking, `Stmt.seal`, implicit `Ty.unit`,
  and `CheckError.returnMismatch`.
- Produces: rejection for non-unit declared functions whose only return path is
  inside a conditional `seal until` body that may be skipped.

- [x] **Step 1: Add failing static example**

Add a checked example for `fn sealReturn(b: bool): num { seal until b { return 1 } }`,
expecting a `returnMismatch` against implicit `unit`.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the previous return-path checker inspected the
conditional seal body but did not account for the already-satisfied exit
condition.

- [x] **Step 3: Treat conditional seal as fallthrough-capable**

After checking body return paths for `Stmt.seal (some _)`, require the declared
return type to accept `unit`; keep bare `seal` body-only.

- [x] **Step 4: Document the invariant**

Record that non-unit functions cannot rely on a conditional `seal until` body
as their only return source.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 138: Compositional Declared Return Fallthrough

**Files:**
- Modify: `Aether/Static.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: declared return path checking, fallthrough-aware `if`/loop/seal
  handling, `inferStmtVars?`, and `CheckError.returnMismatch`.
- Produces: block-level declared-return checking that lets fallthrough continue
  to later statements while still rejecting fallthrough at function-exit points.

- [x] **Step 1: Add failing static example**

Add a checked example for `fn guarded(b: bool): num { if b { return 1 } return 2 }`,
expecting the function to satisfy the declared return contract.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the previous checker rejected the missing `else` before
considering the following unconditional return.

- [x] **Step 3: Split path checks from block-exit checks**

Refactor declared-return checking into path-only statement/block checks and a
complete-block check that reports implicit `unit` only at block-exit points.

- [x] **Step 4: Preserve final return termination**

Handle final explicit `return` statements as terminating the checked block
rather than returning a value and then falling through to `unit`.

- [x] **Step 5: Document the invariant**

Record that missing branches fall through to later statements when present,
but still count as implicit `unit` at function-exit points.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 139: Portable Lean Lexer Line Endings

**Files:**
- Modify: `Aether/Lexer.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: plain `tokenize`, located `tokenizeLocated`, `SourcePos`, line
  comments, block comments, and string scanning.
- Produces: LF, CRLF, and CR line endings as one logical `TokenKind.newline`
  separator with stable located spans.

- [x] **Step 1: Add failing lexer examples**

Add Lean examples proving standalone CR produces a newline token, CRLF remains
a single newline token, and located tokenization advances following diagnostics
to line 2 for both line-ending forms.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because standalone CR was treated as whitespace and CRLF located
spans advanced through the skipped CR before the newline token.

- [x] **Step 3: Add logical newline handling**

Introduce `advanceNewline`, stop skipping CR as whitespace, scan CR and CRLF as
newline tokens, terminate line comments on CR as well as LF, and make raw CR
unterminate strings like LF.

- [x] **Step 4: Preserve block-comment positions**

Advance located block comments across CR and CRLF as one logical line break so
tokens after comments receive stable source positions.

- [x] **Step 5: Document the invariant**

Record that source newlines are platform-portable and that LF, CRLF, and CR all
represent one logical statement separator.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 140: Untyped-Parameter Function Return Annotations

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: proof-core function declarations, untyped parameter lists,
  `AnnTy`, declared-return checking, function signatures, and VM function
  collection.
- Produces: support for `fn name(params): type { ... }`, where parameters stay
  statically unknown but the function result is the declared type.

- [x] **Step 1: Add failing parser and static examples**

Add Lean examples for parsing `fn id(x): num { return x }`, rejecting
`fn bad(x): num { return true }`, and using a declared result type at call
sites.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the AST lacks an untyped-parameter typed-return
function constructor.

- [x] **Step 3: Add the AST and parser form**

Introduce `Stmt.fnDeclReturn` and parse `): type` after ordinary untyped
parameter lists.

- [x] **Step 4: Thread through static checking**

Collect function signatures with unknown parameter types and the declared
result type, validate duplicate untyped parameters, and run declared-return
checking against the body.

- [x] **Step 5: Preserve runtime and VM behavior**

Treat the new declaration as an ordinary function declaration for executable
semantics, function environments, frame-call normalization, and main-frame
lowering.

- [x] **Step 6: Document the implemented grammar**

Record that both untyped and typed parameter lists can carry declared return
annotations in the Lean proof-core parser.

- [x] **Step 7: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 141: Untyped-Parameter Return Annotation Diagnostics

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: `parseProgramDetailed`, parser diagnostic context
  classification, located source diagnostics, typed and untyped function
  parameter-list parsing, and `parseAnnTy`.
- Produces: type-context diagnostics for malformed return annotations after
  ordinary untyped parameter lists, matching typed-parameter return annotation
  diagnostics.

- [x] **Step 1: Add failing parser and source examples**

Add checked examples for `fn id(x): { return x }`, expecting
`ParseContext.typeAnnotation` and a source error at the `{` token after the
colon.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because return-annotation diagnostics only inspect typed
parameter-list remainders.

- [x] **Step 3: Generalize function parameter-list remainders**

Add a helper that returns the remainder after either a typed or untyped
parameter list and use it for malformed return annotation classification and
diagnostic offsets.

- [x] **Step 4: Document coverage**

Record that malformed function return annotations are diagnosed after both
untyped and typed parameter lists.

- [x] **Step 5: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 142: Nested Lean Block Comments

**Files:**
- Modify: `Aether/Lexer.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Lean plain and located block-comment lexing, `SourcePos`, and
  unterminated-comment diagnostics.
- Produces: depth-aware `/* ... */` block comments so nested block comments
  are skipped as one comment region and unterminated nested comments remain
  deterministic lexer errors.

- [x] **Step 1: Add failing lexer examples**

Add Lean examples proving `/* outer /* inner */ still */` skips the whole
comment and `/* outer /* inner */` reports an unterminated block comment.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the previous scanner closed at the first `*/`, leaking
the outer comment tail back into tokenization.

- [x] **Step 3: Implement depth-aware scanning**

Track block-comment depth in both plain and located skippers, incrementing on
nested `/*` and decrementing on `*/`.

- [x] **Step 4: Preserve located behavior**

Keep CRLF and newline position advancement inside comments and add a located
token-kind projection example for nested block comments.

- [x] **Step 5: Document the lexical rule**

Record that Lean proof-core block comments may nest and that unterminated
nested comments are lexer errors.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 143: Implicit Unit Return Mismatch Span

**Files:**
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: located token streams, static `returnMismatch` errors, function
  name matching, and brace-delimited function bodies.
- Produces: source diagnostics for implicit `unit` return mismatches that point
  at the function body's closing brace instead of falling back to the function
  name.

- [x] **Step 1: Add failing source diagnostic example**

Update the `fn bad(): num { }` source diagnostic example to expect the closing
brace range as the fallthrough source.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because implicit `unit` return mismatches had no concrete value
token and fell back to `fnNameSpanWhere`.

- [x] **Step 3: Add function-body close lookup**

Add a located-token helper that finds `fn name`, scans to its body opening
brace, and returns the matching closing brace span with nested brace depth.

- [x] **Step 4: Use only for implicit unit mismatches**

Keep explicit `return unit` diagnostics on the `unit` token, but use the
closing brace fallback when the actual return type is implicit `unit`.

- [x] **Step 5: Document the diagnostic rule**

Record that implicit fallthrough return mismatches point at the matched
function closing brace when available.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 144: Control-Flow Condition Mismatch Spans

**Files:**
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: static `conditionMismatch` errors, located token streams,
  `tokenMatchesTy`, and source diagnostic rendering.
- Produces: source diagnostics for non-boolean `if`, `while`, and
  `seal until` conditions that point at the offending condition token instead
  of the control-flow keyword when the concrete condition type can be matched.

- [x] **Step 1: Add failing source diagnostic examples**

Update the `if 1`, `while 1`, and `seal until 1` source diagnostic examples to
expect the numeric condition token spans.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because condition mismatches previously used the first
control-flow keyword span.

- [x] **Step 3: Add condition-token span lookup**

Scan located tokens for `if <condition>`, `while <condition>`, and
`seal until <condition>` starts, and return the condition token span when it
matches the mismatched static type.

- [x] **Step 4: Keep keyword fallback**

Fall back to the previous keyword span when the condition token cannot be
matched from the static type.

- [x] **Step 5: Document the diagnostic rule**

Record that non-boolean control-flow diagnostics prefer the offending condition
token.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 145: Trailing Binary Operator Parse Spans

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: detailed parser diagnostic offsets, token streams for
  `let`/assignment/`return` expressions, binary operator tokens, and source
  diagnostic rendering.
- Produces: parse diagnostics for expressions ending in a binary operator that
  point at the trailing operator instead of the statement start.

- [x] **Step 1: Add failing parser and source examples**

Update examples for `let x = 1 +`, `let x = self +`, `let x = 1.5 +`, and the
same failure after an earlier valid statement to expect the `+` token.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because incomplete expressions with valid starts previously
fell back to the broad statement-start diagnostic.

- [x] **Step 3: Add trailing binary-operator detection**

Add token helpers that identify binary operators followed by a statement
terminator or EOF and return their offset inside expression-bearing statements.

- [x] **Step 4: Preserve invalid-start behavior**

Keep the existing invalid-expression-start offsets for malformed starts when no
trailing binary operator is present.

- [x] **Step 5: Document the diagnostic rule**

Record that missing right operands at statement end point at the trailing
operator.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 146: Control-Flow Condition Trailing Operator Spans

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: detailed parser diagnostics for `if`, `while`, and `seal until`,
  located source spans, condition expression token streams, and the existing
  binary-operator diagnostic helpers.
- Produces: parse diagnostics for condition expressions ending in a binary
  operator before a body block that point at the trailing operator instead of
  the control-flow keyword or broad block context.

- [x] **Step 1: Add failing parser and source examples**

Add checked examples for `if 1 + { break }`, `while 1 + { break }`, and
`seal until 1 + { break }`, expecting expression diagnostics at the `+` token
and matching source spans.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because control-flow condition failures previously fell back to
the broad keyword/block diagnostic when the expression start was valid.

- [x] **Step 3: Add condition-specific trailing-operator detection**

Add a scanner that treats `{` as a condition-expression terminator and use it
for `if`, `while`, and `seal until` classification and diagnostic offsets.

- [x] **Step 4: Preserve malformed-start behavior**

Keep missing-condition diagnostics such as `if { break }`, `while { break }`,
and `seal until { break }` pointing at the offending `{` token.

- [x] **Step 5: Document the diagnostic rule**

Record that trailing binary operators before control-flow body blocks point at
the operator as the missing-right-operand site.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 147: Nested List Type Annotation Diagnostic Spans

**Files:**
- Modify: `Aether/Parser.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: recursive `parseAnnTy`, local declaration annotations, typed
  function parameter annotations, function return annotations, parser
  diagnostic offsets, and located source rendering.
- Produces: malformed `list[...]` type diagnostics that point at the inner
  token where the element type or closing bracket is missing instead of the
  outer `list` token.

- [x] **Step 1: Add failing parser and source examples**

Add checked examples for `let xs: list[ = [1]`,
`fn id(x: list[) { return x }`, and `fn id(x): list[ { return x }`,
expecting type diagnostics at `=`, `)`, and `{` respectively.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because malformed nested annotation offsets previously reported
the beginning of the annotation rather than the inner failing token.

- [x] **Step 3: Add recursive annotation failure offsets**

Add a helper that follows `list[` annotations into their element type and
returns the token where the element type or closing bracket fails.

- [x] **Step 4: Wire all annotation sites**

Use the helper for local declaration annotations, typed function parameter
annotations, and function return annotations after both typed and untyped
parameter lists.

- [x] **Step 5: Document the diagnostic rule**

Record that nested `list[...]` type annotation failures point at the inner
missing element or bracket location.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 148: Proof-Core `is_empty` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, static method compatibility,
  stack/frame method opcodes, parser postfix method calls, checked source frame
  compilation, and pipeline source execution.
- Produces: zero-argument `.is_empty()` support for proof-core strings and
  lists, returning `bool`, through the evaluator, checker, VM, checked compiler,
  and source pipeline.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `[].is_empty()` and `"open".is_empty()` through
`evalExpr`, `checkExpr`, closed expression bytecode execution, frame expression
compilation, `checkedFrameSourceLocal?`, and `sourceLocal?`.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the shared method evaluator and static compatibility
table only support `.len()` for strings and lists.

- [x] **Step 3: Implement shared runtime support**

Extend `evalMethod` so zero-argument `is_empty` returns `Value.bool` for lists
and strings.

- [x] **Step 4: Implement static method typing**

Extend `compatibleMethod` so zero-argument `is_empty` on strings and lists
checks as `Ty.bool`.

- [x] **Step 5: Document the proof-core method**

Record `.is_empty()` in the proof-core method surface and checked compiler
coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 149: Field and Method Mismatch Member Spans

**Files:**
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: static `fieldMismatch` and `methodMismatch` errors, located token
  streams, flexible member identifiers, and source diagnostic rendering.
- Produces: source diagnostics for unsupported fields and methods that point
  at the member identifier after `.` instead of the dot token when the member
  can be matched.

- [x] **Step 1: Add failing source diagnostic examples**

Update checked examples for `let bad = 1.length` and `let bad = 1.len()` to
expect spans over `length` and `len` rather than the dot.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because static member mismatch spans previously used the last
dot token.

- [x] **Step 3: Add member-name span lookup**

Scan located tokens for `.` followed by the matching field or method name and
return that identifier span.

- [x] **Step 4: Preserve fallback behavior**

Fall back to the previous dot-token span if no matching member identifier can
be recovered.

- [x] **Step 5: Document the diagnostic rule**

Record that field and method mismatch diagnostics prefer the member identifier.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 150: Proof-Core List `first` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, static method compatibility,
  stack/frame method opcodes, parser postfix method calls, checked source frame
  compilation, and pipeline source execution.
- Produces: zero-argument `.first()` support for proof-core lists, returning
  the first runtime value when present and the static list element type.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `[7, 9].first()` through `evalExpr`, `checkExpr`,
closed expression bytecode execution, frame expression compilation,
`checkedFrameSourceLocal?`, and `sourceLocal?`. Include an empty-list runtime
example returning `none`.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the shared method evaluator and static compatibility
table do not yet support `.first()`.

- [x] **Step 3: Implement shared runtime support**

Extend `evalMethod` so zero-argument `first` on a non-empty list returns the
first element and empty lists remain runtime failures.

- [x] **Step 4: Implement static method typing**

Extend `compatibleMethod` so zero-argument `first` on `list[T]` checks as `T`.

- [x] **Step 5: Document the proof-core method**

Record `.first()` in the proof-core method surface and checked compiler
coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 151: Proof-Core List `last` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, static method compatibility,
  stack/frame method opcodes, parser postfix method calls, checked source frame
  compilation, and pipeline source execution.
- Produces: zero-argument `.last()` support for proof-core lists, returning
  the final runtime value when present and the static list element type.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `[7, 9].last()` through `evalExpr`, `checkExpr`,
closed expression bytecode execution, frame expression compilation,
`checkedFrameSourceLocal?`, and `sourceLocal?`. Include an empty-list runtime
example returning `none`.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the shared method evaluator and static compatibility
table do not yet support `.last()`.

- [x] **Step 3: Implement shared runtime support**

Extend `evalMethod` so zero-argument `last` on a non-empty list returns the
last element and empty lists remain runtime failures.

- [x] **Step 4: Implement static method typing**

Extend `compatibleMethod` so zero-argument `last` on `list[T]` checks as `T`.

- [x] **Step 5: Document the proof-core method**

Record `.last()` in the proof-core method surface and checked compiler
coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 152: Proof-Core List `at(index)` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, static method argument type inference,
  stack/frame method opcodes, parser postfix method arguments, checked source
  frame compilation, and pipeline source execution.
- Produces: `.at(index)` support for proof-core lists, returning the selected
  runtime value when the numeric index is in range and the static list element
  type when the index argument checks as numeric.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `[7, 9].at(1)` through `evalExpr`, `checkExpr`, closed
expression bytecode execution, frame expression compilation,
`checkedFrameSourceLocal?`, and `sourceLocal?`. Include out-of-range runtime
failure and a boolean-index static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the shared method evaluator and static compatibility
table do not yet support `.at(index)`.

- [x] **Step 3: Implement shared runtime support**

Extend `evalMethod` so list `at` delegates to indexed list lookup with numeric
indices.

- [x] **Step 4: Implement argument-aware static method typing**

Extend static method compatibility to consume inferred argument types, and use
that path from both detailed and option-returning expression checkers.

- [x] **Step 5: Document the proof-core method**

Record `.at(index)` in the proof-core method surface and checked compiler
coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 153: Proof-Core List `contains(value)` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, static method argument type inference,
  stack/frame method opcodes, parser postfix method arguments, checked source
  frame compilation, and pipeline source execution.
- Produces: `.contains(value)` support for proof-core lists, returning `bool`
  at runtime and statically requiring the searched value to be compatible with
  the list element type.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `[7, 9].contains(9)` through `evalExpr`,
`checkExpr`, closed expression bytecode execution, frame expression
compilation, `checkedFrameSourceLocal?`, and `sourceLocal?`. Include a false
runtime membership case and a mismatched argument static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the shared method evaluator and static compatibility
table do not yet support `.contains(value)`.

- [x] **Step 3: Implement shared runtime support**

Add list membership evaluation over `Value` equality and wire list
`.contains(value)` into `evalMethod`.

- [x] **Step 4: Implement argument-aware static method typing**

Extend static method compatibility so list `.contains(value)` returns `bool`
only when the argument type is compatible with the list element type.

- [x] **Step 5: Document the proof-core method**

Record `.contains(value)` in the proof-core method surface and checked compiler
coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 154: Proof-Core List `tail()` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, static method compatibility, stack/frame
  method opcodes, parser postfix method calls, checked source frame
  compilation, and pipeline source execution.
- Produces: `.tail()` support for proof-core lists, returning the remaining
  runtime list for non-empty lists and statically preserving the list element
  type as `list[T]`.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `[7, 9].tail()` through `evalExpr`, `checkExpr`,
closed expression bytecode execution, frame expression compilation,
`checkedFrameSourceLocal?`, and `sourceLocal?`. Include an empty-list runtime
failure and an arity mismatch static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the shared method evaluator and static compatibility
table do not yet support `.tail()`.

- [x] **Step 3: Implement shared runtime support**

Extend `evalMethod` so zero-argument `tail` on a non-empty list returns the
remaining values as `Value.list`; empty lists remain runtime failures.

- [x] **Step 4: Implement static method typing**

Extend static method compatibility so list `.tail()` returns `list[T]`.

- [x] **Step 5: Document the proof-core method**

Record `.tail()` in the proof-core method surface and checked compiler
coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 155: Proof-Core List `take(count)` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, static method argument type inference,
  stack/frame method opcodes, parser postfix method arguments, checked source
  frame compilation, and pipeline source execution.
- Produces: `.take(count)` support for proof-core lists, returning the prefix
  list at runtime for non-negative numeric counts and statically preserving
  the list element type as `list[T]`.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `[7, 9].take(1)` through `evalExpr`, `checkExpr`,
closed expression bytecode execution, frame expression compilation,
`checkedFrameSourceLocal?`, and `sourceLocal?`. Include oversized-count
runtime behavior, negative-count runtime failure, and boolean-count static
diagnostics.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the shared method evaluator and static compatibility
table do not yet support `.take(count)`.

- [x] **Step 3: Implement shared runtime support**

Add an explicit recursive list-prefix helper and wire list `.take(count)` into
`evalMethod`, rejecting negative counts at runtime.

- [x] **Step 4: Implement argument-aware static method typing**

Extend static method compatibility so list `.take(count)` returns `list[T]`
only when the count argument checks as numeric.

- [x] **Step 5: Document the proof-core method**

Record `.take(count)` in the proof-core method surface and checked compiler
coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 156: Proof-Core List `drop(count)` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, static method argument type inference,
  stack/frame method opcodes, parser postfix method arguments, checked source
  frame compilation, and pipeline source execution.
- Produces: `.drop(count)` support for proof-core lists, returning the suffix
  list at runtime for non-negative numeric counts and statically preserving
  the list element type as `list[T]`.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `[7, 9].drop(1)` through `evalExpr`, `checkExpr`,
closed expression bytecode execution, frame expression compilation,
`checkedFrameSourceLocal?`, and `sourceLocal?`. Include oversized-count
runtime behavior, negative-count runtime failure, and boolean-count static
diagnostics.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the shared method evaluator and static compatibility
table do not yet support `.drop(count)`.

- [x] **Step 3: Implement shared runtime support**

Add an explicit recursive list-suffix helper and wire list `.drop(count)` into
`evalMethod`, rejecting negative counts at runtime.

- [x] **Step 4: Implement argument-aware static method typing**

Extend static method compatibility so list `.drop(count)` returns `list[T]`
only when the count argument checks as numeric.

- [x] **Step 5: Document the proof-core method**

Record `.drop(count)` in the proof-core method surface and checked compiler
coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 157: Lean String Carriage-Return Escape

**Files:**
- Modify: `Aether/Lexer.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: Lean plain and located string literal scanners, token-kind
  projection checks, source parsing, and source execution.
- Produces: `\r` string escape support in the Lean proof-core lexer, preserving
  located-token parity and runtime string values through the pipeline.

- [x] **Step 1: Add failing lexer/source examples**

Add checked examples proving plain tokenization accepts `"row\\rnext"`,
located token-kind projection matches plain tokenization, and `sourceLocal?`
executes a source string containing the escaped carriage return.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because `\r` is still reported as an invalid string escape.

- [x] **Step 3: Implement plain string escape support**

Extend `readString` so escaped `r` appends the carriage-return character.

- [x] **Step 4: Implement located string escape support**

Extend `readStringLocated` with the same escaped `r` handling while preserving
the existing consumed source span.

- [x] **Step 5: Document the lexical rule**

Record `\r` in the proof-core string escape list.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 158: Proof-Core String Indexing

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: parser postfix indexing, shared `evalIndex`, static index
  compatibility, stack/frame index opcodes, checked source frame compilation,
  and pipeline diagnostic rendering.
- Produces: string indexing in the proof-core executable semantics, returning
  a one-character string at runtime for in-range numeric indexes and statically
  typing `str[num]` as `str`.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `"open"[1]` through `evalExpr`, `checkExpr`, closed
expression bytecode execution, and `sourceLocal?`. Include out-of-range
runtime failure and a boolean-index static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because the shared index evaluator and static compatibility
table do not yet support string targets.

- [x] **Step 3: Implement shared runtime support**

Extend `evalIndex` so string targets with non-negative numeric indexes return
the selected character as a one-character `Value.str`.

- [x] **Step 4: Implement static index typing**

Extend static index compatibility so `str` indexed by a numeric value returns
`str`.

- [x] **Step 5: Document string indexing**

Record that proof-core indexing supports both lists and strings.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 159: Proof-Core String `at(index)` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared string indexing, shared method evaluation, argument-aware
  static method compatibility, stack/frame method opcodes, checked source
  frame compilation, and source diagnostic rendering.
- Produces: `.at(index)` support for proof-core strings, returning a
  one-character string at runtime for in-range numeric indexes and statically
  typing `str.at(num)` as `str`.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `"open".at(1)` through `evalExpr`, `checkExpr`,
closed expression bytecode execution, frame expression compilation,
`checkedFrameSourceLocal?`, and `sourceLocal?`. Include out-of-range runtime
failure and a boolean-index static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because string method evaluation and static method
compatibility do not yet support `.at(index)`.

- [x] **Step 3: Implement shared runtime support**

Wire string `.at(index)` through `evalIndex` so method calls and indexing share
the same runtime behavior.

- [x] **Step 4: Implement argument-aware static method typing**

Extend static method compatibility so `str.at(index)` returns `str` only when
the index argument checks as numeric.

- [x] **Step 5: Document the proof-core method**

Record string `.at(index)` in the proof-core method surface and checked
compiler coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.

### Task 160: Proof-Core String `contains(value)` Method

**Files:**
- Modify: `Aether/Core.lean`
- Modify: `Aether/Static.lean`
- Modify: `Aether/VM.lean`
- Modify: `Aether/Pipeline.lean`
- Modify: `docs/FORMAL_CORE.md`

**Interfaces:**
- Consumes: shared method evaluation, argument-aware static method
  compatibility, stack/frame method opcodes, checked source frame compilation,
  and source diagnostic rendering.
- Produces: `.contains(value)` support for proof-core strings, returning
  `bool` at runtime for substring membership and statically requiring a string
  argument.

- [x] **Step 1: Add failing evaluator/static/VM/source examples**

Add checked examples for `"open".contains("pe")` through `evalExpr`,
`checkExpr`, closed expression bytecode execution, frame expression
compilation, `checkedFrameSourceLocal?`, and `sourceLocal?`. Include a false
runtime membership case and a non-string argument static diagnostic.

- [x] **Step 2: Verify red**

Run: `lake build`
Expected: FAIL because string method evaluation and static method
compatibility do not yet support `.contains(value)`.

- [x] **Step 3: Implement shared runtime support**

Add explicit character-list prefix and substring helpers, then wire string
`.contains(value)` into `evalMethod`.

- [x] **Step 4: Implement argument-aware static method typing**

Extend static method compatibility so `str.contains(value)` returns `bool`
only when the value argument checks as `str`.

- [x] **Step 5: Document the proof-core method**

Record string `.contains(value)` in the proof-core method surface and checked
compiler coverage.

- [x] **Step 6: Verify**

Run: `lake build` and `cargo test -p aether-lang -p aether-cli`
Expected: PASS.
