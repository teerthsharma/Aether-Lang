# Aether Formal Core

This document defines the current proof-facing core of Aether-Lang for Lean 4
formalization. It describes the executable subset shared by the parser,
interpreter, and Titan VM as of the current implementation.

## Purpose

The formal core is the stable language fragment intended to receive a Lean 4
syntax, static well-formedness relation, and operational semantics. Surface
forms outside this file may remain experimental until they are lowered into this
core or given their own proof rules.

## Lexical Surface

The core lexer recognizes:

- Identifiers: ASCII identifiers used for variables, parameters, and functions.
- Numbers: integer literals and fixed micro-precision decimal float literals.
- Booleans: `true`, `false`.
- Unit: `unit` as the source spelling for the unit value.
- Strings: string literals for proof-core values and equality checks, with
  `\"`, `\\`, `\n`, `\r`, and `\t` escapes.
- Comments: line comments beginning with `//` and block comments delimited by
  `/*` and `*/`; block comments may nest, and unterminated block comments are
  lexer errors.
- Lists: bracketed list literals whose elements are proof-core expressions;
  newlines may separate elements and a trailing comma before `]` is accepted.
- Statement separators: newline, `~`, and `;`. Newline accepts LF, CRLF, or CR
  as one logical line break.
- Range separator: `..` for integer ranges.
- Seal alias: `seal` and `🦭` both tokenize to the same control-flow keyword.
- Operators: `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `&&`,
  `||`, `!`.

The lexer must preserve `1..10` as `Number(1)`, `DotDot`, `Number(10)` rather
than treating it as a malformed float. Decimal literals such as `1.5` parse as
fixed micro-precision proof-core numeric expressions.

## Core Syntax

The proof core is:

```text
program  ::= stmt*

stmt     ::= "let" ident (":" ann-ty)? "=" newline* expr
           | ident "=" newline* expr
           | "if" newline* expr block ("else" block)?
           | "while" newline* expr block
           | "for" ident "in" signed-int ".." signed-int block
           | "seal" ("until" newline* expr)? block
           | "fn" ident newline* "(" params? ")" (":" ann-ty)? block
           | "return" expr?
           | "break"
           | "continue"
           | expr

block    ::= separator* "{" stmt* "}"
separator ::= newline | "~" | ";"
params   ::= ident ("," newline* ident)* ","? newline*
           | ident ":" ann-ty ("," newline* ident ":" ann-ty)* ","? newline*
ann-ty   ::= "num" | "bool" | "str" | "unit"
           | "list" "[" newline* ann-ty newline* "]"

expr     ::= literal
           | ident
           | ident newline* "(" call-args? ")"
           | "[" list-items? "]"
           | "(" newline* expr newline* ")"
           | expr "[" newline* expr newline* "]"
           | expr "." newline* ident
           | expr "." newline* ident "(" call-args? ")"
           | "-" newline* expr
           | "!" newline* expr
           | expr binop newline* expr

call-args ::= arg ("," newline* arg)* ","? newline*
arg      ::= expr | ident "=" expr
list-items ::= expr ("," newline* expr)* ","? newline*
binop    ::= "+" | "-" | "*" | "/" | "%"
           | "==" | "!=" | "<" | ">" | "<=" | ">="
           | "&&" | "||"
literal  ::= number | float | bool | string | unit
signed-int ::= number | "-" number
```

Parser precedence, from low to high, is:

1. `||`
2. `&&`
3. `==`, `!=`
4. `<`, `>`, `<=`, `>=`
5. `..`
6. `+`, `-`
7. `*`, `/`, `%`
8. unary `-`, `!`
9. postfix `expr[expr]`, `expr.ident`, `expr.ident(args?)`
10. primary expressions

## Runtime Values

The proof core values are:

```text
Value ::= Num int | Float int micros | Bool bool | Str string | List Value* | Unit
```

The interpreter supports additional host values such as manifolds, classes,
tensors, and native functions. Those are outside the first Lean 4 core unless
explicitly modeled as opaque external values. `Float int micros` stores decimal
numeric values in fixed micro-precision, matching lexer float tokens.

## Big-Step Statement Semantics

Statement execution produces a flow:

```text
Flow ::= Value Value | Return Value | Break | Continue
```

The environment maps identifiers to runtime values. Function calls execute in a
call frame whose parameter bindings are local to the call. The VM currently
restores the caller locals after `RET`; the interpreter clones and restores the
variable environment around user function execution.

Core rules to encode first:

- `let x = e` evaluates `e` and binds `x`.
- `x = e` requires an existing binding in interpreter semantics, then updates
  `x`. The VM slot model currently creates a slot if one is missing; the formal
  core should choose the interpreter rule as the stricter source semantics.
- `if c { t } else { f }` executes `t` when `c` is truthy, otherwise `f`.
- `while c { b }` repeatedly executes `b` while `c` is truthy.
- `for i in a..b { body }` binds `i` to each integer value `a <= i < b`;
  `a` and `b` may be signed integer literals.
- `seal until c { body }` checks `c` before each iteration and stops when true.
- `fn f(params) { body }` is a top-level declaration in the proof-core static
  semantics; nested function declarations are rejected before bytecode lowering.
- `return e` produces `Return(v)` and unwinds to the nearest function call.
- `break` and `continue` are handled by the nearest loop.
- An expression statement evaluates the expression and leaves its value as the
  statement result.

Truthiness in the current runtime is:

- `Bool(false)` is false.
- `Num(0.0)` is false.
- Empty strings and empty lists are false.
- `Unit` and unsupported values are false.
- Other booleans, nonzero numbers, non-empty strings, and non-empty lists are true.

The proof-core static checker is stricter for control flow: `if`, `while`, and
`seal until` conditions must have type `bool` when known. Logical `&&` and
`||` operands must also be boolean when known. Conditions and logical operands
whose type is still `unknown` are accepted until more precise annotations or
inference are available. Function signatures retain arity and a conservative
inferred result type from visible `return` statements and final expression
statements; calls to functions with concrete inferred returns use that type,
while imprecise returns remain `unknown`. Unary logical `!` has the same known-boolean requirement. Equality
operators require compatible known operand types, while still allowing
`unknown` on either side. The `unit` literal has type `unit`. List literals carry a static element type when
homogeneous; empty, mixed, and otherwise imprecise lists use `list[unknown]`
while preserving the dynamic runtime list value. List indexing requires a
list-like target and a numeric index when known; successful indexing returns
the known element type for homogeneous lists and `unknown` for imprecise lists.
String indexing also requires a numeric index and returns a one-character
string when the index is in range.
Field access currently supports `.length` for known strings and lists,
producing `num`; unsupported concrete fields are rejected statically, while
fields on `unknown` targets remain `unknown`. Pure method calls currently
support zero-argument `.len()` for known strings and lists, producing `num`,
and zero-argument `.is_empty()` for known strings and lists, producing `bool`;
strings also support zero-argument `.first()` and `.last()`, which return the
first or last character as `str` when present, zero-argument `.tail()`, which
returns the remaining string after the first character when present,
`.take(count)` and `.drop(count)`, which require numeric counts and return the
prefix or suffix string, zero-argument `.reverse()`, which returns the
characters in reverse order, `.at(index)`, which requires a numeric index and returns `str`, plus
`.contains(value)`, `.starts_with(prefix)`, and `.ends_with(suffix)`, which
require string arguments and return `bool`; lists also support zero-argument `.first()` and `.last()`, producing the known
element type statically and the corresponding runtime element when present,
zero-argument `.tail()`, producing `list[T]` statically and the remaining
runtime list for non-empty lists, `.take(count)` and `.drop(count)`, which
require numeric counts and return lists with the same element type,
zero-argument `.reverse()`, producing `list[T]` statically and the reversed
runtime list, `.append(value)`, which requires a value compatible with the list
element type and returns a new list with that value at the end,
`.prepend(value)`, which requires a value compatible with the list element type
and returns a new list with that value at the beginning,
`.concat(other)`, which requires a list argument with compatible element type
and returns a new concatenated list, `.join(separator)`, which requires string
list elements and a string separator and returns `str`, plus `.at(index)`, which requires a numeric index argument and returns the known
element type, and `.contains(value)`, which also requires a value compatible
with the list element type and returns `bool`;
unsupported concrete methods are rejected statically, while methods on
`unknown` targets remain `unknown`.
Annotated local declarations such as `let count: num = 1` and
`let xs: list[num] = [1]` bind the declared type only when the initializer is
compatible with that annotation; incompatible initializers are rejected before
runtime or bytecode execution.

## Bytecode Correspondence

The VM core lowers supported syntax into stack bytecode:

- Numeric constants: `PUSH`.
- Boolean constants: `PUSH_BOOL`.
- Locals: `LOAD`, `STORE`.
- Arithmetic and logic: `ADD`, `SUB`, `MUL`, `DIV`, `MOD`, `NEG`, `EQ`, `NEQ`,
  `LT`, `GT`, `LE`, `GE`, `AND`, `OR`, `NOT`.
- Unit constants: `PUSH Unit`.
- Lists and strings: dynamic list construction plus list and string indexing.
- Fields: postfix field access with executable `.length` on strings and lists.
- Methods: pure postfix `.len()` and `.is_empty()` method calls on strings and
  lists, string `.first()`, `.tail()`, `.last()`, `.take(count)`, `.drop(count)`, `.reverse()`, `.at(index)`, `.contains(value)`, `.starts_with(prefix)`, and `.ends_with(suffix)`, plus list `.first()`, `.tail()`, `.last()`, `.at(index)`,
  `.take(count)`, `.drop(count)`, `.reverse()`, `.append(value)`, `.prepend(value)`, `.concat(other)`, `.join(separator)`, and `.contains(value)`.
- Branching: `JMP`, `JMP_IF_FALSE`.
- Loop flow: `break` and `continue` are lowered by the compiler into patched
  `JMP` instructions targeting the loop exit or continuation point.
- Functions: `CALL(target, arity)` and `RET`.
- Program end: `HALT`.

The first VM proof target should be stack preservation for well-formed bytecode:
if bytecode is produced by the compiler for a well-formed core program, runtime
stack underflow does not occur.

## Current Boundaries

The following are parsed or represented elsewhere but are not yet in the formal
core:

- Classes, methods, object creation, modules, imports.
- Manifold, block, render, regress, topology-specific host operations.
- Named function arguments in user-defined calls.
- General object/class field access beyond proof-core `.length`.
- Mutating or host/object method calls beyond proof-core pure `.len()`.
- Tensor and ML model handles.
- Forward function references before declaration in VM lowering.
- Global variable capture inside VM user functions.

These features should either lower into the core above or receive separate
semantics before being included in Lean 4 proofs.

## Lean 4 Formalization

The checked Lean 4 scaffold lives in:

- `lakefile.lean`: Lake package definition for `aether-formal`.
- `lean-toolchain`: Lean toolchain pin.
- `Aether.lean`: top-level module import.
- `Aether/Lexer.lean`: token-kind model and executable lexical scanner for
  the proof DSL surface.
- `Aether/Core.lean`: core syntax, values, environments, expression evaluation,
  executable function-call evaluation, single-statement and block-flow
  relations, and initial sanity theorems/examples.
- `Aether/Static.lean`: executable static well-formedness and lightweight type
  checking for the current unannotated proof core.
- `Aether/Parser.lean`: executable token-to-core parser for the current
  proof-core expression subset and simple statements.
- `Aether/VM.lean`: core stack-machine bytecode, VM state, one-step execution,
  bounded execution, expression-to-bytecode compilation for the closed
  literal/unary/binary subset plus slot-aware variable expression compilation,
  straight-line statement compilation for `let`, assignment, and expression
  statements, straight-line block compilation, branch-aware `if` compilation,
  bounded `while`, integer-range `for`, and `seal` compilation, and checked
  examples for arithmetic, locals, conditional branch behavior, loop behavior,
  call-frame behavior, and evaluator/compiler agreement examples.
- `Aether/Pipeline.lean`: stage-aware source diagnostics over the executable
  lexer, parser, static checker, checked compiler, and frame VM runner.

Run:

```text
lake build
```

The first formalization target is the executable proof core, not the full host
runtime. `Aether.Core` models exact integer literals and fixed micro-precision
decimal float literals as the proof-friendly numeric subset of the Rust
runtime's `f64` values. Integer-only arithmetic preserves integer results;
mixed integer/float arithmetic converts through micro-units and returns
`Value.float`.

`Aether.Lexer` mirrors the Rust lexer token-kind surface in Lean and provides
an executable `tokenize` scanner for keywords, identifiers, integers, fixed
micro-precision float tokens, strings, line comments, block comments, statement
separators, range tokens, arithmetic/comparison/logical operators, delimiters,
newline, EOF, and lexical errors. Checked examples cover the important `1.5`
versus `1..10` split, keyword/operator scanning, the `🦭` seal alias, comment
handling, nested block comments, tilde separators, portable LF/CRLF/CR line
endings, and string termination errors. `tokenizeLocated` preserves each token's starting and
ending line/column as a `SourceSpan` while keeping the existing `TokenKind`
parser API unchanged. The `tokenKinds` projection maps a located token stream
back to its parser-facing token kinds, and checked examples verify that this
projection agrees with `tokenize` for representative core inputs including
comments, newlines, ranges, strings, lexical errors, and the `🦭` alias.
Checked examples cover ordinary token ranges, newline ranges for LF/CRLF/CR,
the seal emoji alias range, `1..10` token spans, and lexical error ranges.
String escape scanning translates supported escapes to runtime characters and
reports unsupported escape sequences as lexer errors whose located spans point
at the offending escape. Block-comment scanning is depth-aware, advances
line/column positions across newlines, and reports unterminated block comments
with spans from the opening slash through EOF. AST-level source spans and
lexer/parser correctness theorems remain future formalization work.

`Aether.Parser` consumes the Lean token stream and produces `Aether.Core`
syntax for the proof subset. It currently covers precedence-aware expressions
for integer numeric, fixed micro-precision float, boolean, string, unit, and list
literals, postfix list indexing, postfix field access, postfix method calls,
variables including `self`, unary negation/not,
multiplicative/additive,
comparison, equality, logical `&&`, and logical `||`, plus parenthesized
expressions, multiline parenthesized expressions, function calls with positional or named arguments, multiline and
trailing-comma argument lists, and Rust-compatible keyword call syntax for
`embed(...)` and `convergence(...)`. Statement parsing covers `let`,
assignment, `return`, `break`, `continue`, expression statements, newline
separators, tilde separators, EOF termination, brace-delimited blocks,
`if`/`else`, `while`, signed integer-range `for`, conditional `seal until`,
unconditional `seal`, annotated local declarations, untyped `fn` declarations,
`fn` declarations with basic parameter type annotations such as `x: num` and
`xs: list[num]`, untyped-parameter function declarations with declared return
types such as `fn id(x): num`, typed function declarations with declared
return types such as `fn id(x: num): num`, and multiline function parameter
lists with optional trailing commas. `list[...]` type annotations may place
newlines after `[` and before `]`, including in typed parameters and declared
return types. Block-bearing forms may place a statement separator between the
header and opening `{`.
Parser diagnostics for malformed multiline `list[...]` annotations skip
annotation-internal newlines and point at the offending token.
Parenthesized expressions may place newlines after `(` and before `)`.
Postfix index expressions may place newlines after `[` and before `]`.
Postfix field and method expressions may place newlines after `.`.
Function-call expressions may place newlines between the callee name and `(`.
Function declarations may place newlines between the function name and `(`.
Unary expressions may place newlines after `-` or `!`.
Binary expressions may place newlines after the operator before the right-hand
operand.
Let declarations and assignment statements may place newlines after `=` before
the right-hand expression.
Control-flow condition forms may place newlines after `if`, `while`, or
`seal until` before the condition expression.
Checked examples verify arithmetic
precedence, parenthesized boolean expressions, positional call parsing,
multiline binary right-hand-side parsing,
multiline assignment right-hand-side parsing,
multiline control-flow condition parsing,
multiline unary expression parsing,
multiline parenthesized expression parsing,
multiline postfix index parsing,
multiline postfix member parsing,
multiline call opening parsing,
multiline function declaration opening parsing,
multiple tilde-separated statements, newline-separated loop-control statements,
block parsing, `if`/`else` parsing, `while` parsing, signed integer `for` parsing,
both conditional and unconditional `seal` parsing, untyped and typed function declaration
parsing, line-broken block opening parsing, `self` expression parsing, reserved domain keyword field/method names,
keyword call parsing, named argument preservation, multiline function parameter
parsing, multiline call argument parsing, multiline list type annotation
parsing, list literal parsing, and decimal float literal parsing. Method calls beyond proof-core pure `.len()`,
broader object/class field semantics, source spans, full compiler/VM
correspondence proofs, and parser correctness theorems remain future
formalization work.
`parseProgramDetailed` wraps the executable parser with `Except ParseError`,
currently recording the broad context that failed plus a diagnostic token. For
malformed expression starts in `let`, assignment, `return`, `if`, `while`, and
conditional `seal`, the diagnostic token is the offending expression-start
token. The classifier treats `self`, decimal float literals, and `embed` or
`convergence` followed by `(` as valid expression starts, matching the
executable parser's `self` variable, float literal, and keyword-call expression
support while rejecting bare keyword-call names as malformed starts. Incomplete
expressions that end at a binary operator before a statement terminator point
at that trailing operator as the missing-right-operand site. Condition
expressions before `if`, `while`, and `seal until` body blocks use the same
rule when the trailing operator appears before `{`. Other incomplete expressions
whose first token is valid still use the broader statement-start diagnostic
until the parser carries recursive failure locations. Checked examples cover
expression, trailing binary operators in statements and control-flow conditions,
malformed `self`-started and float-started expressions, bare keyword-call names,
malformed integer-ranges, complete `for` ranges missing body blocks, missing
`if`/`while` condition expressions, stray `else` tokens that require a preceding
parsed `if` statement, conditional `seal until` expression, function-parameter
parse failures, malformed type annotations in local declarations, function
parameters, and function return positions after both untyped and typed parameter
lists, and function declarations with valid parameters but missing body blocks.
Malformed nested `list[...]` type annotations recursively point at the token
where an element type or closing bracket is missing, so local, parameter, and
return annotation diagnostics do not collapse to the outer `list` token.
`Aether.Pipeline` attaches the corresponding
`SourceSpan` from the located token stream when surfacing parse diagnostics.
The pipeline walks located statement boundaries so failures after earlier
valid statements point at the later failed statement rather than the beginning
of the file, and invalid expression-start failures point at the offending
token's range without requiring a fully spanned AST.

`Aether.Static` adds the first checked static gate for proof-core programs.
It models `num`, `bool`, `str`, element-aware `list[...]`, `unit`, and
`unknown` types; `unknown` is used for unannotated function parameters,
imprecise function call results, and imprecise list element types. Integer
literals and fixed micro-precision float literals both check as `num`. The
checker validates known arithmetic/comparison/logical/equality operand shapes,
unary operator operands, boolean control-flow conditions, declaration-before-use,
assignment compatibility with refinement of `unknown` assignment targets,
function arity, annotated function parameter argument types, declared function
return types, inferred function call result types,
named function-call argument validation against declared parameter names,
conservative `if`/`else` branch environment joins, `return` placement inside functions,
top-level-only function declarations, and `break`/`continue` placement inside
loops. It checks function bodies after collecting top-level function
signatures, so forward calls by name, arity, and parameter names are
represented. Signature
collection also infers conservative result types from visible return
statements and final expression statements, threading local `let` bindings
and conservative `if`/`else` branch joins through the function body while
merging disagreement to `unknown`. Checked
examples cover valid and invalid expressions, list element checking, list
indexing success and index operand mismatch, supported field access and
unsupported concrete field diagnostics, supported method calls and unsupported
concrete method diagnostics,
undeclared assignment rejection,
top-level `break`/`return` rejection, loop control inside loops, valid function
calls, explicit and implicit inferred call result typing, and arity mismatch
rejection.
`checkProgramDetailed` mirrors the executable checker with
`Except CheckError`, preserving static failure reasons such as undeclared
variables/functions, unary and binary operand mismatches, concrete non-boolean
control-flow conditions, assignment mismatches, arity mismatches, duplicate
top-level function names, duplicate function parameters, unknown or duplicate
named arguments, nested function declarations, and invalid `return`, `break`,
or `continue` placement. The option-returning `checkProgram` wrapper delegates
to the detailed checker and erases the error payload, so both checker APIs
enforce the same duplicate-signature, argument, placement, and declared-return
contracts. Declared-return checking also walks explicit `return` paths and
final-expression returns inside branch and loop bodies before consulting the
merged return summary, so a concrete mismatch in one branch cannot be hidden by
an inferred `unknown` merge. An `if` without an `else` contributes an implicit
unit path only when the conditional is the end of the current block; when later
statements exist, the missing branch falls through to those statements. Thus
non-unit functions must cover both branches explicitly at function-exit points.
A `while` body is also checked with a zero-iteration
fallthrough path, so a non-unit function cannot rely on a loop body as its only
return source. Integer-range `for` loops have the same fallthrough rule because
the range may be empty. Conditional `seal until` bodies also have a skip path
when the exit condition is already true; bare `seal` remains body-only in the
current declared-return checker.
Checked examples cover each major diagnostic class.
Richer source-language type syntax, loop-carried environment joins, and
preservation/progress theorems remain future work.

For `if`/`else`, the static checker joins variables introduced by both branches
only when their types are compatible. Variables introduced by just one branch,
or introduced with incompatible branch types, are not exposed after the
conditional. Existing `unknown` variables assigned compatible concrete values
in both branches are refined by the join; incompatible branch assignments leave
the original imprecise type. Function result inference uses the same join rule
before checking final expression statements.

On successful assignment, the static checker updates the variable environment:
assigning a concrete value to an `unknown` variable refines that variable to
the concrete type for later statements, including nested `list[unknown]`
element refinement. Existing concrete assignment targets keep their declared
static type.

`StepBlock` models ordered statement execution. Value-producing statements
continue to the next statement, the final statement's value is preserved as the
block value, and `return`, `break`, or `continue` stop the block immediately.
`StepStmt` and `StepBlock` are mutually defined so structured `if` statements
can evaluate the selected branch as a block: truthy conditions step through the
then branch, falsey conditions step through the `else` branch when present, and
a falsey condition without `else` produces `unit` without changing the
environment. The same big-step relation includes `while`: falsey conditions
produce `unit`, value-producing bodies recurse, `return` propagates, `break`
exits with `unit`, and `continue` recurses to the next condition check.
Big-step `forRange` binds the iterator to each ascending integer value,
rebinds it to the stop value on normal completion, recurses after ordinary
values or `continue`, exits with `unit` on `break`, and propagates `return`.
Big-step `seal until` stops when its condition is truthy, otherwise executes
the body and recurses after ordinary values or `continue`; bare `seal` recurses
after ordinary values or `continue`. Both forms exit with `unit` on `break` and
propagate `return`. In this env-only big-step relation, `fnDecl` is a
unit-producing statement with no variable-environment effect. Full
function-environment behavior is modeled by the bounded executable `FnEnv`
semantics below and remains a target for a future Prop relation that carries
function bindings explicitly.

`EvalExprWithFnsRel`, `EvalArgsWithFnsRel`, `StepStmtWithFns`, and
`StepBlockWithFns` are the first Prop-level function-environment semantics.
They carry `FnEnv` through statement and block stepping, bind `fn`
declarations into that environment, evaluate positional call arguments, bind
parameters into a call frame, and treat either an ordinary function-body value
or an explicit `return` as the call expression result. Checked call witnesses
cover both explicit `return` and implicit final-expression results. The current
checked slice covers literals, variables, list construction, list/string indexing,
field access through the shared field evaluator, pure `len`, `is_empty`,
string `first`/`tail`/`last`/`take`/`drop`/`reverse`/`at`/`contains`/`starts_with`/`ends_with`, and list `first`/`tail`/`last`/`at`/`take`/`drop`/`reverse`/`append`/`prepend`/`concat`/`join`/`contains` method calls through the shared method evaluator, unary and binary operators through the shared operator
evaluators, function calls, `let`, expression statements, `return`,
declaration sequencing, assignment to existing variable bindings, and
structured `if`/`else` branching with both variable and function
environments threaded through the selected branch. It also covers `while`:
falsey conditions produce `unit`, value-producing bodies recurse, `return`
propagates, `break` exits with `unit`, and `continue` recurses to the next
condition check while preserving both environments. `break` and `continue`
statements now propagate through function-aware blocks. Function-aware
`forRange` binds the iterator to each ascending integer value, rebinds it to
the stop value on normal completion, recurses after ordinary values or
`continue`, exits with `unit` on `break`, and propagates `return` while
threading `FnEnv`. Function-aware `seal until` stops on a truthy condition,
otherwise executes the body and recurses after ordinary values or `continue`;
bare `seal` recurses after ordinary values or `continue`. Both `seal` forms
exit with `unit` on `break` and propagate `return` while preserving both
environments. Initial executable correspondence witnesses check that selected
`EvalExprWithFnsRel` facts for numeric literals, booleans, variables, unary
operators, binary operators, list construction, indexed access, field access,
method calls including `is_empty`, string `first`/`tail`/`last`/`take`/`drop`/`reverse`/`at`/`contains`/`starts_with`/`ends_with`, and list `first`/`tail`/`last`/`at`/`take`/`drop`/`reverse`/`append`/`prepend`/`concat`/`join`/`contains`, explicit-return function calls, and implicit final-expression
function calls agree with `evalExprWithFns` on concrete examples; full
inductive correspondence remains future work.
Initial statement-level executable witnesses check that selected
`StepStmtWithFns` facts for `let`, assignment, expression, `fn` declaration,
`return`, `break`, and `continue` statements agree with projected
`execStmtWithFns` results on concrete examples. Statement checks compare
observable environments and flow, with function declarations checking that a
function binding is added without requiring equality over the function body
payload.
Initial block-level executable witnesses check selected `StepBlockWithFns`
facts for empty blocks, single-statement blocks, ordinary value sequencing, and
early `return`/`break`/`continue` propagation against projected
`execBlockWithFns` results.
Structured statement executable witnesses now also check selected
`StepStmtWithFns` `if` facts for true branches, false branches with `else`, and
false branches without `else` against projected `execStmtWithFns` results.
Loop statement executable witnesses currently check selected non-recursive
`while` facts: false conditions exit with `unit`, body `return` propagates, and
body `break` exits with `unit`.
They also check selected non-recursive `forRange` facts: completed ranges bind
the iterator to the stop value, body `return` propagates, and body `break`
exits with `unit`.
Recursive `forRange` witnesses now additionally cover ordinary value-body
iteration into the completed range case and body `continue` advancing to the
next range value.
`seal until` executable witnesses cover already-satisfied conditions, ordinary
body-value recursion into completion, and body `break` exiting with `unit`.
They also cover body `return` propagation and body `continue` rechecking the
condition before completing.
Bare `seal` executable witnesses cover ordinary value-body recursion into a
later `break`, direct body `return` propagation, and body `break` exiting with
`unit`; they also cover body `continue` advancing to the next bare-seal
iteration.

`evalExprWithFns` and `execBlockWithFns` extend the executable core with a
bounded function environment. `Stmt.fnDecl` binds a function definition,
`Expr.call` preserves positional and named argument nodes, evaluates argument
payload expressions in source order in the caller environment, binds named
arguments to matching function parameters while preserving positional call
behavior, executes the function body, and treats either explicit `return` or
the body's final value as the call result. Checked examples
cover explicit return, implicit final-expression return, arity mismatch, and
parameter shadowing that preserves the caller's outer binding. The bounded
statement executor also evaluates `if`/`else` by running the selected branch as
a block and returning `unit` for a falsey condition without `else`. Bounded
`while` execution rechecks the condition each iteration, returns `unit` on
normal completion or `break`, treats `continue` as the next iteration, and
preserves `return` flow for enclosing function calls. Bounded `forRange`
execution binds the iterator for each ascending integer value `start <= i <
stop`, rebinds the iterator to `stop` on normal completion, treats `continue`
as the next integer, exits with `unit` on `break`, and preserves `return` flow.
Bounded `seal until` execution checks the condition before each iteration and
stops with `unit` when it becomes truthy; bare `seal` repeats until fuel is
exhausted or control flow exits. Both forms treat `continue` as the next
iteration, exit with `unit` on `break`, and preserve `return` flow.

`Aether.VM` is the first formal bytecode model. It currently covers the
proof-core stack instructions needed for constants, locals, arithmetic,
unary operations, dynamic list construction, list indexing, unconditional jumps,
conditional false jumps, and halt.
It also includes `compileExpr` for closed literal, unary, and binary
expressions, with checked examples showing that running compiled bytecode
produces the same value as direct expression evaluation for representative
integer arithmetic, float arithmetic, boolean, string, and list cases.
`compileExprWithSlots` adds explicit variable to
local-slot lookup for expression compilation against VM locals, with checked
examples for successful variable loading and missing-slot failure. Stack and
frame expression compilation both lower indexing as target bytecode followed by
index bytecode and an index opcode, and lower field access as target bytecode
followed by a field opcode. Pure method calls lower target and argument
bytecode followed by a method opcode. Checked examples include zero-argument
`is_empty` calls on strings and lists, string `first`/`tail`/`last`/`take`/`drop`/`reverse`/`at`/`contains`/`starts_with`/`ends_with` calls, and list `first`/`tail`/`last`/`at`/`take`/`drop`/`reverse`/`append`/`prepend`/`concat`/`join`/`contains` calls. `FrameOp`,
`CallFrame`, `FrameState`, `frameStep`, and `runFrame` add a Lean call-frame VM
surface for direct bytecode with `CALL`/`RET` behavior. Checked examples cover
argument passing, explicit return values, implicit unit return, and restoring
caller locals after a function call. `compileFrameProgram` hoists top-level
`fn` declarations after main bytecode, compiles calls to absolute
`CALL target arity` instructions after normalizing named function-call
arguments into parameter order, compiles `return` to `RET`, and runs the
resulting program with `runCompiledFrameProgram`. `FrameOp.jmp` and
`FrameOp.jmpIfFalse` support structured `if`/`else` and bounded `while`
compilation inside frame-compiled functions. Checked examples verify emitted
bytecode, function return values stored by callers, parameter shadowing that
preserves caller locals, branch execution inside a function, and loop execution
inside a function. Frame compilation also supports integer-range `for` loops,
conditional `seal until`, and bare `seal` using the same relative jump scheme
as the non-frame VM; checked examples cover `for` accumulation and conditional
`seal` execution inside functions. Frame compilation carries pending
`break`/`continue` jump sites through nested blocks and branches, then patches
them at the nearest `while`, integer-range `for`, or `seal` loop boundary.
Checked examples cover `break` exiting a compiled function loop and `continue`
skipping the rest of the current loop body. `compileCheckedFrameProgram`
composes `Aether.Static.checkProgramDetailed` with frame compilation, giving
the first checked AST-to-bytecode entrypoint. A Lean theorem records that
successful checked compilation implies the detailed static checker accepted the
source program. `compileCheckedFrameSource`, `runCheckedFrameSource`, and
`checkedFrameSourceLocal?` add the corresponding source-string pipeline:
tokenize, parse, statically check, lower to frame bytecode, and run. Checked
examples show that valid function source code runs through the pipeline while
statically invalid numeric/boolean arithmetic, non-boolean control-flow
conditions, function arity mismatches, and malformed source are rejected before
bytecode execution. Duplicate functions
and duplicate parameters are rejected by the checked compiler path as well as
the diagnostic source pipeline. Checked source examples also verify that the
`🦭 until` alias parses, statically checks, lowers, and runs like `seal until`,
and that string/list `.is_empty()` calls execute through the checked frame
compiler, as do list `.first()`, `.tail()`, `.last()`, `.at(index)`,
`.take(count)`, `.drop(count)`, `.reverse()`, `.append(value)`, `.prepend(value)`, `.concat(other)`, `.join(separator)`, and `.contains(value)` calls, plus string
`.first()`, `.tail()`, `.last()`, `.take(count)`, `.drop(count)`, `.reverse()`, `.at(index)`, `.contains(value)`, `.starts_with(prefix)`, and `.ends_with(suffix)` calls.
General object method calls and full compiler/VM correspondence proofs remain
future work.

`Aether.Pipeline` wraps the source pipeline in `Except Pipeline.Error` so
failures keep their phase. The current phases are `lex message SourceSpan`,
`parse ParseError SourceSpan`,
`static CheckError (Option SourceSpan)`, `compile`, and `runtime`. `parseSource` tokenizes first
with `tokenizeLocated` and reports the first lexer error token with its source
span before invoking the parser. Parser failures carry both the parser
context and the source span of the failed statement start. The located
pipeline replays statement parsing over `LocatedToken` values, preserving the
parser's token-kind API while still reporting later statement failures at their
own line and column range. Parser failures for malformed type annotations point
at the token where the type should begin. Static failures are also rendered
with best-effort source ranges by matching checker errors back to located
tokens for variables, functions, operators, named arguments, duplicate names,
and invalid control-flow conditions. Non-boolean `if`, `while`, and
`seal until` diagnostics prefer the offending condition token when its concrete
type can be matched.
Declared return mismatches prefer the mismatched return expression token,
including explicit `unit` literals; if a non-unit function implicitly returns
unit because no return value is present, the diagnostic points at the closing
brace of the function body when that brace can be matched.
The variable matcher treats the reserved `self` token as the variable name
`self`, and treats `embed`/`convergence` keyword-call tokens as function names,
so undeclared diagnostics for those names retain a source range.
Duplicate function diagnostics use `fn name` token structure, and duplicate
parameter diagnostics search within function parameter lists, so repeated names
in function bodies do not steal the diagnostic range.
Field and method mismatch diagnostics prefer the member identifier after `.`
when it can be matched, falling back to the dot token only if the member name
cannot be recovered from the located token stream.
`compileSource` then parses, runs `checkProgramDetailed`, and lowers to frame
bytecode; `runSource` executes the lowered program; `sourceLocal?` exposes a
checked local value for examples. Checked examples distinguish lexical string
termination, string literal execution, positioned parse failure, concrete static
numeric/boolean/string misuse, concrete static arity mismatch, successful
execution, and fuel-limited runtime state. `errorString`,
`parseSourceErrorString`, `checkSourceErrorString`,
`compileSourceErrorString`, `runSourceErrorString`, and
`sourceLocalErrorString` provide deterministic string rendering for pipeline
errors, including positioned lexical and parser messages, concrete static
diagnostics, compile failures, and runtime/local-access failures. Lexical and
parse renderers print half-open source ranges such as `1:1-1:4`; static
renderers include a range when token lookup can identify a stable source
location. Parser diagnostic rendering names the proof-core lexer keyword
tokens, including domain and object/module keywords, instead of collapsing them
to a generic token label.

`compileStmtWithSlots` covers the
straight-line statement subset: declarations allocate or reuse a local slot and
emit `STORE`, assignments require an existing slot, and expression statements
leave the expression value on the stack. Branching, loops, and functions remain
separate proof targets for the straight-line compiler. `compileBlockWithSlots`
threads the slot table through a sequence of supported straight-line statements
and concatenates their bytecode; unsupported statements fail compilation.
`compileStmtWithBranches`
and `compileBlockWithBranches` add `if` lowering with `JMP_IF_FALSE` and `JMP`,
with checked examples for both true and false branch execution. The same
compiler layer lowers bounded `while` execution to condition bytecode,
`JMP_IF_FALSE` over the body and back jump, body bytecode, and a negative `JMP`
back to the condition, with checked examples for zero-iteration and
multi-iteration execution. It also lowers integer-range `for` loops by
initializing the iterator slot, checking `iterator < end`, running the body,
incrementing the iterator, and jumping back to the condition; checked examples
verify emitted bytecode and final locals for `0..3`. `seal until` lowers to a
pre-body exit check using boolean negation plus `JMP_IF_FALSE`, while bare
`seal` lowers to an unconditional body/back-jump loop; checked examples cover
conditional execution and bare-loop bytecode. Loop-control patching for
`break`/`continue`, functions, and call frames remain separate proof targets.
