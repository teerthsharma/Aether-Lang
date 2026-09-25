# Aether Formal Core

The proof-facing core of Aether-Lang for Lean 4: the executable subset shared by the parser, interpreter, and Titan VM. [Theory →](theory.md#formal)

## Lexical Surface

| Token | Accepted forms |
|---|---|
| Identifiers | ASCII identifiers for variables, parameters, functions |
| Numbers | integer literals; fixed micro-precision decimal floats (`1.5`) |
| Booleans / unit | `true`, `false`; `unit` |
| Strings | escapes `\"`, `\\`, `\n`, `\r`, `\t` |
| Comments | `//` line; nestable `/* */` block; unterminated block = lexer error |
| Lists | `[...]` of core expressions; newlines between elements; trailing comma before `]` |
| Separators | newline (LF, CRLF or CR), `~`, `;` |
| Range | `..` — `1..10` lexes as `Number(1)`, `DotDot`, `Number(10)` |
| Seal alias | `seal` and `🦭` are the same keyword |
| Operators | `+ - * / % == != < > <= >= && \|\| !` |

[Theory →](theory.md#formal-lexical)

## Core Syntax

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

| Precedence (low → high) | Operators |
|---|---|
| 1 | `\|\|` |
| 2 | `&&` |
| 3 | `==`, `!=` |
| 4 | `<`, `>`, `<=`, `>=` |
| 5 | `..` |
| 6 | `+`, `-` |
| 7 | `*`, `/`, `%` |
| 8 | unary `-`, `!` |
| 9 | postfix `expr[expr]`, `expr.ident`, `expr.ident(args?)` |
| 10 | primary expressions |

## Runtime Values

```text
Value ::= Num int | Float int micros | Bool bool | Str string | List Value* | Unit
Flow  ::= Value Value | Return Value | Break | Continue
```

Host values (manifolds, classes, tensors, native functions) sit outside the core. [Theory →](theory.md#formal-values)

## Statement Semantics

Execution maps an environment and a statement to a flow. [Theory →](theory.md#formal-rules)

<div class="ts-viz" data-viz="formal-lookup" data-title="lookup_bind_same and eval_bound_var" data-caption="Bind a name, watch Env.lookup stop at the head cell, and step through the two Lean proofs in Aether/Core.lean 1096-1104."></div>

| Value | Truthy? |
|---|---|
| `Bool(false)`, `Num(0.0)` | false |
| empty string, empty list | false |
| `Unit`, unsupported values | false |
| other booleans, nonzero numbers, non-empty strings and lists | true |

<div class="ts-viz" data-viz="formal-truthy" data-title="truthy, arm by arm" data-caption="Pick a Value constructor and size; the matching arm of truthy in Aether/Core.lean 203-209 fires."></div>

## Static Typing

Types: `num`, `bool`, `str`, `unit`, `list[T]`, `unknown`; `unknown` is accepted wherever a concrete type is still imprecise. [Theory →](theory.md#formal-static-rules)

| Target | Field / pure method | Result |
|---|---|---|
| str, list | `.length`, `.len()` | `num` |
| str, list | `.is_empty()` | `bool` |
| str | `.first()`, `.last()`, `.tail()`, `.reverse()`, `.at(i)`, `.take(n)`, `.drop(n)` | `str` |
| str | `.contains(s)`, `.starts_with(s)`, `.ends_with(s)` | `bool` |
| list[T] | `.first()`, `.last()`, `.at(i)` | `T` |
| list[T] | `.tail()`, `.reverse()`, `.take(n)`, `.drop(n)`, `.append(v)`, `.prepend(v)`, `.concat(xs)` | `list[T]` |
| list[str] | `.join(sep)` | `str` |
| list[T] | `.contains(v)` | `bool` |

## Bytecode Correspondence

| Construct | Bytecode |
|---|---|
| Numeric / boolean / unit constants | `PUSH`, `PUSH_BOOL`, `PUSH Unit` |
| Locals | `LOAD`, `STORE` |
| Arithmetic and logic | `ADD SUB MUL DIV MOD NEG EQ NEQ LT GT LE GE AND OR NOT` |
| Lists, strings, fields, methods | list construction, indexing, `.length`, every pure method in the table above |
| Branching | `JMP`, `JMP_IF_FALSE` |
| `break` / `continue` | patched `JMP` to loop exit / continuation |
| Functions | `CALL(target, arity)`, `RET` |
| Program end | `HALT` |

First VM proof target: compiler output for a well-formed program never underflows the stack. [Theory →](theory.md#formal-bytecode)

## Current Boundaries

Not yet in the formal core:

- Classes, methods, object creation, modules, imports.
- Manifold, block, render, regress, topology-specific host operations.
- Named function arguments in user-defined calls.
- General object/class field access beyond proof-core `.length`.
- Mutating or host/object method calls beyond proof-core pure `.len()`.
- Tensor and ML model handles.
- Forward function references before declaration in VM lowering.
- Global variable capture inside VM user functions.

## Lean 4 Formalization

```text
lake build
```

| File | Contents | Theory |
|---|---|---|
| `lakefile.lean` | Lake package `aether-formal` | |
| `lean-toolchain` | Lean toolchain pin | |
| `Aether.lean` | top-level import | |
| `Aether/Lexer.lean` | token kinds, `tokenize`, `tokenizeLocated` with `SourceSpan` | [→](theory.md#formal-lean-lexer) |
| `Aether/Parser.lean` | token-to-core parser, `parseProgramDetailed` | [→](theory.md#formal-parser) |
| `Aether/Static.lean` | well-formedness and type checking, `checkProgramDetailed` | [→](theory.md#formal-static) |
| `Aether/Core.lean` | syntax, values, envs, eval, `StepStmt`/`StepBlock`, `FnEnv` relations | [→](theory.md#formal-bigstep), [→](theory.md#formal-fnenv), [→](theory.md#formal-bounded) |
| `Aether/VM.lean` | stack VM, frame VM, `compileCheckedFrameProgram` | [→](theory.md#formal-vm), [→](theory.md#formal-slots) |
| `Aether/Pipeline.lean` | stage-aware lex/parse/static/compile/runtime diagnostics | [→](theory.md#formal-pipeline) |

<div class="ts-viz" data-viz="formal-deps" data-title="What imports and uses what" data-caption="Module imports from the Lean sources plus the three theorem kinds; hover or tab to a node to trace its dependencies."></div>

## Executable Witnesses

Each witness checks a Prop-level fact against the bounded executor on a concrete example. [Theory →](theory.md#formal-witnesses)

<div class="ts-viz" data-viz="formal-witness" data-group="expr" data-title="Expression witnesses" data-caption="Each theorem is proved by intro _ then native_decide; this replays the bounded evaluator and lets the fuel drop below Lean's."></div>

<div class="ts-viz" data-viz="formal-witness" data-group="stmt" data-title="Statement witnesses" data-caption="Statement-level witness theorems, replayed step by step against the Lean right-hand side."></div>

<div class="ts-viz" data-viz="formal-witness" data-group="block" data-title="Block witnesses" data-caption="Block-level witnesses: value sequencing continues, return, break and continue stop the block."></div>

<div class="ts-viz" data-viz="formal-witness" data-group="if" data-title="If witnesses" data-caption="The three if/else witness theorems and the branch the executor takes."></div>

<div class="ts-viz" data-viz="formal-witness" data-group="loop" data-title="Loop witnesses" data-caption="while, for and seal witnesses; lower the fuel to see the bounded executor return none."></div>

## Checked Compilation

A successful checked compile implies the static checker accepted the source. [Theory →](theory.md#formal-vm)

<div class="ts-viz" data-viz="formal-checked" data-title="compileCheckedFrameProgram_static_ok" data-caption="The one VM theorem, Aether/VM.lean 804-814: a successful checked compile implies the static checker accepted the program."></div>
