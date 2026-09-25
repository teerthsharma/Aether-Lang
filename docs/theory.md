# Theory

The one page on this site that argues instead of shows. Every other page leads with a visual, a table or a command and links here for the derivation, proof sketch or rationale behind it.

## Topology, ML and the language {#th}

### Derivations {#th-derivations}

The derivations page is deliberately mechanical: each formula names the object, the implementation surface and the claim boundary. For scalar samples \(x(t)\), delay \(\tau\) and dimension \(D\), `TimeDelayEmbedder<D>` produces \(\Phi(t) = [x(t), x(t-\tau), \ldots, x(t-(D-1)\tau)]\). The interpreter fixes \(D=3\), normalizes `tau=0` to `1`, and emits an embedded point only once enough samples exist.

### Block statistics {#th-block-statistics}

For a block \(B = \{x_1,\ldots,x_n\}\) with centroid \(\mu_B\) (`BlockMetadata<D>::from_points`), the mean point-to-centroid distance is

\[
\bar{d} = \frac{1}{n}\sum_{i=1}^{n} d(x_i,\mu_B),
\]

and the distance variance is the second moment minus the squared mean,

\[
\sigma_B^2 = \frac{1}{n}\sum_{i=1}^{n} d(x_i,\mu_B)^2 - \bar{d}^2 .
\]

Concentration \(c_B\) is the mean cosine between each \(x_i\) and \(\mu_B\); zero-norm terms are skipped by implementation guards.

### Cauchy-Schwarz pruning bound {#th-cauchy-schwarz}

Implementation: `BlockMetadata<D>::upper_bound_score`. For a query \(q\) and any point \(x_i \in B\), write \(x_i = \mu_B + (x_i - \mu_B)\) with \(\|x_i-\mu_B\| \le r_B\). Then

\[
q\cdot x_i = q\cdot\mu_B + q\cdot(x_i-\mu_B) \le \|q\|\|\mu_B\| + \|q\| r_B = \|q\|(\|\mu_B\| + r_B).
\]

If this bound is below a threshold, no point in the block can reach it, so the block can be pruned without inspecting every point.

### Governor update {#th-governor}

Implementation: `GeometricGovernor::adapt`. The sparse trigger (`SparseScheduler<D>::should_wake`) uses \(\Delta(t) = \|\mu(t)-\mu(t_{last})\|_2\) against an adaptive threshold \(\epsilon(t)\). The governor adapts \(\epsilon\) as follows.

Observed rate:
\[
R_{actual} = \frac{\Delta(t)}{\epsilon(t)}
\]

Error, as a fraction of the target and clamped so one step is bounded:
\[
e(t) = \mathrm{clamp}\!\left(1 - \frac{R_{actual}}{R_{target}},\, -1,\, 1\right)
\]

The update acts on \(\ln\epsilon\), with the per-step change in error as the derivative term, then clamps \(\epsilon\) into \([0.001, 10]\):

\[
\ln\epsilon(t+1) = \ln\epsilon(t) - \alpha e(t) - \beta\,(e(t) - e(t-1)), \quad \alpha = 0.25,\ \beta = 0.05
\]

Since \(R_{actual}/R_{target} = \epsilon^*/\epsilon\) with \(\epsilon^* = \Delta/R_{target}\), near the fixed point \(e \approx u = \ln(\epsilon/\epsilon^*)\) and

\[
u(t+1) = (1-\alpha-\beta)\,u(t) + \beta\,u(t-1),
\]

whose characteristic roots \(z^2 - 0.70z - 0.05 = 0\), \(z \approx 0.765, -0.065\), lie inside the unit circle. A steady \(\Delta\) therefore converges geometrically to \(\epsilon^*\); a high observed rate raises \(\epsilon\), a low one lowers it. `test_steady_deviation_converges_to_equilibrium` pins this for \(\Delta \in \{5, 50, 2000\}\).

The previous law, \(\epsilon \mathrel{-}= \alpha(R_{target} - R_{actual}) + \beta\,\tfrac{de}{dt}\) with \(\alpha = 0.01\), mixed an error of order \(10^3\) Hz with an \(\epsilon\) of order \(10^{-3}\)–\(10\), and divided the derivative by \(dt \approx 10^{-3}\) s. One step crossed the whole clamp band, so \(\epsilon\) alternated \(0.001, 10, 0.001, \dots\) and the scheduler stopped waking once it sat at 10.

### Binary shape heuristic {#th-shape-heuristic}

Implementation: `crates/aether-core/src/topology.rs`. The gate computes \(density = \beta_0/|B|\) and compares density and approximate loop count against fixed thresholds. Claim boundary: this is a heuristic gate with tests. It is not documented as a production malware detector or a formally complete authentication system, and must not be described as proof of binary safety without external validation, corpora, baselines, and false-positive / false-negative artifacts.

### Persistent homology {#th-persistent-homology}

The engine in `crates/aether-core/src/persistence.rs` consumes a point cloud \(X = \{x_1, \ldots, x_n\}\) of `ManifoldPoint<D>` values, builds a Vietoris-Rips (or lazy witness) filtration up to homology dimension 2, reduces it, and returns a `PersistenceDiagram`.

### Lazy witness mode {#th-lazy-witness}

For lower-load DSL runs, Aether can select landmarks and use all points as witnesses. A simplex filtration value is

\[
f(\sigma) = \min_{w \in X}\left(\max_{\ell \in \sigma} d(w,\ell) - d(w,L)\right)
\]

where \(L\) is the landmark set and \(d(w,L)\) is the distance from witness \(w\) to its nearest landmark. This reduces the selected complex size. It is not the same claim as exact Vietoris-Rips homology over the full point cloud.

### Boundary-matrix reduction {#th-reduction}

The engine sorts simplexes by filtration value and dimension, constructs boundary columns, and reduces them over \(\mathbb{Z}_2\). A reduced empty column births a feature. A later column with a low pivot kills the feature born by that pivot. The resulting pairs are stored as `PersistencePair { dimension, birth, death: Option<f64> }`.

### Betti query {#th-betti}

For radius \(r\), \(\beta_k(r)\) counts the intervals of dimension \(k\) with \(b_i \le r < d_i\). Essential intervals have no death value (`death = None`) and therefore remain live for every \(r \ge b_i\).

### Residual shape heuristic {#th-residual-shape}

For a residual sequence \(r_i = y_i - \hat{y_i}\), the interpreter-level escalating regressor (`EscalatingRegressor::run_escalating`) estimates shape using sign changes and oscillation counts. That is a lightweight residual heuristic, not persistent homology; the persistent-homology path (`topology.ph`, `topology.betti`) is separate.

### Benefit emergence {#th-benefit-emergence}

Aether's benefit model is mechanical. It does not depend on claiming that topology replaces ordinary ML, scheduling, or parsing. The benefit appears when the runtime carries enough structure for a downstream decision to become local, bounded, or auditable: ordinary engineering constraints expose useful behavior when the runtime preserves structure. A benefit is described only through the mechanism that produces it — representation + invariant + gate — and if one part is missing the claim belongs in roadmap text.

### Claim boundaries {#th-claim-boundaries}

Every active claim takes one of three forms: a unit or integration test; a runnable CLI or benchmark artifact; or a docs-only theory or roadmap statement clearly labeled as such. This keeps the project legible without turning planned systems into active claims. Roadmap surfaces (hardware acceleration, production binary-authentication security, end-to-end speedups, full type checking, framework parity, bare-metal bootability) can be implemented, but are not described as active until tests and artifacts cover them. The same rule governs the ML primitives (no claims of beating PyTorch, TensorFlow, sklearn, GUDHI or ripser, production readiness, framework-equivalent semantics, or hardware acceleration without benchmark artifacts and parity tests) and topological convergence (no claim that every training loop terminates by persistent homology, that topology improves model quality on external datasets, that it replaces validation metrics, or that it is benchmarked across model classes).


## Evidence and kernel {#ev}

### Persistence invariants {#ev-invariants}

The invariant suite (`tests/persistence_invariants.rs`) lists the properties that tell a correct persistent homology implementation apart from one that only looks right. Each property is a theorem written as an executable assertion. The stability row is the Cohen-Steiner–Edelsbrunner–Harer bound specialised to Vietoris-Rips: if $\|X - X'\|_\infty \le \varepsilon$ then $d_B(\mathrm{Dgm}(X), \mathrm{Dgm}(X')) \le 2\varepsilon$. The circle row uses the exact regular-polygon chord: the long $H_1$ bar dies at

$$
d = 2r\sin\!\left(\frac{\pi\lceil n/3\rceil}{n}\right),
$$

which equals $\sqrt{3}\,r$ when $3 \mid n$ and tends to it otherwise.

The suite computes bottleneck distance exactly. It binary-searches the candidate cost set and runs Kuhn's augmenting-path matching on each threshold graph, with projection onto the diagonal. Essential-class counts must match exactly.

The suite is not external parity. An implementation that is wrong but self-consistent can satisfy every internal property, so parity needs a bottleneck distance of about 0 against a pinned `ripser`/`gudhi` on shared fixtures.

### Why two diagram mutants first survived {#ev-diagram-mutation}

Two of the five injected diagram defects survived at first. The ordering test used *nested* bars, whose tent values already arrive sorted, so skipping the sort changed nothing. No test referenced $\sigma$ at all, so every other image property held for any fixed kernel width. Both tests were rewritten until the mutants died. The bottleneck mutant that forbids diagonal projection was never run: its infinite costs make the matching search diverge. The rule this leaves: a suite that has not been mutated has unknown strength.

### The routed selector {#ev-routed-selector}

`Selector::TopologicalRouted` splits two jobs that the nearest-neighbour rule mixed together. H0 single-linkage clustering of the **unit-normalised** key directions builds a candidate set, which is norm-invariant by construction. The exact dot product then ranks keys inside that set, which restores the norm sensitivity the geometry threw away.

On uniform keys the router examines every key (cost 0.999 of dense). It becomes dense attention plus clustering overhead, so its +0.94 placement is worth nothing. This is not a clustering bug. Single-linkage chains across a cloud with no density gaps, which is H0 correctly reporting that uniform data has no structure to route on. When the keys do have structure, H0 recovers balanced components and the router gets +0.92 to +0.99 of oracle quality at 0.449× the dense dot-product count. It holds that quality at key-norm spread 8, where the nearest-neighbour rule did worse than random.

The precise claim: topological routing is a real sparsity win exactly when the key distribution has H0 structure, and no win at all when it does not. `routing_is_sparse_only_when_the_keys_have_h0_structure` asserts both halves. The cost contract did not exist until the routed selector posted a 0.999-of-dense "win". Every earlier test measured how good a selection was, and none measured what it cost to make.

### Deciding at runtime, and why the fallback is dense {#ev-routing-plan}

`routing_plan(k, seq, head_dim, clusters, budget, causal)` runs the H0 clustering once per key tensor. That cost is shared by every query, head and layer that reuses the tensor, and the plan reports the routing cost before any query runs. `gap_ratio` is the first merge height above the cut divided by the last one below it. Because it comes from the barcode alone, a runtime can cache that scalar and skip clustering entirely.

The first `Adaptive` fell back to a budget-6 sliding window and measured placement **+0.014** on unstructured keys, no better than random. The fallback code was not at fault. When the keys have no H0 structure, there is no option that is both cheap and good: finding the top-k without computing the scores is exactly what the structure was supposed to allow. So `Adaptive` guarantees it is *never worse than dense, in cost or in quality*, which is the only guarantee safe to enable by default. A caller who prefers to trade quality for cost asks for `Local` explicitly.

The two regimes use different scales on purpose. Placement only means something for budget-limited selectors. Dense attention recovers all of the mass, which sits *above* the budget-limited oracle and makes the ratio blow up. One run reported **+7.6**, which would read as a 700% win over an oracle it never competed with. The unstructured case is therefore asserted as recovered mass.

### Why nearest-neighbour placement is tautological {#ev-nn-ablation}

$$
\|q-k\|^2 = \|q\|^2 + \|k\|^2 - 2\,q\cdot k
$$

When key norms are roughly equal, ranking keys by Euclidean proximity is the same as ranking them by dot product, and uniform random data has roughly equal key norms. So the seed table's +0.86 to +0.94 placement is largely tautological and must not be quoted as a result. Once the key norms vary, the two rankings come apart. Nearest-neighbour in key space is a good proxy for attention mass only when key norms are roughly homogeneous, and real attention does not guarantee that. `the_topological_advantage_collapses_when_key_norms_vary` pins both ends of the curve.

The first run of the ablation was worse: placement −3.6 to −4.2. The cause was an **absolute** `epsilon` of 0.6 against a median query-key distance of 2.4. The selector picked 1.0 keys per row while its baselines picked 5.5, so it lost on budget, not on mechanism. The radius is now relative, and the ablation asserts equal mean budget before it compares mass. A fix to the nearest-neighbour rule needs either key normalisation or a dot-product ranking over a topology-derived candidate set, followed by a re-run of the ablation.

### Scheduled attention port {#ev-scheduled}

`aether_core::scheduled` ports the merged Triton kernel [triton-lang/kernels#22](https://github.com/triton-lang/kernels/pull/22), "Add topology-derived sparse attention kernel". The Python original runs on CUDA. The port runs anywhere `aether-core` does, including `no_std`. It keeps the original's split into a combinatorial CSR schedule and a numeric kernel, and that split is what makes it testable.

The Triton PR measured 56.6% block reduction at seq 1024 and 80.9% at seq 4096 on an RTX 4060, and 1.04×–3.48× sparse-vs-dense-CSR wall-clock. This repository asserts the direction of the reduction at a size a unit test can run. It does not restate the wall-clock numbers, which were measured on hardware this workspace cannot access. The port is a scalar CPU kernel with no SIMD, no threading and no GPU. It reproduces the *answer* and the *block reduction*, not the timing.

**Salience is the elder rule.** Each block records the merge distance at which its component was absorbed, so its score is an H0 death time of the centroid cloud. Exactly one block scores 0, and that follows from an invariant the merge preserves: every component holds exactly one block that has never been written.

**Per-block salience is not permutation-equivariant.** When two components tie on size, index order decides which one is absorbed. So the same centroid can score differently depending on where it sits in the sequence, and the zero-salience block moves too. The **multiset** of saliences is invariant, because it is the H0 barcode. The Triton original has the same tie-breaking, since both follow union-find order. A caller who reorders the sequence gets a different schedule, not a worse one. A fix would need a deterministic tie-break on centroid content instead of on index.

### Why the face lookup mattered {#ev-scale}

The old `find_simplex` did a linear scan, which made the reduction $O(m^2)$ in the simplex count $m$. Indexing the face lookup took `tests/persistence_scale.rs` from 29.07 s to 1.10 s in release, a 26× reduction on identical assertions. The point cap in `scale_probe` is a time budget, not a correctness limit.


## Formal core {#formal}

Theory behind [Formal core](FORMAL_CORE.md): the proof-facing core of Aether-Lang for Lean 4, i.e. the executable subset shared by the parser, interpreter, and Titan VM. Surface forms outside it stay experimental until lowered into this core or given their own proof rules.

### Lexical invariants {#formal-lexical}

The lexer must preserve `1..10` as `Number(1)`, `DotDot`, `Number(10)` rather
than treating it as a malformed float. Decimal literals such as `1.5` parse as
fixed micro-precision proof-core numeric expressions.

### Runtime values beyond the core {#formal-values}

The interpreter supports additional host values such as manifolds, classes,
tensors, and native functions. Those are outside the first Lean 4 core unless
explicitly modeled as opaque external values. `Float int micros` stores decimal
numeric values in fixed micro-precision, matching lexer float tokens.

### Statement rules and call frames {#formal-rules}

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


### Static typing rules {#formal-static-rules}

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

### Bytecode proof target {#formal-bytecode}

The first VM proof target should be stack preservation for well-formed bytecode:
if bytecode is produced by the compiler for a well-formed core program, runtime
stack underflow does not occur.

These features should either lower into the core above or receive separate
semantics before being included in Lean 4 proofs.

### Lean lexer and numerics {#formal-lean-lexer}

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

### Lean parser and parse diagnostics {#formal-parser}

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

### Lean static checker {#formal-static}

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

### Big-step block semantics {#formal-bigstep}

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

### Function-environment semantics {#formal-fnenv}

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

### Executable witnesses {#formal-witnesses}

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

### Bounded executable function semantics {#formal-bounded}

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

### Lean VM and checked frame compiler {#formal-vm}

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

### Source pipeline diagnostics {#formal-pipeline}

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

### Slot and branch compilers {#formal-slots}

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

