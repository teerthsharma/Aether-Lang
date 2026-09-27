# Rounding Certificates

Module: `aether_core::certify`, in `crates/aether-core/src/certify.rs`. It certifies top-k sets, argmins and threshold decisions against floating-point rounding. Evidence: 17 tests in `tests/certify.rs`.

A score evaluated in floating point is not the value its formula defines. If two candidates lie closer together than the rounding of the kernel that scored them, the arithmetic decided which one was returned, not the data. For each decision the module does one of two things. It proves that the returned top-k set, argmin or threshold side is the one exact arithmetic returns on the stored inputs. Or it refuses and names the indices it cannot separate.

## Object

Fix a working format with unit roundoff \(u\): \(2^{-11}\) for binary16, \(2^{-24}\) for binary32, \(2^{-53}\) for binary64. Let \(q \in \mathbb{T}^d\) be a stored query and \(x_1, \ldots, x_n \in \mathbb{T}^d\) the stored corpus. Here \(\mathbb{T}\) is the set of values the format represents, and \(d\) is the vector length. Every kernel scores squared Euclidean distance. \(s_j = \lVert q - x_j \rVert^2\) is the exact real value on the stored inputs, \(D_j\) is the value evaluated in the working format, and \(R_j\) is a radius with

\[
\lvert D_j - s_j \rvert \le R_j .
\]

### Error model

Every operation whose result is normal satisfies \(\mathrm{fl}(a \circ b) = (a \circ b)(1 + \delta)\) with \(\lvert\delta\rvert \le u\). A product of at most \(n\) such factors lies in \([1 - \gamma_n,\ 1 + \gamma_n]\) (Higham, 2nd ed., Lemma 3.1), where

\[
\gamma_n = \frac{nu}{1 - nu}, \quad \text{evaluated in binary64 and rounded upward.}
\]

A subnormal product carries an absolute error of at most half the smallest subnormal \(\sigma\), and a subnormal sum is exact. So every radius carries the unconditional term \(\eta_d = 4(d + 2)\,\sigma\).

### The three kernels

The Gram identity \(D = \mathrm{fl}\big((\lVert q\rVert^2 + \lVert x\rVert^2) - 2\langle x, q\rangle\big)\) routes every summand through at most \(d + 2\) roundings. By Higham's Theorem 3.1:

\[
\lvert D - s \rvert \;\le\; \gamma_{d+2}\Big(\lVert x\rVert^2 + \lVert q\rVert^2 + 2\sum_{l} \lvert x_l\rvert\,\lvert q_l\rvert\Big) \;\le\; \gamma_{d+2}\,\big(\lVert x\rVert + \lVert q\rVert\big)^2 .
\]

The first form is `GramTight` and the second, which follows by Cauchy–Schwarz, is `GramCheap`. The direct sum \(D = \mathrm{fl}\big(\sum_l \mathrm{fl}(q_l - x_l)^2\big)\) has no cancellation, so its bound is relative (`Direct`). The rounded difference enters squared, which is two roundings. The product is one more and the sum at most \(d - 1\), so

\[
\lvert D - s \rvert \;\le\; \gamma_{d+2}\, s \;\le\; \frac{\gamma_{d+2}}{1 - \gamma_{d+2}}\, D .
\]

Norms are evaluated in binary64. A final inflation absorbs the radius's own rounding:

\[
R \leftarrow \operatorname{next\_up}\!\Big(R\,\big(1 + \gamma_{d+2}^{(64)} + 8u_{64}\big) + \eta_d\Big).
\]

## The certificate

Let \(T\) be the indices of the \(k\) smallest entries of \(D\). \(T\) is certified when

\[
\max_{i \in T}\,(D_i + R_i) \;<\; \min_{j \notin T}\,(D_j - R_j) \qquad \text{(strict).}
\]

\(T\) is then the top-k set of every vector in the box \(\prod_i [D_i - R_i,\ D_i + R_i]\), and the exact scores are among them. Any other evaluation of the same formula on the same stored inputs has the same property, provided the bound covers its error. That includes another reduction order, blocking, fused multiply-add and a different thread count. All of these return the same \(T\).

- The argmin is \(k = 1\). The largest-k set and the argmax negate the scores and reuse the one rule. Radii are unsigned and carry over unchanged.
- Order within \(T\) is certified only on request. It requires the \(k - 1\) adjacent enclosures to be disjoint, and transitivity covers the remaining pairs.
- A threshold \(t\), in score units, is decided per score. The score is above \(t\) when \(D_i - R_i > t\), below it when \(D_i + R_i < t\), and undetermined otherwise. The result is a trit, not a boolean.
- Every endpoint \(D_i \pm R_i\) is rounded outward by one ulp before it is compared. The source compares round-to-nearest endpoints.

## Certified, and not

| Certified | Not certified |
| --- | --- |
| The rounding of the named formula did not choose this set, this index or this side. This holds on the stored inputs, in the declared precision. | That the stored inputs are correct, or anything about how they were produced. For example, an embedding from a binary16 forward pass carries error orders of magnitude above the binary32 rounding certified here. |
| Order within a top-k set, when `ordered` is requested | Order within a top-k set otherwise |
| Scores and radii from `enclose_scores` | Scores and radii supplied from outside, beyond the caller's claim that the radii bound the scores. The rule cannot check that claim. |

A refusal is not a finding. It states that this enclosure does not decide the boundary. It does not state that the exact answer differs.

## Negative results

### The source's direct-kernel radius used \(\gamma_{d+1}\), and \(\gamma_{d+2}\) is the sound constant

`separatrix` bounded the direct kernel with \(\gamma_{d+1}\), which counts the rounded difference \(\mathrm{fl}(q_l - x_l)\) once. The difference enters the sum squared, so its rounding passes through twice, and the sound constant is \(\gamma_{d+2}\). The port measured this in the source's own Python: 645 of 199,998 random binary32 pairs escaped the \(\gamma_{d+1}\) radius, and 0 escaped the \(\gamma_{d+2}\) radius. In this repository, `direct_radius_counts_the_rounded_difference_twice` draws 40,000 pairs at each of \(d = 1\) and \(d = 2\). It asserts that none escapes the \(\gamma_{d+2}\) radius and that the \(\gamma_{d+1}\) control is escaped at least once, so the test cannot pass without teeth.

### Comparing the k-th and (k+1)-th enclosures is unsound

A common simplification checks only the rank-\(k\) and rank-\((k{+}1)\) enclosures. That rule is sound when every radius is equal and unsound whenever the radii vary. Take the counterexample

\[
D = [0,\ 1,\ 2,\ 10], \qquad R = [12,\ 0,\ 0,\ 0], \qquad k = 2 .
\]

The rank-2 and rank-3 entries (indices 1 and 2) satisfy \(1 + 0 < 2 - 0\), so the simplified rule certifies \(\{0, 1\}\). Yet the vector \((11, 1, 2, 10)\) lies in the box and has top-2 set \(\{1, 2\}\). The max-in/min-out rule refuses this input. It returns the frontier pair \((0, 2)\) with gap 2 and width 12, and the straddling set \(\{0, 2, 3\}\). The same error in argmin form compares only the runner-up. With \(D = [0, 1, 3]\) and \(R = [0.1, 0.1, 5.0]\), the point \((0.1, 1.0, -1.9)\) lies in the box and has argmin 2. The rule refuses and names \(\{0, 2\}\).

## Refusals

| Variant | Condition |
| --- | --- |
| `BoundaryUndetermined { frontier, straddling }` | The rule fails. `frontier` holds the extreme pair, its outward-rounded intervals, the gap and the width. `straddling` lists every index whose interval crosses the separatrix, a superset of every index whose membership can change inside the box. |
| `NonFiniteInput { operand, index }` (P1) | A non-finite coordinate, score, radius or threshold. The refusal names the operand and index. |
| `BoundVacuous { n }` (P3) | \(nu > \tfrac12\), where the a-priori bound carries no information, or \(\gamma \ge 1\) in the direct kernel's relative form. |
| `RangeUnsafe { headroom, limit }` (P2) | \((\lVert q\rVert + \max_j \lVert x_j\rVert)^2\) exceeds the largest finite value of the working format. This bound dominates every intermediate and is checked before any score is evaluated. |
| `Usage(&str)` | \(k\) outside \(0 < k < n\), radii of the wrong length, a negative radius, or \(\gamma_0\). Each describes the call, not the data. |

## Rust API

```rust
pub enum Precision { F16, F32, F64 }
pub fn unit_roundoff(p: Precision) -> f64;
pub fn gamma(n: usize, p: Precision) -> Result<f64, Refusal>;
pub fn eta(d: usize, p: Precision) -> f64;

pub trait Working: sealed::Sealed + Copy + Into<f64> + Add<Output = Self>
    + Sub<Output = Self> + Mul<Output = Self> {
    const PRECISION: Precision;
    const ZERO: Self;
}                                   // implemented for f32 and f64 only

pub enum Kernel { GramCheap, GramTight, Direct }
pub struct Enclosure { pub scores: Vec<f64>, pub radii: Vec<f64> }

pub fn enclose_scores<T: Working, const D: usize>(
    corpus: &[[T; D]], query: &[T; D], kernel: Kernel,
) -> Result<Enclosure, Refusal>;

pub fn topk_set(scores: &[f64], k: usize, largest: bool) -> Vec<usize>;   // not a certificate
pub fn certified_topk(scores: &[f64], radii: &[f64], k: usize, largest: bool, ordered: bool)
    -> Result<Vec<usize>, Refusal>;
pub fn certified_argmin(scores: &[f64], radii: &[f64]) -> Result<usize, Refusal>;
pub fn certified_threshold(scores: &[f64], radii: &[f64], threshold: f64)
    -> Result<Vec<Trit>, Refusal>;

pub enum Trit { Below = -1, Undetermined = 0, Above = 1 }
pub struct Frontier { pub inside: usize, pub outside: usize, pub inside_lo: f64, pub inside_hi: f64,
                      pub outside_lo: f64, pub outside_hi: f64, pub gap: f64, pub width: f64 }
impl Frontier { pub fn deficit(&self) -> f64; }   // gap − width
```

`radii` holds one radius per score, or a single radius for all of them. `Working` is sealed, because a certificate is only as sound as the unit roundoff its format declares.

## Test evidence

`tests/certify.rs` holds 17 `#[test]` functions. Every certificate is scored against exact integer arithmetic on a dyadic lattice, where the true score of every pair is known before any float runs. Certificates are also checked at the extremum of the error box, where a decision that survives is proved rather than sampled.

```bash
cargo test -p aether-core --test certify
```

| Test | Pins |
| --- | --- |
| `unit_roundoff_and_gamma_are_the_textbook_values_and_refuse_where_vacuous` | \(u\) is exactly \(2^{-11}, 2^{-24}, 2^{-53}\). \(\gamma\) matches the source's published table to \(10^{-6}\) relative. \(\gamma\) refuses past \(nu = \tfrac12\) and not one step early. \(\eta(8, \text{F32}) = 40 \cdot 2^{-149}\) |
| `gram_enclosure_contains_the_exact_score_and_cheap_dominates_tight` | Both Gram radii enclose the exact score on \(60 \times 32\) pairs, and cheap \(\ge\) tight |
| `direct_radius_counts_the_rounded_difference_twice` | 0 escapes of \(\gamma_{d+2}\) and at least one escape of \(\gamma_{d+1}\) over 40,000 pairs at each of \(d = 1, 2\) |
| `no_certified_topk_contradicts_exact_arithmetic_on_adversarial_near_ties` | 300 seeds, three kernels. No certified set differs from the exact set. Rounding changed at least 50 rankings, at least 300 decisions certified, and at least 150 were refused |
| `a_threshold_trit_never_places_a_score_on_the_wrong_side` | A score sitting exactly on \(t\) is `Undetermined`. Every decided side agrees with exact arithmetic |
| `well_separated_data_certifies_and_the_certificate_is_the_exact_set` | Completeness on 200 well-separated seeds, \(k \in \{1, 5\}\) |
| `the_direct_kernel_decides_what_the_gram_identity_cannot` | A pair \(10^{-6}\) apart at norm \(10^6\) cancels to `0.0` under Gram, which refuses. `Direct` certifies |
| `the_boundary_pair_rule_is_a_false_theorem_and_the_refusal_names_the_blocker` | The \([0,1,2,10]\) / \([12,0,0,0]\) counterexample above: frontier \((0, 2)\), gap 2, width 12, straddling \(\{0, 2, 3\}\) |
| `argmin_is_decided_against_every_lower_endpoint_not_the_runner_up` | The runner-up counterexample is refused, naming \(\{0, 2\}\) |
| `a_certified_set_survives_the_worst_corner_of_the_box` | 5,000 random draws. Pushing members outward and non-members inward never changes a certified set. More than 500 certified |
| `order_is_certified_only_when_asked` | A fixed set whose two members can swap inside the box is certified unordered and refused ordered |
| `touching_enclosures_do_not_certify_and_a_zero_radius_certifies_the_floats` | Touching intervals refuse. With radius 0, the rule certifies the floats themselves |
| `the_straddling_set_holds_every_index_whose_membership_can_move` | 1,500 draws, 200 box points each. No index outside the straddling set changes membership |
| `certified_results_are_invariant_under_corpus_permutation` | Scores, radii and certificates permute with the corpus |
| `largest_k_is_smallest_k_of_the_negated_scores` | 4,000 draws with heteroscedastic radii. Frontiers are reported in the caller's orientation |
| `nonfinite_negative_and_empty_inputs_refuse_and_say_where` | NaN, \(\pm\infty\), a negative radius and empty input each refuse with a location |
| `a_range_unsafe_corpus_refuses_before_any_score_is_read` | All three kernels refuse a corpus whose binary32 Gram identity would overflow, which shows the refusal was needed. The same corpus scaled by \(10^{-3}\) encloses |

## Not ported

- **The scaled-integer escalation of `exact.py`.** It needs integers of up to \(2 \cdot 1074\) bits, and so a dependency this crate does not admit. A refusal therefore names the straddling set and does not resolve it.
- **The \(4 \times 4\) canary (P4).** It tests a black-box BLAS path, and this module evaluates its own scores.
- **The per-row radius collapse.** It saves memory only across many queries. The rule accepts a single radius for every score, which is the same collapse supplied by the caller.
- **A binary16 working type.** Rust has no stable binary16 type. `Precision::F16` is reachable through `unit_roundoff`, `gamma` and `eta` for scores evaluated elsewhere.

## Provenance

[separatrix](https://github.com/teerthsharma/separatrix):

- `separatrix/enclose.py`: unit roundoff, \(\gamma\), \(\eta\), the preconditions and both kernels' radii.
- `separatrix/decide.py`: the rule.
- `separatrix/api.py`: argmin and the threshold trit.
- `separatrix/exact.py`: the frontier set.
- `separatrix/verdict.py`: the frontier and the refusal catalogue.

The direct kernel's constant is corrected from \(\gamma_{d+1}\) to \(\gamma_{d+2}\), as recorded above.
