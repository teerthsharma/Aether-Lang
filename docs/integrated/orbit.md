# Orbit Partitions

Module: `aether_core::orbit`, in `crates/aether-core/src/orbit.rs`. It computes the orbit partition of a many-to-one map, and the five bounds that partition certifies against every injective ground truth. Evidence: 15 tests in `tests/orbit.rs`.

## Object

A map \(f : E \to A\) on a finite set \(E = \{0, \ldots, n-1\}\) partitions \(E\) into its fibres, the classes of

\[
e_1 \sim e_2 \iff f(e_1) = f(e_2).
\]

The fibres are \(H_0\) of that equivalence relation. Following the source they are called *orbits*, and \(m\) denotes their number. The same object arises from a relation known only pairwise, as the connected components of a graph. `Partition` is therefore built either from values (`from_map`) or from edges by union-find (`from_edges`).

### Notation

| Symbol | Meaning |
| --- | --- |
| \(R : E \to A\) | The unknown ground relation. *Injective* when distinct elements have distinct correct values |
| \(\operatorname{err}(f)\) | \(\lvert\{e : f(e) \ne R(e)\}\rvert\), the number of wrong answers |
| \(G = R(E)\) | The set of correct values, without the pairing |
| \(m^*\) | \(\lvert f(E) \cap G\rvert\), the number of distinct answered values that are correct for some element |
| \(S\) | \(\{e : \lvert[e]\rvert > 1\}\), the certified set: elements in an orbit of size at least two |
| \(b\) | The number of orbits of size at least two |
| \(b_{\text{adm}}\) | The number of those orbits whose shared value lies in \(G\) |
| \(S^*\) | \(S \cup \{e : f(e) \notin G\}\) |
| \(m_{\text{join}}\) | The block count of the join (common refinement) of several partitions |

## The five bounds

Each bound holds for every injective \(R\) consistent with the observation, and none consults \(R\). The sharpened forms additionally read \(G\), which is a set of values, not an answer key. Theorem numbers are caustic's.

**Bound 1: error floor** (Theorems 1 and 1\*; branchcut Theorem 1).

\[
\operatorname{err}(f) \;\ge\; n - m^* \;\ge\; n - m .
\]

A map that is constant on an orbit agrees with an injective \(R\) on at most one of its members. An orbit of size \(s\) therefore holds at least \(s - 1\) errors, and \(\sum_i (s_i - 1) = n - m\). With \(G\): the correct set \(C\) has \(f|_C = R|_C\) injective and \(f(C) \subseteq f(E) \cap G\), so \(\lvert C\rvert \le m^*\). The bound \(n - m\) is attained exactly when every orbit contains one correct value.

**Bound 2: pooling recovery** (Theorem 2; branchcut Theorem 2). For an orbit of size \(k\), any decoder \(h : A \to E\), and the uniform prior on that orbit,

\[
\Pr\big[h(f(e)) = e\big] \;\le\; \frac{1}{k},
\]

because \(h \circ f\) is constant on the orbit. The decoder that returns one fixed member attains it.

**Bound 3: join recovery** (Theorem 2\*). For maps \(f_1, \ldots, f_T\) whose join has \(m_{\text{join}}\) blocks, and any decoder \(h\) of the value tuple,

\[
\frac{\big\lvert\{e : h(f_1(e), \ldots, f_T(e)) = e\}\big\rvert}{n} \;\le\; \frac{m_{\text{join}}}{n},
\]

because \(h\) is constant on each join block. The join is at least as fine as every component, so \(m_{\text{join}} \ge \max_t m_t\). This ceiling is never below any single coordinate's. One recovery per join block attains it.

**Bound 4: precision of the certified set** (Theorems 6 and 6\*).

\[
\frac{\lvert S \cap \text{wrong}\rvert}{\lvert S\rvert} \;\ge\; \frac{\lvert S\rvert - b_{\text{adm}}}{\lvert S\rvert} \;\ge\; \frac{\lvert S\rvert - b}{\lvert S\rvert} \;=\; \frac{n - m}{\lvert S\rvert}.
\]

Every error that Bound 1 counts lies in \(S\), since a singleton contributes \(s - 1 = 0\). An orbit whose value is not in \(G\) has no correct member at all. Every counted orbit has at least two members, so \(\lvert S\rvert \ge 2b\), and the floor is at least \(\tfrac12\) whenever \(S\) is non-empty. It is attained by the truth that places one correct member in every admissible collapsed orbit.

**Bound 5: recall of \(S^*\)** (Theorem 8). Whenever \(\operatorname{err}(f) \ge 1\),

\[
\frac{\lvert S^* \cap \text{wrong}\rvert}{\lvert \text{wrong}\rvert} \;\ge\; \frac{n - m^*}{n},
\]

because the \(n - m^*\) errors of Bound 1 all lie in \(S^*\), and \(\lvert\text{wrong}\rvert \le n\). The bound is attained, and it is zero exactly when \(m^* = n\). That zero is Theorem 7. Suppose \(f\) is a bijection onto \(G\). Then the truths \(R = f\) and \(R = f \circ \sigma\), for a fixed-point-free permutation \(\sigma\), produce the same observation. Recall is 1 under the first and 0 under the second, so no *uniformly* positive recall floor exists.

## Proved, estimated, and refused

All five inequalities are proved in the source by the elementary counting arguments sketched above. The tests pin each inequality on seeded random instances and exhaustive small cases. They check the proofs and do not replace them. Precision is bounded from the partition alone. Recall is bounded only with \(G\), and never uniformly away from zero. The collision score \((\lvert[e]\rvert - 1)/(n - 1)\) is a per-element score, not a bound.

The certificate is one-sided. It can prove a map wrong and never proves one right.

**Hypothesis.** Injectivity of \(R\) is the hypothesis of every bound, and the partition cannot decide it. On a many-to-one relation, distinct elements *should* share a value, and then \(n - m\) certifies nothing. Injectivity is the caller's assertion. The one violation visible from data is \(\lvert G\rvert < n\), on which `admissible_distinct` and `admissible_collapsed_blocks` return `None`.

**Refusals.** Every bound returns `None` on counts that no partition of these elements can produce:

- \(n = 0\);
- \(m \notin [1, n]\), or \(m^* > n\);
- \(k = 0\), or \(m_{\text{join}} \notin [1, n]\);
- an empty \(S\), whose precision is undefined rather than 1;
- \(\lvert S\rvert > n\), or \(\lvert S\rvert < n - m\);
- \(2\,b_{\text{adm}} > \lvert S\rvert\).

## Rust API

```rust
pub struct Partition { /* canonical: smallest member and size per element */ }
impl Partition {
    pub fn from_map<K: Ord>(values: &[K]) -> Self;                           // O(n log n)
    pub fn from_edges(n: usize, edges: &[(usize, usize)]) -> Option<Self>;  // None: vertex ≥ n
    pub fn join(&self, other: &Self) -> Option<Self>;                        // None: different n
    pub fn n(&self) -> usize;
    pub fn m(&self) -> usize;
    pub fn representative(&self, e: usize) -> usize;
    pub fn block_size(&self, e: usize) -> usize;
    pub fn largest(&self) -> usize;
    pub fn is_flagged(&self, e: usize) -> bool;
    pub fn flagged(&self) -> usize;           // |S|
    pub fn collapsed_blocks(&self) -> usize;  // b
    pub fn collision(&self, e: usize) -> Option<f64>;
}

pub fn admissible_distinct<K: Ord>(values: &[K], gold: &[K]) -> Option<usize>;         // m*
pub fn admissible_collapsed_blocks<K: Ord>(values: &[K], gold: &[K]) -> Option<usize>; // b_adm
pub fn orbit_error_bound(n: usize, m: usize) -> Option<usize>;                          // Bound 1
pub fn admissible_error_bound(n: usize, m_star: usize) -> Option<usize>;                // Bound 1*
pub fn certified_error_floor(n: usize, m: usize) -> Option<f64>;                        // (n − m)/n
pub fn pooling_recovery_bound(k: usize) -> Option<f64>;                                 // Bound 2
pub fn join_recovery_bound(n: usize, m_join: usize) -> Option<f64>;                     // Bound 3
pub fn certified_precision_bound(n: usize, m: usize, flagged: usize) -> Option<f64>;    // Bound 4
pub fn admissible_precision_bound(flagged: usize, b_adm: usize) -> Option<f64>;         // Bound 4*
pub fn recall_floor(n: usize, m_star: usize) -> Option<f64>;                            // Bound 5
```

Two partitions compare equal exactly when they have the same blocks, whatever values or edge order produced them.

## Test evidence

`tests/orbit.rs` holds 15 `#[test]` functions. Each one turns a proved statement into an executable assertion. The truth is consulted only by the check, never by the certificate.

```bash
cargo test -p aether-core --test orbit
```

| Test | Pins |
| --- | --- |
| `from_map_matches_a_brute_force_fibre_computation` | 500 seeded maps agree with brute force |
| `from_edges_matches_a_brute_force_closure_for_any_edge_order` | 500 seeds, for any edge order |
| `orbits_are_invariant_under_relabelling_of_elements_and_values` | Orbits belong to the relation, not to names |
| `collision_is_the_fraction_of_other_elements_sharing_the_value` | \((\lvert[e]\rvert - 1)/(n - 1)\) |
| `bound_1_never_exceeds_the_true_error_count` | 2,000 seeds with \(n \le 16\): \(n - m \le n - m^* \le \operatorname{err}\), and \(n - m = \sum (s - 1)\) |
| `bound_1_is_attained_when_every_orbit_holds_one_correct_value` | Tightness |
| `bound_2_no_decoder_beats_one_over_k_and_the_best_attains_it` | \(1/k\) is a ceiling and is attained |
| `bound_3_no_decoder_of_the_tuple_beats_m_join_over_n` | \(m_{\text{join}}/n\) is a ceiling |
| `bound_4_precision_floor_holds_against_every_injective_truth_and_is_attained` | The floor holds against every truth and is attained |
| `bound_5_recall_floor_holds_exhaustively_and_is_attained` | Every truth and every map over a vocabulary of \(n + 2\) values for \(n = 2..4\), then 2,000 seeded instances. The floor is 0 exactly when \(m^* = n\) |
| `bound_5_is_zero_exactly_at_the_shuffle_witness` | Theorem 7's witness |
| `a_gold_set_smaller_than_n_voids_the_certificate_and_is_refused` | \(\lvert G\rvert < n\) returns `None` |
| `empty_and_singleton_inputs_have_the_trivial_partition` | Degenerate inputs |
| `identity_and_constant_maps_sit_at_the_two_ends_of_every_bound` | Extremes of every bound |
| `every_bound_refuses_counts_no_partition_can_produce` | Each refusal listed above |

## Not ported

The source also reports measurements on language models: how often each bound was attained or loose, and a detection score. Those are properties of particular models and datasets, not of this module, and none is restated here. Nothing in the module depends on a language model. `from_map` sorts instead of hashing, so it needs only `Ord` and no allocator-backed map.

## Provenance

- [caustic](https://github.com/teerthsharma/caustic): `caustic/caustic/theorems.py` and `caustic/caustic/regime.py` (T. Sharma, *Caustic*, doi:10.5281/zenodo.21997746). The theorem numbering follows caustic.
- branchcut (private repository): `branchcut/branchcut/partition.py`, which restates the same results over an arbitrary finite set (`partition_by_key`, `components`, `min_errors`, `recovery_ceiling`, `collision_error_floor`).
