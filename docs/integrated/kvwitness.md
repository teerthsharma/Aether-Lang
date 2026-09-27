# Segment Witnesses

Module: `aether_core::kvwitness`, in `crates/aether-core/src/kvwitness.rs`. It adds segment witnesses to a learned sparse top-k. It is a port of the candidate policy in [vllm-project/vllm#47942](https://github.com/vllm-project/vllm/pull/47942), commit `a41354cb34`, "Add topology witnesses to the sparse MLA indexer top-k". That pull request is open, not merged. Evidence: 15 tests in `tests/kvwitness.rs`.

The learned indexer top-k is a density rule. It concentrates where scores are high, and it places no constraint on how its tokens are distributed over the context. The policy adds a covering constraint on top of it. The original has a torch reference and a fused Triton kernel asserted equal to it. This module ports the torch reference, which is also the original's CPU path.

## Object

A row has context length \(L \ge 1\), and the policy has a segment count \(S \ge 1\). Put \(S_{\text{eff}} = \min(S, L)\) and partition \([0, L)\) into \(S_{\text{eff}}\) consecutive intervals:

\[
e_s = \Big\lceil \frac{sL}{S_{\text{eff}}} \Big\rceil, \qquad I_s = [e_s,\ e_{s+1}), \qquad s = 0, \ldots, S_{\text{eff}} - 1,
\]

with the segment map

\[
\sigma(x) = \Big\lfloor \frac{x\,S_{\text{eff}}}{L} \Big\rfloor, \qquad x \in I_s \iff \sigma(x) = s .
\]

A learned row \(K\) occupies \(O(K) = \{\sigma(x) : x \in K\}\). Its witness set is

\[
W(K) = \{e_s : s < S_{\text{eff}},\ s \notin O(K)\}.
\]

## Invariant and consequences

\(\sigma(e_s) = s\) for every \(s < S_{\text{eff}}\): a segment's left edge, rounded up, lies in that segment. This requires \(S_{\text{eff}} \le L\), which is why \(S\) is capped at \(L\) per row. Rounding down does not round-trip: \(L = 10\), \(S = 4\), \(s = 1\) gives 2, and \(\sigma(2) = 0\). Three properties follow without any scan:

1. \(W(K)\) and \(K\) are disjoint, because a witness is emitted only for a segment that \(K\) does not meet.
2. Witnesses are pairwise distinct, because each lies in its own segment.
3. \(O(K \cup W(K)) = \{0, \ldots, S_{\text{eff}} - 1\}\): every segment holds a selected token.

By (3), a maximal run of unselected tokens meets at most two segments, so its length is at most

\[
2\Big\lceil \frac{L}{S_{\text{eff}}} \Big\rceil - 2 .
\]

The pull request describes this as bounding the uncovered run at \(L/S\) tokens. That wording is loose. The bound above is what segment coverage actually implies.

## Implementation rule and contracts

A row is a fixed-width array of `topk` slots. Slots \([0, \texttt{keep})\) are retained verbatim. Witnesses, in ascending segment order, overwrite slots \(\texttt{keep}, \texttt{keep}+1, \ldots\) up to the replacement budget

\[
B = \min(\texttt{max\_replacements},\ \texttt{topk} - \texttt{keep}),
\]

and any witness beyond \(B\) is dropped. The serving entry point uses \(\texttt{keep} = \texttt{topk} - S\) (saturating) and \(\texttt{max\_replacements} = S\).

| Contract | Statement |
| --- | --- |
| Coordinates | Entries are row-relative token offsets (`i32`), not absolute positions in a flattened KV buffer. Row \(r\) has valid range \([0, L_r)\), and its query sits at \(L_r - 1\). `-1` is padding: it marks no segment and passes through unchanged. Every witness satisfies \(0 \le e_s < L_r\) |
| Budget | At most \(B\) slots change, all inside \([\texttt{keep}, \texttt{keep} + B)\). Slots \([0, \texttt{keep})\) are bit-identical to the input. Row width never changes |
| Disabled | \(S = 0\) is the off switch and the default of `VLLM_SPARSE_MLA_TOPOLOGY_SEGMENTS`. \(B = 0\), or zero rows, returns the learned rows unchanged |
| Fallback | \(L_r \le 0\): that row gets no witnesses. \(L_r < S\): \(S_{\text{eff}} = L_r\), every segment is one token, and every token the learned row does not hold becomes a witness |
| Out-of-range learned entry | A learned \(x \ge L_r\) is neither refused nor removed. It marks segment \(\min(\lfloor x S_{\text{eff}}/L_r\rfloor, S - 1)\) occupied, which lies outside the partition when \(S_{\text{eff}} < S\), and it passes through |

## Caveats, pinned as tests

- **Full coverage is not guaranteed after the merge.** Occupancy is read from the whole learned row, including the tail slots that the merge then overwrites. So (3) holds for \(K \cup W(K)\) but not always for the merged row. A segment whose only learned token sits in an overwritten slot is left uncovered. The test `overwriting_a_sole_tail_occupant_uncovers_its_segment` builds the case with \(L = 16\), \(S = 4\): the merged row is `[0, 1, 2, 3, 4, 8, -1, -1]`, with coverage 0.75. The output covers every segment when no overwritten slot is its segment's sole occupant, for example when the reserved tail is padding.
- **The merge is positional.** The witness set is a set function of the learned row. The output set is invariant under permutations of the retained prefix, but not of the whole row.

Both are the source's behaviour, not defects of the port. They are asserted so that neither is later "fixed" by accident.

## Null models and metrics

A covering policy has to be compared at an identical budget \(k = \lvert\{\text{valid output slots}\}\rvert\). There are two nulls:

- a seeded uniform draw of \(k\) tokens from \([0, L)\) (`random_null`);
- the \(k\) most recent tokens, \([L - k, L)\) (`locality_null`).

Both are row \(L - 1\) of the existing `attention::Selector::Random` and `attention::Selector::Local` baselines, called rather than reimplemented. For a candidate set \(C\) and a dense attention row \(a\) over \([0, L)\):

\[
\operatorname{coverage}(C) = \frac{\big\lvert\{\sigma(x) : x \in C \cap [0, L)\}\big\rvert}{S_{\text{eff}}},
\qquad
\operatorname{recall}(C; a) = \frac{\sum_{x \in C \cap [0, L)} a_x}{\sum_{x < L} a_x}.
\]

## What the source measured, and what is not claimed

The pull request body reports the fused CUDA kernel at 1.12× to 1.62× the cost of copying the same buffer. The runs covered 512 down to 32 rows, with `topk = 2048` and \(S = 64\), on an RTX 4060 Laptop GPU (sm_89). It also reports 1,000/1,000 randomised configurations identical to the torch reference. The pull request ran no model-level evaluation. It states that the evidence shows the change is correct, bounded and cheap, not that it is an improvement. This port measures nothing about quality either. `segment_coverage` and `attention_mass_recall` exist so that an experiment can.

## Refusals

`WitnessError` mirrors the source's `ValueError`s:

| Variant | Condition |
| --- | --- |
| `RowsMismatch { learned_len, topk, rows }` | The context lengths do not cover every learned row, so some row's offsets cannot be checked against a range |
| `SegmentsOutOfRange { num_segments }` | Zero segments, or more than `MAX_TOPOLOGY_SEGMENTS` = 64 at the merge. The fused kernel carries occupancy as a 64-bit mask, and this port enforces the same bound so both paths accept the same inputs |
| `LearnedKeepOutOfRange { learned_keep, topk }` | The retained prefix is wider than the row |

A refused call to `scatter_topology_witnesses` leaves the buffer untouched.

## Rust API

```rust
pub const MAX_TOPOLOGY_SEGMENTS: usize = 64;

pub fn segment_of(token: usize, context_len: usize, num_segments: usize) -> usize;
pub fn topology_witness_indices(learned: &[i32], topk: usize, context_lens: &[i32], num_segments: usize)
    -> Result<Vec<i32>, WitnessError>;                   // row-major [rows, num_segments]
pub fn apply_topology_witnesses(learned: &[i32], topk: usize, context_lens: &[i32],
    learned_keep: usize, num_segments: usize, max_replacements: usize)
    -> Result<Vec<i32>, WitnessError>;
pub fn scatter_topology_witnesses(topk_indices: &mut [i32], topk: usize, context_lens: &[i32],
    num_segments: usize) -> Result<(), WitnessError>;   // serving entry point, in place

pub fn random_null(context_len: usize, budget: usize, seed: u64) -> Vec<i32>;
pub fn locality_null(context_len: usize, budget: usize) -> Vec<i32>;
pub fn segment_coverage(candidates: &[i32], context_len: usize, num_segments: usize) -> f64;
pub fn attention_mass_recall(candidates: &[i32], attention_row: &[f64]) -> f64;
```

`segment_of` reads a length of 0 as 1, as the source's `clamp_min(1)` does, so it never divides by zero. `segment_coverage` returns 1 when there are no segments. `attention_mass_recall` is NaN for a row with no mass.

## Test evidence

`tests/kvwitness.rs` holds 15 `#[test]` functions. The tests were written before the module. The first two tests, and the rejection cases, are the source's own hand examples, so a divergence from the Python original fails before any property is consulted.

```bash
cargo test -p aether-core --test kvwitness
```

| Test | Pins |
| --- | --- |
| `a_witness_fills_only_the_segment_the_learned_row_misses` | \(L = 16\), \(S = 4\), learned `[0, 5, 9, -1]` gives witnesses `[-1, -1, -1, 12]` |
| `the_merge_keeps_the_prefix_and_writes_witnesses_into_the_tail` | `keep = 2`, `max_replacements = 2` gives `[0, 5, 12, -1]` |
| `a_disabled_policy_is_the_identity` | \(S = 0\) and \(B = 0\) leave the buffer unchanged |
| `short_and_empty_rows_follow_the_source_fallback` | A zero-length or negative-length row stays all padding |
| `an_out_of_range_learned_index_is_clamped_rather_than_refused` | \(L = 4\), \(S = 8\) clamps 20 to segment 7, beyond \(S_{\text{eff}} = 4\) |
| `malformed_calls_are_refused` | Each `WitnessError` |
| `every_witness_lands_in_its_own_segment` | \(\sigma(e_s) = s\) for every \(S\) the merge accepts. `segment_of(2, 10, 4) = 0` shows rounding down fails |
| `witnesses_are_causal_distinct_disjoint_and_complete` | Properties (1) to (3) on randomised rows |
| `the_serving_row_is_causal_distinct_ordered_and_within_budget` | 200 randomised cases: the prefix is bit-identical, one ascending run of witnesses starts at `keep`, and the row stays distinct |
| `every_segment_is_covered_when_the_reserved_tail_is_padding` | Coverage 1.0 on 200 randomised cases |
| `overwriting_a_sole_tail_occupant_uncovers_its_segment` | The coverage-0.75 counterexample above |
| `permuting_the_learned_row_leaves_the_witness_set_unchanged` | The witness set is permutation-invariant. Moving a token across the keep boundary changes the retained set from \(\{0, 5, 12\}\) to \(\{5, 9, 12\}\) |
| `the_nulls_spend_exactly_the_policy_budget` | Same budget \(k\). The locality null is exactly \([L - k, L)\) |
| `the_nulls_are_the_last_row_of_the_attention_selectors` | Reuse of `Selector::Random` and `Selector::Local`, asserted |
| `coverage_and_recall_match_a_hand_computed_example` | \(L = 8\), \(S = 4\), dyadic weights. The policy scores coverage 1.0 and recall 0.9375. The locality null at the same budget of four scores 0.5 and 0.25. Padding, duplicates and out-of-range entries count for nothing |

## Provenance

[vllm-project/vllm#47942](https://github.com/vllm-project/vllm/pull/47942) (open), commit `a41354cb34`:

- `vllm/v1/attention/backends/mla/sparse_utils.py`: `topology_witness_indices`, `apply_topology_witnesses`, `merge_topology_witnesses`, `scatter_topology_witnesses_` and `MAX_TOPOLOGY_SEGMENTS`.
- The gate in `vllm/model_executor/layers/sparse_attn_indexer.py`.
- `tests/v1/attention/test_sparse_mla_topology_index.py`: `_segment_of`, `_random_case`, and the coverage predicate of `test_witness_contracts_hold_over_random_rows`.

See [Upstream Contributions](upstream.md).
