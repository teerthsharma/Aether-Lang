//! Segment witnesses for a learned sparse top-k.
//!
//! A port of the candidate policy in vllm-project/vllm#47942, commit
//! `a41354cb34`, "Add topology witnesses to the sparse MLA indexer top-k"
//! (`vllm/v1/attention/backends/mla/sparse_utils.py`, gated from
//! `vllm/model_executor/layers/sparse_attn_indexer.py`). The original has a torch
//! reference and a fused Triton kernel asserted equal to it; this is the torch
//! reference, which is also the original's CPU path.
//!
//! The learned indexer top-k is a density rule: it concentrates where scores are
//! high and places no constraint on how its tokens are distributed over the
//! context. The policy adds a covering constraint on top of it.
//!
//! # Topology object
//!
//! For a row of context length `L >= 1` and a segment count `S >= 1`, put
//! `S_eff = min(S, L)` and partition `[0, L)` into `S_eff` consecutive intervals
//!
//! ```text
//!   e_s = ceil(s * L / S_eff),    I_s = [e_s, e_{s+1}),    s = 0, ..., S_eff - 1,
//! ```
//!
//! with segment map `sigma(x) = floor(x * S_eff / L)`, so that `x in I_s` if and
//! only if `sigma(x) = s` ([`segment_of`]). A learned row `K` occupies
//! `O(K) = { sigma(x) : x in K }`, and the witness set is
//!
//! ```text
//!   W(K) = { e_s : s < S_eff, s not in O(K) }.
//! ```
//!
//! # Invariant
//!
//! `sigma(e_s) = s` for every `s < S_eff`: a segment's left edge, rounded up,
//! lies in that segment. This requires `S_eff <= L`, which is why `S` is capped at
//! `L` per row. Rounding down does not round-trip (`L = 10`, `S = 4`, `s = 1`
//! gives 2, and `sigma(2) = 0`). Three properties follow without any scan:
//!
//! 1. `W(K)` and `K` are disjoint, since a witness is emitted only for a segment
//!    that `K` does not meet;
//! 2. witnesses are pairwise distinct, since each lies in its own segment;
//! 3. `O(K ∪ W(K)) = {0, ..., S_eff - 1}`: every segment holds a selected token.
//!
//! By (3), a maximal run of unselected tokens meets at most two segments, so its
//! length is at most `2 * ceil(L / S_eff) - 2`. The PR describes this as bounding
//! the uncovered run at `L / S` tokens. What segment coverage actually implies is
//! the bound given here.
//!
//! # Implementation rule
//!
//! A row is a fixed-width array of `topk` slots. Slots `[0, keep)` are retained
//! verbatim. Witnesses, in ascending segment order, overwrite slots `keep`,
//! `keep + 1`, ... up to the replacement budget
//!
//! ```text
//!   B = min(max_replacements, topk - keep),
//! ```
//!
//! and any witness beyond `B` is dropped. Occupancy is read from the whole learned
//! row, including the tail slots about to be overwritten, so a tail slot that
//! survives the merge can never equal a witness. The serving entry point
//! [`scatter_topology_witnesses`] uses `keep = topk - S` (saturating) and
//! `max_replacements = S`.
//!
//! Because occupancy counts tail slots that the merge then overwrites, (3) holds
//! for `K ∪ W(K)` but not always for the merged row. A segment whose only learned
//! token sits in an overwritten slot is left uncovered. The output covers every
//! segment when no overwritten slot is its segment's sole occupant, which is the
//! case, for example, when the reserved tail is padding.
//!
//! # Coordinate contract
//!
//! Entries are **row-relative token offsets** (`i32`, relative to the row's
//! `rowStart` in vLLM), not absolute positions in a flattened KV buffer. Row `r`
//! has context length `L_r` and valid range `[0, L_r)`. Its query is at
//! `L_r - 1`, so an entry is causal when `x <= L_r - 1`. vLLM supplies
//! `L_r = seq_lens` for decode and `cu_seqlen_ke - cu_seqlen_ks` for prefill.
//! `-1` is padding: it marks no segment occupied and passes through unchanged.
//! Every witness satisfies `0 <= e_s < L_r`.
//!
//! # Budget contract
//!
//! At most `B` slots change, all inside `[keep, keep + B)`. Slots `[0, keep)` are
//! bit-identical to the input, so at least `keep` learned entries survive. Row
//! width never changes.
//!
//! # Disabled and fallback contract
//!
//! - `S = 0` is the off switch. It is the default of
//!   `VLLM_SPARSE_MLA_TOPOLOGY_SEGMENTS`, and [`scatter_topology_witnesses`]
//!   leaves the buffer untouched.
//! - `B = 0`, or zero rows: the learned rows are returned unchanged.
//! - `L_r <= 0`: that row gets no witnesses.
//! - `L_r < S`: `S_eff = L_r`, every segment is one token, and every token the
//!   learned row does not hold becomes a witness.
//! - A learned `x >= L_r` is not a legal top-k output, but it is neither refused
//!   nor removed. It marks segment `min(floor(x * S_eff / L_r), S - 1)` occupied,
//!   which lies outside the partition when `S_eff < S`, and it passes through.
//! - Refused ([`WitnessError`]): context lengths that do not cover every learned
//!   row; `S` outside `[1, 64]` at the merge, because the fused kernel carries
//!   occupancy as a 64-bit mask; and `keep > topk`.
//!
//! # Null models and metrics
//!
//! A covering policy has to be compared at an identical budget
//! `k = |{valid output slots}|` against a seeded uniform draw of `k` tokens from
//! `[0, L)` ([`random_null`]) and against the `k` most recent tokens
//! `[L - k, L)` ([`locality_null`]). Both are row `L - 1` of the existing
//! [`crate::attention::Selector`] baselines, called rather than reimplemented. For
//! a candidate set `C` and a dense attention row `a` over `[0, L)`:
//!
//! ```text
//!   coverage(C) = |{ sigma(x) : x in C ∩ [0, L) }| / S_eff,
//!   recall(C; a) = sum_{x in C ∩ [0, L)} a_x / sum_{x < L} a_x.
//! ```
//!
//! # What the source measured
//!
//! The PR body reports the fused CUDA kernel at 1.12x to 1.62x the cost of copying
//! the same buffer, for 512 down to 32 rows, `topk = 2048`, `S = 64`, on an RTX
//! 4060 Laptop GPU (sm_89). It also reports 1,000/1,000 randomised configurations
//! identical to the torch reference. The PR ran no model-level evaluation, and it
//! states that the evidence shows the change is correct, bounded and cheap, not
//! that it is an improvement. This port measures nothing about quality either.
//! [`segment_coverage`] and [`attention_mass_recall`] exist so that an experiment
//! can.

#![warn(missing_docs)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::attention::{select_mask, Selector};

/// Largest segment count the merge accepts.
///
/// Ported from vllm a41354cb34:`vllm/v1/attention/backends/mla/sparse_utils.py`
/// (`MAX_TOPOLOGY_SEGMENTS`). The fused kernel carries segment occupancy as an
/// int64 bitmask, one bit per segment. This CPU port has no mask, but it
/// enforces the same bound so that both paths accept the same inputs.
pub const MAX_TOPOLOGY_SEGMENTS: usize = 64;

/// Ways a call can be malformed. These mirror the `ValueError`s of the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WitnessError {
    /// `learned_len != rows * topk`. Some learned row has no context length, so
    /// its row-relative offsets cannot be checked against a range.
    RowsMismatch {
        /// Length of the flattened learned buffer.
        learned_len: usize,
        /// Row width as given.
        topk: usize,
        /// Rows implied by the context lengths.
        rows: usize,
    },
    /// Zero segments, or more than [`MAX_TOPOLOGY_SEGMENTS`] at the merge.
    SegmentsOutOfRange {
        /// The segment count as given.
        num_segments: usize,
    },
    /// The retained prefix is wider than the row.
    LearnedKeepOutOfRange {
        /// The prefix width as given.
        learned_keep: usize,
        /// Row width as given.
        topk: usize,
    },
}

/// The segment map `sigma(x) = floor(x * S_eff / L)`, with
/// `S_eff = min(S, max(L, 1))`.
///
/// Ported from vllm a41354cb34:`tests/v1/attention/test_sparse_mla_topology_index.py`
/// (`_segment_of`). The occupancy pass in `sparse_utils.py` applies the same map
/// and then clamps the result to `S - 1`. A length of 0 is read as 1, as the
/// source's `clamp_min(1)` does, so this never divides by zero.
pub fn segment_of(token: usize, context_len: usize, num_segments: usize) -> usize {
    let len = context_len.max(1) as u64;
    let effective = (num_segments as u64).min(len);
    (token as u64 * effective / len) as usize
}

fn check_rows(learned: &[i32], topk: usize, context_lens: &[i32]) -> Result<(), WitnessError> {
    if learned.len() == context_lens.len() * topk {
        Ok(())
    } else {
        Err(WitnessError::RowsMismatch {
            learned_len: learned.len(),
            topk,
            rows: context_lens.len(),
        })
    }
}

/// One witness per segment the learned row misses, as row-major
/// `[rows, num_segments]`.
///
/// Ported from vllm a41354cb34:`vllm/v1/attention/backends/mla/sparse_utils.py`
/// (`topology_witness_indices`). `learned` is row-major `[rows, topk]` with one
/// context length per row. Entry `(r, s)` is `e_s = ceil(s * L_r / S_eff)` when
/// `s < S_eff`, `L_r > 0` and no learned entry of row `r` falls in segment `s`.
/// Otherwise it is `-1`. Occupancy is read from the whole row, and padding marks
/// nothing. Unlike the merge, this accepts any positive segment count, as the
/// source's reference does.
pub fn topology_witness_indices(
    learned: &[i32],
    topk: usize,
    context_lens: &[i32],
    num_segments: usize,
) -> Result<Vec<i32>, WitnessError> {
    check_rows(learned, topk, context_lens)?;
    if num_segments == 0 {
        return Err(WitnessError::SegmentsOutOfRange { num_segments });
    }

    let mut witnesses = vec![-1i32; context_lens.len() * num_segments];
    let mut occupied = vec![false; num_segments];
    for (row, &raw_len) in context_lens.iter().enumerate() {
        if raw_len <= 0 {
            continue;
        }
        let len = raw_len as usize;
        occupied.fill(false);
        for &token in &learned[row * topk..(row + 1) * topk] {
            if token >= 0 {
                // An index at or past `len` is not a legal top-k output. The
                // source clamps rather than rejects it, and so does this.
                occupied[segment_of(token as usize, len, num_segments).min(num_segments - 1)] =
                    true;
            }
        }

        let effective = num_segments.min(len);
        let out = &mut witnesses[row * num_segments..][..effective];
        for (s, (slot, &hit)) in out.iter_mut().zip(&occupied).enumerate() {
            if !hit {
                // Left edge rounded up, which is what keeps sigma(edge) = s.
                *slot = (s as u64 * len as u64).div_ceil(effective as u64) as i32;
            }
        }
    }
    Ok(witnesses)
}

/// Overwrite a bounded tail of each learned row with that row's witnesses.
///
/// Ported from vllm a41354cb34:`vllm/v1/attention/backends/mla/sparse_utils.py`
/// (`apply_topology_witnesses`), along its CPU path, which is
/// `merge_topology_witnesses` applied to [`topology_witness_indices`]. Slots
/// `[0, learned_keep)` survive untouched. Present witnesses, in ascending segment
/// order, fill slots `learned_keep, learned_keep + 1, ...`, and those beyond
/// `B = min(max_replacements, topk - learned_keep)` are dropped. When `B = 0`, or
/// there are no rows, the learned rows are returned unchanged.
pub fn apply_topology_witnesses(
    learned: &[i32],
    topk: usize,
    context_lens: &[i32],
    learned_keep: usize,
    num_segments: usize,
    max_replacements: usize,
) -> Result<Vec<i32>, WitnessError> {
    check_rows(learned, topk, context_lens)?;
    if num_segments == 0 || num_segments > MAX_TOPOLOGY_SEGMENTS {
        return Err(WitnessError::SegmentsOutOfRange { num_segments });
    }
    if learned_keep > topk {
        return Err(WitnessError::LearnedKeepOutOfRange { learned_keep, topk });
    }

    let mut out = learned.to_vec();
    let budget = max_replacements.min(topk - learned_keep);
    if budget == 0 || context_lens.is_empty() {
        return Ok(out);
    }

    let witnesses = topology_witness_indices(learned, topk, context_lens, num_segments)?;
    for (row, row_witnesses) in witnesses.chunks(num_segments).enumerate() {
        let tail = &mut out[row * topk + learned_keep..][..budget];
        // Zipping the tail with the present witnesses is the source's prefix-sum
        // compaction: the n-th present witness lands in slot keep + n while n < B.
        for (slot, &witness) in tail
            .iter_mut()
            .zip(row_witnesses.iter().filter(|&&w| w >= 0))
        {
            *slot = witness;
        }
    }
    Ok(out)
}

/// The serving entry point. Replaces the last `num_segments` slots of every row
/// in place, or does nothing when `num_segments` is 0.
///
/// Ported from vllm a41354cb34:`vllm/v1/attention/backends/mla/sparse_utils.py`
/// (`scatter_topology_witnesses_`), together with the
/// `if envs.VLLM_SPARSE_MLA_TOPOLOGY_SEGMENTS:` gate at both call sites in
/// `vllm/model_executor/layers/sparse_attn_indexer.py`. It calls
/// [`apply_topology_witnesses`] with `learned_keep = topk - num_segments`
/// (saturating) and `max_replacements = num_segments`. A refused call leaves
/// the buffer untouched.
pub fn scatter_topology_witnesses(
    topk_indices: &mut [i32],
    topk: usize,
    context_lens: &[i32],
    num_segments: usize,
) -> Result<(), WitnessError> {
    if num_segments == 0 {
        return Ok(());
    }
    let merged = apply_topology_witnesses(
        topk_indices,
        topk,
        context_lens,
        topk.saturating_sub(num_segments),
        num_segments,
        num_segments,
    )?;
    topk_indices.copy_from_slice(&merged);
    Ok(())
}

/// Row `context_len - 1` of a causal [`select_mask`], as ascending offsets.
///
/// ponytail: `select_mask` materialises the whole `[L, L]` mask to yield one row,
/// which is O(L^2) in time and memory. That is fine at test sizes. Past a few
/// thousand tokens, the fix is a per-row entry point in `attention.rs`.
fn last_row(selector: Selector, context_len: usize, budget: usize) -> Vec<i32> {
    if context_len == 0 || budget == 0 {
        return Vec::new();
    }
    let mask = select_mask(selector, &[], &[], &[], context_len, 0, true);
    mask[(context_len - 1) * context_len..]
        .iter()
        .enumerate()
        .filter(|&(_, &on)| on)
        .map(|(j, _)| j as i32)
        .collect()
}

/// Same-budget random null: `min(budget, L)` distinct offsets drawn uniformly
/// from `[0, L)`, deterministic in `seed`, ascending.
///
/// Delegates to [`crate::attention::Selector::Random`] (the query row at
/// `L - 1` of a causal sequence of length `L`), so the draw is the same one the
/// attention ablation measures. A zero budget selects nothing.
pub fn random_null(context_len: usize, budget: usize, seed: u64) -> Vec<i32> {
    last_row(Selector::Random { budget, seed }, context_len, budget)
}

/// Same-budget locality-only null: the `min(budget, L)` most recent offsets
/// `[L - budget, L)`, ascending.
///
/// Delegates to [`crate::attention::Selector::Local`], the sliding-window
/// baseline, at the query row `L - 1`. A zero budget selects nothing.
pub fn locality_null(context_len: usize, budget: usize) -> Vec<i32> {
    last_row(Selector::Local { window: budget }, context_len, budget)
}

/// Fraction of the `S_eff = min(S, L)` segments of `[0, L)` that hold at least
/// one candidate.
///
/// The coverage predicate of vllm a41354cb34:
/// `tests/v1/attention/test_sparse_mla_topology_index.py`
/// (`test_witness_contracts_hold_over_random_rows`), expressed as a fraction.
/// Padding and offsets outside `[0, L)` count for nothing. With no segments
/// (`L = 0` or `S = 0`) nothing is left uncovered, and the result is 1.
pub fn segment_coverage(candidates: &[i32], context_len: usize, num_segments: usize) -> f64 {
    let segments = num_segments.min(context_len);
    if segments == 0 {
        return 1.0;
    }
    let mut hit = vec![false; segments];
    for &token in candidates {
        if token >= 0 && (token as usize) < context_len {
            hit[segment_of(token as usize, context_len, num_segments)] = true;
        }
    }
    hit.iter().filter(|&&h| h).count() as f64 / segments as f64
}

/// Share of a dense attention row's mass that falls on a candidate set.
///
/// The per-row term of [`crate::attention::attention_mass_recovered`], taken
/// over an explicit candidate set instead of a selector's mask. `attention_row`
/// covers `[0, L)`. Each distinct in-range candidate counts once, and padding and
/// out-of-range offsets count for nothing. The result is normalised by the row's
/// total, which is NaN for a row with no mass.
pub fn attention_mass_recall(candidates: &[i32], attention_row: &[f64]) -> f64 {
    let mut taken = vec![false; attention_row.len()];
    let mut captured = 0.0;
    for &token in candidates {
        if token >= 0 && (token as usize) < attention_row.len() && !taken[token as usize] {
            taken[token as usize] = true;
            captured += attention_row[token as usize];
        }
    }
    captured / attention_row.iter().sum::<f64>()
}
