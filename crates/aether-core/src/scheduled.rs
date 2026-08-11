//! Topology-derived sparse attention over causal CSR block schedules.
//!
//! A Rust port of the Triton kernel merged as `triton-lang/kernels#22`,
//! "Add topology-derived sparse attention kernel". The Python original runs on
//! CUDA; this runs anywhere `aether-core` does, including `no_std`.
//!
//! The decomposition is the original's, and it is the reason the thing is
//! testable at all:
//!
//!   - **The schedule** is combinatorial. Which key blocks a query block may see
//!     is decided once, on the CPU, and encoded as CSR (`offsets`, `indices`).
//!     Correctness here is exact set equality, not a tolerance.
//!   - **The kernel** is numeric. It walks a row's scheduled blocks with an
//!     online softmax, holding one score tile and the per-row running maximum and
//!     denominator — never a `seq x seq` matrix.
//!
//! The schedule is the union of three sources, all clamped causally:
//!
//!   - **sink blocks**: the first `sink_blocks` blocks, which attention sinks
//!     onto regardless of content;
//!   - **a local window**: `local_radius_blocks` back from the query block, which
//!     always includes the query block itself, so no row is ever empty;
//!   - **salient blocks**: the `topk_topology_blocks` key blocks with the highest
//!     0D-persistence salience over block centroids.
//!
//! The salience is the elder rule: when two components merge, every member of the
//! smaller one records the merge distance and the larger one carries on. A block's
//! score is therefore the H0 death time of the component it belonged to, and a
//! high score means the block stayed distinct from everything else for a long
//! time. `block_salience_is_the_elder_rule_over_centroids` checks it against this
//! crate's persistence engine rather than trusting two implementations of H0.

// Enforced per module rather than per crate. `aether-core` has 215 undocumented
// public items, and switching the whole crate on at once means writing 215 doc
// comments in one pass — which produces sentences restating the signature,
// exactly the kind that later contradicts a measurement nobody re-read them
// against. This module is the repository's headline mechanism and is done; the
// rest arrive a module at a time, each with the lint switched on behind them so
// finished work cannot regress.
#![warn(missing_docs)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use libm::{exp, sqrt};

/// Ways a schedule or a kernel launch can be malformed.
///
/// The Triton wrapper validates the same conditions before launch, because a bad
/// schedule on a GPU is an out-of-bounds read rather than an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleError {
    /// The block size was zero or not a power of two. Both break the index
    /// arithmetic every kernel here does with shifts and masks.
    BlockSizeNotPowerOfTwo {
        /// The offending size.
        block_size: usize,
    },
    /// The sequence does not divide evenly into blocks, which would leave a
    /// partial final block that the CSR schedule has no way to describe.
    SequenceNotDivisible {
        /// Sequence length as given.
        seq: usize,
        /// Block size as given.
        block_size: usize,
    },
    /// A zero-length sequence, which has no query blocks to schedule.
    EmptyInput,
    /// A row scheduled a key block at or beyond the query block's future.
    NonCausalRow {
        /// The query block whose row is malformed.
        q_block: usize,
        /// The future key block it named.
        block: usize,
    },
    /// A row scheduled the same block twice, or listed blocks out of order.
    UnsortedRow {
        /// The query block whose row is malformed.
        q_block: usize,
        /// The block that broke the strict ordering.
        block: usize,
    },
    /// A row scheduled nothing, which would make softmax divide by zero.
    EmptyRow {
        /// The query block with no key blocks.
        q_block: usize,
    },
    /// A row named a key block that does not exist in this schedule.
    BlockIndexOutOfRange {
        /// The query block whose row is malformed.
        q_block: usize,
        /// The out-of-range block index.
        block: usize,
    },
    /// The CSR offsets array is not one longer than the query block count, so
    /// the row boundaries it encodes cannot be read.
    OffsetsLengthMismatch {
        /// Length the block count implies.
        expected: usize,
        /// Length the offsets array actually has.
        actual: usize,
    },
    /// An operand's length disagreed with `seq * head_dim`, or a schedule
    /// covered a different number of query blocks than the launch.
    ShapeMismatch,
}

/// Which key blocks each query block may attend to, in CSR form.
///
/// `offsets` has `num_query_blocks + 1` entries; row `i` owns
/// `indices[offsets[i]..offsets[i + 1]]`. Rows are strictly sorted, causal, and
/// non-empty — enforced at construction so the kernel needs no per-row guards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockSchedule {
    /// Row boundaries, `num_query_blocks + 1` entries. Row `i` owns
    /// `indices[offsets[i]..offsets[i + 1]]`.
    pub offsets: Vec<usize>,
    /// Key block indices, grouped by query block and strictly sorted within
    /// each. The GPU port converts these to f32, which is lossless because a
    /// block index is far below 2^24.
    pub indices: Vec<usize>,
}

impl BlockSchedule {
    /// Build from an explicit per-row list, validating every invariant.
    pub fn from_rows(rows: &[Vec<usize>]) -> Result<Self, ScheduleError> {
        let num_blocks = rows.len();
        let mut offsets = Vec::with_capacity(num_blocks + 1);
        let mut indices = Vec::new();
        offsets.push(0);

        for (q_block, row) in rows.iter().enumerate() {
            if row.is_empty() {
                return Err(ScheduleError::EmptyRow { q_block });
            }
            let mut previous: Option<usize> = None;
            for &block in row {
                if block >= num_blocks {
                    return Err(ScheduleError::BlockIndexOutOfRange { q_block, block });
                }
                if block > q_block {
                    return Err(ScheduleError::NonCausalRow { q_block, block });
                }
                if let Some(prev) = previous {
                    if block <= prev {
                        return Err(ScheduleError::UnsortedRow { q_block, block });
                    }
                }
                previous = Some(block);
                indices.push(block);
            }
            offsets.push(indices.len());
        }
        Ok(Self { offsets, indices })
    }

    /// Key blocks visible to query block `q_block`.
    pub fn row(&self, q_block: usize) -> &[usize] {
        &self.indices[self.offsets[q_block]..self.offsets[q_block + 1]]
    }

    /// Number of query blocks.
    pub fn num_blocks(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
}

/// Every causal key block for every query block: the lower-triangular CSR.
///
/// The dense baseline the sparse schedule is measured against. It is also the
/// degenerate case that pins the kernel: with this schedule, the result must be
/// ordinary causal attention.
pub fn dense_causal_block_schedule(num_blocks: usize) -> BlockSchedule {
    let mut offsets = Vec::with_capacity(num_blocks + 1);
    let mut indices = Vec::with_capacity(num_blocks * (num_blocks + 1) / 2);
    offsets.push(0);
    for q_block in 0..num_blocks {
        indices.extend(0..=q_block);
        offsets.push(indices.len());
    }
    BlockSchedule { offsets, indices }
}

/// How to build a topology-derived schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopologyScheduleConfig {
    /// Query and key block size. Must be a power of two, and must divide `seq`.
    pub block_size: usize,
    /// How many blocks back from the query block the local window reaches. The
    /// window always includes the query block itself, so no row is ever empty.
    pub local_radius_blocks: usize,
    /// Leading blocks every query attends to regardless of content.
    pub sink_blocks: usize,
    /// How many highest-salience key blocks to admit.
    pub topk_topology_blocks: usize,
}

impl Default for TopologyScheduleConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            local_radius_blocks: 1,
            sink_blocks: 1,
            topk_topology_blocks: 2,
        }
    }
}

fn validate_blocking(seq: usize, block_size: usize) -> Result<usize, ScheduleError> {
    if block_size == 0 || !block_size.is_power_of_two() {
        return Err(ScheduleError::BlockSizeNotPowerOfTwo { block_size });
    }
    if seq == 0 {
        return Err(ScheduleError::EmptyInput);
    }
    if !seq.is_multiple_of(block_size) {
        return Err(ScheduleError::SequenceNotDivisible { seq, block_size });
    }
    Ok(seq / block_size)
}

/// 0D-persistence salience of each key block, over block centroids.
///
/// Single-linkage merging in increasing distance; on each merge, every member of
/// the **smaller** component records the merge distance and the larger component
/// continues. That is the elder rule, so a block's score is the H0 death time of
/// the component it belonged to.
///
/// Exactly one block scores 0: the invariant "each component holds exactly one
/// block that has never been written" is preserved by every merge, since the
/// smaller component's sole unwritten block gets written and the larger one's does
/// not. That block is the one that survived to the end, and it is never selected
/// by top-k — which is correct, since it has no finite death.
pub fn block_salience(
    keys: &[f64],
    seq: usize,
    dim: usize,
    block_size: usize,
) -> Result<Vec<f64>, ScheduleError> {
    let num_blocks = validate_blocking(seq, block_size)?;
    if keys.len() != seq * dim {
        return Err(ScheduleError::ShapeMismatch);
    }
    if num_blocks == 1 {
        // One block never merges with anything, so the Triton original returns a
        // constant. Matching that keeps top-k well-defined at the smallest size.
        return Ok(vec![1.0]);
    }

    let mut centroids = vec![0.0f64; num_blocks * dim];
    for block in 0..num_blocks {
        for t in 0..block_size {
            for d in 0..dim {
                centroids[block * dim + d] += keys[(block * block_size + t) * dim + d];
            }
        }
        for d in 0..dim {
            centroids[block * dim + d] /= block_size as f64;
        }
    }

    let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(num_blocks * num_blocks / 2);
    for i in 0..num_blocks {
        for j in (i + 1)..num_blocks {
            let mut sum = 0.0;
            for d in 0..dim {
                let delta = centroids[i * dim + d] - centroids[j * dim + d];
                sum += delta * delta;
            }
            edges.push((sqrt(sum), i, j));
        }
    }
    // Ties broken by index, matching the stable sort the Python builder relies on.
    edges.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

    let mut parent: Vec<usize> = (0..num_blocks).collect();
    let mut members: Vec<Vec<usize>> = (0..num_blocks).map(|i| vec![i]).collect();
    let mut salience = vec![0.0f64; num_blocks];

    for (distance, left, right) in edges {
        let (mut root_left, mut root_right) = (find(&mut parent, left), find(&mut parent, right));
        if root_left == root_right {
            continue;
        }
        // Absorb the smaller into the larger, so the elder rule records the
        // younger component's death.
        if members[root_left].len() > members[root_right].len() {
            core::mem::swap(&mut root_left, &mut root_right);
        }
        for &block in &members[root_left] {
            salience[block] = distance;
        }
        parent[root_left] = root_right;
        let absorbed = core::mem::take(&mut members[root_left]);
        members[root_right].extend(absorbed);
    }
    Ok(salience)
}

fn find(parent: &mut [usize], x: usize) -> usize {
    let mut root = x;
    while parent[root] != root {
        root = parent[root];
    }
    let mut current = x;
    while parent[current] != root {
        let next = parent[current];
        parent[current] = root;
        current = next;
    }
    root
}

/// Build a causal CSR schedule from sink blocks, a local window, and the
/// highest-salience key blocks.
///
/// # The salience ranking is measurably worse than choosing at random
///
/// At an identical per-row budget this schedule recovers *less* attention mass
/// than selecting blocks uniformly at random, and the deficit widens the more
/// budget the topology is given — reaching a placement of −109% at
/// `topk_topology_blocks = 32`, where it recovers 0.7446 against random's 0.8643.
/// The same holds on i.i.d. keys, which controls for a fixture that merely
/// rewards locality. Training a model through the kernel does not close the gap.
///
/// The mechanism is not mysterious. [`block_salience`] scores a block by its H0
/// death time, which measures how *isolated* it is, and attention mass
/// concentrates where a key resembles the query. Ranking by isolation and ranking
/// by attention mass are anti-correlated by construction, which is why
/// [`inverted_topology_block_schedule`] — the same selection with the ranking
/// reversed — beats random by a margin that grows with budget.
///
/// Reproduce with `cargo run -p aether-gpu --example selector_ablation --release`;
/// the tables are in `crates/aether-gpu/FEATURES.md`.
///
/// # Why this function still ranks the way it does
///
/// Flipping the ranking changes what the method *is*, and that belongs to whoever
/// is making the claim rather than to the ablation that found it. The measurement
/// is reported here instead of acted on, because a caller choosing between the
/// two should see the evidence at the point of choosing — the finding was
/// previously recorded only in `FEATURES.md` and the README, which is everywhere
/// except where the decision is made.
pub fn topology_block_schedule(
    keys: &[f64],
    seq: usize,
    dim: usize,
    config: TopologyScheduleConfig,
) -> Result<BlockSchedule, ScheduleError> {
    let num_blocks = validate_blocking(seq, config.block_size)?;
    if keys.len() != seq * dim {
        return Err(ScheduleError::ShapeMismatch);
    }

    let salience = block_salience(keys, seq, dim, config.block_size)?;
    let topk = config.topk_topology_blocks.min(num_blocks);
    let mut ranked: Vec<usize> = (0..num_blocks).collect();
    // Highest salience first; ties by index so the schedule is deterministic.
    ranked.sort_by(|&a, &b| salience[b].total_cmp(&salience[a]).then(a.cmp(&b)));
    let mut is_salient = vec![false; num_blocks];
    for &block in &ranked[..topk] {
        is_salient[block] = true;
    }

    let mut offsets = Vec::with_capacity(num_blocks + 1);
    let mut indices = Vec::new();
    offsets.push(0);

    for q_block in 0..num_blocks {
        // A boolean row rather than a set: it keeps the output sorted and
        // duplicate-free for free, which the CSR invariants require anyway.
        let mut allowed = vec![false; q_block + 1];
        for block in 0..config.sink_blocks.min(num_blocks).min(q_block + 1) {
            allowed[block] = true;
        }
        for block in q_block.saturating_sub(config.local_radius_blocks)..=q_block {
            allowed[block] = true;
        }
        for (block, slot) in allowed.iter_mut().enumerate() {
            if is_salient[block] {
                *slot = true;
            }
        }
        // The local window always contains `q_block`, so this cannot be empty.
        // The guard mirrors the Python original rather than trusting that.
        if !allowed.iter().any(|&on| on) {
            allowed[q_block] = true;
        }

        indices.extend((0..=q_block).filter(|&block| allowed[block]));
        offsets.push(indices.len());
    }
    Ok(BlockSchedule { offsets, indices })
}

/// [`topology_block_schedule`] with the salience ranking reversed.
///
/// Measured against a same-budget random baseline, ranking by *highest* salience
/// recovers less attention mass than choosing blocks at random, and the deficit
/// grows as more budget is given to it. Reversing the ranking turns that around:
/// it beats random by a margin that widens with budget, which is what an
/// informative signal looks like. The numbers are in `aether-gpu/FEATURES.md`
/// and reproduce with `cargo run -p aether-gpu --example selector_ablation`.
///
/// The mechanism is not mysterious. [`block_salience`] scores a block by its H0
/// death time, which measures how *isolated* it is. Attention mass concentrates
/// where a key resembles the query, and a block unlike everything else is unlike
/// the typical query too, so the two rankings are anti-correlated by
/// construction.
///
/// **The zero-salience block is excluded rather than ranked first.** Exactly one
/// block never merges into a larger component and records no death time;
/// [`block_salience`] reports it as 0. Under the original ranking that sentinel
/// sorts last and is never selected, so it costs nothing. Reversed, 0 becomes
/// the minimum and that block would be chosen before every real candidate —
/// which reads a marker for "undefined" as "least isolated". It is filtered out
/// here, and a schedule that wants it can still reach it through the local
/// window or the sinks.
///
/// This is offered alongside [`topology_block_schedule`] rather than replacing
/// its ranking. Changing the sign changes what the method is, and that belongs
/// to whoever is making the claim rather than to the function that measured it.
pub fn inverted_topology_block_schedule(
    keys: &[f64],
    seq: usize,
    dim: usize,
    config: TopologyScheduleConfig,
) -> Result<BlockSchedule, ScheduleError> {
    let num_blocks = validate_blocking(seq, config.block_size)?;
    if keys.len() != seq * dim {
        return Err(ScheduleError::ShapeMismatch);
    }

    let salience = block_salience(keys, seq, dim, config.block_size)?;
    let mut ranked: Vec<usize> = (0..num_blocks).filter(|&b| salience[b] != 0.0).collect();
    // Lowest salience first; ties by index so the schedule is deterministic.
    ranked.sort_by(|&a, &b| salience[a].total_cmp(&salience[b]).then(a.cmp(&b)));

    let topk = config.topk_topology_blocks.min(ranked.len());
    let mut is_salient = vec![false; num_blocks];
    for &block in &ranked[..topk] {
        is_salient[block] = true;
    }

    let mut offsets = Vec::with_capacity(num_blocks + 1);
    let mut indices = Vec::new();
    offsets.push(0);

    for q_block in 0..num_blocks {
        let mut allowed = vec![false; q_block + 1];
        for block in 0..config.sink_blocks.min(num_blocks).min(q_block + 1) {
            allowed[block] = true;
        }
        for block in q_block.saturating_sub(config.local_radius_blocks)..=q_block {
            allowed[block] = true;
        }
        for (block, slot) in allowed.iter_mut().enumerate() {
            if is_salient[block] {
                *slot = true;
            }
        }
        if !allowed.iter().any(|&on| on) {
            allowed[q_block] = true;
        }

        indices.extend((0..=q_block).filter(|&block| allowed[block]));
        offsets.push(indices.len());
    }
    Ok(BlockSchedule { offsets, indices })
}

// ═══════════════════════════════════════════════════════════════════════════════
// The kernel
// ═══════════════════════════════════════════════════════════════════════════════

fn validate_launch(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    seq: usize,
    head_dim: usize,
    schedule: &BlockSchedule,
    block_size: usize,
) -> Result<usize, ScheduleError> {
    let num_blocks = validate_blocking(seq, block_size)?;
    if q.len() != seq * head_dim || k.len() != seq * head_dim || v.len() != seq * head_dim {
        return Err(ScheduleError::ShapeMismatch);
    }
    if schedule.offsets.len() != num_blocks + 1 {
        return Err(ScheduleError::OffsetsLengthMismatch {
            expected: num_blocks + 1,
            actual: schedule.offsets.len(),
        });
    }
    for q_block in 0..num_blocks {
        let row = schedule.row(q_block);
        if row.is_empty() {
            return Err(ScheduleError::EmptyRow { q_block });
        }
        for &block in row {
            if block >= num_blocks {
                return Err(ScheduleError::BlockIndexOutOfRange { q_block, block });
            }
            if block > q_block {
                return Err(ScheduleError::NonCausalRow { q_block, block });
            }
        }
    }
    Ok(num_blocks)
}

/// Forward causal attention over the key/value blocks named in `schedule`.
///
/// Row-major `[seq, head_dim]` in and out. The block loop carries a running
/// maximum `m` and denominator `l` per query row and rescales the accumulator by
/// `alpha = exp(m_prev - m_new)` whenever a later block raises the maximum, so
/// the working set is one `block_size x block_size` score tile plus `block_size`
/// rows of state — never `seq x seq`.
///
/// Positions inside a scheduled block are still masked causally, so a scheduled
/// diagonal block contributes only its lower triangle.
pub fn scheduled_attention(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    seq: usize,
    head_dim: usize,
    schedule: &BlockSchedule,
    block_size: usize,
) -> Result<Vec<f64>, ScheduleError> {
    let num_blocks = validate_launch(q, k, v, seq, head_dim, schedule, block_size)?;
    let scale = 1.0 / sqrt(head_dim as f64);
    let mut out = vec![0.0f64; seq * head_dim];

    // Per-query-row state for the tile currently in flight.
    let mut running_max = vec![f64::NEG_INFINITY; block_size];
    let mut denominator = vec![0.0f64; block_size];
    let mut accumulator = vec![0.0f64; block_size * head_dim];
    let mut scores = vec![0.0f64; block_size];

    for q_block in 0..num_blocks {
        running_max.iter_mut().for_each(|m| *m = f64::NEG_INFINITY);
        denominator.iter_mut().for_each(|l| *l = 0.0);
        accumulator.iter_mut().for_each(|a| *a = 0.0);

        for &k_block in schedule.row(q_block) {
            for local_m in 0..block_size {
                let row = q_block * block_size + local_m;

                // Score tile for this (query row, key block), causally masked.
                let mut tile_max = f64::NEG_INFINITY;
                for (local_n, score) in scores.iter_mut().enumerate() {
                    let col = k_block * block_size + local_n;
                    if col > row {
                        *score = f64::NEG_INFINITY;
                        continue;
                    }
                    let mut dot = 0.0;
                    for d in 0..head_dim {
                        dot += q[row * head_dim + d] * k[col * head_dim + d];
                    }
                    *score = dot * scale;
                    if *score > tile_max {
                        tile_max = *score;
                    }
                }
                // Entirely in the future: contributes nothing, and folding it in
                // would multiply the accumulator by exp(-inf - -inf) = NaN.
                if tile_max == f64::NEG_INFINITY {
                    continue;
                }

                let previous_max = running_max[local_m];
                let new_max = if previous_max > tile_max {
                    previous_max
                } else {
                    tile_max
                };
                // exp(-inf - finite) is 0, which is the right rescale for the
                // first block a row ever sees.
                let alpha = if previous_max == f64::NEG_INFINITY {
                    0.0
                } else {
                    exp(previous_max - new_max)
                };

                denominator[local_m] *= alpha;
                for d in 0..head_dim {
                    accumulator[local_m * head_dim + d] *= alpha;
                }

                for (local_n, &score) in scores.iter().enumerate() {
                    if score == f64::NEG_INFINITY {
                        continue;
                    }
                    let weight = exp(score - new_max);
                    denominator[local_m] += weight;
                    let col = k_block * block_size + local_n;
                    for d in 0..head_dim {
                        accumulator[local_m * head_dim + d] += weight * v[col * head_dim + d];
                    }
                }
                running_max[local_m] = new_max;
            }
        }

        for local_m in 0..block_size {
            let row = q_block * block_size + local_m;
            let l = denominator[local_m];
            for d in 0..head_dim {
                // `l` is positive for every row: the local window always schedules
                // the diagonal block, and a row always sees at least itself.
                out[row * head_dim + d] = accumulator[local_m * head_dim + d] / l;
            }
        }
    }
    Ok(out)
}

/// Reference attention for a CSR schedule: materialise the mask, softmax once.
///
/// Obviously correct and obviously quadratic. The kernel is checked against this,
/// which is the only way to know the online rescale is right.
pub fn dense_masked_attention(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    seq: usize,
    head_dim: usize,
    schedule: &BlockSchedule,
    block_size: usize,
) -> Result<Vec<f64>, ScheduleError> {
    let num_blocks = validate_launch(q, k, v, seq, head_dim, schedule, block_size)?;

    let mut allow = vec![false; seq * seq];
    for q_block in 0..num_blocks {
        for &k_block in schedule.row(q_block) {
            for local_m in 0..block_size {
                let row = q_block * block_size + local_m;
                for local_n in 0..block_size {
                    let col = k_block * block_size + local_n;
                    if col <= row {
                        allow[row * seq + col] = true;
                    }
                }
            }
        }
    }
    Ok(crate::attention::sparse_attention(
        q, k, v, seq, head_dim, &allow,
    ))
}

/// Gradients of a scheduled-attention output with respect to its inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionGradients {
    /// `[seq, head_dim]`, same shape as `q`.
    pub dq: Vec<f64>,
    /// `[seq, head_dim]`, same shape as `k`.
    pub dk: Vec<f64>,
    /// `[seq, head_dim]`, same shape as `v`.
    pub dv: Vec<f64>,
}

/// Reverse mode through [`scheduled_attention`], holding the schedule fixed.
///
/// The schedule is combinatorial and derived from the keys by a procedure with
/// no useful derivative — a block is selected or it is not. Treating it as a
/// constant is the same choice the forward kernel's design already makes, and it
/// is what makes this a well-defined function to differentiate rather than a
/// relaxation of one.
///
/// Which means a gradient here cannot teach the model to schedule differently.
/// It can teach the model to make better use of the blocks it was given, which is
/// the experiment `aether-gpu/examples/recall_training.rs` explicitly could not
/// run: that file freezes attention and trains only a head, and records the
/// inability to train through the kernel as its narrowest assumption.
///
/// # Recomputation rather than saved state
///
/// The forward kernel is a flash-style online softmax that never materialises
/// the score matrix, so there is nothing stored to reuse. This recomputes the
/// scores for each row's scheduled blocks. That is the same trade flash attention
/// makes in reverse — arithmetic in exchange for never holding a `seq x seq`
/// array — and it keeps the working set proportional to a single row.
///
/// The softmax weights are recomputed in one pass here rather than by the
/// forward's running-maximum rescale. Both produce identical weights: the online
/// form exists to keep intermediates in range, not to change the result, which is
/// the property a mutant already confirmed by surviving every test until one used
/// inputs large enough for the difference to matter.
pub fn scheduled_attention_backward(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    seq: usize,
    head_dim: usize,
    schedule: &BlockSchedule,
    block_size: usize,
    d_out: &[f64],
) -> Result<AttentionGradients, ScheduleError> {
    let num_blocks = validate_launch(q, k, v, seq, head_dim, schedule, block_size)?;
    if d_out.len() != seq * head_dim {
        return Err(ScheduleError::ShapeMismatch);
    }
    let _ = num_blocks;

    let scale = 1.0 / sqrt(head_dim as f64);
    let mut dq = vec![0.0f64; seq * head_dim];
    let mut dk = vec![0.0f64; seq * head_dim];
    let mut dv = vec![0.0f64; seq * head_dim];

    // Scratch for one query row: the columns it sees and their weights. Sized
    // once at the widest a row can be, so the loop allocates nothing.
    let mut columns: Vec<usize> = Vec::with_capacity(seq);
    let mut weights: Vec<f64> = Vec::with_capacity(seq);

    for row in 0..seq {
        let q_block = row / block_size;

        columns.clear();
        weights.clear();

        let mut max_score = f64::NEG_INFINITY;
        for &k_block in schedule.row(q_block) {
            for local in 0..block_size {
                let col = k_block * block_size + local;
                if col > row {
                    continue;
                }
                let mut dot = 0.0;
                for d in 0..head_dim {
                    dot += q[row * head_dim + d] * k[col * head_dim + d];
                }
                let score = dot * scale;
                if score > max_score {
                    max_score = score;
                }
                columns.push(col);
                weights.push(score);
            }
        }

        // A causal schedule always contains the diagonal block, and a row always
        // sees at least itself, so this is never empty. The forward kernel relies
        // on the same fact to divide by its denominator.
        let mut denominator = 0.0;
        for weight in weights.iter_mut() {
            *weight = exp(*weight - max_score);
            denominator += *weight;
        }
        for weight in weights.iter_mut() {
            *weight /= denominator;
        }

        // D_i = sum_j p_ij (dOut_i . V_j), the term that makes the softmax
        // Jacobian a rank-one correction rather than a full matrix.
        let mut delta = 0.0;
        for (idx, &col) in columns.iter().enumerate() {
            let mut dp = 0.0;
            for d in 0..head_dim {
                dp += d_out[row * head_dim + d] * v[col * head_dim + d];
            }
            delta += weights[idx] * dp;
        }

        for (idx, &col) in columns.iter().enumerate() {
            let p = weights[idx];

            let mut dp = 0.0;
            for d in 0..head_dim {
                dp += d_out[row * head_dim + d] * v[col * head_dim + d];
                dv[col * head_dim + d] += p * d_out[row * head_dim + d];
            }

            let ds = p * (dp - delta) * scale;
            for d in 0..head_dim {
                dq[row * head_dim + d] += ds * k[col * head_dim + d];
                dk[col * head_dim + d] += ds * q[row * head_dim + d];
            }
        }
    }

    Ok(AttentionGradients { dq, dk, dv })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Same-budget baselines
//
// A sparse attention schedule is easy to make look good. It runs faster than
// dense because it does less work, and the model it feeds still trains. Neither
// fact says the *selection* is doing anything: a schedule picking blocks at
// random is also faster, and also trains.
//
// What separates them is how much of the true attention mass each schedule
// captures at an identical budget. Three schedules bracket the answer:
//
//   - **random** at the same budget. If the topological schedule matches this,
//     the mechanism contributes nothing and any gain is sparsity acting as
//     regularisation.
//   - **oracle** at the same budget, selecting the blocks with genuinely the
//     largest attention mass. Unimplementable in production, since it needs the
//     dense scores the schedule exists to avoid, but as a diagnostic it is the
//     ceiling.
//   - the topological schedule, whose position between those two *is* the
//     result. Reporting that fraction is a stronger claim than any speedup, and
//     it is what a reviewer asks for first.
//
// Budget is matched per row, not on average. A schedule that spends its blocks
// unevenly could otherwise win by spending more of them where they matter, which
// is a different mechanism from the one under test.
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-row normalised attention mass of each causal key block.
///
/// Entry `(q_block, k_block)` is the total softmax weight that the rows of
/// `q_block` place on the columns of `k_block`, with each row normalised over
/// all its causally legal keys. Future blocks score zero.
///
/// The quantity is *additive over key blocks*, which is what makes an exact
/// oracle possible: a schedule's recovered mass is the sum of the scores of the
/// blocks it selects, so maximising it is choosing the largest scores rather
/// than searching over subsets.
fn block_mass_table(
    q: &[f64],
    k: &[f64],
    seq: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<Vec<f64>, ScheduleError> {
    let num_blocks = validate_blocking(seq, block_size)?;
    if q.len() != seq * head_dim || k.len() != seq * head_dim {
        return Err(ScheduleError::ShapeMismatch);
    }

    let scale = 1.0 / sqrt(head_dim as f64);
    let mut table = vec![0.0f64; num_blocks * num_blocks];

    for row in 0..seq {
        let q_block = row / block_size;
        let legal = row + 1;

        let mut logits = vec![0.0f64; legal];
        let mut max_logit = f64::NEG_INFINITY;
        for (col, logit) in logits.iter_mut().enumerate() {
            let mut dot = 0.0;
            for d in 0..head_dim {
                dot += q[row * head_dim + d] * k[col * head_dim + d];
            }
            *logit = dot * scale;
            if *logit > max_logit {
                max_logit = *logit;
            }
        }

        let denominator: f64 = logits.iter().map(|&l| exp(l - max_logit)).sum();
        for (col, &logit) in logits.iter().enumerate() {
            let k_block = col / block_size;
            table[q_block * num_blocks + k_block] += exp(logit - max_logit) / denominator;
        }
    }

    // Each row contributes a total of 1.0 spread across its blocks, so dividing
    // by the rows per block puts the table on the same scale as the per-row
    // average that `block_mass_recovered` reports.
    for entry in table.iter_mut() {
        *entry /= block_size as f64;
    }
    Ok(table)
}

/// Fraction of the true attention mass a schedule captures, averaged over rows.
///
/// 1.0 means the schedule loses nothing against dense causal attention, and is
/// exactly what [`dense_causal_block_schedule`] scores. This is the axis every
/// selector is compared on.
pub fn block_mass_recovered(
    schedule: &BlockSchedule,
    q: &[f64],
    k: &[f64],
    seq: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<f64, ScheduleError> {
    let num_blocks = validate_blocking(seq, block_size)?;
    if schedule.num_blocks() != num_blocks {
        return Err(ScheduleError::ShapeMismatch);
    }
    let table = block_mass_table(q, k, seq, head_dim, block_size)?;

    let mut total = 0.0;
    for q_block in 0..num_blocks {
        for &k_block in schedule.row(q_block) {
            total += table[q_block * num_blocks + k_block];
        }
    }
    Ok(total / num_blocks as f64)
}

/// Blocks each query block spends, read off an existing schedule.
///
/// The budget a baseline must match. Returned rather than assumed because the
/// topological schedule's row lengths vary: a row near the start of the sequence
/// has fewer causal blocks to choose from than the config would otherwise admit.
pub fn schedule_budget(schedule: &BlockSchedule) -> Vec<usize> {
    (0..schedule.num_blocks())
        .map(|q_block| schedule.row(q_block).len())
        .collect()
}

/// A uniformly random causal schedule spending exactly `budget` blocks per row.
///
/// The baseline that answers whether selection matters at all. Deterministic in
/// `seed`, because a baseline that changes between runs cannot be compared
/// against anything.
///
/// Sampling is a partial Fisher-Yates over the causal candidates, which draws
/// without replacement in one pass. Rejection sampling would be shorter and
/// degrades exactly where this is most used: when the budget approaches the
/// number of candidates, as it does for every early query block.
pub fn random_block_schedule(budget: &[usize], seed: u64) -> Result<BlockSchedule, ScheduleError> {
    let num_blocks = budget.len();
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };

    let mut rows = Vec::with_capacity(num_blocks);
    for (q_block, &want) in budget.iter().enumerate() {
        let candidates = q_block + 1;
        if want == 0 {
            return Err(ScheduleError::EmptyRow { q_block });
        }
        // More blocks than exist causally cannot be spent. Clamping rather than
        // erroring keeps a baseline usable against any reference schedule, and a
        // valid reference can never ask for more than this anyway.
        let take = if want > candidates { candidates } else { want };

        let mut pool: Vec<usize> = (0..candidates).collect();
        for i in 0..take {
            let j = i + (next() as usize) % (candidates - i);
            pool.swap(i, j);
        }
        let mut row: Vec<usize> = pool[..take].to_vec();
        row.sort_unstable();
        rows.push(row);
    }

    BlockSchedule::from_rows(&rows)
}

/// The highest-mass causal schedule at a given per-row budget.
///
/// Not implementable in production: it reads the dense scores that a sparse
/// schedule exists to avoid computing. As a diagnostic it is the ceiling, and it
/// is *exactly* the ceiling rather than an approximation of one — recovered mass
/// is additive over key blocks, so taking the largest per-block scores is optimal
/// by construction rather than by search.
///
/// That optimality is asserted as a property rather than trusted. A schedule that
/// ever recovers more mass than this at the same budget would mean the table this
/// ranks by is not measuring what the recovery measures.
pub fn oracle_block_schedule(
    q: &[f64],
    k: &[f64],
    seq: usize,
    head_dim: usize,
    block_size: usize,
    budget: &[usize],
) -> Result<BlockSchedule, ScheduleError> {
    let num_blocks = validate_blocking(seq, block_size)?;
    if budget.len() != num_blocks {
        return Err(ScheduleError::ShapeMismatch);
    }
    let table = block_mass_table(q, k, seq, head_dim, block_size)?;

    let mut rows = Vec::with_capacity(num_blocks);
    for (q_block, &want) in budget.iter().enumerate() {
        let candidates = q_block + 1;
        if want == 0 {
            return Err(ScheduleError::EmptyRow { q_block });
        }
        let take = if want > candidates { candidates } else { want };

        let mut ranked: Vec<usize> = (0..candidates).collect();
        // Descending by mass. Ties break on the lower block index so the result
        // does not depend on the sort's stability guarantees.
        ranked.sort_by(|&a, &b| {
            let ma = table[q_block * num_blocks + a];
            let mb = table[q_block * num_blocks + b];
            mb.partial_cmp(&ma)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut row: Vec<usize> = ranked[..take].to_vec();
        row.sort_unstable();
        rows.push(row);
    }

    BlockSchedule::from_rows(&rows)
}
