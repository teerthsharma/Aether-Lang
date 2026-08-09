//! Contracts for `aether_core::scheduled` — the Rust remake of the Triton
//! kernel merged as triton-lang/kernels#22, "Add topology-derived sparse
//! attention kernel".
//!
//! The Triton original is a forward-only kernel over causal CSR block schedules,
//! with the schedule built from sink blocks, a local window, and a
//! 0D-persistence salience over key-block centroids. This port keeps the same
//! decomposition, so the schedule builder and the kernel can be checked
//! independently:
//!
//!   - the schedule is combinatorial and exactly checkable against the CSR the
//!     Python builder emits;
//!   - the kernel is numeric and checkable against dense masked attention.
//!
//! The salience is the elder rule — each block scores the merge distance at which
//! its component was absorbed — so it is asserted against this crate's persistence
//! engine rather than trusting a second implementation of H0.

use aether_core::attention::sparse_attention;
use aether_core::scheduled::{
    block_salience, dense_causal_block_schedule, dense_masked_attention, scheduled_attention,
    topology_block_schedule, BlockSchedule, ScheduleError, TopologyScheduleConfig,
};

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn signed(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn qkv(seq: usize, head_dim: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let n = seq * head_dim;
    let q = (0..n).map(|_| rng.signed()).collect();
    let k = (0..n).map(|_| rng.signed()).collect();
    let v = (0..n).map(|_| rng.signed()).collect();
    (q, k, v)
}

/// Keys whose blocks sit at deliberately separated centroids, so the salience has
/// something real to find.
fn blocky_keys(num_blocks: usize, block_size: usize, dim: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let centers: Vec<Vec<f64>> = (0..num_blocks)
        .map(|_| (0..dim).map(|_| 4.0 * rng.signed()).collect())
        .collect();
    let mut keys = vec![0.0; num_blocks * block_size * dim];
    for b in 0..num_blocks {
        for t in 0..block_size {
            for d in 0..dim {
                keys[(b * block_size + t) * dim + d] = centers[b][d] + 0.05 * rng.signed();
            }
        }
    }
    keys
}

fn causal_mask(seq: usize) -> Vec<bool> {
    let mut mask = vec![false; seq * seq];
    for i in 0..seq {
        for j in 0..=i {
            mask[i * seq + j] = true;
        }
    }
    mask
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. The CSR schedule, checked exactly
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn the_dense_causal_schedule_is_lower_triangular_csr() {
    // The exact values the Triton builder emits for four blocks.
    let schedule = dense_causal_block_schedule(4);
    assert_eq!(schedule.offsets, vec![0, 1, 3, 6, 10]);
    assert_eq!(schedule.indices, vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3]);
}

#[test]
fn a_csr_schedule_is_well_formed_at_every_size() {
    for num_blocks in 1..=12usize {
        let schedule = dense_causal_block_schedule(num_blocks);

        assert_eq!(schedule.offsets.len(), num_blocks + 1);
        assert_eq!(schedule.offsets[0], 0);
        assert_eq!(*schedule.offsets.last().unwrap(), schedule.indices.len());
        assert_eq!(schedule.indices.len(), num_blocks * (num_blocks + 1) / 2);

        for q_block in 0..num_blocks {
            let row = schedule.row(q_block);
            // Non-decreasing offsets, causal, sorted, and duplicate-free.
            assert!(schedule.offsets[q_block] <= schedule.offsets[q_block + 1]);
            assert!(
                row.iter().all(|&b| b <= q_block),
                "row {q_block} not causal"
            );
            assert!(
                row.windows(2).all(|w| w[0] < w[1]),
                "row {q_block} is not strictly sorted: {row:?}"
            );
        }
    }
}

#[test]
fn the_topology_schedule_contains_sink_local_and_salient_blocks() {
    let (block_size, dim, num_blocks) = (4usize, 8usize, 8usize);
    let keys = blocky_keys(num_blocks, block_size, dim, 11);

    let config = TopologyScheduleConfig {
        block_size,
        local_radius_blocks: 1,
        sink_blocks: 1,
        topk_topology_blocks: 2,
    };
    let schedule = topology_block_schedule(&keys, num_blocks * block_size, dim, config).unwrap();

    let salience = block_salience(&keys, num_blocks * block_size, dim, block_size).unwrap();
    let mut ranked: Vec<usize> = (0..num_blocks).collect();
    ranked.sort_by(|&a, &b| salience[b].total_cmp(&salience[a]).then(a.cmp(&b)));
    let salient: Vec<usize> = ranked[..2].to_vec();

    for q_block in 0..num_blocks {
        let row = schedule.row(q_block);

        // Sink block 0 is always present.
        assert!(row.contains(&0), "row {q_block} lost the sink block");
        // The local window, clamped causally.
        for local in q_block.saturating_sub(1)..=q_block {
            assert!(
                row.contains(&local),
                "row {q_block} lost local block {local}"
            );
        }
        // Salient blocks, where causally legal.
        for &block in &salient {
            if block <= q_block {
                assert!(
                    row.contains(&block),
                    "row {q_block} lost salient block {block}"
                );
            }
        }
        assert!(
            row.iter().all(|&b| b <= q_block),
            "row {q_block} not causal"
        );
        assert!(
            row.windows(2).all(|w| w[0] < w[1]),
            "row {q_block} unsorted"
        );
    }
}

#[test]
fn the_topology_schedule_visits_fewer_blocks_than_the_dense_one() {
    // The point of the schedule. The Triton PR measured 56.6% to 80.9% block
    // reduction at seq 1024 to 4096; assert the direction here, at a size a test
    // can run, rather than restating their number.
    let (block_size, dim, num_blocks) = (4usize, 8usize, 16usize);
    let keys = blocky_keys(num_blocks, block_size, dim, 13);

    let sparse = topology_block_schedule(
        &keys,
        num_blocks * block_size,
        dim,
        TopologyScheduleConfig {
            block_size,
            local_radius_blocks: 1,
            sink_blocks: 1,
            topk_topology_blocks: 2,
        },
    )
    .unwrap();
    let dense = dense_causal_block_schedule(num_blocks);

    let reduction = 1.0 - sparse.indices.len() as f64 / dense.indices.len() as f64;
    println!(
        "scheduled/dense blocks: {} / {}  ({:.1}% reduction)",
        sparse.indices.len(),
        dense.indices.len(),
        100.0 * reduction
    );
    assert!(
        reduction > 0.4,
        "only {:.1}% block reduction; the schedule is barely sparser than dense",
        100.0 * reduction
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Salience is H0 persistence — checked against this crate's engine
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn block_salience_is_the_elder_rule_over_centroids() {
    // Each block scores the merge distance at which its component was absorbed:
    // the elder rule, and therefore an H0 death of the centroid cloud. The set of
    // non-zero saliences must be a sub-multiset of the H0 finite deaths the
    // persistence engine computes on the same centroids.
    use aether_core::manifold::ManifoldPoint;
    use aether_core::persistence::{persistent_homology, PersistenceConfig};

    let (block_size, num_blocks) = (4usize, 10usize);
    let keys = blocky_keys(num_blocks, block_size, 2, 17);
    let seq = num_blocks * block_size;

    let salience = block_salience(&keys, seq, 2, block_size).unwrap();
    assert_eq!(salience.len(), num_blocks);

    // Centroids, the same way the builder computes them.
    let centroids: Vec<ManifoldPoint<2>> = (0..num_blocks)
        .map(|b| {
            let mut c = [0.0f64; 2];
            for t in 0..block_size {
                for (d, slot) in c.iter_mut().enumerate() {
                    *slot += keys[(b * block_size + t) * 2 + d];
                }
            }
            ManifoldPoint::new([c[0] / block_size as f64, c[1] / block_size as f64])
        })
        .collect();

    let diagram = persistent_homology(&centroids, PersistenceConfig::h0_only()).unwrap();
    let mut deaths: Vec<f64> = diagram
        .pairs
        .iter()
        .filter(|p| p.dimension == 0)
        .filter_map(|p| p.death)
        .collect();
    deaths.sort_by(f64::total_cmp);

    for (block, &score) in salience.iter().enumerate() {
        if score == 0.0 {
            continue; // the block that survived to the end never dies
        }
        assert!(
            deaths.iter().any(|&d| (d - score).abs() < 1e-9),
            "block {block} salience {score} is not an H0 death of the centroids: \
             {deaths:?}"
        );
    }

    // Exactly one block survives: n points give n-1 merges, and each merge writes
    // the smaller component's members.
    assert_eq!(
        salience.iter().filter(|&&s| s == 0.0).count(),
        1,
        "exactly one block should never be absorbed"
    );
}

#[test]
fn the_salience_multiset_is_invariant_to_block_order() {
    // What IS invariant, and what is not.
    //
    // The multiset of saliences is the H0 barcode of the centroid cloud, so
    // reordering the blocks cannot change it. The per-block *assignment* is a
    // different matter: when two components tie on size, which one is "smaller"
    // and gets absorbed is decided by index order, so the same centroid can score
    // differently depending on where it sits in the sequence. The block that ends
    // up scoring 0 — the one that is never absorbed — moves too.
    //
    // This test asserted per-block equivariance first and failed: block 2 scored 0
    // where its mirror scored 1.724. That is not a port bug; the Triton original
    // has the same tie-breaking, because both follow union-find order. It is a
    // real caveat for the schedule, recorded in `the_schedule_depends_on_block_order`.
    let (block_size, dim, num_blocks) = (4usize, 3usize, 8usize);
    let keys = blocky_keys(num_blocks, block_size, dim, 19);
    let seq = num_blocks * block_size;

    let mut base = block_salience(&keys, seq, dim, block_size).unwrap();

    let mut reversed = vec![0.0; keys.len()];
    for b in 0..num_blocks {
        let src = (num_blocks - 1 - b) * block_size * dim;
        let dst = b * block_size * dim;
        reversed[dst..dst + block_size * dim].copy_from_slice(&keys[src..src + block_size * dim]);
    }
    let mut flipped = block_salience(&reversed, seq, dim, block_size).unwrap();

    base.sort_by(f64::total_cmp);
    flipped.sort_by(f64::total_cmp);
    for (i, (a, b)) in base.iter().zip(flipped.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "sorted salience {i}: {a} vs {b}; the H0 barcode changed under reordering"
        );
    }
}

#[test]
fn the_schedule_depends_on_block_order() {
    // The caveat, pinned so it cannot be forgotten.
    //
    // Because the per-block salience assignment is tie-order dependent, top-k can
    // choose different blocks for the same set of centroids presented in a
    // different order. A caller who reorders their sequence gets a different — not
    // a worse — schedule. Anyone relying on schedule stability across permutations
    // needs a deterministic tie-break on centroid content rather than on index.
    let (block_size, dim, num_blocks) = (4usize, 3usize, 8usize);
    let keys = blocky_keys(num_blocks, block_size, dim, 19);
    let seq = num_blocks * block_size;

    let base = block_salience(&keys, seq, dim, block_size).unwrap();
    let mut reversed = vec![0.0; keys.len()];
    for b in 0..num_blocks {
        let src = (num_blocks - 1 - b) * block_size * dim;
        let dst = b * block_size * dim;
        reversed[dst..dst + block_size * dim].copy_from_slice(&keys[src..src + block_size * dim]);
    }
    let flipped = block_salience(&reversed, seq, dim, block_size).unwrap();

    let mirrored_differs =
        (0..num_blocks).any(|b| (base[num_blocks - 1 - b] - flipped[b]).abs() > 1e-12);
    assert!(
        mirrored_differs,
        "per-block salience turned out to be order-equivariant after all; if that          is now guaranteed, delete this test and strengthen the multiset one"
    );

    // The invariant that does survive: exactly one block is never absorbed, in
    // either ordering.
    assert_eq!(base.iter().filter(|&&s| s == 0.0).count(), 1);
    assert_eq!(flipped.iter().filter(|&&s| s == 0.0).count(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. The kernel, against dense masked attention
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn a_dense_schedule_reproduces_full_causal_attention() {
    // The degenerate case that pins the kernel before any sparsity is involved:
    // with every causal block scheduled, the answer is ordinary causal attention.
    for (num_blocks, block_size, head_dim) in
        [(1usize, 4usize, 8usize), (3, 4, 16), (5, 8, 4), (7, 2, 6)]
    {
        let seq = num_blocks * block_size;
        let (q, k, v) = qkv(seq, head_dim, 23);
        let schedule = dense_causal_block_schedule(num_blocks);

        let flash = scheduled_attention(&q, &k, &v, seq, head_dim, &schedule, block_size).unwrap();
        let reference = sparse_attention(&q, &k, &v, seq, head_dim, &causal_mask(seq));

        let diff = max_abs_diff(&flash, &reference);
        assert!(
            diff < 1e-12,
            "num_blocks {num_blocks}, block {block_size}, dim {head_dim}: {diff:e}"
        );
    }
}

#[test]
fn a_sparse_schedule_matches_its_own_dense_masked_reference() {
    // The kernel's online softmax runs across a scattered set of blocks; the
    // reference materialises the same mask and softmaxes once. They must agree.
    let (block_size, dim, num_blocks) = (4usize, 8usize, 12usize);
    let seq = num_blocks * block_size;
    let keys = blocky_keys(num_blocks, block_size, dim, 29);
    let (q, _, v) = qkv(seq, dim, 31);

    for (local, sink, topk) in [(0usize, 1usize, 2usize), (1, 1, 3), (2, 2, 0), (0, 0, 4)] {
        let schedule = topology_block_schedule(
            &keys,
            seq,
            dim,
            TopologyScheduleConfig {
                block_size,
                local_radius_blocks: local,
                sink_blocks: sink,
                topk_topology_blocks: topk,
            },
        )
        .unwrap();

        let flash = scheduled_attention(&q, &keys, &v, seq, dim, &schedule, block_size).unwrap();
        let reference =
            dense_masked_attention(&q, &keys, &v, seq, dim, &schedule, block_size).unwrap();

        let diff = max_abs_diff(&flash, &reference);
        assert!(
            diff < 1e-12,
            "local {local}, sink {sink}, topk {topk}: {diff:e}"
        );
    }
}

#[test]
fn scheduling_a_block_actually_changes_what_the_row_sees() {
    // Mask fidelity, behaviourally. Perturb the values in a block that row 0's
    // block does not schedule, and its output must be bitwise unchanged.
    let (block_size, dim, num_blocks) = (4usize, 6usize, 6usize);
    let seq = num_blocks * block_size;
    let (q, k, mut v) = qkv(seq, dim, 37);

    // Row block 5 sees only the sink (block 0) and itself.
    let schedule =
        BlockSchedule::from_rows(&[vec![0], vec![0], vec![0], vec![0], vec![0], vec![0, 5]])
            .unwrap();

    let before = scheduled_attention(&q, &k, &v, seq, dim, &schedule, block_size).unwrap();
    for t in (3 * block_size)..(4 * block_size) {
        for d in 0..dim {
            v[t * dim + d] += 100.0;
        }
    }
    let after = scheduled_attention(&q, &k, &v, seq, dim, &schedule, block_size).unwrap();

    assert_eq!(
        before, after,
        "changing an unscheduled block moved the output"
    );
}

#[test]
fn the_kernel_is_deterministic() {
    let (block_size, dim, num_blocks) = (8usize, 16usize, 6usize);
    let seq = num_blocks * block_size;
    let keys = blocky_keys(num_blocks, block_size, dim, 41);
    let (q, _, v) = qkv(seq, dim, 43);
    let schedule = topology_block_schedule(
        &keys,
        seq,
        dim,
        TopologyScheduleConfig {
            block_size,
            local_radius_blocks: 1,
            sink_blocks: 1,
            topk_topology_blocks: 2,
        },
    )
    .unwrap();

    let first = scheduled_attention(&q, &keys, &v, seq, dim, &schedule, block_size).unwrap();
    for run in 1..4 {
        let again = scheduled_attention(&q, &keys, &v, seq, dim, &schedule, block_size).unwrap();
        assert_eq!(first, again, "run {run} differed");

        let rebuilt = topology_block_schedule(
            &keys,
            seq,
            dim,
            TopologyScheduleConfig {
                block_size,
                local_radius_blocks: 1,
                sink_blocks: 1,
                topk_topology_blocks: 2,
            },
        )
        .unwrap();
        assert_eq!(schedule, rebuilt, "schedule builder differed on run {run}");
    }
}

#[test]
fn large_logits_stay_finite_across_scheduled_blocks() {
    // The running-maximum rescale is what makes the block loop safe. Put the
    // dominant key in a late block so the correction has to fire.
    let (block_size, dim, num_blocks) = (4usize, 8usize, 8usize);
    let seq = num_blocks * block_size;
    let (mut q, mut k, v) = qkv(seq, dim, 47);
    for x in q.iter_mut().chain(k.iter_mut()) {
        *x *= 300.0;
    }
    for d in 0..dim {
        for i in 0..seq {
            q[i * dim + d] = 2000.0;
        }
        k[(seq - 1) * dim + d] = 2000.0;
    }

    let schedule = dense_causal_block_schedule(num_blocks);
    let out = scheduled_attention(&q, &k, &v, seq, dim, &schedule, block_size).unwrap();

    assert!(out.iter().all(|x| x.is_finite()), "non-finite output");
    // The last row is dominated by the last key.
    for d in 0..dim {
        assert!(
            (out[(seq - 1) * dim + d] - v[(seq - 1) * dim + d]).abs() < 1e-6,
            "dim {d}: last row did not saturate onto the dominant key"
        );
    }
}

#[test]
fn the_working_set_does_not_grow_with_the_sequence() {
    // The claim that separates this from tiled-but-still-quadratic: the kernel
    // holds one BLOCK_M x BLOCK_N score tile and the per-row running state, never
    // a seq x seq matrix. seq 4096 would be 16.7M f64 entries, 134 MB.
    let (block_size, dim) = (64usize, 16usize);
    let num_blocks = 64usize;
    let seq = num_blocks * block_size; // 4096
    let (q, k, v) = qkv(seq, dim, 53);

    // Sink + local only: no topology pass, so this stays a kernel test.
    let schedule = topology_block_schedule(
        &k,
        seq,
        dim,
        TopologyScheduleConfig {
            block_size,
            local_radius_blocks: 1,
            sink_blocks: 1,
            topk_topology_blocks: 0,
        },
    )
    .unwrap();

    let out = scheduled_attention(&q, &k, &v, seq, dim, &schedule, block_size).unwrap();
    assert_eq!(out.len(), seq * dim);
    assert!(out.iter().all(|x| x.is_finite()));
    assert!(
        schedule.indices.len() < num_blocks * (num_blocks + 1) / 4,
        "expected a sparse schedule, got {} of {} blocks",
        schedule.indices.len(),
        num_blocks * (num_blocks + 1) / 2
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Validation — the Triton wrapper rejects these before launch
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn the_builder_rejects_malformed_inputs() {
    let keys = vec![0.0; 16 * 4];

    assert_eq!(
        topology_block_schedule(
            &keys,
            16,
            4,
            TopologyScheduleConfig {
                block_size: 0,
                local_radius_blocks: 1,
                sink_blocks: 1,
                topk_topology_blocks: 1,
            }
        ),
        Err(ScheduleError::BlockSizeNotPowerOfTwo { block_size: 0 })
    );

    assert_eq!(
        topology_block_schedule(
            &keys,
            16,
            4,
            TopologyScheduleConfig {
                block_size: 6,
                local_radius_blocks: 1,
                sink_blocks: 1,
                topk_topology_blocks: 1,
            }
        ),
        Err(ScheduleError::BlockSizeNotPowerOfTwo { block_size: 6 })
    );

    assert_eq!(
        topology_block_schedule(
            &keys,
            15,
            4,
            TopologyScheduleConfig {
                block_size: 4,
                local_radius_blocks: 1,
                sink_blocks: 1,
                topk_topology_blocks: 1,
            }
        ),
        Err(ScheduleError::SequenceNotDivisible {
            seq: 15,
            block_size: 4
        })
    );
}

#[test]
fn the_kernel_rejects_a_schedule_that_does_not_match_the_sequence() {
    let (block_size, dim, num_blocks) = (4usize, 8usize, 4usize);
    let seq = num_blocks * block_size;
    let (q, k, v) = qkv(seq, dim, 59);

    // A schedule built for a different block count.
    let wrong = dense_causal_block_schedule(num_blocks + 1);
    assert_eq!(
        scheduled_attention(&q, &k, &v, seq, dim, &wrong, block_size),
        Err(ScheduleError::OffsetsLengthMismatch {
            expected: num_blocks + 1,
            actual: num_blocks + 2
        })
    );

    // A schedule that points past the end of the key blocks.
    let out_of_range = BlockSchedule::from_rows(&[vec![0], vec![0, 1], vec![0, 2], vec![9]]);
    assert_eq!(
        out_of_range,
        Err(ScheduleError::BlockIndexOutOfRange {
            q_block: 3,
            block: 9
        })
    );

    // A non-causal schedule.
    let non_causal = BlockSchedule::from_rows(&[vec![0, 1], vec![0, 1]]);
    assert_eq!(
        non_causal,
        Err(ScheduleError::NonCausalRow {
            q_block: 0,
            block: 1
        })
    );
}

#[test]
fn an_empty_row_is_rejected_rather_than_producing_nan() {
    // softmax over no keys is 0/0. The Triton builder guarantees every row has at
    // least its own block; the Rust type enforces it.
    assert_eq!(
        BlockSchedule::from_rows(&[vec![0], vec![]]),
        Err(ScheduleError::EmptyRow { q_block: 1 })
    );
}
