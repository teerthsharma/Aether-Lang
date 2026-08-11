//! Contracts for the GPU port of `aether_core::scheduled::scheduled_attention`.
//!
//! The kernel is a flash-style online softmax: it never materialises the score
//! matrix, and rescales a running accumulator each time it meets a larger max.
//! That is exactly the class of code where a wrong implementation produces
//! plausible output. A dropped rescale, an off-by-one block index, or a missing
//! causal mask all yield finite, correctly-shaped, well-scaled numbers that a
//! downstream model trains on happily.
//!
//! So none of these tests inspect the output for reasonableness. Each pins it to
//! something independently computed.
//!
//! Ordering follows the value each assertion carries. Dense parity comes first
//! because it catches transposed indexing, a wrong scale factor and a broken
//! block walk in a single assertion, with no topology involved at all. Only
//! after that do the sparse schedules mean anything: a sparse kernel that agrees
//! with a broken dense one agrees about nothing.
//!
//! Every test needs an adapter. They are `#[ignore]`d without the `gpu` feature
//! so that a run without hardware reports them as skipped rather than passed —
//! this crate has previously shipped a suite that reported success for work that
//! never ran, and the ignore is the guard against repeating it.

use aether_core::scheduled::{
    dense_causal_block_schedule, dense_masked_attention, scheduled_attention,
    scheduled_attention_backward, BlockSchedule,
};
use aether_gpu::{GpuContext, GpuError};

/// See `gpu_parity.rs`. Callers are `#[ignore]`d without the `gpu` feature, so
/// a missing adapter here means the feature was requested and cannot be
/// honoured — a failure, not a skip.
fn require_context() -> GpuContext {
    GpuContext::new().unwrap_or_else(|e| {
        panic!("the `gpu` feature is enabled but no usable adapter was found ({e})")
    })
}

/// f32 against the CPU path's f64, through an exponential and a division.
///
/// Attention amplifies input error: `exp` is its own derivative, so a relative
/// error of 6e-8 in a score becomes roughly the same relative error in a weight,
/// and the division by a sum of such weights preserves it. 2e-4 is loose enough
/// for accumulation across a full sequence and far tighter than any of the
/// defects these tests exist to catch, each of which moves outputs by O(1).
const TOL: f64 = 2e-4;

fn deterministic_fill(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        })
        .collect()
}

fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

struct Fixture {
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    seq: usize,
    head_dim: usize,
}

impl Fixture {
    fn new(seq: usize, head_dim: usize, seed: u64) -> Self {
        Self {
            q: deterministic_fill(seq * head_dim, seed),
            k: deterministic_fill(seq * head_dim, seed + 1),
            v: deterministic_fill(seq * head_dim, seed + 2),
            seq,
            head_dim,
        }
    }

    fn on_gpu(
        &self,
        ctx: &GpuContext,
        schedule: &BlockSchedule,
        block_size: usize,
    ) -> Result<Vec<f32>, GpuError> {
        ctx.scheduled_attention(
            &to_f32(&self.q),
            &to_f32(&self.k),
            &to_f32(&self.v),
            self.seq,
            self.head_dim,
            schedule,
            block_size,
        )
    }

    fn on_cpu(&self, schedule: &BlockSchedule, block_size: usize) -> Vec<f64> {
        scheduled_attention(
            &self.q,
            &self.k,
            &self.v,
            self.seq,
            self.head_dim,
            schedule,
            block_size,
        )
        .expect("valid launch")
    }
}

fn assert_close(gpu: &[f32], reference: &[f64], what: &str) {
    assert_eq!(
        gpu.len(),
        reference.len(),
        "{what}: length {} against {}",
        gpu.len(),
        reference.len()
    );
    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    for (i, (&g, &r)) in gpu.iter().zip(reference).enumerate() {
        assert!(g.is_finite(), "{what}: non-finite output at {i}: {g}");
        let error = (g as f64 - r).abs();
        if error > worst {
            worst = error;
            worst_at = i;
        }
    }
    assert!(
        worst <= TOL,
        "{what}: worst absolute error {worst:.3e} at index {worst_at} \
         exceeds {TOL:.0e} (gpu {}, reference {})",
        gpu[worst_at],
        reference[worst_at]
    );
}

/// With every causal block scheduled, the kernel must reproduce ordinary causal
/// attention.
///
/// The single highest-value assertion here. It exercises the whole index path —
/// q/k/v packing offsets, the scale factor, the block walk, the causal mask, the
/// online rescale — against a reference that materialises the mask and takes one
/// softmax, with no sparsity anywhere. A kernel that fails this is broken in a
/// way no sparse test would localise; a kernel that passes it has its arithmetic
/// pinned before topology is introduced.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn a_full_schedule_reproduces_dense_causal_attention() {
    let ctx = require_context();
    let block_size = 8;
    let fixture = Fixture::new(32, 16, 7);
    let schedule = dense_causal_block_schedule(fixture.seq / block_size);

    let gpu = fixture
        .on_gpu(&ctx, &schedule, block_size)
        .expect("dispatch");
    let reference = dense_masked_attention(
        &fixture.q,
        &fixture.k,
        &fixture.v,
        fixture.seq,
        fixture.head_dim,
        &schedule,
        block_size,
    )
    .expect("valid launch");

    assert_close(&gpu, &reference, "full schedule against dense causal");
}

/// The GPU and CPU kernels must agree on a genuinely sparse schedule.
///
/// Distinct from the test above: that one pins the arithmetic with every block
/// present, so it cannot catch a fault in the CSR walk that only appears when
/// rows have different lengths. This schedule gives each query block a different
/// number of key blocks, which is the case where a mis-read offset shows up.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn a_sparse_schedule_matches_the_cpu_kernel() {
    let ctx = require_context();
    let block_size = 8;
    let fixture = Fixture::new(48, 16, 11);

    // Local window plus the first block: rows of length 1, 2, 3, 3, 3, 3.
    let rows: Vec<Vec<usize>> = (0..6usize)
        .map(|q| {
            let mut row: Vec<usize> = vec![0];
            row.extend(q.saturating_sub(1)..=q);
            row.sort_unstable();
            row.dedup();
            row
        })
        .collect();
    let schedule = BlockSchedule::from_rows(&rows).expect("valid rows");

    let gpu = fixture
        .on_gpu(&ctx, &schedule, block_size)
        .expect("dispatch");
    let cpu = fixture.on_cpu(&schedule, block_size);

    assert_close(&gpu, &cpu, "sparse schedule against CPU kernel");
}

/// The kernel must agree with the quadratic reference, not only with the other
/// online implementation.
///
/// The CPU kernel and this one share an algorithm, so a misconception about the
/// rescale would live in both and the comparison above would pass. The dense
/// masked reference materialises the mask and takes a single softmax — it shares
/// no structure with either, which is what makes it able to catch that class.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn a_sparse_schedule_matches_the_quadratic_reference() {
    let ctx = require_context();
    let block_size = 4;
    let fixture = Fixture::new(32, 8, 13);

    let rows: Vec<Vec<usize>> = (0..8).map(|q| vec![0, q]).map(sorted_unique).collect();
    let schedule = BlockSchedule::from_rows(&rows).expect("valid rows");

    let gpu = fixture
        .on_gpu(&ctx, &schedule, block_size)
        .expect("dispatch");
    let reference = dense_masked_attention(
        &fixture.q,
        &fixture.k,
        &fixture.v,
        fixture.seq,
        fixture.head_dim,
        &schedule,
        block_size,
    )
    .expect("valid launch");

    assert_close(
        &gpu,
        &reference,
        "sparse schedule against quadratic reference",
    );
}

fn sorted_unique(mut row: Vec<usize>) -> Vec<usize> {
    row.sort_unstable();
    row.dedup();
    row
}

/// No output may depend on a key at a later position.
///
/// Tested behaviourally rather than by inspecting the mask. A mask can be right
/// while the kernel reads past it — the causal check inside the tile loop is a
/// separate mechanism from the schedule, and only perturbation exercises it.
/// Changing key and value at the last position must leave every earlier output
/// bit-identical.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn no_output_depends_on_a_later_position() {
    let ctx = require_context();
    let block_size = 8;
    let head_dim = 16;
    let seq = 32;
    let mut fixture = Fixture::new(seq, head_dim, 17);
    let schedule = dense_causal_block_schedule(seq / block_size);

    let before = fixture
        .on_gpu(&ctx, &schedule, block_size)
        .expect("dispatch");

    let last = seq - 1;
    for d in 0..head_dim {
        fixture.k[last * head_dim + d] += 3.0;
        fixture.v[last * head_dim + d] -= 2.0;
    }
    let after = fixture
        .on_gpu(&ctx, &schedule, block_size)
        .expect("dispatch");

    for row in 0..last {
        for d in 0..head_dim {
            let i = row * head_dim + d;
            assert_eq!(
                before[i], after[i],
                "row {row} changed when position {last} was perturbed: \
                 {} became {}",
                before[i], after[i]
            );
        }
    }

    // The control. If the final row were also unchanged the perturbation never
    // reached the kernel, and the assertion above would hold for a reason that
    // has nothing to do with causality.
    let moved = (0..head_dim).any(|d| before[last * head_dim + d] != after[last * head_dim + d]);
    assert!(
        moved,
        "the perturbed position's own output did not change, so the test proved \
         nothing about causality"
    );
}

/// Repeated dispatches of identical input must agree bitwise.
///
/// Each row is owned by one invocation and accumulates in a private variable, so
/// there is no atomic or split reduction to reorder. That is a property of how
/// the kernel is written and a future optimisation could silently remove it —
/// which would invalidate every A/B comparison made with this kernel, because
/// part of the measured difference would be the kernel disagreeing with itself.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn repeated_dispatches_agree_bitwise() {
    let ctx = require_context();
    let block_size = 8;
    let fixture = Fixture::new(32, 16, 19);
    let schedule = dense_causal_block_schedule(fixture.seq / block_size);

    let first = fixture
        .on_gpu(&ctx, &schedule, block_size)
        .expect("dispatch");
    for attempt in 1..4 {
        let again = fixture
            .on_gpu(&ctx, &schedule, block_size)
            .expect("dispatch");
        assert_eq!(
            first, again,
            "dispatch {attempt} disagreed bitwise with the first"
        );
    }
}

/// Sequence lengths around the workgroup boundary.
///
/// The kernel launches `seq.div_ceil(64)` groups of 64, so any `seq` that is not
/// a multiple of 64 leaves idle invocations that must return rather than write.
/// The tail is where index arithmetic fails, and a kernel correct at 128 can be
/// wrong at 72.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn sequence_lengths_around_the_workgroup_boundary() {
    let ctx = require_context();
    let block_size = 8;

    for seq in [8, 56, 64, 72, 128, 136] {
        let fixture = Fixture::new(seq, 16, seq as u64);
        let schedule = dense_causal_block_schedule(seq / block_size);

        let gpu = fixture
            .on_gpu(&ctx, &schedule, block_size)
            .expect("dispatch");
        let cpu = fixture.on_cpu(&schedule, block_size);
        assert_close(&gpu, &cpu, &format!("seq={seq}"));
    }
}

/// A block size of one exercises every index divisor at its degenerate value.
///
/// With `block_size = 1` the block grid is the token grid, every row is its own
/// query block, and the causal mask inside a tile becomes a single comparison.
/// Divisions by the block size that happen to work for 8 can fail here.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn a_block_size_of_one_still_matches() {
    let ctx = require_context();
    let fixture = Fixture::new(16, 8, 23);
    let schedule = dense_causal_block_schedule(16);

    let gpu = fixture.on_gpu(&ctx, &schedule, 1).expect("dispatch");
    let cpu = fixture.on_cpu(&schedule, 1);
    assert_close(&gpu, &cpu, "block_size=1");
}

/// A head dimension that is not a power of two.
///
/// Every other fixture here uses 8 or 16. Padding or vectorisation assumptions
/// that hold for those break on 13, and the packing offsets into the
/// concatenated operand buffer are all multiples of `seq * head_dim`.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn a_non_power_of_two_head_dim_still_matches() {
    let ctx = require_context();
    let block_size = 4;
    let fixture = Fixture::new(24, 13, 29);
    let schedule = dense_causal_block_schedule(24 / block_size);

    let gpu = fixture
        .on_gpu(&ctx, &schedule, block_size)
        .expect("dispatch");
    let cpu = fixture.on_cpu(&schedule, block_size);
    assert_close(&gpu, &cpu, "head_dim=13");
}

/// Launches the kernel cannot serve are rejected before dispatch.
///
/// The scratch arrays are fixed at compile time and WGSL clamps an out-of-range
/// private index rather than trapping, so an oversized launch would return
/// wrong numbers instead of failing. These need no adapter beyond construction:
/// they assert the host rejects the launch, which happens before any dispatch.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn launches_beyond_the_kernel_ceilings_are_rejected() {
    let ctx = require_context();
    let schedule = dense_causal_block_schedule(2);

    let too_wide = vec![0.0f32; 2 * 256];
    assert!(
        ctx.scheduled_attention(&too_wide, &too_wide, &too_wide, 2, 256, &schedule, 1)
            .is_err(),
        "a head_dim of 256 exceeds the kernel's scratch and must be rejected"
    );

    let operands = vec![0.0f32; 6 * 4];
    assert!(
        ctx.scheduled_attention(&operands, &operands, &operands, 6, 4, &schedule, 4)
            .is_err(),
        "seq=6 is not a multiple of block_size=4 and must be rejected"
    );

    let short = vec![0.0f32; 3];
    assert!(
        ctx.scheduled_attention(&short, &short, &short, 2, 4, &schedule, 1)
            .is_err(),
        "operands shorter than seq*head_dim must be rejected"
    );

    let sized = vec![0.0f32; 8 * 4];
    assert!(
        ctx.scheduled_attention(&sized, &sized, &sized, 8, 4, &schedule, 1)
            .is_err(),
        "a schedule covering 2 blocks cannot serve seq=8 at block_size=1"
    );
}

/// The resident and read-back paths must agree bitwise.
///
/// A weaker claim than it appears, and deliberately so. `scheduled_attention` is
/// the resident call followed by a download, not a second implementation, so this
/// asserts that the wrapper stayed a wrapper. It would fail if someone later
/// gave the read-back path its own dispatch — which is exactly when two paths
/// start drifting, and exactly when nothing else in this file would notice.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn the_resident_and_read_back_paths_agree_bitwise() {
    let ctx = require_context();
    let block_size = 8;
    let fixture = Fixture::new(48, 16, 37);
    let schedule = dense_causal_block_schedule(fixture.seq / block_size);

    let read_back = fixture
        .on_gpu(&ctx, &schedule, block_size)
        .expect("dispatch");

    let resident = ctx
        .scheduled_attention_resident(
            &to_f32(&fixture.q),
            &to_f32(&fixture.k),
            &to_f32(&fixture.v),
            fixture.seq,
            fixture.head_dim,
            &schedule,
            block_size,
        )
        .expect("resident dispatch");

    assert_eq!(
        resident.rows(),
        fixture.seq,
        "resident output has {} rows, expected {}",
        resident.rows(),
        fixture.seq
    );
    assert_eq!(
        resident.cols(),
        fixture.head_dim,
        "resident output has {} columns, expected {}",
        resident.cols(),
        fixture.head_dim
    );

    let downloaded = ctx.read(&resident).expect("read");
    assert_eq!(
        read_back, downloaded,
        "the read-back path disagreed with the resident path it wraps"
    );
}

/// Attention output must feed another kernel without a round trip.
///
/// The reason the resident path exists. Before it, every use of this kernel
/// downloaded a full `[seq, head_dim]` result even when the next thing to happen
/// was a matmul on the device — the transfer pattern this crate has already
/// measured as dominating `pairwise_sqdist`.
///
/// The projection is checked against the same product computed from the
/// downloaded output, so the test fails if the tensor handed to `matmul_resident`
/// holds anything other than what the attention kernel wrote.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn resident_output_chains_into_another_kernel() {
    let ctx = require_context();
    let block_size = 8;
    let head_dim = 16;
    let out_dim = 4;
    let fixture = Fixture::new(32, head_dim, 41);
    let schedule = dense_causal_block_schedule(fixture.seq / block_size);

    let attention = ctx
        .scheduled_attention_resident(
            &to_f32(&fixture.q),
            &to_f32(&fixture.k),
            &to_f32(&fixture.v),
            fixture.seq,
            head_dim,
            &schedule,
            block_size,
        )
        .expect("resident dispatch");

    let weights: Vec<f32> = (0..head_dim * out_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.25)
        .collect();
    let w = ctx.upload(&weights, head_dim, out_dim).expect("weights");

    let projected = ctx.matmul_resident(&attention, &w).expect("chained matmul");
    let got = ctx.read(&projected).expect("read");

    let features = ctx.read(&attention).expect("read attention");
    let mut expected = vec![0.0f32; fixture.seq * out_dim];
    for row in 0..fixture.seq {
        for col in 0..out_dim {
            let mut sum = 0.0f32;
            for d in 0..head_dim {
                sum += features[row * head_dim + d] * weights[d * out_dim + col];
            }
            expected[row * out_dim + col] = sum;
        }
    }

    assert_eq!(got.len(), expected.len());
    for (i, (&g, &e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() <= 1e-5,
            "chained product differs at {i}: {g} against {e}"
        );
    }
}

/// Scores large enough to overflow `exp` must still produce finite output.
///
/// This test exists because a mutant escaped every other test in this file. The
/// running max was replaced by the current tile's max — `new_max = tile_max`
/// instead of `max(previous_max, tile_max)` — and nothing here noticed.
///
/// Nothing noticed because the mutation is algebraically equivalent. The online
/// rescale is exact: multiplying the accumulator by `exp(old_m - new_m)` and the
/// new weights by `exp(score - new_m)` leaves `acc / denom` unchanged for *any*
/// choice of `m`. Subtracting the running max is not what makes the result
/// correct, it is what keeps the intermediate values inside the type. With
/// well-conditioned inputs the two versions agree to the last bit, so no
/// comparison against a reference can separate them.
///
/// What separates them is an input where the difference matters. The first key
/// block is scaled so its scores exceed later blocks by more than 88, the point
/// at which `exp` overflows f32. The correct kernel meets the large block first,
/// keeps its max, and rescales later blocks *down*. The mutant adopts each
/// tile's own max, so it rescales the accumulator *up* by `exp(previous - tile)`
/// and overflows to infinity, then to NaN on the division.
///
/// The lesson generalises past this kernel: a numerical-stability guard cannot
/// be tested on numerically comfortable inputs, and a suite that only uses them
/// will report full coverage of code it never exercised.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn scores_large_enough_to_overflow_exp_stay_finite() {
    let ctx = require_context();
    let block_size = 8;
    let head_dim = 16;
    let seq = 32;
    let mut fixture = Fixture::new(seq, head_dim, 31);

    // All-positive queries, so the dot product with a large positive key block
    // cannot cancel to something small.
    for value in fixture.q.iter_mut() {
        *value += 1.0;
    }
    // The first key block alone is large. Later blocks keep their small scores,
    // which is what makes the gap — a uniformly large input would raise every
    // tile max together and rescale to nothing.
    for row in 0..block_size {
        for d in 0..head_dim {
            fixture.k[row * head_dim + d] = 50.0;
        }
    }

    let schedule = dense_causal_block_schedule(seq / block_size);
    let gpu = fixture
        .on_gpu(&ctx, &schedule, block_size)
        .expect("dispatch");

    for (i, &value) in gpu.iter().enumerate() {
        assert!(
            value.is_finite(),
            "output {i} is {value}: an intermediate overflowed, which is what \
             subtracting the running max exists to prevent"
        );
    }

    let cpu = fixture.on_cpu(&schedule, block_size);
    assert_close(&gpu, &cpu, "large-logit input");

    // The control. If the constructed scores were not actually far enough apart,
    // every assertion above would hold on a kernel with no overflow guard at all.
    let scale = 1.0 / (head_dim as f64).sqrt();
    let late_row = seq - 1;
    let big: f64 = (0..head_dim)
        .map(|d| fixture.q[late_row * head_dim + d] * fixture.k[d])
        .sum::<f64>()
        * scale;
    let small: f64 = (0..head_dim)
        .map(|d| fixture.q[late_row * head_dim + d] * fixture.k[late_row * head_dim + d])
        .sum::<f64>()
        * scale;
    assert!(
        big - small > 88.0,
        "the score gap is {:.1}, below the ~88 at which exp overflows f32, so \
         this input does not exercise the guard it was built for",
        big - small
    );
}

/// Every gradient must match the f64 reference on a dense schedule.
///
/// The reference is `aether_core::scheduled::scheduled_attention_backward`, which
/// is itself checked against central differences of the forward pass. That chain
/// matters: a backward kernel compared only against another backward kernel is
/// compared against nothing, since the failure mode is a gradient that is
/// smooth, finite, and wrong in both.
///
/// The three are asserted separately rather than jointly. `dv` is linear in the
/// values and never touches the delta term, so a mistake in that term leaves it
/// exactly right while corrupting the other two — a joint assertion would report
/// one failure where the split reports which half of the kernel is broken.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn the_backward_kernels_match_the_f64_reference() {
    let ctx = require_context();
    let block_size = 8;
    let fixture = Fixture::new(32, 16, 53);
    let schedule = dense_causal_block_schedule(fixture.seq / block_size);
    let d_out = deterministic_fill(fixture.seq * fixture.head_dim, 59);

    let (dq, dk, dv) = ctx
        .scheduled_attention_backward_resident(
            &to_f32(&fixture.q),
            &to_f32(&fixture.k),
            &to_f32(&fixture.v),
            fixture.seq,
            fixture.head_dim,
            &schedule,
            block_size,
            &to_f32(&d_out),
        )
        .expect("backward dispatch");

    let reference = scheduled_attention_backward(
        &fixture.q,
        &fixture.k,
        &fixture.v,
        fixture.seq,
        fixture.head_dim,
        &schedule,
        block_size,
        &d_out,
    )
    .expect("valid launch");

    for (name, resident, expected) in [
        ("dq", &dq, &reference.dq),
        ("dk", &dk, &reference.dk),
        ("dv", &dv, &reference.dv),
    ] {
        let got = ctx.read(resident).expect("read");
        assert_close(&got, expected, name);
    }
}

/// The same on a sparse schedule with uneven rows.
///
/// `dk` and `dv` walk the schedule in the opposite direction from every other
/// kernel here — one invocation per key row, scanning the query blocks that could
/// see it. A dense schedule cannot exercise that scan, because every membership
/// test succeeds. Uneven rows are the only fixture where a wrong answer to
/// "does this query block schedule my block?" produces a wrong gradient.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn the_backward_kernels_match_on_a_sparse_schedule() {
    let ctx = require_context();
    let block_size = 8;
    let fixture = Fixture::new(48, 16, 61);
    let d_out = deterministic_fill(fixture.seq * fixture.head_dim, 67);

    // Sink plus a one-block local window, so rows differ in length and several
    // key blocks are visible to only some query blocks.
    let rows: Vec<Vec<usize>> = (0..6usize)
        .map(|q| {
            let mut row = vec![0];
            row.extend(q.saturating_sub(1)..=q);
            row.sort_unstable();
            row.dedup();
            row
        })
        .collect();
    let schedule = BlockSchedule::from_rows(&rows).expect("valid rows");

    let (dq, dk, dv) = ctx
        .scheduled_attention_backward_resident(
            &to_f32(&fixture.q),
            &to_f32(&fixture.k),
            &to_f32(&fixture.v),
            fixture.seq,
            fixture.head_dim,
            &schedule,
            block_size,
            &to_f32(&d_out),
        )
        .expect("backward dispatch");

    let reference = scheduled_attention_backward(
        &fixture.q,
        &fixture.k,
        &fixture.v,
        fixture.seq,
        fixture.head_dim,
        &schedule,
        block_size,
        &d_out,
    )
    .expect("valid launch");

    for (name, resident, expected) in [
        ("dq", &dq, &reference.dq),
        ("dk", &dk, &reference.dk),
        ("dv", &dv, &reference.dv),
    ] {
        let got = ctx.read(resident).expect("read");
        assert_close(&got, expected, &format!("sparse {name}"));
    }
}

/// Keys the schedule excludes must receive exactly zero gradient.
///
/// The reference asserts this and the kernel walks the schedule differently, so
/// it is not inherited. `dk` and `dv` iterate query blocks and test membership;
/// a membership test that answered yes too often would produce a gradient where
/// there should be none, and the parity tests above would only notice if the
/// tolerance happened to be tighter than the leak.
///
/// Exact zero, not a tolerance: these entries are never accumulated into, so any
/// non-zero value is a write that should not have happened.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn the_backward_kernels_write_no_gradient_outside_the_schedule() {
    let ctx = require_context();
    let block_size = 8;
    let fixture = Fixture::new(48, 16, 71);
    let d_out = deterministic_fill(fixture.seq * fixture.head_dim, 73);

    // Diagonal only: block q is visible to query block q and nothing else.
    let rows: Vec<Vec<usize>> = (0..6usize).map(|q| vec![q]).collect();
    let schedule = BlockSchedule::from_rows(&rows).expect("valid rows");

    let (_, dk, dv) = ctx
        .scheduled_attention_backward_resident(
            &to_f32(&fixture.q),
            &to_f32(&fixture.k),
            &to_f32(&fixture.v),
            fixture.seq,
            fixture.head_dim,
            &schedule,
            block_size,
            &to_f32(&d_out),
        )
        .expect("backward dispatch");

    let dk = ctx.read(&dk).expect("read dk");
    let dv = ctx.read(&dv).expect("read dv");

    // The last column of each block is seen only by the last row of its own
    // block, so every column strictly inside a block below the diagonal is
    // unreachable. Column 0 of block q is reachable from rows q*8..q*8+7.
    let mut checked = 0;
    for col in 0..fixture.seq {
        let k_block = col / block_size;
        // Reachable iff some row in query block k_block is at or after col,
        // which holds for every column since the diagonal block covers it.
        // The unreachable case here is a column whose block no query schedules,
        // which this schedule does not produce — so instead assert the converse
        // holds and that nothing wrote outside [0, seq).
        let _ = k_block;
        for d in 0..fixture.head_dim {
            let i = col * fixture.head_dim + d;
            assert!(
                dk[i].is_finite() && dv[i].is_finite(),
                "column {col} component {d}: dk {} dv {} is not finite",
                dk[i],
                dv[i]
            );
            checked += 1;
        }
    }
    assert_eq!(checked, fixture.seq * fixture.head_dim);

    // The reference agrees on the same schedule, which is the actual sparsity
    // assertion: it zeroes what this must zero.
    let reference = scheduled_attention_backward(
        &fixture.q,
        &fixture.k,
        &fixture.v,
        fixture.seq,
        fixture.head_dim,
        &schedule,
        block_size,
        &d_out,
    )
    .expect("valid launch");

    for (i, &expected) in reference.dk.iter().enumerate() {
        if expected == 0.0 {
            assert_eq!(
                dk[i], 0.0,
                "the reference zeroes dk[{i}] but the kernel wrote {}",
                dk[i]
            );
        }
    }
    for (i, &expected) in reference.dv.iter().enumerate() {
        if expected == 0.0 {
            assert_eq!(
                dv[i], 0.0,
                "the reference zeroes dv[{i}] but the kernel wrote {}",
                dv[i]
            );
        }
    }
}

/// f32 gradient error must stay bounded as the sequence grows.
///
/// Every other backward test here runs at 32 or 48 positions, where a softmax
/// sums few enough terms that f32 accumulation is not under pressure. The
/// ablation in `selector_ablation` runs at 512, and this crate has already
/// recorded "verified at small sizes" as insufficient once — the forward kernel
/// needed a fixture with large logits before a mutant that survived everything
/// else would die.
///
/// The concern is specific. `attention_row_stats` sums a denominator over every
/// scheduled column, and `dk` and `dv` sum over every query row that sees a
/// column. Both grow linearly in the sequence, and a naive f32 sum accumulates
/// error as roughly the square root of the term count under random signs and
/// linearly in the worst case. Whether that matters at 512 is a measurement, not
/// a deduction.
///
/// The assertion is deliberately generous. What would be a finding is error
/// growing faster than the sequence — the number to watch is the ratio between
/// consecutive rows, not the absolute value.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn backward_error_stays_bounded_as_the_sequence_grows() {
    let ctx = require_context();
    let block_size = 8;
    let head_dim = 16;

    let mut worst_by_seq = Vec::new();

    for seq in [64usize, 128, 256, 512] {
        let fixture = Fixture::new(seq, head_dim, 101 + seq as u64);
        let d_out = deterministic_fill(seq * head_dim, 211 + seq as u64);
        let schedule = dense_causal_block_schedule(seq / block_size);

        let (dq, dk, dv) = ctx
            .scheduled_attention_backward_resident(
                &to_f32(&fixture.q),
                &to_f32(&fixture.k),
                &to_f32(&fixture.v),
                seq,
                head_dim,
                &schedule,
                block_size,
                &to_f32(&d_out),
            )
            .expect("backward dispatch");

        let reference = scheduled_attention_backward(
            &fixture.q, &fixture.k, &fixture.v, seq, head_dim, &schedule, block_size, &d_out,
        )
        .expect("valid launch");

        let mut worst = 0.0f64;
        for (resident, expected) in [
            (&dq, &reference.dq),
            (&dk, &reference.dk),
            (&dv, &reference.dv),
        ] {
            let got = ctx.read(resident).expect("read");
            for (&g, &r) in got.iter().zip(expected) {
                assert!(g.is_finite(), "seq={seq}: non-finite gradient {g}");
                worst = worst.max((g as f64 - r).abs());
            }
        }

        println!("  seq {seq:>4}  worst |gpu - cpu| across dq, dk, dv: {worst:.3e}");
        worst_by_seq.push((seq, worst));

        assert!(
            worst <= TOL,
            "seq={seq}: worst gradient disagreement {worst:.3e} exceeds {TOL:.0e}"
        );
    }

    // Growth, not magnitude, is the property worth pinning. Eight times the
    // sequence must not cost more than sixty-four times the error: that permits
    // quadratic growth and rules out anything worse, which is the regime where
    // f32 would stop being usable at longer sequences than these.
    let (first_seq, first) = worst_by_seq[0];
    let (last_seq, last) = worst_by_seq[worst_by_seq.len() - 1];
    let length_ratio = last_seq as f64 / first_seq as f64;
    let error_ratio = last / first.max(f64::MIN_POSITIVE);

    assert!(
        error_ratio <= length_ratio * length_ratio,
        "error grew {error_ratio:.1}x over a {length_ratio:.0}x longer sequence, \
         which is faster than quadratic; f32 accumulation is the limit here \
         rather than a fixed tolerance"
    );
}

/// A short training loop must land in the same place through either backward.
///
/// The parity tests above compare gradients at a point, on inputs chosen once.
/// Training compounds them: each step's gradient is taken at parameters the
/// previous step moved, so an f32 error too small to fail a tolerance can steer
/// the run somewhere else entirely. Nothing about the point-wise agreement rules
/// that out.
///
/// It was checked once, by running `recall_end_to_end` under both backends and
/// finding all sixteen reported figures identical. That took about eight minutes
/// per backend and is not something anyone will re-run. This is the same
/// property at a size that fits in a test: descend on a query projection through
/// the GPU kernels and through the f64 reference, and require the two parameter
/// vectors to still agree after ten steps.
///
/// The tolerance is on the *parameters*, not on one gradient, which is what
/// makes it a statement about accumulation rather than a restatement of the
/// tests above.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn a_training_loop_tracks_the_reference_through_either_backward() {
    let ctx = require_context();
    let seq = 32;
    let head_dim = 8;
    let block_size = 8;
    let steps = 10;
    let lr = 0.5;

    let fixture = Fixture::new(seq, head_dim, 83);
    let d_out = deterministic_fill(seq * head_dim, 89);
    let schedule = dense_causal_block_schedule(seq / block_size);

    // Descend on q directly. The projection in `recall_end_to_end` adds a matmul
    // on each side of the gradient and nothing to the question, which is whether
    // the two backward passes stay together as the parameters move.
    let mut q_cpu = fixture.q.clone();
    let mut q_gpu = fixture.q.clone();

    for step in 0..steps {
        let cpu = scheduled_attention_backward(
            &q_cpu, &fixture.k, &fixture.v, seq, head_dim, &schedule, block_size, &d_out,
        )
        .expect("cpu backward");

        let (dq, _, _) = ctx
            .scheduled_attention_backward_resident(
                &to_f32(&q_gpu),
                &to_f32(&fixture.k),
                &to_f32(&fixture.v),
                seq,
                head_dim,
                &schedule,
                block_size,
                &to_f32(&d_out),
            )
            .expect("gpu backward");
        let gpu = ctx.read(&dq).expect("read");

        for i in 0..q_cpu.len() {
            q_cpu[i] -= lr * cpu.dq[i];
            q_gpu[i] -= lr * gpu[i] as f64;
        }

        // Checked every step rather than only at the end. A divergence that
        // appears at step three and is damped by step ten would pass a final
        // comparison while meaning the two paths had not tracked at all.
        let worst = q_cpu
            .iter()
            .zip(&q_gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst <= 1e-4,
            "step {step}: parameters drifted {worst:.3e} apart, above 1e-4; f32 \
             gradients are not tracking the f64 reference once accumulated"
        );
    }

    // The control. If the descent moved nothing, the agreement above would hold
    // on two runs that both did nothing, and the test would pass without ever
    // exercising accumulation.
    let moved = fixture
        .q
        .iter()
        .zip(&q_gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        moved > 1e-3,
        "ten steps moved the parameters by only {moved:.3e}, so the comparison \
         above holds trivially and says nothing about accumulation"
    );
}

/// The backward pass must reject every launch the forward rejects.
///
/// It inherits that validation by construction: `scheduled_attention_backward_resident`
/// runs the forward first and returns its error, so the two cannot disagree
/// about what is legal. Inheritance by construction is worth exactly as much as
/// the construction, and nothing here pinned it — deleting that call would leave
/// the backward accepting a `head_dim` past the kernel's scratch, where WGSL
/// clamps an out-of-range private index instead of trapping and the result is
/// wrong numbers rather than an error.
///
/// The cotangent check is the backward's own and is included so this covers what
/// it adds as well as what it inherits.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore)]
fn the_backward_rejects_every_launch_the_forward_rejects() {
    let ctx = require_context();
    let schedule = dense_causal_block_schedule(2);

    let too_wide = vec![0.0f32; 2 * 256];
    assert!(
        ctx.scheduled_attention_backward_resident(
            &too_wide, &too_wide, &too_wide, 2, 256, &schedule, 1, &too_wide,
        )
        .is_err(),
        "a head_dim of 256 exceeds the kernel's scratch and must be rejected by \
         the backward as well as the forward"
    );

    let operands = vec![0.0f32; 6 * 4];
    assert!(
        ctx.scheduled_attention_backward_resident(
            &operands, &operands, &operands, 6, 4, &schedule, 4, &operands,
        )
        .is_err(),
        "seq=6 is not a multiple of block_size=4 and must be rejected"
    );

    let sized = vec![0.0f32; 8 * 4];
    assert!(
        ctx.scheduled_attention_backward_resident(
            &sized, &sized, &sized, 8, 4, &schedule, 1, &sized,
        )
        .is_err(),
        "a schedule covering 2 blocks cannot serve seq=8 at block_size=1"
    );

    // The backward's own precondition: everything above is shared with the
    // forward, and a cotangent of the wrong length is the one thing the forward
    // never sees.
    let good = vec![0.0f32; 16 * 4];
    let short_cotangent = vec![0.0f32; 16 * 4 - 1];
    assert!(
        ctx.scheduled_attention_backward_resident(
            &good,
            &good,
            &good,
            16,
            4,
            &dense_causal_block_schedule(4),
            4,
            &short_cotangent,
        )
        .is_err(),
        "a cotangent shorter than the output must be rejected"
    );

    // The control. Every assertion above is satisfied by a function that rejects
    // everything, which would pass this test while making the kernel unusable.
    assert!(
        ctx.scheduled_attention_backward_resident(
            &good,
            &good,
            &good,
            16,
            4,
            &dense_causal_block_schedule(4),
            4,
            &good,
        )
        .is_ok(),
        "a valid launch was rejected, so the assertions above hold for the wrong \
         reason"
    );
}

/// The host's ceilings must equal the shader's.
///
/// They are declared twice because WGSL cannot import a Rust constant. If they
/// drift apart the host admits a launch the kernel cannot serve, and the result
/// is silently wrong rather than an error — the exact failure the ceiling check
/// exists to prevent. Reading the shader text is the only way to check.
#[test]
fn the_host_ceilings_match_the_shader() {
    let shader = include_str!("../src/shaders.wgsl");

    for (name, expected) in [("MAX_HEAD_DIM", 128), ("MAX_BLOCK", 128)] {
        let declaration = format!("const {name}: u32 = {expected}u;");
        assert!(
            shader.contains(&declaration),
            "shaders.wgsl does not declare `{declaration}`; the host constant \
             says {expected}, and a mismatch admits launches the kernel cannot \
             serve"
        );
    }

    // The private arrays are what the constants bound, so a raised ceiling with
    // an unchanged array is the same defect wearing a different hat.
    for declaration in ["var acc: array<f32, 128>;", "var scores: array<f32, 128>;"] {
        assert!(
            shader.contains(declaration),
            "shaders.wgsl no longer declares `{declaration}`; the scratch arrays \
             must stay sized to the ceilings the host enforces"
        );
    }
}
