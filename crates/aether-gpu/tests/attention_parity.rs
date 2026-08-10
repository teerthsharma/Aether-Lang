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
    dense_causal_block_schedule, dense_masked_attention, scheduled_attention, BlockSchedule,
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
