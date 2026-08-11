//! Correctness contracts for the GPU backend.
//!
//! The ordering is deliberate. Dense parity against a CPU reference is the
//! highest-value assertion available: it catches transposed strides, a wrong
//! index base, a mis-sized dispatch grid, and a bad uniform layout in a single
//! comparison, before anything interesting is attempted. Everything below it
//! catches a narrower class of bug.
//!
//! # Running these
//!
//! Every test here needs an adapter and is `#[ignore]`d unless the `gpu`
//! feature is on:
//!
//! ```text
//! cargo test -p aether-gpu --features gpu --release
//! ```
//!
//! This paragraph used to claim the suite "never reports success for work that
//! did not happen". That was false. The tests returned early on a missing
//! adapter, and an early return is a pass, so `cargo test --workspace` on a
//! GPU-less runner reported all of them green while running none. Marked
//! ignored, the same run prints `0 passed; 38 ignored`, which is the state the
//! sentence was describing rather than producing.
//!
//! `AETHER_REQUIRE_GPU=1` additionally turns a missing adapter into a failure,
//! for a run that must prove the hardware path was exercised.

use aether_gpu::{cpu_matmul, cpu_pairwise_sqdist, tensor_matmul, GpuContext};

/// f32 accumulation over k terms diverges from a separately-ordered f32
/// accumulation. The bound scales with k, so the tolerance does too rather
/// than being a single constant that is loose at k=4 and wrong at k=512.
fn tolerance(k: usize) -> f32 {
    1e-5 * (k as f32).sqrt().max(1.0)
}

/// The GPU context, or a failure.
///
/// Every caller of this is `#[ignore]`d without the `gpu` feature, so reaching
/// it means the feature was requested. Asking to run the hardware tests and
/// finding no hardware is a failure, not a skip: the alternative is the early
/// return that had this suite reporting green while running nothing.
///
/// This replaced an `Option` plus an `AETHER_REQUIRE_GPU` variable. Two
/// mechanisms guarding one property, neither enforcing the other, is worse than
/// one that cannot be bypassed.
fn require_context() -> GpuContext {
    GpuContext::new().unwrap_or_else(|e| {
        panic!(
            "the `gpu` feature is enabled but no usable adapter was found ({e}). \
             These tests exist to exercise hardware; without it there is nothing \
             to report."
        )
    })
}

/// Deterministic pseudo-random fill. A fixed generator rather than `rand` so a
/// failure reproduces exactly from the seed printed in the test name.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn the_gpu_reports_which_adapter_it_is_using() {
    let ctx = require_context();
    let info = ctx.adapter_info();

    println!("adapter: {}", info.name);
    println!("backend: {}", info.backend);
    println!("device type: {}", info.device_type);

    assert!(!info.name.is_empty(), "adapter must identify itself");
}

/// The claim under test is "runs on GPU", not "runs somewhere".
///
/// wgpu will happily hand back a software rasterizer (lavapipe, WARP) that
/// reports as `Cpu`. A speedup measured against one of those compares two CPU
/// implementations. This test states the requirement explicitly so that a
/// machine without real hardware produces a visible skip rather than a quiet
/// pass that gets reported as a GPU result.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn the_selected_adapter_is_real_hardware_not_a_software_rasterizer() {
    let ctx = require_context();
    let info = ctx.adapter_info();

    if !info.is_hardware() {
        eprintln!(
            "SKIP: adapter '{}' reports device_type={}, which is a software \
             implementation. GPU claims are not supported on this machine.",
            info.name, info.device_type
        );
        return;
    }

    println!(
        "hardware adapter confirmed: {} ({})",
        info.name, info.backend
    );
}

/// The load-bearing test. GPU matmul must equal the CPU reference.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn gpu_matmul_matches_the_cpu_reference() {
    let ctx = require_context();

    for (m, k, n) in [(4, 4, 4), (8, 16, 8), (32, 32, 32), (17, 5, 23)] {
        let a = fill(m * k, 1);
        let b = fill(k * n, 2);

        let gpu = ctx.matmul(&a, &b, m, k, n).expect("matmul dispatch");
        let cpu = cpu_matmul(&a, &b, m, k, n);

        assert_eq!(gpu.len(), cpu.len(), "output length for {m}x{k}x{n}");

        let tol = tolerance(k);
        let worst = gpu
            .iter()
            .zip(&cpu)
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);

        assert!(
            worst <= tol,
            "{m}x{k}x{n}: worst |gpu - cpu| = {worst:e}, tolerance {tol:e}"
        );
        println!("{m}x{k}x{n}: worst abs diff {worst:e} (tol {tol:e})");
    }
}

/// The tail tile is where dispatch arithmetic fails. 16 is the workgroup edge.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn shapes_around_the_workgroup_boundary_are_handled() {
    let ctx = require_context();

    for dim in [1usize, 15, 16, 17, 31, 32, 33] {
        let a = fill(dim * dim, dim as u64);
        let b = fill(dim * dim, dim as u64 + 100);

        let gpu = ctx.matmul(&a, &b, dim, dim, dim).expect("matmul dispatch");
        let cpu = cpu_matmul(&a, &b, dim, dim, dim);

        let tol = tolerance(dim);
        let worst = gpu
            .iter()
            .zip(&cpu)
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);

        assert!(worst <= tol, "dim {dim}: worst {worst:e} > tol {tol:e}");
    }
}

/// A non-square case catches an m/n transposition that square inputs hide.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn a_rectangular_product_is_not_transposed() {
    let ctx = require_context();

    // A is 2x3, B is 3x4, so C is 2x4. Hand-computed.
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![
        1.0, 0.0, 0.0, 1.0, //
        0.0, 1.0, 0.0, 1.0, //
        0.0, 0.0, 1.0, 1.0,
    ];

    let gpu = ctx.matmul(&a, &b, 2, 3, 4).expect("matmul dispatch");
    assert_eq!(gpu, vec![1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0]);
}

/// Same input, same output, bitwise, across repeated dispatches.
///
/// Non-determinism invalidates every A/B measurement downstream, because part
/// of the variance being attributed to a change is the kernel's own.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn repeated_dispatches_are_bitwise_identical() {
    let ctx = require_context();

    let (m, k, n) = (24, 24, 24);
    let a = fill(m * k, 7);
    let b = fill(k * n, 8);

    let first = ctx.matmul(&a, &b, m, k, n).expect("matmul dispatch");
    for run in 1..8 {
        let again = ctx.matmul(&a, &b, m, k, n).expect("matmul dispatch");
        assert_eq!(first, again, "run {run} diverged from run 0");
    }
}

#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn bias_is_broadcast_across_rows_not_down_columns() {
    let ctx = require_context();

    let a = vec![
        1.0, 2.0, 3.0, //
        4.0, 5.0, 6.0,
    ];
    let bias = vec![10.0, 20.0, 30.0];

    let gpu = ctx.add_bias(&a, &bias, 2, 3).expect("bias dispatch");
    assert_eq!(gpu, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn relu_clamps_negatives_and_leaves_positives_alone() {
    let ctx = require_context();

    let a = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
    let gpu = ctx.relu(&a).expect("relu dispatch");
    assert_eq!(gpu, vec![0.0, 0.0, 0.0, 0.5, 2.0]);
}

/// The boundary at exactly zero is a convention, and forward and backward must
/// agree on it. A backward pass that treats 0 as active where the forward
/// treats it as inactive trains to a different optimum, and the loss curve
/// looks fine while it happens.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn relu_backward_is_zero_at_exactly_zero() {
    let ctx = require_context();

    let pre = vec![-1.0, 0.0, 1.0];
    let grad = vec![5.0, 5.0, 5.0];

    let gpu = ctx
        .relu_backward(&pre, &grad)
        .expect("relu_backward dispatch");
    assert_eq!(gpu, vec![0.0, 0.0, 5.0]);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Resident tensors and the tiled kernel
// ═══════════════════════════════════════════════════════════════════════════════

/// The tiled kernel accumulates per 16-wide tile rather than straight down k,
/// so its f32 rounding differs from both the naive kernel and the CPU
/// reference. It must still land inside tolerance -- a tiling bug shows up as
/// a result that is wrong by far more than rounding, usually on the tail tile.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn the_tiled_kernel_matches_the_cpu_reference() {
    let ctx = require_context();

    for (m, k, n) in [(16, 16, 16), (33, 47, 19), (64, 64, 64), (1, 1, 1)] {
        let a = fill(m * k, 11);
        let b = fill(k * n, 12);

        let ga = ctx.upload(&a, m, k).expect("upload a");
        let gb = ctx.upload(&b, k, n).expect("upload b");
        let gc = ctx.matmul_resident(&ga, &gb).expect("tiled matmul");
        let gpu = ctx.read(&gc).expect("readback");

        let cpu = cpu_matmul(&a, &b, m, k, n);
        let tol = tolerance(k);
        let worst = gpu
            .iter()
            .zip(&cpu)
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);

        assert!(
            worst <= tol,
            "tiled {m}x{k}x{n}: worst {worst:e} > tol {tol:e}"
        );
        println!("tiled {m}x{k}x{n}: worst {worst:e} (tol {tol:e})");
    }
}

/// Many tile iterations over many workgroups.
///
/// The tiled kernel writes into workgroup memory, reads it, then overwrites it
/// on the next iteration. The barrier after the read is what stops the next
/// iteration's writes from racing the current reads. A mutation run showed that
/// removing it escaped every suite when the largest case was 64x64x64. This
/// test raises the work until the race surfaces: k=512 is 32 tile iterations
/// and a 128x128 output is 64 concurrent workgroups, and at that size the
/// barrier-free kernel fails here.
///
/// The size is the point. The same defect is invisible at 64x64x64, so a suite
/// whose largest case is small does not merely test less -- it reports a clean
/// pass on a kernel with a data race in it. That the race is caught at all is a
/// property of this adapter and this scheduling; a missing barrier remains
/// undefined behaviour, and this test is evidence of the defect, not a licence
/// to rely on the omission being detectable everywhere.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn the_tiled_kernel_is_correct_across_many_tile_iterations() {
    let ctx = require_context();

    // k = 512 is 32 tile iterations; 128x128 output is 64 concurrent workgroups.
    let (m, k, n) = (128, 512, 128);
    let a = fill(m * k, 81);
    let b = fill(k * n, 82);

    let ga = ctx.upload(&a, m, k).expect("upload");
    let gb = ctx.upload(&b, k, n).expect("upload");
    let gpu = ctx
        .read(&ctx.matmul_resident(&ga, &gb).expect("tiled"))
        .expect("read");

    let cpu = cpu_matmul(&a, &b, m, k, n);
    let tol = tolerance(k);
    let worst = gpu
        .iter()
        .zip(&cpu)
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);

    assert!(worst <= tol, "{m}x{k}x{n}: worst {worst:e} > tol {tol:e}");

    // Repeat: a race is probabilistic, and one clean pass is weak evidence.
    for run in 0..4 {
        let again = ctx
            .read(&ctx.matmul_resident(&ga, &gb).expect("tiled"))
            .expect("read");
        assert_eq!(
            gpu, again,
            "run {run} diverged, which would indicate a race"
        );
    }
}

/// A chain of resident operations must equal the same chain done through the
/// upload-and-read-back API. This is what licenses the training loop to keep
/// intermediates on the device.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn a_resident_chain_equals_the_same_chain_with_readbacks() {
    let ctx = require_context();

    let (m, k, n, p) = (24, 16, 32, 8);
    let a = fill(m * k, 21);
    let b = fill(k * n, 22);
    let c = fill(n * p, 23);

    // Resident: one readback at the very end.
    let ga = ctx.upload(&a, m, k).expect("upload");
    let gb = ctx.upload(&b, k, n).expect("upload");
    let gc = ctx.upload(&c, n, p).expect("upload");
    let ab = ctx.matmul_resident(&ga, &gb).expect("ab");
    let abc = ctx.matmul_resident(&ab, &gc).expect("abc");
    let resident = ctx.read(&abc).expect("readback");

    // Round-tripping: readback between every step.
    let ab_host = ctx.matmul(&a, &b, m, k, n).expect("ab host");
    let abc_host = ctx.matmul(&ab_host, &c, m, n, p).expect("abc host");

    let worst = resident
        .iter()
        .zip(&abc_host)
        .map(|(r, h)| (r - h).abs())
        .fold(0.0f32, f32::max);

    // Tiled and naive round differently, so this is a tolerance, not equality.
    let tol = tolerance(k.max(n));
    assert!(
        worst <= tol,
        "resident chain diverged from round-tripped chain by {worst:e} > {tol:e}"
    );
}

#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn a_resident_matmul_rejects_disagreeing_inner_dimensions() {
    let ctx = require_context();

    let a = ctx.upload(&fill(6, 1), 2, 3).expect("upload");
    let b = ctx.upload(&fill(8, 2), 4, 2).expect("upload");

    assert!(
        ctx.matmul_resident(&a, &b).is_err(),
        "2x3 by 4x2 has no valid product and must be an error"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// The Tensor bridge
// ═══════════════════════════════════════════════════════════════════════════════

/// `tensor_matmul` must agree with `Tensor::matmul` on the same input.
///
/// It is the only thing that makes the bridge usable: a caller swapping one for
/// the other is entitled to the same answer, within the f32 cost the doc states.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn tensor_matmul_matches_the_cpu_path() {
    use aether_core::ml::tensor::Tensor;

    let ctx = require_context();

    for (m, k, n) in [(4usize, 4usize, 4usize), (33, 17, 48), (64, 64, 64)] {
        let a64: Vec<f64> = fill(m * k, 401).iter().map(|v| *v as f64).collect();
        let b64: Vec<f64> = fill(k * n, 402).iter().map(|v| *v as f64).collect();

        let ta = Tensor::new(&a64, &[m, k]);
        let tb = Tensor::new(&b64, &[k, n]);

        let cpu = ta.matmul(&tb);
        let gpu = tensor_matmul(&ctx, &ta, &tb).expect("bridge");

        assert_eq!(gpu.shape, cpu.shape, "{m}x{k}x{n}: shape");

        let c = cpu.data.borrow();
        let g = gpu.data.borrow();
        let scale = c.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));

        let worst = g
            .iter()
            .zip(c.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
            / scale;

        // The f32 cost the doc states, with room for the reduction depth.
        let bound = 8.0 * 1.19e-7 * (k as f64).sqrt();
        assert!(
            worst < bound,
            "{m}x{k}x{n}: relative error {worst:e} exceeds {bound:e}"
        );
        println!("{m}x{k}x{n}: bridge vs Tensor::matmul, relative error {worst:.3e}");
    }
}

/// A non-contiguous tensor must read the same through the bridge as through
/// `Tensor::matmul`.
///
/// Nothing in `aether-core` produces one today — `transpose` copies into a
/// fresh contiguous tensor — so this fixture builds one by hand, swapping the
/// strides to describe a column-major view of the same buffer.
///
/// The test exists because the failure it guards is silent. `Tensor::matmul`
/// indexes through strides; a bridge that read the backing vector in order
/// would return a different answer for the same input, and the only symptom
/// would be a wrong number. Reading flat passes every contiguous test in this
/// file.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn the_tensor_bridge_reads_through_strides_not_the_flat_buffer() {
    use aether_core::ml::tensor::Tensor;

    let ctx = require_context();

    // A 2x3 laid out column-major: the buffer holds columns, and the strides
    // say so. Reading it flat would transpose the operand.
    let mut a = Tensor::new(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]);
    a.strides = vec![1, 2];

    let b = Tensor::new(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[3, 2]);

    let cpu = a.matmul(&b);
    let gpu = tensor_matmul(&ctx, &a, &b).expect("bridge");

    let c = cpu.data.borrow();
    let g = gpu.data.borrow();

    for (i, (x, y)) in g.iter().zip(c.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-6,
            "index {i}: bridge {x} vs Tensor::matmul {y}. The bridge is reading \
             the buffer flat and ignoring the strides."
        );
    }

    // And confirm the fixture is actually strided, so a future change to
    // `Tensor::new` cannot quietly make this test contiguous and vacuous.
    assert_ne!(
        a.strides,
        vec![3, 1],
        "the fixture is contiguous; it no longer tests what it was written for"
    );
}

/// Shapes the bridge cannot honour must be rejected, not silently reshaped.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn the_tensor_bridge_rejects_shapes_it_cannot_multiply() {
    use aether_core::ml::tensor::Tensor;

    let ctx = require_context();

    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let wrong_inner = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert!(tensor_matmul(&ctx, &a, &wrong_inner).is_err());

    let three_d = Tensor::new(&[1.0; 8], &[2, 2, 2]);
    assert!(tensor_matmul(&ctx, &a, &three_d).is_err());
    assert!(tensor_matmul(&ctx, &three_d, &a).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
// What f32 costs a Tensor consumer
//
// The performance case for routing `aether_core::ml::Tensor::matmul` to the GPU
// is measured: crossover at n=128, 38x at n=512 with conversion counted. The
// precision case is not. The topology tests establish that f32 is acceptable for
// *distances*, which is a different operation with a different error growth, and
// carrying that conclusion across would be exactly the kind of transfer this
// file exists to prevent.
//
// `Tensor` is f64. Routing it through an f32 kernel is a semantic change to
// every consumer, so the size of that change is worth knowing before anyone
// argues about whether it is acceptable.
// ═══════════════════════════════════════════════════════════════════════════════

/// f64 reference matmul, independent of `Tensor` so the comparison does not
/// depend on the implementation being replaced.
fn f64_matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// How far an f32 matmul drifts from f64, and how that grows with the reduction
/// depth.
///
/// The expectation is `sqrt(k)` growth: f32 has about 1.2e-7 of relative
/// precision, errors in a k-term dot product accumulate as a random walk, so
/// relative error should scale roughly with the square root of n. Asserting the
/// growth rather than a single tolerance is what distinguishes "f32 behaves like
/// f32" from "f32 behaves like something is wrong".
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn f32_matmul_error_grows_like_the_square_root_of_the_reduction_depth() {
    let ctx = require_context();

    let mut previous: Option<(usize, f64)> = None;

    for n in [16usize, 64, 256] {
        let a32 = fill(n * n, 301);
        let b32 = fill(n * n, 302);
        let a64: Vec<f64> = a32.iter().map(|v| *v as f64).collect();
        let b64: Vec<f64> = b32.iter().map(|v| *v as f64).collect();

        let gpu = ctx.matmul(&a32, &b32, n, n, n).expect("matmul");
        let exact = f64_matmul(&a64, &b64, n);

        // Relative to the magnitude of the result, not element by element: a
        // near-zero entry has unbounded relative error and says nothing.
        let scale = exact.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let worst = gpu
            .iter()
            .zip(&exact)
            .map(|(g, e)| ((*g as f64) - e).abs())
            .fold(0.0f64, f64::max)
            / scale;

        println!("n={n:>4}: worst relative error {worst:.3e}");

        // f32 epsilon is 1.19e-7; sqrt(n) growth with a generous constant.
        let bound = 8.0 * 1.19e-7 * (n as f64).sqrt();
        assert!(
            worst < bound,
            "n={n}: relative error {worst:e} exceeds {bound:e}, which is more \
             than f32 accumulation over {n} terms explains"
        );

        if let Some((pn, pe)) = previous {
            // Growth must not be dramatically faster than sqrt. A quadratic or
            // linear blow-up would indicate a real defect rather than rounding.
            let ratio = worst / pe.max(1e-30);
            let sqrt_ratio = ((n as f64) / (pn as f64)).sqrt();
            assert!(
                ratio < 6.0 * sqrt_ratio,
                "error grew {ratio:.2}x from n={pn} to n={n}, far above the \
                 {sqrt_ratio:.2}x that sqrt accumulation predicts"
            );
        }
        previous = Some((n, worst));
    }
}

/// The consumers that an f32 `Tensor` path would and would not serve.
///
/// This is the assertion that turns a precision number into a decision. The
/// relative error at n=256 is around 1e-6, which is:
///
/// - fine for neural network training, where gradients are noisy by several
///   orders of magnitude more than that, and where this crate already trains to
///   the same accuracy as the f64 CPU path;
/// - fine for clustering and classification, which threshold and argmax;
/// - **not** fine for anything asserting to 1e-9 or tighter, which includes the
///   persistence engine's own invariant suite.
///
/// So the honest recommendation is per-consumer, not per-crate, and this test
/// pins the number the recommendation rests on rather than leaving it in a
/// comment that drifts.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn f32_matmul_precision_is_stated_as_a_number_not_an_adjective() {
    let ctx = require_context();

    let n = 256;
    let a32 = fill(n * n, 311);
    let b32 = fill(n * n, 312);
    let a64: Vec<f64> = a32.iter().map(|v| *v as f64).collect();
    let b64: Vec<f64> = b32.iter().map(|v| *v as f64).collect();

    let gpu = ctx.matmul(&a32, &b32, n, n, n).expect("matmul");
    let exact = f64_matmul(&a64, &b64, n);

    let scale = exact.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let worst = gpu
        .iter()
        .zip(&exact)
        .map(|(g, e)| ((*g as f64) - e).abs())
        .fold(0.0f64, f64::max)
        / scale;

    println!("n=256 relative error: {worst:.3e}");

    // Comfortably above anything asserting at 1e-9, comfortably below anything
    // that matters to a thresholding consumer. Both directions are asserted, so
    // the test fails if the kernel silently becomes either much worse or much
    // better than the recommendation assumes.
    assert!(
        worst > 1e-9,
        "relative error {worst:e} is below 1e-9, so the claim that an f32 path \
         is unsuitable for 1e-9 assertions no longer holds and the \
         recommendation in FEATURES.md needs revisiting"
    );
    assert!(
        worst < 1e-4,
        "relative error {worst:e} is above 1e-4, which would make an f32 path \
         unsuitable for thresholding consumers too"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// The optimizer
//
// A mutation run found that flipping `sgd_update` from descent to ascent
// escaped every suite: no test asserted the direction of the parameter update,
// and both gradient checks stop at the gradients without ever applying one.
// The training examples would have diverged, but nothing under `cargo test`
// would have said so.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn sgd_subtracts_the_scaled_gradient() {
    let ctx = require_context();

    let params = ctx.upload(&[1.0, 2.0, 3.0, -1.0], 1, 4).expect("params");
    let grads = ctx.upload(&[0.5, -1.0, 2.0, 0.0], 1, 4).expect("grads");

    let updated = ctx
        .sgd_update_resident(&params, &grads, 0.1)
        .expect("update");
    let got = ctx.read(&updated).expect("read");

    // p - lr*g, elementwise, by hand.
    let want = [
        1.0 - 0.1 * 0.5,  // 0.95
        2.0 - 0.1 * -1.0, // 2.10
        3.0 - 0.1 * 2.0,  // 2.80
        -1.0 - 0.1 * 0.0, // -1.00
    ];

    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < 1e-6,
            "index {i}: got {g}, expected {w}. A sign here is the difference \
             between gradient descent and gradient ascent."
        );
    }
}

/// The update must move against the gradient, for any gradient. Stated as a
/// property rather than a fixture so it holds beyond the four values above.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn sgd_moves_every_parameter_against_its_gradient() {
    let ctx = require_context();

    let p_host = fill(64, 71);
    let g_host = fill(64, 72);

    let params = ctx.upload(&p_host, 8, 8).expect("params");
    let grads = ctx.upload(&g_host, 8, 8).expect("grads");

    let updated = ctx
        .sgd_update_resident(&params, &grads, 0.25)
        .expect("update");
    let got = ctx.read(&updated).expect("read");

    for i in 0..64 {
        let delta = got[i] - p_host[i];
        if g_host[i].abs() < 1e-6 {
            continue;
        }
        assert!(
            delta * g_host[i] < 0.0,
            "index {i}: gradient {} and step {delta} share a sign, which is ascent",
            g_host[i]
        );
    }
}

/// A zero learning rate must be a no-op. Catches a rate that is ignored or
/// hardcoded, which a single non-zero rate cannot distinguish from correct.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn a_zero_learning_rate_leaves_parameters_untouched() {
    let ctx = require_context();

    let p_host = fill(32, 73);
    let params = ctx.upload(&p_host, 4, 8).expect("params");
    let grads = ctx.upload(&fill(32, 74), 4, 8).expect("grads");

    let updated = ctx
        .sgd_update_resident(&params, &grads, 0.0)
        .expect("update");
    let got = ctx.read(&updated).expect("read");

    assert_eq!(got, p_host, "lr=0 must leave every parameter unchanged");
}

/// Doubling the rate must double the step. Catches a rate applied at the wrong
/// magnitude, which the direction property above cannot see.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn the_step_scales_linearly_with_the_learning_rate() {
    let ctx = require_context();

    let p_host = fill(16, 75);
    let params = ctx.upload(&p_host, 4, 4).expect("params");
    let grads = ctx.upload(&fill(16, 76), 4, 4).expect("grads");

    let one = ctx
        .read(&ctx.sgd_update_resident(&params, &grads, 0.1).expect("u1"))
        .expect("r1");
    let two = ctx
        .read(&ctx.sgd_update_resident(&params, &grads, 0.2).expect("u2"))
        .expect("r2");

    for i in 0..16 {
        let step_one = one[i] - p_host[i];
        let step_two = two[i] - p_host[i];
        assert!(
            (step_two - 2.0 * step_one).abs() < 1e-5,
            "index {i}: doubling the rate gave {step_two} against {step_one} doubled"
        );
    }
}

#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn sgd_rejects_a_gradient_of_the_wrong_length() {
    let ctx = require_context();

    let params = ctx.upload(&fill(12, 77), 3, 4).expect("params");
    let grads = ctx.upload(&fill(8, 78), 2, 4).expect("grads");

    assert!(ctx.sgd_update_resident(&params, &grads, 0.1).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Adam
//
// Checked against a CPU implementation of the same update rather than only for
// plausible behaviour. Adam's characteristic defects -- a missing bias
// correction, epsilon inside the square root instead of outside, a step counter
// that does not advance -- all still train. They produce a different optimiser
// that converges somewhere slightly worse, which no smoke test distinguishes
// from an unlucky seed.
// ═══════════════════════════════════════════════════════════════════════════════

const B1: f64 = 0.9;
const B2: f64 = 0.999;
const EPS: f64 = 1e-8;

/// The reference update, in f64, matching the kernel exactly.
fn adam_cpu(params: &mut [f64], grads: &[f64], m: &mut [f64], v: &mut [f64], t: u32, lr: f64) {
    for i in 0..params.len() {
        m[i] = B1 * m[i] + (1.0 - B1) * grads[i];
        v[i] = B2 * v[i] + (1.0 - B2) * grads[i] * grads[i];

        let mhat = m[i] / (1.0 - B1.powi(t as i32));
        let vhat = v[i] / (1.0 - B2.powi(t as i32));

        params[i] -= lr * mhat / (vhat.sqrt() + EPS);
    }
}

#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn adam_matches_a_cpu_reference_over_many_steps() {
    let ctx = require_context();

    let n = 40;
    let p0 = fill(n, 201);
    let grads: Vec<Vec<f32>> = (0..12).map(|s| fill(n, 300 + s)).collect();
    let lr = 0.01f32;

    let mut gp = ctx.upload(&p0, 1, n).expect("params");
    let mut state = ctx.adam_state(&gp).expect("state");

    let mut cp: Vec<f64> = p0.iter().map(|v| *v as f64).collect();
    let mut cm = vec![0.0f64; n];
    let mut cv = vec![0.0f64; n];

    for (step, g) in grads.iter().enumerate() {
        let gg = ctx.upload(g, 1, n).expect("grad");
        gp = ctx
            .adam_update_resident(&gp, &gg, &mut state, lr)
            .expect("adam");

        let g64: Vec<f64> = g.iter().map(|v| *v as f64).collect();
        adam_cpu(&mut cp, &g64, &mut cm, &mut cv, step as u32 + 1, lr as f64);

        assert_eq!(state.step(), step as u32 + 1, "step counter must advance");
    }

    let got = ctx.read(&gp).expect("read");
    let mut worst = 0.0f64;
    for i in 0..n {
        let rel = ((got[i] as f64) - cp[i]).abs() / cp[i].abs().max(1e-4);
        worst = worst.max(rel);
    }

    assert!(
        worst < 5e-3,
        "worst relative divergence from the CPU reference after 12 steps: {worst:e}"
    );
    println!("adam: 12 steps, worst relative divergence {worst:.3e}");
}

/// The first step is where bias correction is visible and nowhere else.
///
/// With both moments at zero, the uncorrected first moment is `0.1 * g` and the
/// uncorrected second is `0.001 * g^2`, so an implementation without correction
/// takes a first step roughly `sqrt(0.001)/0.1` times the right size. The
/// correction decays as `t` grows, so a comparison after many steps cannot see
/// whether it is there.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn the_first_adam_step_applies_bias_correction() {
    let ctx = require_context();

    let n = 8;
    let params = vec![0.0f32; n];
    let grads = vec![1.0f32; n];

    let gp = ctx.upload(&params, 1, n).expect("params");
    let gg = ctx.upload(&grads, 1, n).expect("grad");
    let mut state = ctx.adam_state(&gp).expect("state");

    let updated = ctx
        .adam_update_resident(&gp, &gg, &mut state, 0.1)
        .expect("adam");
    let got = ctx.read(&updated).expect("read");

    // With correction, mhat = g and vhat = g^2, so the step is exactly lr for a
    // unit gradient. Without it the step would be about 0.0316 * lr.
    for (i, v) in got.iter().enumerate() {
        assert!(
            (v + 0.1).abs() < 1e-4,
            "index {i}: first step gave {v}, expected -0.1. A step near -0.00316 \
             means the bias correction is missing."
        );
    }
}

/// Adam's step size is nearly independent of gradient magnitude, which is the
/// property that distinguishes it from SGD. Scaling every gradient by 100 must
/// not scale the step by 100.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn adam_normalises_away_the_gradient_scale() {
    let ctx = require_context();

    let n = 16;
    let small = vec![0.01f32; n];
    let large = vec![1.0f32; n];

    let step_for = |g: &Vec<f32>| -> f32 {
        let gp = ctx.upload(&vec![0.0f32; n], 1, n).expect("params");
        let gg = ctx.upload(g, 1, n).expect("grad");
        let mut state = ctx.adam_state(&gp).expect("state");
        let out = ctx
            .adam_update_resident(&gp, &gg, &mut state, 0.1)
            .expect("adam");
        ctx.read(&out).expect("read")[0]
    };

    let s_small = step_for(&small);
    let s_large = step_for(&large);

    // A hundredfold change in gradient must move the step by well under 2x.
    let ratio = (s_small / s_large).abs();
    assert!(
        (0.5..2.0).contains(&ratio),
        "gradients differing by 100x gave steps {s_small} and {s_large}, ratio {ratio}"
    );
}

/// Epsilon must be added to `sqrt(vhat)`, not folded inside the root.
///
/// At ordinary gradient magnitudes the two forms are indistinguishable: with
/// `vhat` around 1e-4 they differ by roughly 5e-5 relative, which sits inside
/// any tolerance the other Adam tests can justify, and a mutation run confirmed
/// `sqrt(vhat + eps)` escaped every suite.
///
/// The difference is only visible where `vhat` is comparable to epsilon. With a
/// gradient of 1e-6 the corrected second moment is 1e-12, so `sqrt(vhat)` is
/// 1e-6 and epsilon is a one percent correction outside the root -- while
/// inside it, `sqrt(1e-12 + 1e-8)` is about 1e-4, a hundredfold larger
/// denominator and a hundredfold smaller step.
///
/// This is the general shape of an epsilon-placement bug: invisible in the
/// regime the code normally runs in, decisive in the regime epsilon exists for.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn adams_epsilon_sits_outside_the_square_root() {
    let ctx = require_context();

    let n = 8;
    let lr = 0.1f32;
    let g = 1e-6f32;

    let gp = ctx.upload(&vec![0.0f32; n], 1, n).expect("params");
    let gg = ctx.upload(&vec![g; n], 1, n).expect("grad");
    let mut state = ctx.adam_state(&gp).expect("state");

    let out = ctx
        .adam_update_resident(&gp, &gg, &mut state, lr)
        .expect("adam");
    let got = ctx.read(&out).expect("read");

    // Step 1, bias-corrected: mhat = g, vhat = g^2, so sqrt(vhat) = g.
    let correct = -(lr as f64) * (g as f64) / (g as f64 + 1e-8);
    let inside = -(lr as f64) * (g as f64) / ((g as f64).powi(2) + 1e-8).sqrt();

    for (i, v) in got.iter().enumerate() {
        let d_correct = ((*v as f64) - correct).abs();
        let d_inside = ((*v as f64) - inside).abs();
        assert!(
            d_correct < d_inside,
            "index {i}: step {v} is closer to the epsilon-inside-root value \
             {inside:.6e} than to the correct {correct:.6e}"
        );
        assert!(
            d_correct / correct.abs() < 1e-3,
            "index {i}: step {v} differs from the expected {correct:.6e}"
        );
    }
}

#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn adam_rejects_state_sized_for_a_different_parameter_count() {
    let ctx = require_context();

    let p = ctx.upload(&fill(10, 211), 1, 10).expect("params");
    let other = ctx.upload(&fill(4, 212), 1, 4).expect("other");
    let g = ctx.upload(&fill(10, 213), 1, 10).expect("grad");

    let mut state = ctx.adam_state(&other).expect("state");
    assert!(ctx.adam_update_resident(&p, &g, &mut state, 0.01).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Softmax and categorical cross-entropy
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn softmax_rows_are_probability_distributions() {
    let ctx = require_context();

    let (rows, classes) = (7, 4);
    let logits = ctx
        .upload(&fill(rows * classes, 41), rows, classes)
        .expect("upload");
    let p = ctx
        .read(&ctx.softmax_resident(&logits).expect("softmax"))
        .expect("read");

    for r in 0..rows {
        let row = &p[r * classes..(r + 1) * classes];
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "row {r} sums to {sum}, expected 1"
        );
        for (j, v) in row.iter().enumerate() {
            assert!(
                *v > 0.0 && *v < 1.0,
                "row {r} class {j} is {v}, outside (0, 1)"
            );
        }
    }
}

/// Softmax is invariant to adding a constant to every logit in a row. This is
/// the identity the max-subtraction relies on, so it is asserted rather than
/// assumed.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn softmax_is_invariant_to_a_constant_shift() {
    let ctx = require_context();

    let (rows, classes) = (5, 3);
    let base = fill(rows * classes, 42);
    let shifted: Vec<f32> = base.iter().map(|v| v + 12.5).collect();

    let a = ctx.upload(&base, rows, classes).expect("upload");
    let b = ctx.upload(&shifted, rows, classes).expect("upload");

    let pa = ctx
        .read(&ctx.softmax_resident(&a).expect("softmax"))
        .expect("read");
    let pb = ctx
        .read(&ctx.softmax_resident(&b).expect("softmax"))
        .expect("read");

    for (i, (x, y)) in pa.iter().zip(&pb).enumerate() {
        assert!(
            (x - y).abs() < 1e-6,
            "index {i}: shifted logits gave {y} vs {x}"
        );
    }
}

/// Without the max subtraction, a logit above roughly 88 overflows f32 `exp`
/// to inf, and inf/inf is NaN. This is the test that would fail if someone
/// simplified the kernel back to the naive form.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn large_logits_do_not_produce_nan_in_softmax() {
    let ctx = require_context();

    let logits = vec![
        1000.0, 999.0, 998.0, //
        -1000.0, -999.0, -998.0,
    ];
    let g = ctx.upload(&logits, 2, 3).expect("upload");
    let p = ctx
        .read(&ctx.softmax_resident(&g).expect("softmax"))
        .expect("read");

    for (i, v) in p.iter().enumerate() {
        assert!(v.is_finite(), "index {i} is {v}");
    }

    for r in 0..2 {
        let sum: f32 = p[r * 3..(r + 1) * 3].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "row {r} sums to {sum}");
    }
}

/// The fused gradient must equal `softmax(z) - y` divided by the batch size.
/// Computed here from the separately-tested softmax kernel, so the two paths
/// have to agree.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn the_fused_softmax_gradient_equals_softmax_minus_target() {
    let ctx = require_context();

    let (rows, classes) = (6, 3);
    let logits_host = fill(rows * classes, 43);

    // One-hot targets, class = row index mod classes.
    let mut targets = vec![0.0f32; rows * classes];
    for r in 0..rows {
        targets[r * classes + (r % classes)] = 1.0;
    }

    let gl = ctx.upload(&logits_host, rows, classes).expect("upload");
    let gt = ctx.upload(&targets, rows, classes).expect("upload");

    let fused = ctx
        .read(&ctx.softmax_xent_grad_resident(&gl, &gt).expect("grad"))
        .expect("read");
    let p = ctx
        .read(&ctx.softmax_resident(&gl).expect("softmax"))
        .expect("read");

    for i in 0..rows * classes {
        let want = (p[i] - targets[i]) / rows as f32;
        assert!(
            (fused[i] - want).abs() < 1e-6,
            "index {i}: fused {} vs softmax-minus-target {want}",
            fused[i]
        );
    }
}

/// A correct cross-entropy gradient sums to zero across each row, because the
/// softmax probabilities sum to one and the one-hot target sums to one. This
/// catches a missing or doubled target term that a magnitude check would not.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn the_softmax_gradient_sums_to_zero_across_each_row() {
    let ctx = require_context();

    let (rows, classes) = (8, 5);
    let logits_host = fill(rows * classes, 44);
    let mut targets = vec![0.0f32; rows * classes];
    for r in 0..rows {
        targets[r * classes + (r % classes)] = 1.0;
    }

    let gl = ctx.upload(&logits_host, rows, classes).expect("upload");
    let gt = ctx.upload(&targets, rows, classes).expect("upload");
    let g = ctx
        .read(&ctx.softmax_xent_grad_resident(&gl, &gt).expect("grad"))
        .expect("read");

    for r in 0..rows {
        let sum: f32 = g[r * classes..(r + 1) * classes].iter().sum();
        assert!(
            sum.abs() < 1e-6,
            "row {r} gradient sums to {sum}, expected 0"
        );
    }
}

#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn the_softmax_gradient_rejects_mismatched_target_shapes() {
    let ctx = require_context();

    let logits = ctx.upload(&fill(12, 45), 4, 3).expect("upload");
    let targets = ctx.upload(&fill(8, 46), 4, 2).expect("upload");

    assert!(ctx.softmax_xent_grad_resident(&logits, &targets).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Batched submission
// ═══════════════════════════════════════════════════════════════════════════════

/// Asserts the batching is real, by counting recorded dispatches rather than
/// by observing that something got faster. A faster kernel would also produce
/// a timing improvement, so timing alone cannot distinguish the two.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn resident_operations_accumulate_into_one_submission() {
    let ctx = require_context();

    let a = ctx.upload(&fill(64, 1), 8, 8).expect("upload");
    let b = ctx.upload(&fill(64, 2), 8, 8).expect("upload");

    assert_eq!(
        ctx.pending_dispatches(),
        0,
        "nothing recorded before any work"
    );

    let mut cur = ctx.matmul_resident(&a, &b).expect("matmul");
    for _ in 0..5 {
        cur = ctx.matmul_resident(&cur, &b).expect("matmul");
    }

    assert_eq!(
        ctx.pending_dispatches(),
        6,
        "six matmuls must accumulate, not submit one at a time"
    );

    let _ = ctx.read(&cur).expect("read");

    assert_eq!(
        ctx.pending_dispatches(),
        0,
        "read must flush the batch it depends on"
    );
}

/// Deferring work must not change it. Same chain, forced to submit after every
/// step, has to produce the same bytes as the batched version.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn batching_does_not_change_the_result() {
    let ctx = require_context();

    let a = ctx.upload(&fill(256, 3), 16, 16).expect("upload");
    let b = ctx.upload(&fill(256, 4), 16, 16).expect("upload");

    let mut batched = ctx.matmul_resident(&a, &b).expect("matmul");
    for _ in 0..3 {
        batched = ctx.matmul_resident(&batched, &b).expect("matmul");
    }
    let batched = ctx.read(&batched).expect("read");

    let mut stepwise = ctx.matmul_resident(&a, &b).expect("matmul");
    ctx.flush();
    for _ in 0..3 {
        stepwise = ctx.matmul_resident(&stepwise, &b).expect("matmul");
        ctx.flush();
    }
    let stepwise = ctx.read(&stepwise).expect("read");

    assert_eq!(
        batched, stepwise,
        "batching changed the result; submission grouping must be invisible"
    );
}

/// Work that is recorded and never flushed is never executed. That is a
/// deliberate property of the design and is pinned so it cannot regress into
/// an implicit flush that silently costs a submission per operation.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn flush_is_required_for_recorded_work_to_execute() {
    let ctx = require_context();

    let a = ctx.upload(&fill(64, 5), 8, 8).expect("upload");
    let b = ctx.upload(&fill(64, 6), 8, 8).expect("upload");

    let _ = ctx.matmul_resident(&a, &b).expect("matmul");
    assert_eq!(ctx.pending_dispatches(), 1);

    ctx.flush();
    assert_eq!(ctx.pending_dispatches(), 0);

    // Flushing an empty batch is a no-op, not an error.
    ctx.flush();
    assert_eq!(ctx.pending_dispatches(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pairwise distance -- the op that matters to this repository specifically
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn the_distance_matrix_matches_the_cpu_reference() {
    let ctx = require_context();

    for (n, d) in [(8usize, 2usize), (32, 3), (17, 5), (64, 8)] {
        let pts = fill(n * d, 31 + n as u64);

        let g = ctx.upload(&pts, n, d).expect("upload");
        let gd = ctx.pairwise_sqdist_resident(&g).expect("sqdist");
        let gpu = ctx.read(&gd).expect("readback");

        let cpu = cpu_pairwise_sqdist(&pts, n, d);
        let worst = gpu
            .iter()
            .zip(&cpu)
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);

        assert!(worst <= tolerance(d), "n={n} d={d}: worst {worst:e}");
        println!("sqdist n={n} d={d}: worst {worst:e}");
    }
}

/// A distance matrix is symmetric with a zero diagonal. Both properties are
/// free to check and both fail loudly on an index transposition, which is the
/// bug this kernel is most likely to have.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn the_distance_matrix_is_symmetric_with_a_zero_diagonal() {
    let ctx = require_context();

    let (n, d) = (48, 4);
    let pts = fill(n * d, 99);

    let g = ctx.upload(&pts, n, d).expect("upload");
    let gd = ctx.pairwise_sqdist_resident(&g).expect("sqdist");
    let m = ctx.read(&gd).expect("readback");

    for i in 0..n {
        assert_eq!(
            m[i * n + i],
            0.0,
            "diagonal at {i} is {}, expected exactly 0",
            m[i * n + i]
        );
        for j in 0..n {
            // Bitwise, not within a tolerance.
            //
            // IEEE-754 subtraction is exactly antisymmetric: `a - b` and `b - a`
            // differ only in sign bit, so their squares are bitwise identical,
            // and the kernel accumulates both orders over the same `d` range.
            // Symmetry is therefore a property of the arithmetic rather than an
            // approximation, and asserting it to 1e-5 -- as this test did
            // originally -- would accept a real indexing bug that produced
            // slightly different entries.
            assert_eq!(
                m[i * n + j].to_bits(),
                m[j * n + i].to_bits(),
                "asymmetry at ({i},{j}): {} vs {}",
                m[i * n + j],
                m[j * n + i]
            );
        }
    }
}

/// Distances are non-negative and obey the triangle inequality in the
/// un-squared metric. Checked on the root, since the squared form does not
/// satisfy the triangle inequality.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn distances_are_non_negative_and_satisfy_the_triangle_inequality() {
    let ctx = require_context();

    let (n, d) = (24, 3);
    let pts = fill(n * d, 77);

    let g = ctx.upload(&pts, n, d).expect("upload");
    let gd = ctx.pairwise_sqdist_resident(&g).expect("sqdist");
    let sq = ctx.read(&gd).expect("readback");

    for v in &sq {
        assert!(*v >= -1e-6, "negative squared distance {v}");
    }

    let dist = |i: usize, j: usize| sq[i * n + j].max(0.0).sqrt();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let (ij, ik, kj) = (dist(i, j), dist(i, k), dist(k, j));
                assert!(
                    ij <= ik + kj + 1e-4,
                    "triangle inequality violated: d({i},{j})={ij} > {ik} + {kj}"
                );
            }
        }
    }
}

#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn mismatched_shapes_are_rejected_rather_than_dispatched() {
    let ctx = require_context();

    let a = vec![1.0; 6];
    let b = vec![1.0; 6];

    assert!(
        ctx.matmul(&a, &b, 2, 3, 3).is_err(),
        "b is 6 elements but k*n = 9; this must be an error, not a dispatch"
    );
    assert!(
        ctx.matmul(&a, &b, 5, 5, 5).is_err(),
        "a is 6 elements but m*k = 25; this must be an error, not a dispatch"
    );
}

/// The fused gradient must be the derivative of the loss, not merely `p - y`.
///
/// `the_fused_gradient_equals_softmax_minus_target` checks the kernel implements
/// that formula, with both sides computed from the GPU's own softmax. That pins
/// the implementation against the intention and cannot notice if the intention
/// is wrong: `p - y` is the gradient of mean categorical cross-entropy, and
/// nothing here differentiates a cross-entropy to confirm it.
///
/// The equivalent gap on the CPU side was where a real defect hid — a softmax
/// layer whose gradient was identically zero passed every test that compared it
/// against a formula, and fell only to a finite difference of the loss.
///
/// So this differences the loss. Cross-entropy is computed host-side in f64 from
/// the logits, and each logit is perturbed to compare `dL/dz` against what the
/// kernel produced. No GPU softmax appears on the reference side, which is what
/// makes it an independent check rather than a restatement.
///
/// The independence is demonstrated rather than argued. Replacing `exp(x)` with
/// `exp(2x)` at all three inline softmax sites in the shader is a common-mode
/// defect: the rows are still a valid probability distribution, the fused
/// gradient still equals `p - y` because both sides of that comparison move
/// together, and the row sums are still zero. Three existing tests pass. This one
/// fails, because its reference never touches the GPU.
///
/// A first attempt to show this used a different defect — dropping the `/ rows`
/// averaging — and did not separate them, since the formula test encodes the
/// same division on its reference side. Common-mode is the shape that matters:
/// two wrong things agreeing is invisible to any check that compares them to
/// each other.
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn the_fused_gradient_is_the_derivative_of_cross_entropy() {
    let ctx = require_context();

    let (rows, classes) = (4, 3);
    let logits: Vec<f64> = fill(rows * classes, 61).iter().map(|&v| v as f64).collect();
    let mut targets = vec![0.0f64; rows * classes];
    for r in 0..rows {
        targets[r * classes + (r % classes)] = 1.0;
    }

    // Mean categorical cross-entropy, in f64, with no GPU involved. The kernel
    // averages over rows, so the loss it is the gradient of must too.
    let loss = |z: &[f64]| -> f64 {
        let mut total = 0.0;
        for r in 0..rows {
            let row = &z[r * classes..(r + 1) * classes];
            let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let denom: f64 = row.iter().map(|v| (v - max).exp()).sum();
            for c in 0..classes {
                if targets[r * classes + c] != 0.0 {
                    total -= (row[c] - max) - denom.ln();
                }
            }
        }
        total / rows as f64
    };

    let gl = ctx
        .upload(
            &logits.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
            rows,
            classes,
        )
        .expect("upload");
    let gt = ctx
        .upload(
            &targets.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
            rows,
            classes,
        )
        .expect("upload");
    let analytic = ctx
        .read(&ctx.softmax_xent_grad_resident(&gl, &gt).expect("grad"))
        .expect("read");

    let h = 1e-5;
    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    for i in 0..rows * classes {
        let mut plus = logits.clone();
        let mut minus = logits.clone();
        plus[i] += h;
        minus[i] -= h;

        let numerical = (loss(&plus) - loss(&minus)) / (2.0 * h);
        let error = (analytic[i] as f64 - numerical).abs();
        if error > worst {
            worst = error;
            worst_at = i;
        }
    }

    // f32 kernel against an f64 difference, so the tolerance is the kernel's
    // precision rather than the difference's.
    assert!(
        worst <= 2e-5,
        "logit {worst_at}: kernel {} against a central difference of the loss, \
         worst disagreement {worst:.3e}. The fused gradient is not the derivative \
         of the cross-entropy it claims to fuse with.",
        analytic[worst_at]
    );

    let magnitude = analytic.iter().fold(0.0f32, |m, g| m.max(g.abs()));
    assert!(
        magnitude > 1e-6,
        "the gradient is {magnitude:.3e}, indistinguishable from zero"
    );
}

/// The softmax kernel must compute softmax, not merely something shaped like it.
///
/// The three tests above pin properties: the rows sum to one and are positive,
/// the result is invariant to a constant shift, and large logits do not produce
/// NaN. Every one of those is satisfied by a family of functions, not by softmax
/// alone — `exp(2x) / sum(exp(2x))` is a probability distribution, is
/// shift-invariant, and is finite.
///
/// That is not hypothetical. Injecting exactly that defect to demonstrate the
/// cross-entropy gradient check left all three passing, which makes this the same
/// gap one level down: the forward kernel is described by its properties and
/// compared against nothing.
///
/// So this compares it against softmax computed host-side in f64. The reference
/// subtracts the row maximum for the same reason the kernel does, which is a
/// shared technique rather than a shared implementation — the arithmetic that
/// could be wrong is not shared.
///
/// Confirmed against that defect:
///
/// ```text
/// softmax_rows_are_probability_distributions      ok
/// softmax_is_invariant_to_a_constant_shift        ok
/// large_logits_do_not_produce_nan_in_softmax      ok
/// softmax_matches_a_host_reference            FAILED
/// ```
#[test]
#[cfg_attr(not(feature = "gpu"), ignore = "needs a GPU adapter: --features gpu")]
fn softmax_matches_a_host_reference() {
    let ctx = require_context();

    let (rows, classes) = (6, 4);
    let logits: Vec<f64> = fill(rows * classes, 71).iter().map(|&v| v as f64).collect();

    let uploaded = ctx
        .upload(
            &logits.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
            rows,
            classes,
        )
        .expect("upload");
    let kernel = ctx
        .read(&ctx.softmax_resident(&uploaded).expect("softmax"))
        .expect("read");

    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    for r in 0..rows {
        let row = &logits[r * classes..(r + 1) * classes];
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let denom: f64 = row.iter().map(|v| (v - max).exp()).sum();

        for c in 0..classes {
            let want = (row[c] - max).exp() / denom;
            let i = r * classes + c;
            let error = (kernel[i] as f64 - want).abs();
            if error > worst {
                worst = error;
                worst_at = i;
            }
        }
    }

    assert!(
        worst <= 2e-7,
        "index {worst_at}: kernel {} against an f64 host reference, worst \
         disagreement {worst:.3e}. The kernel produces a distribution that is \
         not softmax.",
        kernel[worst_at]
    );
}
