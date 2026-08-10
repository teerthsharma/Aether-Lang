//! Correctness contracts for the GPU backend.
//!
//! The ordering is deliberate. Dense parity against a CPU reference is the
//! highest-value assertion available: it catches transposed strides, a wrong
//! index base, a mis-sized dispatch grid, and a bad uniform layout in a single
//! comparison, before anything interesting is attempted. Everything below it
//! catches a narrower class of bug.
//!
//! Every test skips rather than fails when no adapter exists, so the suite is
//! meaningful on a developer box with a GPU and honest on a headless runner
//! without one. A skipped test prints why. It never reports success for work
//! that did not happen.

use aether_gpu::{cpu_matmul, GpuContext};

/// f32 accumulation over k terms diverges from a separately-ordered f32
/// accumulation. The bound scales with k, so the tolerance does too rather
/// than being a single constant that is loose at k=4 and wrong at k=512.
fn tolerance(k: usize) -> f32 {
    1e-5 * (k as f32).sqrt().max(1.0)
}

fn context() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIP: no usable GPU adapter ({e})");
            None
        }
    }
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
fn the_gpu_reports_which_adapter_it_is_using() {
    let Some(ctx) = context() else { return };
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
fn the_selected_adapter_is_real_hardware_not_a_software_rasterizer() {
    let Some(ctx) = context() else { return };
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
fn gpu_matmul_matches_the_cpu_reference() {
    let Some(ctx) = context() else { return };

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
fn shapes_around_the_workgroup_boundary_are_handled() {
    let Some(ctx) = context() else { return };

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
fn a_rectangular_product_is_not_transposed() {
    let Some(ctx) = context() else { return };

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
fn repeated_dispatches_are_bitwise_identical() {
    let Some(ctx) = context() else { return };

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
fn bias_is_broadcast_across_rows_not_down_columns() {
    let Some(ctx) = context() else { return };

    let a = vec![
        1.0, 2.0, 3.0, //
        4.0, 5.0, 6.0,
    ];
    let bias = vec![10.0, 20.0, 30.0];

    let gpu = ctx.add_bias(&a, &bias, 2, 3).expect("bias dispatch");
    assert_eq!(gpu, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn relu_clamps_negatives_and_leaves_positives_alone() {
    let Some(ctx) = context() else { return };

    let a = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
    let gpu = ctx.relu(&a).expect("relu dispatch");
    assert_eq!(gpu, vec![0.0, 0.0, 0.0, 0.5, 2.0]);
}

/// The boundary at exactly zero is a convention, and forward and backward must
/// agree on it. A backward pass that treats 0 as active where the forward
/// treats it as inactive trains to a different optimum, and the loss curve
/// looks fine while it happens.
#[test]
fn relu_backward_is_zero_at_exactly_zero() {
    let Some(ctx) = context() else { return };

    let pre = vec![-1.0, 0.0, 1.0];
    let grad = vec![5.0, 5.0, 5.0];

    let gpu = ctx
        .relu_backward(&pre, &grad)
        .expect("relu_backward dispatch");
    assert_eq!(gpu, vec![0.0, 0.0, 5.0]);
}

#[test]
fn mismatched_shapes_are_rejected_rather_than_dispatched() {
    let Some(ctx) = context() else { return };

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
