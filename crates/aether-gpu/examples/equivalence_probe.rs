//! Does a mutation change what the kernels compute, bit for bit?
//!
//! Run:
//!   cargo run -p aether-gpu --example equivalence_probe --release --features gpu
//!
//! `mutants-mechanical.sh` flips every comparison operator in the shader and
//! reports which flips no suite catches. 52 of 90 survive, against 0 of 24 for
//! the hand-written set, and that number means nothing on its own: a flip that
//! does not change the computed result is an *equivalent mutant*, and no test can
//! catch it because there is nothing to catch. Counting one as a coverage hole
//! would be the same error as counting a skipped test as a passing one.
//!
//! Distinguishing the two needs evidence the suites cannot give. A suite reports
//! pass or fail against a tolerance, so "survives" means the outputs agreed to
//! within that tolerance rather than that they were identical. This prints a
//! checksum over the exact bits of each kernel's output, so running it on a clean
//! tree and again under a mutant answers the question directly: identical
//! checksums mean the mutation changed nothing and is equivalent, and different
//! checksums under a passing suite mean the tolerance absorbed a real difference,
//! which is a coverage hole with a measurement behind it.
//!
//! The inputs are fixed and the checksum is over raw `f32` bits rather than
//! rounded values, because the entire question is whether anything changed at
//! all.
//!
//! # Every kernel is dispatched, and that is load-bearing
//!
//! A mutation in a kernel this never runs produces an identical checksum because
//! nothing ran it, which is indistinguishable in the output from a mutation that
//! ran and changed nothing. An earlier version dispatched five kernels of the
//! twenty and could therefore only classify a survivor that happened to land in
//! one of them — reading its verdict for any other site would have been the same
//! error the rest of this crate documents, taking silence for evidence.
//!
//! All twenty are now covered, and `assert_all_kernels_covered` fails the run if
//! the shader gains one this does not reach. Without that check the gap returns
//! silently the first time a kernel is added, and every later verdict is quietly
//! narrower than it claims.
//!
//! Two are reached only indirectly and are named here so the coverage claim can
//! be audited: `matmul_tiled` runs via `matmul_resident` (the non-resident
//! `matmul` uses the untiled kernel), and `adam_moments` runs as the first of the
//! two dispatches inside `adam_update_resident`.

use aether_core::scheduled::dense_causal_block_schedule;
use aether_gpu::{AttentionPath, GpuContext};

/// FNV-1a over the raw bits of every output element.
///
/// Bits, not values: two `f32`s that print identically can differ in the last
/// place, and a probe for "did anything change" that rounds first would report
/// equivalence for the differences most worth finding. NaN is not special-cased
/// for the same reason — its payload is part of what changed.
fn checksum(values: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in values {
        for byte in v.to_bits().to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        })
        .collect()
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("No GPU adapter: {e}");
            std::process::exit(1);
        }
    };

    println!(
        "kernel output checksums, adapter {}",
        ctx.adapter_info().name
    );
    println!();

    let mut all: u64 = 0xcbf2_9ce4_8422_2325;
    let mut report = |name: &str, out: &[f32]| {
        let c = checksum(out);
        println!("  {name:<28} {c:#018x}  {} elements", out.len());
        all ^= c;
        all = all.wrapping_mul(0x0000_0100_0000_01b3);
    };

    // Shapes are deliberately not multiples of the 16-wide workgroup. A kernel
    // whose bounds guard is wrong is only wrong for the threads past the end,
    // and a shape that divides evenly has none of them — the guard would be
    // unreachable and every flip of it equivalent for that reason alone, which
    // is a property of the input rather than of the code.
    let (m, k, n) = (37usize, 23usize, 19usize);
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);

    report("matmul", &ctx.matmul(&a, &b, m, k, n).expect("matmul"));

    let bias = fill(n, 3);
    let mm = ctx.matmul(&a, &b, m, k, n).expect("matmul");
    report(
        "add_bias",
        &ctx.add_bias(&mm, &bias, m, n).expect("add_bias"),
    );

    // Spans zero exactly, so the `x >= 0` / `x > 0` branch boundary is reached
    // rather than assumed to be.
    let kinked: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
    report("relu", &ctx.relu(&kinked).expect("relu"));

    let grad = fill(kinked.len(), 4);
    report(
        "relu_backward",
        &ctx.relu_backward(&kinked, &grad).expect("relu_backward"),
    );

    // matmul_tiled, which the non-resident matmul above does not touch.
    let ta = ctx.upload(&a, m, k).expect("upload a");
    let tb = ctx.upload(&b, k, n).expect("upload b");
    let tiled = ctx.matmul_resident(&ta, &tb).expect("matmul_resident");
    report("matmul_tiled", &ctx.read(&tiled).expect("read"));

    // add_broadcast_row, reached through the resident bias add.
    let tbias = ctx.upload(&bias, 1, n).expect("upload bias");
    let biased = ctx
        .add_bias_resident(&tiled, &tbias)
        .expect("add_bias_resident");
    report("add_broadcast_row", &ctx.read(&biased).expect("read"));

    let tk = ctx.upload(&kinked, 1, kinked.len()).expect("upload kinked");
    report(
        "sigmoid",
        &ctx.read(&ctx.sigmoid_resident(&tk).expect("sigmoid"))
            .expect("read"),
    );

    // Logits and one-hot targets share a shape, so the same pair drives both the
    // sigmoid and the softmax gradient kernels.
    let (bs, classes) = (7usize, 5usize);
    let logits = fill(bs * classes, 8);
    let tlogits = ctx.upload(&logits, bs, classes).expect("upload logits");

    let mut onehot = vec![0.0f32; bs * classes];
    for r in 0..bs {
        onehot[r * classes + (r % classes)] = 1.0;
    }
    let tonehot = ctx.upload(&onehot, bs, classes).expect("upload onehot");

    report(
        "softmax_rows",
        &ctx.read(&ctx.softmax_resident(&tlogits).expect("softmax"))
            .expect("read"),
    );
    report(
        "softmax_xent_grad",
        &ctx.read(
            &ctx.softmax_xent_grad_resident(&tlogits, &tonehot)
                .expect("xent grad"),
        )
        .expect("read"),
    );
    report(
        "sigmoid_bce_grad",
        &ctx.read(
            &ctx.sigmoid_bce_grad_resident(&tlogits, &tonehot)
                .expect("bce grad"),
        )
        .expect("read"),
    );

    report(
        "transpose",
        &ctx.read(&ctx.transpose_resident(&ta).expect("transpose"))
            .expect("read"),
    );
    report(
        "column_sums",
        &ctx.read(&ctx.column_sums_resident(&ta).expect("column_sums"))
            .expect("read"),
    );

    // Points are a small cloud, not the matmul operand, so the distance kernel
    // sees a shape it would actually be called with.
    let (pts, dim) = (11usize, 3usize);
    let cloud = ctx
        .upload(&fill(pts * dim, 9), pts, dim)
        .expect("upload pts");
    report(
        "pairwise_sqdist",
        &ctx.read(&ctx.pairwise_sqdist_resident(&cloud).expect("sqdist"))
            .expect("read"),
    );

    let param = ctx.upload(&fill(31, 10), 1, 31).expect("upload param");
    let grad31 = ctx.upload(&fill(31, 11), 1, 31).expect("upload grad");
    report(
        "sgd_update",
        &ctx.read(&ctx.sgd_update_resident(&param, &grad31, 0.05).expect("sgd"))
            .expect("read"),
    );

    // Two steps rather than one. Adam's bias correction divides by
    // `1 - beta^t`, so a defect in the step counter or the correction is
    // identical to a correct implementation at t=1 and diverges afterwards.
    let mut state = ctx.adam_state(&param).expect("adam_state");
    let once = ctx
        .adam_update_resident(&param, &grad31, &mut state, 0.05)
        .expect("adam 1");
    let twice = ctx
        .adam_update_resident(&once, &grad31, &mut state, 0.05)
        .expect("adam 2");
    report(
        "adam_moments + adam_update",
        &ctx.read(&twice).expect("read"),
    );

    let (seq, head_dim, block) = (24usize, 8usize, 8usize);
    let q = fill(seq * head_dim, 5);
    let kk = fill(seq * head_dim, 6);
    let v = fill(seq * head_dim, 7);
    let schedule = dense_causal_block_schedule(seq / block);
    report(
        "scheduled_attention",
        &ctx.scheduled_attention(&q, &kk, &v, seq, head_dim, &schedule, block)
            .expect("attention"),
    );

    // attention_row_stats, attention_dq, attention_dk, attention_dv. The
    // backward entry point takes f64 and narrows internally.
    let wide = |xs: &[f32]| xs.iter().map(|&x| f64::from(x)).collect::<Vec<f64>>();
    let d_out = fill(seq * head_dim, 12);
    let (grads, path) = ctx
        .scheduled_attention_backward_or_cpu(
            &wide(&q),
            &wide(&kk),
            &wide(&v),
            seq,
            head_dim,
            &schedule,
            block,
            &wide(&d_out),
        )
        .expect("attention backward");

    // The backward path falls back to CPU under conditions this shape should not
    // meet. If it ever does, the four backward kernels are not being measured and
    // the run must say so rather than print a checksum that describes the CPU.
    assert!(
        matches!(path, AttentionPath::Gpu),
        "attention backward ran on {path:?}, so the four backward kernels were \
         not dispatched and their sites cannot be classified from this run"
    );

    let narrow = |xs: &[f64]| xs.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    report("attention_dq", &narrow(&grads.dq));
    report("attention_dk", &narrow(&grads.dk));
    report("attention_dv", &narrow(&grads.dv));

    println!();
    println!("  {:<28} {all:#018x}", "COMBINED");

    assert_all_kernels_covered();

    println!();
    println!("Run on a clean tree, then again with a mutant applied. Identical");
    println!("combined checksums mean the mutation changed no output and is an");
    println!("equivalent mutant. A different checksum while the suites still pass");
    println!("means a tolerance absorbed a real difference.");
}

/// Fail if the shader declares a kernel this probe does not dispatch.
///
/// The probe's verdict is only as wide as its coverage, and a gap is invisible
/// in its output: an unreached kernel reports the same identical checksum as one
/// that ran and changed nothing. Reading the shader is what makes the coverage
/// claim checkable rather than a comment that was true when written.
///
/// `attention_row_stats` is dispatched inside the backward call and produces no
/// separately readable buffer, so it is listed as covered on the strength of the
/// three gradient checksums that depend on it.
fn assert_all_kernels_covered() {
    const COVERED: &[&str] = &[
        "matmul",
        "matmul_tiled",
        "pairwise_sqdist",
        "sigmoid",
        "sigmoid_bce_grad",
        "softmax_rows",
        "softmax_xent_grad",
        "transpose",
        "column_sums",
        "sgd_update",
        "adam_moments",
        "adam_update",
        "add_broadcast_row",
        "relu",
        "relu_backward",
        "scheduled_attention",
        "attention_row_stats",
        "attention_dq",
        "attention_dk",
        "attention_dv",
    ];

    // Only entry points, which means the `fn` on the line after a `@compute`
    // attribute. Collecting every `fn` would sweep in the shader's helper
    // functions and report them as uncovered kernels, which is a failure that
    // looks exactly like the one this is for.
    let src = include_str!("../src/shaders.wgsl");
    let mut declared = Vec::new();
    let mut previous_was_compute = false;

    for line in src.lines() {
        if previous_was_compute {
            if let Some(rest) = line.trim_start().strip_prefix("fn ") {
                if let Some(name) = rest.split('(').next() {
                    declared.push(name.trim());
                }
            }
        }
        previous_was_compute = line.trim_start().starts_with("@compute");
    }

    assert!(
        !declared.is_empty(),
        "no @compute entry points found in the shader, so this check passes by \
         examining nothing — the attribute's spelling or placement has changed"
    );

    let missing: Vec<&str> = declared
        .iter()
        .copied()
        .filter(|k| !COVERED.contains(k))
        .collect();

    assert!(
        missing.is_empty(),
        "the shader declares kernels this probe does not dispatch: {missing:?}. \
         Every verdict it prints would be narrower than it appears, because an \
         unreached kernel checksums identically to an unchanged one."
    );
}
