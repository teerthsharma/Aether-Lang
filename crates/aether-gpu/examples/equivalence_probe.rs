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
//! # What an identical checksum does not prove
//!
//! This runs five kernels of the twenty in the shader: `matmul`, `add_bias`,
//! `relu`, `relu_backward` and `scheduled_attention`. A mutation in a kernel it
//! never dispatches produces an identical checksum because nothing ran it, which
//! is indistinguishable here from a mutation that ran and changed nothing.
//!
//! So a verdict of "equivalent" is only valid for a site inside one of those five.
//! Reading it for a site in `softmax_rows`, `adam_update`, `transpose`,
//! `column_sums`, `sigmoid`, `pairwise_sqdist` or any backward kernel would be
//! the same class of error the rest of this crate documents: taking silence for
//! evidence. Extending the coverage is the fix; asserting the conclusion anyway
//! is not.

use aether_core::scheduled::dense_causal_block_schedule;
use aether_gpu::GpuContext;

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

    println!();
    println!("  {:<28} {all:#018x}", "COMBINED");
    println!();
    println!("Run on a clean tree, then again with a mutant applied. Identical");
    println!("combined checksums mean the mutation changed no output and is an");
    println!("equivalent mutant. A different checksum while the suites still pass");
    println!("means a tolerance absorbed a real difference.");
}
