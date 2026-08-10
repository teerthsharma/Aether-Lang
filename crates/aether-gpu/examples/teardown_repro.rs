//! Minimal reproducer for the teardown access violation.
//!
//! Run:
//!   cargo run -p aether-gpu --example teardown_repro --release [pattern]
//!
//! `gpu_bench` faults at process exit with STATUS_ACCESS_VIOLATION on roughly
//! one run in five, but takes about fifteen seconds per run, which makes the
//! thirty-runs-per-variant needed to compare rates impractically slow.
//!
//! This strips the benchmark to the pattern that correlates with the fault:
//! many large result tensors allocated and dropped without ever being read.
//! Patterns are selectable so the suspect can be isolated rather than assumed.
//!
//! Exit code is the finding. 0 is a clean run; the harness counts.

use aether_gpu::GpuContext;

fn fill(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 97) as f32 * 0.01).collect()
}

fn main() {
    let pattern = std::env::args().nth(1).unwrap_or_else(|| "drop".into());

    // Buffer edge length. The result matrix is n*n f32, so n=2048 is 16 MB and
    // n=512 is 1 MB. Parameterised because size dependence is one of the two
    // questions a wgpu maintainer would ask first, and it was unmeasured.
    let n: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);

    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no adapter: {e}");
            std::process::exit(2);
        }
    };

    // Printed so a run recorded from a harness carries its own backend. The
    // other question a maintainer asks first is whether this is
    // backend-specific, and `WGPU_BACKEND` selects it.
    if std::env::var("REPRO_QUIET").is_err() {
        let info = ctx.adapter_info();
        eprintln!(
            "adapter {} | {} | n={n} | pattern={pattern}",
            info.name, info.backend
        );
    }

    let pts = fill(n * 3);
    let g = ctx.upload(&pts, n, 3).expect("upload");

    match pattern.as_str() {
        // The suspect: allocate 16 MB results and drop them unread. Recorded
        // dispatches reference buffers that are freed before anything forces
        // the queue to finish with them.
        "drop" => {
            for _ in 0..10 {
                for _ in 0..10 {
                    let _ = ctx.pairwise_sqdist_resident(&g).expect("sqdist");
                }
            }
        }

        // Same allocation volume, but every result is read, which flushes and
        // polls to completion before the buffer is dropped.
        "read" => {
            for _ in 0..10 {
                for _ in 0..10 {
                    let m = ctx.pairwise_sqdist_resident(&g).expect("sqdist");
                    let _ = ctx.read(&m).expect("read");
                }
            }
        }

        // Same as "drop" but flushing after each batch, so work is submitted
        // even though nothing is read. Separates "never submitted" from
        // "submitted but the buffer went away".
        "flush" => {
            for _ in 0..10 {
                for _ in 0..10 {
                    let _ = ctx.pairwise_sqdist_resident(&g).expect("sqdist");
                }
                ctx.flush();
            }
        }

        // Allocation churn with no dispatches at all, to separate buffer
        // lifetime from recorded-command lifetime.
        "alloc" => {
            for _ in 0..100 {
                let _ = ctx.upload(&vec![0.0f32; n * n], 1, n * n).expect("upload");
            }
        }

        other => {
            eprintln!("unknown pattern: {other}");
            std::process::exit(2);
        }
    }

    println!("{pattern}: completed");
}
