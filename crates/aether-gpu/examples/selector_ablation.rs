//! Does the topological selector do any work, or is it sparsity in a costume?
//!
//! Run:
//!   cargo run -p aether-gpu --example selector_ablation --release
//!
//! Every claim this repository makes about topology-derived attention rests on
//! one question that a speedup number cannot answer. A sparse schedule is faster
//! than dense because it does less work — a schedule choosing blocks at random
//! is faster too, and a model trained on either still converges. Speed says
//! nothing about whether the *selection* is informed.
//!
//! What answers it is recovered attention mass at an identical budget, bracketed
//! by two baselines:
//!
//!   - **random** at the same per-row budget. The topological schedule matching
//!     this means the mechanism contributes nothing, and whatever gain the
//!     method shows is sparsity acting as a regulariser.
//!   - **oracle** at the same budget, selecting the blocks that genuinely hold
//!     the most mass. Unimplementable in production — it reads the dense scores
//!     the schedule exists to avoid — but exactly the ceiling, because recovered
//!     mass is additive over key blocks.
//!
//! The reported position is `(topological - random) / (oracle - random)`: the
//! fraction of the achievable gain the selector actually captures. Zero means
//! random would have done as well. One means it found what the oracle found.
//!
//! Budget is matched per row rather than on average. A schedule spending its
//! blocks unevenly could otherwise win by spending more where they matter, which
//! is a different mechanism from the one under test.
//!
//! The GPU column exists to confirm the port computes the same attention the
//! ablation is reasoning about. It is a correctness check, not a timing: this
//! machine cannot produce a trustworthy magnitude, for reasons recorded in
//! FEATURES.md.

use aether_core::scheduled::{
    block_mass_recovered, block_salience, dense_causal_block_schedule, oracle_block_schedule,
    random_block_schedule, schedule_budget, scheduled_attention, topology_block_schedule,
    BlockSchedule, TopologyScheduleConfig,
};
use aether_gpu::GpuContext;

/// Correlated keys, so the sequence has structure a selector could find.
///
/// Independent noise is the wrong fixture for this measurement. With unstructured
/// keys every block holds roughly equal mass, the oracle has nothing to find, and
/// the gap it defines collapses — which would make the selector look neither good
/// nor bad, and would say more about the fixture than the method. Drifting the
/// mean produces blocks that genuinely differ in how much attention they draw.
fn drifting_keys(seq: usize, head_dim: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut out = Vec::with_capacity(seq * head_dim);
    for row in 0..seq {
        // A slow sweep across the sequence, so nearby rows resemble each other
        // and distant ones do not.
        let phase = row as f64 / seq as f64;
        for d in 0..head_dim {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            let drift = if d % 2 == 0 { phase } else { -phase };
            out.push(noise * 0.4 + drift);
        }
    }
    out
}

/// Independent keys with no positional structure.
///
/// The control for [`drifting_keys`]. Drift makes position a strong predictor of
/// attention, which rewards the local window and could on its own explain a
/// selector that looks for distant structure scoring badly. Without it, position
/// carries no information and anything left is the salience signal.
fn iid_keys(seq: usize, head_dim: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..seq * head_dim)
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

/// Worst absolute disagreement between the GPU kernel and the CPU kernel.
fn gpu_agreement(
    ctx: &GpuContext,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    seq: usize,
    head_dim: usize,
    schedule: &BlockSchedule,
    block_size: usize,
) -> f64 {
    let gpu = ctx
        .scheduled_attention(
            &to_f32(q),
            &to_f32(k),
            &to_f32(v),
            seq,
            head_dim,
            schedule,
            block_size,
        )
        .expect("dispatch");
    let cpu =
        scheduled_attention(q, k, v, seq, head_dim, schedule, block_size).expect("cpu kernel");

    gpu.iter()
        .zip(&cpu)
        .map(|(&g, &c)| (g as f64 - c).abs())
        .fold(0.0f64, f64::max)
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("No GPU adapter: {e}");
            std::process::exit(1);
        }
    };

    let head_dim = 32;
    let block_size = 8;
    // Enough random draws that the baseline is a distribution rather than one
    // sample. A selector credited for beating a single unlucky draw has been
    // credited for nothing.
    let draws = 32;

    let info = ctx.adapter_info();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Topological selector against same-budget random and oracle");
    println!("  adapter {}  |  {}", info.name, info.backend);
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
    println!("  recovered attention mass, per-row budget matched exactly");
    println!();
    println!(
        "  {:>5}  {:>7}  {:>9}  {:>9}  {:>9}  {:>9}  {:>10}",
        "seq", "density", "random", "topology", "oracle", "position", "gpu vs cpu"
    );
    println!(
        "  {:->5}  {:->7}  {:->9}  {:->9}  {:->9}  {:->9}  {:->10}",
        "", "", "", "", "", "", ""
    );

    for seq in [64usize, 128, 256, 512] {
        let q = drifting_keys(seq, head_dim, 1);
        let k = drifting_keys(seq, head_dim, 2);
        let v = drifting_keys(seq, head_dim, 3);

        let config = TopologyScheduleConfig {
            block_size,
            local_radius_blocks: 1,
            sink_blocks: 1,
            topk_topology_blocks: 2,
        };
        let topological = topology_block_schedule(&k, seq, head_dim, config).expect("valid config");
        let budget = schedule_budget(&topological);

        let dense = dense_causal_block_schedule(seq / block_size);
        let scheduled_blocks: usize = budget.iter().sum();
        let dense_blocks: usize = schedule_budget(&dense).iter().sum();
        let density = scheduled_blocks as f64 / dense_blocks as f64;

        let mass = |s: &BlockSchedule| {
            block_mass_recovered(s, &q, &k, seq, head_dim, block_size).expect("valid shapes")
        };

        let topo_mass = mass(&topological);
        let oracle = oracle_block_schedule(&q, &k, seq, head_dim, block_size, &budget)
            .expect("valid budget");
        let oracle_mass = mass(&oracle);

        let random_mass = (0..draws)
            .map(|d| mass(&random_block_schedule(&budget, 900 + d as u64).expect("valid budget")))
            .sum::<f64>()
            / draws as f64;

        // The achievable gain is what the oracle found beyond chance. If it is
        // vanishing there is nothing to capture and a position is meaningless,
        // so report that rather than a ratio of two small numbers.
        let headroom = oracle_mass - random_mass;
        let position = if headroom > 1e-9 {
            format!("{:.1}%", 100.0 * (topo_mass - random_mass) / headroom)
        } else {
            "n/a".to_string()
        };

        let worst = gpu_agreement(&ctx, &q, &k, &v, seq, head_dim, &topological, block_size);

        println!(
            "  {seq:>5}  {:>6.1}%  {random_mass:>9.4}  {topo_mass:>9.4}  \
             {oracle_mass:>9.4}  {position:>9}  {worst:>10.2e}",
            100.0 * density
        );
    }

    // The table above shows the position collapsing as the sequence lengthens.
    // There are two explanations and they call for opposite responses: either
    // the salience signal degrades with sequence length, which would be a defect
    // in the method, or the configuration holds `topk_topology_blocks` at 2
    // while the number of candidate blocks grows, so the topology-derived share
    // of the budget shrinks toward nothing and the schedule becomes mostly local
    // window and sink.
    //
    // Those are separable. Holding the sequence fixed and varying only the
    // topology allowance changes the second and not the first.
    println!();
    println!("───────────────────────────────────────────────────────────────────────");
    println!("  seq = 512, varying only the topology allowance");
    println!();
    println!(
        "  {:>8}  {:>5}  {:>7}  {:>9}  {:>9}  {:>9}  {:>9}",
        "keys", "top-k", "density", "random", "topology", "oracle", "position"
    );
    println!(
        "  {:->8}  {:->5}  {:->7}  {:->9}  {:->9}  {:->9}  {:->9}",
        "", "", "", "", "", "", ""
    );

    let seq = 512;
    let dense_blocks: usize = schedule_budget(&dense_causal_block_schedule(seq / block_size))
        .iter()
        .sum();

    for (fixture, q, k) in [
        (
            "drifting",
            drifting_keys(seq, head_dim, 1),
            drifting_keys(seq, head_dim, 2),
        ),
        // The control. Drifting keys make position a strong predictor of
        // attention, which favours the local window and could by itself explain
        // a selector that looks for distant structure doing badly. Independent
        // keys remove that: any residual effect is the salience signal rather
        // than the fixture rewarding locality.
        (
            "iid",
            iid_keys(seq, head_dim, 1),
            iid_keys(seq, head_dim, 2),
        ),
    ] {
        for topk in [2usize, 4, 8, 16, 32] {
            let config = TopologyScheduleConfig {
                block_size,
                local_radius_blocks: 1,
                sink_blocks: 1,
                topk_topology_blocks: topk,
            };
            let topological =
                topology_block_schedule(&k, seq, head_dim, config).expect("valid config");
            let budget = schedule_budget(&topological);
            let density = budget.iter().sum::<usize>() as f64 / dense_blocks as f64;

            let mass = |s: &BlockSchedule| {
                block_mass_recovered(s, &q, &k, seq, head_dim, block_size).expect("valid shapes")
            };
            let topo_mass = mass(&topological);
            let oracle_mass = mass(
                &oracle_block_schedule(&q, &k, seq, head_dim, block_size, &budget)
                    .expect("valid budget"),
            );
            let random_mass = (0..draws)
                .map(|d| {
                    mass(&random_block_schedule(&budget, 900 + d as u64).expect("valid budget"))
                })
                .sum::<f64>()
                / draws as f64;

            let headroom = oracle_mass - random_mass;
            let position = if headroom > 1e-9 {
                format!("{:.1}%", 100.0 * (topo_mass - random_mass) / headroom)
            } else {
                "n/a".to_string()
            };

            println!(
                "  {fixture:>8}  {topk:>5}  {:>6.1}%  {random_mass:>9.4}  {topo_mass:>9.4}  \
             {oracle_mass:>9.4}  {position:>9}",
                100.0 * density
            );
        }
        println!();
    }

    // Both fixtures put the selector below random, which needs an explanation
    // rather than a louder measurement.
    //
    // `block_salience` scores a block by its H0 death time under single-linkage
    // merging: how long its component survives before joining another. A high
    // score means an *isolated* block, far in feature space from the rest. But
    // attention mass concentrates where the key resembles the query, and a block
    // unlike everything else is unlike the typical query too. Ranking by
    // isolation and ranking by attention mass are then anti-correlated by
    // construction, and taking the top of one is close to taking the bottom of
    // the other.
    //
    // That predicts something specific and cheap to check: selecting the
    // *lowest*-salience blocks at the same budget should beat both the current
    // selector and random. If it does, the signal is real and its sign is
    // reversed. If it merely matches random, the salience carries no usable
    // information about attention and the sign was never the issue.
    println!();
    println!("───────────────────────────────────────────────────────────────────────");
    println!("  seq = 512, drifting keys: ranking by salience against ranking by");
    println!("  inverted salience, at an identical per-row budget");
    println!();
    println!(
        "  {:>5}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}",
        "top-k", "random", "highest", "lowest", "oracle", "lowest pos"
    );
    println!(
        "  {:->5}  {:->9}  {:->9}  {:->9}  {:->9}  {:->9}",
        "", "", "", "", "", ""
    );

    let q = drifting_keys(seq, head_dim, 1);
    let k = drifting_keys(seq, head_dim, 2);
    let salience = block_salience(&k, seq, head_dim, block_size).expect("valid blocking");

    for topk in [2usize, 4, 8, 16] {
        let config = TopologyScheduleConfig {
            block_size,
            local_radius_blocks: 1,
            sink_blocks: 1,
            topk_topology_blocks: topk,
        };
        let highest = topology_block_schedule(&k, seq, head_dim, config).expect("valid config");
        let budget = schedule_budget(&highest);

        // The same schedule shape with the salience ranking reversed: sinks and
        // the local window are kept, and the discretionary slots go to the
        // least isolated causal blocks instead of the most.
        let num_blocks = seq / block_size;
        let rows: Vec<Vec<usize>> = (0..num_blocks)
            .map(|q_block| {
                let mut row: Vec<usize> = (0..config.sink_blocks.min(q_block + 1)).collect();
                row.extend(q_block.saturating_sub(config.local_radius_blocks)..=q_block);
                row.sort_unstable();
                row.dedup();

                let mut candidates: Vec<usize> =
                    (0..=q_block).filter(|c| !row.contains(c)).collect();
                candidates.sort_by(|&a, &b| {
                    salience[a]
                        .partial_cmp(&salience[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(&b))
                });
                let want = budget[q_block].saturating_sub(row.len());
                row.extend(candidates.iter().take(want));
                row.sort_unstable();
                row
            })
            .collect();
        let lowest = BlockSchedule::from_rows(&rows).expect("valid rows");

        let mass = |s: &BlockSchedule| {
            block_mass_recovered(s, &q, &k, seq, head_dim, block_size).expect("valid shapes")
        };
        let random_mass = (0..draws)
            .map(|d| mass(&random_block_schedule(&budget, 900 + d as u64).expect("valid budget")))
            .sum::<f64>()
            / draws as f64;
        let oracle_mass = mass(
            &oracle_block_schedule(&q, &k, seq, head_dim, block_size, &budget)
                .expect("valid budget"),
        );
        let high_mass = mass(&highest);
        let low_mass = mass(&lowest);

        let headroom = oracle_mass - random_mass;
        let low_position = if headroom > 1e-9 {
            format!("{:.1}%", 100.0 * (low_mass - random_mass) / headroom)
        } else {
            "n/a".to_string()
        };

        println!(
            "  {topk:>5}  {random_mass:>9.4}  {high_mass:>9.4}  {low_mass:>9.4}  \
             {oracle_mass:>9.4}  {low_position:>9}"
        );
    }

    println!();
    println!("  A position that recovers as top-k rises means the collapse above is");
    println!("  the fixed allowance being diluted by sequence length, not the");
    println!("  salience signal failing. A position that stays flat means the");
    println!("  signal itself degrades, and raising the budget cannot fix it.");

    println!();
    println!("───────────────────────────────────────────────────────────────────────");
    println!("  position = (topology - random) / (oracle - random)");
    println!();
    println!("    0%   random selection would have recovered as much mass, so the");
    println!("         topology contributes nothing and the gain is sparsity alone");
    println!("  100%   the selector found what the oracle found");
    println!();
    println!("  'gpu vs cpu' is the worst absolute disagreement between the WGSL");
    println!("  kernel and the f64 CPU kernel on the same schedule. It is a");
    println!("  correctness check on the port, not a timing.");
    println!();
    println!("  random is the mean of {draws} draws at the identical per-row budget.");
    println!("  Reporting one draw would credit the selector for beating a sample.");
    println!("═══════════════════════════════════════════════════════════════════════");
}
