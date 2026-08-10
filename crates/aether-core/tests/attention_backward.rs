//! Contracts for reverse mode through the scheduled-attention kernel.
//!
//! A wrong backward pass is the least visible defect in this whole codebase. The
//! forward stays correct, the loss still falls, and the model converges to a
//! plausible worse optimum — there is no crash, no NaN, and no obviously bad
//! number to notice. Nothing about a training curve distinguishes it from a
//! model that is simply learning something hard.
//!
//! So the load-bearing test here is finite differences. Every analytic gradient
//! is checked against a central difference of the forward kernel it claims to
//! differentiate, which is the only check that does not share an assumption with
//! the code under test: the reference is the forward pass itself, already pinned
//! against a quadratic reference elsewhere.
//!
//! The structural tests around it catch what finite differences cannot afford to
//! sample. A gradient that leaks into a key the schedule excluded would need the
//! difference check to happen to perturb that exact coordinate; asserting the
//! sparsity directly covers every excluded position at once.

use aether_core::scheduled::{
    dense_causal_block_schedule, scheduled_attention, scheduled_attention_backward, BlockSchedule,
};

fn fill(n: usize, seed: u64) -> Vec<f64> {
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

struct Case {
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    d_out: Vec<f64>,
    seq: usize,
    head_dim: usize,
    block_size: usize,
}

impl Case {
    fn new(seq: usize, head_dim: usize, block_size: usize, seed: u64) -> Self {
        Self {
            q: fill(seq * head_dim, seed),
            k: fill(seq * head_dim, seed + 1),
            v: fill(seq * head_dim, seed + 2),
            // An arbitrary upstream gradient. A vector of ones would make several
            // terms cancel and would hide a sign error in the rank-one softmax
            // correction, which is exactly the term hardest to get right.
            d_out: fill(seq * head_dim, seed + 3),
            seq,
            head_dim,
            block_size,
        }
    }

    /// The scalar the gradients are gradients of: `<attention(q, k, v), d_out>`.
    ///
    /// Differentiating an inner product with a fixed cotangent is what makes a
    /// single finite difference comparable to a single entry of the analytic
    /// gradient, without needing to form a Jacobian.
    fn loss(&self, q: &[f64], k: &[f64], v: &[f64], schedule: &BlockSchedule) -> f64 {
        let out = scheduled_attention(
            q,
            k,
            v,
            self.seq,
            self.head_dim,
            schedule,
            self.block_size,
        )
        .expect("valid launch");
        out.iter().zip(&self.d_out).map(|(a, b)| a * b).sum()
    }

    fn analytic(&self, schedule: &BlockSchedule) -> aether_core::scheduled::AttentionGradients {
        scheduled_attention_backward(
            &self.q,
            &self.k,
            &self.v,
            self.seq,
            self.head_dim,
            schedule,
            self.block_size,
            &self.d_out,
        )
        .expect("valid launch")
    }
}

/// Step size for the central difference.
///
/// Central differences carry truncation error O(h^2) and rounding error
/// O(eps/h), which in f64 balance near h = eps^(1/3) ~= 6e-6. 1e-5 sits in that
/// basin and leaves the observed disagreement several orders below the tolerance
/// asserted, so neither term is what the test is measuring.
const H: f64 = 1e-5;

/// Tolerance on |analytic - numerical|.
///
/// Set from the observed error rather than chosen: across every fixture in this
/// file the worst disagreement is below 1e-9, and a defect in any of the three
/// gradients moves entries by O(0.1) or more. Two orders of headroom above what
/// is observed, seven below what a bug would produce.
const TOL: f64 = 1e-7;

fn check_operand(case: &Case, schedule: &BlockSchedule, which: Operand) {
    let analytic = case.analytic(schedule);
    let (grad, base) = match which {
        Operand::Q => (&analytic.dq, &case.q),
        Operand::K => (&analytic.dk, &case.k),
        Operand::V => (&analytic.dv, &case.v),
    };

    let mut worst = 0.0f64;
    let mut worst_at = 0usize;

    for i in 0..base.len() {
        let mut plus = base.clone();
        let mut minus = base.clone();
        plus[i] += H;
        minus[i] -= H;

        let (lp, lm) = match which {
            Operand::Q => (
                case.loss(&plus, &case.k, &case.v, schedule),
                case.loss(&minus, &case.k, &case.v, schedule),
            ),
            Operand::K => (
                case.loss(&case.q, &plus, &case.v, schedule),
                case.loss(&case.q, &minus, &case.v, schedule),
            ),
            Operand::V => (
                case.loss(&case.q, &case.k, &plus, schedule),
                case.loss(&case.q, &case.k, &minus, schedule),
            ),
        };

        let numerical = (lp - lm) / (2.0 * H);
        let error = (grad[i] - numerical).abs();
        if error > worst {
            worst = error;
            worst_at = i;
        }
    }

    assert!(
        worst <= TOL,
        "{which:?}: worst |analytic - numerical| = {worst:.3e} at index \
         {worst_at}, above {TOL:.0e}. analytic {}, numerical differs by that \
         much, so the backward pass does not differentiate the forward pass.",
        grad[worst_at]
    );
}

#[derive(Debug, Clone, Copy)]
enum Operand {
    Q,
    K,
    V,
}

/// Every gradient must match a central difference of the forward kernel.
///
/// The dense causal schedule first, because it exercises the whole backward path
/// with no sparsity involved. A backward that is wrong here is wrong everywhere,
/// and no sparse fixture would localise it better.
#[test]
fn gradients_match_finite_differences_on_a_dense_schedule() {
    let case = Case::new(16, 4, 4, 3);
    let schedule = dense_causal_block_schedule(case.seq / case.block_size);

    for operand in [Operand::Q, Operand::K, Operand::V] {
        check_operand(&case, &schedule, operand);
    }
}

/// The same, on a schedule where rows see different numbers of blocks.
///
/// Distinct from the dense case: with every block present, an error in which
/// columns a row accumulates into cannot show, because the answer is "all of
/// them". Uneven rows are where a mis-walked CSR appears.
#[test]
fn gradients_match_finite_differences_on_a_sparse_schedule() {
    let case = Case::new(24, 4, 4, 7);

    // Sink plus a one-block local window: rows of length 1, 2, 3, 3, 3, 3.
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

    for operand in [Operand::Q, Operand::K, Operand::V] {
        check_operand(&case, &schedule, operand);
    }
}

/// A key the schedule never selects must receive exactly zero gradient.
///
/// Structural, and asserted directly rather than left to the difference check to
/// stumble on. A leak into an excluded position would only show there if that
/// exact coordinate happened to be perturbed, whereas this covers every excluded
/// position at once.
///
/// Exact zero rather than a tolerance: nothing is ever accumulated into these
/// entries, so anything other than zero is a write that should not have happened
/// and not a rounding artefact.
#[test]
fn keys_outside_the_schedule_receive_no_gradient() {
    let case = Case::new(24, 4, 4, 11);

    // Only the diagonal block, so every row sees exactly its own block and the
    // excluded set is large and easy to enumerate.
    let rows: Vec<Vec<usize>> = (0..6usize).map(|q| vec![q]).collect();
    let schedule = BlockSchedule::from_rows(&rows).expect("valid rows");
    let grads = case.analytic(&schedule);

    for col in 0..case.seq {
        let k_block = col / case.block_size;

        // A column is reachable if any query block scheduling its block holds a
        // row at or after it, since the causal mask drops the rest.
        let reachable = (0..6usize).any(|q_block| {
            schedule.row(q_block).contains(&k_block)
                && q_block * case.block_size + case.block_size - 1 >= col
        });
        if reachable {
            continue;
        }

        for d in 0..case.head_dim {
            let i = col * case.head_dim + d;
            assert_eq!(
                grads.dk[i], 0.0,
                "column {col} is scheduled by no query row but dk[{i}] is {}",
                grads.dk[i]
            );
            assert_eq!(
                grads.dv[i], 0.0,
                "column {col} is scheduled by no query row but dv[{i}] is {}",
                grads.dv[i]
            );
        }
    }
}

/// The value gradient must equal the attention weights transposed.
///
/// `dv_j = sum_i p_ij dOut_i`, exactly and with no softmax correction involved,
/// because the output is linear in `v`. Feeding a one-hot cotangent isolates a
/// single row, so `dv` becomes that row's attention weights spread across the
/// head dimension — a closed form the implementation is checked against rather
/// than another numerical estimate.
///
/// It also pins something the difference tests cannot: that the weights used in
/// the backward are the same weights the forward applied, rather than a
/// separately normalised set that happens to sum to one.
#[test]
fn the_value_gradient_reproduces_the_attention_weights() {
    let seq = 16;
    let head_dim = 4;
    let block_size = 4;
    let mut case = Case::new(seq, head_dim, block_size, 13);
    let schedule = dense_causal_block_schedule(seq / block_size);

    // One-hot on the final row, first component.
    let target_row = seq - 1;
    case.d_out = vec![0.0; seq * head_dim];
    case.d_out[target_row * head_dim] = 1.0;

    let grads = case.analytic(&schedule);

    // With this cotangent, dv[j][0] is exactly p_{target_row, j} and every other
    // component is zero.
    let mut total = 0.0;
    for col in 0..seq {
        let weight = grads.dv[col * head_dim];
        assert!(
            weight >= -1e-15,
            "column {col} carries a negative attention weight {weight}"
        );
        total += weight;

        for d in 1..head_dim {
            assert_eq!(
                grads.dv[col * head_dim + d],
                0.0,
                "component {d} of column {col} is non-zero under a one-hot \
                 cotangent on component 0"
            );
        }
    }

    assert!(
        (total - 1.0).abs() < 1e-12,
        "the recovered attention weights sum to {total}, not 1; the backward is \
         using a differently normalised distribution from the forward"
    );
}

/// Shape violations are rejected rather than producing a wrong-length gradient.
#[test]
fn a_mismatched_cotangent_is_rejected() {
    let case = Case::new(16, 4, 4, 17);
    let schedule = dense_causal_block_schedule(case.seq / case.block_size);

    let short = vec![0.0; case.seq * case.head_dim - 1];
    assert!(
        scheduled_attention_backward(
            &case.q,
            &case.k,
            &case.v,
            case.seq,
            case.head_dim,
            &schedule,
            case.block_size,
            &short,
        )
        .is_err(),
        "a cotangent shorter than the output was accepted"
    );
}
