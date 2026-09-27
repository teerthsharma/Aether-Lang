# Resolvent Operator

Module: `aether_core::resolvent`, in `crates/aether-core/src/resolvent.rs`. It implements one causal attention head with three switches. Three familiar operators are settings of this single operator: softmax attention, unnormalized-kernel attention, and the exact path product of a gated chain. The source's statements are proved in Lean 4. This module mirrors them numerically. Evidence: 17 tests in `tests/resolvent.rs`.

## Object

The head acts on a causal sequence of length \(n\). Its inputs are queries and keys \(q_i, k_j \in \mathbb{R}^d\), raw gate magnitudes \(u_k \in \mathbb{R}\), phases \(\theta_k \in \mathbb{R}\), and switches \((\beta, \mathrm{qk}, g)\). Here \(\mathrm{qk}\) is the scalar content scale on the logit, `qk` in the code. Real values \(V_j\) are read out. For \(0 \le j \le i < n\):

\[
\begin{aligned}
s_{ij} &= \langle q_i, k_j\rangle / \sqrt{d} && \text{scaled logit}\\
m_k &= \operatorname{clamp}\big(\operatorname{lerp}(1, u_k, g),\, 0,\, 1\big), \quad \theta'_k = g\,\theta_k && \text{blended gate}\\
a_k &= m_k\, e^{i\theta'_k} && \text{the gate}\\
G_{ij} &= \textstyle\prod_{k=j+1}^{i} a_k, \quad G_{ii} = 1 && \text{the path product}\\
R_{ij} &= \textstyle\prod_{k=j+1}^{i} m_k = \lvert G_{ij}\rvert && \text{its modulus}\\
Z_i &= \textstyle\sum_{j \le i} R_{ij}\, e^{\mathrm{qk}\, s_{ij}} && \text{the row normaliser}\\
W_{ij} &= G_{ij}\, e^{\mathrm{qk}\, s_{ij}} / Z_i^{\beta} \ \ (j \le i), \qquad W_{ij} = 0 \ \ (j > i)\\
O_i &= \textstyle\sum_{j \le i} W_{ij}\, V_j && \text{the readout}
\end{aligned}
\]

### The three corners

| Corner | \((\beta, \mathrm{qk}, g)\) | Operator |
| --- | --- | --- |
| Softmax attention | \((1, 1, 0)\) | \(W_{ij} = e^{s_{ij}} / \sum_{j' \le i} e^{s_{ij'}}\) |
| Unnormalized-kernel attention | \((0, 1, 0)\) | \(W_{ij} = e^{s_{ij}}\) |
| Path product | \((0, 0, 1)\) | \(W_{ij} = G_{ij} = \prod_{k=j+1}^{i} a_k\) |

The \(\beta = 0\) kernel corner is not linear attention in the \(O(n)\) sense. It is the unnormalized exponential kernel, which has no finite feature map, and it costs what the softmax corner costs.

### The path-product corner is a resolvent

At the path-product corner, with values \(b\) and \(b_0 = 0\), the readout is the gated chain

\[
y_0 = 0, \qquad y_i = a_i\, y_{i-1} + b_i .
\]

That recurrence is forward substitution of \((I - A)\,y = b\), where \(A_{i,i-1} = a_i\) and every other entry of \(A\) is zero. \(A\) is strictly lower triangular and therefore nilpotent, so the Neumann series terminates:

\[
G = (I - A)^{-1} = \sum_{k < n} A^k .
\]

The corner is the resolvent of the gate shift.

## What is machine-proved and what is only tested

The following are proved in the source's Lean 4, in `lean/CEQ/V16Domain.lean` unless another file is named. Each is mirrored numerically in `tests/resolvent.rs`.

| Lean theorem | Statement |
| --- | --- |
| `three_corners_containment` (`corner_softmax`, `corner_linear`, `corner_path_product`) | The family contains the three named operators |
| `corners_are_distinct` | At \(g \equiv 0\), \(\mathrm{qk} = 0\), entry \((1, 0)\) reads \(1/2\) at \(\beta = 1\) and \(1\) at \(\beta = 0\) |
| `softmax_row_sum_one`, `beta_one_row_is_one` | At \(\beta = 1\), rows sum to 1 |
| `gate_zero_beta_zero_is_linear_attention`, `gate_zero_beta_zero_row_not_one` | At \(g \equiv 0\), \(\beta = 0\) the head is the bare kernel. At \(\mathrm{qk} = 0\), row \(i\) sums to \(i + 1\). So \(\beta\), not \(g\), decides softmax-class membership |
| `pathProd_abs`, `pathProd_eq_zero_iff`, `prefix_logit_mask_restated` | \(\lvert G_{ij}\rvert = \prod m_k\) and \(\lvert G_{ij}\rvert \le 1\). \(\lvert G_{ij}\rvert = 1\) iff every magnitude on the path is 1. \(G_{ij} = 0\) exactly iff some \(m_k = 0\) on the path |
| `no_prefix_scan_represents_a_zero_gate` | \(\exp(C_i - C_j)\) is never zero, so the path product is evaluated as a product and never through a logarithm |
| `bedM_gate_exact`, `negative_draw_is_on_the_band` | \(-1, 0, +1\) are the gates \((m, \theta) = (1, \pi), (0, 0), (1, 0)\) |
| `constant_phase_gate_is_rope`, `cumulative_phase_is_separable` | A unit gate of constant phase \(\omega\) gives \(e^{i\omega(i-j)}\). Any phase schedule gives \(e^{i(\Theta_i - \Theta_j)}\), with \(\Theta\) the prefix sum |
| `PhaseH.scalar_gate_commutes` | A scalar path product is invariant under permuting the gates inside its window |
| `CEQ.V15.chain_path_product`, `CEQ.V15Fork.chain_eq_sum` | The chain equals \(\sum_s \big(\prod_{k=s+1}^{i} a_k\big) b_s\), over the reals, with no hypothesis on \(a\) |
| `CEQ.Nilpotent.occupancy_is_exact_inverse` | For strictly lower-triangular \(A\), \(\sum_{k<n} A^k\) is the exact two-sided inverse of \(I - A\) |

The following are tested only, not proved:

- Lean's `Hop` carries the gate as \(\exp(C_i - C_j)\) over the reals. Corners 1 and 2 (\(g = 0\), where both forms equal 1 on the causal triangle) are exactly its statements. Corner 3 with a zero magnitude or a nonzero phase rests on `pathProd`'s theorems. Its identity with this operator is a bitwise test, as it is in the source.
- The complex-gate chain identity; the Lean `chain` is real-valued.
- \(G(I - A) = I\) as a single matrix statement. It is a corollary of `chain_path_product` rather than a named theorem.
- Every property of the float64 evaluation: exact corners, causality, and the large-logit behaviour below.
- Anything for \(\beta\) strictly between 0 and 1. Lean's `Hop` makes no statement there either.

## Evaluation and refusals

\(Z_i^\beta\) is evaluated in the log domain, as \(W_{ij} = G_{ij}\exp(\mathrm{qk}\, s_{ij} - \beta \log Z_i)\). \(\log Z_i\) comes from a sum shifted by its maximum, taken over the live entries (\(R_{ij} > 0\)). This keeps the \(\beta = 1\) corner finite, with \(\lvert W_{ij}\rvert \le 1\), at logits of \(\pm 10^4\), where the dense form overflows.

| Case | Behaviour |
| --- | --- |
| All-masked row | Cannot occur. \(G_{ii} = R_{ii} = 1\) is the empty product, so every row has a live diagonal and \(Z_i > 0\). Closing every gate leaves the identity, not an empty row |
| Non-finite input | Refused with `NonFinite`. This covers a NaN or infinite query, key, magnitude, phase or switch, a blended phase \(g\theta_k\) that overflows, and a logit \(\mathrm{qk}\, s_{ij}\) that overflows. The source's `ceqjepa/operator.py::build_operator` refuses non-finite logits the same way |
| Empty sequence | Returns an empty operator, readout and chain |
| Overflow away from \(\beta = 1\) | A live entry whose true value exceeds the f64 range reads \(\pm\infty\); that is its value. It never reads NaN: a dead entry is exactly 0, and a lane of \(G_{ij}\) that is exactly 0 stays 0 rather than becoming \(0 \cdot \infty\). `readout` of such an operator is not defined and is not guarded |
| Shape mismatch, `head_dim = 0` with \(n > 0\) | Panics, as in `aether_core::attention` |

## Rust API

```rust
pub struct Complex { pub re: f64, pub im: f64 }
pub struct Switches { pub beta: f64, pub qk: f64, pub g: f64 }
impl Switches {
    pub const SOFTMAX: Switches;             // β = 1, qk = 1, g = 0
    pub const UNNORMALIZED_KERNEL: Switches; // β = 0, qk = 1, g = 0
    pub const PATH_PRODUCT: Switches;        // β = 0, qk = 0, g = 1
}
pub struct NonFinite;

pub fn blend(u: f64, theta: f64, g: f64) -> (f64, f64);
pub fn gate(m: f64, theta: f64) -> Complex;
pub fn path_product(a: &[Complex]) -> Vec<Complex>;                  // row-major [n, n]
pub fn operator(q: &[f64], k: &[f64], u: &[f64], theta: &[f64],
                seq: usize, head_dim: usize, switches: Switches)
    -> Result<Vec<Complex>, NonFinite>;                              // row-major [seq, seq]
pub fn readout(w: &[Complex], v: &[f64], seq: usize, value_dim: usize) -> Vec<Complex>;
pub fn chain_label(a: &[Complex], b: &[f64]) -> Vec<Complex>;
```

`path_product` accumulates from \(k = i\) down to \(k = j + 1\), the order the source fixes, because a complex product is not associative in floating point. \(a_0\) is never read.

## Test evidence

`tests/resolvent.rs` holds 17 `#[test]` functions. The references are written in the test file and share no code with the module. The softmax corner is also checked against the crate's own `attention::sparse_attention`.

```bash
cargo test -p aether-core --test resolvent
```

| Test | Mirrors | Pins |
| --- | --- | --- |
| `the_softmax_corner_is_causal_softmax_and_matches_the_crate_reference` | `corner_softmax` | Agreement to \(10^{-14}\), with an imaginary part of exactly 0 at \(g = 0\) |
| `the_unnormalized_kernel_corner_is_the_bare_exponential_bitwise` | `corner_linear` | Bitwise \(e^{s_{ij}}\) |
| `the_path_product_corner_is_the_gate_product_bitwise` | `corner_path_product`, `prefix_logit_mask_restated` | Bitwise against a triple loop, including closed and unit gates |
| `the_three_corners_are_distinct_operators` | `corners_are_distinct` | Entry \((1,0)\) is \(1/2\) against 1. Random inputs separate every pair of corners by more than 0.5 |
| `beta_alone_decides_row_stochasticity` | `softmax_row_sum_one`, `beta_one_row_is_one`, `gate_zero_beta_zero_row_not_one` | Rows sum to 1 within \(10^{-14}\) with gates on or off. The bare row \(i\) sums to exactly \(i + 1\) |
| `the_path_product_modulus_is_the_product_of_magnitudes` | `pathProd_abs` | Tolerance \(10^{-14}\) |
| `a_closed_gate_annihilates_every_path_through_it_exactly` | `pathProd_eq_zero_iff`, `no_prefix_scan_represents_a_zero_gate` | The zero survives every switch setting as an exact 0 |
| `the_bedm_values_are_ordinary_points_of_the_gate` | `bedM_gate_exact`, `negative_draw_is_on_the_band` | Exact real parts. The imaginary part left by \(\sin\pi\) is bounded by \(1.3 \times 10^{-16}\) |
| `a_constant_phase_is_the_rope_kernel_and_every_phase_schedule_separates` | `constant_phase_gate_is_rope`, `cumulative_phase_is_separable` | Toeplitz structure; tolerance \(10^{-13}\) |
| `scalar_gates_commute_within_a_window` | `PhaseH.scalar_gate_commutes` | Tolerance \(10^{-15}\). Windows away from the permutation are bitwise unchanged |
| `the_path_product_corner_reads_out_the_chain_and_inverts_the_gate_shift` | `chain_path_product`, `chain_eq_sum`, `occupancy_is_exact_inverse` | Readout equals the chain to \(10^{-12}\), and \(G(I - A) = I\) to \(10^{-14}\). As a planted negative, \(\beta = 1\) breaks the chain |
| `perturbing_the_future_never_moves_the_past` | (float64 only) | Rows up to \(t\) are bitwise unchanged when later inputs change. Entries above the diagonal are exactly 0 |
| `an_off_switch_deletes_its_term_exactly` | (float64 only) | \(g = 0\) removes the gates exactly, and \(\mathrm{qk} = 0\) removes \(q\) and \(k\) exactly |
| `logits_of_ten_thousand_stay_finite_at_beta_one` | (float64 only) | Logits past \(\pm 10^4\) stay finite. Rows are stochastic to \(10^{-9}\), and agree with `sparse_attention` to \(10^{-9}\) |
| `an_overflowing_logit_never_becomes_nan` | (float64 only) | At \(\beta = 0\), a logit of \(2 \times 10^4\) reads \(+\infty\), never NaN |
| `non_finite_input_is_refused` | (float64 only) | Each non-finite input class is refused. With \(\mathrm{qk} = 0\), an overflowing dot product is accepted |
| `an_empty_sequence_is_empty_and_no_row_is_ever_all_masked` | (float64 only) | Empty in, empty out. Closing every gate leaves the identity |

## Scope boundary

The module ports `arm_smprime.py` (operator, blend, gate, path product, readout) and `arm_phase.py::chain_label`. It does not port the Lean proofs, which remain in the source repository and are mirrored here as numerical tests. The module documentation names no other omitted component. It makes no statement for \(\beta \in (0, 1)\).

## Provenance

[resolvent](https://github.com/teerthsharma/resolvent), commit `8c19735`: `resolvent/ceq/arm_smprime.py` (`blend`, `magnitude`, `gate`, `path_product`, `operator`, `hop`, `numerator`, `block_summary`, `read_summary`, `readout`), `resolvent/ceq/arm_phase.py` (`chain_label`) and the Lean 4 development under `resolvent/lean/CEQ/`. The non-finite refusal follows `ceqjepa/operator.py::build_operator`.
