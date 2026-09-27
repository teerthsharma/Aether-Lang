# Linking Number

Module: `aether_core::linking`, in `crates/aether-core/src/linking.rs`. It computes the Gauss linking number, the writhe and the knot determinant of closed polygons in \(\mathbb{R}^3\). It also states the rule under which a floating-point linking number may be reported as an integer. Evidence: 18 tests in `tests/linking.rs`.

<div class="ts-viz" data-viz="int-link" data-title="Twisted-band link and its certified linking number" data-caption="The two boundary curves of an annulus with k full twists, which form the (2, 2k) torus link, drawn with the twisted_band generator from tests/linking.rs. The browser recomputes the closed-form sum, the error bound B and the certify rule from linking.rs. Reversing B negates lk."></div>

## Object

A closed polygon is a vertex list \(a_0, \ldots, a_{n-1} \in \mathbb{R}^3\) with \(n \ge 3\). The list is read cyclically: segment \(i\) runs from \(a_i\) to \(a_{(i+1) \bmod n}\). For two disjoint closed curves \(A\) and \(B\), with \(r_A\) a point on \(A\) and \(r_B\) a point on \(B\), the Gauss double integral

\[
\operatorname{Lk}(A,B) = \frac{1}{4\pi} \oint_A \oint_B \frac{(r_A - r_B)\cdot(\mathrm{d}r_A \times \mathrm{d}r_B)}{\lVert r_A - r_B \rVert^3}
\]

is an integer, and it is invariant under isotopies that keep the curves disjoint.

### Closed form over segment pairs

Take segment \(p_1 \to p_2\) of \(A\) and segment \(p_3 \to p_4\) of \(B\). Write \(r_{ij} = p_j - p_i\) and \(\hat r_{ij} = r_{ij} / \lVert r_{ij} \rVert\). For unit vectors \(a, b, c\), the Van Oosterom–Strackee formula gives the signed solid angle of their spherical triangle:

\[
\Omega(a,b,c) = 2\,\operatorname{atan2}\!\big(a\cdot(b\times c),\; 1 + a\cdot b + a\cdot c + b\cdot c\big).
\]

The Gauss map sweeps a spherical quadrilateral with vertices \(\hat r_{13}, \hat r_{14}, \hat r_{24}, \hat r_{23}\). Fanning it from \(\hat r_{13}\) gives

\[
\omega_{ij} = -\Big[\Omega(\hat r_{13}, \hat r_{14}, \hat r_{24}) + \Omega(\hat r_{13}, \hat r_{24}, \hat r_{23})\Big],
\qquad
\widehat{\operatorname{Lk}} = \frac{1}{4\pi} \sum_{i} \sum_{j} \omega_{ij}.
\]

The sum has no quadrature term and no length constant. The leading minus sign orients it to the integral. nerve fixed that sign against an independent midpoint quadrature, and `linking_number_matches_midpoint_quadrature` pins it again here.

The writhe applies the same sum to one curve against itself, over non-adjacent segment pairs:

\[
\operatorname{Wr}(A) = \frac{2}{4\pi} \sum_{i < j,\ \text{non-adjacent}} \omega_{ij}.
\]

Adjacent segments are coplanar with their shared vertex and contribute zero. The writhe depends on the embedding, is not an invariant, and is never rounded.

### Error bound

\(u = 2^{-53}\) is the binary64 unit roundoff. The bound assumes IEEE-754 round-to-nearest, a correctly rounded `sqrt` and an `atan2` accurate to 2 ulp. It treats the vertices as exact, so it covers the arithmetic and not the process that produced the coordinates. Terms of order \(u^2\) are dropped.

1. Each difference \(p_j - p_i\) is rounded componentwise, and normalising it costs at most \(5u\) per component. Each unit direction therefore lies within \(7u\) of the exact direction.
2. With directions perturbed by \(7u\), the computed numerator \(N\) and denominator \(D\) of each triangle lie within \(47u\) and \(60u\) of their exact values. Both are bounded by \(K = 128u\).
3. Suppose the box of half-width \(K\) about \((D, N)\) misses the branch cut \(\{N = 0,\ D \le 0\}\), and \(\rho = \operatorname{hypot}(N, D) > 2K\). Then `atan2` is smooth on the box, and the triangle's error is
   \[
   e = 2\left(\frac{\sqrt{2}\,K}{\rho - \sqrt{2}\,K} + 8u\right).
   \]
4. Recursive summation of \(m\) terms adds at most \(\gamma_m \sum \lvert \omega_{ij} \rvert\), with \(\gamma_m = mu/(1 - mu)\) (Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 4). The final division by \(4\pi\) adds \(2u\,\lvert\widehat{\operatorname{Lk}}\rvert\).

\[
B = \frac{1}{4\pi}\Big(\sum_{ij} e_{ij} + \gamma_m \sum_{ij} \lvert\omega_{ij}\rvert\Big) + 2u\,\big\lvert\widehat{\operatorname{Lk}}\big\rvert,
\]

where \(e_{ij}\) sums the two triangle errors of pair \(ij\) and the rounding \(u\,\lvert\Omega_1 + \Omega_2\rvert\) of their sum. nerve measured the deviation from the integer on \((2, 2n)\) torus links, at most \(2.16 \times 10^{-13}\) at 1024 segments and growing with the segment count, and rounded on that evidence. This port carries \(B\) instead.

## What is certified

Let \(n = \operatorname{round}(\widehat{\operatorname{Lk}})\). `GaussLinking::certify` rounds only when

\[
\big\lvert \widehat{\operatorname{Lk}} - n \big\rvert + B < \tfrac{1}{2}.
\]

The interval \([\widehat{\operatorname{Lk}} - B,\ \widehat{\operatorname{Lk}} + B]\) then lies inside \((n - \tfrac12,\ n + \tfrac12)\), so \(n\) is the only integer it can contain. Half a unit is the threshold that `nerve-periodic` names as the only defensible one for a closed polygon.

| Verdict | Condition | Meaning |
| --- | --- | --- |
| `Linked { lk }` | rule holds, \(n \ne 0\) | \(\operatorname{Lk} = n\). A split link has \(\operatorname{Lk} = 0\), so no isotopy that keeps the curves disjoint can separate them (tangle's theorem T3, read contrapositively). |
| `ZeroLinking` | rule holds, \(n = 0\) | \(\operatorname{Lk} = 0\) is proven, and it certifies nothing. The Whitehead link has \(\operatorname{Lk} = 0\) and is not split. |
| `Undetermined { lk_estimate, error_bound }` | rule fails, including NaN | The estimate is returned. No claim is made. |

No verdict reads "unlinked", because the linking number cannot supply the converse of the certificate. The certificate is conditional on the first-order bound above. It is not interval arithmetic.

| Quantity | Status |
| --- | --- |
| Integer \(\operatorname{Lk}\) | Certified, conditional on the first-order bound |
| \(\widehat{\operatorname{Lk}}\), \(B\) | Computed value and a first-order bound on its error |
| Writhe | Estimate, no certificate |
| Knot determinant | Exact `i128` arithmetic on the diagram. Extracting the diagram uses an absolute tolerance of \(10^{-9}\) and carries no error bound. |

## Knot determinant

Consider a knot diagram in which crossing \(c\) has over-arc \(o\), incoming under-arc \(a\) and outgoing under-arc \(b\). The Alexander matrix rows are

\[
\text{positive: } a \mapsto t,\ \ b \mapsto -1,\ \ o \mapsto 1 - t;
\qquad
\text{negative: } a \mapsto 1,\ \ b \mapsto -t,\ \ o \mapsto t - 1.
\]

At \(t = -1\) these become \((-1, -1, 2)\) and \((1, 1, -2)\), which are exact negatives. Negating a row leaves \(\lvert\det\rvert\) unchanged, so for a diagram with \(c\) crossings

\[
\det(K) = \lvert \Delta_K(-1) \rvert = \big\lvert \det M' \big\rvert,
\]

where \(M'\) is any \((c-1)\times(c-1)\) minor of the \(c \times c\) matrix at \(t = -1\). No crossing sign is computed. The minor is evaluated exactly in `i128` by Bareiss elimination. The diagram comes from a projection along \(z\), retried over eight fixed reorientations when that projection is not generic. The projection tolerance \(10^{-9}\) is the source's and is not scale-free. The determinant is an invariant but not a complete one: \(4_1\) and \(5_1\) both give 5, and a value of 1 does not certify the unknot.

## Refusals

Every refusal is a `LinkingError`, never a number.

| Variant | Condition |
| --- | --- |
| `TooFewVertices { curve, len }` | A curve has fewer than three vertices. nerve returned `0.0`, or `Some(1)` for the determinant. |
| `NonFinite { curve, vertex }` | A coordinate is NaN or infinite. |
| `Intersecting { segment_a, segment_b }` | Two segments meet, or come within rounding of meeting. This is the branch-cut condition of step 3, and it involves no length scale. |
| `OutOfRange { segment_a, segment_b }` | A difference vector's squared length is not a normal binary64 number, that is, a length outside roughly \([1.5 \times 10^{-154},\ 1.3 \times 10^{154}]\). The analysis does not hold there. |
| `NoGenericProjection` | None of the eight reorientations gives a generic projection with an `i128`-representable determinant. |

## Rust API

```rust
pub fn linking_number(a: &[[f64; 3]], b: &[[f64; 3]]) -> Result<GaussLinking, LinkingError>;
pub fn writhe(a: &[[f64; 3]]) -> Result<f64, LinkingError>;
#[cfg(feature = "alloc")]
pub fn knot_determinant(curve: &[[f64; 3]]) -> Result<u128, LinkingError>;

pub struct GaussLinking { pub value: f64, pub error_bound: f64 }
impl GaussLinking { pub fn certify(&self) -> LinkVerdict; }

pub enum LinkVerdict {
    Linked { lk: i64 },
    ZeroLinking,
    Undetermined { lk_estimate: f64, error_bound: f64 },
}

pub enum LinkingError {
    TooFewVertices { curve: usize, len: usize },
    NonFinite { curve: usize, vertex: usize },
    Intersecting { segment_a: usize, segment_b: usize },
    OutOfRange { segment_a: usize, segment_b: usize },
    NoGenericProjection,
}
```

`linking_number` visits \(O(\lvert a\rvert\,\lvert b\rvert)\) segment pairs and does not allocate. `writhe` visits \(O(\lvert a\rvert^2)\) pairs and does not allocate. Reversing one curve negates the linking number, and swapping the curves leaves it unchanged.

## Test evidence

`tests/linking.rs` holds 18 `#[test]` functions. The generators `twisted_band`, `rotate`, `torus_knot` and `figure_eight` are nerve's. The braid words are tangle's.

```bash
cargo test -p aether-core --test linking
```

| Test | Pins |
| --- | --- |
| `hopf_link_certifies_linked_with_lk_of_magnitude_one` | The Hopf link (one twist, 200 vertices) certifies \(\lvert lk \rvert = 1\) |
| `torus_link_2_2k_certifies_lk_k_with_one_sign_across_the_family` | Coplanar rings sum to exactly `0.0` and give `ZeroLinking`. The \((2,2k)\) links certify \(\lvert lk\rvert = k\) for \(k = 1..4\), all with one sign |
| `whitehead_link_has_lk_zero_and_is_never_certified_as_separable` | Braid controls give \(\lvert lk\rvert\) of 1 and 2. The Whitehead braid `[1, -2, 1, -2, -2]` gives `ZeroLinking`, with its value inside \(B\) |
| `linking_number_matches_midpoint_quadrature` | The closed form agrees with a midpoint rule within \(2\times10^{-2}\) at 500 vertices, for 0 to 2 twists. This pins the sign and the \(1/(4\pi)\) |
| `error_bound_covers_the_measured_distance_to_the_integer` | For 1 to 3 twists at \(m \in \{64, 256, 512\}\), the deviation from the integer is at most \(B\), and \(B < 10^{-6}\) |
| `certification_rounds_only_when_the_bound_proves_the_rounding` | \((0.9, 0.2)\) certifies 1. \((0.7, 0.2)\), \((0.5, 10^{-12})\), \((3.0, 0.5)\), NaN and \(\infty\) give `Undetermined` |
| `linking_number_is_invariant_under_rigid_rotation` | Value agrees within \(10^{-10}\), with the same verdict |
| `linking_number_is_invariant_under_translation` | Translation by \((123.5, -7.25, 4096)\) keeps the value within \(10^{-10}\) |
| `linking_number_and_bound_are_bitwise_invariant_under_power_of_two_scaling` | Value and bound are bitwise identical for \(c \in \{0.25, 0.5, 2, 1024, 1/1024, 2^{400}\}\) |
| `linking_number_is_invariant_under_cyclic_shift_of_the_start_vertex` | Value agrees within \(10^{-12}\) |
| `reversing_one_orientation_or_reflecting_negates_lk` | Reversing \(A\), reversing \(B\) or reflecting negates \(lk\). Reversing both preserves it |
| `swapping_the_curves_keeps_lk` | Value agrees within \(10^{-12}\) |
| `a_strand_through_the_other_curve_is_refused_and_either_side_is_certified` | Offsets of \(\pm 1/1024\) certify \(\lvert lk\rvert = 1\) or `ZeroLinking`. Offset 0 and a shared vertex are `Intersecting` |
| `too_few_vertices_and_non_finite_coordinates_are_refused_everywhere` | All three functions refuse, and the refusal names the curve and vertex |
| `out_of_range_differences_are_refused_rather_than_summed_as_zero` | Scales \(2^{600}\) and \(2^{-600}\) give `OutOfRange`, not a false zero |
| `writhe_is_real_valued_odd_under_reflection_and_refuses_self_intersection` | A planar circle's writhe is below \(10^{-12}\). A twisted curve's writhe is not integral. Reflection negates it. A lemniscate of Gerono is refused |
| `knot_determinant_matches_the_classical_table` | Unknot 1, \(3_1\) 3, \(4_1\) 5, \(5_1\) 5, \(7_1\) 7, and braid closures give 3 and 5 |
| `knot_determinant_is_projection_invariant_mirror_blind_and_incomplete` | Four rotations of the trefoil give 3. The mirror trefoil gives 3. \(4_1\) and \(5_1\) agree |

## Deviations from the sources

The error bound \(B\) and the rounding rule built on it appear in neither source. They are derived for this port. nerve returned `0.0` from the linking number for fewer than three vertices and for degenerate segment pairs, and `Some(1)` from the determinant for fewer than three vertices. Here each of these is a typed refusal. The module documentation names no source component that was left unported.

## Provenance

- [nerve](https://github.com/teerthsharma/nerve): `nerve/crates/nerve-topo/src/lib.rs` (`linking_number_closed`, `omega`, `solid_angle`, `writhe_closed`), `nerve/crates/nerve-melt/src/lib.rs` (`alexander_det_minus_one`, `det_for_orientation`, `det_bareiss`) and `nerve/crates/nerve-periodic/src/lib.rs` (the half-unit threshold).
- [tangle](https://github.com/teerthsharma/tangle): `tangle/tangle/certify.py`, the one-directional `LINKED` certificate (`CERTIFIED LINKED` when \(lk \ne 0\), `NOT CERTIFIED LK_ZERO` when \(lk = 0\)), its theorem T3, and the braid words of `tests/test_certify.py` and `tests/test_alexander.py`.
