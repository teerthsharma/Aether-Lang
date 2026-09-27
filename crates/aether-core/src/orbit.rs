//! Orbit partitions of many-to-one maps, and the five bounds they certify.
//!
//! A map `f : E → A` on a finite set `E = {0, …, n − 1}` partitions `E` into its
//! fibres, the classes of `e₁ ~ e₂ ⇔ f(e₁) = f(e₂)`. The fibres are `H₀` of that
//! equivalence relation. Following the source they are called *orbits*, and `m`
//! is their number. The same object arises from a relation known only pairwise,
//! as the connected components of a graph, so [`Partition`] is built either way:
//! [`Partition::from_map`] from values, [`Partition::from_edges`] by union-find.
//!
//! Ported from `caustic/caustic/theorems.py` and `caustic/caustic/regime.py`
//! (T. Sharma, *Caustic*, doi:10.5281/zenodo.21997746), and from
//! `branchcut/branchcut/partition.py`, which restates the same results over an
//! arbitrary finite set. Theorem numbers below are caustic's. Nothing in this
//! module depends on a language model.
//!
//! # Notation
//!
//! `R : E → A` is the unknown ground relation, *injective* when distinct
//! elements have distinct correct values, and `err(f) = |{e : f(e) ≠ R(e)}|`.
//! `G = R(E)` is the set of correct values without the pairing, and
//! `m* = |f(E) ∩ G|`. `S = {e : |[e]| > 1}` is the certified set, `b` the
//! number of orbits of size at least two, `b_adm` the number of those whose
//! shared value lies in `G`, and `S* = S ∪ {e : f(e) ∉ G}`.
//!
//! # The five bounds
//!
//! One bound per quantity bounded. Each holds for every injective `R`
//! consistent with the observation and is computed without consulting `R`.
//! Where the source has a sharpened form, it additionally reads `G`, which is a
//! set of values and not an answer key.
//!
//! **Bound 1 — error floor** (Theorems 1 and 1*; branchcut Theorem 1).
//!
//! ```text
//! err(f)  ≥  n − m*  ≥  n − m
//! ```
//!
//! A map constant on an orbit agrees with an injective `R` on at most one of
//! its members, so an orbit of size `s` holds at least `s − 1` errors and
//! `Σ (sᵢ − 1) = n − m`. With `G`: the correct set `C` has `f|_C = R|_C`
//! injective and `f(C) ⊆ f(E) ∩ G`, so `|C| ≤ m*`. `n − m` is attained exactly
//! when every orbit contains one correct value.
//! [`orbit_error_bound`], [`admissible_error_bound`], [`certified_error_floor`].
//!
//! **Bound 2 — pooling recovery** (Theorem 2; branchcut Theorem 2). For an
//! orbit of size `k`, any `h : A → E`, and the uniform prior on that orbit,
//!
//! ```text
//! Pr[ h(f(e)) = e ]  ≤  1 / k
//! ```
//!
//! since `h ∘ f` is constant on the orbit. Attained by the decoder that returns
//! one fixed member. [`pooling_recovery_bound`].
//!
//! **Bound 3 — join recovery** (Theorem 2*). For maps `f₁, …, f_T` whose join
//! (common refinement, [`Partition::join`]) has `m_join` blocks, and any `h` of
//! the value tuple,
//!
//! ```text
//! |{e : h(f₁(e), …, f_T(e)) = e}| / n  ≤  m_join / n
//! ```
//!
//! since `h` is constant on each join block. The join is at least as fine as
//! every component, so `m_join ≥ max_t m_t` and this ceiling is never below any
//! single coordinate's. Attained at one recovery per join block.
//! [`join_recovery_bound`].
//!
//! **Bound 4 — precision of the certified set** (Theorems 6 and 6*).
//!
//! ```text
//! |S ∩ wrong| / |S|  ≥  (|S| − b_adm) / |S|  ≥  (|S| − b) / |S|  =  (n − m) / |S|
//! ```
//!
//! Every error Bound 1 counts lies in `S`, since a singleton contributes
//! `s − 1 = 0`. An orbit whose value is not in `G` has no correct member at all.
//! Every counted orbit has two or more members, so `|S| ≥ 2b` and the floor is
//! at least `1/2` whenever `S` is non-empty. Attained by the truth that places
//! one correct member in every admissible collapsed orbit.
//! [`certified_precision_bound`], [`admissible_precision_bound`].
//!
//! **Bound 5 — recall of `S*`** (Theorem 8). Whenever `err(f) ≥ 1`,
//!
//! ```text
//! |S* ∩ wrong| / |wrong|  ≥  (n − m*) / n
//! ```
//!
//! since the `n − m*` errors of Bound 1 all lie in `S*` and `|wrong| ≤ n`.
//! Attained, and zero exactly when `m* = n`. That zero is Theorem 7: with `f` a
//! bijection onto `G`, the truths `R = f` and `R = f ∘ σ` for a fixed-point-free
//! `σ` produce the same observation while recall is 1 in the first and 0 in the
//! second, so no *uniformly* positive recall floor exists. [`recall_floor`].
//!
//! # Proved, and not
//!
//! All five inequalities are proved in the source by the elementary counting
//! arguments sketched above. The source also reports measurements on language
//! models: how often each bound was attained or loose, and a detection score.
//! Those are properties of particular models and datasets, not of this module,
//! and none is restated here. The tests pin each inequality on seeded random
//! instances and exhaustive small cases; they check the proofs and do not
//! replace them.
//!
//! Precision is bounded from the partition alone. Recall is bounded only with
//! `G`, and never uniformly away from zero.
//!
//! # Hypothesis and refusal
//!
//! Injectivity of `R` is the hypothesis of every bound, and the partition cannot
//! decide it: on a many-to-one relation distinct elements *should* share a
//! value, and `n − m` then certifies nothing. It is the caller's assertion. The
//! one violation visible from data is `|G| < n`, on which
//! [`admissible_distinct`] and [`admissible_collapsed_blocks`] return `None`.
//!
//! Every bound returns `None` on counts no partition of these elements can
//! produce, rather than a number: `n = 0`; `m ∉ [1, n]`; `m* > n`; `k = 0`;
//! `m_join ∉ [1, n]`; an empty `S`, whose precision is undefined rather than 1;
//! `|S| > n` or `|S| < n − m`; and `2·b_adm > |S|`.
//!
//! The certificate is one-sided. It can prove a map wrong and never proves one
//! right.

#![warn(missing_docs)]

use alloc::vec;
use alloc::vec::Vec;

/// The partition of `{0, …, n − 1}` into orbits.
///
/// Stored canonically: every element records the smallest member of its orbit
/// and the orbit's size. Two partitions compare equal exactly when they have the
/// same blocks, whatever values or edge order produced them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partition {
    representative: Vec<usize>,
    size: Vec<usize>,
}

impl Partition {
    /// The fibres of `e ↦ values[e]`.
    ///
    /// Ported from `branchcut.partition.partition_by_key` and
    /// `caustic.regime.orbit_partition`. Sorting replaces hashing so the
    /// function needs only `Ord` and no allocator-backed map. `O(n log n)`.
    pub fn from_map<K: Ord>(values: &[K]) -> Self {
        let n = values.len();
        let mut order: Vec<usize> = (0..n).collect();
        // Stable, so each run of equal values keeps ascending element order and
        // its first entry is the orbit's smallest member.
        order.sort_by(|&a, &b| values[a].cmp(&values[b]));

        let mut representative = vec![0; n];
        let mut size = vec![0; n];
        for run in order.chunk_by(|&a, &b| values[a] == values[b]) {
            for &e in run {
                representative[e] = run[0];
                size[e] = run.len();
            }
        }
        Self {
            representative,
            size,
        }
    }

    /// Connected components of `n` vertices under `edges`.
    ///
    /// Ported from `branchcut.partition.components`: union by size with path
    /// halving, near-linear in `edges.len()`. The result depends on the edge
    /// set only, not on edge order or repetition.
    ///
    /// Returns `None` if an edge names a vertex `≥ n`.
    pub fn from_edges(n: usize, edges: &[(usize, usize)]) -> Option<Self> {
        let mut parent: Vec<usize> = (0..n).collect();
        let mut size = vec![1; n];
        for &(a, b) in edges {
            if a >= n || b >= n {
                return None;
            }
            let (mut ra, mut rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra == rb {
                continue;
            }
            if size[ra] < size[rb] {
                core::mem::swap(&mut ra, &mut rb);
            }
            parent[rb] = ra;
            size[ra] += size[rb];
        }

        // The union-find root is arbitrary. Re-key each component by its
        // smallest member, which is the first one met in ascending order.
        let mut smallest = vec![usize::MAX; n];
        let mut representative = Vec::with_capacity(n);
        let mut block = Vec::with_capacity(n);
        for e in 0..n {
            let root = find(&mut parent, e);
            if smallest[root] == usize::MAX {
                smallest[root] = e;
            }
            representative.push(smallest[root]);
            block.push(size[root]);
        }
        Some(Self {
            representative,
            size: block,
        })
    }

    /// The join: the common refinement of `self` and `other`.
    ///
    /// Two elements share a block exactly when they share one in both. Folding
    /// this over `T` partitions gives the partition of the value tuple that
    /// Bound 3 is stated on. Ported from `caustic.regime.join_partition`.
    ///
    /// Returns `None` if the partitions are over different numbers of elements.
    pub fn join(&self, other: &Self) -> Option<Self> {
        if self.n() != other.n() {
            return None;
        }
        let pairs: Vec<(usize, usize)> = self
            .representative
            .iter()
            .copied()
            .zip(other.representative.iter().copied())
            .collect();
        Some(Self::from_map(&pairs))
    }

    /// `n`, the number of elements.
    pub fn n(&self) -> usize {
        self.representative.len()
    }

    /// `m`, the number of orbits.
    pub fn m(&self) -> usize {
        self.representative
            .iter()
            .enumerate()
            .filter(|&(e, &r)| e == r)
            .count()
    }

    /// The smallest member of the orbit of `e`.
    ///
    /// # Panics
    ///
    /// If `e ≥ n`.
    pub fn representative(&self, e: usize) -> usize {
        self.representative[e]
    }

    /// The size of the orbit of `e`, the `k` of Bound 2.
    ///
    /// # Panics
    ///
    /// If `e ≥ n`.
    pub fn block_size(&self, e: usize) -> usize {
        self.size[e]
    }

    /// The size of the largest orbit, `0` when `n = 0`. Bound 2 on this orbit is
    /// the recovery ceiling of the whole map.
    pub fn largest(&self) -> usize {
        self.size.iter().copied().max().unwrap_or(0)
    }

    /// Whether `e` lies in the certified set `S`, i.e. in an orbit of size at
    /// least two. Ported from `caustic.regime.OrbitReport.certified_set`.
    ///
    /// # Panics
    ///
    /// If `e ≥ n`.
    pub fn is_flagged(&self, e: usize) -> bool {
        self.size[e] > 1
    }

    /// `|S|`, the number of elements in orbits of size at least two.
    pub fn flagged(&self) -> usize {
        self.size.iter().filter(|&&s| s > 1).count()
    }

    /// `b`, the number of orbits of size at least two.
    pub fn collapsed_blocks(&self) -> usize {
        self.representative
            .iter()
            .zip(&self.size)
            .enumerate()
            .filter(|&(e, (&r, &s))| e == r && s > 1)
            .count()
    }

    /// The collision score of `e`: the fraction of *other* elements sharing its
    /// value, `(|[e]| − 1) / (n − 1)`.
    ///
    /// Ported from `caustic.regime.symmetry_scores`, `collision_scored`. A
    /// per-element score, not a bound: under an injective `R` any non-zero value
    /// places `e` in `S`. `None` when `n < 2`, where no other element exists.
    ///
    /// # Panics
    ///
    /// If `e ≥ n`.
    pub fn collision(&self, e: usize) -> Option<f64> {
        let n = self.n();
        let others = self.size[e] - 1;
        (n >= 2).then(|| others as f64 / (n - 1) as f64)
    }
}

/// Union-find root with path halving.
fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// `G` as a sorted, de-duplicated set, or `None` if it holds fewer than `n`
/// distinct values and so cannot be the image of an injective `R` on `n`
/// elements.
fn gold_set<K: Ord>(gold: &[K], n: usize) -> Option<Vec<&K>> {
    let mut set: Vec<&K> = gold.iter().collect();
    set.sort();
    set.dedup();
    (set.len() >= n).then_some(set)
}

/// `m* = |f(E) ∩ G|`, the input to Bounds 1 and 5, for `f(e) = values[e]`.
///
/// Ported from `caustic.regime.admissible_distinct`. Distinct values are
/// counted, not elements: an orbit sharing an admissible value holds at most one
/// correct element.
///
/// Returns `None` when `gold` holds fewer than `values.len()` distinct values.
/// That is the precondition failing where it can be seen: two elements share a
/// correct value, `R` is not injective, and every bound is void.
pub fn admissible_distinct<K: Ord>(values: &[K], gold: &[K]) -> Option<usize> {
    let gold = gold_set(gold, values.len())?;
    let mut image: Vec<&K> = values.iter().collect();
    image.sort();
    image.dedup();
    Some(
        image
            .into_iter()
            .filter(|v| gold.binary_search(v).is_ok())
            .count(),
    )
}

/// `b_adm`, the number of orbits of size at least two whose shared value lies in
/// `G`, for `f(e) = values[e]`. The input to Bound 4's sharpened form, as
/// defined in caustic Theorem 6*.
///
/// Returns `None` under the same precondition failure as [`admissible_distinct`].
pub fn admissible_collapsed_blocks<K: Ord>(values: &[K], gold: &[K]) -> Option<usize> {
    let gold = gold_set(gold, values.len())?;
    let p = Partition::from_map(values);
    Some(
        (0..p.n())
            .filter(|&e| p.representative(e) == e && p.is_flagged(e))
            .filter(|&e| gold.binary_search(&&values[e]).is_ok())
            .count(),
    )
}

/// Bound 1: `err(f) ≥ n − m`, for injective `R`.
///
/// Ported from `caustic.theorems.orbit_error_bound` (Theorem 1) and
/// `branchcut.partition.min_errors`. `None` unless `n ≥ 1` and `1 ≤ m ≤ n`.
pub fn orbit_error_bound(n: usize, m: usize) -> Option<usize> {
    (n >= 1 && (1..=n).contains(&m)).then(|| n - m)
}

/// Bound 1, sharpened by `G`: `err(f) ≥ n − m*`, for injective `R`.
///
/// Ported from `caustic.theorems.admissible_error_bound` (Theorem 1*). Never
/// weaker than [`orbit_error_bound`], since `m* ≤ m`. `m* = 0` is allowed: it is
/// total collapse onto values no element's truth could be, and certifies all
/// `n`. `None` unless `n ≥ 1` and `m* ≤ n`.
pub fn admissible_error_bound(n: usize, m_star: usize) -> Option<usize> {
    (n >= 1 && m_star <= n).then(|| n - m_star)
}

/// Bound 1 as a rate, `(n − m) / n`: a lower bound on the error rate.
///
/// Ported from `caustic.theorems.certified_error_floor` and
/// `branchcut.partition.collision_error_floor`. Refuses what
/// [`orbit_error_bound`] refuses.
pub fn certified_error_floor(n: usize, m: usize) -> Option<f64> {
    orbit_error_bound(n, m).map(|e| e as f64 / n as f64)
}

/// Bound 2: no decoder recovers an element of a `k`-orbit from its value with
/// probability above `1 / k`.
///
/// Ported from `caustic.theorems.pooling_recovery_bound` (Theorem 2) and
/// `branchcut.partition.recovery_ceiling`. `None` when `k = 0`.
pub fn pooling_recovery_bound(k: usize) -> Option<f64> {
    (k >= 1).then(|| 1.0 / k as f64)
}

/// Bound 3: no decoder of the value tuple recovers more than a fraction
/// `m_join / n` of the elements.
///
/// Ported from `caustic.theorems.join_recovery_bound` (Theorem 2*). `None`
/// unless `n ≥ 1` and `1 ≤ m_join ≤ n`.
pub fn join_recovery_bound(n: usize, m_join: usize) -> Option<f64> {
    (n >= 1 && (1..=n).contains(&m_join)).then(|| m_join as f64 / n as f64)
}

/// Bound 4: `precision(S) ≥ (n − m) / |S|`, for injective `R`.
///
/// Ported from `caustic.theorems.certified_precision_bound` (Theorem 6).
/// Equal to `1 − b / |S|`, and at least `1/2` on any partition that produces
/// it. `None` when [`orbit_error_bound`] refuses, when `flagged = 0` (the
/// precision of an empty set is undefined, not 1), when `flagged > n`, or when
/// `flagged < n − m`, which no partition with these counts can produce.
pub fn certified_precision_bound(n: usize, m: usize, flagged: usize) -> Option<f64> {
    let errors = orbit_error_bound(n, m)?;
    (flagged >= 1 && flagged <= n && flagged >= errors).then(|| errors as f64 / flagged as f64)
}

/// Bound 4, sharpened by `G`: `precision(S) ≥ (|S| − b_adm) / |S|`, for
/// injective `R`.
///
/// Ported from `caustic.theorems.admissible_precision_bound` (Theorem 6*).
/// Never below [`certified_precision_bound`], since `b_adm ≤ b`. `None` when
/// `flagged = 0`, or when `2·b_adm > flagged`, which is impossible because every
/// counted orbit has at least two members.
pub fn admissible_precision_bound(flagged: usize, b_adm: usize) -> Option<f64> {
    (flagged >= 1 && 2 * b_adm <= flagged)
        .then(|| (flagged - b_adm) as f64 / flagged as f64)
}

/// Bound 5: `recall(S*) ≥ (n − m*) / n` whenever `f` errs at all, for injective
/// `R`.
///
/// Ported from `caustic.theorems.recall_floor` (Theorem 8). Zero exactly when
/// `m* = n`, which is the witness of Theorem 7. Refuses what
/// [`admissible_error_bound`] refuses.
pub fn recall_floor(n: usize, m_star: usize) -> Option<f64> {
    admissible_error_bound(n, m_star).map(|e| e as f64 / n as f64)
}
