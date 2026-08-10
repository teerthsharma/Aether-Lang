//! Metrics and vectorizations for persistence diagrams.
//!
//! A diagram is not directly usable by a model: it is a multiset of variable
//! size with no natural vector-space structure. This module supplies the two
//! things that fix that:
//!
//!   - **Metrics** — bottleneck and p-Wasserstein, for comparison, clustering,
//!     and kernel methods.
//!   - **Vectorizations** — landscapes and images, which map a diagram to a
//!     fixed-length real vector while remaining stable under perturbation of the
//!     underlying point cloud.
//!
//! Stability is the whole point. Both metrics reproduce the Cohen-Steiner–
//! Edelsbrunner–Harer bound, and the landscape is 1-Lipschitz with respect to the
//! bottleneck distance, so a small change to the data produces a small change to
//! the feature vector. A vectorization without that guarantee is a hash, not a
//! feature.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use libm::{exp, pow, sqrt};

use crate::persistence::PersistenceDiagram;

/// A finite bar, as the metrics see it.
type Bar = (f64, f64);

/// Finite bars of one homology dimension, birth-sorted.
fn finite_bars(diagram: &PersistenceDiagram, dimension: usize) -> Vec<Bar> {
    let mut bars: Vec<Bar> = diagram
        .pairs
        .iter()
        .filter(|pair| pair.dimension == dimension)
        .filter_map(|pair| pair.death.map(|death| (pair.birth, death)))
        .collect();
    bars.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    bars
}

// ═══════════════════════════════════════════════════════════════════════════════
// Metrics
// ═══════════════════════════════════════════════════════════════════════════════

/// Cost matrix for the standard diagram-matching construction on `n + m` nodes
/// per side: real bars may match each other or be projected to the diagonal, and
/// the surplus diagonal slots match each other at zero cost.
fn cost_matrix(a: &[Bar], b: &[Bar]) -> Vec<Vec<f64>> {
    let (n, m) = (a.len(), b.len());
    let size = n + m;
    let mut cost = vec![vec![0.0f64; size]; size];

    for (i, row) in cost.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            *entry = match (i < n, j < m) {
                // Both real: L-infinity distance in the (birth, death) plane.
                (true, true) => {
                    let (b0, d0) = a[i];
                    let (b1, d1) = b[j];
                    fmax((b0 - b1).abs(), (d0 - d1).abs())
                }
                // Real against the diagonal: half the bar's persistence, the
                // L-infinity distance to the nearest diagonal point.
                (true, false) => (a[i].1 - a[i].0) / 2.0,
                (false, true) => (b[j].1 - b[j].0) / 2.0,
                (false, false) => 0.0,
            };
        }
    }
    cost
}

fn fmax(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else {
        b
    }
}

/// Bottleneck distance between the dimension-`dimension` parts of two diagrams.
///
/// The minimum, over all matchings that may send bars to the diagonal, of the
/// largest single matched cost. Computed exactly: binary search over the finite
/// set of candidate costs, with feasibility decided by augmenting-path matching
/// on the threshold graph.
///
/// Essential (infinite-death) classes are ignored. Compare their counts
/// separately if that matters for the application; there is no finite distance
/// between an essential class and a finite one.
pub fn bottleneck_distance(
    left: &PersistenceDiagram,
    right: &PersistenceDiagram,
    dimension: usize,
) -> f64 {
    let a = finite_bars(left, dimension);
    let b = finite_bars(right, dimension);
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }

    let cost = cost_matrix(&a, &b);
    let mut candidates: Vec<f64> = cost.iter().flatten().copied().collect();
    candidates.sort_by(f64::total_cmp);
    candidates.dedup();

    let (mut lo, mut hi) = (0usize, candidates.len() - 1);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if perfect_matching_exists(&cost, candidates[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    candidates[lo]
}

/// p-Wasserstein distance between the dimension-`dimension` parts of two
/// diagrams: the p-th root of the minimum total p-th power of matched costs.
///
/// `p` must be at least 1. The bottleneck distance is the `p -> infinity` limit,
/// so `wasserstein_distance >= bottleneck_distance` always holds.
///
/// The optimal assignment is solved exactly by the Hungarian algorithm on the
/// same `(n + m)` square cost matrix, so this is O((n + m)^3).
///
/// ponytail: Hungarian is cubic. Bar count is bounded by `PersistenceConfig`'s
/// `max_points`, which `persistence.rs` rejects past — 512 in the widest preset
/// (`h0_only`), 128 in `h1_dense` — so `(n + m)` stays in the hundreds. Trigger:
/// any preset's `max_points` raised past 2048, or a caller assembling diagrams
/// outside those caps. Then swap in an auction or Sinkhorn solver — the tests
/// here pin the answer either way.
pub fn wasserstein_distance(
    left: &PersistenceDiagram,
    right: &PersistenceDiagram,
    dimension: usize,
    p: f64,
) -> f64 {
    assert!(p >= 1.0, "p-Wasserstein requires p >= 1, got {p}");

    let a = finite_bars(left, dimension);
    let b = finite_bars(right, dimension);
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }

    let base = cost_matrix(&a, &b);
    let powered: Vec<Vec<f64>> = base
        .iter()
        .map(|row| row.iter().map(|&c| pow(c, p)).collect())
        .collect();

    let total = hungarian_min_cost(&powered);
    pow(total, 1.0 / p)
}

/// Kuhn's augmenting-path matching restricted to edges with `cost <= threshold`.
/// Returns whether a perfect matching exists.
fn perfect_matching_exists(cost: &[Vec<f64>], threshold: f64) -> bool {
    let size = cost.len();
    let mut match_right: Vec<Option<usize>> = vec![None; size];

    for left in 0..size {
        let mut seen = vec![false; size];
        if !augment(left, cost, threshold, &mut seen, &mut match_right) {
            return false;
        }
    }
    true
}

fn augment(
    left: usize,
    cost: &[Vec<f64>],
    threshold: f64,
    seen: &mut [bool],
    match_right: &mut [Option<usize>],
) -> bool {
    for right in 0..cost.len() {
        if seen[right] || cost[left][right] > threshold {
            continue;
        }
        seen[right] = true;
        let free = match match_right[right] {
            None => true,
            Some(other) => augment(other, cost, threshold, seen, match_right),
        };
        if free {
            match_right[right] = Some(left);
            return true;
        }
    }
    false
}

/// Hungarian algorithm (Jonker-Volgenant potentials form) on a square matrix.
/// Returns the minimum total cost of a perfect matching.
fn hungarian_min_cost(cost: &[Vec<f64>]) -> f64 {
    let n = cost.len();
    // 1-indexed potentials and assignment, the classic formulation: `u` and `v`
    // are the dual potentials, `assignment[j]` is the row matched to column j.
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut assignment = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        assignment[0] = i;
        let mut j0 = 0usize;
        let mut min_cost = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];

        loop {
            used[j0] = true;
            let i0 = assignment[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;

            for j in 1..=n {
                if used[j] {
                    continue;
                }
                let current = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if current < min_cost[j] {
                    min_cost[j] = current;
                    way[j] = j0;
                }
                if min_cost[j] < delta {
                    delta = min_cost[j];
                    j1 = j;
                }
            }

            for j in 0..=n {
                if used[j] {
                    u[assignment[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_cost[j] -= delta;
                }
            }

            j0 = j1;
            if assignment[j0] == 0 {
                break;
            }
        }

        // Walk the alternating path back, re-assigning as we go.
        loop {
            let j1 = way[j0];
            assignment[j0] = assignment[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    (1..=n).map(|j| cost[assignment[j] - 1][j - 1]).sum()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Persistence landscapes
// ═══════════════════════════════════════════════════════════════════════════════

/// Sampling grid for [`persistence_landscape`].
#[derive(Debug, Clone, Copy)]
pub struct LandscapeConfig {
    /// How many landscape levels to return.
    pub levels: usize,
    /// Number of evenly spaced samples across `[min_t, max_t]`, endpoints included.
    pub resolution: usize,
    pub min_t: f64,
    pub max_t: f64,
}

impl Default for LandscapeConfig {
    fn default() -> Self {
        Self {
            levels: 3,
            resolution: 64,
            min_t: 0.0,
            max_t: 1.0,
        }
    }
}

/// Persistence landscape: `levels` functions sampled on a fixed grid.
///
/// Each bar `[b, d]` contributes the tent function
///
/// ```text
/// lambda_bar(t) = max(0, min(t - b, d - t))
/// ```
///
/// and level `k` at each `t` is the `k`-th largest tent value there. Levels are
/// pointwise non-increasing by construction.
///
/// The result is `levels` rows of `resolution` samples, flattened per level. It
/// is 1-Lipschitz in the bottleneck distance: two diagrams within `eps` produce
/// landscapes within `eps` in the sup norm.
pub fn persistence_landscape(
    diagram: &PersistenceDiagram,
    dimension: usize,
    config: LandscapeConfig,
) -> Vec<Vec<f64>> {
    let bars = finite_bars(diagram, dimension);
    let mut levels = vec![vec![0.0f64; config.resolution]; config.levels];
    if config.resolution == 0 || config.levels == 0 {
        return levels;
    }

    let step = if config.resolution > 1 {
        (config.max_t - config.min_t) / (config.resolution - 1) as f64
    } else {
        0.0
    };

    let mut tents = Vec::with_capacity(bars.len());
    for sample in 0..config.resolution {
        let t = config.min_t + step * sample as f64;

        tents.clear();
        for &(birth, death) in &bars {
            let value = fmax(0.0, fmin(t - birth, death - t));
            if value > 0.0 {
                tents.push(value);
            }
        }
        // Descending: level 1 is the largest tent value at this t.
        tents.sort_by(|a, b| b.total_cmp(a));

        for (level, row) in levels.iter_mut().enumerate() {
            row[sample] = tents.get(level).copied().unwrap_or(0.0);
        }
    }
    levels
}

fn fmin(a: f64, b: f64) -> f64 {
    if a < b {
        a
    } else {
        b
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Persistence images
// ═══════════════════════════════════════════════════════════════════════════════

/// Grid and kernel for [`persistence_image`].
#[derive(Debug, Clone, Copy)]
pub struct ImageConfig {
    pub width: usize,
    pub height: usize,
    /// Standard deviation of the Gaussian deposited at each bar.
    pub sigma: f64,
    /// Horizontal window, in birth coordinates.
    pub min_birth: f64,
    pub max_birth: f64,
    /// Vertical window, in persistence (`death - birth`) coordinates, from 0.
    pub max_persistence: f64,
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            width: 32,
            height: 32,
            sigma: 0.1,
            min_birth: 0.0,
            max_birth: 1.0,
            max_persistence: 1.0,
        }
    }
}

/// Persistence image: a fixed-size raster of the diagram in birth-persistence
/// coordinates, row-major with `height * width` entries.
///
/// Each bar is transformed to `(birth, death - birth)`, weighted linearly by its
/// persistence, and smeared with an isotropic Gaussian of width `sigma`. The
/// linear weight is what suppresses near-diagonal noise: without it the image is
/// dominated by the many short bars any sampling produces.
///
/// Row 0 is the lowest persistence band. Values are non-negative.
pub fn persistence_image(
    diagram: &PersistenceDiagram,
    dimension: usize,
    config: ImageConfig,
) -> Vec<f64> {
    let mut image = vec![0.0f64; config.width * config.height];
    if config.width == 0 || config.height == 0 || !(config.sigma > 0.0) {
        return image;
    }

    let bars = finite_bars(diagram, dimension);
    let birth_step = (config.max_birth - config.min_birth) / config.width as f64;
    let persistence_step = config.max_persistence / config.height as f64;
    let two_sigma_sq = 2.0 * config.sigma * config.sigma;
    // Normalising so the deposited mass is comparable across sigma values.
    let norm = 1.0 / (two_sigma_sq * core::f64::consts::PI);

    for &(birth, death) in &bars {
        let persistence = death - birth;
        if persistence <= 0.0 {
            continue;
        }
        // Linear weighting in persistence: the standard choice, and the one that
        // makes the image stable rather than noise-dominated.
        let weight = persistence;

        for row in 0..config.height {
            let py = persistence_step * (row as f64 + 0.5);
            for col in 0..config.width {
                let px = config.min_birth + birth_step * (col as f64 + 0.5);
                let dx = px - birth;
                let dy = py - persistence;
                let value = weight * norm * exp(-(dx * dx + dy * dy) / two_sigma_sq);
                image[row * config.width + col] += value;
            }
        }
    }
    image
}

/// Total persistence: the sum of bar lengths in one dimension. The cheapest
/// scalar summary, and a useful sanity signal next to the vectorizations above.
pub fn total_persistence(diagram: &PersistenceDiagram, dimension: usize) -> f64 {
    finite_bars(diagram, dimension)
        .iter()
        .map(|&(birth, death)| death - birth)
        .sum()
}

/// Persistent entropy: the Shannon entropy of the normalised bar lengths.
///
/// Low entropy means one bar dominates (a single clear feature); high entropy
/// means persistence is spread across many bars (noise, or genuinely multiscale
/// structure). Returns 0 for an empty or zero-length diagram.
pub fn persistent_entropy(diagram: &PersistenceDiagram, dimension: usize) -> f64 {
    let bars = finite_bars(diagram, dimension);
    let total: f64 = bars.iter().map(|&(b, d)| d - b).sum();
    if total <= 0.0 {
        return 0.0;
    }

    -bars
        .iter()
        .map(|&(birth, death)| (death - birth) / total)
        .filter(|&p| p > 0.0)
        .map(|p| p * libm::log(p))
        .sum::<f64>()
}

/// Euclidean norm of a landscape, flattened across levels. A convenient scalar
/// feature, and the norm the landscape stability bound is usually quoted in.
pub fn landscape_norm(levels: &[Vec<f64>]) -> f64 {
    sqrt(levels.iter().flatten().map(|v| v * v).sum::<f64>())
}
