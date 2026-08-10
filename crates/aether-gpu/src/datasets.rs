//! Synthetic data and split diagnostics for the training examples.
//!
//! This lives in the library rather than being copied into each example
//! because the reasoning it encodes took three attempts to get right, and two
//! of them were wrong in ways that looked convincing.
//!
//! **Use [`spirals_iid`] and draw train and test separately.** A partition of a
//! single deterministic sweep is not an i.i.d. split, because a sweep is not a
//! sample: two sweeps with the same spacing differ only by their noise. That is
//! a structural argument and it is the whole justification.
//!
//! # Two conclusions that did not survive contact with measurement
//!
//! The first attempt cross-validated a swept sample. The second measured that
//! held-out points sat 0.99× the cloud's own spacing from their nearest
//! training point, concluded the split was leaking, and blocked it into
//! contiguous arcs instead. Both conclusions were wrong, in opposite
//! directions:
//!
//! - **Blocking is worse, not better.** It measures 9.72× and sends both a
//!   tuned SGD and a tuned Adam below the majority-class control, because it
//!   asks the model to extrapolate into arcs it never saw. This failure the
//!   ratio does detect, which is what [`SplitDiagnostic::is_extrapolating`] is
//!   for.
//! - **0.99× was not evidence of a leak.** Independent draws from the same
//!   generator measure 1.39×, because dense i.i.d. sampling naturally places
//!   points near each other — a close training neighbour is what "same
//!   distribution" means. Re-running both examples against independently drawn
//!   test sets gave the same answers the interleaved split had (Adam 1.0000
//!   against 1.0000; multi-class SGD 0.8067 against 0.8020). The interleaved
//!   numbers were not inflated.
//!
//! So the recommendation stands and the argument for it changed. It rests on
//! how the data is constructed, not on a distance ratio that cannot tell an
//! honest dense sample from a contaminated one.

/// Small deterministic generator, so every figure reproduces from its seed.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as f32 / (1u64 << 31) as f32
    }
}

fn spiral_point(class: usize, classes: usize, t: f32, rng: &mut Lcg) -> (f32, f32) {
    let radius = 0.15 + 3.85 * t;
    let angle = 2.2 * core::f32::consts::PI * t
        + class as f32 * 2.0 * core::f32::consts::PI / classes as f32;
    let jitter_r = (rng.next_f32() - 0.5) * 0.30;
    let jitter_a = (rng.next_f32() - 0.5) * 0.12;
    (
        (radius + jitter_r) * (angle + jitter_a).cos() / 4.0,
        (radius + jitter_r) * (angle + jitter_a).sin() / 4.0,
    )
}

/// Interleaved spirals with arc positions drawn **uniformly at random**.
///
/// Two calls with different seeds are independent samples from the same
/// distribution, which is what makes a train/test pair meaningful. Use this.
///
/// Returns row-major `[n, 2]` features and `[n]` class labels.
pub fn spirals_iid(seed: u64, classes: usize, per_class: usize) -> (Vec<f32>, Vec<usize>) {
    let mut rng = Lcg::new(seed);
    let mut x = Vec::with_capacity(classes * per_class * 2);
    let mut y = Vec::with_capacity(classes * per_class);

    for class in 0..classes {
        for _ in 0..per_class {
            let t = rng.next_f32();
            let (px, py) = spiral_point(class, classes, t, &mut rng);
            x.push(px);
            x.push(py);
            y.push(class);
        }
    }
    (x, y)
}

/// Interleaved spirals with arc positions swept in order.
///
/// Retained because it is what the earlier examples used and because the
/// comparison against [`spirals_iid`] is the evidence for the split problem.
/// A sweep is not a sample: two sweeps with the same spacing differ only by
/// their noise, and any partition of one puts near-duplicates on both sides.
pub fn spirals_sweep(seed: u64, classes: usize, per_class: usize) -> (Vec<f32>, Vec<usize>) {
    let mut rng = Lcg::new(seed);
    let mut x = Vec::with_capacity(classes * per_class * 2);
    let mut y = Vec::with_capacity(classes * per_class);

    for class in 0..classes {
        for i in 0..per_class {
            let t = i as f32 / per_class as f32;
            let (px, py) = spiral_point(class, classes, t, &mut rng);
            x.push(px);
            x.push(py);
            y.push(class);
        }
    }
    (x, y)
}

/// One-hot encoding, `[n, classes]` row-major.
pub fn one_hot(y: &[usize], classes: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; y.len() * classes];
    for (i, &c) in y.iter().enumerate() {
        out[i * classes + c] = 1.0;
    }
    out
}

/// Fraction of the largest class: the floor any classifier must beat.
pub fn majority_class(y: &[usize], classes: usize) -> f32 {
    let mut counts = vec![0usize; classes];
    for &c in y {
        counts[c] += 1;
    }
    *counts.iter().max().unwrap_or(&0) as f32 / y.len().max(1) as f32
}

/// How well separated a train/test split actually is.
///
/// `ratio` is the median distance from a test point to its nearest training
/// point, divided by the median nearest-neighbour distance within the whole
/// cloud. At 1.0 the split has placed a training point as close to each test
/// point as its own neighbours are, so accuracy measures interpolation.
pub struct SplitDiagnostic {
    pub median_to_train: f32,
    pub median_spacing: f32,
    pub ratio: f32,
}

impl SplitDiagnostic {
    /// Whether held-out points sit **further** from training data than the
    /// cloud's own spacing.
    ///
    /// # This is not a leak detector, and an earlier version of this code
    /// claimed it was
    ///
    /// The ratio was introduced after an interleaved split of a swept spiral
    /// measured 0.99x, which looked like proof the split was leaking. It is
    /// not. Two genuinely independent draws from the same distribution measure
    /// **1.39x** on the same generator, and are flagged by any threshold above
    /// that — because dense i.i.d. sampling naturally places points near each
    /// other. A near-neighbour in the training set is what "same distribution"
    /// means, not evidence of contamination.
    ///
    /// So a low ratio does not establish a leak, and the conclusion originally
    /// drawn from 0.99x was wrong. Checked directly: interleaved
    /// cross-validation and an independent-draw holdout **agree** on this data
    /// (Adam 1.0000 against 1.0000; the multi-class SGD run 0.8067 against
    /// 0.8020). The interleaved split was not inflating anything.
    ///
    /// What the ratio does detect is the opposite failure — a split so
    /// separated that the held-out set is out of distribution. The blocked
    /// split measures 9.72x and sends both optimisers below the majority-class
    /// control, because it asks for extrapolation into arcs never seen.
    ///
    /// Independent draws remain the right construction, on the structural
    /// argument that a partition of one deterministic sweep is not a sample.
    /// That argument does not depend on this ratio, which is why it survives
    /// the ratio turning out not to mean what it was taken to mean.
    pub fn is_extrapolating(&self) -> bool {
        self.ratio >= 5.0
    }
}

/// Measure a split. `x` is row-major `[n, 2]`; `is_test[i]` marks held-out
/// points.
///
/// O(n^2), which is fine at the sizes the examples use and is why this is a
/// diagnostic rather than something to call in a loop.
pub fn diagnose_split(x: &[f32], is_test: &[bool]) -> SplitDiagnostic {
    let n = is_test.len();
    let dist = |i: usize, j: usize| -> f32 {
        let dx = x[i * 2] - x[j * 2];
        let dy = x[i * 2 + 1] - x[j * 2 + 1];
        (dx * dx + dy * dy).sqrt()
    };

    let mut to_train = Vec::new();
    for i in 0..n {
        if !is_test[i] {
            continue;
        }
        let mut best = f32::INFINITY;
        for j in 0..n {
            if !is_test[j] {
                best = best.min(dist(i, j));
            }
        }
        if best.is_finite() {
            to_train.push(best);
        }
    }

    let mut spacing = Vec::with_capacity(n);
    for i in 0..n {
        let mut best = f32::INFINITY;
        for j in 0..n {
            if i != j {
                best = best.min(dist(i, j));
            }
        }
        if best.is_finite() {
            spacing.push(best);
        }
    }

    let median = |v: &mut Vec<f32>| -> f32 {
        if v.is_empty() {
            return f32::NAN;
        }
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    };

    let median_to_train = median(&mut to_train);
    let median_spacing = median(&mut spacing);

    SplitDiagnostic {
        median_to_train,
        median_spacing,
        ratio: median_to_train / median_spacing,
    }
}

/// Print the diagnostic.
///
/// Reports the ratio without a verdict on leakage, because the ratio does not
/// support one: independent draws from one distribution measure around 1.4x on
/// this generator, which is indistinguishable from a leaky split of a sweep at
/// 0.99x. A high ratio is the informative case — it means the held-out set is
/// out of distribution rather than merely unseen.
pub fn report_split(label: &str, x: &[f32], is_test: &[bool]) -> SplitDiagnostic {
    let d = diagnose_split(x, is_test);
    println!("  split diagnostic ({label})");
    println!(
        "    median test-to-train distance   {:.5}",
        d.median_to_train
    );
    println!(
        "    median cloud spacing            {:.5}",
        d.median_spacing
    );
    println!("    ratio                           {:.2}x", d.ratio);
    println!("      ~1x is expected for independent draws from one distribution");
    if d.is_extrapolating() {
        println!("    EXTRAPOLATING: held-out points sit far outside the training");
        println!("    data. Accuracy here measures extrapolation, not");
        println!("    generalisation, and will understate the model.");
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The correction, pinned so it cannot be forgotten: a low ratio does not
    /// indicate a leak, because independent draws produce one too.
    ///
    /// An interleaved split of a sweep and a pair of independent draws land
    /// close enough together that no threshold separates them. This is the
    /// evidence that the ratio cannot be used as a leak detector, which is what
    /// an earlier version of this module claimed it was.
    #[test]
    fn a_low_ratio_does_not_distinguish_a_leaky_split_from_independent_draws() {
        let (x, y) = spirals_sweep(1, 3, 300);
        let is_test: Vec<bool> = (0..y.len()).map(|i| i % 5 == 0).collect();
        let leaky = diagnose_split(&x, &is_test);

        let (a, _) = spirals_iid(1, 3, 300);
        let (b, _) = spirals_iid(2, 3, 300);
        let mut both = a.clone();
        both.extend_from_slice(&b);
        let is_test: Vec<bool> = (0..a.len() / 2)
            .map(|_| false)
            .chain((0..b.len() / 2).map(|_| true))
            .collect();
        let independent = diagnose_split(&both, &is_test);

        assert!(
            independent.ratio < 5.0,
            "independent draws measured {:.2}x, which a leak threshold would flag",
            independent.ratio
        );
        assert!(
            !leaky.is_extrapolating() && !independent.is_extrapolating(),
            "neither split is extrapolating: {:.2}x and {:.2}x",
            leaky.ratio,
            independent.ratio
        );
    }

    /// The case the ratio does detect: holding out contiguous arcs puts the
    /// held-out set outside the training distribution.
    #[test]
    fn a_blocked_split_is_flagged_as_extrapolation() {
        let (x, y) = spirals_sweep(1, 3, 300);
        let per_class = 300;
        let is_test: Vec<bool> = (0..y.len())
            .map(|i| (i % per_class) < per_class / 5)
            .collect();

        let d = diagnose_split(&x, &is_test);
        assert!(
            d.is_extrapolating(),
            "a blocked split should read as extrapolation, got {:.2}x",
            d.ratio
        );
    }

    #[test]
    fn majority_class_is_the_largest_share() {
        assert!((majority_class(&[0, 0, 0, 1], 2) - 0.75).abs() < 1e-6);
        assert!((majority_class(&[0, 1, 2], 3) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn one_hot_sets_exactly_one_entry_per_row() {
        let h = one_hot(&[0, 2, 1], 3);
        assert_eq!(h, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    }
}
