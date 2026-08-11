//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Data Loaders
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Efficient data loading with batching and shuffling.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
// Aether-Lang — invented by Teerth Sharma
// https://github.com/teerthsharma/Aether-Lang
// Copyright (c) 2026 Teerth Sharma. All Rights Reserved.
// ═══════════════════════════════════════════════════════════════════════════════
//

#![allow(dead_code)]

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use super::tensor::Tensor;

/// Data Loader
#[derive(Debug, Clone)]
pub struct DataLoader {
    pub features: Vec<Tensor>,
    pub targets: Vec<Tensor>,
    pub batch_size: usize,
    pub shuffle: bool,
}

impl DataLoader {
    /// Create new DataLoader
    pub fn new(
        features: Vec<Tensor>,
        targets: Vec<Tensor>,
        batch_size: usize,
        shuffle: bool,
    ) -> Self {
        assert_eq!(
            features.len(),
            targets.len(),
            "Features and targets must have same length"
        );
        Self {
            features,
            targets,
            batch_size,
            shuffle,
        }
    }

    /// Convert raw slices to DataLoader
    pub fn from_slice(x: &[Tensor], y: &[Tensor], batch_size: usize, shuffle: bool) -> Self {
        Self {
            features: x.to_vec(),
            targets: y.to_vec(),
            batch_size,
            shuffle,
        }
    }

    /// Iterate over batches
    pub fn iter(&self) -> BatchIterator<'_> {
        let n = self.features.len();
        let mut indices: Vec<usize> = (0..n).collect();

        if self.shuffle {
            // Simple Linear Congruential Generator for no_std compatibility
            // X_{n+1} = (aX_n + c) % m
            let mut rng = 42u64; // Should ideally take a seed
            for i in (1..n).rev() {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                // High bits, not low. A power-of-two-modulus LCG has period
                // 2^(k+1) in bit k, so `rng as usize` hands the modulo the
                // worst bits it has: bit 0 of this sequence is
                // 1010101010101010101, and the low two bits cycle 3,0,1,2. The
                // final swaps of a Fisher-Yates are the ones with the smallest
                // bounds, so they were driven entirely by those bits -- at
                // i = 1 the choice was `rng % 2`, which alternates rather than
                // chooses. `datasets::Lcg` in aether-gpu already takes `>> 33`
                // for this reason.
                let j = ((rng >> 33) as usize) % (i + 1);
                indices.swap(i, j);
            }
        }

        BatchIterator {
            loader: self,
            indices,
            current_idx: 0,
        }
    }
}

/// Iterator over batches
pub struct BatchIterator<'a> {
    loader: &'a DataLoader,
    indices: Vec<usize>,
    current_idx: usize,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = (Vec<Tensor>, Vec<Tensor>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.loader.features.len() {
            return None;
        }

        let start = self.current_idx;
        let end = (start + self.loader.batch_size).min(self.loader.features.len());
        self.current_idx = end;

        let mut batch_x = Vec::with_capacity(end - start);
        let mut batch_y = Vec::with_capacity(end - start);

        for i in start..end {
            let idx = self.indices[i];
            batch_x.push(self.loader.features[idx].clone());
            batch_y.push(self.loader.targets[idx].clone());
        }

        Some((batch_x, batch_y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataloader() {
        let x = vec![
            Tensor::zeros(&[1]),
            Tensor::zeros(&[1]),
            Tensor::zeros(&[1]),
            Tensor::zeros(&[1]),
            Tensor::zeros(&[1]),
        ];
        let y = x.clone();

        let loader = DataLoader::new(x, y, 2, false);
        let mut iter = loader.iter();

        let batch1 = iter.next();
        assert!(batch1.is_some());
        assert_eq!(batch1.unwrap().0.len(), 2);

        let batch2 = iter.next();
        assert!(batch2.is_some());
        assert_eq!(batch2.unwrap().0.len(), 2);

        let batch3 = iter.next(); // Last batch of 1
        assert!(batch3.is_some());
        assert_eq!(batch3.unwrap().0.len(), 1);

        assert!(iter.next().is_none());
    }

    /// One sample per index, so a batch reveals which samples it holds.
    ///
    /// The test above builds five `Tensor::zeros`, which are indistinguishable.
    /// It can therefore check how many samples come back and nothing about
    /// which — a loader that returned the first sample five times, dropped the
    /// last, or paired feature 3 with target 1 would satisfy every assertion in
    /// it. These use distinct values so the answers differ.
    fn labelled(n: usize) -> (Vec<Tensor>, Vec<Tensor>) {
        let features = (0..n)
            .map(|i| Tensor::new(&[i as f64], &[1]))
            .collect::<Vec<_>>();
        // Targets are offset so a feature paired with the wrong target is
        // visible: sample i is (i, i + 100).
        let targets = (0..n)
            .map(|i| Tensor::new(&[i as f64 + 100.0], &[1]))
            .collect::<Vec<_>>();
        (features, targets)
    }

    fn drain(loader: &DataLoader) -> Vec<(f64, f64)> {
        let mut seen = Vec::new();
        for (xs, ys) in loader.iter() {
            assert_eq!(
                xs.len(),
                ys.len(),
                "a batch returned {} features against {} targets",
                xs.len(),
                ys.len()
            );
            for (x, y) in xs.iter().zip(&ys) {
                seen.push((x.get(&[0]), y.get(&[0])));
            }
        }
        seen
    }

    /// Every sample exactly once, shuffled or not.
    ///
    /// The property that matters for training: a loader that drops a sample
    /// trains on less data than it reports, and one that repeats a sample
    /// weights it twice. Neither changes any batch's length, so neither is
    /// visible to a size check.
    #[test]
    fn every_sample_appears_exactly_once() {
        for shuffle in [false, true] {
            for (n, batch) in [(5usize, 2usize), (6, 3), (1, 4), (7, 1)] {
                let (x, y) = labelled(n);
                let loader = DataLoader::new(x, y, batch, shuffle);

                let mut ids: Vec<f64> = drain(&loader).iter().map(|(f, _)| *f).collect();
                assert_eq!(
                    ids.len(),
                    n,
                    "n={n} batch={batch} shuffle={shuffle}: got {} samples",
                    ids.len()
                );

                ids.sort_by(f64::total_cmp);
                let expected: Vec<f64> = (0..n).map(|i| i as f64).collect();
                assert_eq!(
                    ids, expected,
                    "n={n} batch={batch} shuffle={shuffle}: the samples returned are not                      the samples given, so one was dropped, repeated, or invented"
                );
            }
        }
    }

    /// A feature must arrive with its own target.
    ///
    /// Shuffling features and targets through separate index sequences is the
    /// defect this guards: every batch is the right size, every sample appears
    /// once, and the model learns a permuted labelling. Training would converge
    /// to something and be wrong, which is the hardest kind of wrong to notice.
    #[test]
    fn shuffling_keeps_each_feature_with_its_own_target() {
        let (x, y) = labelled(9);
        let loader = DataLoader::new(x, y, 4, true);

        for (feature, target) in drain(&loader) {
            assert_eq!(
                target,
                feature + 100.0,
                "feature {feature} came back paired with target {target}, which                  belongs to sample {}",
                target - 100.0
            );
        }
    }

    /// A batch larger than the data yields one short batch, not an empty one.
    #[test]
    fn a_batch_larger_than_the_dataset_returns_everything_once() {
        let (x, y) = labelled(3);
        let loader = DataLoader::new(x, y, 10, false);
        let mut it = loader.iter();

        let (xs, ys) = it
            .next()
            .expect("a loader with three samples yielded no batch");
        assert_eq!((xs.len(), ys.len()), (3, 3));
        assert!(
            it.next().is_none(),
            "a second batch appeared after all samples were returned"
        );
    }

    /// The shuffle is the same every time, and that is a limitation not a feature.
    ///
    /// `iter` seeds its generator with a literal 42, so two epochs over one
    /// loader see the same order — which for a training loop means the shuffle
    /// stops being a shuffle after the first pass. Pinned here so the behaviour
    /// is recorded rather than discovered, and so that giving `DataLoader` a seed
    /// is a change that fails this test and has to be acknowledged.
    #[test]
    fn the_shuffle_repeats_because_its_seed_is_a_literal() {
        let (x, y) = labelled(8);
        let loader = DataLoader::new(x, y, 3, true);

        let first = drain(&loader);
        let second = drain(&loader);
        assert_eq!(
            first, second,
            "the two passes differ, so the seed is no longer fixed — if that was              deliberate this test should be replaced by one asserting the seed is              respected"
        );

        let identity: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let order: Vec<f64> = first.iter().map(|(f, _)| *f).collect();
        assert_ne!(
            order, identity,
            "shuffle=true returned the samples in input order"
        );
    }

    /// The shuffle must draw from the generator's high bits.
    ///
    /// `iter` seeds a literal, so there is one permutation per dataset size and
    /// no distribution to sample — which is why this varies `n` instead. The
    /// number of draws is `n - 1`, so consecutive sizes differ by one step of the
    /// generator, and a defect in *which bits* the modulo sees shows up as
    /// structure across `n`.
    ///
    /// Reading `rng as usize` takes the low bits, and a power-of-two-modulus LCG
    /// has period 2^(k+1) in bit k — bit 0 alternates. The last swap of a
    /// Fisher-Yates has bound 2, so under that defect it is decided by an
    /// alternating bit, and the count of fixed points alternates with it: for
    /// n = 5..21 it reads 0 1 0 3 0 1 0 2 0 1 0 2 0 2 0, a zero at every other
    /// size. Drawing from `rng >> 33` produces no such pattern.
    ///
    /// The statistic is the number of adjacent sizes that *both* leave a point
    /// fixed. Alternating zeros make that nearly impossible; over n = 2..25 the
    /// low-bit version scores 3 and the high-bit version 15. The threshold is 8,
    /// far from both — this separates two regimes rather than pinning a
    /// permutation, so it survives a change of seed while still failing if the
    /// shift is dropped.
    #[test]
    fn the_shuffle_draws_from_the_high_bits() {
        let fixed_points = |n: usize| -> usize {
            let (x, y) = labelled(n);
            let loader = DataLoader::new(x, y, n, true);
            drain(&loader)
                .iter()
                .enumerate()
                .filter(|(slot, (feature, _))| *feature as usize == *slot)
                .count()
        };

        let counts: Vec<usize> = (2..26).map(fixed_points).collect();
        let adjacent = counts.windows(2).filter(|w| w[0] > 0 && w[1] > 0).count();

        assert!(
            adjacent >= 8,
            "only {adjacent} adjacent dataset sizes both leave a point fixed,              against 15 when the shuffle reads `rng >> 33` and 3 when it reads              `rng as usize`. The low bits of this generator alternate, so a              modulo that sees them makes the last swap follow a pattern rather              than a draw. Counts were: {counts:?}"
        );
    }
}
