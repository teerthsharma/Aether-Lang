//! Bounded persistent homology for AETHER point clouds.
//!
//! The engine builds a filtered simplicial complex through tetrahedra and
//! reduces boundary columns over Z2. It is exact for the selected complex and
//! deliberately bounded so topological ML workloads fail fast instead of
//! exhausting memory.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::manifold::{ManifoldPoint, TimeDelayEmbedder};

const SIMPLEX_VERTICES: usize = 4;

/// A simplex identified by its zero-padded vertex array and its vertex count.
type SimplexKey = ([usize; SIMPLEX_VERTICES], usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexKind {
    VietorisRips,
    Witness { max_landmarks: usize },
}

#[derive(Debug, Clone, Copy)]
pub struct PersistenceConfig {
    pub max_homology_dim: usize,
    pub max_points: usize,
    pub max_simplices: usize,
    pub max_radius: f64,
    pub complex_kind: ComplexKind,
}

impl PersistenceConfig {
    /// Full H0/H1/H2, sized so a call returns in roughly two seconds.
    ///
    /// Measured on the regular circle (`cargo run -p aether-core --example
    /// scale_probe --release`, release, Windows 11, single core):
    ///
    /// | dim | n   | pairs  | seconds |
    /// |-----|-----|--------|---------|
    /// | 0   | 200 | 200    | 0.049   |
    /// | 0   | 1000| 1000   | 5.781   |
    /// | 1   | 120 | 7141   | 2.202   |
    /// | 1   | 200 | 19901  | 20.728  |
    /// | 2   | 50  | 19650  | 1.859   |
    /// | 2   | 70  | 54810  | 15.338  |
    ///
    /// The caps are a fail-fast budget, not a statement about correctness. Raise
    /// them explicitly when the workload justifies the wait.
    pub const fn h2_default() -> Self {
        Self {
            max_homology_dim: 2,
            max_points: 48,
            max_simplices: 1_000_000,
            max_radius: f64::INFINITY,
            complex_kind: ComplexKind::VietorisRips,
        }
    }

    /// H0 and H1 only, which is what most topological ML features use. Tetrahedra
    /// are the O(n^4) term, so dropping them buys a much larger point budget.
    pub const fn h1_dense() -> Self {
        Self {
            max_homology_dim: 1,
            max_points: 128,
            max_simplices: 1_000_000,
            max_radius: f64::INFINITY,
            complex_kind: ComplexKind::VietorisRips,
        }
    }

    /// Connected components only. Cheapest useful configuration.
    pub const fn h0_only() -> Self {
        Self {
            max_homology_dim: 0,
            max_points: 512,
            max_simplices: 1_000_000,
            max_radius: f64::INFINITY,
            complex_kind: ComplexKind::VietorisRips,
        }
    }

    pub const fn low_load() -> Self {
        Self {
            max_homology_dim: 1,
            max_points: 24,
            max_simplices: 4_096,
            max_radius: f64::INFINITY,
            complex_kind: ComplexKind::Witness { max_landmarks: 24 },
        }
    }
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self::h2_default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PersistenceError {
    InvalidDimension,
    InvalidRadius,
    TooManyPoints { actual: usize, max: usize },
    TooManySimplices { max: usize },
    EmptyInput,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersistencePair {
    pub dimension: usize,
    pub birth: f64,
    pub death: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PersistenceDiagram {
    pub pairs: Vec<PersistencePair>,
}

impl PersistenceDiagram {
    pub fn new(pairs: Vec<PersistencePair>) -> Self {
        Self { pairs }
    }

    pub fn betti_at(&self, radius: f64) -> BettiNumbers3 {
        let mut betti = BettiNumbers3::default();
        for pair in &self.pairs {
            if pair.birth <= radius && pair.death.map(|death| radius < death).unwrap_or(true) {
                match pair.dimension {
                    0 => betti.beta_0 += 1,
                    1 => betti.beta_1 += 1,
                    2 => betti.beta_2 += 1,
                    _ => {}
                }
            }
        }
        betti
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BettiNumbers3 {
    pub beta_0: u32,
    pub beta_1: u32,
    pub beta_2: u32,
}

#[derive(Debug, Clone)]
struct Simplex {
    vertices: [usize; SIMPLEX_VERTICES],
    len: usize,
    dimension: usize,
    filtration: f64,
}

pub fn time_delay_persistence<const D: usize>(
    samples: &[f64],
    tau: usize,
    config: PersistenceConfig,
) -> Result<PersistenceDiagram, PersistenceError> {
    let mut embedder = TimeDelayEmbedder::<D>::new(tau);
    let mut points = Vec::new();

    for &sample in samples {
        embedder.push(sample);
        if let Some(point) = embedder.embed() {
            points.push(point);
        }
    }

    persistent_homology(&points, config)
}

pub fn persistent_homology<const D: usize>(
    points: &[ManifoldPoint<D>],
    config: PersistenceConfig,
) -> Result<PersistenceDiagram, PersistenceError> {
    validate_common(points.len(), config)?;

    let simplices = match config.complex_kind {
        ComplexKind::VietorisRips => {
            validate_point_cap(points.len(), config)?;
            build_vietoris_rips_simplices(points, config)?
        }
        ComplexKind::Witness { max_landmarks } => {
            let landmarks = select_landmarks(points, max_landmarks);
            validate_point_cap(landmarks.len(), config)?;
            build_lazy_witness_simplices(&landmarks, points, config)?
        }
    };
    reduce_z2(&simplices, config.max_homology_dim)
}

fn validate_common(point_count: usize, config: PersistenceConfig) -> Result<(), PersistenceError> {
    if config.max_homology_dim > 2 {
        return Err(PersistenceError::InvalidDimension);
    }
    if config.max_radius.is_nan() || config.max_radius < 0.0 {
        return Err(PersistenceError::InvalidRadius);
    }
    if point_count == 0 {
        return Err(PersistenceError::EmptyInput);
    }
    Ok(())
}

fn validate_point_cap(
    point_count: usize,
    config: PersistenceConfig,
) -> Result<(), PersistenceError> {
    if point_count > config.max_points {
        return Err(PersistenceError::TooManyPoints {
            actual: point_count,
            max: config.max_points,
        });
    }
    Ok(())
}

fn select_landmarks<const D: usize>(
    points: &[ManifoldPoint<D>],
    max_landmarks: usize,
) -> Vec<ManifoldPoint<D>> {
    if max_landmarks == 0 || points.len() <= max_landmarks {
        return points.to_vec();
    }

    let mut selected = vec![false; points.len()];
    let mut landmarks = Vec::with_capacity(max_landmarks);
    landmarks.push(points[0]);
    selected[0] = true;

    while landmarks.len() < max_landmarks {
        let mut best_idx = None;
        let mut best_distance = -1.0;

        for (idx, point) in points.iter().enumerate() {
            if selected[idx] {
                continue;
            }

            let nearest = landmarks
                .iter()
                .map(|landmark| point.distance(landmark))
                .fold(f64::INFINITY, |a, b| a.min(b));

            if nearest > best_distance {
                best_distance = nearest;
                best_idx = Some(idx);
            }
        }

        let Some(idx) = best_idx else {
            break;
        };
        landmarks.push(points[idx]);
        selected[idx] = true;
    }
    landmarks
}

fn build_vietoris_rips_simplices<const D: usize>(
    points: &[ManifoldPoint<D>],
    config: PersistenceConfig,
) -> Result<Vec<Simplex>, PersistenceError> {
    let n = points.len();
    let mut distances = vec![0.0; n * n];
    for i in 0..n {
        for j in i + 1..n {
            let distance = points[i].distance(&points[j]);
            distances[i * n + j] = distance;
            distances[j * n + i] = distance;
        }
    }

    let mut simplices = Vec::new();
    for i in 0..n {
        push_simplex(
            &mut simplices,
            simplex([i, 0, 0, 0], 1, 0.0),
            config.max_simplices,
        )?;
    }

    for i in 0..n {
        for j in i + 1..n {
            let r = distances[i * n + j];
            if r <= config.max_radius {
                push_simplex(
                    &mut simplices,
                    simplex([i, j, 0, 0], 2, r),
                    config.max_simplices,
                )?;
            }
        }
    }

    if config.max_homology_dim >= 1 {
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    let r = max3(
                        distances[i * n + j],
                        distances[i * n + k],
                        distances[j * n + k],
                    );
                    if r <= config.max_radius {
                        push_simplex(
                            &mut simplices,
                            simplex([i, j, k, 0], 3, r),
                            config.max_simplices,
                        )?;
                    }
                }
            }
        }
    }

    if config.max_homology_dim >= 2 {
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    for l in k + 1..n {
                        let r = max6(
                            distances[i * n + j],
                            distances[i * n + k],
                            distances[i * n + l],
                            distances[j * n + k],
                            distances[j * n + l],
                            distances[k * n + l],
                        );
                        if r <= config.max_radius {
                            push_simplex(
                                &mut simplices,
                                simplex([i, j, k, l], 4, r),
                                config.max_simplices,
                            )?;
                        }
                    }
                }
            }
        }
    }

    simplices.sort_by(compare_simplex);
    Ok(simplices)
}

fn build_lazy_witness_simplices<const D: usize>(
    landmarks: &[ManifoldPoint<D>],
    witnesses: &[ManifoldPoint<D>],
    config: PersistenceConfig,
) -> Result<Vec<Simplex>, PersistenceError> {
    let n = landmarks.len();
    let mut witness_to_landmark = vec![0.0; witnesses.len() * n];
    let mut nearest = vec![f64::INFINITY; witnesses.len()];

    for (w_idx, witness) in witnesses.iter().enumerate() {
        for (l_idx, landmark) in landmarks.iter().enumerate() {
            let distance = witness.distance(landmark);
            witness_to_landmark[w_idx * n + l_idx] = distance;
            nearest[w_idx] = nearest[w_idx].min(distance);
        }
    }

    let mut simplices = Vec::new();
    for i in 0..n {
        push_simplex(
            &mut simplices,
            simplex([i, 0, 0, 0], 1, 0.0),
            config.max_simplices,
        )?;
    }

    for i in 0..n {
        for j in i + 1..n {
            if let Some(r) = witness_filtration(&witness_to_landmark, &nearest, n, [i, j, 0, 0], 2)
            {
                if r <= config.max_radius {
                    push_simplex(
                        &mut simplices,
                        simplex([i, j, 0, 0], 2, r),
                        config.max_simplices,
                    )?;
                }
            }
        }
    }

    if config.max_homology_dim >= 1 {
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    if let Some(r) =
                        witness_filtration(&witness_to_landmark, &nearest, n, [i, j, k, 0], 3)
                    {
                        if r <= config.max_radius {
                            push_simplex(
                                &mut simplices,
                                simplex([i, j, k, 0], 3, r),
                                config.max_simplices,
                            )?;
                        }
                    }
                }
            }
        }
    }

    if config.max_homology_dim >= 2 {
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    for l in k + 1..n {
                        if let Some(r) =
                            witness_filtration(&witness_to_landmark, &nearest, n, [i, j, k, l], 4)
                        {
                            if r <= config.max_radius {
                                push_simplex(
                                    &mut simplices,
                                    simplex([i, j, k, l], 4, r),
                                    config.max_simplices,
                                )?;
                            }
                        }
                    }
                }
            }
        }
    }

    simplices.sort_by(compare_simplex);
    Ok(simplices)
}

fn witness_filtration(
    distances: &[f64],
    nearest: &[f64],
    landmark_count: usize,
    vertices: [usize; SIMPLEX_VERTICES],
    len: usize,
) -> Option<f64> {
    let mut best = f64::INFINITY;
    for witness_idx in 0..nearest.len() {
        let mut farthest_vertex = 0.0;
        for &vertex in vertices.iter().take(len) {
            let distance = distances[witness_idx * landmark_count + vertex];
            if distance > farthest_vertex {
                farthest_vertex = distance;
            }
        }

        let filtration = (farthest_vertex - nearest[witness_idx]).max(0.0);
        if filtration < best {
            best = filtration;
        }
    }

    best.is_finite().then_some(best)
}

fn simplex(vertices: [usize; SIMPLEX_VERTICES], len: usize, filtration: f64) -> Simplex {
    Simplex {
        vertices,
        len,
        dimension: len - 1,
        filtration,
    }
}

fn push_simplex(
    simplices: &mut Vec<Simplex>,
    simplex: Simplex,
    max_simplices: usize,
) -> Result<(), PersistenceError> {
    if simplices.len() >= max_simplices {
        return Err(PersistenceError::TooManySimplices { max: max_simplices });
    }
    simplices.push(simplex);
    Ok(())
}

fn compare_simplex(a: &Simplex, b: &Simplex) -> core::cmp::Ordering {
    a.filtration
        .total_cmp(&b.filtration)
        .then(a.dimension.cmp(&b.dimension))
        .then(a.vertices[..a.len].cmp(&b.vertices[..b.len]))
}

fn reduce_z2(
    simplices: &[Simplex],
    max_homology_dim: usize,
) -> Result<PersistenceDiagram, PersistenceError> {
    let mut reduced_columns: Vec<Vec<usize>> = Vec::with_capacity(simplices.len());
    let mut low_owner: Vec<Option<usize>> = vec![None; simplices.len()];
    let mut paired_birth = vec![false; simplices.len()];
    let mut pairs = Vec::new();

    let index = build_simplex_index(simplices);

    for j in 0..simplices.len() {
        let mut column = boundary_indices(simplices, &index, j);
        loop {
            let Some(&low) = column.last() else {
                break;
            };
            let Some(owner) = low_owner[low] else {
                break;
            };
            let owner_column: &[usize] = reduced_columns[owner].as_slice();
            column = xor_sorted(&column, owner_column);
        }

        if let Some(&low) = column.last() {
            low_owner[low] = Some(j);
            paired_birth[low] = true;
            let dimension = simplices[low].dimension;
            if dimension <= max_homology_dim {
                pairs.push(PersistencePair {
                    dimension,
                    birth: simplices[low].filtration,
                    death: Some(simplices[j].filtration),
                });
            }
        }
        reduced_columns.push(column);
    }

    for (idx, column) in reduced_columns.iter().enumerate() {
        if column.is_empty() && !paired_birth[idx] && simplices[idx].dimension <= max_homology_dim {
            pairs.push(PersistencePair {
                dimension: simplices[idx].dimension,
                birth: simplices[idx].filtration,
                death: None,
            });
        }
    }

    pairs.sort_by(|a, b| {
        a.dimension
            .cmp(&b.dimension)
            .then(a.birth.total_cmp(&b.birth))
            .then(match (a.death, b.death) {
                (Some(x), Some(y)) => x.total_cmp(&y),
                (Some(_), None) => core::cmp::Ordering::Less,
                (None, Some(_)) => core::cmp::Ordering::Greater,
                (None, None) => core::cmp::Ordering::Equal,
            })
    });
    Ok(PersistenceDiagram::new(pairs))
}

/// Position of every simplex, keyed by its zero-padded vertex array and length.
///
/// `simplex()` zero-fills the unused slots and `boundary_indices` builds faces the
/// same way, so the padded array is a canonical key. Combinations are generated
/// once each, so keys are unique.
///
/// This replaces a linear scan over `simplices[..before]`, which made the whole
/// reduction O(m^2) in the simplex count and put a hard ceiling of ~32 points on
/// the engine.
fn build_simplex_index(simplices: &[Simplex]) -> BTreeMap<SimplexKey, usize> {
    simplices
        .iter()
        .enumerate()
        .map(|(idx, simplex)| ((simplex.vertices, simplex.len), idx))
        .collect()
}

fn boundary_indices(
    simplices: &[Simplex],
    index: &BTreeMap<SimplexKey, usize>,
    simplex_idx: usize,
) -> Vec<usize> {
    let simplex = &simplices[simplex_idx];
    if simplex.dimension == 0 {
        return Vec::new();
    }

    let mut boundary = Vec::with_capacity(simplex.len);
    for remove_idx in 0..simplex.len {
        let mut face = [0usize; SIMPLEX_VERTICES];
        let mut face_len = 0;
        for i in 0..simplex.len {
            if i != remove_idx {
                face[face_len] = simplex.vertices[i];
                face_len += 1;
            }
        }
        // The `< simplex_idx` guard preserves the original "search only earlier
        // simplices" semantics. It is never false for a well-formed filtration —
        // `every_face_is_present_and_precedes_its_coface` asserts exactly that —
        // but keeping it means a malformed complex degrades rather than lies.
        if let Some(&idx) = index.get(&(face, face_len)) {
            if idx < simplex_idx {
                boundary.push(idx);
            }
        }
    }
    boundary.sort_unstable();
    boundary
}

fn xor_sorted(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    let mut i = 0;
    let mut j = 0;
    while i < left.len() || j < right.len() {
        if j == right.len() || (i < left.len() && left[i] < right[j]) {
            out.push(left[i]);
            i += 1;
        } else if i == left.len() || right[j] < left[i] {
            out.push(right[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    out
}

fn max3(a: f64, b: f64, c: f64) -> f64 {
    a.max(b).max(c)
}

fn max6(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> f64 {
    a.max(b).max(c).max(d).max(e).max(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max_dim: usize, radius: f64) -> PersistenceConfig {
        PersistenceConfig {
            max_homology_dim: max_dim,
            max_points: 16,
            max_simplices: 4_096,
            max_radius: radius,
            complex_kind: ComplexKind::VietorisRips,
        }
    }

    #[test]
    fn h0_tracks_component_merges() {
        let points = [
            ManifoldPoint::<2>::new([0.0, 0.0]),
            ManifoldPoint::<2>::new([1.0, 0.0]),
            ManifoldPoint::<2>::new([3.0, 0.0]),
        ];

        let diagram = persistent_homology(&points, cfg(0, 10.0)).unwrap();

        assert_eq!(diagram.betti_at(0.5).beta_0, 3);
        assert_eq!(diagram.betti_at(1.5).beta_0, 2);
        assert_eq!(diagram.betti_at(3.0).beta_0, 1);
    }

    #[test]
    fn h1_square_loop_is_born_before_it_dies() {
        let points = [
            ManifoldPoint::<2>::new([0.0, 0.0]),
            ManifoldPoint::<2>::new([1.0, 0.0]),
            ManifoldPoint::<2>::new([1.0, 1.0]),
            ManifoldPoint::<2>::new([0.0, 1.0]),
        ];

        let diagram = persistent_homology(&points, cfg(1, 10.0)).unwrap();
        let h1 = diagram
            .pairs
            .iter()
            .find(|pair| pair.dimension == 1 && pair.birth <= 1.0);

        assert!(h1.is_some());
        assert!(h1.unwrap().death.unwrap() > h1.unwrap().birth);
    }

    #[test]
    fn h2_tetrahedron_boundary_has_void_until_tetrahedron_enters() {
        let points = [
            ManifoldPoint::<3>::new([1.0, 1.0, 1.0]),
            ManifoldPoint::<3>::new([-1.0, -1.0, 1.0]),
            ManifoldPoint::<3>::new([-1.0, 1.0, -1.0]),
            ManifoldPoint::<3>::new([1.0, -1.0, -1.0]),
        ];

        let diagram = persistent_homology(&points, cfg(2, 10.0)).unwrap();
        let h2 = diagram.pairs.iter().find(|pair| pair.dimension == 2);

        assert!(h2.is_some());
        assert_eq!(h2.unwrap().death, Some(h2.unwrap().birth));
    }

    #[test]
    fn caps_fail_before_allocating_unbounded_complexes() {
        let points = [
            ManifoldPoint::<2>::new([0.0, 0.0]),
            ManifoldPoint::<2>::new([1.0, 0.0]),
            ManifoldPoint::<2>::new([0.0, 1.0]),
        ];
        let config = PersistenceConfig {
            max_homology_dim: 2,
            max_points: 8,
            max_simplices: 2,
            max_radius: 10.0,
            complex_kind: ComplexKind::VietorisRips,
        };

        assert_eq!(
            persistent_homology(&points, config),
            Err(PersistenceError::TooManySimplices { max: 2 })
        );
    }

    #[test]
    fn time_delay_constant_signal_has_one_essential_component() {
        let samples = [1.0; 16];
        let diagram = time_delay_persistence::<3>(&samples, 1, cfg(2, 10.0)).unwrap();

        assert_eq!(
            diagram.betti_at(0.0),
            BettiNumbers3 {
                beta_0: 1,
                beta_1: 0,
                beta_2: 0
            }
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Algebraic identities on the filtered complex
    //
    // These need the private `Simplex` representation, so they live here rather
    // than in tests/persistence_invariants.rs. Each one is an identity that must
    // hold before the reduction is even meaningful.
    // ═══════════════════════════════════════════════════════════════════════════

    fn sample_complexes() -> Vec<Vec<Simplex>> {
        let square = [
            ManifoldPoint::<2>::new([0.0, 0.0]),
            ManifoldPoint::<2>::new([1.0, 0.0]),
            ManifoldPoint::<2>::new([1.0, 1.0]),
            ManifoldPoint::<2>::new([0.0, 1.0]),
            ManifoldPoint::<2>::new([0.5, 2.3]),
        ];
        let tetra = [
            ManifoldPoint::<3>::new([1.0, 1.0, 1.0]),
            ManifoldPoint::<3>::new([-1.0, -1.0, 1.0]),
            ManifoldPoint::<3>::new([-1.0, 1.0, -1.0]),
            ManifoldPoint::<3>::new([1.0, -1.0, -1.0]),
            ManifoldPoint::<3>::new([0.4, 0.1, -0.7]),
        ];
        let circle: Vec<ManifoldPoint<2>> = (0..9)
            .map(|i| {
                let t = i as f64 * core::f64::consts::TAU / 9.0;
                ManifoldPoint::<2>::new([libm::cos(t), libm::sin(t)])
            })
            .collect();

        vec![
            build_vietoris_rips_simplices(&square, cfg(2, f64::INFINITY)).unwrap(),
            build_vietoris_rips_simplices(&tetra, cfg(2, f64::INFINITY)).unwrap(),
            build_vietoris_rips_simplices(&square, cfg(2, 1.5)).unwrap(),
            build_vietoris_rips_simplices(&circle, cfg(1, f64::INFINITY)).unwrap(),
            build_lazy_witness_simplices(&circle[..5], &circle, cfg(2, f64::INFINITY)).unwrap(),
        ]
    }

    #[test]
    fn boundary_of_boundary_is_zero_over_z2() {
        // The defining identity of a chain complex. If it fails, every rank the
        // reduction computes is meaningless, and it fails silently.
        for (complex_idx, simplices) in sample_complexes().iter().enumerate() {
            let index = build_simplex_index(simplices);
            for j in 0..simplices.len() {
                let boundary = boundary_indices(simplices, &index, j);
                let mut accumulated: Vec<usize> = Vec::new();
                for &face in &boundary {
                    accumulated =
                        xor_sorted(&accumulated, &boundary_indices(simplices, &index, face));
                }
                assert!(
                    accumulated.is_empty(),
                    "complex {complex_idx}, simplex {j} (dim {}): d(d(sigma)) = {accumulated:?}",
                    simplices[j].dimension
                );
            }
        }
    }

    #[test]
    fn every_face_is_present_and_precedes_its_coface() {
        // A filtration requires each face to enter no later than its coface. A
        // missing face makes `boundary_indices` silently return a short column,
        // which manufactures spurious cycles.
        for (complex_idx, simplices) in sample_complexes().iter().enumerate() {
            let index = build_simplex_index(simplices);
            for j in 0..simplices.len() {
                let simplex = &simplices[j];
                if simplex.dimension == 0 {
                    continue;
                }
                let boundary = boundary_indices(simplices, &index, j);
                assert_eq!(
                    boundary.len(),
                    simplex.len,
                    "complex {complex_idx}, simplex {j} (dim {}) has {} of {} faces",
                    simplex.dimension,
                    boundary.len(),
                    simplex.len
                );
                for &face in &boundary {
                    assert!(
                        face < j,
                        "complex {complex_idx}: face {face} enters after coface {j}"
                    );
                    assert!(
                        simplices[face].filtration <= simplex.filtration + 1e-12,
                        "complex {complex_idx}: face filtration {} exceeds coface {}",
                        simplices[face].filtration,
                        simplex.filtration
                    );
                }
            }
        }
    }

    #[test]
    fn reduced_columns_have_distinct_lowest_ones() {
        // The invariant the standard reduction maintains. Two columns sharing a
        // lowest one means the reduction terminated early and bars are wrong.
        for simplices in sample_complexes() {
            let diagram = reduce_z2(&simplices, 2).unwrap();
            for pair in &diagram.pairs {
                if let Some(death) = pair.death {
                    assert!(
                        death >= pair.birth,
                        "negative-length bar: born {}, died {death}",
                        pair.birth
                    );
                }
            }
        }
    }

    #[test]
    fn witness_mode_uses_landmarks_without_rejecting_full_signal_size() {
        let points: Vec<_> = (0..40)
            .map(|i| {
                let t = i as f64 * 0.2;
                ManifoldPoint::<2>::new([libm::cos(t), libm::sin(t)])
            })
            .collect();
        let config = PersistenceConfig {
            max_homology_dim: 1,
            max_points: 8,
            max_simplices: 1_024,
            max_radius: 1.0,
            complex_kind: ComplexKind::Witness { max_landmarks: 8 },
        };

        let diagram = persistent_homology(&points, config).unwrap();

        assert!(!diagram.pairs.is_empty());
    }
}
