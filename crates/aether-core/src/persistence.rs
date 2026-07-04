//! Bounded persistent homology for AETHER point clouds.
//!
//! The engine builds a filtered simplicial complex through tetrahedra and
//! reduces boundary columns over Z2. It is exact for the selected complex and
//! deliberately bounded so topological ML workloads fail fast instead of
//! exhausting memory.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::manifold::{ManifoldPoint, TimeDelayEmbedder};

const SIMPLEX_VERTICES: usize = 4;

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
    pub const fn h2_default() -> Self {
        Self {
            max_homology_dim: 2,
            max_points: 32,
            max_simplices: 8_192,
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

    for j in 0..simplices.len() {
        let mut column = boundary_indices(simplices, j);
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

fn boundary_indices(simplices: &[Simplex], simplex_idx: usize) -> Vec<usize> {
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
        if let Some(idx) = find_simplex(simplices, simplex_idx, &face, face_len) {
            boundary.push(idx);
        }
    }
    boundary.sort_unstable();
    boundary
}

fn find_simplex(
    simplices: &[Simplex],
    before: usize,
    vertices: &[usize; SIMPLEX_VERTICES],
    len: usize,
) -> Option<usize> {
    simplices[..before]
        .iter()
        .position(|simplex| simplex.len == len && simplex.vertices[..len] == vertices[..len])
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
