//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Topological Data Manifold
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Transforms massive data streams into topological shapes using sparse attention
//! and geometric concentration. This is the core of "making data 3D for everyone".
//!
//! Key Concepts:
//!   - Sparse Attention: Only attend to topologically significant points
//!   - Concentration: Collapse high-dim data onto low-dim manifold
//!   - Shape Extraction: Persistent homology for structure discovery
//!
//! Mathematical Foundation:
//!   - Time-Delay Embedding: Φ(x) = [x(t), x(t-τ), x(t-2τ), ...]
//!   - Sparse Attention: A(i,j) = 1 iff d(xᵢ, xⱼ) < ε (geometric locality)
//!   - Concentration: Project to principal manifold via local PCA
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

use libm::sqrt;

// ═══════════════════════════════════════════════════════════════════════════════
// Manifold Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Default embedding dimension for time-delay embedding
const DEFAULT_EMBED_DIM: usize = 3;

/// Default time delay (τ) for embedding
const DEFAULT_TAU: usize = 1;

/// Default neighborhood radius for sparse attention
const DEFAULT_EPSILON: f64 = 0.5;

/// Maximum points to track (memory constraint for no_std)
const MAX_POINTS: usize = 256;

// ═══════════════════════════════════════════════════════════════════════════════
// Point Cloud Representation
// ═══════════════════════════════════════════════════════════════════════════════

/// A point in the embedded manifold space
#[derive(Debug, Clone, Copy)]
pub struct ManifoldPoint<const D: usize> {
    pub coords: [f64; D],
}

impl<const D: usize> ManifoldPoint<D> {
    pub const fn zero() -> Self {
        Self { coords: [0.0; D] }
    }
    
    pub fn new(coords: [f64; D]) -> Self {
        Self { coords }
    }
    
    /// Euclidean distance to another point
    pub fn distance(&self, other: &Self) -> f64 {
        let mut sum = 0.0;
        for i in 0..D {
            let d = self.coords[i] - other.coords[i];
            sum += d * d;
        }
        sqrt(sum)
    }
    
    /// Check if within epsilon-neighborhood (sparse attention criterion)
    pub fn is_neighbor(&self, other: &Self, epsilon: f64) -> bool {
        self.distance(other) < epsilon
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Time-Delay Embedding
// ═══════════════════════════════════════════════════════════════════════════════

/// Embeds 1D signal into D-dimensional manifold using Takens' theorem
/// 
/// Φ(t) = [x(t), x(t-τ), x(t-2τ), ..., x(t-(D-1)τ)]
/// 
/// This transforms temporal data into geometric shapes that reveal
/// the underlying dynamical system's attractor.
#[derive(Debug)]
pub struct TimeDelayEmbedder<const D: usize> {
    /// Time delay parameter τ
    tau: usize,
    
    /// Circular buffer for recent values
    buffer: [f64; 256],
    buffer_pos: usize,
    buffer_len: usize,
}

impl<const D: usize> TimeDelayEmbedder<D> {
    pub fn new(tau: usize) -> Self {
        Self {
            tau: if tau == 0 { 1 } else { tau },
            buffer: [0.0; 256],
            buffer_pos: 0,
            buffer_len: 0,
        }
    }
    
    /// Add a new sample to the buffer
    pub fn push(&mut self, value: f64) {
        self.buffer[self.buffer_pos] = value;
        self.buffer_pos = (self.buffer_pos + 1) % 256;
        if self.buffer_len < 256 {
            self.buffer_len += 1;
        }
    }
    
    /// Get embedded point from current buffer state
    pub fn embed(&self) -> Option<ManifoldPoint<D>> {
        let required = D * self.tau;
        if self.buffer_len < required {
            return None;
        }
        
        let mut point = ManifoldPoint::zero();
        for i in 0..D {
            let offset = i * self.tau;
            let idx = (self.buffer_pos + 256 - 1 - offset) % 256;
            point.coords[i] = self.buffer[idx];
        }
        
        Some(point)
    }
    
    /// Reset the embedder
    pub fn reset(&mut self) {
        self.buffer = [0.0; 256];
        self.buffer_pos = 0;
        self.buffer_len = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sparse Attention Graph
// ═══════════════════════════════════════════════════════════════════════════════

/// Sparse attention matrix using geometric locality
/// 
/// Instead of O(n²) dense attention, we only connect points within
/// ε-neighborhood. This is the key to handling massive data.
/// 
/// A(i,j) = 1 iff d(pᵢ, pⱼ) < ε
#[derive(Debug)]
pub struct SparseAttentionGraph<const D: usize> {
    /// Point cloud
    points: [ManifoldPoint<D>; MAX_POINTS],
    point_count: usize,
    
    /// Epsilon neighborhood radius
    epsilon: f64,
    
    /// Sparse adjacency (bit-packed for memory efficiency)
    /// adjacency[i] is bitmask of neighbors for point i
    adjacency: [u64; MAX_POINTS],
}

impl<const D: usize> SparseAttentionGraph<D> {
    pub fn new(epsilon: f64) -> Self {
        Self {
            points: [ManifoldPoint::zero(); MAX_POINTS],
            point_count: 0,
            epsilon,
            adjacency: [0; MAX_POINTS],
        }
    }
    
    /// Add a point and compute its sparse attention edges
    pub fn add_point(&mut self, point: ManifoldPoint<D>) -> Option<usize> {
        if self.point_count >= MAX_POINTS {
            return None;
        }
        
        let idx = self.point_count;
        self.points[idx] = point;
        
        // Compute sparse edges (only to nearby points)
        let mut mask = 0u64;
        for i in 0..idx {
            if point.is_neighbor(&self.points[i], self.epsilon) {
                // Set bit for neighbor relationship
                if i < 64 {
                    mask |= 1 << i;
                    // Symmetric: add reverse edge
                    self.adjacency[i] |= 1 << (idx % 64);
                }
            }
        }
        self.adjacency[idx] = mask;
        
        self.point_count += 1;
        Some(idx)
    }
    
    /// Get number of neighbors (degree) for a point
    pub fn degree(&self, idx: usize) -> u32 {
        if idx >= self.point_count {
            return 0;
        }
        self.adjacency[idx].count_ones()
    }
    
    /// Check if two points are connected
    pub fn are_neighbors(&self, i: usize, j: usize) -> bool {
        if i >= self.point_count || j >= self.point_count || j >= 64 {
            return false;
        }
        (self.adjacency[i] & (1 << j)) != 0
    }
    
    /// Compute connected components (β₀) using Union-Find
    pub fn compute_betti_0(&self) -> u32 {
        if self.point_count == 0 {
            return 0;
        }
        
        // Simple DFS-based component counting
        let mut visited = [false; MAX_POINTS];
        let mut components = 0u32;
        
        for start in 0..self.point_count {
            if visited[start] {
                continue;
            }
            
            // BFS/DFS from this point
            components += 1;
            let mut stack = [0usize; 64];
            let mut stack_top = 0;
            
            stack[0] = start;
            stack_top = 1;
            
            while stack_top > 0 {
                stack_top -= 1;
                let current = stack[stack_top];
                
                if visited[current] {
                    continue;
                }
                visited[current] = true;
                
                // Add unvisited neighbors
                for neighbor in 0..64.min(self.point_count) {
                    if !visited[neighbor] && self.are_neighbors(current, neighbor) {
                        if stack_top < 64 {
                            stack[stack_top] = neighbor;
                            stack_top += 1;
                        }
                    }
                }
            }
        }
        
        components
    }
    
    /// Estimate β₁ (cycles) using Euler characteristic
    /// χ = V - E + F, for planar: β₀ - β₁ + β₂ = χ
    /// Simplified: β₁ ≈ E - V + β₀ (ignoring higher homology)
    pub fn estimate_betti_1(&self) -> u32 {
        let v = self.point_count as i32;
        let mut e = 0i32;
        
        for i in 0..self.point_count {
            e += self.adjacency[i].count_ones() as i32;
        }
        e /= 2; // Edges counted twice
        
        let b0 = self.compute_betti_0() as i32;
        
        // β₁ ≈ E - V + β₀ (simplified)
        let b1 = e - v + b0;
        if b1 > 0 { b1 as u32 } else { 0 }
    }
    
    /// Get the topological shape signature
    pub fn shape(&self) -> (u32, u32) {
        (self.compute_betti_0(), self.estimate_betti_1())
    }
    
    /// Clear the graph
    pub fn clear(&mut self) {
        self.point_count = 0;
        self.adjacency = [0; MAX_POINTS];
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Concentration (Dimension Reduction)
// ═══════════════════════════════════════════════════════════════════════════════

/// Geometric concentration: collapse high-dim data to principal axes
/// Uses streaming mean and variance for memory efficiency
#[derive(Debug)]
pub struct GeometricConcentrator<const D: usize> {
    /// Running mean
    mean: [f64; 8],
    
    /// Running variance (for principal direction)
    variance: [f64; 8],
    
    /// Sample count
    count: u64,
}

impl<const D: usize> GeometricConcentrator<D> {
    pub fn new() -> Self {
        Self {
            mean: [0.0; 8],
            variance: [0.0; 8],
            count: 0,
        }
    }
    
    /// Update statistics with new point (Welford's algorithm)
    pub fn update(&mut self, point: &ManifoldPoint<D>) {
        self.count += 1;
        let n = self.count as f64;
        
        for i in 0..D.min(8) {
            let delta = point.coords[i] - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = point.coords[i] - self.mean[i];
            self.variance[i] += delta * delta2;
        }
    }
    
    /// Get the principal dimension (highest variance)
    pub fn principal_dimension(&self) -> usize {
        let mut max_var = 0.0;
        let mut max_dim = 0;
        
        for i in 0..D.min(8) {
            if self.variance[i] > max_var {
                max_var = self.variance[i];
                max_dim = i;
            }
        }
        
        max_dim
    }
    
    /// Project point onto principal axis (1D concentration)
    pub fn concentrate_1d(&self, point: &ManifoldPoint<D>) -> f64 {
        let dim = self.principal_dimension();
        if dim < D {
            point.coords[dim] - self.mean[dim]
        } else {
            0.0
        }
    }
    
    /// Get concentration ratio (how much variance is in principal dim)
    pub fn concentration_ratio(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        
        let total: f64 = self.variance.iter().take(D.min(8)).sum();
        if total == 0.0 {
            return 0.0;
        }
        
        let principal = self.variance[self.principal_dimension()];
        principal / total
    }
    
    pub fn reset(&mut self) {
        self.mean = [0.0; 8];
        self.variance = [0.0; 8];
        self.count = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Complete Manifold Pipeline
// ═══════════════════════════════════════════════════════════════════════════════

/// Full pipeline: Stream → Embed → Sparse Attention → Shape
pub struct TopologicalPipeline<const D: usize> {
    embedder: TimeDelayEmbedder<D>,
    graph: SparseAttentionGraph<D>,
    concentrator: GeometricConcentrator<D>,
}

impl<const D: usize> TopologicalPipeline<D> {
    pub fn new(tau: usize, epsilon: f64) -> Self {
        Self {
            embedder: TimeDelayEmbedder::new(tau),
            graph: SparseAttentionGraph::new(epsilon),
            concentrator: GeometricConcentrator::new(),
        }
    }
    
    /// Process a new data sample
    pub fn push(&mut self, value: f64) -> Option<(u32, u32)> {
        self.embedder.push(value);
        
        if let Some(point) = self.embedder.embed() {
            self.concentrator.update(&point);
            self.graph.add_point(point)?;
            Some(self.graph.shape())
        } else {
            None
        }
    }
    
    /// Get current shape (β₀, β₁)
    pub fn shape(&self) -> (u32, u32) {
        self.graph.shape()
    }
    
    /// Get concentration ratio
    pub fn concentration(&self) -> f64 {
        self.concentrator.concentration_ratio()
    }
    
    /// Reset pipeline
    pub fn reset(&mut self) {
        self.embedder.reset();
        self.graph.clear();
        self.concentrator.reset();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_point_distance() {
        let p1 = ManifoldPoint::<3>::new([0.0, 0.0, 0.0]);
        let p2 = ManifoldPoint::<3>::new([3.0, 4.0, 0.0]);
        
        assert!((p1.distance(&p2) - 5.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_sparse_attention_neighbor() {
        let p1 = ManifoldPoint::<3>::new([0.0, 0.0, 0.0]);
        let p2 = ManifoldPoint::<3>::new([0.1, 0.1, 0.0]);
        
        assert!(p1.is_neighbor(&p2, 0.5));
        assert!(!p1.is_neighbor(&p2, 0.1));
    }
    
    #[test]
    fn test_embedding() {
        let mut emb = TimeDelayEmbedder::<3>::new(1);
        
        for i in 0..10 {
            emb.push(i as f64);
        }
        
        let point = emb.embed().unwrap();
        // Should be [9, 8, 7] (last 3 values with τ=1)
        assert!((point.coords[0] - 9.0).abs() < 1e-10);
        assert!((point.coords[1] - 8.0).abs() < 1e-10);
        assert!((point.coords[2] - 7.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_single_component() {
        let mut graph = SparseAttentionGraph::<3>::new(1.0);
        
        // Add points that are all within epsilon of each other
        graph.add_point(ManifoldPoint::new([0.0, 0.0, 0.0]));
        graph.add_point(ManifoldPoint::new([0.5, 0.0, 0.0]));
        graph.add_point(ManifoldPoint::new([0.5, 0.5, 0.0]));
        
        // Should be single connected component
        assert_eq!(graph.compute_betti_0(), 1);
    }
}
