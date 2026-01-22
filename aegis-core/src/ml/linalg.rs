//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Linear Algebra Library
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Complete linear algebra primitives for ML algorithms, optimized for no_std.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

use libm::{sqrt, fabs, exp};

/// Maximum vector/matrix size for stack allocation
pub const MAX_DIM: usize = 64;
pub const MAX_MATRIX: usize = 32;

// ═══════════════════════════════════════════════════════════════════════════════
// Vector Operations
// ═══════════════════════════════════════════════════════════════════════════════

/// Dense vector with fixed maximum size
#[derive(Debug, Clone, Copy)]
pub struct Vector {
    pub data: [f64; MAX_DIM],
    pub len: usize,
}

impl Vector {
    pub const fn zeros(len: usize) -> Self {
        Self {
            data: [0.0; MAX_DIM],
            len,
        }
    }
    
    pub fn from_slice(slice: &[f64]) -> Self {
        let mut v = Self::zeros(slice.len().min(MAX_DIM));
        for (i, &val) in slice.iter().enumerate().take(MAX_DIM) {
            v.data[i] = val;
        }
        v
    }
    
    /// Dot product
    pub fn dot(&self, other: &Self) -> f64 {
        let n = self.len.min(other.len);
        let mut sum = 0.0;
        for i in 0..n {
            sum += self.data[i] * other.data[i];
        }
        sum
    }
    
    /// Euclidean norm (L2)
    pub fn norm(&self) -> f64 {
        sqrt(self.dot(self))
    }
    
    /// L1 norm (Manhattan)
    pub fn norm_l1(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.len {
            sum += fabs(self.data[i]);
        }
        sum
    }
    
    /// Normalize to unit length
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-10 {
            return *self;
        }
        self.scale(1.0 / n)
    }
    
    /// Scalar multiplication
    pub fn scale(&self, s: f64) -> Self {
        let mut result = *self;
        for i in 0..self.len {
            result.data[i] *= s;
        }
        result
    }
    
    /// Vector addition
    pub fn add(&self, other: &Self) -> Self {
        let n = self.len.min(other.len);
        let mut result = Self::zeros(n);
        for i in 0..n {
            result.data[i] = self.data[i] + other.data[i];
        }
        result
    }
    
    /// Vector subtraction
    pub fn sub(&self, other: &Self) -> Self {
        let n = self.len.min(other.len);
        let mut result = Self::zeros(n);
        for i in 0..n {
            result.data[i] = self.data[i] - other.data[i];
        }
        result
    }
    
    /// Element-wise multiplication (Hadamard product)
    pub fn hadamard(&self, other: &Self) -> Self {
        let n = self.len.min(other.len);
        let mut result = Self::zeros(n);
        for i in 0..n {
            result.data[i] = self.data[i] * other.data[i];
        }
        result
    }
    
    /// Euclidean distance to another vector
    pub fn distance(&self, other: &Self) -> f64 {
        self.sub(other).norm()
    }
    
    /// Cosine similarity
    pub fn cosine_similarity(&self, other: &Self) -> f64 {
        let na = self.norm();
        let nb = other.norm();
        if na < 1e-10 || nb < 1e-10 {
            return 0.0;
        }
        self.dot(other) / (na * nb)
    }
    
    /// Sum of elements
    pub fn sum(&self) -> f64 {
        let mut s = 0.0;
        for i in 0..self.len {
            s += self.data[i];
        }
        s
    }
    
    /// Mean of elements
    pub fn mean(&self) -> f64 {
        if self.len == 0 { return 0.0; }
        self.sum() / self.len as f64
    }
    
    /// Variance
    pub fn variance(&self) -> f64 {
        if self.len == 0 { return 0.0; }
        let m = self.mean();
        let mut sum = 0.0;
        for i in 0..self.len {
            let diff = self.data[i] - m;
            sum += diff * diff;
        }
        sum / self.len as f64
    }
    
    /// Standard deviation
    pub fn std(&self) -> f64 {
        sqrt(self.variance())
    }
    
    /// Max element
    pub fn max(&self) -> f64 {
        let mut m = f64::NEG_INFINITY;
        for i in 0..self.len {
            if self.data[i] > m {
                m = self.data[i];
            }
        }
        m
    }
    
    /// Min element
    pub fn min(&self) -> f64 {
        let mut m = f64::INFINITY;
        for i in 0..self.len {
            if self.data[i] < m {
                m = self.data[i];
            }
        }
        m
    }
    
    /// Argmax
    pub fn argmax(&self) -> usize {
        let mut idx = 0;
        let mut m = f64::NEG_INFINITY;
        for i in 0..self.len {
            if self.data[i] > m {
                m = self.data[i];
                idx = i;
            }
        }
        idx
    }
    
    /// Apply ReLU activation
    pub fn relu(&self) -> Self {
        let mut result = *self;
        for i in 0..self.len {
            if result.data[i] < 0.0 {
                result.data[i] = 0.0;
            }
        }
        result
    }
    
    /// Apply Sigmoid activation
    pub fn sigmoid(&self) -> Self {
        let mut result = *self;
        for i in 0..self.len {
            result.data[i] = 1.0 / (1.0 + exp(-result.data[i]));
        }
        result
    }
    
    /// Softmax (numerically stable)
    pub fn softmax(&self) -> Self {
        let max_val = self.max();
        let mut result = *self;
        let mut sum = 0.0;
        
        for i in 0..self.len {
            result.data[i] = exp(result.data[i] - max_val);
            sum += result.data[i];
        }
        
        for i in 0..self.len {
            result.data[i] /= sum;
        }
        result
    }
}

impl Default for Vector {
    fn default() -> Self {
        Self::zeros(0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Matrix Operations
// ═══════════════════════════════════════════════════════════════════════════════

/// Dense matrix with fixed maximum size
#[derive(Debug, Clone, Copy)]
pub struct Matrix {
    pub data: [[f64; MAX_MATRIX]; MAX_MATRIX],
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub const fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: [[0.0; MAX_MATRIX]; MAX_MATRIX],
            rows,
            cols,
        }
    }
    
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n.min(MAX_MATRIX) {
            m.data[i][i] = 1.0;
        }
        m
    }
    
    /// Matrix-vector multiplication
    pub fn matvec(&self, v: &Vector) -> Vector {
        let mut result = Vector::zeros(self.rows);
        for i in 0..self.rows {
            let mut sum = 0.0;
            for j in 0..self.cols.min(v.len) {
                sum += self.data[i][j] * v.data[j];
            }
            result.data[i] = sum;
        }
        result
    }
    
    /// Matrix-matrix multiplication
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        let mut result = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        result
    }
    
    /// Transpose
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }
    
    /// Outer product of two vectors
    pub fn outer(a: &Vector, b: &Vector) -> Matrix {
        let mut result = Matrix::zeros(a.len, b.len);
        for i in 0..a.len {
            for j in 0..b.len {
                result.data[i][j] = a.data[i] * b.data[j];
            }
        }
        result
    }
    
    /// Add matrices
    pub fn add(&self, other: &Matrix) -> Matrix {
        let rows = self.rows.min(other.rows);
        let cols = self.cols.min(other.cols);
        let mut result = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        result
    }
    
    /// Subtract matrices
    pub fn sub(&self, other: &Matrix) -> Matrix {
        let rows = self.rows.min(other.rows);
        let cols = self.cols.min(other.cols);
        let mut result = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                result.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        result
    }
    
    /// Scale by scalar
    pub fn scale(&self, s: f64) -> Matrix {
        let mut result = *self;
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] *= s;
            }
        }
        result
    }
    
    /// Frobenius norm
    pub fn frobenius_norm(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.rows {
            for j in 0..self.cols {
                sum += self.data[i][j] * self.data[i][j];
            }
        }
        sqrt(sum)
    }
    
    /// Get column as vector
    pub fn get_col(&self, j: usize) -> Vector {
        let mut v = Vector::zeros(self.rows);
        for i in 0..self.rows {
            v.data[i] = self.data[i][j];
        }
        v
    }
    
    /// Get row as vector
    pub fn get_row(&self, i: usize) -> Vector {
        let mut v = Vector::zeros(self.cols);
        for j in 0..self.cols {
            v.data[j] = self.data[i][j];
        }
        v
    }
}

impl Default for Matrix {
    fn default() -> Self {
        Self::zeros(0, 0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Loss Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Mean Squared Error
pub fn mse(y_true: &Vector, y_pred: &Vector) -> f64 {
    let diff = y_true.sub(y_pred);
    diff.dot(&diff) / y_true.len as f64
}

/// Mean Absolute Error
pub fn mae(y_true: &Vector, y_pred: &Vector) -> f64 {
    let mut sum = 0.0;
    let n = y_true.len.min(y_pred.len);
    for i in 0..n {
        sum += fabs(y_true.data[i] - y_pred.data[i]);
    }
    sum / n as f64
}

/// Root Mean Squared Error
pub fn rmse(y_true: &Vector, y_pred: &Vector) -> f64 {
    sqrt(mse(y_true, y_pred))
}

/// Binary Cross-Entropy
pub fn binary_cross_entropy(y_true: &Vector, y_pred: &Vector) -> f64 {
    let mut sum = 0.0;
    let n = y_true.len.min(y_pred.len);
    for i in 0..n {
        let p = y_pred.data[i].max(1e-7).min(1.0 - 1e-7);
        let y = y_true.data[i];
        sum -= y * libm::log(p) + (1.0 - y) * libm::log(1.0 - p);
    }
    sum / n as f64
}

/// Hinge Loss (for SVM)
pub fn hinge_loss(y_true: &Vector, y_pred: &Vector) -> f64 {
    let mut sum = 0.0;
    let n = y_true.len.min(y_pred.len);
    for i in 0..n {
        let margin = 1.0 - y_true.data[i] * y_pred.data[i];
        if margin > 0.0 {
            sum += margin;
        }
    }
    sum / n as f64
}

// ═══════════════════════════════════════════════════════════════════════════════
// Gradient Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Numerical gradient of f at x
pub fn numerical_gradient<F>(f: F, x: &Vector, epsilon: f64) -> Vector
where
    F: Fn(&Vector) -> f64,
{
    let mut grad = Vector::zeros(x.len);
    
    for i in 0..x.len {
        let mut x_plus = *x;
        let mut x_minus = *x;
        x_plus.data[i] += epsilon;
        x_minus.data[i] -= epsilon;
        grad.data[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * epsilon);
    }
    
    grad
}

/// MSE gradient for linear model
pub fn mse_gradient(x_mat: &Matrix, y: &Vector, weights: &Vector) -> Vector {
    let n = x_mat.rows as f64;
    let mut grad = Vector::zeros(weights.len);
    
    for i in 0..x_mat.rows {
        let x_i = x_mat.get_row(i);
        let pred = x_i.dot(weights);
        let error = pred - y.data[i];
        
        for j in 0..weights.len {
            grad.data[j] += 2.0 * error * x_i.data[j] / n;
        }
    }
    
    grad
}

// ═══════════════════════════════════════════════════════════════════════════════
// Distance Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Euclidean distance
pub fn euclidean_distance(a: &Vector, b: &Vector) -> f64 {
    a.distance(b)
}

/// Manhattan distance (L1)
pub fn manhattan_distance(a: &Vector, b: &Vector) -> f64 {
    a.sub(b).norm_l1()
}

/// Chebyshev distance (L∞)
pub fn chebyshev_distance(a: &Vector, b: &Vector) -> f64 {
    let diff = a.sub(b);
    let mut max = 0.0;
    for i in 0..diff.len {
        let abs_val = fabs(diff.data[i]);
        if abs_val > max {
            max = abs_val;
        }
    }
    max
}

/// RBF kernel value
pub fn rbf_kernel(a: &Vector, b: &Vector, gamma: f64) -> f64 {
    let dist_sq = a.sub(b).dot(&a.sub(b));
    exp(-gamma * dist_sq)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vector_dot() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
        assert!((a.dot(&b) - 32.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_vector_norm() {
        let v = Vector::from_slice(&[3.0, 4.0]);
        assert!((v.norm() - 5.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_matrix_matvec() {
        let mut m = Matrix::zeros(2, 2);
        m.data[0][0] = 1.0; m.data[0][1] = 2.0;
        m.data[1][0] = 3.0; m.data[1][1] = 4.0;
        
        let v = Vector::from_slice(&[1.0, 1.0]);
        let result = m.matvec(&v);
        
        assert!((result.data[0] - 3.0).abs() < 1e-10);
        assert!((result.data[1] - 7.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_softmax() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let s = v.softmax();
        assert!((s.sum() - 1.0).abs() < 1e-10);
    }
}
