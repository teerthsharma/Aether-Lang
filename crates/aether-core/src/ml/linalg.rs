//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Linear Algebra Library
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Complete linear algebra primitives for ML algorithms.
//! Now powered by dynamic Tensors (Rc<RefCell> backend).
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
#[allow(unused_imports)]
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use libm::log;

use super::tensor::Tensor;

// ═══════════════════════════════════════════════════════════════════════════════
// Loss Functions
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossConfig {
    MSE,
    MAE,
    BinaryCrossEntropy,
    Hinge,
}

impl LossConfig {
    /// Compute loss value
    pub fn compute(&self, y_true: &Tensor, y_pred: &Tensor) -> f64 {
        match self {
            LossConfig::MSE => mse(y_true, y_pred),
            LossConfig::MAE => mae(y_true, y_pred),
            LossConfig::BinaryCrossEntropy => binary_cross_entropy(y_true, y_pred),
            LossConfig::Hinge => hinge_loss(y_true, y_pred),
        }
    }

    /// Compute derivative (gradient) w.r.t prediction
    pub fn derivative(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        assert_eq!(y_true.shape, y_pred.shape);
        let true_data = y_true.data.borrow();
        let pred_data = y_pred.data.borrow();
        let n = true_data.len() as f64;

        match self {
            LossConfig::MSE => Tensor::from_vec(
                pred_data
                    .iter()
                    .zip(true_data.iter())
                    .map(|(p, y)| (p - y) * (2.0 / n))
                    .collect(),
                y_pred.shape.clone(),
            ),
            LossConfig::MAE => Tensor::from_vec(
                pred_data
                    .iter()
                    .zip(true_data.iter())
                    .map(|(p, y)| {
                        let diff = p - y;
                        if diff > 0.0 {
                            1.0 / n
                        } else if diff < 0.0 {
                            -1.0 / n
                        } else {
                            0.0
                        }
                    })
                    .collect(),
                y_pred.shape.clone(),
            ),
            LossConfig::BinaryCrossEntropy => Tensor::from_vec(
                pred_data
                    .iter()
                    .zip(true_data.iter())
                    .map(|(p_raw, y)| {
                        let p = p_raw.clamp(1e-7, 1.0 - 1e-7); // Avoid div by zero
                        let grad = -(y / p) + ((1.0 - y) / (1.0 - p));
                        grad / n
                    })
                    .collect(),
                y_pred.shape.clone(),
            ),
            LossConfig::Hinge => Tensor::from_vec(
                pred_data
                    .iter()
                    .zip(true_data.iter())
                    .map(|(p, y)| if 1.0 - y * p > 0.0 { -y / n } else { 0.0 })
                    .collect(),
                y_pred.shape.clone(),
            ),
        }
    }
}

/// Mean Squared Error
pub fn mse(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    assert_eq!(y_true.shape, y_pred.shape);
    let true_data = y_true.data.borrow();
    let pred_data = y_pred.data.borrow();
    let n = true_data.len();

    let sum: f64 = true_data
        .iter()
        .zip(pred_data.iter())
        .map(|(y, p)| {
            let diff = y - p;
            diff * diff
        })
        .sum();
    sum / n as f64
}

/// Mean Absolute Error
pub fn mae(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    assert_eq!(y_true.shape, y_pred.shape);
    let true_data = y_true.data.borrow();
    let pred_data = y_pred.data.borrow();
    let n = true_data.len();

    let sum: f64 = true_data
        .iter()
        .zip(pred_data.iter())
        .map(|(y, p)| fabs(y - p))
        .sum();
    sum / n as f64
}

/// Root Mean Squared Error
pub fn rmse(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    sqrt(mse(y_true, y_pred))
}

/// Binary Cross-Entropy
pub fn binary_cross_entropy(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    assert_eq!(y_true.shape, y_pred.shape);
    let true_data = y_true.data.borrow();
    let pred_data = y_pred.data.borrow();
    let n = true_data.len();

    let sum: f64 = true_data
        .iter()
        .zip(pred_data.iter())
        .map(|(y, p_raw)| {
            let p = p_raw.clamp(1e-7, 1.0 - 1e-7);
            let y = *y;
            #[cfg(not(feature = "std"))]
            {
                -(y * log(p) + (1.0 - y) * log(1.0 - p))
            }
            #[cfg(feature = "std")]
            {
                -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
            }
        })
        .sum();
    sum / n as f64
}

/// Hinge Loss (for SVM)
pub fn hinge_loss(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    assert_eq!(y_true.shape, y_pred.shape);
    let true_data = y_true.data.borrow();
    let pred_data = y_pred.data.borrow();
    let n = true_data.len();

    let sum: f64 = true_data
        .iter()
        .zip(pred_data.iter())
        .map(|(y, p)| {
            let margin = 1.0 - y * p;
            if margin > 0.0 {
                margin
            } else {
                0.0
            }
        })
        .sum();
    sum / n as f64
}

// ═══════════════════════════════════════════════════════════════════════════════
// Gradient Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Numerical gradient of f at x
pub fn numerical_gradient<F>(f: F, x: &Tensor, epsilon: f64) -> Tensor
where
    F: Fn(&Tensor) -> f64,
{
    // Clone structure
    let grad = Tensor::zeros(&x.shape);
    let n = x.shape.iter().product();

    let mut grad_data = grad.data.borrow_mut();

    // We need a deep copy to mutate independent probe.
    let x_plus = Tensor::new(&x.data.borrow(), &x.shape);
    let x_minus = Tensor::new(&x.data.borrow(), &x.shape);

    {
        let xp_data = x_plus.data.borrow_mut();
        let xm_data = x_minus.data.borrow_mut();

        drop(xp_data);
        drop(xm_data);

        for i in 0..n {
            let original = x.data.borrow()[i];

            x_plus.data.borrow_mut()[i] = original + epsilon;
            x_minus.data.borrow_mut()[i] = original - epsilon;

            grad_data[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * epsilon);

            // Restore
            x_plus.data.borrow_mut()[i] = original;
            x_minus.data.borrow_mut()[i] = original;
        }
    }

    drop(grad_data);
    grad
}

// ═══════════════════════════════════════════════════════════════════════════════
// Distance Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Euclidean distance
pub fn euclidean_distance(a: &Tensor, b: &Tensor) -> f64 {
    assert_eq!(a.shape, b.shape);
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();

    let sum: f64 = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(a_val, b_val)| {
            let diff = a_val - b_val;
            diff * diff
        })
        .sum();
    sqrt(sum)
}

/// Manhattan distance (L1)
pub fn manhattan_distance(a: &Tensor, b: &Tensor) -> f64 {
    assert_eq!(a.shape, b.shape);
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();

    a_data
        .iter()
        .zip(b_data.iter())
        .map(|(a_val, b_val)| fabs(a_val - b_val))
        .sum()
}

/// Chebyshev distance (L∞)
pub fn chebyshev_distance(a: &Tensor, b: &Tensor) -> f64 {
    assert_eq!(a.shape, b.shape);
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();

    a_data
        .iter()
        .zip(b_data.iter())
        .map(|(a_val, b_val)| fabs(a_val - b_val))
        .fold(
            0.0,
            |max_val, abs_val| if abs_val > max_val { abs_val } else { max_val },
        )
}

/// RBF kernel value
pub fn rbf_kernel(a: &Tensor, b: &Tensor, gamma: f64) -> f64 {
    assert_eq!(a.shape, b.shape);
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();

    let sum: f64 = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(a_val, b_val)| {
            let diff = a_val - b_val;
            diff * diff
        })
        .sum();
    exp(-gamma * sum)
}

fn fabs(x: f64) -> f64 {
    #[cfg(feature = "std")]
    return x.abs();
    #[cfg(not(feature = "std"))]
    return libm::fabs(x);
}

fn sqrt(x: f64) -> f64 {
    #[cfg(feature = "std")]
    return x.sqrt();
    #[cfg(not(feature = "std"))]
    return libm::sqrt(x);
}

fn exp(x: f64) -> f64 {
    #[cfg(feature = "std")]
    return x.exp();
    #[cfg(not(feature = "std"))]
    return libm::exp(x);
}
