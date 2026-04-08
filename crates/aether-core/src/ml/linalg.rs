//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Linear Algebra Library
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Complete linear algebra primitives for ML algorithms.
//! Now powered by dynamic Tensors (Rc<RefCell> backend).
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use libm::{exp, log}; // Keep only what's not redefined locally or needed
#[cfg(feature = "std")]
use std::f64;

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
        match self {
            LossConfig::MSE => {
                let true_data = y_true.data.borrow();
                let pred_data = y_pred.data.borrow();
                let n = true_data.len();
                let mut grad_data = Vec::with_capacity(n);
                let factor = 2.0 / n as f64;
                for i in 0..n {
                    grad_data.push((pred_data[i] - true_data[i]) * factor);
                }
                Tensor::new(&grad_data, &y_pred.shape)
            }
            LossConfig::MAE => {
                let true_data = y_true.data.borrow();
                let pred_data = y_pred.data.borrow();
                let n = true_data.len();
                let mut grad_data = Vec::with_capacity(n);
                let factor = 1.0 / n as f64;
                for i in 0..n {
                    let diff = pred_data[i] - true_data[i];
                    if diff > 0.0 {
                        grad_data.push(factor);
                    } else if diff < 0.0 {
                        grad_data.push(-factor);
                    } else {
                        grad_data.push(0.0);
                    }
                }
                Tensor::new(&grad_data, &y_pred.shape)
            }
            LossConfig::BinaryCrossEntropy => {
                // dL/dp = (1-y)/(1-p) - y/p
                let true_data = y_true.data.borrow();
                let pred_data = y_pred.data.borrow();
                let n = true_data.len();
                let mut grad_data = Vec::with_capacity(n); // Fixed: using Vec instead of let mut

                for i in 0..n {
                    let y = true_data[i];
                    let p = pred_data[i].clamp(1e-7, 1.0 - 1e-7); // Avoid div by zero

                    let grad = -(y / p) + ((1.0 - y) / (1.0 - p));
                    grad_data.push(grad / n as f64);
                }
                Tensor::new(&grad_data, &y_pred.shape)
            }
            LossConfig::Hinge => {
                // L = max(0, 1 - y*p)
                // dL/dp = -y if 1 - y*p > 0 else 0
                let true_data = y_true.data.borrow();
                let pred_data = y_pred.data.borrow();
                let n = true_data.len();
                let mut grad_data = Vec::with_capacity(n);
                let factor = 1.0 / n as f64;

                for i in 0..n {
                    let y = true_data[i];
                    let p = pred_data[i];

                    if 1.0 - y * p > 0.0 {
                        grad_data.push(-y * factor);
                    } else {
                        grad_data.push(0.0);
                    }
                }
                Tensor::new(&grad_data, &y_pred.shape)
            }
        }
    }
}

/// Mean Squared Error
pub fn mse(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    assert_eq!(y_true.shape, y_pred.shape);
    let mut sum = 0.0;
    let true_data = y_true.data.borrow();
    let pred_data = y_pred.data.borrow();
    let n = true_data.len();

    for i in 0..n {
        let diff = true_data[i] - pred_data[i];
        sum += diff * diff;
    }
    sum / n as f64
}

/// Mean Absolute Error
pub fn mae(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    assert_eq!(y_true.shape, y_pred.shape);
    let mut sum = 0.0;
    let true_data = y_true.data.borrow();
    let pred_data = y_pred.data.borrow();
    let n = true_data.len();

    for i in 0..n {
        sum += fabs(true_data[i] - pred_data[i]);
    }
    sum / n as f64
}

/// Root Mean Squared Error
pub fn rmse(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    sqrt(mse(y_true, y_pred))
}

/// Binary Cross-Entropy
pub fn binary_cross_entropy(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    assert_eq!(y_true.shape, y_pred.shape);
    let mut sum = 0.0;
    let true_data = y_true.data.borrow();
    let pred_data = y_pred.data.borrow();
    let n = true_data.len();

    for i in 0..n {
        let p = pred_data[i].clamp(1e-7, 1.0 - 1e-7);
        let y = true_data[i];

        #[cfg(not(feature = "std"))]
        {
            sum -= y * log(p) + (1.0 - y) * log(1.0 - p);
        }
        #[cfg(feature = "std")]
        {
            sum -= y * p.ln() + (1.0 - y) * (1.0 - p).ln();
        }
    }
    sum / n as f64
}

/// Hinge Loss (for SVM)
pub fn hinge_loss(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    assert_eq!(y_true.shape, y_pred.shape);
    let mut sum = 0.0;
    let true_data = y_true.data.borrow();
    let pred_data = y_pred.data.borrow();
    let n = true_data.len();

    for i in 0..n {
        let margin = 1.0 - true_data[i] * pred_data[i];
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
pub fn numerical_gradient<F>(f: F, x: &Tensor, epsilon: f64) -> Tensor
where
    F: Fn(&Tensor) -> f64,
{
    // Clone structure
    let grad = Tensor::zeros(&x.shape);
    let n = x.shape.iter().product();

    let mut grad_data = grad.data.borrow_mut();

    // We need a deep copy to mutate independent probe.
    let mut x_plus = Tensor::new(&x.data.borrow(), &x.shape);
    let mut x_minus = Tensor::new(&x.data.borrow(), &x.shape);

    {
        let mut xp_data = x_plus.data.borrow_mut();
        let mut xm_data = x_minus.data.borrow_mut();

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
    let mut sum = 0.0;
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();
    for i in 0..a_data.len() {
        let diff = a_data[i] - b_data[i];
        sum += diff * diff;
    }
    sqrt(sum)
}

/// Manhattan distance (L1)
pub fn manhattan_distance(a: &Tensor, b: &Tensor) -> f64 {
    assert_eq!(a.shape, b.shape);
    let mut sum = 0.0;
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();
    for i in 0..a_data.len() {
        sum += fabs(a_data[i] - b_data[i]);
    }
    sum
}

/// Chebyshev distance (L∞)
pub fn chebyshev_distance(a: &Tensor, b: &Tensor) -> f64 {
    assert_eq!(a.shape, b.shape);
    let mut max = 0.0;
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();
    for i in 0..a_data.len() {
        let abs_val = fabs(a_data[i] - b_data[i]);
        if abs_val > max {
            max = abs_val;
        }
    }
    max
}

/// RBF kernel value
pub fn rbf_kernel(a: &Tensor, b: &Tensor, gamma: f64) -> f64 {
    assert_eq!(a.shape, b.shape);
    let mut dist_sq = 0.0;
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();
    for i in 0..a_data.len() {
        let diff = a_data[i] - b_data[i];
        dist_sq += diff * diff;
    }
    exp(-gamma * dist_sq)
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
