import re

with open('crates/aether-core/src/ml/linalg.rs', 'r') as f:
    content = f.read()

mse_old = """pub fn mse(y_true: &Tensor, y_pred: &Tensor) -> f64 {
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
}"""

mse_new = """pub fn mse(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    assert_eq!(y_true.shape, y_pred.shape);
    let true_data = y_true.data.borrow();
    let pred_data = y_pred.data.borrow();
    let n = true_data.len();

    let sum: f64 = true_data.iter().zip(pred_data.iter()).map(|(y, p)| {
        let diff = y - p;
        diff * diff
    }).sum();
    sum / n as f64
}"""

content = content.replace(mse_old, mse_new)

mae_old = """pub fn mae(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    assert_eq!(y_true.shape, y_pred.shape);
    let mut sum = 0.0;
    let true_data = y_true.data.borrow();
    let pred_data = y_pred.data.borrow();
    let n = true_data.len();

    for i in 0..n {
        sum += fabs(true_data[i] - pred_data[i]);
    }
    sum / n as f64
}"""

mae_new = """pub fn mae(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    assert_eq!(y_true.shape, y_pred.shape);
    let true_data = y_true.data.borrow();
    let pred_data = y_pred.data.borrow();
    let n = true_data.len();

    let sum: f64 = true_data.iter().zip(pred_data.iter()).map(|(y, p)| fabs(y - p)).sum();
    sum / n as f64
}"""

content = content.replace(mae_old, mae_new)

bce_old = """pub fn binary_cross_entropy(y_true: &Tensor, y_pred: &Tensor) -> f64 {
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
}"""

bce_new = """pub fn binary_cross_entropy(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    assert_eq!(y_true.shape, y_pred.shape);
    let true_data = y_true.data.borrow();
    let pred_data = y_pred.data.borrow();
    let n = true_data.len();

    let sum: f64 = true_data.iter().zip(pred_data.iter()).map(|(y, p_raw)| {
        let p = p_raw.clamp(1e-7, 1.0 - 1e-7);

        #[cfg(not(feature = "std"))]
        {
            -(y * log(p) + (1.0 - y) * log(1.0 - p))
        }
        #[cfg(feature = "std")]
        {
            -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
        }
    }).sum();
    sum / n as f64
}"""

content = content.replace(bce_old, bce_new)

hinge_old = """pub fn hinge_loss(y_true: &Tensor, y_pred: &Tensor) -> f64 {
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
}"""

hinge_new = """pub fn hinge_loss(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    assert_eq!(y_true.shape, y_pred.shape);
    let true_data = y_true.data.borrow();
    let pred_data = y_pred.data.borrow();
    let n = true_data.len();

    let sum: f64 = true_data.iter().zip(pred_data.iter()).map(|(y, p)| {
        let margin = 1.0 - y * p;
        if margin > 0.0 { margin } else { 0.0 }
    }).sum();
    sum / n as f64
}"""

content = content.replace(hinge_old, hinge_new)


euclidean_old = """pub fn euclidean_distance(a: &Tensor, b: &Tensor) -> f64 {
    assert_eq!(a.shape, b.shape);
    let mut sum = 0.0;
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();

    for i in 0..a_data.len() {
        let diff = a_data[i] - b_data[i];
        sum += diff * diff;
    }
    sqrt(sum)
}"""

euclidean_new = """pub fn euclidean_distance(a: &Tensor, b: &Tensor) -> f64 {
    assert_eq!(a.shape, b.shape);
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();

    let sum: f64 = a_data.iter().zip(b_data.iter()).map(|(a_val, b_val)| {
        let diff = a_val - b_val;
        diff * diff
    }).sum();
    sqrt(sum)
}"""

content = content.replace(euclidean_old, euclidean_new)


manhattan_old = """pub fn manhattan_distance(a: &Tensor, b: &Tensor) -> f64 {
    assert_eq!(a.shape, b.shape);
    let mut sum = 0.0;
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();

    for i in 0..a_data.len() {
        sum += fabs(a_data[i] - b_data[i]);
    }
    sum
}"""

manhattan_new = """pub fn manhattan_distance(a: &Tensor, b: &Tensor) -> f64 {
    assert_eq!(a.shape, b.shape);
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();

    a_data.iter().zip(b_data.iter()).map(|(a_val, b_val)| fabs(a_val - b_val)).sum()
}"""

content = content.replace(manhattan_old, manhattan_new)


chebyshev_old = """pub fn chebyshev_distance(a: &Tensor, b: &Tensor) -> f64 {
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
}"""

chebyshev_new = """pub fn chebyshev_distance(a: &Tensor, b: &Tensor) -> f64 {
    assert_eq!(a.shape, b.shape);
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();

    a_data.iter().zip(b_data.iter()).fold(0.0, |max_val, (a_val, b_val)| {
        let abs_val = fabs(a_val - b_val);
        if abs_val > max_val { abs_val } else { max_val }
    })
}"""

content = content.replace(chebyshev_old, chebyshev_new)


rbf_old = """pub fn rbf_kernel(a: &Tensor, b: &Tensor, gamma: f64) -> f64 {
    assert_eq!(a.shape, b.shape);
    let mut sum = 0.0;
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();

    for i in 0..a_data.len() {
        let diff = a_data[i] - b_data[i];
        sum += diff * diff;
    }
    exp(-gamma * sum)
}"""

rbf_new = """pub fn rbf_kernel(a: &Tensor, b: &Tensor, gamma: f64) -> f64 {
    assert_eq!(a.shape, b.shape);
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();

    let sum: f64 = a_data.iter().zip(b_data.iter()).map(|(a_val, b_val)| {
        let diff = a_val - b_val;
        diff * diff
    }).sum();
    exp(-gamma * sum)
}"""

content = content.replace(rbf_old, rbf_new)

with open('crates/aether-core/src/ml/linalg.rs', 'w') as f:
    f.write(content)
