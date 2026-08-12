import sys

content = open("crates/aether-core/src/ml/linalg.rs").read()

search_mse = """/// Mean Squared Error
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
}"""
replace_mse = """/// Mean Squared Error
pub fn mse(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    assert_eq!(y_true.shape, y_pred.shape);
    let true_data = y_true.data.borrow();
    let pred_data = y_pred.data.borrow();
    let n = true_data.len();

    // ⚡ Bolt: Use iterator zip to elide bounds checks and enable auto-vectorization
    let sum: f64 = true_data
        .iter()
        .zip(pred_data.iter())
        .map(|(&y, &p)| {
            let diff = y - p;
            diff * diff
        })
        .sum();
    sum / n as f64
}"""

search_mae = """/// Mean Absolute Error
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
}"""
replace_mae = """/// Mean Absolute Error
pub fn mae(y_true: &Tensor, y_pred: &Tensor) -> f64 {
    assert_eq!(y_true.shape, y_pred.shape);
    let true_data = y_true.data.borrow();
    let pred_data = y_pred.data.borrow();
    let n = true_data.len();

    // ⚡ Bolt: Use iterator zip to elide bounds checks and enable auto-vectorization
    let sum: f64 = true_data
        .iter()
        .zip(pred_data.iter())
        .map(|(&y, &p)| fabs(y - p))
        .sum();
    sum / n as f64
}"""

content = content.replace(search_mse, replace_mse)
content = content.replace(search_mae, replace_mae)

open("crates/aether-core/src/ml/linalg.rs", "w").write(content)
