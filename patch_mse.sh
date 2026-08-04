#!/bin/bash
sed -i 's/let mut sum = 0.0;/let sum: f64 = true_data.iter().zip(pred_data.iter()).map(|(y, p)| { let diff = y - p; diff * diff }).sum();/' crates/aether-core/src/ml/linalg.rs
