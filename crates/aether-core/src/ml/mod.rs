//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS ML Engine - Complete Machine Learning Library
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! A comprehensive ML library built from scratch for AEGIS:
//!
//! Core Modules:
//!   - linalg: Vector/Matrix ops, loss functions, gradients
//!   - regressor: Manifold regression (Linear, Polynomial, RBF, GP, Geodesic)
//!   - convergence: Topological convergence via Betti numbers
//!   - clustering: K-Means, DBSCAN, Hierarchical, Auto-K
//!   - classification: LogisticRegression, KNN, Perceptron, NaiveBayes, AdaBoost
//!   - neural: MLP, DenseLayer, Activations, Adam/SGD optimizers
//!   - benchmark: Escalating benchmark system
//!
//! All algorithms use seal-loop style convergence where applicable.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
// Aether-Lang — invented by Teerth Sharma
// https://github.com/teerthsharma/Aether-Lang
// Copyright (c) 2026 Teerth Sharma. All Rights Reserved.
// ═══════════════════════════════════════════════════════════════════════════════
//

#![allow(dead_code)]

// Core modules
pub mod autograd;
pub mod benchmark;
pub mod convergence;
pub mod convolution;
pub mod linalg;
pub mod regressor;
pub mod tensor;

// Extended ML library
pub mod classification;
pub mod clustering;
pub mod neural;

// Re-export key types
pub use benchmark::*;
pub use classification::{
    AdaBoost, GaussianNB, KNNClassifier, LogisticRegression, NearestCentroid, Perceptron,
};
pub use clustering::{
    AgglomerativeClustering, DBSCANResult, KMeans, KMeansResult, Linkage, DBSCAN,
};
pub use convergence::*;
// pub use linalg::{Matrix, Vector}; // Removed
pub use neural::{Activation, DenseLayer, OptimizerConfig, TrainingResult, MLP};
pub use regressor::*;
pub use tensor::Tensor;
pub mod gossip;
