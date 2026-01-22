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

#![allow(dead_code)]

// Core modules
pub mod linalg;
pub mod regressor;
pub mod convergence;
pub mod benchmark;

// Extended ML library
pub mod clustering;
pub mod classification;
pub mod neural;

// Re-export key types
pub use regressor::*;
pub use convergence::*;
pub use benchmark::*;
pub use linalg::{Vector, Matrix};
pub use clustering::{KMeans, KMeansResult, DBSCAN, DBSCANResult, AgglomerativeClustering, Linkage};
pub use classification::{LogisticRegression, KNNClassifier, Perceptron, GaussianNB, AdaBoost, NearestCentroid};
pub use neural::{MLP, DenseLayer, Activation, TrainingResult, AdamParams};

