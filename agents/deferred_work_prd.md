# AEGIS Deep Learning: Deferred Features & Gap Analysis (PRD)
**Version:** 0.1.0-Deferred
**Status:** Backlog
**Parent:** AEGIS Deep Learning Roadmap

---

## 1. Executive Summary
This document outlines the features from the original "AEGIS Deep Learning Roadmap" that are **NOT** being implemented in Phase 1, as well as critical bugs and limitations identified in the current codebase (`aegis-core`). These items are deferred to future phases (Synapse, Acceleration).

## 2. Deferred Scope (What We Are NOT Doing Yet)

### 2.1 Hardware Acceleration (`aegis-compute`)
- **Status**: Deferred to Phase 2.
- **Description**: We are strictly sticking to CPU-based execution for Phase 1. `wgpu` integration and CUDA kernels are out of scope.
- **Impact**: Training performance will be slow; real-time topology (manifolds) will happen on CPU.

### 2.2 The "Synapse" Layer Library
- **Status**: Deferred to Phase 3.
- **Description**: High-level modular layers (`TransformerBlock`, `LSTM`, `GRU`, `Dropout`, `BatchNorm`) are not being built yet.
- **Current State**: We are focusing only on the low-level `Autograd` engine and a basic `Dense` layer refactor.
- **Impact**: Users cannot easily build complex architectures like GPT-2 or ResNet without manually defining the graph.

### 2.3 Advanced Optimizers
- **Status**: Partially Deferred.
- **Description**: While `Adam` code exists in fragments, we are primarily supporting `SGD` for the Phase 1 Autograd rollout. `RMSProp` and `Adagrad` are not planned.
- **Impact**: Slower convergence on complex landscapes.

### 2.4 Modular Loss Functions
- **Status**: Deferred.
- **Description**: `CrossEntropy`, `KL-Div`, and `PersistenceLoss` will be implemented as functions first, not as modular objects with state.

### 2.5 Data Loaders
- **Status**: Deferred.
- **Description**: No `DataLoader` with batching, shuffling, or pre-fetching. Data will be passed as raw Tensors.

## 3. Known Bugs & Critical Limitations

### 3.1 severe Scalability Limitation (`MAX_NEURONS`)
- **Severity**: 🔴 CRITICAL
- **Location**: `aegis-core/src/ml/neural.rs`
- **Issue**: The current implementation mandates `const MAX_NEURONS: usize = 64;`.
- **Consequence**: **It is impossible to train on MNIST** (which requires 784 input neurons). The library is currently limited to toy problems (XOR, Iris).
- **Fix Required**: Move from stack-allocated arrays `[f64; 64]` to heap-allocated dynamic `Vec<f64>` or the new `Tensor` struct.

### 3.2 Stack Overflow Risk
- **Severity**: 🔴 CRITICAL
- **Location**: `aegis-core/src/ml/linalg.rs`, `neural.rs`
- **Issue**: Large structs (`Matrix` with `[f64; 32][32]`) are passed by value or allocated on the stack. In a `no_std` kernel environment, this will blow the stack immediately with deep networks.
- **Fix Required**: Use `Box` or `Rc` for storage, or strict reference passing.

### 3.3 Dead Code: Disconnected `Adam` Optimizer
- **Severity**: 🟡 MEDIUM
- **Location**: `aegis-core/src/ml/neural.rs`
- **Issue**: `struct AdamState` and `fn adam_update` exist but are **not integrated** into the `MLP` struct. `MLP` hardcodes `layer.backward(...)`, which looks like standard SGD.
- **Consequence**: Users cannot actually use Adam despite the code being there.

### 3.4 Poor Randomness Initialization
- **Severity**: 🟢 LOW
- **Location**: `DenseLayer::new`
- **Issue**: Uses a crude Linear Congruential Generator (LCG) seeded with `42`.
- **Consequence**: All models initialize exactly the same way (deterministic), but the distribution quality is poor.

### 3.5 Manual Backpropagation
- **Severity**: 🟡 MEDIUM
- **Location**: `DenseLayer::backward`
- **Issue**: The backward pass is hardcoded for the specific `DenseLayer` math. It does not support automatic differentiation for arbitrary graphs (which is the goal of Phase 1).

## 4. Next Steps
The immediate priority is to address **3.1 (MAX_NEURONS)** and **3.5 (Manual Backprop)** by implementing the new `Tensor` and `Autograd` system outlined in `implementation_plan.md`.
