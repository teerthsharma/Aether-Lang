// ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Interpreter - Runtime execution of AEGIS programs
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Executes parsed AEGIS AST, managing:
//! - 3D manifold workspaces
//! - Block geometry computations
//! - Escalating regression benchmarks
//! - Topological convergence detection
//!   ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap;
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
#[cfg(not(feature = "std"))]
use alloc::{format, vec};
#[cfg(not(feature = "std"))]
use alloc::string::ToString;

#[cfg(not(feature = "std"))]
macro_rules! println {
    ($($arg:tt)*) => {};
}

#[cfg(feature = "std")]
use std::boxed::Box;
#[cfg(feature = "std")]
use std::collections::BTreeMap;
#[cfg(feature = "std")]
use std::string::String;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::ast::*;
use aegis_core::aether::{BlockMetadata, DriftDetector, HierarchicalBlockTree};
use aegis_core::manifold::{ManifoldPoint, TimeDelayEmbedder};
use aegis_core::ml::{MLP, KMeans, Activation, OptimizerConfig};
use aegis_core::ml::linalg::LossConfig;
use aegis_core::ml::convolution::Conv2D;
use libm::{fabs, sqrt};

#[cfg(feature = "std")]
// use reqwest::blocking::Client;
#[cfg(feature = "std")]
#[cfg(feature = "std")]
use safetensors::SafeTensors;
#[cfg(feature = "std")]
use std::sync::Arc;
#[cfg(not(feature = "std"))]
use alloc::sync::Arc;

#[cfg(feature = "ml")]
use candle_core::{Device, Tensor as CandleTensor};
#[cfg(feature = "ml")]
use candle_transformers::models::quantized_llama::ModelWeights as LlamaWeights;
#[cfg(feature = "ml")]
use tokenizers::Tokenizer;

/// Embedding dimension
const DIM: usize = 3;

// ═══════════════════════════════════════════════════════════════════════════════
// Runtime Values
// ═══════════════════════════════════════════════════════════════════════════════

/// Runtime value types
#[derive(Debug, Clone)]
pub enum Value {
    /// Numeric value
    Num(f64),
    /// Boolean
    Bool(bool),
    /// String
    Str(String),
    /// 3D Manifold reference
    Manifold(ManifoldHandle),
    /// Geometric block reference  
    Block(BlockHandle),
    /// 3D Point
    Point([f64; DIM]),
    /// Regression result
    RegressionResult(RegressionOutput),
    /// Class Definition
    Class(ClassHandle),
    /// Object Instance
    Object(ObjectHandle),
    /// Native Function (for Standard Library)
    NativeFn(NativeFunction),
    /// Dynamic List (Python-like)
    List(Vec<Value>),
    /// ML Types
    Mlp(Box<MLP>),
    KMeans(Box<KMeans<DIM>>),
    Conv2D(Box<Conv2D>),
    /// Void/Unit
    Unit,
    /// Module Namespace
    /// Module Namespace
    Module(String),
    /// Dynamic Tensor
    Tensor(Arc<Tensor>),
    /// Llama Model (Wrapped)
    #[cfg(feature = "ml")]
    LlamaModel(Arc<LlamaContext>),
}

#[cfg(feature = "ml")]
#[derive(Debug)]
pub struct LlamaContext {
    pub model: LlamaWeights,
    pub tokenizer: Tokenizer,
    pub name: String,
}

/// Simple Dynamic Tensor
#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self { shape, data }
    }
    
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        if self.shape.len() != 2 || other.shape.len() != 2 { return Tensor::new(vec![], vec![]); }
        let m = self.shape[0];
        let k = self.shape[1];
        let k2 = other.shape[0];
        let n = other.shape[1];
        if k != k2 { return Tensor::new(vec![], vec![]); }
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k { sum += self.data[i*k+l] * other.data[l*n+j]; }
                out[i*n+j] = sum;
            }
        }
        Tensor::new(vec![m, n], out)
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
       if self.shape != other.shape { return Tensor::new(vec![], vec![]); }
       Tensor::new(self.shape.clone(), self.data.iter().zip(&other.data).map(|(a,b)| a+b).collect())
    }
    
    pub fn relu(&self) -> Tensor {
        Tensor::new(self.shape.clone(), self.data.iter().map(|x| if *x > 0.0 { *x } else { 0.0 }).collect())
    }

    pub fn softmax(&self) -> Tensor {
        // Row-wise softmax for 2D, or global for 1D
        let mut out = self.data.clone();
        if self.shape.len() == 2 {
            let rows = self.shape[0];
            let cols = self.shape[1];
            for i in 0..rows {
                let row_start = i * cols;
                let row_end = row_start + cols;
                let max_val = self.data[row_start..row_end].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0;
                for j in row_start..row_end {
                    out[j] = libm::exp((out[j] - max_val) as f64) as f32;
                    sum += out[j];
                }
                for j in row_start..row_end { out[j] /= sum; }
            }
        }
        Tensor::new(self.shape.clone(), out)
    }

}

/// Native function pointer type
#[derive(Debug, Clone)]
pub enum NativeFunction {
    MathSin,
    MathCos,
    MathSqrt,
    MathExp,
    TopoBetti,
    Print,
    // ML Constructors
    MlpNew,
    KMeansNew,
    Conv2DNew,
    // Seal Functions
    SealTrain,
    // Tensor Ops
    MlLoadWeights, // (url, key) -> Tensor
    MlMatMul,      // (a, b) -> Tensor
    MlAdd,         // (a, b) -> Tensor
    MlRelu,        // (x) -> Tensor
    MlSoftmax,     // (x) -> Tensor
    MlEmbed,       // (ids, table) -> Tensor
    MlAttention,   // (q, k, v) -> Tensor (AETHER Sparse)
    MlGpuCheck,    // () -> bool
    MlBackward,    // (tensor) -> Tensor (Gradient)
    MlUpdate,      // (weights, grads, lr) -> Tensor (Updated Weights)
    MlLoadLlama,   // (repo_id) -> LlamaModel
    MlGenerate,    // (model, prompt, max_tokens) -> String


}

/// Handle to a manifold workspace
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManifoldHandle(pub usize);

/// Handle to a geometric block
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockHandle(pub usize);

/// Handle to a class definition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassHandle(pub usize);

/// Handle to an object instance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectHandle(pub usize);

/// Class Definition Runtime
#[derive(Debug, Clone)]
pub struct ClassDef {
    pub name: String,
    pub fields: Vec<VarDecl>,
    pub methods: BTreeMap<String, FnDecl>,
}

/// Object Instance Runtime
#[derive(Debug, Clone)]
pub struct ObjectInstance {
    pub class: ClassHandle,
    pub fields: BTreeMap<String, Value>,
}

/// Regression output with convergence info
#[derive(Debug, Clone)]
pub struct RegressionOutput {
    /// Final coefficients
    pub coefficients: [f64; 8],
    /// Number of epochs to converge
    pub epochs: u32,
    /// Final error
    pub final_error: f64,
    /// Converged?
    pub converged: bool,
    /// Betti numbers at convergence
    pub betti: (u32, u32),
}

// ═══════════════════════════════════════════════════════════════════════════════
// Manifold Workspace
// ═══════════════════════════════════════════════════════════════════════════════

/// 3D Manifold workspace containing embedded points
#[derive(Debug)]
pub struct ManifoldWorkspace {
    /// Embedded points in 3D
    pub points: Vec<ManifoldPoint<DIM>>,
    /// Hierarchical block tree for AETHER
    pub block_tree: HierarchicalBlockTree<DIM>,
    /// Drift detector for convergence
    pub drift: DriftDetector<DIM>,
    /// Time-delay embedder
    pub embedder: TimeDelayEmbedder<DIM>,
    /// Current centroid
    pub centroid: [f64; DIM],
}

impl ManifoldWorkspace {
    pub fn new(tau: usize) -> Self {
        Self {
            points: Vec::new(),
            block_tree: HierarchicalBlockTree::new(),
            drift: DriftDetector::new(),
            embedder: TimeDelayEmbedder::new(tau),
            centroid: [0.0; DIM],
        }
    }

    /// Embed raw data into 3D manifold
    pub fn embed_data(&mut self, data: &[f64]) {
        self.points.clear();
        self.embedder.reset();

        for &val in data {
            self.embedder.push(val);
            if let Some(point) = self.embedder.embed() {
                self.points.push(point);
            }
        }

        self.update_centroid();
    }

    /// Update centroid from points
    fn update_centroid(&mut self) {
        if self.points.is_empty() {
            return;
        }

        let mut sum = [0.0; DIM];
        for p in &self.points {
            for (d, s) in sum.iter_mut().enumerate().take(DIM) {
                *s += p.coords[d];
            }
        }

        let n = self.points.len() as f64;
        for (d, s) in sum.iter().enumerate().take(DIM) {
            self.centroid[d] = s / n;
        }
    }

    /// Extract block from index range
    pub fn extract_block(&self, start: usize, end: usize) -> BlockMetadata<DIM> {
        let end = end.min(self.points.len());
        let start = start.min(end);

        if start >= end {
            return BlockMetadata::empty();
        }

        // Convert points to array format for calculation
        // For no_std compatibility with AETHER, we create a temporary buffer
        // Note: In a real alloc environment, we'd pass a slice or Vec
        // Here we assume BlockMetadata::from_points can take a slice of arrays

        let mut block_points = Vec::new();
        for i in start..end {
            block_points.push(self.points[i].coords);
        }

        // Assuming from_points accepts &[ [f64; DIM] ]
        BlockMetadata::from_points(&block_points)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Escalating Regression Engine
// ═══════════════════════════════════════════════════════════════════════════════

/// Regression model types
#[derive(Debug, Clone, Copy)]
pub enum RegressionModel {
    Linear,
    Polynomial { degree: u8 },
    Rbf { gamma: f64 },
}

/// Escalating benchmark system
pub struct EscalatingRegressor {
    /// Current model complexity
    current_level: u32,
    /// Target for regression
    target: Vec<f64>,
    /// Predictions
    predictions: Vec<f64>,
    /// Convergence epsilon
    epsilon: f64,
    /// Betti stability window
    betti_history: Vec<(u32, u32)>,
}

impl EscalatingRegressor {
    pub fn new(epsilon: f64) -> Self {
        Self {
            current_level: 0,
            target: Vec::new(),
            predictions: Vec::new(),
            epsilon,
            betti_history: Vec::new(),
        }
    }

    /// Set target values for regression
    pub fn set_target(&mut self, data: &[f64]) {
        self.target.clear();
        for &v in data {
            self.target.push(v);
        }
    }

    /// Run escalating regression until convergence
    pub fn run_escalating(
        &mut self,
        manifold: &ManifoldWorkspace,
        max_epochs: u32,
    ) -> RegressionOutput {
        let mut coefficients = [0.0f64; 8];
        let mut error = f64::MAX;
        let mut converged = false;
        let mut epochs = 0u32;

        for epoch in 0..max_epochs {
            epochs = epoch;

            // Escalate model complexity
            let model = self.escalate_model(epoch);

            // Fit model
            coefficients = self.fit_model(manifold, &model);

            // Compute error
            error = self.compute_error(manifold, &coefficients, &model);

            // Check topological convergence
            let betti = self.compute_residual_betti(manifold, &coefficients, &model);
            self.betti_history.push(betti);
            // Keep history small
            if self.betti_history.len() > 10 {
                self.betti_history.remove(0);
            }

            if self.is_converged(error, &betti) {
                converged = true;
                break;
            }
        }

        RegressionOutput {
            coefficients,
            epochs,
            final_error: error,
            converged,
            betti: *self.betti_history.last().unwrap_or(&(0, 0)),
        }
    }

    /// Escalate model complexity based on epoch
    fn escalate_model(&self, epoch: u32) -> RegressionModel {
        match epoch {
            0 => RegressionModel::Linear,
            1 => RegressionModel::Polynomial { degree: 2 },
            2 => RegressionModel::Polynomial { degree: 3 },
            3 => RegressionModel::Polynomial { degree: 4 },
            4..=6 => RegressionModel::Rbf {
                gamma: 0.1 * (epoch as f64),
            },
            _ => RegressionModel::Rbf { gamma: 1.0 },
        }
    }

    /// Fit model to manifold data
    fn fit_model(&self, manifold: &ManifoldWorkspace, model: &RegressionModel) -> [f64; 8] {
        let mut coeffs = [0.0f64; 8];

        if manifold.points.is_empty() || self.target.is_empty() {
            return coeffs;
        }

        // Simple least squares for demonstration
        // In production, use proper matrix methods
        match model {
            RegressionModel::Linear => {
                // y = a + b*x
                let n = manifold.points.len().min(self.target.len()) as f64;
                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                let mut sum_xy = 0.0;
                let mut sum_xx = 0.0;

                for (i, p) in manifold.points.iter().enumerate() {
                    if i >= self.target.len() {
                        break;
                    }
                    let x = p.coords[0]; // Use x-axis
                    let y = self.target[i];
                    sum_x += x;
                    sum_y += y;
                    sum_xy += x * y;
                    sum_xx += x * x;
                }

                let denom = n * sum_xx - sum_x * sum_x;
                if fabs(denom) > 1e-10 {
                    coeffs[1] = (n * sum_xy - sum_x * sum_y) / denom;
                    coeffs[0] = (sum_y - coeffs[1] * sum_x) / n;
                }
            }
            RegressionModel::Polynomial { degree } => {
                // Use linear coefficients as approximation
                // For proper poly, would need Vandermonde matrix
                coeffs = self.fit_model(manifold, &RegressionModel::Linear);
                coeffs[*degree as usize] = 0.01;
            }
            RegressionModel::Rbf { gamma: _ } => {
                // RBF kernel - approximate with polynomial
                coeffs = self.fit_model(manifold, &RegressionModel::Polynomial { degree: 3 });
            }
        }

        coeffs
    }

    /// Compute mean squared error
    fn compute_error(
        &self,
        manifold: &ManifoldWorkspace,
        coeffs: &[f64; 8],
        model: &RegressionModel,
    ) -> f64 {
        let mut mse = 0.0;
        let mut count = 0;

        for (i, p) in manifold.points.iter().enumerate() {
            if i >= self.target.len() {
                break;
            }

            let pred = self.predict(p.coords[0], coeffs, model);
            let err = pred - self.target[i];
            mse += err * err;
            count += 1;
        }

        if count > 0 {
            mse /= count as f64;
            sqrt(mse)
        } else {
            f64::MAX
        }
    }

    /// Predict value at x
    fn predict(&self, x: f64, coeffs: &[f64; 8], model: &RegressionModel) -> f64 {
        match model {
            RegressionModel::Linear => coeffs[0] + coeffs[1] * x,
            RegressionModel::Polynomial { degree } => {
                let mut y = coeffs[0];
                let mut x_pow = x;
                for coeff in coeffs.iter().take((*degree as usize).min(7) + 1).skip(1) {
                    y += coeff * x_pow;
                    x_pow *= x;
                }
                y
            }
            RegressionModel::Rbf { gamma: _ } => {
                // Approximate
                self.predict(x, coeffs, &RegressionModel::Polynomial { degree: 3 })
            }
        }
    }

    /// Compute Betti numbers of residual manifold
    fn compute_residual_betti(
        &self,
        manifold: &ManifoldWorkspace,
        coeffs: &[f64; 8],
        model: &RegressionModel,
    ) -> (u32, u32) {
        // Simplified: count sign changes (β₀) and oscillations (β₁)
        let mut sign_changes = 0u32;
        let mut oscillations = 0u32;
        let mut prev_residual = 0.0;
        let mut prev_sign = true;

        for (i, p) in manifold.points.iter().enumerate() {
            if i >= self.target.len() {
                break;
            }

            let pred = self.predict(p.coords[0], coeffs, model);
            let residual = self.target[i] - pred;

            let sign = residual >= 0.0;
            if i > 0 && sign != prev_sign {
                sign_changes += 1;
            }

            // Detect oscillation: residual changes direction
            if i > 1 {
                let delta = residual - prev_residual;
                let prev_delta = prev_residual;
                if (delta > 0.0) != (prev_delta > 0.0) {
                    oscillations += 1;
                }
            }

            prev_residual = residual;
            prev_sign = sign;
        }

        (sign_changes / 2 + 1, oscillations / 4)
    }

    /// Check if converged via topology
    fn is_converged(&self, error: f64, current_betti: &(u32, u32)) -> bool {
        // Error below threshold
        if error < self.epsilon {
            return true;
        }

        // Betti numbers stable for last 3 epochs
        if self.betti_history.len() >= 3 {
            let recent: Vec<&(u32, u32)> = self.betti_history.iter().rev().take(3).collect();

            if recent.iter().all(|b| **b == *current_betti) {
                // Topological stability achieved
                return true;
            }
        }

        false
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Interpreter
// ═══════════════════════════════════════════════════════════════════════════════

/// Runtime environment
pub struct Interpreter {
    /// Variable bindings
    variables: BTreeMap<String, Value>,
    /// Manifold workspaces
    manifolds: Vec<ManifoldWorkspace>,
    /// Block geometries
    blocks: Vec<BlockMetadata<DIM>>,
    /// Class definitions
    classes: Vec<ClassDef>,
    /// Object instances
    objects: Vec<ObjectInstance>,
    /// Sample data (for demo) - dynamic list
    sample_data: Vec<f64>,
}

impl Interpreter {
    pub fn new() -> Self {
        // Generate sample data (sine wave for demo)
        let mut data = Vec::new();
        for i in 0..64 {
            let x = (i as f64) * 0.1;
            data.push(libm::sin(x));
        }

        Self {
            variables: BTreeMap::new(),
            manifolds: Vec::new(),
            blocks: Vec::new(),
            classes: Vec::new(),
            objects: Vec::new(),
            sample_data: data,
        }
    }

    /// Execute a program
    pub fn execute(&mut self, program: &Program) -> Result<Value, String> {
        let mut last_value = Value::Unit;

        for stmt in &program.statements {
            last_value = self.execute_statement(stmt)?;
        }

        Ok(last_value)
    }

    fn execute_statement(&mut self, stmt: &Statement) -> Result<Value, String> {
        match stmt {
            Statement::Manifold(decl) => self.execute_manifold(decl),
            Statement::Block(decl) => self.execute_block(decl),
            Statement::Var(decl) => self.execute_var(decl),
            Statement::Regress(stmt) => self.execute_regress(stmt),
            Statement::Render(stmt) => self.execute_render(stmt),

            // Class
            Statement::Class(decl) => self.execute_class(decl),

            // Import
            Statement::Import(stmt) => self.execute_import(stmt),

            // Control Flow
            Statement::If(stmt) => self.execute_if(stmt),
            Statement::While(stmt) => self.execute_while(stmt),
            Statement::Loop(stmt) => self.execute_seal(stmt),
            // For/Fn/Return/Break/Continue would require more complex runtime state (stack frames)
            // Implementing basic support or stubs for now
            Statement::For(_) => Ok(Value::Unit),
            Statement::Fn(_) => Ok(Value::Unit),
            Statement::Return(_) => Ok(Value::Unit),
            Statement::Break(_) => Ok(Value::Unit),
            Statement::Continue(_) => Ok(Value::Unit),
            Statement::Expr(expr) => {
                self.evaluate_expr(expr)?;
                Ok(Value::Unit)
            }

            Statement::Empty => Ok(Value::Unit),
        }
    }

    fn execute_class(&mut self, decl: &ClassDecl) -> Result<Value, String> {
        let mut methods = BTreeMap::new();
        for m in &decl.methods {
            methods.insert(m.name.clone(), m.clone());
        }

        let class_def = ClassDef {
            name: decl.name.clone(),
            fields: decl.fields.clone(),
            methods,
        };

        let handle = ClassHandle(self.classes.len());
        self.classes.push(class_def);
        self.variables
            .insert(decl.name.clone(), Value::Class(handle));

        Ok(Value::Class(handle))
    }

    #[allow(unused_variables)]
    fn evaluate_new(&mut self, class_name: &String, args: &[Expr]) -> Result<Value, String> {
        // Find class handle
        let class_handle = if let Some(Value::Class(h)) = self.variables.get(class_name) {
            *h
        } else {
            return Err(format!("Class '{}' not found", class_name));
        };

        // Use clone to avoid borrow checker issues with self.classes vs self.execute
        let class_def = self.classes[class_handle.0].clone();

        // Initialize fields with default values
        let mut fields = BTreeMap::new();
        for field in &class_def.fields {
            let val = self.evaluate_expr(&field.value)?;
            fields.insert(field.name.clone(), val);
        }

        // Create object
        let obj_handle = ObjectHandle(self.objects.len());
        self.objects.push(ObjectInstance {
            class: class_handle,
            fields,
        });

        // Run init method if exists
        // Note: Simple method call simulation.
        // Real implementation needs Executing context (stack frame) to bind 'self'.
        // For now, we skip the actual execution of 'init' logic inside the object context
        // to keep this MVP "Python-level" demonstration simple but valid.
        // A full implementation would push a stack frame and bind 'self' to obj_handle.

        Ok(Value::Object(obj_handle))
    }

    fn execute_import(&mut self, stmt: &ImportStmt) -> Result<Value, String> {
        let mod_name = stmt.module.as_str();

        match mod_name {
            "math" => self.import_math(stmt),
            "topology" => self.import_topology(stmt),
            "ml" | "Ml" => self.import_ml(stmt),
            "Seal" => self.import_seal(stmt),
            _ => Err(format!("Module '{}' not found", mod_name)),
        }
    }

    fn import_math(&mut self, stmt: &ImportStmt) -> Result<Value, String> {
        // Built-in Math library
        // Exports: pi, sin, cos, sqrt

        if let Some(symbol) = &stmt.symbol {
            match symbol.as_str() {
                "pi" => {
                    self.variables
                        .insert(String::from("pi"), Value::Num(core::f64::consts::PI));
                }
                "sin" => {
                    self.variables.insert(
                        String::from("sin"),
                        Value::NativeFn(NativeFunction::MathSin),
                    );
                }
                "cos" => {
                    self.variables.insert(
                        String::from("cos"),
                        Value::NativeFn(NativeFunction::MathCos),
                    );
                }
                "sqrt" => {
                    self.variables.insert(
                        String::from("sqrt"),
                        Value::NativeFn(NativeFunction::MathSqrt),
                    );
                }
                "exp" => {
                    self.variables.insert(
                        String::from("exp"),
                        Value::NativeFn(NativeFunction::MathExp),
                    );
                }
                _ => return Err(format!("Symbol '{}' not found in math", symbol)),
            }
        } else {
            // Import all into global namespace
            self.variables
                .insert(String::from("pi"), Value::Num(core::f64::consts::PI));
            self.variables.insert(
                String::from("sin"),
                Value::NativeFn(NativeFunction::MathSin),
            );
            self.variables.insert(
                String::from("cos"),
                Value::NativeFn(NativeFunction::MathCos),
            );
            self.variables.insert(
                String::from("sqrt"),
                Value::NativeFn(NativeFunction::MathSqrt),
            );
        }
        Ok(Value::Unit)
    }

    fn import_topology(&mut self, stmt: &ImportStmt) -> Result<Value, String> {
        // Built-in Topology library
        if let Some(symbol) = &stmt.symbol {
            match symbol.as_str() {
                "Betti" => self.variables.insert(
                    String::from("Betti"),
                    Value::NativeFn(NativeFunction::TopoBetti),
                ),
                _ => return Err(format!("Symbol '{}' not found in topology", symbol)),
            };
        }
        Ok(Value::Unit)
    }

    fn import_ml(&mut self, stmt: &ImportStmt) -> Result<Value, String> {
        if let Some(symbol) = &stmt.symbol {
            match symbol.as_str() {
                 "MLP" => { self.variables.insert(String::from("MLP"), Value::NativeFn(NativeFunction::MlpNew)); },
                 "KMeans" => { self.variables.insert(String::from("KMeans"), Value::NativeFn(NativeFunction::KMeansNew)); },
                 "Conv2D" => { self.variables.insert(String::from("Conv2D"), Value::NativeFn(NativeFunction::Conv2DNew)); },
                 _ => return Err(format!("Symbol '{}' not found in ml", symbol)),
            };
        } else {
             self.variables.insert(String::from("MLP"), Value::NativeFn(NativeFunction::MlpNew));
            self.variables.insert(String::from("KMeans"), Value::NativeFn(NativeFunction::KMeansNew));
            self.variables.insert(String::from("Conv2D"), Value::NativeFn(NativeFunction::Conv2DNew));
            
            // Tensor functions
            self.variables.insert(String::from("load_weights"), Value::NativeFn(NativeFunction::MlLoadWeights));
            self.variables.insert(String::from("matmul"), Value::NativeFn(NativeFunction::MlMatMul));
            self.variables.insert(String::from("add"), Value::NativeFn(NativeFunction::MlAdd));
            self.variables.insert(String::from("relu"), Value::NativeFn(NativeFunction::MlRelu));
            self.variables.insert(String::from("softmax"), Value::NativeFn(NativeFunction::MlSoftmax));
            self.variables.insert(String::from("attention"), Value::NativeFn(NativeFunction::MlAttention));
            self.variables.insert(String::from("gpu_check"), Value::NativeFn(NativeFunction::MlGpuCheck));
            self.variables.insert(String::from("backward"), Value::NativeFn(NativeFunction::MlBackward));
            self.variables.insert(String::from("update"), Value::NativeFn(NativeFunction::MlUpdate));
            self.variables.insert(String::from("load_llama"), Value::NativeFn(NativeFunction::MlLoadLlama));
            self.variables.insert(String::from("generate"), Value::NativeFn(NativeFunction::MlGenerate));
            
            self.variables.insert(String::from("Ml"), Value::Module(String::from("Ml")));
        }
        Ok(Value::Unit)
    }

    fn import_seal(&mut self, stmt: &ImportStmt) -> Result<Value, String> {
        if let Some(symbol) = &stmt.symbol {
             match symbol.as_str() {
                  "train" => { self.variables.insert(String::from("train"), Value::NativeFn(NativeFunction::SealTrain)); },
                  _ => return Err(format!("Symbol '{}' not found in Seal", symbol)),
             };
         } else {
             self.variables.insert(String::from("Seal"), Value::Module(String::from("Seal")));
         }
         Ok(Value::Unit)
    }

    fn execute_manifold(&mut self, decl: &ManifoldDecl) -> Result<Value, String> {
        // Extract tau from initialization
        let tau = self.extract_tau(&decl.init).unwrap_or(3);

        // Create new manifold workspace
        let mut workspace = ManifoldWorkspace::new(tau);

        // Embed sample data
        workspace.embed_data(&self.sample_data);

        // Store workspace
        let handle = ManifoldHandle(self.manifolds.len());
        self.manifolds.push(workspace);

        // Bind variable
        self.variables
            .insert(decl.name.clone(), Value::Manifold(handle));

        Ok(Value::Manifold(handle))
    }

    fn extract_tau(&self, expr: &Expr) -> Option<usize> {
        if let Expr::Call { args, .. } = expr {
            for arg in args {
                if let CallArg::Named { name, value } = arg {
                    if name.as_str() == "tau" {
                        if let Expr::Num(Number::Int(n)) = value {
                            return Some(*n as usize);
                        }
                    }
                }
            }
        }
        None
    }

    fn execute_block(&mut self, decl: &BlockDecl) -> Result<Value, String> {
        // Get range from source expression
        let (manifold_handle, start, end) = self.extract_block_range(&decl.source)?;

        // Extract block from manifold
        if let Some(workspace) = self.manifolds.get(manifold_handle.0) {
            let block = workspace.extract_block(start, end);
            let handle = BlockHandle(self.blocks.len());
            self.blocks.push(block);
            self.variables
                .insert(decl.name.clone(), Value::Block(handle));
            Ok(Value::Block(handle))
        } else {
            let mut err = String::new();
            err.push_str("manifold not found");
            Err(err)
        }
    }

    fn extract_block_range(&self, expr: &Expr) -> Result<(ManifoldHandle, usize, usize), String> {
        match expr {
            Expr::MethodCall { object, args, .. } => {
                let handle = self.get_manifold_handle(object)?;
                let (start, end) = self.extract_range_from_args(args);
                Ok((handle, start, end))
            }
            Expr::Index { object, range } => {
                let handle = self.get_manifold_handle(object)?;
                let start = range.start.as_f64() as usize;
                let end = range.end.as_f64() as usize;
                Ok((handle, start, end))
            }
            _ => {
                let mut err = String::new();
                err.push_str("invalid block source");
                Err(err)
            }
        }
    }

    fn get_manifold_handle(&self, name: &String) -> Result<ManifoldHandle, String> {
        if let Some(Value::Manifold(h)) = self.variables.get(name) {
            Ok(*h)
        } else {
            let mut err = String::new();
            err.push_str("variable is not a manifold");
            Err(err)
        }
    }

    fn extract_range_from_args(&self, args: &[CallArg]) -> (usize, usize) {
        // Default range
        let mut start = 0usize;
        let mut end = 64usize;

        for (i, arg) in args.iter().enumerate() {
            if let CallArg::Positional(Expr::Num(n)) = arg {
                if i == 0 {
                    start = n.as_f64() as usize;
                }
                if i == 1 {
                    end = n.as_f64() as usize;
                }
            } else if let CallArg::Positional(Expr::Range(r)) = arg {
                start = r.start.as_f64() as usize;
                end = r.end.as_f64() as usize;
            }
        }

        (start, end)
    }

    fn execute_var(&mut self, decl: &VarDecl) -> Result<Value, String> {
        let value = self.evaluate_expr(&decl.value)?;
        self.variables.insert(decl.name.clone(), value.clone());
        Ok(value)
    }

    fn execute_regress(&mut self, stmt: &RegressStmt) -> Result<Value, String> {
        let config = &stmt.config;

        // Get epsilon from convergence condition
        let epsilon = match &config.until {
            Some(ConvergenceCond::Epsilon(n)) => n.as_f64(),
            _ => 1e-6,
        };

        // Create regressor
        let mut regressor = EscalatingRegressor::new(epsilon);

        // Set target (use sample data projection for demo)
        regressor.set_target(&self.sample_data);

        // Get first manifold as source
        if let Some(workspace) = self.manifolds.first() {
            let max_epochs = if config.escalate { 100 } else { 10 };
            let result = regressor.run_escalating(workspace, max_epochs);

            Ok(Value::RegressionResult(result))
        } else {
            let mut err = String::new();
            err.push_str("no manifold for regression");
            Err(err)
        }
    }

    fn execute_render(&mut self, stmt: &RenderStmt) -> Result<Value, String> {
        // In no_std, we just acknowledge the render
        let _target = &stmt.target;
        let _config = &stmt.config;

        // TODO: ASCII render or WebGL export
        Ok(Value::Unit)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Control Flow Execution
    // ═══════════════════════════════════════════════════════════════════════════

    fn execute_stmt_block(&mut self, block: &Block) -> Result<Value, String> {
        let mut last_val = Value::Unit;
        for stmt in &block.statements {
            last_val = self.execute_statement(stmt)?;
        }
        Ok(last_val)
    }

    fn execute_if(&mut self, stmt: &IfStmt) -> Result<Value, String> {
        let cond_val = self.evaluate_expr(&stmt.condition)?;

        let is_true = match cond_val {
            Value::Bool(b) => b,
            _ => return Err(String::from("condition must be boolean")),
        };

        if is_true {
            self.execute_stmt_block(&stmt.then_branch)
        } else if let Some(else_branch) = &stmt.else_branch {
            self.execute_stmt_block(else_branch)
        } else {
            Ok(Value::Unit)
        }
    }

    fn execute_while(&mut self, stmt: &WhileStmt) -> Result<Value, String> {
        let mut last_val = Value::Unit;

        loop {
            let cond_val = self.evaluate_expr(&stmt.condition)?;
            let is_true = match cond_val {
                Value::Bool(b) => b,
                _ => return Err(String::from("condition must be boolean")),
            };

            if !is_true {
                break;
            }

            last_val = self.execute_stmt_block(&stmt.body)?;
        }

        Ok(last_val)
    }

    /// Execute 'seal' loop - runs until topological convergence
    fn execute_seal(&mut self, stmt: &LoopStmt) -> Result<Value, String> {
        // In a full implementation, this would check Betti numbers of the active manifold
        // For now, we limit iterations to avoiding hanging if no break condition
        let max_iters = 1000;
        let mut last_val = Value::Unit;

        for _ in 0..max_iters {
            last_val = self.execute_stmt_block(&stmt.body)?;
            // If we had a break signal, we would handle it here
        }

        Ok(last_val)
    }

    fn evaluate_expr(&mut self, expr: &Expr) -> Result<Value, String> {
        match expr {
            Expr::Num(n) => Ok(Value::Num(n.as_f64())),
            Expr::Bool(b) => Ok(Value::Bool(*b)),
            Expr::Str(s) => Ok(Value::Str(s.clone())),
            Expr::Ident(name) => {
                if let Some(v) = self.variables.get(name) {
                    Ok(v.clone())
                } else {
                    Ok(Value::Unit)
                }
            }
            Expr::FieldAccess { object, field } => self.evaluate_field_access(object, field),
            Expr::Call { name, args } => self.evaluate_call(name, args),
            Expr::New { class, args } => self.evaluate_new(class, args),
            Expr::List(elements) => self.evaluate_list(elements),
            Expr::MethodCall {
                object,
                method,
                args,
            } => self.evaluate_method_call(object, method, args),
            Expr::Range(_) => Err(String::from("Ranges cannot be evaluated directly as values")),
            _ => Ok(Value::Unit),
        }
    }

    fn evaluate_list(&mut self, elements: &Vec<Expr>) -> Result<Value, String> {
        let mut values = Vec::new();
        for expr in elements {
            values.push(self.evaluate_expr(expr)?);
        }
        Ok(Value::List(values))
    }

    fn evaluate_call(&mut self, name: &Ident, args: &Vec<CallArg>) -> Result<Value, String> {
        // Check if variable exists (could be function or native fn)
        if let Some(val) = self.variables.get(name) {
            match val.clone() {
                Value::NativeFn(func) => self.execute_native_fn(func, args),
                _ => Ok(Value::Unit), // Todo: Handle user defined functions
            }
        } else {
            Ok(Value::Unit)
        }
    }

    fn execute_native_fn(
        &mut self,
        func: NativeFunction,
        args: &[CallArg],
    ) -> Result<Value, String> {
        // Helper to get first arg as f64
        let mut get_f64 = |args: &[CallArg]| -> Result<f64, String> {
            if let Some(CallArg::Positional(Expr::Num(n))) = args.first() {
                Ok(n.as_f64())
            } else if let Some(CallArg::Positional(expr)) = args.first() {
                // Evaluate first
                let val = self.evaluate_expr(expr)?;
                if let Value::Num(n) = val {
                    Ok(n)
                } else {
                    Err(String::from("Expected number rule"))
                }
            } else {
                Err(String::from("Expected number"))
            }
        };

        match func {
            NativeFunction::MathSin => {
                let n = get_f64(args)?;
                Ok(Value::Num(libm::sin(n)))
            }
            NativeFunction::MathCos => {
                let n = get_f64(args)?;
                Ok(Value::Num(libm::cos(n)))
            }
            NativeFunction::MathSqrt => {
                let n = get_f64(args)?;
                Ok(Value::Num(libm::sqrt(n)))
            }
            NativeFunction::MathExp => {
                let n = get_f64(args)?;
                Ok(Value::Num(libm::exp(n)))
            }
            NativeFunction::TopoBetti => {
                // Mock calculation
                Ok(Value::List(vec![Value::Num(1.0), Value::Num(0.0)]))
            }
            NativeFunction::Print => {
                // No-op in no_std for now, or write to serial
                Ok(Value::Unit)
            }
            NativeFunction::MlpNew => {
                 let lr = get_f64(args).unwrap_or(0.01);
                 let config = OptimizerConfig::SGD { learning_rate: lr, momentum: 0.9 };
                 Ok(Value::Mlp(Box::new(MLP::new(config, LossConfig::MSE))))
            }
            NativeFunction::KMeansNew => {
                 let k = get_f64(args).unwrap_or(2.0) as usize;
                 Ok(Value::KMeans(Box::new(KMeans::new(k))))
            }
            NativeFunction::Conv2DNew => {
                 Ok(Value::Conv2D(Box::new(Conv2D::new(1, 1, 3, 1, 1, Activation::ReLU))))
            }

            NativeFunction::SealTrain => Err("Seal training via native fn not implemented yet".into()),
            // LLM Operations
            NativeFunction::MlLoadWeights => {
                #[cfg(feature = "std")]
                {
                    // args: url, key
                    let url = match args.get(0) { Some(CallArg::Positional(Expr::Str(s))) => s, _ => return Err("Expected URL string".into()) };
                    let key = match args.get(1) { Some(CallArg::Positional(Expr::Str(s))) => s, _ => return Err("Expected Key string".into()) };
                    
                    // Simple blocking download
                    let bytes = reqwest::blocking::get(url)
                        .map_err(|e| format!("Network error: {}", e))?
                        .bytes()
                        .map_err(|e| format!("Bytes error: {}", e))?;
                    
                    let tensors = SafeTensors::deserialize(&bytes).map_err(|e| format!("SafeTensors error: {}", e))?;
                    let tensor_view = tensors.tensor(key).map_err(|e| format!("Key '{}' not found", key))?;
                    
                    // Convert to our simple Tensor (f32)
                    let shape = tensor_view.shape().to_vec();
                    // Assume f32 for now. In real app, we need to handle Dtypes.
                    // Safetensors gives byte slice.
                     let data_bytes = tensor_view.data();
                     // Safety: Align check needed ideally.
                     let data: Vec<f32> = list_from_u8(data_bytes);
                     
                     Ok(Value::Tensor(Arc::new(Tensor::new(shape, data))))
                }
                #[cfg(not(feature = "std"))]
                Err("MlLoadWeights requires std".into())
            }
            NativeFunction::MlMatMul => {
                let a = match self.get_arg_val(args, 0)? { Value::Tensor(t) => t, _ => return Err("Arg 0 must be Tensor".into()) };
                let b = match self.get_arg_val(args, 1)? { Value::Tensor(t) => t, _ => return Err("Arg 1 must be Tensor".into()) };
                Ok(Value::Tensor(Arc::new(a.matmul(&b))))
            }
            NativeFunction::MlAdd => {
                let a = match self.get_arg_val(args, 0)? { Value::Tensor(t) => t, _ => return Err("Arg 0 must be Tensor".into()) };
                let b = match self.get_arg_val(args, 1)? { Value::Tensor(t) => t, _ => return Err("Arg 1 must be Tensor".into()) };
                Ok(Value::Tensor(Arc::new(a.add(&b))))
            }
            NativeFunction::MlRelu => {
                let a = match self.get_arg_val(args, 0)? { Value::Tensor(t) => t, _ => return Err("Arg 0 must be Tensor".into()) };
                Ok(Value::Tensor(Arc::new(a.relu())))
            }
            NativeFunction::MlSoftmax => {
                let a = match self.get_arg_val(args, 0)? { Value::Tensor(t) => t, _ => return Err("Arg 0 must be Tensor".into()) };
                Ok(Value::Tensor(Arc::new(a.softmax())))
            }
            NativeFunction::MlEmbed => {
                 // args: ids (Tensor), table (Tensor)
                 // Simplified embedding lookup
                 let ids = match self.get_arg_val(args, 0)? { Value::Tensor(t) => t, _ => return Err("Arg 0 must be Tensor ids".into()) };
                 let table = match self.get_arg_val(args, 1)? { Value::Tensor(t) => t, _ => return Err("Arg 1 must be Tensor table".into()) };
                 
                 // manual embedding
                 let hidden_dim = table.shape[1];
                 let seq_len = ids.shape[0]; // assuming 1D ids
                 let mut out = vec![0.0; seq_len * hidden_dim];
                 
                 for i in 0..seq_len {
                     let id = ids.data[i] as usize;
                     if id < table.shape[0] {
                         for j in 0..hidden_dim {
                             out[i * hidden_dim + j] = table.data[id * hidden_dim + j];
                         }
                     }
                 }
                 Ok(Value::Tensor(Arc::new(Tensor::new(vec![seq_len, hidden_dim], out))))
            }
            NativeFunction::MlAttention => {
                let q = match self.get_arg_val(args, 0)? { Value::Tensor(t) => t, _ => return Err("Arg 0 (Q) must be Tensor".into()) };
                let k = match self.get_arg_val(args, 1)? { Value::Tensor(t) => t, _ => return Err("Arg 1 (K) must be Tensor".into()) };
                let v = match self.get_arg_val(args, 2)? { Value::Tensor(t) => t, _ => return Err("Arg 2 (V) must be Tensor".into()) };
                
                // AETHER Sparse Attention Implementation
                // 1. We treat rows of Q and K as points in the manifold.
                // 2. We use AETHER blocks to prune non-interacting regions.
                // Note: For this demo, we assume D <= 64 to use BlockMetadata<64>, OR we just force 64.
                
                let seq_len = q.shape[0];
                let d_k = q.shape[1]; 
                
                // Simplified Sparse Logic:
                // Instead of computing Q*K^T (N^2), we:
                // i. Iterating Q rows.
                // ii. Only attending to K rows that are topologically close?
                // For exact attention we need all, but "Sparse-Event" implies approximation.
                
                // Naive implementation of "Flash Attention" style tiling or just standard attention for correctness first.
                // To show "Aether" usage, we would build a tree.
                
                // Standard Attention: Softmax(Q*K^T / sqrt(d_k)) * V
                // [N, D] * [D, N] -> [N, N]
                // [N, N] * [N, D] -> [N, D]
                
                let k_t = self.transpose_tensor(&k);
                let scores = q.matmul(&k_t);
                
                // Scale
                let scale = 1.0 / (d_k as f32).sqrt();
                let scaled = Tensor::new(scores.shape.clone(), scores.data.iter().map(|x| x * scale).collect());
                
                // Softmax
                let probs = scaled.softmax();
                
                // Output
                let out = probs.matmul(&v);
                
                Ok(Value::Tensor(Arc::new(out)))
            }
            NativeFunction::MlGpuCheck => {
                #[cfg(feature = "std")]
                {
                    // Check for WGPU adapter
                    let instance = wgpu::Instance::default();
                    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));
                     match adapter {
                        Some(a) => {
                             println!("GPU Adapter Found: {:?}", a.get_info());
                             Ok(Value::Bool(true))
                        },
                        None => {
                            println!("No GPU Adapter found.");
                            Ok(Value::Bool(false))
                        }
                    }
                }
                #[cfg(not(feature = "std"))]
                Ok(Value::Bool(false))
            }
            NativeFunction::MlBackward => {
                // Simulating autograd: returns random gradients of same shape
                let t = match self.get_arg_val(args, 0)? { Value::Tensor(t) => t, _ => return Err("Arg 0 must be Tensor".into()) };
                let len = t.data.len();
                // Random gradients -0.01 to 0.01
                // For demo visualization, let's make them deterministic enough to show change
                let grads: Vec<f32> = t.data.iter().map(|v| v * 0.01).collect(); // specific 'gradient'
                Ok(Value::Tensor(Arc::new(Tensor::new(t.shape.clone(), grads))))
            }
            NativeFunction::MlUpdate => {
                // SGD: w = w - lr * g
                let w = match self.get_arg_val(args, 0)? { Value::Tensor(t) => t, _ => return Err("Arg 0 (Weights) must be Tensor".into()) };
                let g = match self.get_arg_val(args, 1)? { Value::Tensor(t) => t, _ => return Err("Arg 1 (Grads) must be Tensor".into()) };
                let lr = match self.get_arg_val(args, 2)? { Value::Num(n) => n as f32, _ => 0.01 }; // Default lr
                
                if w.shape != g.shape { return Err(format!("Shape mismatch {:?} vs {:?}", w.shape, g.shape)); }
                
                let new_data: Vec<f32> = w.data.iter().zip(&g.data).map(|(weight, grad)| weight - lr * grad).collect();
                Ok(Value::Tensor(Arc::new(Tensor::new(w.shape.clone(), new_data))))
            }
            NativeFunction::MlLoadLlama => {
                #[cfg(feature = "ml")]
                {
                   let repo_id = match args.get(0) { Some(CallArg::Positional(Expr::Str(s))) => s.clone(), _ => "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string() };
                   
                   println!("Initializing hf-hub api for {}...", repo_id);
                   
                   let api = hf_hub::api::sync::Api::new().map_err(|e| format!("API error: {}", e))?;
                   let repo = api.model(repo_id);
                   
                   println!("Fetching tokenizer...");
                   let tokenizer_path = repo.get("tokenizer.json").map_err(|e| format!("Tokenizer fetch error: {}", e))?;
                   let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| format!("Tokenizer load error: {}", e))?;
                   
                   println!("Fetching model weights (safetensors/gguf)...");
                   // Try quantized gguf first if user asks, but standard safetensors for simplicity in this demo
                   // For TinyLlama, standard weights:
                   
                   // Let's force GGUF for the demo as it's faster on CPU.
                   let quantization_repo = api.model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string());
                   let gguf_path = quantization_repo.get("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf").map_err(|e| format!("GGUF download failed: {}", e))?;
                   
                   // Load generic GGUF
                   let mut file = std::fs::File::open(&gguf_path).map_err(|e| e.to_string())?;
                   let model = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(&mut file, &mut file).map_err(|e| e.to_string())?;
                   
                   println!(">> Llama Model Loaded Successfully via Candle!");
                   
                   Ok(Value::LlamaModel(Arc::new(LlamaContext { model, tokenizer, name: repo_id })))
                }
                #[cfg(not(feature = "ml"))]
                Err("Requires ml feature".into())
            }
            NativeFunction::MlGenerate => {
                 #[cfg(feature = "ml")]
                 {
                     let ctx = match self.get_arg_val(args, 0)? { Value::LlamaModel(c) => c, _ => return Err("Arg 0 must be LlamaModel".into()) };
                     let prompt = match args.get(1) { Some(CallArg::Positional(Expr::Str(s))) => s, _ => "" };
                     let max_tokens = match args.get(2) { Some(CallArg::Positional(Expr::Num(n))) => n.as_f64() as usize, _ => 50 };
                     
                     println!("Generating response for prompt: '{}'", prompt); 
                     
                     let mut tokens = ctx.tokenizer.encode(prompt, true).map_err(|e| e.to_string())?.get_ids().to_vec();
                     let mut output = String::new();
                     
                     for _ in 0..max_tokens {
                         let input = CandleTensor::new(tokens.as_slice(), &Device::Cpu).map_err(|e| e.to_string())?.unsqueeze(0).map_err(|e| e.to_string())?;
                         let logits = ctx.model.forward(&input, tokens.len()).map_err(|e| e.to_string())?;
                         let logits = logits.squeeze(0).map_err(|e| e.to_string())?;
                         
                         let next_token = logits.argmax(0).map_err(|e| e.to_string())?.to_scalar::<u32>().map_err(|e| e.to_string())?;
                         tokens.push(next_token);
                         
                         if let Some(text) = ctx.tokenizer.decode(&[next_token], true).ok() {
                             output.push_str(&text);
                             print!("{}", text); 
                             use std::io::Write;
                             std::io::stdout().flush().ok();
                         }
                     }
                     println!(); 
                     Ok(Value::Str(output))
                 }
                 #[cfg(not(feature = "ml"))]
                 Err("Requires ml feature".into())
            }
        }
    }
    
    fn transpose_tensor(&self, t: &Tensor) -> Tensor {
         if t.shape.len() != 2 { return Tensor::new(vec![], vec![]); }
         let rows = t.shape[0];
         let cols = t.shape[1];
         let mut out = vec![0.0; rows * cols];
         for i in 0..rows {
             for j in 0..cols {
                 out[j * rows + i] = t.data[i * cols + j];
             }
         }
         Tensor::new(vec![cols, rows], out)
    }

    
    fn get_arg_val(&mut self, args: &[CallArg], idx: usize) -> Result<Value, String> {
        if let Some(CallArg::Positional(expr)) = args.get(idx) {
            self.evaluate_expr(expr)?;
        } else {
            return Err(format!("Missing argument {}", idx));
        }

        // Seal.train(model, data, targets)
        if args.len() < 3 {
            return Err(String::from("Seal.train requires model, data, and targets"));
        }

        // 1. Get Model
        let model_val = if let CallArg::Positional(expr) = &args[0] {
             self.evaluate_expr(expr)?
        } else {
             return Err(String::from("Model must be positional arg"));
        };

        // 2. Get Data
        let data_val = if let CallArg::Positional(expr) = &args[1] {
             self.evaluate_expr(expr)?
        } else {
             return Err(String::from("Data must be positional arg"));
        };

        // 3. Get Targets
        let targets_val = if let CallArg::Positional(expr) = &args[2] {
             self.evaluate_expr(expr)?
        } else {
             return Err(String::from("Targets must be positional arg"));
        };

        match model_val {
            Value::Mlp(mut mlp) => {
                 // Convert data to fixed size arrays (MAX_NEURONS constraint for now)
                 // In a real impl, we'd handle this better or use Alloc
                 let x = self.value_to_tensor(&data_val)?;
                 let y = self.value_to_tensor(&targets_val)?;
                 let n_samples = x.len();
                 let output_size = 1; // Infer from y?

                 let x_tensors: Vec<aegis_core::ml::Tensor> = x.iter().map(|arr| aegis_core::ml::Tensor::new(&arr.to_vec(), &[64])).collect();
                 let y_tensors: Vec<aegis_core::ml::Tensor> = y.iter().map(|arr| aegis_core::ml::Tensor::new(&arr.to_vec(), &[1])).collect();
                 let result = mlp.fit(&x_tensors, &y_tensors, 1000);
                 
                 // Update the model variable if it was passed by reference-ish logic?
                 // Since we have ownership of the Box<MLP> here, we need to put it back?
                 // Calling evaluate_expr gave us a CLONE of the value (Value derives Clone).
                 // So `mlp` is a clone. Training it won't affect the original variable!
                 // WE NEED MUTABLE ACCESS TO VARIABLES.
                 // This is a limitation of the current `evaluate_expr` returning clones.
                 
                 // FIX: Re-assign the trained model to the variable *if* the first arg was an identifier.
                 if let CallArg::Positional(Expr::Ident(name)) = &args[0] {
                     self.variables.insert(name.clone(), Value::Mlp(mlp));
                 }
                 
                 Ok(Value::Num(result.final_loss))
            }
            _ => Err(String::from("Seal.train only supports MLP currently")),
        }
    }

    fn value_to_tensor(&self, val: &Value) -> Result<Vec<[f64; 64]>, String> { // 64 = MAX_NEURONS
        match val {
            Value::List(rows) => {
                let mut tensor = Vec::new();
                for row in rows {
                    if let Value::List(cols) = row {
                        let mut arr = [0.0; 64];
                        for (i, v) in cols.iter().enumerate().take(64) {
                            if let Value::Num(n) = v {
                                arr[i] = *n;
                            }
                        }
                        tensor.push(arr);
                    } else {
                        return Err(String::from("Data must be 2D list"));
                    }
                }
                Ok(tensor)
            }
            _ => Err(String::from("Data must be a List")),
        }
    }

    fn evaluate_method_call(
        &mut self,
        object_name: &String,
        method: &String,
        args: &[CallArg],
    ) -> Result<Value, String> {
        // DEBUG
        println!("Method call: {}.{}", object_name, method);

        let val = if let Some(v) = self.variables.get(object_name) {
            v.clone()
        } else {
            return Err(format!("Object '{}' not found", object_name));
        };

        println!("Object Type: {:?}", val);

        match val {
            Value::List(mut list) => {
                // List methods: push, pop, len
                let res = match method.as_str() {
                    "push" => {
                        if let Some(CallArg::Positional(expr)) = args.first() {
                            let val = self.evaluate_expr(expr)?;
                            list.push(val);
                            // Update variable
                            self.variables
                                .insert(object_name.clone(), Value::List(list));
                            Ok(Value::Unit)
                        } else {
                            Err(String::from("push requires 1 argument"))
                        }
                    }
                    "pop" => {
                        let val = list.pop().unwrap_or(Value::Unit);
                        self.variables
                            .insert(object_name.clone(), Value::List(list));
                        Ok(val)
                    }
                    "len" => Ok(Value::Num(list.len() as f64)),
                    _ => Err(format!("Method '{}' not found on List", method)),
                };
                res
            }
            Value::Object(handle) => {
                // User-defined methods
                let obj = &self.objects[handle.0];
                let class_def = &self.classes[obj.class.0];

                if let Some(_method_decl) = class_def.methods.get(method) {
                    // TODO: Implement user-defined method execution
                    // This requires setting up a stack frame, binding 'self', etc.
                    // For now, return Unit to prevent crash
                    Ok(Value::Unit)
                } else {
                    Err(format!(
                        "Method '{}' not found on class '{}'",
                        method, class_def.name
                    ))
                }
            }
            Value::Mlp(mut boxed_mlp) => {
                 match method.as_str() {
                     "add_layer" => {
                        // args: in, out
                        let in_size = 2; // simplified
                        let out_size = 1;
                        boxed_mlp.add_layer(in_size, out_size, Activation::ReLU, None);
                        self.variables.insert(object_name.clone(), Value::Mlp(boxed_mlp));
                        Ok(Value::Unit)
                     }
                     "train" => {
                         // Mock training
                         Ok(Value::Num(0.1))
                     }
                     "predict" => {
                         // Mock inference
                         Ok(Value::Num(0.0))
                     }
                     _ => Err(format!("Method '{}' not found on MLP", method)),
                 }
            }
            Value::Module(mod_name) => {
                 match (mod_name.as_str(), method.as_str()) {
                     ("Ml", "MLP") => self.execute_native_fn(NativeFunction::MlpNew, args),
                     ("Ml", "KMeans") => self.execute_native_fn(NativeFunction::KMeansNew, args),
                     ("Ml", "Conv2D") => self.execute_native_fn(NativeFunction::Conv2DNew, args),
                     ("Seal", "train") => self.execute_native_fn(NativeFunction::SealTrain, args),
                     _ => Err(format!("Method '{}' not found in module '{}'", method, mod_name)),
                 }
            }
            _ => Err("Type cannot handle method calls".to_string()),
        }
    }

    fn evaluate_field_access(&self, object: &String, field: &String) -> Result<Value, String> {
        if let Some(Value::Object(handle)) = self.variables.get(object) {
            if let Some(obj) = self.objects.get(handle.0) {
                if let Some(val) = obj.fields.get(field) {
                    return Ok(val.clone());
                }
            }
        }

        if let Some(Value::Block(handle)) = self.variables.get(object) {
            if let Some(block) = self.blocks.get(handle.0) {
                match field.as_str() {
                    "center" => Ok(Value::Point(block.centroid)),
                    "spread" => Ok(Value::Num(block.radius)),
                    _ => Ok(Value::Unit),
                }
            } else {
                Ok(Value::Unit)
            }
        } else if let Some(Value::Manifold(handle)) = self.variables.get(object) {
            if let Some(workspace) = self.manifolds.get(handle.0) {
                match field.as_str() {
                    "center" => Ok(Value::Point(workspace.centroid)),
                    _ => Ok(Value::Unit),
                }
            } else {
                Ok(Value::Unit)
            }
        } else if let Some(Value::Module(name)) = self.variables.get(object) {
             match (name.as_str(), field.as_str()) {
                 ("Seal", "train") => Ok(Value::NativeFn(NativeFunction::SealTrain)),
                 ("Ml", "MLP") => Ok(Value::NativeFn(NativeFunction::MlpNew)),
                 ("Ml", "KMeans") => Ok(Value::NativeFn(NativeFunction::KMeansNew)),
                 ("Ml", "Conv2D") => Ok(Value::NativeFn(NativeFunction::Conv2DNew)),
                 _ => Ok(Value::Unit),
             }
        } else {
             Ok(Value::Unit)
        }
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

fn list_from_u8(bytes: &[u8]) -> Vec<f32> {
    let mut data = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().unwrap();
        data.push(f32::from_le_bytes(arr));
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_import_and_usage() {
        let mut interpreter = Interpreter::new();
        let script = "import ml\nlet nn = MLP(0.01)";
        let mut parser = crate::parser::Parser::new(script);
        let program = parser.parse().unwrap();
        
        let result = interpreter.execute(&program);
        assert!(result.is_ok());
        
        match interpreter.variables.get("nn") {
            Some(Value::Mlp(_)) => assert!(true),
            _ => assert!(false, "Expected MLP value"),
        }
    }

    #[test]
    fn test_seal_demo() {
        let mut interpreter = Interpreter::new();
        let script = r#"
            import Ml
            import Seal

            let network = Ml.MLP(0.1)
            network.add_layer(2, 4, "relu")
            network.add_layer(4, 1, "sigmoid")

            let data = [[0, 0], [0, 1], [1, 0], [1, 1]]
            let targets = [[0], [1], [1], [0]]

            let loss = Seal.train(network, data, targets)
        "#;
        
        let mut parser = crate::parser::Parser::new(script);
        let program = parser.parse().expect("Failed to parse seal demo");
        
        let result = interpreter.execute(&program);
        assert!(result.is_ok(), "Execution failed: {:?}", result.err());
        
        // Check finding loss variable
        match interpreter.variables.get("loss") {
             Some(Value::Num(loss)) => {
                 println!("Final loss: {}", loss);
                 assert!(*loss >= 0.0);
             }
             _ => panic!("Expected loss to be a number"),
        }
    }
}
