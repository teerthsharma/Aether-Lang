// ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Interpreter - Runtime execution of AEGIS programs
// ═══════════════════════════════════════════════════════════════════════════════
//!
//! Executes parsed AEGIS AST, managing:
//! - 3D manifold workspaces
//! - Block geometry computations
//! - Escalating regression benchmarks
//! - Topological convergence detection
// ═══════════════════════════════════════════════════════════════════════════════

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
use aether_core::aether::{BlockMetadata, DriftDetector, HierarchicalBlockTree};
use aether_core::manifold::{ManifoldPoint, TimeDelayEmbedder};
use aether_core::ml::{MLP, KMeans, Activation, OptimizerConfig};
use aether_core::ml::tensor::Tensor;
use aether_core::ml::linalg::LossConfig;
use aether_core::ml::convolution::Conv2D;
use libm::{fabs, sqrt};

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
    Module(String),
    /// Dynamic Tensor
    Tensor(Tensor),
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
    MlLoadWeights,
    MlMatMul,
    MlAdd,
    MlForward,
    MlRelu,
    MlSoftmax,
    MlEmbed,
    MlAttention,
    MlGpuCheck,
    MlBackward,
    MlUpdate,
    MlLoadLlama,
    MlGenerate,
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

        let mut block_points = Vec::new();
        for i in start..end {
            block_points.push(self.points[i].coords);
        }

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
            let model = self.escalate_model(epoch);
            coefficients = self.fit_model(manifold, &model);
            error = self.compute_error(manifold, &coefficients, &model);
            let betti = self.compute_residual_betti(manifold, &coefficients, &model);
            self.betti_history.push(betti);
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

    fn fit_model(&self, manifold: &ManifoldWorkspace, model: &RegressionModel) -> [f64; 8] {
        let mut coeffs = [0.0f64; 8];

        if manifold.points.is_empty() || self.target.is_empty() {
            return coeffs;
        }

        match model {
            RegressionModel::Linear => {
                let n = manifold.points.len().min(self.target.len()) as f64;
                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                let mut sum_xy = 0.0;
                let mut sum_xx = 0.0;

                for (i, p) in manifold.points.iter().enumerate() {
                    if i >= self.target.len() { break; }
                    let x = p.coords[0];
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
                coeffs = self.fit_model(manifold, &RegressionModel::Linear);
                coeffs[*degree as usize] = 0.01;
            }
            RegressionModel::Rbf { .. } => {
                coeffs = self.fit_model(manifold, &RegressionModel::Polynomial { degree: 3 });
            }
        }

        coeffs
    }

    fn compute_error(
        &self,
        manifold: &ManifoldWorkspace,
        coeffs: &[f64; 8],
        model: &RegressionModel,
    ) -> f64 {
        let mut mse = 0.0;
        let mut count = 0;

        for (i, p) in manifold.points.iter().enumerate() {
            if i >= self.target.len() { break; }
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
            RegressionModel::Rbf { .. } => {
                self.predict(x, coeffs, &RegressionModel::Polynomial { degree: 3 })
            }
        }
    }

    fn compute_residual_betti(
        &self,
        manifold: &ManifoldWorkspace,
        coeffs: &[f64; 8],
        model: &RegressionModel,
    ) -> (u32, u32) {
        let mut sign_changes = 0u32;
        let mut oscillations = 0u32;
        let mut prev_residual = 0.0;
        let mut prev_sign = true;

        for (i, p) in manifold.points.iter().enumerate() {
            if i >= self.target.len() { break; }
            let pred = self.predict(p.coords[0], coeffs, model);
            let residual = self.target[i] - pred;
            let sign = residual >= 0.0;
            if i > 0 && sign != prev_sign {
                sign_changes += 1;
            }
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

    fn is_converged(&self, error: f64, current_betti: &(u32, u32)) -> bool {
        if error < self.epsilon { return true; }
        if self.betti_history.len() >= 3 {
            let recent: Vec<&(u32, u32)> = self.betti_history.iter().rev().take(3).collect();
            if recent.iter().all(|b| **b == *current_betti) { return true; }
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
    pub variables: BTreeMap<String, Value>, // Made public for tests
    /// Manifold workspaces
    manifolds: Vec<ManifoldWorkspace>,
    /// Block geometries
    blocks: Vec<BlockMetadata<DIM>>,
    /// Class definitions
    classes: Vec<ClassDef>,
    /// Object instances
    objects: Vec<ObjectInstance>,
    /// Sample data (for demo)
    sample_data: Vec<f64>,
}

impl Interpreter {
    pub fn new() -> Self {
        let mut data = Vec::new();
        for i in 0..64 {
            let x = (i as f64) * 0.1;
            data.push(libm::sin(x));
        }

        let mut variables = BTreeMap::new();
        variables.insert(String::from("print"), Value::NativeFn(NativeFunction::Print));

        Self {
            variables,
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
        match &stmt.node {
            StmtKind::Manifold(decl) => self.execute_manifold(decl),
            StmtKind::Block(decl) => self.execute_block(decl),
            StmtKind::Var(decl) => self.execute_var(decl),
            StmtKind::Regress(stmt) => self.execute_regress(stmt),
            StmtKind::Render(stmt) => self.execute_render(stmt),
            StmtKind::Class(decl) => self.execute_class(decl),
            StmtKind::Import(stmt) => self.execute_import(stmt),
            StmtKind::If(stmt) => self.execute_if(stmt),
            StmtKind::While(stmt) => self.execute_while(stmt),
            StmtKind::Loop(stmt) => self.execute_seal(stmt),
            StmtKind::For(_) => Ok(Value::Unit),
            StmtKind::Fn(_) => Ok(Value::Unit),
            StmtKind::Return(_) => Ok(Value::Unit),
            StmtKind::Break(_) => Ok(Value::Unit),
            StmtKind::Continue(_) => Ok(Value::Unit),
            StmtKind::Expr(expr) => {
                self.evaluate_expr(expr)?;
                Ok(Value::Unit)
            },
            StmtKind::Empty => Ok(Value::Unit),
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
        self.variables.insert(decl.name.clone(), Value::Class(handle));
        Ok(Value::Class(handle))
    }

    #[allow(unused_variables)]
    fn evaluate_new(&mut self, class_name: &String, args: &[Expr]) -> Result<Value, String> {
        let class_handle = if let Some(Value::Class(h)) = self.variables.get(class_name) { *h } else {
            return Err(format!("Class '{}' not found", class_name));
        };

        let class_def = self.classes[class_handle.0].clone();
        let mut fields = BTreeMap::new();
        for field in &class_def.fields {
            let val = self.evaluate_expr(&field.value)?;
            fields.insert(field.name.clone(), val);
        }

        let obj_handle = ObjectHandle(self.objects.len());
        self.objects.push(ObjectInstance {
            class: class_handle,
            fields,
        });

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
        if let Some(symbol) = &stmt.symbol {
            match symbol.as_str() {
                "pi" => { self.variables.insert(String::from("pi"), Value::Num(core::f64::consts::PI)); },
                "sin" => { self.variables.insert(String::from("sin"), Value::NativeFn(NativeFunction::MathSin)); },
                "cos" => { self.variables.insert(String::from("cos"), Value::NativeFn(NativeFunction::MathCos)); },
                "sqrt" => { self.variables.insert(String::from("sqrt"), Value::NativeFn(NativeFunction::MathSqrt)); },
                "exp" => { self.variables.insert(String::from("exp"), Value::NativeFn(NativeFunction::MathExp)); },
                _ => return Err(format!("Symbol '{}' not found in math", symbol)),
            }
        } else {
             self.variables.insert(String::from("pi"), Value::Num(core::f64::consts::PI));
             self.variables.insert(String::from("sin"), Value::NativeFn(NativeFunction::MathSin));
             self.variables.insert(String::from("cos"), Value::NativeFn(NativeFunction::MathCos));
             self.variables.insert(String::from("sqrt"), Value::NativeFn(NativeFunction::MathSqrt));
        }
        Ok(Value::Unit)
    }

    fn import_topology(&mut self, stmt: &ImportStmt) -> Result<Value, String> {
        if let Some(symbol) = &stmt.symbol {
            match symbol.as_str() {
                "Betti" => self.variables.insert(String::from("Betti"), Value::NativeFn(NativeFunction::TopoBetti)),
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
        let tau = self.extract_tau(&decl.init).unwrap_or(3);
        let mut workspace = ManifoldWorkspace::new(tau);
        workspace.embed_data(&self.sample_data);
        let handle = ManifoldHandle(self.manifolds.len());
        self.manifolds.push(workspace);
        self.variables.insert(decl.name.clone(), Value::Manifold(handle));
        Ok(Value::Manifold(handle))
    }

    fn extract_tau(&self, expr: &Expr) -> Option<usize> {
        if let ExprKind::Call { args, .. } = &expr.node {
            for arg in args {
                if let CallArg::Named { name, value } = arg {
                    if name.as_str() == "tau" {
                        if let ExprKind::Literal(Literal::Num(n)) = &value.node {
                            return Some(*n as usize);
                        }
                    }
                }
            }
        }
        None
    }

    fn execute_block(&mut self, decl: &BlockDecl) -> Result<Value, String> {
        let (manifold_handle, start, end) = self.extract_block_range(&decl.source)?;
        if let Some(workspace) = self.manifolds.get(manifold_handle.0) {
            let block = workspace.extract_block(start, end);
            let handle = BlockHandle(self.blocks.len());
            self.blocks.push(block);
            self.variables.insert(decl.name.clone(), Value::Block(handle));
            Ok(Value::Block(handle))
        } else {
            Err("manifold not found".to_string())
        }
    }

    fn extract_block_range(&self, expr: &Expr) -> Result<(ManifoldHandle, usize, usize), String> {
        match &expr.node {
            ExprKind::MethodCall { object, args, .. } => {
                let handle = self.get_manifold_handle(object)?;
                let (start, end) = self.extract_range_from_args(args);
                Ok((handle, start, end))
            }
            ExprKind::Index { object, range } => {
                let handle = self.get_manifold_handle(object)?;
                let start = range.start.as_f64() as usize;
                let end = range.end.as_f64() as usize;
                Ok((handle, start, end))
            }
            _ => Err("invalid block source".to_string())
        }
    }

    fn get_manifold_handle(&self, name: &String) -> Result<ManifoldHandle, String> {
        if let Some(Value::Manifold(h)) = self.variables.get(name) {
            Ok(*h)
        } else {
            Err("variable is not a manifold".to_string())
        }
    }

    fn extract_range_from_args(&self, args: &[CallArg]) -> (usize, usize) {
        let mut start = 0usize;
        let mut end = 64usize;

        for (i, arg) in args.iter().enumerate() {
            if let CallArg::Positional(expr) = arg {
                match &expr.node {
                    ExprKind::Literal(Literal::Num(n)) => {
                        if i == 0 { start = *n as usize; }
                        if i == 1 { end = *n as usize; }
                    }
                    ExprKind::Range(r) => {
                        start = r.start.as_f64() as usize;
                        end = r.end.as_f64() as usize;
                    }
                    _ => {}
                }
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
        let epsilon = match &config.until {
            Some(ConvergenceCond::Epsilon(n)) => n.as_f64(),
            _ => 1e-6,
        };
        let mut regressor = EscalatingRegressor::new(epsilon);
        regressor.set_target(&self.sample_data);
        if let Some(workspace) = self.manifolds.first() {
            let max_epochs = if config.escalate { 100 } else { 10 };
            let result = regressor.run_escalating(workspace, max_epochs);
            Ok(Value::RegressionResult(result))
        } else {
            Err("no manifold for regression".to_string())
        }
    }

    fn execute_render(&mut self, _: &RenderStmt) -> Result<Value, String> {
        Ok(Value::Unit)
    }

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
            if !is_true { break; }
            last_val = self.execute_stmt_block(&stmt.body)?;
        }
        Ok(last_val)
    }

    fn execute_seal(&mut self, stmt: &LoopStmt) -> Result<Value, String> {
        let max_iters = 1000;
        let mut last_val = Value::Unit;
        for _ in 0..max_iters {
            last_val = self.execute_stmt_block(&stmt.body)?;
        }
        Ok(last_val)
    }

    fn evaluate_expr(&mut self, expr: &Expr) -> Result<Value, String> {
        match &expr.node {
            ExprKind::Literal(lit) => match lit {
                Literal::Num(n) => Ok(Value::Num(*n)),
                Literal::Bool(b) => Ok(Value::Bool(*b)),
                Literal::Str(s) => Ok(Value::Str(s.clone())),
            },
            ExprKind::Ident(name) => {
                if let Some(v) = self.variables.get(name) {
                    Ok(v.clone())
                } else {
                    Ok(Value::Unit)
                }
            }
            ExprKind::FieldAccess { object, field } => self.evaluate_field_access(object, field),
            ExprKind::Call { name, args } => self.evaluate_call(name, args),
            ExprKind::New { class, args } => self.evaluate_new(class, args),
            ExprKind::List(elements) => self.evaluate_list(elements),
            ExprKind::MethodCall { object, method, args } => self.evaluate_method_call(object, method, args),
            ExprKind::Range(_) => Err(String::from("Ranges cannot be evaluated directly as values")),
            ExprKind::BinaryOp(left, op, right) => {
                 let l = self.evaluate_expr(left)?;
                 let r = self.evaluate_expr(right)?;
                 self.evaluate_binary(l, *op, r)
            },
            ExprKind::UnaryOp(_, _) => Err(String::from("Unary ops not implemented yet")),
            ExprKind::Index { object, range } => {
                // Simplified: returns a descriptive string or handle? 
                // For now, let's treat it as a lookup that returns a sub-manifold or block value
                let handle = self.get_manifold_handle(object)?;
                let start = range.start.as_f64() as usize;
                let end = range.end.as_f64() as usize;
                if let Some(workspace) = self.manifolds.get(handle.0) {
                    let block = workspace.extract_block(start, end);
                    let block_handle = BlockHandle(self.blocks.len());
                    self.blocks.push(block);
                    Ok(Value::Block(block_handle))
                } else {
                    Err(format!("Manifold '{}' not found", object))
                }
            }
            ExprKind::Config(_) => Err(String::from("Raw config blocks cannot be evaluated as expressions")),
        }
    }
    
    fn evaluate_binary(&self, left: Value, op: BinaryOp, right: Value) -> Result<Value, String> {
        match (left, op, right) {
            (Value::Num(a), BinaryOp::Add, Value::Num(b)) => Ok(Value::Num(a + b)),
            (Value::Num(a), BinaryOp::Sub, Value::Num(b)) => Ok(Value::Num(a - b)),
            (Value::Num(a), BinaryOp::Mul, Value::Num(b)) => Ok(Value::Num(a * b)),
            (Value::Num(a), BinaryOp::Div, Value::Num(b)) => Ok(Value::Num(a / b)),
            _ => Err("Invalid binary operation".into())
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
        if let Some(val) = self.variables.get(name) {
            match val.clone() {
                Value::NativeFn(func) => self.execute_native_fn(func, args),
                _ => Ok(Value::Unit),
            }
        } else {
            Ok(Value::Unit)
        }
    }

    fn execute_native_fn(&mut self, func: NativeFunction, args: &[CallArg]) -> Result<Value, String> {
        let mut get_f64 = |args: &[CallArg]| -> Result<f64, String> {
            if let Some(CallArg::Positional(expr)) = args.first() {
                let val = self.evaluate_expr(expr)?;
                if let Value::Num(n) = val { Ok(n) } else { Err(String::from("Expected number")) }
            } else {
                Err(String::from("Expected number"))
            }
        };

        match func {
            NativeFunction::MathSin => Ok(Value::Num(libm::sin(get_f64(args)?))),
            NativeFunction::MathCos => Ok(Value::Num(libm::cos(get_f64(args)?))),
            NativeFunction::MathSqrt => Ok(Value::Num(libm::sqrt(get_f64(args)?))),
            NativeFunction::MathExp => Ok(Value::Num(libm::exp(get_f64(args)?))),
            NativeFunction::TopoBetti => Ok(Value::List(vec![Value::Num(1.0), Value::Num(0.0)])),
            NativeFunction::Print => {
                 for arg in args {
                     if let CallArg::Positional(expr) = arg {
                         let val = self.evaluate_expr(expr)?;
                         #[cfg(feature = "std")]
                         println!("{:?}", val);
                     }
                 }
                 Ok(Value::Unit)
            },
            NativeFunction::MlpNew => {
                let lr = get_f64(args).unwrap_or(0.01);
                let config = OptimizerConfig::SGD { learning_rate: lr, momentum: 0.9 };
                Ok(Value::Mlp(Box::new(MLP::new(config, LossConfig::MSE))))
            },
            NativeFunction::KMeansNew => {
                 let k = get_f64(args).unwrap_or(2.0) as usize;
                 Ok(Value::KMeans(Box::new(KMeans::new(k))))
            },
            NativeFunction::Conv2DNew => Ok(Value::Conv2D(Box::new(Conv2D::new(1, 1, 3, 1, 1, Activation::ReLU)))),
            
            // ML Ops
            NativeFunction::MlMatMul => {
                let a = self.get_tensor_arg(args, 0)?;
                let b = self.get_tensor_arg(args, 1)?;
                Ok(Value::Tensor(a.matmul(&b)))
            },
            NativeFunction::MlAdd => {
                let a = self.get_tensor_arg(args, 0)?;
                let b = self.get_tensor_arg(args, 1)?;
                Ok(Value::Tensor(a.add(&b)))
            },
            NativeFunction::MlRelu => {
                let a = self.get_tensor_arg(args, 0)?;
                Ok(Value::Tensor(Activation::ReLU.apply(&a)))
            },
            NativeFunction::MlSoftmax => {
                let a = self.get_tensor_arg(args, 0)?;
                Ok(Value::Tensor(Activation::Softmax.apply(&a)))
            },
            NativeFunction::MlLoadWeights => {
                // Acts as "Create Tensor from List"
                 if let Some(CallArg::Positional(expr)) = args.first() {
                     let val = self.evaluate_expr(expr)?;
                     let t = self.value_to_tensor_core(&val)?;
                     Ok(Value::Tensor(t))
                 } else {
                     Err("Missing argument".into())
                 }
            },
            NativeFunction::MlForward => { // Map "forward"
                 // Args: model, input
                 // Actually this might be MethodCall on Mlp object
                 Ok(Value::Unit)
            },
            _ => Ok(Value::Unit)
        }
    }

    fn get_tensor_arg(&mut self, args: &[CallArg], index: usize) -> Result<Tensor, String> {
        if let Some(CallArg::Positional(expr)) = args.get(index) {
             let val = self.evaluate_expr(expr)?;
             self.value_to_tensor_core(&val)
        } else {
             Err(format!("Missing argument {}", index))
        }
    }

    fn value_to_tensor_core(&self, val: &Value) -> Result<Tensor, String> {
         match val {
            Value::Tensor(t) => Ok(t.clone()),
            Value::List(rows) => {
                 if rows.is_empty() { return Ok(Tensor::zeros(&[0])); }
                 
                 let mut data = Vec::new();
                 
                 // Check if 2D or 1D
                 if let Value::List(_) = &rows[0] {
                     // 2D
                     let rows_cnt = rows.len();
                     let mut cols_cnt = 0;
                     for (i, row) in rows.iter().enumerate() {
                         if let Value::List(cols) = row {
                             if i == 0 { cols_cnt = cols.len(); }
                             else if cols.len() != cols_cnt { return Err("Ragged tensor".into()); }
                             for c in cols {
                                 if let Value::Num(n) = c { data.push(*n); }
                                 else { return Err("Tensor must contain numbers".into()); }
                             }
                         } else { return Err("Expected 2D list".into()); }
                     }
                     Ok(Tensor::new(&data, &[rows_cnt, cols_cnt]))
                 } else {
                     // 1D
                      for c in rows {
                         if let Value::Num(n) = c { data.push(*n); }
                         else { return Err("Tensor must contain numbers".into()); }
                     }
                     Ok(Tensor::new(&data, &[rows.len()]))
                 }
            },
            _ => Err("Expected Tensor or List".into())
         }
    }

    fn evaluate_method_call(&mut self, object_name: &String, method: &String, args: &[CallArg]) -> Result<Value, String> {
        let val = if let Some(v) = self.variables.get(object_name) { v.clone() } else { return Err(format!("Object '{}' not found", object_name)); };
        match val {
            Value::List(mut list) => {
                let res = match method.as_str() {
                    "push" => {
                        if let Some(CallArg::Positional(expr)) = args.first() {
                            let val = self.evaluate_expr(expr)?;
                            list.push(val);
                            self.variables.insert(object_name.clone(), Value::List(list));
                            Ok(Value::Unit)
                        } else { Err(String::from("push requires 1 argument")) }
                    }
                    "pop" => {
                        let val = list.pop().unwrap_or(Value::Unit);
                        self.variables.insert(object_name.clone(), Value::List(list));
                        Ok(val)
                    }
                    "len" => Ok(Value::Num(list.len() as f64)),
                    _ => Err(format!("Method '{}' not found on List", method)),
                };
                res
            }
            Value::Mlp(mut mlp) => {
                match method.as_str() {
                    "add_layer" => {
                         // input, output, activation key string
                         // Default to Tanh if not string
                         let input = self.get_arg_num(args, 0)? as usize;
                         let output = self.get_arg_num(args, 1)? as usize;
                         let act_str = self.get_arg_str(args, 2).unwrap_or("tanh".to_string());
                         let act = match act_str.as_str() {
                             "relu" => Activation::ReLU,
                             "sigmoid" => Activation::Sigmoid,
                             "softmax" => Activation::Softmax,
                             _ => Activation::Tanh
                         };
                         mlp.add_layer(input, output, act, None);
                         self.variables.insert(object_name.clone(), Value::Mlp(mlp)); // Update
                         Ok(Value::Unit)
                    },
                    "train" => {
                        // inputs (List/Tensor), targets (List/Tensor), epochs
                        let input = self.get_tensor_arg(args, 0)?;
                        let target = self.get_tensor_arg(args, 1)?;
                        let epochs = self.get_arg_num(args, 2).unwrap_or(1.0) as usize;
                        let res = mlp.fit(&[input], &[target], epochs); // fit expects slice of tensors
                        Ok(Value::Num(res.final_loss))
                    },
                    "forward" | "predict" => {
                        let input = self.get_tensor_arg(args, 0)?;
                         let output = mlp.forward(&input);
                         Ok(Value::Tensor(output))
                    },
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
            _ => Ok(Value::Unit), 
        }
    }

    fn get_arg_num(&mut self, args: &[CallArg], index: usize) -> Result<f64, String> {
         if let Some(CallArg::Positional(expr)) = args.get(index) {
             let val = self.evaluate_expr(expr)?;
             if let Value::Num(n) = val { Ok(n) } else { Err("Expected number".into()) }
         } else { Err("Missing arg".into()) }
    }
    
    fn get_arg_str(&mut self, args: &[CallArg], index: usize) -> Result<String, String> {
          if let Some(CallArg::Positional(expr)) = args.get(index) {
             let val = self.evaluate_expr(expr)?;
             if let Value::Str(s) = val { Ok(s) } else { Err("Expected string".into()) }
         } else { Err("Missing arg".into()) }
    }

    fn evaluate_field_access(&self, object: &String, field: &String) -> Result<Value, String> {
        if let Some(Value::Object(handle)) = self.variables.get(object) {
            if let Some(obj) = self.objects.get(handle.0) {
                if let Some(val) = obj.fields.get(field) { return Ok(val.clone()); }
            }
        }
        if let Some(Value::Module(name)) = self.variables.get(object) {
             match (name.as_str(), field.as_str()) {
                 ("Seal", "train") => Ok(Value::NativeFn(NativeFunction::SealTrain)),
                 ("Ml", "MLP") => Ok(Value::NativeFn(NativeFunction::MlpNew)),
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

// Tests helper
fn list_from_u8(bytes: &[u8]) -> Vec<f32> {
    let mut data = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().unwrap();
        data.push(f32::from_le_bytes(arr));
    }
    data
}
