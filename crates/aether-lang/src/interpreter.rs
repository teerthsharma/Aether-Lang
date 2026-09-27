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

// ═══════════════════════════════════════════════════════════════════════════════
// Aether-Lang — invented by Teerth Sharma
// https://github.com/teerthsharma/Aether-Lang
// Copyright (c) 2026 Teerth Sharma. All Rights Reserved.
// ═══════════════════════════════════════════════════════════════════════════════
//

#![allow(dead_code)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap;
#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::string::ToString;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::boxed::Box;
#[cfg(feature = "std")]
use std::collections::BTreeMap;
#[cfg(feature = "std")]
use std::string::String;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::ast::*;
use crate::natives::{self, NativeArgs, NativeId};
use aether_core::aether::{BlockMetadata, DriftDetector, HierarchicalBlockTree};
use aether_core::manifold::{ManifoldPoint, TimeDelayEmbedder};
use aether_core::ml::convolution::Conv2D;
use aether_core::ml::tensor::Tensor;
use aether_core::ml::{KMeans, MLP};
use aether_core::persistence::PersistenceDiagram;
use libm::{fabs, sqrt};

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
    /// Persistent homology diagram
    Persistence(PersistenceDiagram),
    /// Class Definition
    Class(ClassHandle),
    /// Object Instance
    Object(ObjectHandle),
    /// Native Function (for Standard Library)
    NativeFn(NativeFunction),
    /// User-defined function
    Function(FnDecl),
    /// Dynamic List (Python-like)
    List(Vec<Value>),
    /// Named fields returned by a native function, read with `r.field`
    Record(BTreeMap<String, Value>),
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
    /// A TitanVM function, by index into the VM's function table.
    Compiled(usize),
    /// Llama Model (Wrapped)
    #[cfg(feature = "ml")]
    LlamaModel(Arc<LlamaContext>),
}

/// How `print` and the REPL show a value: the value itself, not its Rust
/// representation. Numbers use the shortest round-trip form (`1`, `0.25`), and
/// switch to exponent form outside `[1e-4, 1e16)` so an error bound prints as
/// `7.072564457345712e-14` rather than fourteen zeros. Handles and models fall
/// back to `Debug`, since they have no literal syntax to print back.
impl core::fmt::Display for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fn seq<'a>(
            f: &mut core::fmt::Formatter<'_>,
            items: impl Iterator<Item = &'a Value>,
        ) -> core::fmt::Result {
            for (i, v) in items.enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{v}")?;
            }
            Ok(())
        }
        match self {
            Value::Num(n) if *n != 0.0 && n.is_finite() && !(1e-4..1e16).contains(&n.abs()) => {
                write!(f, "{n:e}")
            }
            Value::Num(n) => write!(f, "{n}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Str(s) => f.write_str(s),
            Value::Point(p) => write!(f, "({}, {}, {})", p[0], p[1], p[2]),
            Value::List(items) => {
                f.write_str("[")?;
                seq(f, items.iter())?;
                f.write_str("]")
            }
            Value::Record(fields) => {
                f.write_str("{")?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
                f.write_str("}")
            }
            Value::Unit => f.write_str("()"),
            other => write!(f, "{other:?}"),
        }
    }
}

#[cfg(feature = "ml")]
#[derive(Debug)]
pub struct LlamaContext {
    pub model: LlamaWeights,
    pub tokenizer: Tokenizer,
    pub name: String,
}

/// A function value the engine provides rather than the program.
#[derive(Debug, Clone, Copy)]
pub enum NativeFunction {
    /// `print`, a special form: each argument is evaluated, then printed.
    Print,
    /// A native of [`crate::natives`].
    Native(NativeId),
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
                    if i >= self.target.len() {
                        break;
                    }
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
            if i >= self.target.len() {
                break;
            }
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
        if error < self.epsilon {
            return true;
        }
        if self.betti_history.len() >= 3 {
            let recent: Vec<&(u32, u32)> = self.betti_history.iter().rev().take(3).collect();
            if recent.iter().all(|b| **b == *current_betti) {
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

enum RuntimeFlow {
    Value(Value),
    Return(Value),
    Break,
    Continue,
}

impl Interpreter {
    pub fn new() -> Self {
        let mut data = Vec::new();
        for i in 0..64 {
            let x = (i as f64) * 0.1;
            data.push(libm::sin(x));
        }

        let mut variables = BTreeMap::new();
        variables.insert(
            String::from("print"),
            Value::NativeFn(NativeFunction::Print),
        );

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
            match self.execute_statement(stmt)? {
                RuntimeFlow::Value(value) => last_value = value,
                RuntimeFlow::Return(value) => return Ok(value),
                RuntimeFlow::Break => return Err(String::from("break outside loop")),
                RuntimeFlow::Continue => return Err(String::from("continue outside loop")),
            }
        }
        Ok(last_value)
    }

    fn execute_statement(&mut self, stmt: &Statement) -> Result<RuntimeFlow, String> {
        match &stmt.node {
            StmtKind::Manifold(decl) => self.execute_manifold(decl).map(RuntimeFlow::Value),
            StmtKind::Block(decl) => self.execute_block(decl).map(RuntimeFlow::Value),
            StmtKind::Var(decl) => self.execute_var(decl).map(RuntimeFlow::Value),
            StmtKind::Assign(stmt) => self.execute_assign(stmt).map(RuntimeFlow::Value),
            StmtKind::Regress(stmt) => self.execute_regress(stmt).map(RuntimeFlow::Value),
            StmtKind::Render(stmt) => self.execute_render(stmt).map(RuntimeFlow::Value),
            StmtKind::Class(decl) => self.execute_class(decl).map(RuntimeFlow::Value),
            StmtKind::Import(stmt) => self.execute_import(stmt).map(RuntimeFlow::Value),
            StmtKind::If(stmt) => self.execute_if(stmt),
            StmtKind::While(stmt) => self.execute_while(stmt).map(RuntimeFlow::Value),
            StmtKind::Loop(stmt) => self.execute_seal(stmt).map(RuntimeFlow::Value),
            StmtKind::For(stmt) => self.execute_for(stmt).map(RuntimeFlow::Value),
            StmtKind::Fn(decl) => self.execute_fn_decl(decl).map(RuntimeFlow::Value),
            StmtKind::Return(stmt) => self.execute_return(stmt),
            StmtKind::Break(_) => Ok(RuntimeFlow::Break),
            StmtKind::Continue(_) => Ok(RuntimeFlow::Continue),
            StmtKind::Expr(expr) => self.evaluate_expr(expr).map(RuntimeFlow::Value),
            StmtKind::Empty => Ok(RuntimeFlow::Value(Value::Unit)),
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

    fn execute_fn_decl(&mut self, decl: &FnDecl) -> Result<Value, String> {
        self.variables
            .insert(decl.name.clone(), Value::Function(decl.clone()));
        Ok(Value::Unit)
    }

    fn execute_return(&mut self, stmt: &ReturnStmt) -> Result<RuntimeFlow, String> {
        let value = if let Some(expr) = &stmt.value {
            self.evaluate_expr(expr)?
        } else {
            Value::Unit
        };
        Ok(RuntimeFlow::Return(value))
    }

    #[allow(unused_variables)]
    fn evaluate_new(&mut self, class_name: &String, args: &[Expr]) -> Result<Value, String> {
        let class_handle = if let Some(Value::Class(h)) = self.variables.get(class_name) {
            *h
        } else {
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
        let module = stmt.module.as_str();
        let names =
            natives::exports(module).ok_or_else(|| format!("Module '{}' not found", module))?;
        if let Some(symbol) = &stmt.symbol {
            let shown = if module == "Ml" { "ml" } else { module };
            let value = binding(module, symbol)
                .ok_or_else(|| format!("Symbol '{}' not found in {}", symbol, shown))?;
            self.variables.insert(symbol.clone(), value);
            return Ok(Value::Unit);
        }
        // Every module but math is also a value, for `topology.ph(M)`; ml is
        // bound as `Ml`. The module goes first, so `import track` leaves
        // `track` the function.
        match module {
            "math" => {}
            "ml" | "Ml" => {
                self.variables
                    .insert(String::from("Ml"), Value::Module(String::from("Ml")));
            }
            _ => {
                self.variables
                    .insert(String::from(module), Value::Module(String::from(module)));
            }
        }
        for name in names {
            let value = binding(module, name)
                .ok_or_else(|| format!("{}: '{}' is exported but not defined", module, name))?;
            self.variables.insert(String::from(*name), value);
        }
        Ok(Value::Unit)
    }

    fn execute_manifold(&mut self, decl: &ManifoldDecl) -> Result<Value, String> {
        let tau = self.extract_tau(&decl.init).unwrap_or(3);
        let data = match self.extract_embed_data(&decl.init)? {
            Some(data) => data,
            None => self.sample_data.clone(),
        };
        let manifold = natives::embed_manifold(self, data, DIM, tau)?;
        self.variables.insert(decl.name.clone(), manifold.clone());
        Ok(manifold)
    }

    fn extract_embed_data(&mut self, expr: &Expr) -> Result<Option<Vec<f64>>, String> {
        let ExprKind::Call { name, args } = &expr.node else {
            return Ok(None);
        };
        if name.as_str() != "embed" {
            return Ok(None);
        }

        for arg in args {
            match arg {
                CallArg::Positional(expr) => {
                    let value = self.evaluate_expr(expr)?;
                    return self.value_to_f64_vec(value).map(Some);
                }
                CallArg::Named { name, value } if name.as_str() == "data" => {
                    let value = self.evaluate_expr(value)?;
                    return self.value_to_f64_vec(value).map(Some);
                }
                _ => {}
            }
        }

        Ok(None)
    }

    fn value_to_f64_vec(&self, value: Value) -> Result<Vec<f64>, String> {
        match value {
            Value::List(values) => {
                let mut out = Vec::with_capacity(values.len());
                for value in values {
                    match value {
                        Value::Num(n) => out.push(n),
                        _ => return Err(String::from("embed data must be a numeric list")),
                    }
                }
                Ok(out)
            }
            _ => Err(String::from("embed expects a numeric list")),
        }
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
            self.variables
                .insert(decl.name.clone(), Value::Block(handle));
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
            _ => Err("invalid block source".to_string()),
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
                        if i == 0 {
                            start = *n as usize;
                        }
                        if i == 1 {
                            end = *n as usize;
                        }
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

    fn execute_assign(&mut self, stmt: &AssignStmt) -> Result<Value, String> {
        if !self.variables.contains_key(&stmt.name) {
            return Err(format!("cannot assign undefined variable '{}'", stmt.name));
        }

        let value = self.evaluate_expr(&stmt.value)?;
        self.variables.insert(stmt.name.clone(), value.clone());
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

    fn execute_stmt_block(&mut self, block: &Block) -> Result<RuntimeFlow, String> {
        let mut last_value = Value::Unit;
        for stmt in &block.statements {
            match self.execute_statement(stmt)? {
                RuntimeFlow::Value(value) => last_value = value,
                RuntimeFlow::Return(value) => return Ok(RuntimeFlow::Return(value)),
                RuntimeFlow::Break => return Ok(RuntimeFlow::Break),
                RuntimeFlow::Continue => return Ok(RuntimeFlow::Continue),
            }
        }
        Ok(RuntimeFlow::Value(last_value))
    }

    fn execute_if(&mut self, stmt: &IfStmt) -> Result<RuntimeFlow, String> {
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
            Ok(RuntimeFlow::Value(Value::Unit))
        }
    }

    fn execute_while(&mut self, stmt: &WhileStmt) -> Result<Value, String> {
        let mut last_value = Value::Unit;
        loop {
            let cond_val = self.evaluate_expr(&stmt.condition)?;
            let is_true = match cond_val {
                Value::Bool(b) => b,
                _ => return Err(String::from("condition must be boolean")),
            };
            if !is_true {
                break;
            }
            match self.execute_stmt_block(&stmt.body)? {
                RuntimeFlow::Value(value) => last_value = value,
                RuntimeFlow::Return(value) => return Ok(value),
                RuntimeFlow::Break => break,
                RuntimeFlow::Continue => continue,
            }
        }
        Ok(last_value)
    }

    fn execute_for(&mut self, stmt: &ForStmt) -> Result<Value, String> {
        let start = stmt.range.start.as_f64() as i64;
        let end = stmt.range.end.as_f64() as i64;
        let step = if start <= end { 1 } else { -1 };
        let mut current = start;
        let mut last_value = Value::Unit;

        while (step > 0 && current < end) || (step < 0 && current > end) {
            self.variables
                .insert(stmt.iterator.clone(), Value::Num(current as f64));
            match self.execute_stmt_block(&stmt.body)? {
                RuntimeFlow::Value(value) => last_value = value,
                RuntimeFlow::Return(value) => return Ok(value),
                RuntimeFlow::Break => break,
                RuntimeFlow::Continue => {
                    current += step;
                    continue;
                }
            }
            current += step;
        }

        self.variables
            .insert(stmt.iterator.clone(), Value::Num(current as f64));
        Ok(last_value)
    }

    fn execute_seal(&mut self, stmt: &LoopStmt) -> Result<Value, String> {
        // `until convergence(eps)`: seal once a pass of the body changes the
        // body's value by at most `eps` in the max norm. The loop's value is
        // always the last pass's value, so `previous` doubles as it.
        if let Some(LoopCond::Convergence(tolerance)) = &stmt.until {
            let eps = match self.evaluate_expr(tolerance)? {
                Value::Num(eps) if eps >= 0.0 => eps,
                other => {
                    return Err(format!(
                        "convergence() takes a non-negative tolerance, got {other}"
                    ))
                }
            };
            let mut previous: Option<Value> = None;
            for _ in 0..natives::seal::MAX_PASSES {
                let value = match self.execute_stmt_block(&stmt.body)? {
                    RuntimeFlow::Value(value) => value,
                    RuntimeFlow::Return(value) => return Ok(value),
                    RuntimeFlow::Break => break,
                    RuntimeFlow::Continue => continue,
                };
                let settled = match &previous {
                    Some(prev) => natives::seal::max_change(prev, &value)? <= eps,
                    None => false,
                };
                previous = Some(value);
                if settled {
                    break;
                }
            }
            return Ok(previous.unwrap_or(Value::Unit));
        }

        let mut last_value = Value::Unit;
        let mut previous: Option<Value> = None;
        for _ in 0..natives::seal::MAX_PASSES {
            match &stmt.until {
                // `until stable(expr)`: seal once `expr` reads the same before
                // two consecutive passes, i.e. a pass left the invariant unchanged.
                Some(LoopCond::Stable(expr)) => {
                    let now = self.evaluate_expr(expr)?;
                    if let Some(prev) = &previous {
                        if natives::seal::same(prev, &now)? {
                            break;
                        }
                    }
                    previous = Some(now);
                }
                Some(LoopCond::Expr(condition)) => {
                    if self.evaluate_condition(condition)? {
                        break;
                    }
                }
                Some(LoopCond::Convergence(_)) | None => {}
            }
            match self.execute_stmt_block(&stmt.body)? {
                RuntimeFlow::Value(value) => last_value = value,
                RuntimeFlow::Return(value) => return Ok(value),
                RuntimeFlow::Break => break,
                RuntimeFlow::Continue => continue,
            }
        }
        Ok(last_value)
    }

    fn evaluate_condition(&mut self, expr: &Expr) -> Result<bool, String> {
        match self.evaluate_expr(expr)? {
            Value::Bool(value) => Ok(value),
            _ => Err(String::from("condition must be boolean")),
        }
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
            ExprKind::MethodCall {
                object,
                method,
                args,
            } => self.evaluate_method_call(object, method, args),
            ExprKind::Range(_) => Err(String::from(
                "Ranges cannot be evaluated directly as values",
            )),
            ExprKind::BinaryOp(left, op, right) => {
                let l = self.evaluate_expr(left)?;
                let r = self.evaluate_expr(right)?;
                self.evaluate_binary(l, *op, r)
            }
            ExprKind::UnaryOp(op, expr) => {
                let value = self.evaluate_expr(expr)?;
                self.evaluate_unary(*op, value)
            }
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
            ExprKind::Config(_) => Err(String::from(
                "Raw config blocks cannot be evaluated as expressions",
            )),
            ExprKind::Member { object, field } => {
                let value = self.evaluate_expr(object)?;
                natives::field(&value, field)
            }
            ExprKind::Element { object, index } => {
                let list = self.evaluate_expr(object)?;
                let index = self.evaluate_expr(index)?;
                natives::element(&list, &index)
            }
        }
    }

    fn evaluate_binary(&self, left: Value, op: BinaryOp, right: Value) -> Result<Value, String> {
        match (left, op, right) {
            (Value::Num(a), BinaryOp::Add, Value::Num(b)) => Ok(Value::Num(a + b)),
            (Value::Num(a), BinaryOp::Sub, Value::Num(b)) => Ok(Value::Num(a - b)),
            (Value::Num(a), BinaryOp::Mul, Value::Num(b)) => Ok(Value::Num(a * b)),
            (Value::Num(a), BinaryOp::Div, Value::Num(b)) => Ok(Value::Num(a / b)),
            (Value::Num(a), BinaryOp::Mod, Value::Num(b)) => Ok(Value::Num(a % b)),
            (Value::Num(a), BinaryOp::Eq, Value::Num(b)) => Ok(Value::Bool(a == b)),
            (Value::Num(a), BinaryOp::Neq, Value::Num(b)) => Ok(Value::Bool(a != b)),
            (Value::Num(a), BinaryOp::Lt, Value::Num(b)) => Ok(Value::Bool(a < b)),
            (Value::Num(a), BinaryOp::Gt, Value::Num(b)) => Ok(Value::Bool(a > b)),
            (Value::Num(a), BinaryOp::Le, Value::Num(b)) => Ok(Value::Bool(a <= b)),
            (Value::Num(a), BinaryOp::Ge, Value::Num(b)) => Ok(Value::Bool(a >= b)),
            (Value::Bool(a), BinaryOp::Eq, Value::Bool(b)) => Ok(Value::Bool(a == b)),
            (Value::Bool(a), BinaryOp::Neq, Value::Bool(b)) => Ok(Value::Bool(a != b)),
            (Value::Bool(a), BinaryOp::And, Value::Bool(b)) => Ok(Value::Bool(a && b)),
            (Value::Bool(a), BinaryOp::Or, Value::Bool(b)) => Ok(Value::Bool(a || b)),
            (Value::Str(a), BinaryOp::Eq, Value::Str(b)) => Ok(Value::Bool(a == b)),
            (Value::Str(a), BinaryOp::Neq, Value::Str(b)) => Ok(Value::Bool(a != b)),
            _ => Err("Invalid binary operation".into()),
        }
    }

    fn evaluate_unary(&self, op: UnaryOp, value: Value) -> Result<Value, String> {
        match (op, value) {
            (UnaryOp::Neg, Value::Num(n)) => Ok(Value::Num(-n)),
            (UnaryOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
            _ => Err("Invalid unary operation".into()),
        }
    }

    fn evaluate_list(&mut self, elements: &Vec<Expr>) -> Result<Value, String> {
        let mut values = Vec::new();
        for expr in elements {
            values.push(self.evaluate_expr(expr)?);
        }
        Ok(Value::List(values))
    }

    fn evaluate_call(&mut self, name: &Ident, args: &[CallArg]) -> Result<Value, String> {
        match self.variables.get(name) {
            Some(Value::NativeFn(NativeFunction::Print)) => self.print(args),
            Some(Value::NativeFn(NativeFunction::Native(id))) => {
                let id = *id;
                let args = self.evaluate_args(args)?;
                natives::call(id, args, self)
            }
            Some(Value::Function(func)) => {
                let func = func.clone();
                self.execute_user_fn(&func, args)
            }
            _ => Ok(Value::Unit),
        }
    }

    /// `print` is a special form: it evaluates one argument, prints it, then
    /// moves to the next, so output interleaves with the arguments' own.
    fn print(&mut self, args: &[CallArg]) -> Result<Value, String> {
        for arg in args {
            if let CallArg::Positional(expr) = arg {
                // Bound with a leading underscore because the only reader is
                // the `std` println below, and evaluating is the point
                // regardless: the expression may have side effects and `?`
                // propagates its error. Discarding the binding would discard
                // those too.
                let _val = self.evaluate_expr(expr)?;
                #[cfg(feature = "std")]
                println!("{_val}");
            }
        }
        Ok(Value::Unit)
    }

    /// A native's arguments, evaluated left to right.
    fn evaluate_args(&mut self, args: &[CallArg]) -> Result<NativeArgs, String> {
        let mut out = NativeArgs {
            positional: Vec::new(),
            named: Vec::new(),
        };
        for arg in args {
            match arg {
                CallArg::Positional(expr) => out.positional.push(self.evaluate_expr(expr)?),
                CallArg::Named { name, value } => {
                    let value = self.evaluate_expr(value)?;
                    out.named.push((name.clone(), value));
                }
            }
        }
        Ok(out)
    }

    fn execute_user_fn(&mut self, func: &FnDecl, args: &[CallArg]) -> Result<Value, String> {
        let mut values = Vec::with_capacity(args.len());
        for arg in args {
            let CallArg::Positional(expr) = arg else {
                return Err(format!(
                    "function '{}' does not accept named arguments",
                    func.name
                ));
            };
            values.push(self.evaluate_expr(expr)?);
        }
        self.call_user_fn(func, values)
    }

    /// Call a user function on already-evaluated arguments. Native functions
    /// that take a program's function (a sampler, a map) call back through here.
    fn call_user_fn(&mut self, func: &FnDecl, args: Vec<Value>) -> Result<Value, String> {
        if args.len() != func.params.len() {
            return Err(format!(
                "function '{}' expected {} arguments, got {}",
                func.name,
                func.params.len(),
                args.len()
            ));
        }

        let mut frame = self.variables.clone();
        for (param, value) in func.params.iter().zip(args) {
            frame.insert(param.clone(), value);
        }

        let outer = core::mem::replace(&mut self.variables, frame);
        let result = match self.execute_stmt_block(&func.body) {
            Ok(RuntimeFlow::Return(value)) | Ok(RuntimeFlow::Value(value)) => Ok(value),
            Ok(RuntimeFlow::Break) => Err(String::from("break outside loop")),
            Ok(RuntimeFlow::Continue) => Err(String::from("continue outside loop")),
            Err(err) => Err(err),
        };
        self.variables = outer;
        result
    }

    fn evaluate_method_call(
        &mut self,
        object_name: &String,
        method: &String,
        args: &[CallArg],
    ) -> Result<Value, String> {
        let module = match self.variables.get(object_name) {
            None => return Err(format!("Object '{}' not found", object_name)),
            Some(Value::Module(module)) => Some(module.clone()),
            Some(_) => None,
        };
        let args = self.evaluate_args(args)?;
        // A module is a name, not state: its natives run on a copy, so the
        // module stays bound if one of them calls back into the program.
        if let Some(module) = module {
            return natives::method(&mut Value::Module(module), method, args, self);
        }
        // Anything else is taken out and put back, so the method mutates it in
        // place rather than a copy.
        let mut receiver = self
            .variables
            .remove(object_name)
            .ok_or_else(|| format!("Object '{}' not found", object_name))?;
        let result = natives::method(&mut receiver, method, args, self);
        self.variables.insert(object_name.clone(), receiver);
        result
    }

    fn evaluate_field_access(&self, object: &String, field: &String) -> Result<Value, String> {
        match self.variables.get(object) {
            Some(record @ Value::Record(_)) => natives::field(record, field),
            Some(Value::Object(handle)) => Ok(self
                .objects
                .get(handle.0)
                .and_then(|obj| obj.fields.get(field))
                .cloned()
                .unwrap_or(Value::Unit)),
            Some(Value::Module(module)) => {
                Ok(natives::lookup(module, field).map_or(Value::Unit, native_value))
            }
            _ => Ok(Value::Unit),
        }
    }
}

fn native_value(id: NativeId) -> Value {
    Value::NativeFn(NativeFunction::Native(id))
}

/// What `import module` binds `name` to: a constant (`math.pi`) or a native.
fn binding(module: &str, name: &str) -> Option<Value> {
    natives::constant(module, name).or_else(|| natives::lookup(module, name).map(native_value))
}

impl natives::Host for Interpreter {
    fn call_function(&mut self, f: &Value, args: Vec<Value>) -> Result<Value, String> {
        match f {
            Value::Function(decl) => self.call_user_fn(decl, args),
            _ => Err(String::from(
                "only a fn defined in the program can be called back",
            )),
        }
    }

    fn manifolds(&mut self) -> &mut Vec<ManifoldWorkspace> {
        &mut self.manifolds
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    #[test]
    fn executes_comparison_logical_unary_and_modulo_expressions() {
        let mut parser = Parser::new("let x = 10 % 4~\nlet ok = x == 2 && !false~");
        let program = parser.parse().expect("program should parse");
        let mut interpreter = Interpreter::new();

        interpreter
            .execute(&program)
            .expect("program should execute");

        assert!(matches!(
            interpreter.variables.get("x"),
            Some(Value::Num(2.0))
        ));
        assert!(matches!(
            interpreter.variables.get("ok"),
            Some(Value::Bool(true))
        ));
    }

    #[test]
    fn executes_reassignment_statement() {
        let mut parser = Parser::new("let count = 0~\ncount = count + 1~");
        let program = parser.parse().expect("program should parse");
        let mut interpreter = Interpreter::new();

        interpreter
            .execute(&program)
            .expect("program should execute");

        assert!(matches!(
            interpreter.variables.get("count"),
            Some(Value::Num(1.0))
        ));
    }

    #[test]
    fn executes_while_loop_with_assignment() {
        let mut parser = Parser::new("let count = 0~\nwhile count < 3 { count = count + 1~ }");
        let program = parser.parse().expect("program should parse");
        let mut interpreter = Interpreter::new();

        interpreter
            .execute(&program)
            .expect("program should execute");

        assert!(matches!(
            interpreter.variables.get("count"),
            Some(Value::Num(3.0))
        ));
    }

    #[test]
    fn executes_for_loop_over_integer_range() {
        let mut parser = Parser::new("let sum = 0~\nfor i in 0..4 { sum = sum + i~ }");
        let program = parser.parse().expect("program should parse");
        let mut interpreter = Interpreter::new();

        interpreter
            .execute(&program)
            .expect("program should execute");

        assert!(matches!(
            interpreter.variables.get("sum"),
            Some(Value::Num(6.0))
        ));
        assert!(matches!(
            interpreter.variables.get("i"),
            Some(Value::Num(4.0))
        ));
    }

    #[test]
    fn break_exits_loop_and_continue_skips_remaining_body() {
        let mut parser = Parser::new(
            "let i = 0~
             let sum = 0~
             while i < 5 {
                 i = i + 1~
                 if i == 2 { continue~ }
                 if i == 4 { break~ }
                 sum = sum + i~
             }",
        );
        let program = parser.parse().expect("program should parse");
        let mut interpreter = Interpreter::new();

        interpreter
            .execute(&program)
            .expect("program should execute");

        assert!(matches!(
            interpreter.variables.get("i"),
            Some(Value::Num(4.0))
        ));
        assert!(matches!(
            interpreter.variables.get("sum"),
            Some(Value::Num(4.0))
        ));
    }

    #[test]
    fn seal_until_stops_when_condition_becomes_true() {
        let mut parser =
            Parser::new("let count = 0~\nseal until count >= 3 { count = count + 1~ }");
        let program = parser.parse().expect("program should parse");
        let mut interpreter = Interpreter::new();

        interpreter
            .execute(&program)
            .expect("program should execute");

        assert!(matches!(
            interpreter.variables.get("count"),
            Some(Value::Num(3.0))
        ));
    }

    #[test]
    fn executes_user_defined_function_with_return() {
        let mut parser = Parser::new("fn add(a, b) { return a + b~ }\nlet result = add(2, 3)~");
        let program = parser.parse().expect("program should parse");
        let mut interpreter = Interpreter::new();

        interpreter
            .execute(&program)
            .expect("program should execute");

        assert!(matches!(
            interpreter.variables.get("result"),
            Some(Value::Num(5.0))
        ));
    }

    #[test]
    fn function_without_explicit_return_uses_last_value() {
        let mut parser = Parser::new("fn one() { let x = 1~ x~ }\nlet result = one()~");
        let program = parser.parse().expect("program should parse");
        let mut interpreter = Interpreter::new();

        interpreter
            .execute(&program)
            .expect("program should execute");

        assert!(matches!(
            interpreter.variables.get("result"),
            Some(Value::Num(1.0))
        ));
    }

    #[test]
    fn function_parameters_do_not_overwrite_outer_variables() {
        let mut parser = Parser::new("let x = 10~\nfn id(x) { return x~ }\nlet y = id(3)~");
        let program = parser.parse().expect("program should parse");
        let mut interpreter = Interpreter::new();

        interpreter
            .execute(&program)
            .expect("program should execute");

        assert!(matches!(
            interpreter.variables.get("x"),
            Some(Value::Num(10.0))
        ));
        assert!(matches!(
            interpreter.variables.get("y"),
            Some(Value::Num(3.0))
        ));
    }

    #[test]
    fn manifold_embed_uses_user_numeric_list() {
        let mut parser = Parser::new(
            "let data = [1.0, 2.0, 3.0, 4.0]~
             manifold M = embed(data, tau=1)~",
        );
        let program = parser.parse().expect("program should parse");
        let mut interpreter = Interpreter::new();

        interpreter
            .execute(&program)
            .expect("program should execute");

        let Value::Manifold(handle) = interpreter.variables.get("M").unwrap() else {
            panic!("expected manifold handle");
        };
        let workspace = &interpreter.manifolds[handle.0];
        assert_eq!(workspace.points.len(), 2);
        assert_eq!(workspace.points[0].coords, [3.0, 2.0, 1.0]);
        assert_eq!(workspace.points[1].coords, [4.0, 3.0, 2.0]);
    }

    #[test]
    fn topology_betti_uses_persistent_homology_engine() {
        let mut parser = Parser::new(
            "import topology~
             let data = [1.0, 1.0, 1.0, 1.0, 1.0]~
             manifold M = embed(data, tau=1)~
             let diagram = topology.ph(M, max_dim=2, mode=\"vr\", max_points=16)~
             let b = topology.betti(diagram, radius=0.0)~",
        );
        let program = parser.parse().expect("program should parse");
        let mut interpreter = Interpreter::new();

        interpreter
            .execute(&program)
            .expect("program should execute");

        let Some(Value::Persistence(diagram)) = interpreter.variables.get("diagram") else {
            panic!("expected persistence diagram");
        };
        assert!(!diagram.pairs.is_empty());

        let Some(Value::List(betti)) = interpreter.variables.get("b") else {
            panic!("expected Betti list");
        };
        assert!(matches!(
            betti.as_slice(),
            [Value::Num(1.0), Value::Num(0.0), Value::Num(0.0)]
        ));
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
