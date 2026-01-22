//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Interpreter - Runtime execution of AEGIS programs
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Executes parsed AEGIS AST, managing:
//! - 3D manifold workspaces
//! - Block geometry computations
//! - Escalating regression benchmarks
//! - Topological convergence detection
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

use crate::lang::ast::*;
use crate::manifold::{TimeDelayEmbedder, ManifoldPoint};
use crate::aether::{BlockMetadata, HierarchicalBlockTree, DriftDetector};
use heapless::{String, FnvIndexMap};
use libm::{sqrt, fabs};

/// Maximum variables in scope
const MAX_VARS: usize = 32;
/// Maximum points in manifold
const MAX_POINTS: usize = 256;
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
    Str(String<128>),
    /// 3D Manifold reference
    Manifold(ManifoldHandle),
    /// Geometric block reference  
    Block(BlockHandle),
    /// 3D Point
    Point([f64; DIM]),
    /// Regression result
    RegressionResult(RegressionOutput),
    /// Void/Unit
    Unit,
}

/// Handle to a manifold workspace
#[derive(Debug, Clone, Copy)]
pub struct ManifoldHandle(pub usize);

/// Handle to a geometric block
#[derive(Debug, Clone, Copy)]
pub struct BlockHandle(pub usize);

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
    pub points: heapless::Vec<ManifoldPoint<DIM>, MAX_POINTS>,
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
            points: heapless::Vec::new(),
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
                let _ = self.points.push(point);
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
            for d in 0..DIM {
                sum[d] += p.coords[d];
            }
        }
        
        let n = self.points.len() as f64;
        for d in 0..DIM {
            self.centroid[d] = sum[d] / n;
        }
    }
    
    /// Extract block from index range
    pub fn extract_block(&self, start: usize, end: usize) -> BlockMetadata<DIM> {
        let end = end.min(self.points.len());
        let start = start.min(end);
        
        if start >= end {
            return BlockMetadata::empty();
        }
        
        // Convert points to array format
        let mut block_points: heapless::Vec<[f64; DIM], 64> = heapless::Vec::new();
        for i in start..end.min(start + 64) {
            let _ = block_points.push(self.points[i].coords);
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
    target: heapless::Vec<f64, MAX_POINTS>,
    /// Predictions
    predictions: heapless::Vec<f64, MAX_POINTS>,
    /// Convergence epsilon
    epsilon: f64,
    /// Betti stability window
    betti_history: heapless::Vec<(u32, u32), 16>,
}

impl EscalatingRegressor {
    pub fn new(epsilon: f64) -> Self {
        Self {
            current_level: 0,
            target: heapless::Vec::new(),
            predictions: heapless::Vec::new(),
            epsilon,
            betti_history: heapless::Vec::new(),
        }
    }
    
    /// Set target values for regression
    pub fn set_target(&mut self, data: &[f64]) {
        self.target.clear();
        for &v in data.iter().take(MAX_POINTS) {
            let _ = self.target.push(v);
        }
    }
    
    /// Run escalating regression until convergence
    pub fn run_escalating(&mut self, manifold: &ManifoldWorkspace, max_epochs: u32) -> RegressionOutput {
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
            let _ = self.betti_history.push(betti);
            
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
            4..=6 => RegressionModel::Rbf { gamma: 0.1 * (epoch as f64) },
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
                    if i >= self.target.len() { break; }
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
    fn compute_error(&self, manifold: &ManifoldWorkspace, coeffs: &[f64; 8], model: &RegressionModel) -> f64 {
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
    
    /// Predict value at x
    fn predict(&self, x: f64, coeffs: &[f64; 8], model: &RegressionModel) -> f64 {
        match model {
            RegressionModel::Linear => coeffs[0] + coeffs[1] * x,
            RegressionModel::Polynomial { degree } => {
                let mut y = coeffs[0];
                let mut x_pow = x;
                for i in 1..=(*degree as usize).min(7) {
                    y += coeffs[i] * x_pow;
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
    fn compute_residual_betti(&self, manifold: &ManifoldWorkspace, coeffs: &[f64; 8], model: &RegressionModel) -> (u32, u32) {
        // Simplified: count sign changes (β₀) and oscillations (β₁)
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
            let recent: heapless::Vec<&(u32, u32), 3> = 
                self.betti_history.iter().rev().take(3).collect();
            
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
    variables: FnvIndexMap<String<64>, Value, MAX_VARS>,
    /// Manifold workspaces
    manifolds: heapless::Vec<ManifoldWorkspace, 4>,
    /// Block geometries
    blocks: heapless::Vec<BlockMetadata<DIM>, 16>,
    /// Sample data (for demo)
    sample_data: [f64; 64],
}

impl Interpreter {
    pub fn new() -> Self {
        // Generate sample data (sine wave for demo)
        let mut data = [0.0f64; 64];
        for (i, d) in data.iter_mut().enumerate() {
            let x = (i as f64) * 0.1;
            *d = libm::sin(x);
        }
        
        Self {
            variables: FnvIndexMap::new(),
            manifolds: heapless::Vec::new(),
            blocks: heapless::Vec::new(),
            sample_data: data,
        }
    }
    
    /// Execute a program
    pub fn execute(&mut self, program: &Program) -> Result<Value, String<128>> {
        let mut last_value = Value::Unit;
        
        for stmt in &program.statements {
            last_value = self.execute_statement(stmt)?;
        }
        
        Ok(last_value)
    }
    
    fn execute_statement(&mut self, stmt: &Statement) -> Result<Value, String<128>> {
        match stmt {
            Statement::Manifold(decl) => self.execute_manifold(decl),
            Statement::Block(decl) => self.execute_block(decl),
            Statement::Var(decl) => self.execute_var(decl),
            Statement::Regress(stmt) => self.execute_regress(stmt),
            Statement::Render(stmt) => self.execute_render(stmt),
            Statement::Empty => Ok(Value::Unit),
        }
    }
    
    fn execute_manifold(&mut self, decl: &ManifoldDecl) -> Result<Value, String<128>> {
        // Extract tau from initialization
        let tau = self.extract_tau(&decl.init).unwrap_or(3);
        
        // Create new manifold workspace
        let mut workspace = ManifoldWorkspace::new(tau);
        
        // Embed sample data
        workspace.embed_data(&self.sample_data);
        
        // Store workspace
        let handle = ManifoldHandle(self.manifolds.len());
        let _ = self.manifolds.push(workspace);
        
        // Bind variable
        let _ = self.variables.insert(decl.name.clone(), Value::Manifold(handle));
        
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
    
    fn execute_block(&mut self, decl: &BlockDecl) -> Result<Value, String<128>> {
        // Get range from source expression
        let (manifold_handle, start, end) = self.extract_block_range(&decl.source)?;
        
        // Extract block from manifold
        if let Some(workspace) = self.manifolds.get(manifold_handle.0) {
            let block = workspace.extract_block(start, end);
            let handle = BlockHandle(self.blocks.len());
            let _ = self.blocks.push(block);
            let _ = self.variables.insert(decl.name.clone(), Value::Block(handle));
            Ok(Value::Block(handle))
        } else {
            let mut err = String::new();
            let _ = err.push_str("manifold not found");
            Err(err)
        }
    }
    
    fn extract_block_range(&self, expr: &Expr) -> Result<(ManifoldHandle, usize, usize), String<128>> {
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
                let _ = err.push_str("invalid block source");
                Err(err)
            }
        }
    }
    
    fn get_manifold_handle(&self, name: &String<64>) -> Result<ManifoldHandle, String<128>> {
        if let Some(Value::Manifold(h)) = self.variables.get(name) {
            Ok(*h)
        } else {
            let mut err = String::new();
            let _ = err.push_str("variable is not a manifold");
            Err(err)
        }
    }
    
    fn extract_range_from_args(&self, args: &heapless::Vec<CallArg, MAX_ARGS>) -> (usize, usize) {
        // Default range
        let mut start = 0usize;
        let mut end = 64usize;
        
        for (i, arg) in args.iter().enumerate() {
            if let CallArg::Positional(Expr::Num(n)) = arg {
                if i == 0 { start = n.as_f64() as usize; }
                if i == 1 { end = n.as_f64() as usize; }
            }
        }
        
        (start, end)
    }
    
    fn execute_var(&mut self, decl: &VarDecl) -> Result<Value, String<128>> {
        let value = self.evaluate_expr(&decl.value)?;
        let _ = self.variables.insert(decl.name.clone(), value.clone());
        Ok(value)
    }
    
    fn execute_regress(&mut self, stmt: &RegressStmt) -> Result<Value, String<128>> {
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
            let _ = err.push_str("no manifold for regression");
            Err(err)
        }
    }
    
    fn execute_render(&mut self, stmt: &RenderStmt) -> Result<Value, String<128>> {
        // In no_std, we just acknowledge the render
        // Real implementation would output to serial/display
        let _target = &stmt.target;
        let _config = &stmt.config;
        
        // TODO: ASCII render or WebGL export
        Ok(Value::Unit)
    }
    
    fn evaluate_expr(&self, expr: &Expr) -> Result<Value, String<128>> {
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
            Expr::FieldAccess { object, field } => {
                self.evaluate_field_access(object, field)
            }
            _ => Ok(Value::Unit),
        }
    }
    
    fn evaluate_field_access(&self, object: &String<64>, field: &String<64>) -> Result<Value, String<128>> {
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
