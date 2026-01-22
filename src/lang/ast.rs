//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Abstract Syntax Tree
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! AST nodes representing the structure of AEGIS programs.
//!
//! Core Constructs:
//! - ManifoldDecl: 3D embedded space definition
//! - BlockDecl: Geometric cluster extraction
//! - RegressStmt: Non-linear regression with escalation
//! - RenderStmt: 3D visualization directives
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

use heapless::String;
use heapless::Vec as HVec;

/// Maximum nesting depth for AST
pub const MAX_DEPTH: usize = 16;
/// Maximum statements per program
pub const MAX_STATEMENTS: usize = 64;
/// Maximum arguments per call
pub const MAX_ARGS: usize = 8;

// ═══════════════════════════════════════════════════════════════════════════════
// Expression Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Numeric value (integer or fixed-point float)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Number {
    Int(i64),
    /// Fixed-point: value = int_part + frac_part / 1_000_000
    Float { int_part: i64, frac_part: i64 },
}

impl Number {
    pub fn as_f64(&self) -> f64 {
        match self {
            Number::Int(i) => *i as f64,
            Number::Float { int_part, frac_part } => {
                *int_part as f64 + (*frac_part as f64 / 1_000_000.0)
            }
        }
    }
}

/// Identifier (variable name, field access, etc.)
pub type Ident = String<64>;

/// Range expression: start:end
#[derive(Debug, Clone)]
pub struct Range {
    pub start: Number,
    pub end: Number,
}

/// Key-value pair in configuration blocks
#[derive(Debug, Clone)]
pub struct ConfigPair {
    pub key: Ident,
    pub value: Expr,
}

/// Expression in AEGIS
#[derive(Debug, Clone)]
pub enum Expr {
    /// Numeric literal: 42, 3.14159
    Num(Number),
    
    /// Boolean literal: true, false
    Bool(bool),
    
    /// String literal: "polynomial"
    Str(String<128>),
    
    /// Identifier: M, data, dim
    Ident(Ident),
    
    /// Field access: M.center, B.spread
    FieldAccess {
        object: Ident,
        field: Ident,
    },
    
    /// Function call: embed(data, dim=3)
    Call {
        name: Ident,
        args: HVec<CallArg, MAX_ARGS>,
    },
    
    /// Method call: M.cluster(0:64)
    MethodCall {
        object: Ident,
        method: Ident,
        args: HVec<CallArg, MAX_ARGS>,
    },
    
    /// Index/slice: M[0:64]
    Index {
        object: Ident,
        range: Range,
    },
    
    /// Configuration block: { model: "rbf", escalate: true }
    Config(HVec<ConfigPair, MAX_ARGS>),
}

/// Argument in function/method call (positional or named)
#[derive(Debug, Clone)]
pub enum CallArg {
    Positional(Expr),
    Named { name: Ident, value: Expr },
}

// ═══════════════════════════════════════════════════════════════════════════════
// Statement Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Manifold declaration: manifold M = embed(data, dim=3, tau=5)
#[derive(Debug, Clone)]
pub struct ManifoldDecl {
    pub name: Ident,
    pub init: Expr,
}

/// Block declaration: block B = M.cluster(0:64)
#[derive(Debug, Clone)]
pub struct BlockDecl {
    pub name: Ident,
    pub source: Expr,
}

/// Variable assignment: centroid C = B.center
#[derive(Debug, Clone)]
pub struct VarDecl {
    pub type_hint: Option<Ident>,
    pub name: Ident,
    pub value: Expr,
}

/// Regression statement with configuration
#[derive(Debug, Clone)]
pub struct RegressStmt {
    pub config: RegressConfig,
}

/// Regression configuration
#[derive(Debug, Clone)]
pub struct RegressConfig {
    /// Model type: "polynomial", "rbf", "gp"
    pub model: String<32>,
    /// Polynomial degree (if applicable)
    pub degree: Option<u8>,
    /// Target expression
    pub target: Option<Expr>,
    /// Enable escalating difficulty
    pub escalate: bool,
    /// Convergence condition
    pub until: Option<ConvergenceCond>,
}

impl Default for RegressConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            degree: None,
            target: None,
            escalate: false,
            until: None,
        }
    }
}

/// Convergence condition
#[derive(Debug, Clone)]
pub enum ConvergenceCond {
    /// Epsilon-based: convergence(epsilon=1e-6)
    Epsilon(Number),
    /// Betti stability: betti_stable(epochs=10)
    BettiStable { epochs: u32 },
    /// Custom expression
    Custom(Expr),
}

/// Render statement: render M { color: by_density }
#[derive(Debug, Clone)]
pub struct RenderStmt {
    pub target: Ident,
    pub config: RenderConfig,
}

/// Render configuration
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Color mode: by_density, gradient, cluster
    pub color: Option<String<32>>,
    /// Highlight specific block
    pub highlight: Option<Ident>,
    /// Show trajectory
    pub trajectory: bool,
    /// Projection axis (for 2D views)
    pub axis: Option<u8>,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            color: None,
            highlight: None,
            trajectory: false,
            axis: None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Top-Level AST
// ═══════════════════════════════════════════════════════════════════════════════

/// Any statement in an AEGIS program
#[derive(Debug, Clone)]
pub enum Statement {
    Manifold(ManifoldDecl),
    Block(BlockDecl),
    Var(VarDecl),
    Regress(RegressStmt),
    Render(RenderStmt),
    /// Empty line or comment
    Empty,
}

/// Complete AEGIS program
#[derive(Debug)]
pub struct Program {
    pub statements: HVec<Statement, MAX_STATEMENTS>,
}

impl Program {
    pub fn new() -> Self {
        Self {
            statements: HVec::new(),
        }
    }
    
    pub fn push(&mut self, stmt: Statement) -> Result<(), Statement> {
        self.statements.push(stmt)
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// AST Visitors (for interpretation and analysis)
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for visiting AST nodes
pub trait AstVisitor {
    type Output;
    type Error;
    
    fn visit_program(&mut self, prog: &Program) -> Result<Self::Output, Self::Error>;
    fn visit_statement(&mut self, stmt: &Statement) -> Result<(), Self::Error>;
    fn visit_expr(&mut self, expr: &Expr) -> Result<Self::Output, Self::Error>;
}
