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
//!   ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;

// ═══════════════════════════════════════════════════════════════════════════════
// Expression Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Numeric value (integer or fixed-point float)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Number {
    Int(i64),
    /// Fixed-point: value = int_part + frac_part / 1_000_000
    Float {
        int_part: i64,
        frac_part: i64,
    },
}

impl Number {
    pub fn as_f64(&self) -> f64 {
        match self {
            Number::Int(i) => *i as f64,
            Number::Float {
                int_part,
                frac_part,
            } => *int_part as f64 + (*frac_part as f64 / 1_000_000.0),
        }
    }
}

/// Identifier (variable name, field access, etc.)
pub type Ident = String;

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
    Str(String),

    /// Identifier: M, data, dim
    Ident(Ident),

    /// Field access: M.center, B.spread
    FieldAccess { object: Ident, field: Ident },

    /// Function call: embed(data, dim=3)
    Call { name: Ident, args: Vec<CallArg> },

    /// Method call: M.cluster(0:64)
    MethodCall {
        object: Ident,
        method: Ident,
        args: Vec<CallArg>,
    },

    /// Index/slice: M[0:64]
    Index { object: Ident, range: Range },

    /// Configuration block: { model: "rbf", escalate: true }
    Config(Vec<ConfigPair>),

    /// Object instantiation: new Point(1, 2)
    New { class: Ident, args: Vec<Expr> },

    /// Range expression: 0:64 (can be passed as argument)
    Range(Range),

    /// List literal: [1, 2, 3]
    List(Vec<Expr>),
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
#[derive(Debug, Clone, Default)]
pub struct RegressConfig {
    /// Model type: "polynomial", "rbf", "gp"
    pub model: String,
    /// Polynomial degree (if applicable)
    pub degree: Option<u8>,
    /// Target expression
    pub target: Option<Expr>,
    /// Enable escalating difficulty
    pub escalate: bool,
    /// Convergence condition
    pub until: Option<ConvergenceCond>,
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
#[derive(Debug, Clone, Default)]
pub struct RenderConfig {
    /// Color mode: by_density, gradient, cluster
    pub color: Option<String>,
    /// Highlight specific block
    pub highlight: Option<Ident>,
    /// Show trajectory
    pub trajectory: bool,
    /// Projection axis (for 2D views)
    pub axis: Option<u8>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Control Flow Statements
// ═══════════════════════════════════════════════════════════════════════════════

/// Block of statements (nested scope)
#[derive(Debug, Clone)]
pub struct Block {
    pub statements: Vec<Statement>,
}

/// If statement: if x > 0 { ... } else { ... }
#[derive(Debug, Clone)]
pub struct IfStmt {
    pub condition: Expr,
    pub then_branch: Block,
    pub else_branch: Option<Block>,
}

/// While loop: while x < 10 { ... }
#[derive(Debug, Clone)]
pub struct WhileStmt {
    pub condition: Expr,
    pub body: Block,
}

/// For loop: for i in 0..10 { ... }
#[derive(Debug, Clone)]
pub struct ForStmt {
    pub iterator: Ident,
    pub range: Range, // Simplified: currently only iterating ranges
    pub body: Block,
}

/// Seal loop (topological): seal { ... }
#[derive(Debug, Clone)]
pub struct LoopStmt {
    pub body: Block,
}

/// Function declaration: fn add(a, b) { ... }
#[derive(Debug, Clone)]
pub struct FnDecl {
    pub name: Ident,
    pub params: Vec<Ident>,
    pub body: Block,
}

/// Return statement: return x
#[derive(Debug, Clone)]
pub struct ReturnStmt {
    pub value: Option<Expr>,
}

/// Break statement
#[derive(Debug, Clone)]
pub struct BreakStmt;

/// Continue statement
#[derive(Debug, Clone)]
pub struct ContinueStmt;

/// Class declaration: class Point { x, y, fn init(self) { ... } }
#[derive(Debug, Clone)]
pub struct ClassDecl {
    pub name: Ident,
    pub fields: Vec<VarDecl>, // Fields with default values
    pub methods: Vec<FnDecl>,
}

/// Import statement: import math; from topology import Betti;
#[derive(Debug, Clone)]
pub struct ImportStmt {
    /// Module name: "math"
    pub module: Ident,
    /// Specific symbol: "Betti" (if From import)
    pub symbol: Option<Ident>,
    // Alias: "m" (import math as m) - Future work
}

/// Any statement in an AEGIS program
#[derive(Debug, Clone)]
pub enum Statement {
    Manifold(ManifoldDecl),
    Block(BlockDecl),
    Var(VarDecl),
    Regress(RegressStmt),
    Render(RenderStmt),

    Class(ClassDecl),
    Import(ImportStmt),

    // Control Flow
    If(IfStmt),
    While(WhileStmt),
    For(ForStmt),
    Loop(LoopStmt),
    Fn(FnDecl),
    Return(ReturnStmt),
    Break(BreakStmt),
    Continue(ContinueStmt),

    /// Continue statement
    Continue(ContinueStmt),

    /// Expression statement (method call, assignment, etc.)
    Expr(Expr),

    /// Empty line or comment
    Empty,
}

/// Complete AEGIS program
#[derive(Debug)]
pub struct Program {
    pub statements: Vec<Statement>,
}

impl Program {
    pub fn new() -> Self {
        Self {
            statements: Vec::new(),
        }
    }

    pub fn push(&mut self, stmt: Statement) {
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
