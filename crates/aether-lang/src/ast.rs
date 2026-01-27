//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Abstract Syntax Tree
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! AST nodes representing the structure of AEGIS programs.
//! Now supporting Source Spans and Titan VM Opcodes.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::boxed::Box;

/// Source Span for diagnostics (Line, Column, Length)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub col: usize,
}

/// Wrapper for AST nodes to map back to Source DNA
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Expression Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Numeric value (integer or fixed-point float) - Legacy support
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Number {
    Int(i64),
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
#[derive(Debug, Clone, PartialEq)]
pub struct Range {
    pub start: Number,
    pub end: Number,
}

/// Key-value pair in configuration blocks
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigPair {
    pub key: Ident,
    pub value: Expr,
}

/// Binary Operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add, Sub, Mul, Div,
    Eq, Neq, Lt, Gt, Le, Ge,
    And, Or,
}

/// Unary Operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg, Not,
}



// ═══════════════════════════════════════════════════════════════════════════════
// Expression Types
// ═══════════════════════════════════════════════════════════════════════════════

/// AEGIS Expression wrapped with source span
pub type Expr = Spanned<ExprKind>;

/// Expression kinds in AEGIS
#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    /// Literal value
    Literal(Literal),
    
    /// Identifier: M, data, dim
    Ident(Ident),

    /// Binary Operation: a + b
    BinaryOp(Box<Expr>, BinaryOp, Box<Expr>),
    
    /// Unary Operation: -a
    UnaryOp(UnaryOp, Box<Expr>),

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

/// Literals
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Num(f64),
    Bool(bool),
    Str(String),
}

/// Argument in function/method call (positional or named)
#[derive(Debug, Clone, PartialEq)]
pub enum CallArg {
    Positional(Expr),
    Named { name: Ident, value: Expr },
}

// ═══════════════════════════════════════════════════════════════════════════════
// Statement Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Manifold declaration: manifold M = embed(data, dim=3, tau=5)
#[derive(Debug, Clone, PartialEq)]
pub struct ManifoldDecl {
    pub name: Ident,
    pub init: Expr,
}

/// Block declaration: block B = M.cluster(0:64)
#[derive(Debug, Clone, PartialEq)]
pub struct BlockDecl {
    pub name: Ident,
    pub source: Expr,
}

/// Variable assignment: centroid C = B.center
#[derive(Debug, Clone, PartialEq)]
pub struct VarDecl {
    pub type_hint: Option<Ident>,
    pub name: Ident,
    pub value: Expr,
}

/// Regression statement with configuration
#[derive(Debug, Clone, PartialEq)]
pub struct RegressStmt {
    pub config: RegressConfig,
}

/// Regression configuration
#[derive(Debug, Clone, Default, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceCond {
    /// Epsilon-based: convergence(epsilon=1e-6)
    Epsilon(Number),
    /// Betti stability: betti_stable(epochs=10)
    BettiStable { epochs: u32 },
    /// Custom expression
    Custom(Expr),
}

/// Render statement: render M { color: by_density }
#[derive(Debug, Clone, PartialEq)]
pub struct RenderStmt {
    pub target: Ident,
    pub config: RenderConfig,
}

/// Render configuration
#[derive(Debug, Clone, Default, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub statements: Vec<Statement>,
}

/// If statement: if x > 0 { ... } else { ... }
#[derive(Debug, Clone, PartialEq)]
pub struct IfStmt {
    pub condition: Expr,
    pub then_branch: Block,
    pub else_branch: Option<Block>,
}

/// While loop: while x < 10 { ... }
#[derive(Debug, Clone, PartialEq)]
pub struct WhileStmt {
    pub condition: Expr,
    pub body: Block,
}

/// For loop: for i in 0..10 { ... }
#[derive(Debug, Clone, PartialEq)]
pub struct ForStmt {
    pub iterator: Ident,
    pub range: Range, // Simplified: currently only iterating ranges
    pub body: Block,
}

/// Seal loop (topological): seal { ... }
#[derive(Debug, Clone, PartialEq)]
pub struct LoopStmt {
    pub body: Block,
}

/// Function declaration: fn add(a, b) { ... }
#[derive(Debug, Clone, PartialEq)]
pub struct FnDecl {
    pub name: Ident,
    pub params: Vec<Ident>,
    pub body: Block,
}

/// Return statement: return x
#[derive(Debug, Clone, PartialEq)]
pub struct ReturnStmt {
    pub value: Option<Expr>,
}

/// Break statement
#[derive(Debug, Clone, PartialEq)]
pub struct BreakStmt;

/// Continue statement
#[derive(Debug, Clone, PartialEq)]
pub struct ContinueStmt;

/// Class declaration: class Point { x, y, fn init(self) { ... } }
#[derive(Debug, Clone, PartialEq)]
pub struct ClassDecl {
    pub name: Ident,
    pub fields: Vec<VarDecl>, // Fields with default values
    pub methods: Vec<FnDecl>,
}

/// Import statement: import math; from topology import Betti;
#[derive(Debug, Clone, PartialEq)]
pub struct ImportStmt {
    pub module: Ident,
    pub symbol: Option<Ident>,
}

/// AEGIS Statement wrapped with source span
pub type Statement = Spanned<StmtKind>;

/// Statement kinds in AEGIS
#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
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

    /// Expression statement (method call, assignment, etc.)
    Expr(Expr),

    /// Empty line or comment
    Empty,
}

/// Complete AEGIS program
#[derive(Debug, Clone)]
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
