// ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Parser - Recursive descent parser for .aegis scripts
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Converts token stream to AST for interpretation.
//!
//! Grammar (simplified):
//!   program     → statement* EOF
//!   statement   → manifold_decl | block_decl | regress_stmt | render_stmt | var_decl
//!   manifold_decl → "manifold" IDENT "=" expr
//!   regress_stmt → "regress" config_block
//!   config_block → "{" (IDENT ":" expr ",")* "}"
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

extern crate alloc;
use crate::ast::*;
use crate::lexer::{Lexer, Token, TokenKind};
use alloc::string::String;
use alloc::vec::Vec;


#[cfg(not(feature = "std"))]
use alloc::{format, vec};
#[cfg(not(feature = "std"))]
use alloc::string::ToString;

#[cfg(not(feature = "std"))]
macro_rules! println {
    ($($arg:tt)*) => {};
}

/// Parser error
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

impl ParseError {
    pub fn new(msg: &str, line: usize, column: usize) -> Self {
        let mut message = String::new();
        message.push_str(msg);
        Self {
            message,
            line,
            column,
        }
    }
}

/// AEGIS Parser
pub struct Parser<'a> {
    tokens: Vec<Token>,
    current: usize,
    _source: &'a str,
}

impl<'a> Parser<'a> {
    /// Create parser from source text
    pub fn new(source: &'a str) -> Self {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();

        Self {
            tokens,
            current: 0,
            _source: source,
        }
    }

    /// Parse entire program
    pub fn parse(&mut self) -> Result<Program, ParseError> {
        let mut program = Program::new();

        while !self.is_at_end() {
            // Skip empty lines
            while self.check(TokenKind::Newline) {
                self.advance();
            }

            if self.is_at_end() {
                break;
            }

            let stmt = self.parse_statement()?;
            if !matches!(stmt, Statement::Empty) {
                program.push(stmt);
            }
        }

        Ok(program)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Statement Parsing
    // ═══════════════════════════════════════════════════════════════════════════

    fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        let token = self.peek();

        match &token.kind {
            TokenKind::Manifold => self.parse_manifold_decl(),
            TokenKind::Block => self.parse_block_decl(),
            TokenKind::Regress => self.parse_regress_stmt(),
            TokenKind::Render => self.parse_render_stmt(),
            TokenKind::Identifier(_) => self.parse_ident_start_stmt(),

            // Class Declaration
            TokenKind::Class => self.parse_class_decl(),

            // Modules
            TokenKind::Import => self.parse_import_stmt(),
            TokenKind::From => self.parse_from_import_stmt(),

            // Control Flow
            TokenKind::If => self.parse_if_stmt(),
            TokenKind::While => self.parse_while_stmt(),
            TokenKind::For => self.parse_for_stmt(),
            TokenKind::Seal => self.parse_seal_stmt(),
            TokenKind::Fn => self.parse_fn_decl(),
            TokenKind::Return => self.parse_return_stmt(),
            TokenKind::Break => self.parse_break_stmt(),
            TokenKind::Continue => self.parse_continue_stmt(),
            TokenKind::Let => self.parse_let_decl(),

            TokenKind::Newline | TokenKind::Eof => Ok(Statement::Empty),
            _ => {
                let line = token.line;
                let col = token.column;
                Err(ParseError::new(
                    &format!("unexpected token: {:?}", token.kind),
                    line,
                    col,
                ))
            }
        }
    }

    /// manifold_decl → "manifold" IDENT "=" expr
    fn parse_manifold_decl(&mut self) -> Result<Statement, ParseError> {
        self.expect(TokenKind::Manifold)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::Equals)?;
        let init = self.parse_expr()?;

        Ok(Statement::Manifold(ManifoldDecl { name, init }))
    }

    /// block_decl → "block" IDENT "=" expr
    fn parse_block_decl(&mut self) -> Result<Statement, ParseError> {
        self.expect(TokenKind::Block)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::Equals)?;
        let source = self.parse_expr()?;

        Ok(Statement::Block(BlockDecl { name, source }))
    }

    /// ident_start_stmt → var_decl | expr_stmt
    fn parse_ident_start_stmt(&mut self) -> Result<Statement, ParseError> {
        let first_ident = self.expect_ident()?;

        // 1. Check for type hint: Ident Ident = Expr
        if self.check_ident() && self.peek_next_is(TokenKind::Equals) {
            let type_hint = Some(first_ident);
            let name = self.expect_ident()?;
            self.expect(TokenKind::Equals)?;
            let value = self.parse_expr()?;

            Ok(Statement::Var(VarDecl {
                type_hint,
                name,
                value,
            }))
        } 
        // 2. Check for Var Decl without type: Ident = Expr
        else if self.check(TokenKind::Equals) {
            self.expect(TokenKind::Equals)?;
            let value = self.parse_expr()?;

            Ok(Statement::Var(VarDecl {
                type_hint: None,
                name: first_ident,
                value,
            }))
        }
        // 3. Expression Statement (e.g., method call) starting with Ident
        else {
            // We consumed the identifier. Parse the rest as an expression starting with this ident.
            let expr = self.parse_ident_expr(first_ident)?;
            Ok(Statement::Expr(expr))
        }
    }

    /// let_decl → "let" IDENT "=" expr
    fn parse_let_decl(&mut self) -> Result<Statement, ParseError> {
        self.expect(TokenKind::Let)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::Equals)?;
        let value = self.parse_expr()?;

        Ok(Statement::Var(VarDecl {
            type_hint: None,
            name,
            value,
        }))
    }

    /// regress_stmt → "regress" config_block
    fn parse_regress_stmt(&mut self) -> Result<Statement, ParseError> {
        self.expect(TokenKind::Regress)?;
        let config = self.parse_regress_config()?;

        Ok(Statement::Regress(RegressStmt { config }))
    }

    fn parse_regress_config(&mut self) -> Result<RegressConfig, ParseError> {
        self.expect(TokenKind::LBrace)?;

        let mut config = RegressConfig::default();

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            // Skip newlines inside block
            if self.check(TokenKind::Newline) {
                self.advance();
                continue;
            }

            let token = self.peek().clone();
            let key = match &token.kind {
                TokenKind::Identifier(s) => s.clone(),
                TokenKind::Model => String::from("model"),
                TokenKind::Escalate => String::from("escalate"),
                TokenKind::Until => String::from("until"),
                // Add any other keywords used as keys here
                _ => return Err(ParseError::new("expected identifier or config key", token.line, token.column)),
            };
            self.advance(); // consume key
            self.expect(TokenKind::Colon)?;

            match key.as_str() {
                "model" => {
                    if let Expr::Str(s) = self.parse_expr()? {
                        config.model = s;
                    }
                }
                "degree" => {
                    if let Expr::Num(Number::Int(n)) = self.parse_expr()? {
                        config.degree = Some(n as u8);
                    }
                }
                "target" => {
                    config.target = Some(self.parse_expr()?);
                }
                "escalate" => {
                    if let Expr::Bool(b) = self.parse_expr()? {
                        config.escalate = b;
                    }
                }
                "until" => {
                    config.until = Some(self.parse_convergence()?);
                }
                _ => {
                    // Skip unknown config keys
                    let _ = self.parse_expr()?;
                }
            }

            // Optional comma
            if self.check(TokenKind::Comma) {
                self.advance();
            }
        }

        self.expect(TokenKind::RBrace)?;
        Ok(config)
    }

    fn parse_convergence(&mut self) -> Result<ConvergenceCond, ParseError> {
        // convergence(epsilon=1e-6) or convergence(1e-6)
        if self.check(TokenKind::Convergence) {
            self.advance();
            self.expect(TokenKind::LParen)?;
            let value = self.parse_expr()?;
            self.expect(TokenKind::RParen)?;

            if let Expr::Num(n) = value {
                return Ok(ConvergenceCond::Epsilon(n));
            }
        }

        // Fallback to custom expression
        let expr = self.parse_expr()?;
        Ok(ConvergenceCond::Custom(expr))
    }

    /// render_stmt → "render" IDENT config_block?
    fn parse_render_stmt(&mut self) -> Result<Statement, ParseError> {
        self.expect(TokenKind::Render)?;
        let target = self.expect_ident()?;

        let config = if self.check(TokenKind::LBrace) {
            self.parse_render_config()?
        } else {
            RenderConfig::default()
        };

        Ok(Statement::Render(RenderStmt { target, config }))
    }

    fn parse_render_config(&mut self) -> Result<RenderConfig, ParseError> {
        self.expect(TokenKind::LBrace)?;

        let mut config = RenderConfig::default();

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            if self.check(TokenKind::Newline) {
                self.advance();
                continue;
            }

            let key = self.expect_flexible_ident()?;
            self.expect(TokenKind::Colon)?;

            match key.as_str() {
                "color" => {
                    let expr = self.parse_expr()?;
                    if let Expr::Ident(id) = expr {
                        config.color = Some(id);
                    }
                }
                "highlight" => {
                    if let Expr::Ident(id) = self.parse_expr()? {
                        config.highlight = Some(id);
                    }
                }
                "trajectory" => {
                    let expr = self.parse_expr()?;
                    if let Expr::Bool(b) = &expr {
                        config.trajectory = *b;
                    } else if let Expr::Ident(id) = &expr {
                        config.trajectory = id.as_str() == "on";
                    }
                }
                "axis" => {
                    if let Expr::Num(Number::Int(n)) = self.parse_expr()? {
                        config.axis = Some(n as u8);
                    }
                }
                _ => {
                    let _ = self.parse_expr()?;
                }
            }

            if self.check(TokenKind::Comma) {
                self.advance();
            }
        }

        self.expect(TokenKind::RBrace)?;
        Ok(config)
    }

    /// class_decl → "class" IDENT "{" (var_decl | fn_decl)* "}"
    fn parse_class_decl(&mut self) -> Result<Statement, ParseError> {
        self.expect(TokenKind::Class)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::LBrace)?;

        let mut fields = Vec::new();
        let mut methods = Vec::new();

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            if self.check(TokenKind::Newline) {
                self.advance();
                continue;
            }

            if self.check(TokenKind::Fn) {
                // Method
                if let Statement::Fn(f) = self.parse_fn_decl()? {
                    methods.push(f);
                }
            } else if self.check_ident() {
                // Field with optional default value: x = 0
                // We'll reuse parse_var_decl logic but simpler
                let field_name = self.expect_ident()?;
                let value = if self.check(TokenKind::Equals) {
                    self.advance();
                    self.parse_expr()?
                } else {
                    Expr::Bool(false) // Default to false if no value
                };

                // Allow comma or newline separator
                if self.check(TokenKind::Comma) {
                    self.advance();
                }

                fields.push(VarDecl {
                    type_hint: None,
                    name: field_name,
                    value,
                });
            } else {
                // Unexpected token
                let t = self.peek();
                return Err(ParseError::new(
                    "expected field or method",
                    t.line,
                    t.column,
                ));
            }
        }

        self.expect(TokenKind::RBrace)?;

        Ok(Statement::Class(ClassDecl {
            name,
            fields,
            methods,
        }))
    }

    /// import_stmt -> "import" IDENT
    fn parse_import_stmt(&mut self) -> Result<Statement, ParseError> {
        self.expect(TokenKind::Import)?;
        let module = self.expect_ident()?;
        // Todo: handle 'as'

        Ok(Statement::Import(ImportStmt {
            module,
            symbol: None,
        }))
    }

    /// from_import_stmt -> "from" IDENT "import" IDENT
    fn parse_from_import_stmt(&mut self) -> Result<Statement, ParseError> {
        self.expect(TokenKind::From)?;
        let module = self.expect_ident()?;
        self.expect(TokenKind::Import)?;
        let symbol = self.expect_ident()?;

        Ok(Statement::Import(ImportStmt {
            module,
            symbol: Some(symbol),
        }))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Expression Parsing
    // ═══════════════════════════════════════════════════════════════════════════

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_primary()?;

        if self.check(TokenKind::Colon) {
            self.advance();
            // AST Range currently only supports Numbers
            if let Expr::Num(start) = left {
                let right = self.parse_primary()?;
                if let Expr::Num(end) = right {
                    return Ok(Expr::Range(Range { start, end }));
                } else {
                    let token = self.peek();
                    return Err(ParseError::new("range end must be a number", token.line, token.column));
                }
            } else {
                 let token = self.peek();
                 return Err(ParseError::new("range start must be a number", token.line, token.column));
            }
        }
        
        Ok(left)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let token = self.advance();

        match token.kind {
            TokenKind::Number(n) => Ok(Expr::Num(Number::Int(n))),
            TokenKind::Float(int, frac) => Ok(Expr::Num(Number::Float {
                int_part: int,
                frac_part: frac,
            })),
            TokenKind::True => Ok(Expr::Bool(true)),
            TokenKind::False => Ok(Expr::Bool(false)),
            TokenKind::StringLit(s) => Ok(Expr::Str(s)),
            TokenKind::Self_ => Ok(Expr::Ident(String::from("self"))),

            TokenKind::Identifier(name) => self.parse_ident_expr(name),

            // New Object Instantiation
            TokenKind::New => {
                let class = self.expect_ident()?;
                self.expect(TokenKind::LParen)?;
                let mut args = Vec::new();
                while !self.check(TokenKind::RParen) && !self.is_at_end() {
                    args.push(self.parse_expr()?);
                    if self.check(TokenKind::Comma) {
                        self.advance();
                    }
                }
                self.expect(TokenKind::RParen)?;
                Ok(Expr::New { class, args })
            }

            // Function keywords that act as function names
            TokenKind::Embed => self.parse_call_expr({
                let mut s = String::new();
                s.push_str("embed");
                s
            }),
            TokenKind::Convergence => self.parse_call_expr({
                let mut s = String::new();
                s.push_str("convergence");
                s
            }),

            // List literals: [1, 2, 3]
            TokenKind::LBracket => self.parse_list_literal(),

            _ => Err(ParseError::new(
                "expected expression",
                token.line,
                token.column,
            )),
        }
    }

    fn parse_list_literal(&mut self) -> Result<Expr, ParseError> {
        // LBracket already consumed by advance() before calling parse_primary? 
        // No, parse_primary called advance() which returned the token. 
        // So we are inside parse_primary match.
        // Wait, parse_primary calls `self.advance()` at the start.
        // So LBracket is already consumed.

        let mut elements = Vec::new();

        while !self.check(TokenKind::RBracket) && !self.is_at_end() {
             // Handle newlines inside lists
             if self.check(TokenKind::Newline) {
                 self.advance();
                 continue;
             }
             
             elements.push(self.parse_expr()?);

             if self.check(TokenKind::Comma) {
                 self.advance();
             }
        }

        self.expect(TokenKind::RBracket)?;
        Ok(Expr::List(elements))
    }

    fn parse_ident_expr(&mut self, name: String) -> Result<Expr, ParseError> {
        // DEBUG
        println!("Parsing ident expr for: {}", name);
        if !self.is_at_end() {
             println!("Next token: {:?}", self.peek().kind);
        }

        // Check for method call: M.cluster(...)
        if self.check(TokenKind::Dot) {
            self.advance();
            let method = self.expect_flexible_ident()?;

            if self.check(TokenKind::LParen) {
                // Method call
                let args = self.parse_call_args()?;
                return Ok(Expr::MethodCall {
                    object: name,
                    method,
                    args,
                });
            } else {
                // Field access
                return Ok(Expr::FieldAccess {
                    object: name,
                    field: method,
                });
            }
        }

        // Check for function call: embed(...)
        if self.check(TokenKind::LParen) {
            return self.parse_call_expr(name);
        }

        // Check for index: M[0:64]
        if self.check(TokenKind::LBracket) {
            self.advance();
            let start = self.parse_number()?;
            self.expect(TokenKind::Colon)?;
            let end = self.parse_number()?;
            self.expect(TokenKind::RBracket)?;

            return Ok(Expr::Index {
                object: name,
                range: Range { start, end },
            });
        }

        Ok(Expr::Ident(name))
    }

    fn parse_call_expr(&mut self, name: String) -> Result<Expr, ParseError> {
        let args = self.parse_call_args()?;
        Ok(Expr::Call { name, args })
    }

    fn parse_call_args(&mut self) -> Result<Vec<CallArg>, ParseError> {
        self.expect(TokenKind::LParen)?;

        let mut args = Vec::new();

        while !self.check(TokenKind::RParen) && !self.is_at_end() {
            // Check for named argument: dim=3
            if self.check_flexible_ident() {
                let saved_pos = self.current;
                let name = self.expect_flexible_ident()?;

                if self.check(TokenKind::Equals) {
                    self.advance();
                    let value = self.parse_expr()?;
                    args.push(CallArg::Named { name, value });
                } else {
                    // Rewind - it's a positional arg that's an identifier/keyword looks like arg
                    self.current = saved_pos;
                    let expr = self.parse_expr()?;
                    args.push(CallArg::Positional(expr));
                }
            } else {
                let expr = self.parse_expr()?;
                args.push(CallArg::Positional(expr));
            }

            if self.check(TokenKind::Comma) {
                self.advance();
            }
        }

        self.expect(TokenKind::RParen)?;
        Ok(args)
    }

    fn parse_number(&mut self) -> Result<Number, ParseError> {
        let token = self.advance();
        match token.kind {
            TokenKind::Number(n) => Ok(Number::Int(n)),
            TokenKind::Float(int, frac) => Ok(Number::Float {
                int_part: int,
                frac_part: frac,
            }),
            _ => Err(ParseError::new("expected number", token.line, token.column)),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Helper Methods
    // ═══════════════════════════════════════════════════════════════════════════

    fn peek(&self) -> &Token {
        self.tokens.get(self.current).unwrap_or(&Token {
            kind: TokenKind::Eof,
            line: 0,
            column: 0,
        })
    }

    fn advance(&mut self) -> Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.tokens.get(self.current - 1).cloned().unwrap_or(Token {
            kind: TokenKind::Eof,
            line: 0,
            column: 0,
        })
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Eof)
    }

    fn check(&self, kind: TokenKind) -> bool {
        if self.is_at_end() {
            return false;
        }
        core::mem::discriminant(&self.peek().kind) == core::mem::discriminant(&kind)
    }

    fn check_ident(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Identifier(_))
    }

    fn peek_next_is(&self, kind: TokenKind) -> bool {
        self.tokens
            .get(self.current + 1)
            .map(|t| core::mem::discriminant(&t.kind) == core::mem::discriminant(&kind))
            .unwrap_or(false)
    }

    fn expect(&mut self, kind: TokenKind) -> Result<Token, ParseError> {
        if self.check(kind.clone()) {
            Ok(self.advance())
        } else {
            let token = self.peek();
            Err(ParseError::new(
                "unexpected token",
                token.line,
                token.column,
            ))
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        let token = self.advance();
        match token.kind {
            TokenKind::Identifier(s) => Ok(s),
            _ => Err(ParseError::new(
                "expected identifier",
                token.line,
                token.column,
            )),
        }
    }

    fn check_flexible_ident(&self) -> bool {
        match &self.peek().kind {
            TokenKind::Identifier(_) => true,
            // Allow specific keywords as argument names
            TokenKind::Dim
            | TokenKind::Tau
            | TokenKind::Model
            | TokenKind::Color
            | TokenKind::Axis
            | TokenKind::Project
            | TokenKind::Cluster
            | TokenKind::Center
            | TokenKind::Spread
            | TokenKind::Format
            | TokenKind::Output
            | TokenKind::Escalate
            | TokenKind::Convergence => true,
            _ => false,
        }
    }

    fn expect_flexible_ident(&mut self) -> Result<String, ParseError> {
        let token = self.advance();
        match token.kind {
            TokenKind::Identifier(s) => Ok(s),
            TokenKind::Dim => Ok(String::from("dim")),
            TokenKind::Tau => Ok(String::from("tau")),
            TokenKind::Model => Ok(String::from("model")),
            TokenKind::Color => Ok(String::from("color")),
            TokenKind::Axis => Ok(String::from("axis")),
            TokenKind::Project => Ok(String::from("project")),
            TokenKind::Cluster => Ok(String::from("cluster")),
            TokenKind::Center => Ok(String::from("center")),
            TokenKind::Spread => Ok(String::from("spread")),
            TokenKind::Format => Ok(String::from("format")),
            TokenKind::Output => Ok(String::from("output")),
            TokenKind::Escalate => Ok(String::from("escalate")),
            TokenKind::Convergence => Ok(String::from("convergence")),
            _ => Err(ParseError::new(
                "expected argument name",
                token.line,
                token.column,
            )),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Control Flow Parsing
    // ═══════════════════════════════════════════════════════════════════════════

    /// Parse a block of statements: { stmt* }
    fn parse_block_stmts(&mut self) -> Result<Block, ParseError> {
        self.expect(TokenKind::LBrace)?;

        let mut statements = Vec::new();

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            // Skip newlines
            while self.check(TokenKind::Newline) {
                self.advance();
            }

            if self.check(TokenKind::RBrace) {
                break;
            }

            let stmt = self.parse_statement()?;
            if !matches!(stmt, Statement::Empty) {
                statements.push(stmt);
            }

            // Consume optional tilde statement terminator
            if self.check(TokenKind::Tilde) {
                self.advance();
            }
        }

        self.expect(TokenKind::RBrace)?;

        Ok(Block {
            statements: statements.into_iter().collect(),
        })
    }

    /// if expr { block } else { block }
    fn parse_if_stmt(&mut self) -> Result<Statement, ParseError> {
        self.expect(TokenKind::If)?;
        let condition = self.parse_comparison()?;
        let then_branch = self.parse_block_stmts()?;

        let else_branch = if self.check(TokenKind::Else) {
            self.advance();
            Some(self.parse_block_stmts()?)
        } else {
            None
        };

        Ok(Statement::If(IfStmt {
            condition,
            then_branch,
            else_branch,
        }))
    }

    /// while expr { block }
    fn parse_while_stmt(&mut self) -> Result<Statement, ParseError> {
        self.expect(TokenKind::While)?;
        let condition = self.parse_comparison()?;
        let body = self.parse_block_stmts()?;

        Ok(Statement::While(WhileStmt { condition, body }))
    }

    /// for ident in start..end { block }
    fn parse_for_stmt(&mut self) -> Result<Statement, ParseError> {
        self.expect(TokenKind::For)?;
        let iterator = self.expect_ident()?;
        self.expect(TokenKind::In)?;

        let start = self.parse_number()?;
        self.expect(TokenKind::DotDot)?;
        let end = self.parse_number()?;

        let body = self.parse_block_stmts()?;

        Ok(Statement::For(ForStmt {
            iterator,
            range: Range { start, end },
            body,
        }))
    }

    /// seal until expr { block } or 🦭 until expr { block }
    fn parse_seal_stmt(&mut self) -> Result<Statement, ParseError> {
        self.expect(TokenKind::Seal)?;

        // Optional 'until' with convergence condition (parsed but ignored for now)
        if self.check(TokenKind::Until) {
            self.advance();
            let _ = self.parse_expr()?; // Consume condition
        }

        let body = self.parse_block_stmts()?;

        Ok(Statement::Loop(LoopStmt { body }))
    }

    /// fn name(params) { block }
    fn parse_fn_decl(&mut self) -> Result<Statement, ParseError> {
        self.expect(TokenKind::Fn)?;
        let name = self.expect_ident()?;

        self.expect(TokenKind::LParen)?;
        let mut params = Vec::<String>::new();

        while !self.check(TokenKind::RParen) && !self.is_at_end() {
            let param = self.expect_ident()?;
            params.push(param);

            if self.check(TokenKind::Comma) {
                self.advance();
            }
        }
        self.expect(TokenKind::RParen)?;

        let body = self.parse_block_stmts()?;

        Ok(Statement::Fn(FnDecl {
            name,
            params: params.into_iter().collect(),
            body,
        }))
    }

    /// return expr?
    fn parse_return_stmt(&mut self) -> Result<Statement, ParseError> {
        self.expect(TokenKind::Return)?;

        let value = if !self.check(TokenKind::Newline)
            && !self.check(TokenKind::Tilde)
            && !self.is_at_end()
        {
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(Statement::Return(ReturnStmt { value }))
    }

    /// break
    fn parse_break_stmt(&mut self) -> Result<Statement, ParseError> {
        self.expect(TokenKind::Break)?;
        Ok(Statement::Break(BreakStmt))
    }

    /// continue
    fn parse_continue_stmt(&mut self) -> Result<Statement, ParseError> {
        self.expect(TokenKind::Continue)?;
        Ok(Statement::Continue(ContinueStmt))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Comparison & Expression Parsing (for conditions)
    // ═══════════════════════════════════════════════════════════════════════════

    fn parse_comparison(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_expr()?;

        // Check for comparison operators
        if self.check(TokenKind::Less)
            || self.check(TokenKind::Greater)
            || self.check(TokenKind::LessEq)
            || self.check(TokenKind::GreaterEq)
            || self.check(TokenKind::EqEq)
            || self.check(TokenKind::NotEq)
        {
            let op_token = self.advance();
            let right = self.parse_expr()?;

            // For now, encode comparison as a method call
            let op_name = match op_token.kind {
                TokenKind::Less => "__lt__",
                TokenKind::Greater => "__gt__",
                TokenKind::LessEq => "__le__",
                TokenKind::GreaterEq => "__ge__",
                TokenKind::EqEq => "__eq__",
                TokenKind::NotEq => "__ne__",
                _ => "__cmp__",
            };

            let mut name = String::new();
            name.push_str(op_name);

            let args = vec![CallArg::Positional(left), CallArg::Positional(right)];

            return Ok(Expr::Call { name, args });
        }

        Ok(left)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_manifold() {
        let mut parser = Parser::new("manifold M = embed(data, dim=3)");
        let program = parser.parse().unwrap();

        assert_eq!(program.statements.len(), 1);
        assert!(matches!(program.statements[0], Statement::Manifold(_)));
    }

    #[test]
    fn test_parse_regress() {
        let source = r#"
regress {
    model: "polynomial",
    degree: 3,
    escalate: true
}
"#;
        let mut parser = Parser::new(source);
        let program = parser.parse().unwrap();

        assert!(matches!(program.statements[0], Statement::Regress(_)));
    }
}
