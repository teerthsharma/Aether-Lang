// ═══════════════════════════════════════════════════════════════════════════════
//! AETHER Parser - Recursive descent parser for .aether scripts
// ═══════════════════════════════════════════════════════════════════════════════
//!
//! Converts token stream to AST for interpretation.
//! Now produces Spanned nodes for precise error reporting.
//!
//! Grammar (simplified):
//!   program     → statement* EOF
//!   statement   → manifold_decl | block_decl | regress_stmt | render_stmt | var_decl
//!   manifold_decl → "manifold" IDENT "=" expr
//!   regress_stmt → "regress" config_block
//!   config_block → "{" (IDENT ":" expr ",")* "}"
// ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

extern crate alloc;
use crate::ast::*;
use crate::lexer::{Lexer, Token, TokenKind};
use alloc::string::String;
use alloc::vec::Vec;
use alloc::boxed::Box;

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
             // Skip empty lines (though Lexer currently emits Newline tokens, we might consume them)
             // Actually, grammar says program -> statement*.
             // Our parse_statement handles newline/empty specially.
            
             // Consume leading newlines strictly
             while self.check(TokenKind::Newline) {
                 self.advance();
             }

            if self.is_at_end() {
                break;
            }

            let stmt = self.parse_statement()?;
            // We only push non-empty statements
            if !matches!(stmt.node, StmtKind::Empty) {
                program.push(stmt);
            }
        }

        Ok(program)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════════════

    fn make_span(&self, start: &Token, end: &Token) -> Span {
        Span {
            start: start.start,
            end: end.end,
            line: start.line,
            col: start.column,
        }
    }

    fn wrap_stmt(&self, kind: StmtKind, start_token: &Token) -> Statement {
        let end_token = self.previous();
        let span = self.make_span(start_token, end_token);
        Statement { node: kind, span }
    }
    
    fn wrap_expr(&self, kind: ExprKind, start_token: &Token) -> Expr {
        let end_token = self.previous();
        let span = self.make_span(start_token, end_token);
        Expr { node: kind, span }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Statement Parsing
    // ═══════════════════════════════════════════════════════════════════════════

    fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        let token = self.peek().clone();
        
        let kind = match &token.kind {
            TokenKind::Manifold => self.parse_manifold_decl()?,
            TokenKind::Block => self.parse_block_decl()?,
            TokenKind::Regress => self.parse_regress_stmt()?,
            TokenKind::Render => self.parse_render_stmt()?,
            TokenKind::Identifier(_) => self.parse_ident_start_stmt()?,

            // Class Declaration
            TokenKind::Class => self.parse_class_decl()?,

            // Modules
            TokenKind::Import => self.parse_import_stmt()?,
            TokenKind::From => self.parse_from_import_stmt()?,

            // Control Flow
            TokenKind::If => self.parse_if_stmt()?,
            TokenKind::While => self.parse_while_stmt()?,
            TokenKind::For => self.parse_for_stmt()?,
            TokenKind::Seal => self.parse_seal_stmt()?,
            TokenKind::Fn => self.parse_fn_decl()?,
            TokenKind::Return => self.parse_return_stmt()?,
            TokenKind::Break => {
                self.advance();
                StmtKind::Break(BreakStmt)
            },
            TokenKind::Continue => {
                self.advance();
                StmtKind::Continue(ContinueStmt)
            },
            TokenKind::Let => self.parse_let_decl()?,

            TokenKind::Newline | TokenKind::Eof => StmtKind::Empty,
            _ => {
                return Err(ParseError::new(
                    &format!("unexpected token: {:?}", token.kind),
                    token.line,
                    token.column,
                ));
            }
        };
        
        // Special case: if Empty, just return a dummy empty statement with current token span
        if matches!(kind, StmtKind::Empty) {
            let _t = self.previous(); // Might be Newline we just consumed or previous
            // Doing it properly:
            return Ok(Statement { 
                node: StmtKind::Empty, 
                span: self.make_span(&token, &token) 
            });
        }

        // For statements that we parsed, we want them wrapped. 
        // Note: parse_manifold_decl etc currently return StmtKind, need to adapt helper methods.
        // Actually, let's make specific parsers return StmtKind and wrap here?
        // Wait, parse_ident_start_stmt consumes tokens inside.
        // Better to have parse functions return StmtKind.
        
        Ok(self.wrap_stmt(kind, &token))
    }

    /// manifold_decl → "manifold" IDENT "=" expr
    fn parse_manifold_decl(&mut self) -> Result<StmtKind, ParseError> {
        self.expect(TokenKind::Manifold)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::Equals)?;
        let init = self.parse_expr()?;

        Ok(StmtKind::Manifold(ManifoldDecl { name, init }))
    }

    /// block_decl → "block" IDENT "=" expr
    fn parse_block_decl(&mut self) -> Result<StmtKind, ParseError> {
        self.expect(TokenKind::Block)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::Equals)?;
        let source = self.parse_expr()?;

        Ok(StmtKind::Block(BlockDecl { name, source }))
    }

    /// ident_start_stmt → var_decl | expr_stmt
    fn parse_ident_start_stmt(&mut self) -> Result<StmtKind, ParseError> {
        let first_ident = self.expect_ident()?;

        // 1. Check for type hint: Ident Ident = Expr
        if self.check_ident() && self.peek_next_is(TokenKind::Equals) {
            let type_hint = Some(first_ident);
            let name = self.expect_ident()?;
            self.expect(TokenKind::Equals)?;
            let value = self.parse_expr()?;

            Ok(StmtKind::Var(VarDecl {
                type_hint,
                name,
                value,
            }))
        } 
        // 2. Check for Var Decl without type: Ident = Expr
        else if self.check(TokenKind::Equals) {
            self.expect(TokenKind::Equals)?;
            let value = self.parse_expr()?;

            Ok(StmtKind::Var(VarDecl {
                type_hint: None,
                name: first_ident,
                value,
            }))
        }
        // 3. Expression Statement (e.g., method call) starting with Ident
        else {
            // We consumed the identifier. Parse the rest as an expression starting with this ident.
            let start_token = self.tokens[self.current-1].clone();
            let kind = self.parse_ident_expr_cont(first_ident, &start_token)?;
            Ok(StmtKind::Expr(self.wrap_expr(kind, &start_token)))
        }
    }

    /// let_decl → "let" IDENT "=" expr
    fn parse_let_decl(&mut self) -> Result<StmtKind, ParseError> {
        self.expect(TokenKind::Let)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::Equals)?;
        let value = self.parse_expr()?;

        Ok(StmtKind::Var(VarDecl {
            type_hint: None,
            name,
            value,
        }))
    }

    /// regress_stmt → "regress" config_block
    fn parse_regress_stmt(&mut self) -> Result<StmtKind, ParseError> {
        self.expect(TokenKind::Regress)?;
        let config = self.parse_regress_config()?;

        Ok(StmtKind::Regress(RegressStmt { config }))
    }

    /// render_stmt → "render" IDENT config_block?
    fn parse_render_stmt(&mut self) -> Result<StmtKind, ParseError> {
        self.expect(TokenKind::Render)?;
        let target = self.expect_ident()?;

        let config = if self.check(TokenKind::LBrace) {
            self.parse_render_config()?
        } else {
            RenderConfig::default()
        };

        Ok(StmtKind::Render(RenderStmt { target, config }))
    }

    /// class_decl → "class" IDENT "{" (var_decl | fn_decl)* "}"
    fn parse_class_decl(&mut self) -> Result<StmtKind, ParseError> {
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
                if let StmtKind::Fn(f) = self.parse_fn_decl()? {
                    methods.push(f);
                }
            } else if self.check_ident() {
                // Field
                let field_name = self.expect_ident()?;
                let value = if self.check(TokenKind::Equals) {
                    self.advance();
                    self.parse_expr()?
                } else {
                    // Default to false wrapped 
                    let t = self.peek().clone(); // span might be slightly off
                    self.wrap_expr(ExprKind::Literal(Literal::Bool(false)), &t)
                };

                if self.check(TokenKind::Comma) {
                    self.advance();
                }

                fields.push(VarDecl {
                    type_hint: None,
                    name: field_name,
                    value,
                });
            } else {
                 let t = self.peek();
                 return Err(ParseError::new("expected field or method", t.line, t.column));
            }
        }

        self.expect(TokenKind::RBrace)?;

        Ok(StmtKind::Class(ClassDecl {
            name,
            fields,
            methods,
        }))
    }
    
    // Modules
    fn parse_import_stmt(&mut self) -> Result<StmtKind, ParseError> {
        self.expect(TokenKind::Import)?;
        let module = self.expect_ident()?;
         Ok(StmtKind::Import(ImportStmt {
            module,
            symbol: None,
        }))
    }

    fn parse_from_import_stmt(&mut self) -> Result<StmtKind, ParseError> {
        self.expect(TokenKind::From)?;
        let module = self.expect_ident()?;
        self.expect(TokenKind::Import)?;
        let symbol = self.expect_ident()?;

        Ok(StmtKind::Import(ImportStmt {
            module,
            symbol: Some(symbol),
        }))
    }
    
    // Control Flow
    fn parse_if_stmt(&mut self) -> Result<StmtKind, ParseError> {
        self.expect(TokenKind::If)?;
        let condition = self.parse_expr()?;
        let then_branch = self.parse_block_stmts()?;
        
        let else_branch = if self.check(TokenKind::Else) {
            self.advance();
            Some(self.parse_block_stmts()?)
        } else {
            None
        };
        
        Ok(StmtKind::If(IfStmt { condition, then_branch, else_branch }))
    }
    
    fn parse_while_stmt(&mut self) -> Result<StmtKind, ParseError> {
        self.expect(TokenKind::While)?;
        let condition = self.parse_expr()?;
        let body = self.parse_block_stmts()?;
        Ok(StmtKind::While(WhileStmt { condition, body }))
    }
    
    fn parse_for_stmt(&mut self) -> Result<StmtKind, ParseError> {
        self.expect(TokenKind::For)?;
        let iterator = self.expect_ident()?;
        self.expect(TokenKind::In)?;
        
        // Currently expecting Range. 
        // We parse expr, verify range.
        let expr = self.parse_expr()?;
        let range = match expr.node {
            ExprKind::Range(r) => r,
            _ => {
                 return Err(ParseError::new("expected range in for loop", expr.span.line, expr.span.col));
            }
        };
        
        let body = self.parse_block_stmts()?;
        Ok(StmtKind::For(ForStmt { iterator, range, body }))
    }
    
    fn parse_seal_stmt(&mut self) -> Result<StmtKind, ParseError> {
        self.expect(TokenKind::Seal)?;
        let body = self.parse_block_stmts()?;
        Ok(StmtKind::Loop(LoopStmt { body }))
    }
    
    fn parse_fn_decl(&mut self) -> Result<StmtKind, ParseError> {
         self.expect(TokenKind::Fn)?;
         let name = self.expect_ident()?;
         self.expect(TokenKind::LParen)?;
         
         let mut params = Vec::new();
         while !self.check(TokenKind::RParen) && !self.is_at_end() {
             params.push(self.expect_ident()?);
             if self.check(TokenKind::Comma) { self.advance(); }
         }
         self.expect(TokenKind::RParen)?;
         
         let body = self.parse_block_stmts()?;
         Ok(StmtKind::Fn(FnDecl { name, params, body }))
    }
    
    fn parse_return_stmt(&mut self) -> Result<StmtKind, ParseError> {
        self.expect(TokenKind::Return)?;
        let value = if self.check(TokenKind::Newline) || self.check(TokenKind::RBrace) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        Ok(StmtKind::Return(ReturnStmt { value }))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Expression Parsing
    // ═══════════════════════════════════════════════════════════════════════════

    // Start with lowest precedence
    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_range()
    }
    
    fn parse_range(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_arithmetic()?; // Using arithmetic as base for range
        
        if self.check(TokenKind::Colon) {
             self.advance();
             // range start/end must be numbers, but parse_arithmetic returns Spanned<ExprKind>
             // We need to extract number values if possible, or return Error
             let right = self.parse_arithmetic()?;
             
             let start_val = self.expr_to_number(&left)?;
             let end_val = self.expr_to_number(&right)?;
             
             let kind = ExprKind::Range(Range { start: start_val, end: end_val });
             
             let span = Span {
                 start: left.span.start,
                 end: right.span.end,
                 line: left.span.line,
                 col: left.span.col,
             };
             return Ok(Expr { node: kind, span });
        }
        
        Ok(left)
    }

    fn parse_arithmetic(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_term()?;

        while self.check(TokenKind::Plus) || self.check(TokenKind::Minus) {
            let op_token = self.advance();
            let op = match op_token.kind {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                _ => unreachable!(),
            };
            let right = self.parse_term()?;
            
             let span = Span {
                 start: left.span.start,
                 end: right.span.end,
                 line: left.span.line,
                 col: left.span.col,
             };
             
             let kind = ExprKind::BinaryOp(Box::new(left.clone()), op, Box::new(right));
             left = Expr { node: kind, span };
        }
        
        Ok(left)
    }

    fn parse_term(&mut self) -> Result<Expr, ParseError> {
         let mut left = self.parse_primary()?;
         
         while self.check(TokenKind::Star) || self.check(TokenKind::Slash) {
            let op_token = self.advance();
            let op = match op_token.kind {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                _ => unreachable!(),
            };
            let right = self.parse_primary()?;
            
             let span = Span {
                 start: left.span.start,
                 end: right.span.end,
                 line: left.span.line,
                 col: left.span.col,
             };
             
             let kind = ExprKind::BinaryOp(Box::new(left.clone()), op, Box::new(right));
             left = Expr { node: kind, span };
         }
         
         Ok(left)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let token = self.advance();
        let kind = match token.kind {
            TokenKind::Number(n) => ExprKind::Literal(Literal::Num(n as f64)),
            TokenKind::Float(int, frac) => {
                 let val = int as f64 + (frac as f64 / 1_000_000.0);
                 ExprKind::Literal(Literal::Num(val))
            },
            TokenKind::True => ExprKind::Literal(Literal::Bool(true)),
            TokenKind::False => ExprKind::Literal(Literal::Bool(false)),
            TokenKind::StringLit(ref s) => ExprKind::Literal(Literal::Str(s.clone())),
            TokenKind::Self_ => ExprKind::Ident(String::from("self")),
            
            TokenKind::Identifier(ref name) => {
                let name_clone = name.clone();
                // Return result of parse_ident_expr_cont which returns ExprKind
                return self.parse_ident_expr_cont(name_clone, &token).map(|kind| {
                    // Span adjustment needed because parse_ident_expr_cont consumes more
                    let end_token = self.previous();
                    let span = self.make_span(&token, end_token);
                    Expr { node: kind, span }
                });
            },

            // New Object Instantiation
            TokenKind::New => {
                let class = self.expect_ident()?;
                self.expect(TokenKind::LParen)?;
                 let mut args = Vec::new();
                while !self.check(TokenKind::RParen) && !self.is_at_end() {
                    args.push(self.parse_expr()?);
                    if self.check(TokenKind::Comma) { self.advance(); }
                }
                self.expect(TokenKind::RParen)?;
                ExprKind::New { class, args }
            },
            
            // List
            TokenKind::LBracket => self.parse_list_literal_cont()?,

            // Embed/Convergence keywords used as functions
            TokenKind::Embed => {
                 return self.parse_call_expr_cont(String::from("embed"), &token);
            },
            TokenKind::Convergence => {
                 return self.parse_call_expr_cont(String::from("convergence"), &token);
            },
            
            _ => return Err(ParseError::new("expected expression", token.line, token.column)),
        };
        
        Ok(self.wrap_expr(kind, &token))
    }
    
    fn parse_list_literal_cont(&mut self) -> Result<ExprKind, ParseError> {
          let mut elements = Vec::new();

        while !self.check(TokenKind::RBracket) && !self.is_at_end() {
             if self.check(TokenKind::Newline) { self.advance(); continue; }
             elements.push(self.parse_expr()?);
             if self.check(TokenKind::Comma) { self.advance(); }
        }
        self.expect(TokenKind::RBracket)?;
        Ok(ExprKind::List(elements))
    }

    fn parse_ident_expr_cont(&mut self, name: String, _start_token: &Token) -> Result<ExprKind, ParseError> {
        // Method call: M.cluster(...)
        if self.check(TokenKind::Dot) {
            self.advance();
            let method = self.expect_flexible_ident()?;

            if self.check(TokenKind::LParen) {
                let args = self.parse_call_args()?;
                return Ok(ExprKind::MethodCall {
                    object: name,
                    method,
                    args,
                });
            } else {
                 return Ok(ExprKind::FieldAccess {
                    object: name,
                    field: method,
                });
            }
        }

        // Call: embed(...)
        if self.check(TokenKind::LParen) {
             let args = self.parse_call_args()?;
             return Ok(ExprKind::Call { name, args });
        }

        // Index: M[0:64]
        if self.check(TokenKind::LBracket) {
            self.advance();
            let start = self.parse_number()?;
            self.expect(TokenKind::Colon)?;
            let end = self.parse_number()?;
            self.expect(TokenKind::RBracket)?;

            return Ok(ExprKind::Index {
                object: name,
                range: Range { start, end },
            });
        }

        Ok(ExprKind::Ident(name))
    }
    
    fn parse_call_expr_cont(&mut self, name: String, start_token: &Token) -> Result<Expr, ParseError> {
         let args = self.parse_call_args()?;
         let kind = ExprKind::Call { name, args };
         Ok(self.wrap_expr(kind, start_token))
    }

    fn parse_call_args(&mut self) -> Result<Vec<CallArg>, ParseError> {
        self.expect(TokenKind::LParen)?;

        let mut args = Vec::new();

        while !self.check(TokenKind::RParen) && !self.is_at_end() {
             // Check for named argument
             if self.check_flexible_ident() {
                  let saved_pos = self.current;
                  let name = self.expect_flexible_ident()?;
                  
                  if self.check(TokenKind::Equals) {
                      self.advance();
                      let value = self.parse_expr()?;
                      args.push(CallArg::Named { name, value });
                  } else {
                      // Backtrack
                      self.current = saved_pos;
                      let expr = self.parse_expr()?;
                      args.push(CallArg::Positional(expr));
                  }
             } else {
                 let expr = self.parse_expr()?;
                 args.push(CallArg::Positional(expr));
             }
             
             if self.check(TokenKind::Comma) { self.advance(); }
        }
        self.expect(TokenKind::RParen)?;
        Ok(args)
    }

    // Helper to extract Number from Expr (for Range and Config compatibility)
    fn expr_to_number(&self, expr: &Expr) -> Result<Number, ParseError> {
        match &expr.node {
            ExprKind::Literal(Literal::Num(f)) => {
                 // Convert f64 back to Number enum just for internal usage in Range?
                 // Wait, Range struct in ast.rs expects Number enum.
                 // So I must construct Number.
                 // f64 to Number::Float
                 let int_part = *f as i64;
                 let frac_part = ((*f - int_part as f64) * 1_000_000.0) as i64;
                 Ok(Number::Float { int_part, frac_part })
            },
            _ => Err(ParseError::new("expected number", expr.span.line, expr.span.col))
        }
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
    
    // Config Block Parsers (Simplified for brevity but maintaining logic)
    fn parse_regress_config(&mut self) -> Result<RegressConfig, ParseError> {
         self.expect(TokenKind::LBrace)?;
         let mut config = RegressConfig::default();
         while !self.check(TokenKind::RBrace) && !self.is_at_end() {
             if self.check(TokenKind::Newline) { self.advance(); continue; }
             
             let key = self.expect_flexible_ident()?;
             self.expect(TokenKind::Colon)?;
             
             match key.as_str() {
                 "model" => {
                     if let ExprKind::Literal(Literal::Str(s)) = self.parse_expr()?.node {
                         config.model = s;
                     }
                 },
                 "degree" => {
                      let expr = self.parse_expr()?;
                      if let Ok(num) = self.expr_to_number(&expr) {
                          if let Number::Int(n) = num { config.degree = Some(n as u8); }
                      }
                 },
                 "target" => config.target = Some(self.parse_expr()?),
                 "escalate" => {
                      if let ExprKind::Literal(Literal::Bool(b)) = self.parse_expr()?.node {
                          config.escalate = b;
                      }
                 },
                 "until" => {
                     // Parse convergence
                     // convergence(..) or custom expr
                     // Look at parse_expr() handling
                     let expr = self.parse_expr()?;
                     // Check if it is a call 'convergence'
                     // Actually convergence is special keyword in lexer but parsed as call
                     config.until = Some(ConvergenceCond::Custom(expr)); // Simplified
                 },
                 _ => { self.parse_expr()?; }
             }
             
             if self.check(TokenKind::Comma) { self.advance(); }
         }
         self.expect(TokenKind::RBrace)?;
         Ok(config)
    }

    fn parse_render_config(&mut self) -> Result<RenderConfig, ParseError> {
        self.expect(TokenKind::LBrace)?;
        let mut config = RenderConfig::default();
         while !self.check(TokenKind::RBrace) && !self.is_at_end() {
             if self.check(TokenKind::Newline) { self.advance(); continue; }
             
             let key = self.expect_flexible_ident()?;
             self.expect(TokenKind::Colon)?;
             let expr = self.parse_expr()?;
             
             match key.as_str() {
                 "color" => {
                     if let ExprKind::Ident(id) = expr.node { config.color = Some(id); }
                 },
                 "highlight" => {
                      if let ExprKind::Ident(id) = expr.node { config.highlight = Some(id); }
                 },
                 "trajectory" => {
                      if let ExprKind::Literal(Literal::Bool(b)) = expr.node { config.trajectory = b; }
                 },
                 "axis" => {
                      if let Ok(Number::Int(n)) = self.expr_to_number(&expr) { config.axis = Some(n as u8); }
                 },
                 _ => {}
             }
             
             if self.check(TokenKind::Comma) { self.advance(); }
         }
         self.expect(TokenKind::RBrace)?;
         Ok(config)
    }

    // Helper Methods
    fn parse_block_stmts(&mut self) -> Result<Block, ParseError> {
        self.expect(TokenKind::LBrace)?;
        let mut statements = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            while self.check(TokenKind::Newline) { self.advance(); }
            if self.check(TokenKind::RBrace) { break; }
            let stmt = self.parse_statement()?;
             if !matches!(stmt.node, StmtKind::Empty) {
                statements.push(stmt);
            }
        }
        self.expect(TokenKind::RBrace)?;
        Ok(Block { statements })
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.current).unwrap_or(&self.tokens[self.tokens.len()-1])
    }
    
    fn previous(&self) -> &Token {
        if self.current == 0 { return &self.tokens[0]; }
        &self.tokens[self.current - 1]
    }

    fn advance(&mut self) -> Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous().clone()
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
            _ => Err(ParseError::new("expected identifier", token.line, token.column)),
        }
    }
    
    fn check_flexible_ident(&self) -> bool {
         matches!(self.peek().kind, TokenKind::Identifier(_) | TokenKind::Dim | TokenKind::Tau | TokenKind::Model | TokenKind::Color | TokenKind::Axis | TokenKind::Project | TokenKind::Cluster | TokenKind::Center | TokenKind::Spread | TokenKind::Format | TokenKind::Output | TokenKind::Escalate | TokenKind::Convergence)
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
             _ => Err(ParseError::new("expected argument name", token.line, token.column))
         }
    }
}
