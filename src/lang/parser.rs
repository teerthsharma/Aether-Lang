//! ═══════════════════════════════════════════════════════════════════════════════
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

use crate::lang::lexer::{Token, TokenKind, Lexer};
use crate::lang::ast::*;
use heapless::{String, Vec as HVec};

/// Parser error
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String<128>,
    pub line: usize,
    pub column: usize,
}

impl ParseError {
    pub fn new(msg: &str, line: usize, column: usize) -> Self {
        let mut message = String::new();
        let _ = message.push_str(msg);
        Self { message, line, column }
    }
}

/// AEGIS Parser
pub struct Parser<'a> {
    tokens: HVec<Token, 256>,
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
                let _ = program.push(stmt);
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
            TokenKind::Identifier(_) => self.parse_var_decl(),
            TokenKind::Newline | TokenKind::Eof => Ok(Statement::Empty),
            _ => {
                let line = token.line;
                let col = token.column;
                Err(ParseError::new("unexpected token", line, col))
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
    
    /// var_decl → type_hint? IDENT "=" expr
    fn parse_var_decl(&mut self) -> Result<Statement, ParseError> {
        let first_ident = self.expect_ident()?;
        
        // Check for type hint (e.g., "centroid C = ...")
        if self.check_ident() && self.peek_next_is(TokenKind::Equals) {
            let type_hint = Some(first_ident);
            let name = self.expect_ident()?;
            self.expect(TokenKind::Equals)?;
            let value = self.parse_expr()?;
            
            Ok(Statement::Var(VarDecl { type_hint, name, value }))
        } else {
            self.expect(TokenKind::Equals)?;
            let value = self.parse_expr()?;
            
            Ok(Statement::Var(VarDecl { type_hint: None, name: first_ident, value }))
        }
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
            
            let key = self.expect_ident()?;
            self.expect(TokenKind::Colon)?;
            
            match key.as_str() {
                "model" => {
                    if let Expr::Str(s) = self.parse_expr()? {
                        config.model = {
                            let mut m = String::<32>::new();
                            let _ = m.push_str(s.as_str());
                            m
                        };
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
            
            let key = self.expect_ident()?;
            self.expect(TokenKind::Colon)?;
            
            match key.as_str() {
                "color" => {
                    let expr = self.parse_expr()?;
                    if let Expr::Ident(id) = expr {
                        config.color = Some({
                            let mut s = String::<32>::new();
                            let _ = s.push_str(id.as_str());
                            s
                        });
                    }
                }
                "highlight" => {
                    if let Expr::Ident(id) = self.parse_expr()? {
                        config.highlight = Some(id);
                    }
                }
                "trajectory" => {
                    if let Expr::Bool(b) = self.parse_expr()? {
                        config.trajectory = b;
                    } else if let Expr::Ident(id) = self.parse_expr()? {
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
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Expression Parsing
    // ═══════════════════════════════════════════════════════════════════════════
    
    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_primary()
    }
    
    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let token = self.advance();
        
        match token.kind {
            TokenKind::Number(n) => Ok(Expr::Num(Number::Int(n))),
            TokenKind::Float(int, frac) => Ok(Expr::Num(Number::Float { 
                int_part: int, 
                frac_part: frac 
            })),
            TokenKind::True => Ok(Expr::Bool(true)),
            TokenKind::False => Ok(Expr::Bool(false)),
            TokenKind::StringLit(s) => Ok(Expr::Str(s)),
            
            TokenKind::Identifier(name) => self.parse_ident_expr(name),
            
            // Function keywords that act as function names
            TokenKind::Embed => self.parse_call_expr({
                let mut s = String::<64>::new();
                let _ = s.push_str("embed");
                s
            }),
            TokenKind::Convergence => self.parse_call_expr({
                let mut s = String::<64>::new();
                let _ = s.push_str("convergence");
                s
            }),
            
            _ => {
                Err(ParseError::new("expected expression", token.line, token.column))
            }
        }
    }
    
    fn parse_ident_expr(&mut self, name: String<64>) -> Result<Expr, ParseError> {
        // Check for method call: M.cluster(...)
        if self.check(TokenKind::Dot) {
            self.advance();
            let method = self.expect_ident()?;
            
            if self.check(TokenKind::LParen) {
                // Method call
                let args = self.parse_call_args()?;
                return Ok(Expr::MethodCall { object: name, method, args });
            } else {
                // Field access
                return Ok(Expr::FieldAccess { object: name, field: method });
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
                range: Range { start, end } 
            });
        }
        
        Ok(Expr::Ident(name))
    }
    
    fn parse_call_expr(&mut self, name: String<64>) -> Result<Expr, ParseError> {
        let args = self.parse_call_args()?;
        Ok(Expr::Call { name, args })
    }
    
    fn parse_call_args(&mut self) -> Result<HVec<CallArg, MAX_ARGS>, ParseError> {
        self.expect(TokenKind::LParen)?;
        
        let mut args = HVec::new();
        
        while !self.check(TokenKind::RParen) && !self.is_at_end() {
            // Check for named argument: dim=3
            if self.check_ident() {
                let saved_pos = self.current;
                let name = self.expect_ident()?;
                
                if self.check(TokenKind::Equals) {
                    self.advance();
                    let value = self.parse_expr()?;
                    let _ = args.push(CallArg::Named { name, value });
                } else {
                    // Rewind - it's a positional arg that's an identifier
                    self.current = saved_pos;
                    let expr = self.parse_expr()?;
                    let _ = args.push(CallArg::Positional(expr));
                }
            } else {
                let expr = self.parse_expr()?;
                let _ = args.push(CallArg::Positional(expr));
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
            TokenKind::Float(int, frac) => Ok(Number::Float { int_part: int, frac_part: frac }),
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
        self.tokens.get(self.current + 1)
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
    
    fn expect_ident(&mut self) -> Result<String<64>, ParseError> {
        let token = self.advance();
        match token.kind {
            TokenKind::Identifier(s) => Ok(s),
            _ => Err(ParseError::new("expected identifier", token.line, token.column)),
        }
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
