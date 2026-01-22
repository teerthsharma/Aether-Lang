// ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Lexer - Tokenizer for .aegis scripts
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Converts source text into a stream of tokens for the parser.
//!
//! Token Categories:
//! - Keywords: manifold, block, regress, render, until, escalate, embed
//! - Control flow: seal (🦭), for, while, if, else, fn, return, break, continue
//! - Operators: =, :, {, }, [, ], (, ), , +, -, *, /, <, >, ==, !=, &&, ||
//! - Literals: numbers, strings, identifiers
//! - Comments: // single-line
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;
use core::str::Chars;
use core::iter::Peekable;

/// Token kinds in the AEGIS language
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords - 3D manifold primitives
    Manifold,       // manifold
    Block,          // block
    Regress,        // regress
    Render,         // render
    Embed,          // embed
    Until,          // until
    Escalate,       // escalate
    Convergence,    // convergence
    True,           // true
    False,          // false
    
    // Classes & Objects - NEW
    Class,          // class
    New,            // new
    Self_,          // self
    
    // Modules - NEW
    Import,         // import
    From,           // from
    As,             // as
    
    // Control flow keywords - NEW for full language
    Seal,           // seal or 🦭
    For,            // for
    While,          // while
    If,             // if
    Else,           // else
    Fn,             // fn
    Return,         // return
    Break,          // break
    Continue,       // continue
    In,             // in
    Let,            // let
    
    // 3D-specific keywords
    Dim,            // dim
    Tau,            // tau
    Model,          // model
    Color,          // color
    Axis,           // axis
    Project,        // project
    Cluster,        // cluster
    Center,         // center
    Spread,         // spread
    Format,         // format
    Output,         // output
    
    // Literals
    Identifier(String),
    Number(i64),
    Float(i64, i64),  // (integer_part, fractional_part * 1000000)
    StringLit(String),
    
    // Operators & Punctuation
    Equals,         // =
    Colon,          // :
    Comma,          // ,
    Dot,            // .
    LBrace,         // {
    RBrace,         // }
    LBracket,       // [
    RBracket,       // ]
    LParen,         // (
    RParen,         // )
    
    // Arithmetic operators - NEW
    Plus,           // +
    Minus,          // -
    Star,           // *
    Slash,          // /
    Percent,        // %
    
    // Comparison operators - NEW
    Less,           // <
    Greater,        // >
    LessEq,         // <=
    GreaterEq,      // >=
    EqEq,           // ==
    NotEq,          // !=
    
    // Logical operators - NEW
    And,            // &&
    Or,             // ||
    Not,            // !
    
    // Range operator - NEW
    DotDot,         // ..
    Tilde,          // ~
    
    // End markers
    Newline,
    Eof,
    
    // Error
    Error(String),
}

/// A token with its position in source
#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub line: usize,
    pub column: usize,
}

impl Token {
    pub fn new(kind: TokenKind, line: usize, column: usize) -> Self {
        Self { kind, line, column }
    }
}

/// Lexer for AEGIS scripts
pub struct Lexer<'a> {
    source: &'a str,
    chars: Peekable<Chars<'a>>,
    current_pos: usize,
    line: usize,
    column: usize,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer for the given source
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            chars: source.chars().peekable(),
            current_pos: 0,
            line: 1,
            column: 1,
        }
    }
    
    /// Advance to next character
    fn advance(&mut self) -> Option<char> {
        let c = self.chars.next()?;
        self.current_pos += c.len_utf8();
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(c)
    }
    
    /// Peek at current character without consuming
    fn peek(&mut self) -> Option<&char> {
        self.chars.peek()
    }
    
    /// Skip whitespace (except newlines which are tokens)
    fn skip_whitespace(&mut self) {
        while let Some(&c) = self.peek() {
            if c == ' ' || c == '\t' || c == '\r' {
                self.advance();
            } else {
                break;
            }
        }
    }
    
    /// Skip single-line comment
    fn skip_comment(&mut self) {
        while let Some(&c) = self.peek() {
            if c == '\n' {
                break;
            }
            self.advance();
        }
    }
    
    /// Read an identifier or keyword
    fn read_identifier(&mut self, first: char) -> TokenKind {
        let mut name = String::new();
        name.push(first);
        
        while let Some(&c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                name.push(c);
                self.advance();
            } else {
                break;
            }
        }
        
        // Check for keywords
        match name.as_str() {
            "manifold" => TokenKind::Manifold,
            "block" => TokenKind::Block,
            "regress" => TokenKind::Regress,
            "render" => TokenKind::Render,
            "embed" => TokenKind::Embed,
            "until" => TokenKind::Until,
            "escalate" => TokenKind::Escalate,
            "convergence" => TokenKind::Convergence,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            // Classes
            "class" => TokenKind::Class,
            "new" => TokenKind::New,
            "self" => TokenKind::Self_,
            // Modules
            "import" => TokenKind::Import,
            "from" => TokenKind::From,
            "as" => TokenKind::As,
            // Control flow - NEW
            "seal" => TokenKind::Seal,
            "for" => TokenKind::For,
            "while" => TokenKind::While,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "fn" => TokenKind::Fn,
            "return" => TokenKind::Return,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            "in" => TokenKind::In,
            "let" => TokenKind::Let,
            // 3D keywords
            "dim" => TokenKind::Dim,
            "tau" => TokenKind::Tau,
            "model" => TokenKind::Model,
            "color" => TokenKind::Color,
            "axis" => TokenKind::Axis,
            "project" => TokenKind::Project,
            "cluster" => TokenKind::Cluster,
            "center" => TokenKind::Center,
            "spread" => TokenKind::Spread,
            "format" => TokenKind::Format,
            "output" => TokenKind::Output,
            _ => TokenKind::Identifier(name),
        }
    }
    
    /// Read a number (integer or float)
    fn read_number(&mut self, first: char) -> TokenKind {
        let mut int_part: i64 = (first as i64) - ('0' as i64);
        
        // Read integer part
        while let Some(&c) = self.peek() {
            if c.is_ascii_digit() {
                int_part = int_part * 10 + (c as i64 - '0' as i64);
                self.advance();
            } else {
                break;
            }
        }
        
        // Check for decimal point
        if let Some(&'.') = self.peek() {
            self.advance();
            let mut frac_part: i64 = 0;
            let mut frac_digits = 0;
            
            while let Some(&c) = self.peek() {
                if c.is_ascii_digit() && frac_digits < 6 {
                    frac_part = frac_part * 10 + (c as i64 - '0' as i64);
                    frac_digits += 1;
                    self.advance();
                } else {
                    break;
                }
            }
            
            // Normalize to 6 decimal places
            while frac_digits < 6 {
                frac_part *= 10;
                frac_digits += 1;
            }
            
            TokenKind::Float(int_part, frac_part)
        } else {
            TokenKind::Number(int_part)
        }
    }
    
    /// Read a string literal
    fn read_string(&mut self) -> TokenKind {
        let mut s = String::new();
        
        while let Some(&c) = self.peek() {
            if c == '"' {
                self.advance();
                return TokenKind::StringLit(s);
            } else if c == '\n' {
                let mut err = String::new();
                err.push_str("unterminated string");
                return TokenKind::Error(err);
            } else {
                s.push(c);
                self.advance();
            }
        }
        
        let mut err = String::new();
        err.push_str("unexpected EOF in string");
        TokenKind::Error(err)
    }
    
    /// Get next token
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();
        
        let line = self.line;
        let column = self.column;
        
        let c = match self.advance() {
            Some(c) => c,
            None => return Token::new(TokenKind::Eof, line, column),
        };
        
        let kind = match c {
            // Comments and division
            '/' if self.peek() == Some(&'/') => {
                self.skip_comment();
                return self.next_token();
            }
            '/' => TokenKind::Slash,
            
            // Seal emoji 🦭
            '\u{1F9AD}' => TokenKind::Seal,

            // Tilde ~
            '~' => TokenKind::Tilde,
            
            // Two-char operators
            '=' if self.peek() == Some(&'=') => { self.advance(); TokenKind::EqEq }
            '!' if self.peek() == Some(&'=') => { self.advance(); TokenKind::NotEq }
            '!' => TokenKind::Not,
            '<' if self.peek() == Some(&'=') => { self.advance(); TokenKind::LessEq }
            '<' => TokenKind::Less,
            '>' if self.peek() == Some(&'=') => { self.advance(); TokenKind::GreaterEq }
            '>' => TokenKind::Greater,
            '&' if self.peek() == Some(&'&') => { self.advance(); TokenKind::And }
            '|' if self.peek() == Some(&'|') => { self.advance(); TokenKind::Or }
            '.' if self.peek() == Some(&'.') => { self.advance(); TokenKind::DotDot }
            
            // Single-char tokens
            '=' => TokenKind::Equals,
            ':' => TokenKind::Colon,
            ',' => TokenKind::Comma,
            '.' => TokenKind::Dot,
            '{' => TokenKind::LBrace,
            '}' => TokenKind::RBrace,
            '[' => TokenKind::LBracket,
            ']' => TokenKind::RBracket,
            '(' => TokenKind::LParen,
            ')' => TokenKind::RParen,
            '\n' => TokenKind::Newline,
            
            // Arithmetic operators - NEW
            '+' => TokenKind::Plus,
            '-' => TokenKind::Minus,
            '*' => TokenKind::Star,
            '%' => TokenKind::Percent,
            
            // String literals
            '"' => self.read_string(),
            
            // Numbers
            c if c.is_ascii_digit() => self.read_number(c),
            
            // Identifiers and keywords
            c if c.is_alphabetic() || c == '_' => self.read_identifier(c),
            
            // Unknown
            _ => {
                let mut err = String::new();
                err.push_str("unexpected char: ");
                err.push(c);
                TokenKind::Error(err)
            }
        };
        
        Token::new(kind, line, column)
    }
    
    /// Tokenize entire source into a vector
    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        
        loop {
            let token = self.next_token();
            let is_eof = matches!(token.kind, TokenKind::Eof);
            tokens.push(token);
            
            if is_eof {
                break;
            }
        }
        
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lex_manifold() {
        let mut lexer = Lexer::new("manifold M = embed(data, dim=3)");
        let tokens = lexer.tokenize();
        
        assert!(matches!(tokens[0].kind, TokenKind::Manifold));
        assert!(matches!(tokens[1].kind, TokenKind::Identifier(_)));
        assert!(matches!(tokens[2].kind, TokenKind::Equals));
        assert!(matches!(tokens[3].kind, TokenKind::Embed));
    }
    
    #[test]
    fn test_lex_float() {
        let mut lexer = Lexer::new("1.5");
        let token = lexer.next_token();
        
        assert!(matches!(token.kind, TokenKind::Float(1, 500000)));
    }

    #[test]
    fn test_lex_tilde() {
        let mut lexer = Lexer::new("~");
        let token = lexer.next_token();
        assert!(matches!(token.kind, TokenKind::Tilde));
    }
}
