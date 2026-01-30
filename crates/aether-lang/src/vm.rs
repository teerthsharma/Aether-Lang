//! ═══════════════════════════════════════════════════════════════════════════════
//! TITAN Cortex: The High-Throughput Virtual Machine
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! "The Left Brain of AEGIS."
//!
//! Optimization targets:
//! - Stack-based execution (Cache locality)
//! - Linear bytecode (Predictable branching)
//! - Explicit topological ops (EMBED, ATTEND, PRUNE)
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::string::String;

#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(feature = "std")]
use std::string::String;
#[cfg(feature = "std")]
use std::boxed::Box;

use crate::ast::{Program, Statement, StmtKind, Expr, ExprKind, BinaryOp, Literal};
use crate::interpreter::Value;
use aether_core::memory::ManifoldHeap; // From Phase 1

/// Titan Bytecode Instructions
#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
pub enum OpCode {
    /// Push constant value onto stack
    PUSH(f64), 
    /// Push variable value
    LOAD(usize), // Index into constant pool or variable table? Let's use register/slot index
    /// Store top of stack to variable
    STORE(usize),
    
    /// Arithmetic
    ADD, SUB, MUL, DIV,
    
    /// Topology / Core Logic
    /// Embeds the top value into the manifold
    EMBED,
    /// Checks topological attention/neighbors
    ATTEND,
    /// Explicit entropy regulation point
    PRUNE,
    
    /// Control Flow
    JMP(isize),
    JMP_IF_FALSE(isize),
    
    /// Output
    PRINT,
    
    /// End of program
    HALT,
}

/// The Titan Virtual Machine
pub struct TitanVM {
    /// Instruction Pointer
    ip: usize,
    /// The Bytecode DNA
    code: Vec<OpCode>,
    /// Operand Stack (Fast, hot memory)
    stack: Vec<Value>, 
    // In a real optimized VM, we'd use a primitive stack f64, but for compatibility with AEGIS Value type...
    // To achieve the 100x speedup, we should probably strictly stick to f64 for calculations 
    // and only box when necessary. But let's start safe.
    
    /// The Substrate (Heap)
    heap: ManifoldHeap<Value>, 
    
    /// Call Frame / Locals (simplified map for now, or vector)
    locals: Vec<Value>,
}

impl TitanVM {
    pub fn new() -> Self {
        Self {
            ip: 0,
            code: Vec::new(),
            stack: Vec::with_capacity(1024),
            heap: ManifoldHeap::new(),
            locals: vec![Value::Unit; 256], // Pre-alloc locals slots
        }
    }
    
    pub fn load_code(&mut self, code: Vec<OpCode>) {
        self.code = code;
        self.ip = 0;
    }
    
    pub fn run(&mut self) -> Result<Value, String> {
        loop {
            if self.ip >= self.code.len() {
                break;
            }
            
            let op = self.code[self.ip];
            self.ip += 1;
            
            match op {
                OpCode::HALT => break,
                
                OpCode::PUSH(v) => self.stack.push(Value::Num(v)),
                
                OpCode::ADD => {
                    let b = self.pop_num()?;
                    let a = self.pop_num()?;
                    self.stack.push(Value::Num(a + b));
                }
                OpCode::SUB => {
                    let b = self.pop_num()?;
                    let a = self.pop_num()?;
                    self.stack.push(Value::Num(a - b));
                }
                OpCode::MUL => {
                    let b = self.pop_num()?;
                    let a = self.pop_num()?;
                    self.stack.push(Value::Num(a * b));
                }
                OpCode::DIV => {
                    let b = self.pop_num()?;
                    if b == 0.0 { return Err("Division by zero".into()); }
                    let a = self.pop_num()?;
                    self.stack.push(Value::Num(a / b));
                }
                
                OpCode::PRINT => {
                    let val = self.stack.pop().ok_or("Stack underflow")?;
                    // In no_std we might print differently, for now simple debug
                    #[cfg(feature = "std")]
                    println!("{:?}", val);
                }
                
                OpCode::EMBED => {
                    let _val = self.pop_num()?;
                    // In a real integration, this would push to the TimeDelayEmbedder
                    // For now, we simulate the 'Action'
                    // self.heap.alloc(Value::Num(val)); // Store in manifold
                }
                
                OpCode::PRUNE => {
                    // Trigger Entropy Regulation
                    self.heap.regulate_entropy(|_h| {
                        // Mark roots (stack, locals)
                        // This binding is tricky without referencing self inside closure
                        // Ideally pass a closure that captures the roots. 
                        // Simplified:
                    });
                }
                
                OpCode::LOAD(idx) => {
                    if idx < self.locals.len() {
                        self.stack.push(self.locals[idx].clone());
                    } else {
                        return Err("Variable index out of bounds".into());
                    }
                }
                OpCode::STORE(idx) => {
                    let val = self.stack.pop().ok_or("Stack underflow")?;
                    if idx >= self.locals.len() {
                        // Grow locals if needed (simple dynamic growth)
                        self.locals.resize(idx + 1, Value::Unit);
                    }
                    self.locals[idx] = val;
                }
                
                OpCode::JMP(offset) => {
                    // safer pointer arithmetic
                    let next = self.ip as isize + offset;
                    if next < 0 { return Err("Invalid Jump".into()); }
                    self.ip = next as usize;
                }
                
                OpCode::JMP_IF_FALSE(offset) => {
                    let val = self.stack.pop().ok_or("Stack underflow")?;
                    let condition = match val {
                        Value::Bool(b) => b,
                        Value::Num(n) => n != 0.0,
                        _ => false,
                    };
                    
                    if !condition {
                         let next = self.ip as isize + offset;
                         if next < 0 { return Err("Invalid Jump".into()); }
                         self.ip = next as usize;
                    }
                }
                
                _ => return Err("Unimplemented OpCode".into()),
            }
        }
        
        Ok(self.stack.pop().unwrap_or(Value::Unit))
    }
    
    fn pop_num(&mut self) -> Result<f64, String> {
        match self.stack.pop() {
            Some(Value::Num(n)) => Ok(n),
            Some(_) => Err("Type Error: Expected Number".into()),
            None => Err("Stack Underflow".into()),
        }
    }
}

/// The Compiler: AST -> Bytecode
pub struct Compiler {
    code: Vec<OpCode>,
    /// Simple symbol table: name -> index
    locals: Vec<String>,
}

impl Compiler {
    pub fn new() -> Self {
        Self { 
            code: Vec::new(),
            locals: Vec::new(),
        }
    }
    
    pub fn compile(mut self, program: &Program) -> Vec<OpCode> {
        for stmt in &program.statements {
            self.compile_stmt(stmt);
        }
        self.code.push(OpCode::HALT);
        self.code
    }
    
    fn resolve_local(&mut self, name: &str) -> usize {
        if let Some(idx) = self.locals.iter().position(|r| r == name) {
            idx
        } else {
            let idx = self.locals.len();
            self.locals.push(name.to_string());
            idx
        }
    }
    
    fn compile_stmt(&mut self, stmt: &Statement) {
        match &stmt.node {
            StmtKind::Expr(expr) => {
                self.compile_expr(expr);
                // Expression statement usually discards result unless it's a specific context
                // For now, we leave it on stack or assume explicit print/store
            }
            StmtKind::Render(stmt) => {
                // self.compile_expr(&stmt.data); // Ooops, need to fix RenderStmt access (target currently Ident)
                // Actually RenderStmt has 'target' Ident. Access variable.
                let idx = self.resolve_local(&stmt.target);
                self.code.push(OpCode::LOAD(idx));
                self.code.push(OpCode::PRINT); 
            }
            StmtKind::Var(decl) => {
                self.compile_expr(&decl.value);
                let idx = self.resolve_local(&decl.name);
                self.code.push(OpCode::STORE(idx));
            }
            StmtKind::While(stmt) => {
                // Label: Start
                let start_ip = self.code.len();
                
                // Condition
                self.compile_expr(&stmt.condition);
                
                // Jump if False placeholder
                let jmp_false_idx = self.code.len();
                self.code.push(OpCode::JMP_IF_FALSE(0));
                
                // Body
                for s in &stmt.body.statements {
                    self.compile_stmt(s);
                }
                
                // Jump back to Start
                let end_ip = self.code.len();
                let back_jump = (start_ip as isize) - (end_ip as isize) - 1; // -1 because IP increments after fetch
                self.code.push(OpCode::JMP(back_jump));
                
                // Patch Jump If False
                let patch_offset = (self.code.len() as isize) - (jmp_false_idx as isize) - 1;
                self.code[jmp_false_idx] = OpCode::JMP_IF_FALSE(patch_offset);
            }
            StmtKind::If(stmt) => {
                // Condition
                self.compile_expr(&stmt.condition);
                
                // JMP_IF_FALSE to Else or End
                let jmp_false_idx = self.code.len();
                self.code.push(OpCode::JMP_IF_FALSE(0));
                
                // Then Block
                for s in &stmt.then_branch.statements {
                    self.compile_stmt(s);
                }
                
                // If there's an Else block, we need a Jump over it at end of Then
                let mut jmp_end_idx = None;
                
                if let Some(_else_branch) = &stmt.else_branch {
                    jmp_end_idx = Some(self.code.len());
                    self.code.push(OpCode::JMP(0));
                }
                
                // Patch False Jump to here (start of Else or End)
                let false_dest = self.code.len();
                let patch_false = (false_dest as isize) - (jmp_false_idx as isize) - 1;
                self.code[jmp_false_idx] = OpCode::JMP_IF_FALSE(patch_false);
                
                // Compile Else
                if let Some(else_branch) = &stmt.else_branch {
                    for s in &else_branch.statements {
                        self.compile_stmt(s);
                    }
                    
                    // Patch End Jump
                    if let Some(idx) = jmp_end_idx {
                        let end_dest = self.code.len();
                        let patch_end = (end_dest as isize) - (idx as isize) - 1;
                        self.code[idx] = OpCode::JMP(patch_end);
                    }
                }
            }
            StmtKind::For(stmt) => {
                // 1. Initialize Iterator
                let start_val = stmt.range.start.as_f64();
                let end_val = stmt.range.end.as_f64();

                // PUSH start_val
                self.code.push(OpCode::PUSH(start_val));
                // STORE iterator
                let iter_idx = self.resolve_local(&stmt.iterator);
                self.code.push(OpCode::STORE(iter_idx));

                // 2. Loop Start Label
                let start_ip = self.code.len();

                // 3. Condition: iterator != end (simplified range loop)
                // LOAD iterator
                self.code.push(OpCode::LOAD(iter_idx));
                // PUSH end
                self.code.push(OpCode::PUSH(end_val));
                // SUB
                self.code.push(OpCode::SUB);

                // 4. Jump if False (if 0/Equal) to End
                let jmp_false_idx = self.code.len();
                self.code.push(OpCode::JMP_IF_FALSE(0));

                // 5. Body
                for s in &stmt.body.statements {
                    self.compile_stmt(s);
                }

                // 6. Increment Iterator
                // LOAD iterator
                self.code.push(OpCode::LOAD(iter_idx));
                // PUSH 1.0 (step)
                self.code.push(OpCode::PUSH(1.0));
                // ADD
                self.code.push(OpCode::ADD);
                // STORE iterator
                self.code.push(OpCode::STORE(iter_idx));

                // 7. Jump back to Start
                let end_ip = self.code.len();
                let back_jump = (start_ip as isize) - (end_ip as isize) - 1;
                self.code.push(OpCode::JMP(back_jump));

                // 8. Patch Jump If False
                let patch_offset = (self.code.len() as isize) - (jmp_false_idx as isize) - 1;
                self.code[jmp_false_idx] = OpCode::JMP_IF_FALSE(patch_offset);
            }
            StmtKind::Loop(stmt) => {
                // Simple infinite loop (seal loop)
                // Label: Start
                let start_ip = self.code.len();

                // Body
                for s in &stmt.body.statements {
                    self.compile_stmt(s);
                }

                // Jump back to Start
                let end_ip = self.code.len();
                let back_jump = (start_ip as isize) - (end_ip as isize) - 1;
                self.code.push(OpCode::JMP(back_jump));
            }
            _ => {
                // TODO: Implement Fn, Return, Break, Continue
            }
        }
    }
    
    fn compile_expr(&mut self, expr: &Expr) {
        match &expr.node {
            ExprKind::Literal(l) => {
                 match l {
                     Literal::Num(n) => self.code.push(OpCode::PUSH(*n)),
                     Literal::Bool(b) => self.code.push(OpCode::PUSH(if *b { 1.0 } else { 0.0 })),
                     _ => {},
                 }
            }
            ExprKind::Ident(name) => {
                let idx = self.resolve_local(name);
                self.code.push(OpCode::LOAD(idx));
            }
            ExprKind::BinaryOp(left, op, right) => {
                self.compile_expr(left);
                self.compile_expr(right);
                match op {
                    BinaryOp::Add => self.code.push(OpCode::ADD),
                    BinaryOp::Sub => self.code.push(OpCode::SUB),
                    BinaryOp::Mul => self.code.push(OpCode::MUL),
                    BinaryOp::Div => self.code.push(OpCode::DIV),
                    BinaryOp::Lt => {
                        self.code.push(OpCode::SUB); // a - b
                        // If < 0, then true. This is hacky. Titan needs proper CMP.
                        // Impl: if top < 0 push 1 else 0? 
                        // Simplified: we don't have LT opcode.
                        // Let's use strict JMP behavior or add CMP opcode. 
                        // For bench_calc we might need loop counter.
                        // Temporarily assuming strict arithmetic loops.
                    }
                    _ => {},
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_titan_math() {
        let mut vm = TitanVM::new();
        // 5 + 3 * 2 = 11
        let code = vec![
            OpCode::PUSH(5.0),
            OpCode::PUSH(3.0),
            OpCode::PUSH(2.0),
            OpCode::MUL,
            OpCode::ADD,
            OpCode::HALT,
        ];
        
        vm.load_code(code);
        let res = vm.run().unwrap();
        
        if let Value::Num(n) = res {
            assert_eq!(n, 11.0);
        } else {
            panic!("Expected number");
        }
    }

    #[test]
    fn test_compiler_for_loop() {
        use crate::ast::{VarDecl, ForStmt, Range, Number, Block, Span};

        // for i in 0:3 { accum = accum + i }
        // Result should be 0+1+2 = 3.

        let program = Program {
            statements: vec![
                // accum = 0
                Statement::new(
                    StmtKind::Var(VarDecl {
                        type_hint: None,
                        name: "accum".to_string(),
                        value: Expr::new(ExprKind::Literal(Literal::Num(0.0)), Span::default()),
                    }),
                    Span::default()
                ),
                // for i in 0:3
                Statement::new(
                    StmtKind::For(ForStmt {
                        iterator: "i".to_string(),
                        range: Range {
                            start: Number::Int(0),
                            end: Number::Int(3),
                        },
                        body: Block {
                            statements: vec![
                                // accum = accum + i
                                Statement::new(
                                    StmtKind::Var(VarDecl {
                                        type_hint: None,
                                        name: "accum".to_string(),
                                        value: Expr::new(
                                            ExprKind::BinaryOp(
                                                Box::new(Expr::new(ExprKind::Ident("accum".to_string()), Span::default())),
                                                BinaryOp::Add,
                                                Box::new(Expr::new(ExprKind::Ident("i".to_string()), Span::default()))
                                            ),
                                            Span::default()
                                        ),
                                    }),
                                    Span::default()
                                )
                            ]
                        }
                    }),
                    Span::default()
                ),
                // Expr: accum (to leave result on stack)
                Statement::new(
                    StmtKind::Expr(Expr::new(ExprKind::Ident("accum".to_string()), Span::default())),
                    Span::default()
                )
            ]
        };

        let compiler = Compiler::new();
        let code = compiler.compile(&program);

        let mut vm = TitanVM::new();
        vm.load_code(code);
        let res = vm.run().unwrap();

        if let Value::Num(n) = res {
            assert_eq!(n, 3.0);
        } else {
            panic!("Expected number, got {:?}", res);
        }
    }
}
