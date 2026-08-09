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

// ═══════════════════════════════════════════════════════════════════════════════
// Aether-Lang — invented by Teerth Sharma
// https://github.com/teerthsharma/Aether-Lang
// Copyright (c) 2026 Teerth Sharma. All Rights Reserved.
// ═══════════════════════════════════════════════════════════════════════════════
//

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::string::{String, ToString};
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::string::String;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::ast::{BinaryOp, Expr, ExprKind, Literal, Program, Statement, StmtKind, UnaryOp};
use crate::interpreter::Value;
use aether_core::memory::ManifoldHeap; // From Phase 1

/// Titan Bytecode Instructions
#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
pub enum OpCode {
    /// Push constant value onto stack
    PUSH(f64),
    /// Push boolean value onto stack
    PUSH_BOOL(bool),
    /// Push variable value
    LOAD(usize), // Index into constant pool or variable table? Let's use register/slot index
    /// Store top of stack to variable
    STORE(usize),

    /// Arithmetic
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    NEG,
    EQ,
    NEQ,
    LT,
    GT,
    LE,
    GE,
    AND,
    OR,
    NOT,

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
    CALL(usize, usize),
    RET,

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
    frames: Vec<CallFrame>,
}

struct CallFrame {
    return_ip: usize,
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
            frames: Vec::new(),
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
                OpCode::PUSH_BOOL(v) => self.stack.push(Value::Bool(v)),

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
                    if b == 0.0 {
                        return Err("Division by zero".into());
                    }
                    let a = self.pop_num()?;
                    self.stack.push(Value::Num(a / b));
                }
                OpCode::MOD => {
                    let b = self.pop_num()?;
                    if b == 0.0 {
                        return Err("Modulo by zero".into());
                    }
                    let a = self.pop_num()?;
                    self.stack.push(Value::Num(a % b));
                }
                OpCode::NEG => {
                    let value = self.pop_num()?;
                    self.stack.push(Value::Num(-value));
                }
                OpCode::EQ => {
                    let b = self.stack.pop().ok_or("Stack underflow")?;
                    let a = self.stack.pop().ok_or("Stack underflow")?;
                    self.stack.push(Value::Bool(values_equal(&a, &b)));
                }
                OpCode::NEQ => {
                    let b = self.stack.pop().ok_or("Stack underflow")?;
                    let a = self.stack.pop().ok_or("Stack underflow")?;
                    self.stack.push(Value::Bool(!values_equal(&a, &b)));
                }
                OpCode::LT => {
                    let b = self.pop_num()?;
                    let a = self.pop_num()?;
                    self.stack.push(Value::Bool(a < b));
                }
                OpCode::GT => {
                    let b = self.pop_num()?;
                    let a = self.pop_num()?;
                    self.stack.push(Value::Bool(a > b));
                }
                OpCode::LE => {
                    let b = self.pop_num()?;
                    let a = self.pop_num()?;
                    self.stack.push(Value::Bool(a <= b));
                }
                OpCode::GE => {
                    let b = self.pop_num()?;
                    let a = self.pop_num()?;
                    self.stack.push(Value::Bool(a >= b));
                }
                OpCode::AND => {
                    let b = self.pop_truthy()?;
                    let a = self.pop_truthy()?;
                    self.stack.push(Value::Bool(a && b));
                }
                OpCode::OR => {
                    let b = self.pop_truthy()?;
                    let a = self.pop_truthy()?;
                    self.stack.push(Value::Bool(a || b));
                }
                OpCode::NOT => {
                    let value = self.pop_truthy()?;
                    self.stack.push(Value::Bool(!value));
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
                    if next < 0 {
                        return Err("Invalid Jump".into());
                    }
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
                        if next < 0 {
                            return Err("Invalid Jump".into());
                        }
                        self.ip = next as usize;
                    }
                }

                OpCode::CALL(target, arity) => {
                    if target >= self.code.len() {
                        return Err("Function target out of bounds".into());
                    }

                    let mut args = Vec::with_capacity(arity);
                    for _ in 0..arity {
                        args.push(self.stack.pop().ok_or("Stack underflow")?);
                    }
                    args.reverse();

                    let frame = CallFrame {
                        return_ip: self.ip,
                        locals: core::mem::replace(&mut self.locals, vec![Value::Unit; 256]),
                    };
                    self.frames.push(frame);

                    if self.locals.len() < arity {
                        self.locals.resize(arity, Value::Unit);
                    }
                    for (idx, value) in args.into_iter().enumerate() {
                        self.locals[idx] = value;
                    }

                    self.ip = target;
                }

                OpCode::RET => {
                    let value = self.stack.pop().unwrap_or(Value::Unit);
                    let frame = self.frames.pop().ok_or("Return outside function")?;
                    self.locals = frame.locals;
                    self.ip = frame.return_ip;
                    self.stack.push(value);
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

    fn pop_truthy(&mut self) -> Result<bool, String> {
        match self.stack.pop() {
            Some(Value::Bool(value)) => Ok(value),
            Some(Value::Num(value)) => Ok(value != 0.0),
            Some(_) => Err("Type Error: Expected Boolean".into()),
            None => Err("Stack Underflow".into()),
        }
    }
}

fn values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Num(a), Value::Num(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Str(a), Value::Str(b)) => a == b,
        _ => false,
    }
}

/// The Compiler: AST -> Bytecode
pub struct Compiler {
    code: Vec<OpCode>,
    /// Simple symbol table: name -> index
    locals: Vec<String>,
    functions: Vec<CompiledFunction>,
    loop_stack: Vec<LoopContext>,
}

struct CompiledFunction {
    name: String,
    target: usize,
    arity: usize,
}

struct LoopContext {
    break_jumps: Vec<usize>,
    continue_jumps: Vec<usize>,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            code: Vec::new(),
            locals: Vec::new(),
            functions: Vec::new(),
            loop_stack: Vec::new(),
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

    fn emit_jump_placeholder(&mut self) -> usize {
        let idx = self.code.len();
        self.code.push(OpCode::JMP(0));
        idx
    }

    fn patch_jump_to(&mut self, idx: usize, target: usize) {
        let offset = (target as isize) - (idx as isize) - 1;
        self.code[idx] = OpCode::JMP(offset);
    }

    fn push_loop_context(&mut self) {
        self.loop_stack.push(LoopContext {
            break_jumps: Vec::new(),
            continue_jumps: Vec::new(),
        });
    }

    fn patch_loop_context(&mut self, continue_target: usize, break_target: usize) {
        if let Some(context) = self.loop_stack.pop() {
            for idx in context.continue_jumps {
                self.patch_jump_to(idx, continue_target);
            }
            for idx in context.break_jumps {
                self.patch_jump_to(idx, break_target);
            }
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
            StmtKind::Assign(stmt) => {
                self.compile_expr(&stmt.value);
                let idx = self.resolve_local(&stmt.name);
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
                self.push_loop_context();
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
                self.patch_loop_context(start_ip, self.code.len());
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
                self.push_loop_context();
                for s in &stmt.body.statements {
                    self.compile_stmt(s);
                }

                // 6. Increment Iterator
                let continue_target = self.code.len();
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
                self.patch_loop_context(continue_target, self.code.len());
            }
            StmtKind::Loop(stmt) => {
                // Label: Start
                let start_ip = self.code.len();
                let mut jmp_until_idx = None;

                if let Some(condition) = &stmt.until {
                    self.compile_expr(condition);
                    self.code.push(OpCode::NOT);
                    jmp_until_idx = Some(self.code.len());
                    self.code.push(OpCode::JMP_IF_FALSE(0));
                }

                // Body
                self.push_loop_context();
                for s in &stmt.body.statements {
                    self.compile_stmt(s);
                }

                // Jump back to Start
                let end_ip = self.code.len();
                let back_jump = (start_ip as isize) - (end_ip as isize) - 1;
                self.code.push(OpCode::JMP(back_jump));

                if let Some(idx) = jmp_until_idx {
                    let patch_offset = (self.code.len() as isize) - (idx as isize) - 1;
                    self.code[idx] = OpCode::JMP_IF_FALSE(patch_offset);
                }
                self.patch_loop_context(start_ip, self.code.len());
            }
            StmtKind::Fn(decl) => {
                let jmp_over_idx = self.code.len();
                self.code.push(OpCode::JMP(0));

                let target = self.code.len();
                self.functions.push(CompiledFunction {
                    name: decl.name.clone(),
                    target,
                    arity: decl.params.len(),
                });

                let outer_locals = core::mem::replace(&mut self.locals, decl.params.clone());
                for s in &decl.body.statements {
                    self.compile_stmt(s);
                }
                self.code.push(OpCode::RET);
                self.locals = outer_locals;

                let patch_offset = (self.code.len() as isize) - (jmp_over_idx as isize) - 1;
                self.code[jmp_over_idx] = OpCode::JMP(patch_offset);
            }
            StmtKind::Return(stmt) => {
                if let Some(value) = &stmt.value {
                    self.compile_expr(value);
                }
                self.code.push(OpCode::RET);
            }
            StmtKind::Break(_) => {
                let idx = self.emit_jump_placeholder();
                if let Some(context) = self.loop_stack.last_mut() {
                    context.break_jumps.push(idx);
                }
            }
            StmtKind::Continue(_) => {
                let idx = self.emit_jump_placeholder();
                if let Some(context) = self.loop_stack.last_mut() {
                    context.continue_jumps.push(idx);
                }
            }
            _ => {
                // TODO: Implement unsupported surface forms in the VM compiler.
            }
        }
    }

    fn compile_expr(&mut self, expr: &Expr) {
        match &expr.node {
            ExprKind::Literal(l) => match l {
                Literal::Num(n) => self.code.push(OpCode::PUSH(*n)),
                Literal::Bool(b) => self.code.push(OpCode::PUSH_BOOL(*b)),
                _ => {}
            },
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
                    BinaryOp::Mod => self.code.push(OpCode::MOD),
                    BinaryOp::Eq => self.code.push(OpCode::EQ),
                    BinaryOp::Neq => self.code.push(OpCode::NEQ),
                    BinaryOp::Lt => self.code.push(OpCode::LT),
                    BinaryOp::Gt => self.code.push(OpCode::GT),
                    BinaryOp::Le => self.code.push(OpCode::LE),
                    BinaryOp::Ge => self.code.push(OpCode::GE),
                    BinaryOp::And => self.code.push(OpCode::AND),
                    BinaryOp::Or => self.code.push(OpCode::OR),
                }
            }
            ExprKind::UnaryOp(op, expr) => {
                self.compile_expr(expr);
                match op {
                    UnaryOp::Neg => self.code.push(OpCode::NEG),
                    UnaryOp::Not => self.code.push(OpCode::NOT),
                }
            }
            ExprKind::Call { name, args } => {
                let function_idx = self
                    .functions
                    .iter()
                    .position(|function| function.name == *name);
                if let Some(function_idx) = function_idx {
                    for arg in args {
                        if let crate::ast::CallArg::Positional(expr) = arg {
                            self.compile_expr(expr);
                        }
                    }
                    let function = &self.functions[function_idx];
                    self.code
                        .push(OpCode::CALL(function.target, function.arity));
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    fn run_source(source: &str) -> Value {
        let mut parser = Parser::new(source);
        let program = parser.parse().expect("source should parse");
        let compiler = Compiler::new();
        let code = compiler.compile(&program);
        let mut vm = TitanVM::new();
        vm.load_code(code);
        vm.run().expect("vm should run")
    }

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
        use crate::ast::{Block, ForStmt, Number, Range, Span, VarDecl};

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
                    Span::default(),
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
                                                Box::new(Expr::new(
                                                    ExprKind::Ident("accum".to_string()),
                                                    Span::default(),
                                                )),
                                                BinaryOp::Add,
                                                Box::new(Expr::new(
                                                    ExprKind::Ident("i".to_string()),
                                                    Span::default(),
                                                )),
                                            ),
                                            Span::default(),
                                        ),
                                    }),
                                    Span::default(),
                                ),
                            ],
                        },
                    }),
                    Span::default(),
                ),
                // Expr: accum (to leave result on stack)
                Statement::new(
                    StmtKind::Expr(Expr::new(
                        ExprKind::Ident("accum".to_string()),
                        Span::default(),
                    )),
                    Span::default(),
                ),
            ],
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

    #[test]
    fn test_vm_modulo_comparison_and_boolean_ops() {
        let res = run_source("let ok = 10 % 4 == 2 && !false~ ok~");

        assert!(matches!(res, Value::Bool(true)));
    }

    #[test]
    fn test_vm_assignment_if_and_while_match_interpreter_surface() {
        let res = run_source(
            "let i = 0~
             if 1 < 2 { i = i + 1~ }
             while i < 3 { i = i + 1~ }
             i~",
        );

        assert!(matches!(res, Value::Num(3.0)));
    }

    #[test]
    fn test_vm_seal_until_condition() {
        let res = run_source("let count = 0~ seal until count >= 3 { count = count + 1~ } count~");

        assert!(matches!(res, Value::Num(3.0)));
    }

    #[test]
    fn test_vm_user_function_explicit_return() {
        let res = run_source("fn add(a, b) { return a + b~ } let result = add(2, 3)~ result~");

        assert!(matches!(res, Value::Num(5.0)));
    }

    #[test]
    fn test_vm_user_function_implicit_last_expression_return() {
        let res = run_source("fn one() { let x = 1~ x~ } one()~");

        assert!(matches!(res, Value::Num(1.0)));
    }

    #[test]
    fn test_vm_user_function_parameters_are_call_frame_local() {
        let res = run_source("let x = 10~ fn id(x) { return x~ } let y = id(3)~ let z = x + y~ z~");

        assert!(matches!(res, Value::Num(13.0)));
    }

    #[test]
    fn test_vm_break_exits_while_loop() {
        let res = run_source(
            "let i = 0~
             let sum = 0~
             while i < 10 {
                 i = i + 1~
                 if i == 4 { break~ }
                 sum = sum + i~
             }
             sum~",
        );

        assert!(matches!(res, Value::Num(6.0)));
    }

    #[test]
    fn test_vm_continue_skips_to_next_while_iteration() {
        let res = run_source(
            "let i = 0~
             let sum = 0~
             while i < 5 {
                 i = i + 1~
                 if i == 3 { continue~ }
                 sum = sum + i~
             }
             sum~",
        );

        assert!(matches!(res, Value::Num(12.0)));
    }
}
