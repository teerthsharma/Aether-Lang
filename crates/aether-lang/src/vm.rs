//! ═══════════════════════════════════════════════════════════════════════════════
//! TitanVM: the bytecode engine
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! `Compiler::compile` turns a parsed program into a `Chunk` and refuses, with a
//! `CompileError` naming the construct and its position, anything it does not
//! compile. `TitanVM::run` executes the chunk with the semantics of the tree
//! interpreter (`interpreter.rs` is the reference):
//!
//! - the value of a program, block or function body is the value of its last
//!   statement, kept in the LAST register; expression statements pop into it,
//!   so the operand stack is empty at every statement boundary;
//! - `if`, `while` and `seal until` conditions must be booleans;
//! - a function is bound when its `fn` statement executes, and sees the globals
//!   read-only plus its own parameters and locals; its writes stay local;
//! - natives come from `natives`, bound by `import` at compile time and called
//!   with evaluated arguments; they call program functions back through `Host`.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
// Aether-Lang — invented by Teerth Sharma
// https://github.com/teerthsharma/Aether-Lang
// Copyright (c) 2026 Teerth Sharma. All Rights Reserved.
// ═══════════════════════════════════════════════════════════════════════════════
//

use alloc::collections::BTreeMap;
#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec, vec::Vec};

use crate::ast::{
    BinaryOp, Block, CallArg, Expr, ExprKind, FnDecl, ImportStmt, Literal, LoopCond, Program, Span,
    Statement, StmtKind, UnaryOp,
};
use crate::interpreter::{ManifoldWorkspace, Value};
use crate::natives::{self, seal, Host, NativeArgs, NativeId};

/// Frames deeper than this are refused rather than grown without bound.
const MAX_DEPTH: usize = 10_000;

/// A construct the compiler does not translate, and where it is.
#[derive(Debug, Clone, PartialEq)]
pub struct CompileError {
    pub construct: String,
    pub line: usize,
    pub col: usize,
}

impl core::fmt::Display for CompileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "titan cannot compile {} at line {}, column {}",
            self.construct, self.line, self.col
        )
    }
}

fn refuse<T>(construct: impl Into<String>, span: Span) -> Result<T, CompileError> {
    Err(CompileError {
        construct: construct.into(),
        line: span.line,
        col: span.col,
    })
}

/// Where a name's value lives.
#[derive(Debug, Clone, Copy)]
enum Slot {
    Global(usize),
    /// Relative to the current frame's base.
    Local(usize),
    /// A local that shares a global's name: reads fall back to the global
    /// until the function writes the local.
    Shadow(usize, usize),
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Num(f64),
    Bool(bool),
    Unit,
    /// Push a constant: strings, function values, argument-name lists.
    Const(usize),
    Dup,
    /// Push a slot's value; an unbound name reads as Unit.
    Load(Slot),
    Store(Slot),
    /// Unbind a hidden local.
    Clear(usize),
    /// Fail with the text of the constant unless the slot is bound.
    Bound(Slot, usize),
    /// Pop into LAST.
    SetLast,
    PushLast,
    Bin(BinaryOp),
    Neg,
    Not,
    List(usize),
    /// Pop a record, push the field named by the constant.
    Field(usize),
    Element,
    Jump(usize),
    /// Pop a Bool and jump when it equals the flag. Anything else is refused.
    JumpIf(bool, usize),
    /// Fail with the text of the constant.
    Fail(usize),
    Print,
    /// Push the function in the slot; the constant names it in errors.
    Callee(Slot, usize),
    /// Call the function below `argc` arguments.
    CallValue(usize),
    Ret,
    /// Native, argc, constant listing each argument's name (Unit if positional).
    CallNative(NativeId, usize, usize),
    /// Receiver slot, method-name constant, argc, argument-name constant.
    Method(Slot, usize, usize, usize),
    /// Pop the series, push the manifold `natives::embed_manifold` builds.
    Manifold {
        dim: usize,
        tau: usize,
    },
    /// Count a seal pass in the hidden local; push whether one was left.
    SealPass(usize),
    /// Pop the watched value, push whether it equals the previous pass's
    /// (kept in the hidden local); otherwise remember it.
    SealStable(usize),
    /// Pop and check a convergence tolerance into the hidden local.
    SealEps(usize),
    /// Push whether LAST moved at most `eps` from the previous pass's body
    /// value, then remember LAST.
    SealConverge {
        prev: usize,
        eps: usize,
    },
}

#[derive(Debug)]
struct Function {
    name: String,
    entry: usize,
    arity: usize,
    locals: usize,
}

/// A compiled program.
#[derive(Debug)]
pub struct Chunk {
    code: Vec<Op>,
    consts: Vec<Value>,
    functions: Vec<Function>,
    globals: usize,
    main_locals: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Compiler
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Default)]
struct Loop {
    breaks: Vec<usize>,
    continues: Vec<usize>,
}

/// AST -> bytecode.
pub struct Compiler {
    code: Vec<Op>,
    consts: Vec<Value>,
    functions: Vec<Function>,
    /// Function bodies still to compile. They compile after the top level, so
    /// they see every import and every global.
    pending: Vec<(usize, FnDecl)>,
    /// Global slots: name, and whether top-level code writes it.
    globals: Vec<(String, bool)>,
    /// Names bound to natives by top-level imports.
    natives: BTreeMap<String, NativeId>,
    /// Modules bound by `import module~`, for `module.f(...)`.
    modules: Vec<String>,
    /// Local slots of the code being compiled, parameters first; "" is hidden.
    frame: Vec<String>,
    in_fn: bool,
    loops: Vec<Loop>,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            code: Vec::new(),
            consts: Vec::new(),
            functions: Vec::new(),
            pending: Vec::new(),
            globals: Vec::new(),
            natives: BTreeMap::new(),
            modules: Vec::new(),
            frame: Vec::new(),
            in_fn: false,
            loops: Vec::new(),
        }
    }

    pub fn compile(mut self, program: &Program) -> Result<Chunk, CompileError> {
        for s in &program.statements {
            match &s.node {
                StmtKind::Import(i) => self.import(i, s.span)?,
                _ => self.stmt(s)?,
            }
        }
        self.emit(Op::PushLast);
        self.emit(Op::Ret);
        let main_locals = self.frame.len();
        while let Some((idx, decl)) = self.pending.pop() {
            self.function(idx, &decl)?;
        }
        Ok(Chunk {
            code: self.code,
            consts: self.consts,
            functions: self.functions,
            globals: self.globals.len(),
            main_locals,
        })
    }

    fn function(&mut self, idx: usize, decl: &FnDecl) -> Result<(), CompileError> {
        let mut frame = decl.params.clone();
        self.scan_block(&decl.body, &mut frame);
        self.frame = frame;
        self.in_fn = true;
        let entry = self.code.len();
        self.block(&decl.body)?;
        self.emit(Op::PushLast);
        self.emit(Op::Ret);
        let f = &mut self.functions[idx];
        f.entry = entry;
        f.locals = self.frame.len();
        Ok(())
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    fn emit(&mut self, op: Op) -> usize {
        self.code.push(op);
        self.code.len() - 1
    }

    fn patch(&mut self, at: usize, to: usize) {
        if let Op::Jump(t) | Op::JumpIf(_, t) = &mut self.code[at] {
            *t = to;
        }
    }

    fn konst(&mut self, v: Value) -> usize {
        self.consts.push(v);
        self.consts.len() - 1
    }

    fn text(&mut self, s: String) -> usize {
        self.konst(Value::Str(s))
    }

    fn hidden(&mut self) -> usize {
        self.frame.push(String::new());
        self.frame.len() - 1
    }

    fn global(&mut self, name: &str) -> usize {
        match self.globals.iter().position(|(n, _)| n == name) {
            Some(g) => g,
            None => {
                self.globals.push((String::from(name), false));
                self.globals.len() - 1
            }
        }
    }

    fn is_local(&self, name: &str) -> bool {
        self.in_fn && self.frame.iter().any(|n| n == name)
    }

    fn slot(&mut self, name: &str) -> Slot {
        if self.in_fn {
            if let Some(l) = self.frame.iter().position(|n| n == name) {
                return match self.globals.iter().position(|(n, _)| n == name) {
                    Some(g) => Slot::Shadow(l, g),
                    None => Slot::Local(l),
                };
            }
        }
        Slot::Global(self.global(name))
    }

    /// The slot a write to `name` goes to: a local inside a function, else a
    /// global that no import holds.
    fn write_slot(&mut self, name: &str, span: Span) -> Result<Slot, CompileError> {
        if name == "print" {
            return refuse("a binding named 'print'", span);
        }
        if self.in_fn {
            if !self.is_local(name) {
                self.frame.push(String::from(name));
            }
            return Ok(self.slot(name));
        }
        if self.natives.contains_key(name) || self.modules.iter().any(|m| m == name) {
            return refuse(
                format!("assignment to '{name}', which an import binds"),
                span,
            );
        }
        let g = self.global(name);
        self.globals[g].1 = true;
        Ok(Slot::Global(g))
    }

    // ── statements ───────────────────────────────────────────────────────────

    fn block(&mut self, b: &Block) -> Result<(), CompileError> {
        if b.statements.is_empty() {
            self.emit(Op::Unit);
            self.emit(Op::SetLast);
        }
        for s in &b.statements {
            self.stmt(s)?;
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Statement) -> Result<(), CompileError> {
        let span = s.span;
        match &s.node {
            StmtKind::Expr(e) => {
                self.expr(e)?;
                self.emit(Op::SetLast);
            }
            StmtKind::Var(d) => {
                self.expr(&d.value)?;
                let slot = self.write_slot(&d.name, span)?;
                self.emit(Op::Dup);
                self.emit(Op::Store(slot));
                self.emit(Op::SetLast);
            }
            StmtKind::Assign(a) => {
                let slot = self.write_slot(&a.name, span)?;
                let msg = self.text(format!("cannot assign undefined variable '{}'", a.name));
                self.emit(Op::Bound(slot, msg));
                self.expr(&a.value)?;
                self.emit(Op::Dup);
                self.emit(Op::Store(slot));
                self.emit(Op::SetLast);
            }
            StmtKind::Manifold(d) => {
                let ExprKind::Call { name, args } = &d.init.node else {
                    return refuse("a manifold initialiser other than embed(...)", span);
                };
                if name != "embed" {
                    return refuse("a manifold initialiser other than embed(...)", span);
                }
                // As the interpreter: only literal `dim=`/`tau=` count, default 3.
                let literal = |key: &str| {
                    args.iter().find_map(|a| match a {
                        CallArg::Named { name, value } if name == key => match value.node {
                            ExprKind::Literal(Literal::Num(n)) => Some(n as usize),
                            _ => None,
                        },
                        _ => None,
                    })
                };
                let (dim, tau) = (literal("dim").unwrap_or(3), literal("tau").unwrap_or(3));
                let data = args.iter().find_map(|a| match a {
                    CallArg::Positional(e) => Some(e),
                    CallArg::Named { name, value } if name == "data" => Some(value),
                    CallArg::Named { .. } => None,
                });
                match data {
                    Some(e) => self.expr(e)?,
                    None => {
                        // The interpreter's built-in series: sin(0.1 i), i < 64.
                        let sample = (0..64)
                            .map(|i| Value::Num(libm::sin(i as f64 * 0.1)))
                            .collect();
                        let k = self.konst(Value::List(sample));
                        self.emit(Op::Const(k));
                    }
                }
                self.emit(Op::Manifold { dim, tau });
                let slot = self.write_slot(&d.name, span)?;
                self.emit(Op::Dup);
                self.emit(Op::Store(slot));
                self.emit(Op::SetLast);
            }
            StmtKind::Render(_) | StmtKind::Empty => {
                self.emit(Op::Unit);
                self.emit(Op::SetLast);
            }
            StmtKind::If(i) => {
                self.expr(&i.condition)?;
                let skip = self.emit(Op::JumpIf(false, 0));
                self.block(&i.then_branch)?;
                let end = self.emit(Op::Jump(0));
                self.patch(skip, self.code.len());
                match &i.else_branch {
                    Some(b) => self.block(b)?,
                    None => {
                        self.emit(Op::Unit);
                        self.emit(Op::SetLast);
                    }
                }
                self.patch(end, self.code.len());
            }
            StmtKind::While(w) => {
                let acc = self.loop_start();
                let top = self.code.len();
                self.expr(&w.condition)?;
                let exit = self.emit(Op::JumpIf(false, 0));
                let ctx = self.loop_body(&w.body, acc)?;
                self.loop_end(ctx, top, top, &[exit], acc);
            }
            StmtKind::For(f) => {
                // As the interpreter: i64 bounds, a step of +1 or -1, and the
                // iterator left at the first value that failed the test.
                let start = f.range.start.as_f64() as i64;
                let end = f.range.end.as_f64() as i64;
                let (step, test) = if start <= end {
                    (1.0, BinaryOp::Lt)
                } else {
                    (-1.0, BinaryOp::Gt)
                };
                let it = self.write_slot(&f.iterator, span)?;
                let acc = self.loop_start();
                let cur = Slot::Local(self.hidden());
                self.emit(Op::Num(start as f64));
                self.emit(Op::Store(cur));
                let top = self.code.len();
                self.emit(Op::Load(cur));
                self.emit(Op::Num(end as f64));
                self.emit(Op::Bin(test));
                let exit = self.emit(Op::JumpIf(false, 0));
                self.emit(Op::Load(cur));
                self.emit(Op::Store(it));
                let ctx = self.loop_body(&f.body, acc)?;
                let cont = self.code.len();
                self.emit(Op::Load(cur));
                self.emit(Op::Num(step));
                self.emit(Op::Bin(BinaryOp::Add));
                self.emit(Op::Store(cur));
                self.loop_end(ctx, top, cont, &[exit], acc);
                self.emit(Op::Load(cur));
                self.emit(Op::Store(it));
            }
            StmtKind::Loop(l) => {
                let acc = self.loop_start();
                let count = self.hidden();
                self.emit(Op::Clear(count));
                let prev = self.hidden();
                self.emit(Op::Clear(prev));
                let eps = match &l.until {
                    Some(LoopCond::Convergence(e)) => {
                        self.expr(e)?;
                        let eps = self.hidden();
                        self.emit(Op::SealEps(eps));
                        Some(eps)
                    }
                    Some(LoopCond::Expr(_) | LoopCond::Stable(_)) | None => None,
                };
                let top = self.code.len();
                self.emit(Op::SealPass(count));
                let mut exits = vec![self.emit(Op::JumpIf(false, 0))];
                match &l.until {
                    Some(LoopCond::Expr(c)) => {
                        self.expr(c)?;
                        exits.push(self.emit(Op::JumpIf(true, 0)));
                    }
                    Some(LoopCond::Stable(e)) => {
                        self.expr(e)?;
                        self.emit(Op::SealStable(prev));
                        exits.push(self.emit(Op::JumpIf(true, 0)));
                    }
                    Some(LoopCond::Convergence(_)) | None => {}
                }
                let ctx = self.loop_body(&l.body, acc)?;
                if let Some(eps) = eps {
                    self.emit(Op::SealConverge { prev, eps });
                    exits.push(self.emit(Op::JumpIf(true, 0)));
                }
                self.loop_end(ctx, top, top, &exits, acc);
            }
            StmtKind::Fn(d) => {
                for (i, p) in d.params.iter().enumerate() {
                    if d.params[..i].contains(p) {
                        return refuse(format!("duplicate parameter '{p}'"), span);
                    }
                }
                let idx = self.functions.len();
                self.functions.push(Function {
                    name: d.name.clone(),
                    entry: 0,
                    arity: d.params.len(),
                    locals: 0,
                });
                self.pending.push((idx, d.clone()));
                let k = self.konst(Value::Compiled(idx));
                let slot = self.write_slot(&d.name, span)?;
                self.emit(Op::Const(k));
                self.emit(Op::Store(slot));
                self.emit(Op::Unit);
                self.emit(Op::SetLast);
            }
            StmtKind::Return(r) => {
                // A return inside a loop leaves the function: `Ret` truncates
                // the frame's stack and locals, loop slots included.
                match &r.value {
                    Some(e) => self.expr(e)?,
                    None => {
                        self.emit(Op::Unit);
                    }
                }
                self.emit(Op::Ret);
            }
            StmtKind::Break(_) | StmtKind::Continue(_) => {
                let brk = matches!(s.node, StmtKind::Break(_));
                if self.loops.is_empty() {
                    self.fail(if brk {
                        "break outside loop"
                    } else {
                        "continue outside loop"
                    });
                } else {
                    let j = self.emit(Op::Jump(0));
                    if let Some(l) = self.loops.last_mut() {
                        let jumps = if brk { &mut l.breaks } else { &mut l.continues };
                        jumps.push(j);
                    }
                }
            }
            StmtKind::Import(_) => return refuse("import inside a block or function", span),
            StmtKind::Block(_) => return refuse("block declaration", span),
            StmtKind::Regress(_) => return refuse("regress", span),
            StmtKind::Class(_) => return refuse("class", span),
        }
        Ok(())
    }

    fn fail(&mut self, msg: &str) {
        let k = self.text(String::from(msg));
        self.emit(Op::Fail(k));
    }

    /// A loop's value slot, starting at Unit (a loop that never completes a
    /// pass is worth Unit).
    fn loop_start(&mut self) -> Slot {
        let acc = Slot::Local(self.hidden());
        self.emit(Op::Unit);
        self.emit(Op::Store(acc));
        acc
    }

    /// The body; a pass that completes stores its value in `acc`. `break` and
    /// `continue` skip that store, as in the interpreter.
    fn loop_body(&mut self, body: &Block, acc: Slot) -> Result<Loop, CompileError> {
        self.loops.push(Loop::default());
        let compiled = self.block(body);
        let ctx = self.loops.pop().unwrap_or_default();
        compiled?;
        self.emit(Op::PushLast);
        self.emit(Op::Store(acc));
        Ok(ctx)
    }

    fn loop_end(&mut self, ctx: Loop, top: usize, cont: usize, exits: &[usize], acc: Slot) {
        self.emit(Op::Jump(top));
        let exit = self.code.len();
        for &j in exits.iter().chain(&ctx.breaks) {
            self.patch(j, exit);
        }
        for &j in &ctx.continues {
            self.patch(j, cont);
        }
        self.emit(Op::Load(acc));
        self.emit(Op::SetLast);
    }

    /// `import m~` and `from m import f~`. Top level only: bindings are
    /// resolved at compile time, in program order.
    fn import(&mut self, i: &ImportStmt, span: Span) -> Result<(), CompileError> {
        match (natives::exports(&i.module), &i.symbol) {
            (None, _) => self.fail(&format!("Module '{}' not found", i.module)),
            (Some(names), Some(sym)) if !names.contains(&sym.as_str()) => {
                self.fail(&format!("Symbol '{}' not found in {}", sym, i.module))
            }
            (Some(_), Some(sym)) => self.bind_native(&i.module, sym, span)?,
            (Some(names), None) => {
                self.bind_name(&i.module, span)?;
                self.modules.push(i.module.clone());
                for name in names {
                    self.bind_native(&i.module, name, span)?;
                }
            }
        }
        self.emit(Op::Unit);
        self.emit(Op::SetLast);
        Ok(())
    }

    fn bind_name(&mut self, name: &str, span: Span) -> Result<(), CompileError> {
        if self
            .globals
            .iter()
            .any(|(n, written)| n == name && *written)
        {
            return refuse(
                format!("import of '{name}', which the program also assigns"),
                span,
            );
        }
        Ok(())
    }

    fn bind_native(&mut self, module: &str, name: &str, span: Span) -> Result<(), CompileError> {
        self.bind_name(name, span)?;
        // A constant (`math.pi`) binds as a global value, as the interpreter
        // binds it; only functions become native call sites.
        if let Some(Value::Num(v)) = natives::constant(module, name) {
            self.emit(Op::Num(v));
            let slot = self.write_slot(name, span)?;
            self.emit(Op::Store(slot));
            return Ok(());
        }
        let Some(id) = natives::lookup(module, name) else {
            return refuse(
                format!("{module}.{name}, which exports lists but lookup does not"),
                span,
            );
        };
        self.natives.insert(String::from(name), id);
        Ok(())
    }

    // ── expressions ──────────────────────────────────────────────────────────

    fn expr(&mut self, e: &Expr) -> Result<(), CompileError> {
        let span = e.span;
        match &e.node {
            ExprKind::Literal(Literal::Num(n)) => {
                self.emit(Op::Num(*n));
            }
            ExprKind::Literal(Literal::Bool(b)) => {
                self.emit(Op::Bool(*b));
            }
            ExprKind::Literal(Literal::Str(s)) => {
                let k = self.text(s.clone());
                self.emit(Op::Const(k));
            }
            ExprKind::Ident(name) => self.load(name, span)?,
            ExprKind::BinaryOp(a, op, b) => {
                // Both sides, no short circuit, as the interpreter.
                self.expr(a)?;
                self.expr(b)?;
                self.emit(Op::Bin(*op));
            }
            ExprKind::UnaryOp(op, a) => {
                self.expr(a)?;
                self.emit(match op {
                    UnaryOp::Neg => Op::Neg,
                    UnaryOp::Not => Op::Not,
                });
            }
            ExprKind::List(items) => {
                for x in items {
                    self.expr(x)?;
                }
                self.emit(Op::List(items.len()));
            }
            ExprKind::FieldAccess { object, field } => {
                if !self.is_local(object) && self.modules.contains(object) {
                    return refuse(format!("native {object}.{field} as a value"), span);
                }
                self.load(object, span)?;
                let k = self.text(field.clone());
                self.emit(Op::Field(k));
            }
            ExprKind::Member { object, field } => {
                self.expr(object)?;
                let k = self.text(field.clone());
                self.emit(Op::Field(k));
            }
            ExprKind::Element { object, index } => {
                self.expr(object)?;
                self.expr(index)?;
                self.emit(Op::Element);
            }
            ExprKind::Call { name, args } => self.call(name, args, span)?,
            ExprKind::MethodCall {
                object,
                method,
                args,
            } => {
                if !self.is_local(object) && self.modules.contains(object) {
                    match natives::lookup(object, method) {
                        Some(id) => self.call_native(id, args)?,
                        None => {
                            self.fail(&format!("Method '{method}' not found in module '{object}'"))
                        }
                    }
                    return Ok(());
                }
                let slot = self.slot(object);
                let msg = self.text(format!("Object '{object}' not found"));
                self.emit(Op::Bound(slot, msg));
                let names = self.args(args)?;
                let k = self.text(method.clone());
                self.emit(Op::Method(slot, k, args.len(), names));
            }
            ExprKind::Index { .. } => return refuse("manifold slice M[a:b]", span),
            ExprKind::New { .. } => return refuse("new (objects)", span),
            ExprKind::Config(_) => return refuse("config block as a value", span),
            ExprKind::Range(_) => return refuse("range as a value", span),
        }
        Ok(())
    }

    fn load(&mut self, name: &str, span: Span) -> Result<(), CompileError> {
        if !self.is_local(name) {
            if name == "print" || self.modules.iter().any(|m| m == name) {
                return refuse(format!("'{name}' as a value"), span);
            }
            // A bare native (`pi`) is its zero-argument call.
            if let Some(&id) = self.natives.get(name) {
                let names = self.konst(Value::List(Vec::new()));
                self.emit(Op::CallNative(id, 0, names));
                return Ok(());
            }
        }
        let slot = self.slot(name);
        self.emit(Op::Load(slot));
        Ok(())
    }

    fn call(&mut self, name: &str, args: &[CallArg], span: Span) -> Result<(), CompileError> {
        if name == "print" {
            // One argument at a time: evaluate, then print.
            for a in args {
                let CallArg::Positional(x) = a else {
                    return refuse("a named argument to print", span);
                };
                self.expr(x)?;
                self.emit(Op::Print);
            }
            self.emit(Op::Unit);
            return Ok(());
        }
        if !self.is_local(name) {
            if let Some(&id) = self.natives.get(name) {
                return self.call_native(id, args);
            }
        }
        let slot = self.slot(name);
        let k = self.text(String::from(name));
        self.emit(Op::Callee(slot, k));
        if args.iter().any(|a| matches!(a, CallArg::Named { .. })) {
            // The interpreter resolves the callee, then refuses before
            // evaluating any argument; so does this.
            self.fail(&format!(
                "function '{name}' does not accept named arguments"
            ));
            return Ok(());
        }
        self.args(args)?;
        self.emit(Op::CallValue(args.len()));
        Ok(())
    }

    fn call_native(&mut self, id: NativeId, args: &[CallArg]) -> Result<(), CompileError> {
        let names = self.args(args)?;
        self.emit(Op::CallNative(id, args.len(), names));
        Ok(())
    }

    /// Evaluate arguments in source order; return the constant naming them.
    fn args(&mut self, args: &[CallArg]) -> Result<usize, CompileError> {
        let mut names = Vec::with_capacity(args.len());
        for a in args {
            match a {
                CallArg::Positional(x) => {
                    self.expr(x)?;
                    names.push(Value::Unit);
                }
                CallArg::Named { name, value } => {
                    self.expr(value)?;
                    names.push(Value::Str(name.clone()));
                }
            }
        }
        Ok(self.konst(Value::List(names)))
    }

    // ── locals of a function body ────────────────────────────────────────────

    /// Every name a body writes is a local of the function, including method
    /// receivers (a method writes its receiver back). Nested fn bodies are
    /// their own functions; only their names count here.
    fn scan_block(&self, b: &Block, out: &mut Vec<String>) {
        for s in &b.statements {
            self.scan_stmt(s, out);
        }
    }

    fn scan_stmt(&self, s: &Statement, out: &mut Vec<String>) {
        match &s.node {
            StmtKind::Var(d) => {
                add(out, &d.name);
                self.scan_expr(&d.value, out);
            }
            StmtKind::Assign(a) => {
                add(out, &a.name);
                self.scan_expr(&a.value, out);
            }
            StmtKind::Manifold(d) => {
                add(out, &d.name);
                self.scan_expr(&d.init, out);
            }
            StmtKind::Fn(d) => add(out, &d.name),
            StmtKind::For(f) => {
                add(out, &f.iterator);
                self.scan_block(&f.body, out);
            }
            StmtKind::If(i) => {
                self.scan_expr(&i.condition, out);
                self.scan_block(&i.then_branch, out);
                if let Some(b) = &i.else_branch {
                    self.scan_block(b, out);
                }
            }
            StmtKind::While(w) => {
                self.scan_expr(&w.condition, out);
                self.scan_block(&w.body, out);
            }
            StmtKind::Loop(l) => {
                if let Some(LoopCond::Expr(e) | LoopCond::Stable(e) | LoopCond::Convergence(e)) =
                    &l.until
                {
                    self.scan_expr(e, out);
                }
                self.scan_block(&l.body, out);
            }
            StmtKind::Return(r) => {
                if let Some(e) = &r.value {
                    self.scan_expr(e, out);
                }
            }
            StmtKind::Expr(e) => self.scan_expr(e, out),
            // Refused by `stmt`, or bind nothing.
            StmtKind::Block(_)
            | StmtKind::Regress(_)
            | StmtKind::Render(_)
            | StmtKind::Class(_)
            | StmtKind::Import(_)
            | StmtKind::Break(_)
            | StmtKind::Continue(_)
            | StmtKind::Empty => {}
        }
    }

    fn scan_expr(&self, e: &Expr, out: &mut Vec<String>) {
        match &e.node {
            ExprKind::MethodCall { object, args, .. } => {
                if !self.modules.contains(object) {
                    add(out, object);
                }
                self.scan_args(args, out);
            }
            ExprKind::Call { args, .. } => self.scan_args(args, out),
            ExprKind::BinaryOp(a, _, b)
            | ExprKind::Element {
                object: a,
                index: b,
            } => {
                self.scan_expr(a, out);
                self.scan_expr(b, out);
            }
            ExprKind::UnaryOp(_, a) | ExprKind::Member { object: a, .. } => self.scan_expr(a, out),
            ExprKind::List(xs) | ExprKind::New { args: xs, .. } => {
                for x in xs {
                    self.scan_expr(x, out);
                }
            }
            ExprKind::Literal(_)
            | ExprKind::Ident(_)
            | ExprKind::FieldAccess { .. }
            | ExprKind::Index { .. }
            | ExprKind::Config(_)
            | ExprKind::Range(_) => {}
        }
    }

    fn scan_args(&self, args: &[CallArg], out: &mut Vec<String>) {
        for a in args {
            match a {
                CallArg::Positional(x) | CallArg::Named { value: x, .. } => self.scan_expr(x, out),
            }
        }
    }
}

fn add(out: &mut Vec<String>, name: &str) {
    if !out.iter().any(|n| n == name) {
        out.push(String::from(name));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Virtual machine
// ═══════════════════════════════════════════════════════════════════════════════

struct Frame {
    ret_ip: usize,
    /// Index of the frame's first local.
    base: usize,
    /// Stack height to restore on return.
    stack_base: usize,
}

/// The Titan virtual machine.
pub struct TitanVM {
    chunk: Chunk,
    ip: usize,
    /// Base of the current frame's locals.
    bp: usize,
    stack: Vec<Value>,
    /// Every frame's locals, back to back; None is unbound.
    locals: Vec<Option<Value>>,
    globals: Vec<Option<Value>>,
    frames: Vec<Frame>,
    /// Value of the last statement executed.
    last: Value,
    manifolds: Vec<ManifoldWorkspace>,
}

impl TitanVM {
    pub fn new() -> Self {
        Self {
            chunk: Chunk {
                code: vec![Op::PushLast, Op::Ret],
                consts: Vec::new(),
                functions: Vec::new(),
                globals: 0,
                main_locals: 0,
            },
            ip: 0,
            bp: 0,
            stack: Vec::new(),
            locals: Vec::new(),
            globals: Vec::new(),
            frames: Vec::new(),
            last: Value::Unit,
            manifolds: Vec::new(),
        }
    }

    pub fn load(&mut self, chunk: Chunk) {
        self.chunk = chunk;
    }

    /// Run the loaded chunk from the top. The result is the value of the last
    /// top-level statement, or of a top-level `return`.
    pub fn run(&mut self) -> Result<Value, String> {
        self.globals = vec![None; self.chunk.globals];
        self.locals = vec![None; self.chunk.main_locals];
        self.frames = vec![Frame {
            ret_ip: 0,
            base: 0,
            stack_base: 0,
        }];
        self.stack.clear();
        self.manifolds.clear();
        self.last = Value::Unit;
        self.ip = 0;
        self.bp = 0;
        self.run_until_return(1)
    }

    /// Execute until the frame stack drops below `depth`, and return the value
    /// the frame at `depth` returned. Natives re-enter here to call back.
    fn run_until_return(&mut self, depth: usize) -> Result<Value, String> {
        loop {
            let op = self.chunk.code[self.ip];
            self.ip += 1;
            match op {
                Op::Num(n) => self.stack.push(Value::Num(n)),
                Op::Bool(b) => self.stack.push(Value::Bool(b)),
                Op::Unit => self.stack.push(Value::Unit),
                Op::Const(k) => self.stack.push(self.chunk.consts[k].clone()),
                Op::Dup => {
                    let v = self.stack.last().cloned().ok_or("stack underflow")?;
                    self.stack.push(v);
                }
                Op::Load(s) => {
                    let v = self.read(s).cloned().unwrap_or(Value::Unit);
                    self.stack.push(v);
                }
                Op::Store(s) => {
                    let v = self.pop()?;
                    self.write(s, v);
                }
                Op::Clear(l) => self.locals[self.bp + l] = None,
                Op::Bound(s, msg) => {
                    if self.read(s).is_none() {
                        return Err(self.text(msg));
                    }
                }
                Op::SetLast => self.last = self.pop()?,
                Op::PushLast => self.stack.push(self.last.clone()),
                Op::Bin(o) => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(binary(a, o, b)?);
                }
                Op::Neg => match self.pop()? {
                    Value::Num(n) => self.stack.push(Value::Num(-n)),
                    _ => return Err(String::from("Invalid unary operation")),
                },
                Op::Not => match self.pop()? {
                    Value::Bool(b) => self.stack.push(Value::Bool(!b)),
                    _ => return Err(String::from("Invalid unary operation")),
                },
                Op::List(n) => {
                    let items = self.pop_n(n)?;
                    self.stack.push(Value::List(items));
                }
                Op::Field(k) => {
                    let v = self.pop()?;
                    let name = self.text(k);
                    self.stack.push(field(&v, &name)?);
                }
                Op::Element => {
                    let i = self.pop()?;
                    let xs = self.pop()?;
                    self.stack.push(element(&xs, &i)?);
                }
                Op::Jump(t) => self.ip = t,
                Op::JumpIf(when, t) => match self.pop()? {
                    Value::Bool(b) => {
                        if b == when {
                            self.ip = t;
                        }
                    }
                    _ => return Err(String::from("condition must be boolean")),
                },
                Op::Fail(k) => return Err(self.text(k)),
                Op::Print => {
                    let _v = self.pop()?;
                    #[cfg(feature = "std")]
                    println!("{_v}");
                }
                Op::Callee(s, k) => match self.read(s) {
                    Some(f @ Value::Compiled(_)) => {
                        let f = f.clone();
                        self.stack.push(f);
                    }
                    Some(_) => return Err(format!("'{}' is not a function", self.text(k))),
                    None => return Err(format!("undefined function '{}'", self.text(k))),
                },
                Op::CallValue(argc) => {
                    let at = self
                        .stack
                        .len()
                        .checked_sub(argc + 1)
                        .ok_or("stack underflow")?;
                    let Value::Compiled(idx) = self.stack[at] else {
                        return Err(String::from("call of a value that is not a function"));
                    };
                    self.check_call(idx, argc)?;
                    let base = self.locals.len();
                    self.locals.extend(self.stack.drain(at + 1..).map(Some));
                    self.stack.pop();
                    self.enter(idx, base);
                }
                Op::Ret => {
                    let v = self.pop()?;
                    let f = self.frames.pop().ok_or("return with no frame")?;
                    self.locals.truncate(f.base);
                    self.stack.truncate(f.stack_base);
                    self.ip = f.ret_ip;
                    self.bp = self.frames.last().map_or(0, |f| f.base);
                    if self.frames.len() < depth {
                        return Ok(v);
                    }
                    self.stack.push(v);
                }
                Op::CallNative(id, argc, names) => {
                    let args = self.native_args(argc, names)?;
                    let v = natives::call(id, args, self)?;
                    self.stack.push(v);
                }
                Op::Method(s, k, argc, names) => {
                    let args = self.native_args(argc, names)?;
                    let name = self.text(k);
                    // Taken, not cloned, so `xs.push(x)` in a loop stays linear.
                    // ponytail: a callback reading the receiver's global while
                    // the method runs sees Unit; no method calls back today.
                    let mut receiver = self.take(s);
                    let v = natives::method(&mut receiver, &name, args, self);
                    self.write(s, receiver);
                    self.stack.push(v?);
                }
                Op::Manifold { dim, tau } => {
                    let data = match self.pop()? {
                        Value::List(xs) => xs
                            .into_iter()
                            .map(|v| match v {
                                Value::Num(n) => Ok(n),
                                _ => Err(String::from("embed data must be a numeric list")),
                            })
                            .collect::<Result<Vec<f64>, String>>()?,
                        _ => return Err(String::from("embed expects a numeric list")),
                    };
                    let m = natives::embed_manifold(self, data, dim, tau)?;
                    self.stack.push(m);
                }
                Op::SealPass(l) => {
                    let n = match self.locals[self.bp + l] {
                        Some(Value::Num(n)) => n,
                        _ => 0.0,
                    };
                    let more = n < seal::MAX_PASSES as f64;
                    if more {
                        self.locals[self.bp + l] = Some(Value::Num(n + 1.0));
                    }
                    self.stack.push(Value::Bool(more));
                }
                Op::SealStable(prev) => {
                    let now = self.pop()?;
                    let same = match &self.locals[self.bp + prev] {
                        Some(p) => seal::same(p, &now)?,
                        None => false,
                    };
                    if !same {
                        self.locals[self.bp + prev] = Some(now);
                    }
                    self.stack.push(Value::Bool(same));
                }
                Op::SealEps(l) => match self.pop()? {
                    Value::Num(eps) if eps >= 0.0 => {
                        self.locals[self.bp + l] = Some(Value::Num(eps))
                    }
                    other => {
                        return Err(format!(
                            "convergence() takes a non-negative tolerance, got {other}"
                        ))
                    }
                },
                Op::SealConverge { prev, eps } => {
                    let now = self.last.clone();
                    let eps = match self.locals[self.bp + eps] {
                        Some(Value::Num(e)) => e,
                        _ => 0.0,
                    };
                    let settled = match &self.locals[self.bp + prev] {
                        Some(p) => seal::max_change(&now, p)? <= eps,
                        None => false,
                    };
                    self.locals[self.bp + prev] = Some(now);
                    self.stack.push(Value::Bool(settled));
                }
            }
        }
    }

    fn pop(&mut self) -> Result<Value, String> {
        self.stack
            .pop()
            .ok_or_else(|| String::from("stack underflow"))
    }

    fn pop_n(&mut self, n: usize) -> Result<Vec<Value>, String> {
        let at = self.stack.len().checked_sub(n).ok_or("stack underflow")?;
        Ok(self.stack.split_off(at))
    }

    fn text(&self, k: usize) -> String {
        match &self.chunk.consts[k] {
            Value::Str(s) => s.clone(),
            other => format!("{other}"),
        }
    }

    fn native_args(&mut self, argc: usize, names: usize) -> Result<NativeArgs, String> {
        let values = self.pop_n(argc)?;
        let names = match &self.chunk.consts[names] {
            Value::List(names) => names.as_slice(),
            _ => &[][..],
        };
        let mut args = NativeArgs {
            positional: Vec::new(),
            named: Vec::new(),
        };
        for (i, v) in values.into_iter().enumerate() {
            match names.get(i) {
                Some(Value::Str(name)) => args.named.push((name.clone(), v)),
                _ => args.positional.push(v),
            }
        }
        Ok(args)
    }

    fn read(&self, s: Slot) -> Option<&Value> {
        match s {
            Slot::Global(g) => self.globals[g].as_ref(),
            Slot::Local(l) => self.locals[self.bp + l].as_ref(),
            Slot::Shadow(l, g) => self.locals[self.bp + l]
                .as_ref()
                .or(self.globals[g].as_ref()),
        }
    }

    fn write(&mut self, s: Slot, v: Value) {
        match s {
            Slot::Global(g) => self.globals[g] = Some(v),
            Slot::Local(l) | Slot::Shadow(l, _) => self.locals[self.bp + l] = Some(v),
        }
    }

    fn take(&mut self, s: Slot) -> Value {
        let v = match s {
            Slot::Global(g) => self.globals[g].take(),
            Slot::Local(l) => self.locals[self.bp + l].take(),
            Slot::Shadow(l, g) => self.locals[self.bp + l]
                .take()
                .or_else(|| self.globals[g].clone()),
        };
        v.unwrap_or(Value::Unit)
    }

    fn check_call(&self, idx: usize, argc: usize) -> Result<(), String> {
        let f = self
            .chunk
            .functions
            .get(idx)
            .ok_or("function value from another program")?;
        if argc != f.arity {
            return Err(format!(
                "function '{}' expected {} arguments, got {}",
                f.name, f.arity, argc
            ));
        }
        if self.frames.len() >= MAX_DEPTH {
            return Err(format!("call depth exceeds {MAX_DEPTH}"));
        }
        Ok(())
    }

    /// Push a frame for function `idx`, whose arguments already sit at `base`.
    fn enter(&mut self, idx: usize, base: usize) {
        let f = &self.chunk.functions[idx];
        self.locals.resize(base + f.locals, None);
        self.frames.push(Frame {
            ret_ip: self.ip,
            base,
            stack_base: self.stack.len(),
        });
        self.bp = base;
        self.ip = f.entry;
    }
}

impl Host for TitanVM {
    fn call_function(&mut self, f: &Value, args: Vec<Value>) -> Result<Value, String> {
        let Value::Compiled(idx) = *f else {
            return Err(format!(
                "expected a fn defined in the program, got {}",
                kind(f)
            ));
        };
        self.check_call(idx, args.len())?;
        let (ip, depth, height) = (self.ip, self.frames.len(), self.stack.len());
        let base = self.locals.len();
        self.locals.extend(args.into_iter().map(Some));
        self.enter(idx, base);
        let result = self.run_until_return(depth + 1);
        if result.is_err() {
            // Unwind whatever the failed call left, so the caller can go on.
            self.frames.truncate(depth);
            self.locals.truncate(base);
            self.stack.truncate(height);
            self.bp = self.frames.last().map_or(0, |f| f.base);
        }
        self.ip = ip;
        result
    }

    fn manifolds(&mut self) -> &mut Vec<ManifoldWorkspace> {
        &mut self.manifolds
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Value operations, as the interpreter defines them
// ═══════════════════════════════════════════════════════════════════════════════

fn binary(left: Value, op: BinaryOp, right: Value) -> Result<Value, String> {
    use BinaryOp::*;
    use Value::{Bool, Num, Str};
    Ok(match (left, op, right) {
        (Num(a), Add, Num(b)) => Num(a + b),
        (Num(a), Sub, Num(b)) => Num(a - b),
        (Num(a), Mul, Num(b)) => Num(a * b),
        (Num(a), Div, Num(b)) => Num(a / b),
        (Num(a), Mod, Num(b)) => Num(a % b),
        (Num(a), Eq, Num(b)) => Bool(a == b),
        (Num(a), Neq, Num(b)) => Bool(a != b),
        (Num(a), Lt, Num(b)) => Bool(a < b),
        (Num(a), Gt, Num(b)) => Bool(a > b),
        (Num(a), Le, Num(b)) => Bool(a <= b),
        (Num(a), Ge, Num(b)) => Bool(a >= b),
        (Bool(a), Eq, Bool(b)) => Bool(a == b),
        (Bool(a), Neq, Bool(b)) => Bool(a != b),
        (Bool(a), And, Bool(b)) => Bool(a && b),
        (Bool(a), Or, Bool(b)) => Bool(a || b),
        (Str(a), Eq, Str(b)) => Bool(a == b),
        (Str(a), Neq, Str(b)) => Bool(a != b),
        _ => return Err(String::from("Invalid binary operation")),
    })
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Num(_) => "number",
        Value::Bool(_) => "bool",
        Value::Str(_) => "string",
        Value::List(_) => "list",
        Value::Record(_) => "record",
        Value::Unit => "unit",
        Value::Function(_) | Value::NativeFn(_) | Value::Compiled(_) => "function",
        _ => "value",
    }
}

/// `r.field` and `xs[i]` share their implementation, and so their error
/// text, with the interpreter.
fn field(v: &Value, name: &str) -> Result<Value, String> {
    natives::field(v, name)
}

fn element(list: &Value, i: &Value) -> Result<Value, String> {
    natives::element(list, i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Number, Range};
    use crate::parser::Parser;

    fn compile(source: &str) -> Result<Chunk, CompileError> {
        let program = Parser::new(source).parse().expect("source should parse");
        Compiler::new().compile(&program)
    }

    fn run(source: &str) -> Result<Value, String> {
        let mut vm = TitanVM::new();
        vm.load(compile(source).expect("source should compile"));
        vm.run()
    }

    fn num(source: &str) -> f64 {
        match run(source).expect("vm should run") {
            Value::Num(n) => n,
            other => panic!("expected a number, got {other:?}"),
        }
    }

    fn shown(source: &str) -> String {
        format!("{}", run(source).expect("vm should run"))
    }

    #[test]
    fn test_titan_math() {
        // 5 + 3 * 2 = 11
        let mut vm = TitanVM::new();
        vm.load(Chunk {
            code: vec![
                Op::Num(5.0),
                Op::Num(3.0),
                Op::Num(2.0),
                Op::Bin(BinaryOp::Mul),
                Op::Bin(BinaryOp::Add),
                Op::SetLast,
                Op::PushLast,
                Op::Ret,
            ],
            consts: Vec::new(),
            functions: Vec::new(),
            globals: 0,
            main_locals: 0,
        });
        assert!(matches!(vm.run(), Ok(Value::Num(11.0))));
    }

    #[test]
    fn test_compiler_for_loop() {
        // let accum = 0~ for i in 0:3 { let accum = accum + i~ } accum~  => 0+1+2
        let ident = |n: &str| Expr::new(ExprKind::Ident(n.into()), Span::default());
        let stmt = |k| Statement::new(k, Span::default());
        let decl = |value| {
            stmt(StmtKind::Var(crate::ast::VarDecl {
                type_hint: None,
                name: "accum".into(),
                value,
            }))
        };
        let program = Program {
            statements: vec![
                decl(Expr::new(
                    ExprKind::Literal(Literal::Num(0.0)),
                    Span::default(),
                )),
                stmt(StmtKind::For(crate::ast::ForStmt {
                    iterator: "i".into(),
                    range: Range {
                        start: Number::Int(0),
                        end: Number::Int(3),
                    },
                    body: Block {
                        statements: vec![decl(Expr::new(
                            ExprKind::BinaryOp(
                                Box::new(ident("accum")),
                                BinaryOp::Add,
                                Box::new(ident("i")),
                            ),
                            Span::default(),
                        ))],
                    },
                })),
                stmt(StmtKind::Expr(ident("accum"))),
            ],
        };
        let mut vm = TitanVM::new();
        vm.load(Compiler::new().compile(&program).expect("should compile"));
        assert!(matches!(vm.run(), Ok(Value::Num(3.0))));
    }

    #[test]
    fn test_vm_modulo_comparison_and_boolean_ops() {
        assert!(matches!(
            run("let ok = 10 % 4 == 2 && !false~ ok~"),
            Ok(Value::Bool(true))
        ));
    }

    #[test]
    fn test_vm_assignment_if_and_while_match_interpreter_surface() {
        let n = num("let i = 0~
             if 1 < 2 { i = i + 1~ }
             while i < 3 { i = i + 1~ }
             i~");
        assert_eq!(n, 3.0);
    }

    #[test]
    fn test_vm_seal_until_condition() {
        assert_eq!(
            num("let count = 0~ seal until count >= 3 { count = count + 1~ } count~"),
            3.0
        );
    }

    #[test]
    fn test_vm_user_function_explicit_return() {
        assert_eq!(
            num("fn add(a, b) { return a + b~ } let result = add(2, 3)~ result~"),
            5.0
        );
    }

    #[test]
    fn test_vm_user_function_implicit_last_expression_return() {
        assert_eq!(num("fn one() { let x = 1~ x~ } one()~"), 1.0);
    }

    #[test]
    fn test_vm_user_function_parameters_are_call_frame_local() {
        assert_eq!(
            num("let x = 10~ fn id(x) { return x~ } let y = id(3)~ let z = x + y~ z~"),
            13.0
        );
    }

    #[test]
    fn test_vm_break_exits_while_loop() {
        let n = num("let i = 0~
             let sum = 0~
             while i < 10 {
                 i = i + 1~
                 if i == 4 { break~ }
                 sum = sum + i~
             }
             sum~");
        assert_eq!(n, 6.0);
    }

    #[test]
    fn test_vm_continue_skips_to_next_while_iteration() {
        let n = num("let i = 0~
             let sum = 0~
             while i < 5 {
                 i = i + 1~
                 if i == 3 { continue~ }
                 sum = sum + i~
             }
             sum~");
        assert_eq!(n, 12.0);
    }

    #[test]
    fn stack_is_empty_after_a_thousand_passes() {
        let mut vm = TitanVM::new();
        vm.load(
            compile(
                "fn inc(x) { return x + 1~ }
                 let s = 0~
                 for i in 0..1000 { s~ inc(i)~ s = s + 1~ }
                 seal { let t = s + 1~ }
                 s~",
            )
            .expect("should compile"),
        );
        assert!(matches!(vm.run(), Ok(Value::Num(1000.0))));
        assert!(vm.stack.is_empty(), "{} values leaked", vm.stack.len());
    }

    #[test]
    fn value_is_the_last_statement() {
        assert!(matches!(run(""), Ok(Value::Unit)));
        assert_eq!(num("fn f() { let y = 5~ } f()~"), 5.0);
        assert_eq!(num("let a = 2~ if a > 1 { let b = a * 3~ }"), 6.0);
        assert!(matches!(run("let a = 2~ if a < 1 { a~ }"), Ok(Value::Unit)));
        assert!(matches!(run("fn g() {} g()~"), Ok(Value::Unit)));
        assert!(matches!(
            run("let a = 1~ while false { a~ }"),
            Ok(Value::Unit)
        ));
        // A pass cut by `break` does not count; the last whole pass does.
        assert_eq!(
            num("let i = 0~ while true { i = i + 1~ if i == 3 { break~ } let t = i * 10~ }"),
            20.0
        );
        assert_eq!(num("print(1)~ let a = 4~ return a + 5~ let b = 6~"), 9.0);
    }

    #[test]
    fn descending_range_steps_down_and_leaves_the_iterator() {
        assert_eq!(
            shown("let s = 0~ for i in 5..0 { s = s + i~ } let r = [s, i]~"),
            "[15, 0]"
        );
        assert_eq!(num("let n = 0~ for i in 2.7 .. 0 { n = n + 1~ } n~"), 2.0);
        assert_eq!(
            shown(
                "let s = 0~
                 for i in 0..10 { if i == 2 { continue~ } if i == 5 { break~ } s = s + i~ }
                 let r = [s, i]~"
            ),
            "[8, 5]"
        );
    }

    #[test]
    fn functions_bind_when_their_fn_statement_runs() {
        assert_eq!(num("fn a() { b()~ } fn b() { let v = 42~ } a()~"), 42.0);
        // Globals are read at call time and writes stay local.
        assert_eq!(
            shown("let x = 1~ fn f() { x = x + 1~ x~ } let y = f()~ x = 5~ let r = [x, y, f()]~"),
            "[5, 2, 6]"
        );
    }

    #[test]
    fn calling_an_unbound_name_is_a_runtime_error() {
        assert_eq!(
            run("fn a() { b()~ } a()~ fn b() { let v = 1~ }").unwrap_err(),
            "undefined function 'b'"
        );
        assert_eq!(run("nope(1)~").unwrap_err(), "undefined function 'nope'");
    }

    #[test]
    fn unsupported_constructs_are_compile_errors() {
        let err = compile("regress { model: \"linear\" }~").unwrap_err();
        assert_eq!(
            err.to_string(),
            "titan cannot compile regress at line 1, column 1"
        );
        assert_eq!(compile("class P { x }").unwrap_err().construct, "class");
    }

    #[test]
    fn return_inside_a_loop_leaves_the_function() {
        assert_eq!(
            num("fn f() { for i in 0..10 { if i == 3 { return i~ } } return 99~ } f()~"),
            3.0
        );
        assert_eq!(
            num("fn g() { let k = 0~ while true { k = k + 1~ if k == 4 { return k~ } } } g()~"),
            4.0
        );
    }

    #[test]
    fn division_by_zero_and_numeric_conditions_follow_the_interpreter() {
        assert_eq!(num("let r = 1 / 0~"), f64::INFINITY);
        assert_eq!(
            run("if 1 { let b = 2~ }").unwrap_err(),
            "condition must be boolean"
        );
        assert_eq!(run("break~").unwrap_err(), "break outside loop");
    }

    #[test]
    fn seal_loops_stop_on_stable_convergence_and_the_pass_cap() {
        assert_eq!(
            num("let x = 0~ seal until stable(x) { if x < 3 { x = x + 1~ } } x~"),
            3.0
        );
        assert_eq!(
            num("let x = 1~ seal until convergence(0.001) { x = x / 2~ }"),
            1.0 / 1024.0
        );
        assert_eq!(num("let n = 0~ seal { n = n + 1~ } n~"), 1000.0);
    }

    #[test]
    fn natives_and_host_callbacks_run_on_titan() {
        assert_eq!(num("from math import sin~ sin(0)~"), 0.0);
        let mut vm = TitanVM::new();
        vm.load(compile("fn double(x) { return x * 2~ } double~").expect("should compile"));
        let f = vm.run().expect("vm should run");
        assert!(matches!(
            vm.call_function(&f, vec![Value::Num(21.0)]),
            Ok(Value::Num(42.0))
        ));
    }
}
