import Aether.Core
import Aether.Static
import Aether.Parser

namespace Aether
namespace VM

inductive Op where
  | push : Value -> Op
  | load : Nat -> Op
  | store : Nat -> Op
  | bin : BinOp -> Op
  | unary : UnOp -> Op
  | list : Nat -> Op
  | index : Op
  | field : Ident -> Op
  | method : Ident -> Nat -> Op
  | jmp : Int -> Op
  | jmpIfFalse : Int -> Op
  | halt : Op
  deriving Repr, BEq, DecidableEq

structure State where
  ip : Nat
  code : List Op
  stack : List Value
  locals : List Value
  halted : Bool
  deriving Repr, BEq, DecidableEq

def initialState (code : List Op) : State :=
  { ip := 0, code := code, stack := [], locals := [], halted := false }

def listGet? (values : List α) (idx : Nat) : Option α :=
  values[idx]?

def listSet (values : List α) (idx : Nat) (value : α) (fill : α) : List α :=
  if idx < values.length then
    values.set idx value
  else
    values ++ List.replicate (idx - values.length) fill ++ [value]

def jumpTarget (ip : Nat) (offset : Int) : Option Nat :=
  let target := (ip : Int) + offset
  if target < 0 then none else some target.toNat

def popValues : Nat -> List Value -> Option (List Value × List Value)
  | 0, stack => some ([], stack)
  | fuel + 1, value :: rest => do
      let (values, remaining) <- popValues fuel rest
      some (values ++ [value], remaining)
  | _ + 1, [] => none

def step (state : State) : Option State :=
  if state.halted then
    some state
  else
    match state.code[state.ip]? with
    | none => some { state with halted := true }
    | some Op.halt => some { state with ip := state.ip + 1, halted := true }
    | some (Op.push value) =>
        some { state with ip := state.ip + 1, stack := value :: state.stack }
    | some (Op.load idx) => do
        let value <- listGet? state.locals idx
        some { state with ip := state.ip + 1, stack := value :: state.stack }
    | some (Op.store idx) =>
        match state.stack with
        | value :: rest =>
            let locals := listSet state.locals idx value Value.unit
            some { state with ip := state.ip + 1, stack := rest, locals := locals }
        | [] => none
    | some (Op.bin op) =>
        match state.stack with
        | right :: left :: rest => do
            let value <- evalBinOp op left right
            some { state with ip := state.ip + 1, stack := value :: rest }
        | _ => none
    | some (Op.unary op) =>
        match state.stack with
        | value :: rest => do
            let result <- evalUnOp op value
            some { state with ip := state.ip + 1, stack := result :: rest }
        | [] => none
    | some (Op.list length) => do
        let (values, restStack) <- popValues length state.stack
        some { state with ip := state.ip + 1, stack := Value.list values :: restStack }
    | some Op.index =>
        match state.stack with
        | index :: target :: rest => do
            let value <- evalIndex target index
            some { state with ip := state.ip + 1, stack := value :: rest }
        | _ => none
    | some (Op.field field) =>
        match state.stack with
        | target :: rest => do
            let value <- evalField target field
            some { state with ip := state.ip + 1, stack := value :: rest }
        | [] => none
    | some (Op.method method arity) => do
        let (values, restStack) <- popValues (arity + 1) state.stack
        match values with
        | target :: args => do
            let value <- evalMethod target method args
            some { state with ip := state.ip + 1, stack := value :: restStack }
        | [] => none
    | some (Op.jmp offset) => do
        let target <- jumpTarget (state.ip + 1) offset
        some { state with ip := target }
    | some (Op.jmpIfFalse offset) =>
        match state.stack with
        | value :: rest =>
            if truthy value then
              some { state with ip := state.ip + 1, stack := rest }
            else do
              let target <- jumpTarget (state.ip + 1) offset
              some { state with ip := target, stack := rest }
        | [] => none

def runFuel : Nat -> State -> Option State
  | 0, state => some state
  | Nat.succ fuel, state =>
      if state.halted then
        some state
      else do
        let next <- step state
        runFuel fuel next

def run (fuel : Nat) (code : List Op) : Option State :=
  runFuel fuel (initialState code)

inductive FrameOp where
  | push : Value -> FrameOp
  | load : Nat -> FrameOp
  | store : Nat -> FrameOp
  | bin : BinOp -> FrameOp
  | unary : UnOp -> FrameOp
  | list : Nat -> FrameOp
  | index : FrameOp
  | field : Ident -> FrameOp
  | method : Ident -> Nat -> FrameOp
  | jmp : Int -> FrameOp
  | jmpIfFalse : Int -> FrameOp
  | call : Nat -> Nat -> FrameOp
  | ret : FrameOp
  | halt : FrameOp
  deriving Repr, BEq, DecidableEq

structure CallFrame where
  returnIp : Nat
  locals : List Value
  deriving Repr, BEq, DecidableEq

structure FrameState where
  ip : Nat
  code : List FrameOp
  stack : List Value
  locals : List Value
  frames : List CallFrame
  halted : Bool
  deriving Repr, BEq, DecidableEq

def initialFrameState (code : List FrameOp) : FrameState :=
  { ip := 0, code := code, stack := [], locals := [], frames := [], halted := false }

def popArgs : Nat -> List Value -> Option (List Value × List Value)
  | 0, stack => some ([], stack)
  | fuel + 1, value :: rest => do
      let (args, remaining) <- popArgs fuel rest
      some (args ++ [value], remaining)
  | _ + 1, [] => none

def frameStep (state : FrameState) : Option FrameState :=
  if state.halted then
    some state
  else
    match state.code[state.ip]? with
    | none => some { state with halted := true }
    | some FrameOp.halt => some { state with ip := state.ip + 1, halted := true }
    | some (FrameOp.push value) =>
        some { state with ip := state.ip + 1, stack := value :: state.stack }
    | some (FrameOp.load idx) => do
        let value <- listGet? state.locals idx
        some { state with ip := state.ip + 1, stack := value :: state.stack }
    | some (FrameOp.store idx) =>
        match state.stack with
        | value :: rest =>
            let locals := listSet state.locals idx value Value.unit
            some { state with ip := state.ip + 1, stack := rest, locals := locals }
        | [] => none
    | some (FrameOp.bin op) =>
        match state.stack with
        | right :: left :: rest => do
            let value <- evalBinOp op left right
            some { state with ip := state.ip + 1, stack := value :: rest }
        | _ => none
    | some (FrameOp.unary op) =>
        match state.stack with
        | value :: rest => do
            let result <- evalUnOp op value
            some { state with ip := state.ip + 1, stack := result :: rest }
        | [] => none
    | some (FrameOp.list length) => do
        let (values, restStack) <- popArgs length state.stack
        some { state with ip := state.ip + 1, stack := Value.list values :: restStack }
    | some FrameOp.index =>
        match state.stack with
        | index :: target :: rest => do
            let value <- evalIndex target index
            some { state with ip := state.ip + 1, stack := value :: rest }
        | _ => none
    | some (FrameOp.field field) =>
        match state.stack with
        | target :: rest => do
            let value <- evalField target field
            some { state with ip := state.ip + 1, stack := value :: rest }
        | [] => none
    | some (FrameOp.method method arity) => do
        let (values, restStack) <- popArgs (arity + 1) state.stack
        match values with
        | target :: args => do
            let value <- evalMethod target method args
            some { state with ip := state.ip + 1, stack := value :: restStack }
        | [] => none
    | some (FrameOp.jmp offset) => do
        let target <- jumpTarget (state.ip + 1) offset
        some { state with ip := target }
    | some (FrameOp.jmpIfFalse offset) =>
        match state.stack with
        | value :: rest =>
            if truthy value then
              some { state with ip := state.ip + 1, stack := rest }
            else do
              let target <- jumpTarget (state.ip + 1) offset
              some { state with ip := target, stack := rest }
        | [] => none
    | some (FrameOp.call target arity) => do
        if target < state.code.length then
          let (args, restStack) <- popArgs arity state.stack
          some
            { state with
              ip := target
              stack := restStack
              locals := args
              frames := { returnIp := state.ip + 1, locals := state.locals } :: state.frames }
        else
          none
    | some FrameOp.ret =>
        let value :=
          match state.stack with
          | result :: _ => result
          | [] => Value.unit
        match state.frames with
        | frame :: restFrames =>
            some
              { state with
                ip := frame.returnIp
                stack := value :: state.stack.drop 1
                locals := frame.locals
                frames := restFrames }
        | [] => none

def runFrameFuel : Nat -> FrameState -> Option FrameState
  | 0, state => some state
  | Nat.succ fuel, state =>
      if state.halted then
        some state
      else do
        let next <- frameStep state
        runFrameFuel fuel next

def runFrame (fuel : Nat) (code : List FrameOp) : Option FrameState :=
  runFrameFuel fuel (initialFrameState code)

abbrev SlotEnv := List (Ident × Nat)

def SlotEnv.lookup (slots : SlotEnv) (name : Ident) : Option Nat :=
  match slots with
  | [] => none
  | (key, slot) :: rest =>
      if key == name then some slot else SlotEnv.lookup rest name

def SlotEnv.resolve (slots : SlotEnv) (name : Ident) : SlotEnv × Nat :=
  match SlotEnv.lookup slots name with
  | some slot => (slots, slot)
  | none =>
      let slot := slots.length
      (slots ++ [(name, slot)], slot)

structure FrameFunction where
  name : Ident
  params : List Ident
  body : List Stmt
  deriving Repr, BEq

abbrev FrameFnEnv := List (Ident × Nat × Nat)

def FrameFnEnv.lookup (fns : FrameFnEnv) (name : Ident) : Option (Nat × Nat) :=
  match fns with
  | [] => none
  | (key, target, arity) :: rest =>
      if key == name then some (target, arity) else FrameFnEnv.lookup rest name

def collectFrameFunctions : List Stmt -> List FrameFunction
  | [] => []
  | Stmt.fnDecl name params body :: rest =>
      { name := name, params := params, body := body } :: collectFrameFunctions rest
  | Stmt.fnDeclReturn name params _ body :: rest =>
      { name := name, params := params, body := body } :: collectFrameFunctions rest
  | Stmt.fnDeclTyped name params body :: rest =>
      { name := name, params := params.map Prod.fst, body := body } :: collectFrameFunctions rest
  | Stmt.fnDeclTypedReturn name params _ body :: rest =>
      { name := name, params := params.map Prod.fst, body := body } :: collectFrameFunctions rest
  | _ :: rest => collectFrameFunctions rest

abbrev FrameParamEnv := List (Ident × List Ident)

def FrameParamEnv.lookup (params : FrameParamEnv) (name : Ident) : Option (List Ident) :=
  match params with
  | [] => none
  | (key, fnParams) :: rest =>
      if key == name then some fnParams else FrameParamEnv.lookup rest name

def collectFrameParamEnv (functions : List FrameFunction) : FrameParamEnv :=
  functions.map (fun fn => (fn.name, fn.params))

def argNodeLookup (bindings : List (Ident × Arg)) (name : Ident) : Option Arg :=
  match bindings with
  | [] => none
  | (key, arg) :: rest =>
      if key == name then some arg else argNodeLookup rest name

def bindArgNodeName
    (params : List Ident)
    (name : Ident)
    (arg : Arg)
    (bindings : List (Ident × Arg)) :
    Option (List (Ident × Arg)) :=
  if identInList name params then
    match argNodeLookup bindings name with
    | none => some ((name, arg) :: bindings)
    | some _ => none
  else
    none

def firstUnboundArgParam (params : List Ident) (bindings : List (Ident × Arg)) : Option Ident :=
  match params with
  | [] => none
  | name :: rest =>
      match argNodeLookup bindings name with
      | none => some name
      | some _ => firstUnboundArgParam rest bindings

def bindArgNodes
    (params : List Ident)
    (args : List Arg)
    (bindings : List (Ident × Arg)) :
    Option (List (Ident × Arg)) :=
  match args with
  | [] => some bindings
  | Arg.positional expr :: rest => do
      let name <- firstUnboundArgParam params bindings
      let updated <- bindArgNodeName params name (Arg.positional expr) bindings
      bindArgNodes params rest updated
  | Arg.named name expr :: rest => do
      let updated <- bindArgNodeName params name (Arg.named name expr) bindings
      bindArgNodes params rest updated

def orderedBoundArgs (params : List Ident) (bindings : List (Ident × Arg)) :
    Option (List Arg) :=
  match params with
  | [] => some []
  | name :: rest => do
      let arg <- argNodeLookup bindings name
      let args <- orderedBoundArgs rest bindings
      some (arg :: args)

def reorderCallArgsForParams (params : List Ident) (args : List Arg) : Option (List Arg) :=
  if argsAllPositional args then
    some args
  else do
    let bindings <- bindArgNodes params args []
    orderedBoundArgs params bindings

mutual
  partial def normalizeFrameArgCalls (params : FrameParamEnv) : Arg -> Option Arg
    | Arg.positional expr => do
        let normalized <- normalizeFrameExprCalls params expr
        some (Arg.positional normalized)
    | Arg.named name expr => do
        let normalized <- normalizeFrameExprCalls params expr
        some (Arg.named name normalized)

  partial def normalizeFrameArgsCalls (params : FrameParamEnv) : List Arg -> Option (List Arg)
    | [] => some []
    | arg :: rest => do
        let normalizedArg <- normalizeFrameArgCalls params arg
        let normalizedRest <- normalizeFrameArgsCalls params rest
        some (normalizedArg :: normalizedRest)

  partial def normalizeFrameExprsCalls (params : FrameParamEnv) : List Expr -> Option (List Expr)
    | [] => some []
    | expr :: rest => do
        let normalizedExpr <- normalizeFrameExprCalls params expr
        let normalizedRest <- normalizeFrameExprsCalls params rest
        some (normalizedExpr :: normalizedRest)

  partial def normalizeFrameExprCalls (params : FrameParamEnv) : Expr -> Option Expr
    | Expr.num n => some (Expr.num n)
    | Expr.float intPart fracMicros => some (Expr.float intPart fracMicros)
    | Expr.bool b => some (Expr.bool b)
    | Expr.str value => some (Expr.str value)
    | Expr.unit => some Expr.unit
    | Expr.var name => some (Expr.var name)
    | Expr.list exprs => do
        let normalized <- normalizeFrameExprsCalls params exprs
        some (Expr.list normalized)
    | Expr.unary op expr => do
        let normalized <- normalizeFrameExprCalls params expr
        some (Expr.unary op normalized)
    | Expr.binary left op right => do
        let normalizedLeft <- normalizeFrameExprCalls params left
        let normalizedRight <- normalizeFrameExprCalls params right
        some (Expr.binary normalizedLeft op normalizedRight)
    | Expr.index target index => do
        let normalizedTarget <- normalizeFrameExprCalls params target
        let normalizedIndex <- normalizeFrameExprCalls params index
        some (Expr.index normalizedTarget normalizedIndex)
    | Expr.field target field => do
        let normalizedTarget <- normalizeFrameExprCalls params target
        some (Expr.field normalizedTarget field)
    | Expr.method target method args => do
        let normalizedTarget <- normalizeFrameExprCalls params target
        let normalizedArgs <- normalizeFrameArgsCalls params args
        some (Expr.method normalizedTarget method normalizedArgs)
    | Expr.call name args => do
        let normalizedArgs <- normalizeFrameArgsCalls params args
        match FrameParamEnv.lookup params name with
        | none => some (Expr.call name normalizedArgs)
        | some fnParams => do
            let reordered <- reorderCallArgsForParams fnParams normalizedArgs
            some (Expr.call name reordered)

  partial def normalizeFrameStmtCalls (params : FrameParamEnv) : Stmt -> Option Stmt
    | Stmt.letDecl name expr => do
        let normalized <- normalizeFrameExprCalls params expr
        some (Stmt.letDecl name normalized)
    | Stmt.letDeclTyped name ty expr => do
        let normalized <- normalizeFrameExprCalls params expr
        some (Stmt.letDeclTyped name ty normalized)
    | Stmt.assign name expr => do
        let normalized <- normalizeFrameExprCalls params expr
        some (Stmt.assign name normalized)
    | Stmt.ifThenElse condition thenBranch elseBranch => do
        let normalizedCondition <- normalizeFrameExprCalls params condition
        let normalizedThen <- normalizeFrameStmtsCalls params thenBranch
        match elseBranch with
        | none => some (Stmt.ifThenElse normalizedCondition normalizedThen none)
        | some statements => do
            let normalizedElse <- normalizeFrameStmtsCalls params statements
            some (Stmt.ifThenElse normalizedCondition normalizedThen (some normalizedElse))
    | Stmt.while condition body => do
        let normalizedCondition <- normalizeFrameExprCalls params condition
        let normalizedBody <- normalizeFrameStmtsCalls params body
        some (Stmt.while normalizedCondition normalizedBody)
    | Stmt.forRange iterator start stop body => do
        let normalizedBody <- normalizeFrameStmtsCalls params body
        some (Stmt.forRange iterator start stop normalizedBody)
    | Stmt.seal condition body => do
        let normalizedCondition <-
          match condition with
          | none => some none
          | some expr => do
              let normalized <- normalizeFrameExprCalls params expr
              some (some normalized)
        let normalizedBody <- normalizeFrameStmtsCalls params body
        some (Stmt.seal normalizedCondition normalizedBody)
    | Stmt.fnDecl name fnParams body => do
        let normalizedBody <- normalizeFrameStmtsCalls params body
        some (Stmt.fnDecl name fnParams normalizedBody)
    | Stmt.fnDeclReturn name fnParams returnTy body => do
        let normalizedBody <- normalizeFrameStmtsCalls params body
        some (Stmt.fnDeclReturn name fnParams returnTy normalizedBody)
    | Stmt.fnDeclTyped name fnParams body => do
        let normalizedBody <- normalizeFrameStmtsCalls params body
        some (Stmt.fnDeclTyped name fnParams normalizedBody)
    | Stmt.fnDeclTypedReturn name fnParams returnTy body => do
        let normalizedBody <- normalizeFrameStmtsCalls params body
        some (Stmt.fnDeclTypedReturn name fnParams returnTy normalizedBody)
    | Stmt.ret none => some (Stmt.ret none)
    | Stmt.ret (some expr) => do
        let normalized <- normalizeFrameExprCalls params expr
        some (Stmt.ret (some normalized))
    | Stmt.break => some Stmt.break
    | Stmt.continue => some Stmt.continue
    | Stmt.expr expr => do
        let normalized <- normalizeFrameExprCalls params expr
        some (Stmt.expr normalized)

  partial def normalizeFrameStmtsCalls (params : FrameParamEnv) : List Stmt -> Option (List Stmt)
    | [] => some []
    | stmt :: rest => do
        let normalizedStmt <- normalizeFrameStmtCalls params stmt
        let normalizedRest <- normalizeFrameStmtsCalls params rest
        some (normalizedStmt :: normalizedRest)
end

def dummyFrameFns (functions : List FrameFunction) : FrameFnEnv :=
  functions.map (fun fn => (fn.name, 0, fn.params.length))

def paramsToSlots : List Ident -> SlotEnv :=
  let rec go : List Ident -> Nat -> List (Ident × Nat)
    | [], _ => []
    | name :: rest, idx => (name, idx) :: go rest (idx + 1)
  fun params => go params 0

structure FrameCompile where
  slots : SlotEnv
  code : List FrameOp
  breaks : List Nat
  continues : List Nat
  deriving Repr, BEq

def FrameCompile.done (slots : SlotEnv) (code : List FrameOp) : FrameCompile :=
  { slots := slots, code := code, breaks := [], continues := [] }

def shiftFrameIndices (base : Nat) (indices : List Nat) : List Nat :=
  indices.map (fun idx => idx + base)

def appendFrameCompile (left right : FrameCompile) : FrameCompile :=
  { slots := right.slots
  , code := left.code ++ right.code
  , breaks := left.breaks ++ shiftFrameIndices left.code.length right.breaks
  , continues := left.continues ++ shiftFrameIndices left.code.length right.continues
  }

def patchFrameJump (code : List FrameOp) (idx target : Nat) : List FrameOp :=
  let offset : Int := Int.ofNat target - Int.ofNat (idx + 1)
  listSet code idx (FrameOp.jmp offset) FrameOp.halt

def patchFrameJumps (code : List FrameOp) (indices : List Nat) (target : Nat) : List FrameOp :=
  indices.foldl (fun patched idx => patchFrameJump patched idx target) code

mutual
  def compileFrameExprs (slots : SlotEnv) (fns : FrameFnEnv) : List Expr -> Option (List FrameOp)
    | [] => some []
    | expr :: rest => do
        let exprCode <- compileFrameExpr slots fns expr
        let restCode <- compileFrameExprs slots fns rest
        some (exprCode ++ restCode)

  def compileFrameArg (slots : SlotEnv) (fns : FrameFnEnv) : Arg -> Option (List FrameOp)
    | Arg.positional expr => compileFrameExpr slots fns expr
    | Arg.named _ expr => compileFrameExpr slots fns expr

  def compileFrameArgs (slots : SlotEnv) (fns : FrameFnEnv) : List Arg -> Option (List FrameOp)
    | [] => some []
    | arg :: rest => do
        let exprCode <- compileFrameArg slots fns arg
        let restCode <- compileFrameArgs slots fns rest
        some (exprCode ++ restCode)

  def compileFrameExpr (slots : SlotEnv) (fns : FrameFnEnv) : Expr -> Option (List FrameOp)
    | Expr.num n => some [FrameOp.push (Value.num n)]
    | Expr.float intPart fracMicros => some [FrameOp.push (Value.float intPart fracMicros)]
    | Expr.bool b => some [FrameOp.push (Value.bool b)]
    | Expr.str value => some [FrameOp.push (Value.str value)]
    | Expr.unit => some [FrameOp.push Value.unit]
    | Expr.list exprs => do
        let code <- compileFrameExprs slots fns exprs
        some (code ++ [FrameOp.list exprs.length])
    | Expr.var name => do
        let slot <- SlotEnv.lookup slots name
        some [FrameOp.load slot]
    | Expr.unary op expr => do
        let code <- compileFrameExpr slots fns expr
        some (code ++ [FrameOp.unary op])
    | Expr.binary left op right => do
        let leftCode <- compileFrameExpr slots fns left
        let rightCode <- compileFrameExpr slots fns right
        some (leftCode ++ rightCode ++ [FrameOp.bin op])
    | Expr.index target index => do
        let targetCode <- compileFrameExpr slots fns target
        let indexCode <- compileFrameExpr slots fns index
        some (targetCode ++ indexCode ++ [FrameOp.index])
    | Expr.field target field => do
        let targetCode <- compileFrameExpr slots fns target
        some (targetCode ++ [FrameOp.field field])
    | Expr.method target method args => do
        let targetCode <- compileFrameExpr slots fns target
        let argCode <- compileFrameArgs slots fns args
        some (targetCode ++ argCode ++ [FrameOp.method method args.length])
    | Expr.call name args => do
        let (target, arity) <- FrameFnEnv.lookup fns name
        if arity == args.length then
          let argCode <- compileFrameArgs slots fns args
          some (argCode ++ [FrameOp.call target arity])
        else
          none
end

mutual
  def compileFrameStmtFlow (slots : SlotEnv) (fns : FrameFnEnv) : Stmt -> Option FrameCompile
    | Stmt.ifThenElse condition thenBranch elseBranch => do
        let conditionCode <- compileFrameExpr slots fns condition
        let thenResult <- compileFrameBlockFlow slots fns thenBranch
        match elseBranch with
        | none =>
            let thenBase := conditionCode.length + 1
            let falseOffset : Int := Int.ofNat thenResult.code.length
            some
              { slots := thenResult.slots
              , code := conditionCode ++ [FrameOp.jmpIfFalse falseOffset] ++ thenResult.code
              , breaks := shiftFrameIndices thenBase thenResult.breaks
              , continues := shiftFrameIndices thenBase thenResult.continues
              }
        | some elseStatements => do
            let elseResult <- compileFrameBlockFlow thenResult.slots fns elseStatements
            let thenBase := conditionCode.length + 1
            let elseBase := conditionCode.length + 1 + thenResult.code.length + 1
            let falseOffset : Int := Int.ofNat (thenResult.code.length + 1)
            let endOffset : Int := Int.ofNat elseResult.code.length
            some
              { slots := elseResult.slots
              , code := conditionCode
                ++ [FrameOp.jmpIfFalse falseOffset]
                ++ thenResult.code
                ++ [FrameOp.jmp endOffset]
                ++ elseResult.code
              , breaks :=
                  shiftFrameIndices thenBase thenResult.breaks
                    ++ shiftFrameIndices elseBase elseResult.breaks
              , continues :=
                  shiftFrameIndices thenBase thenResult.continues
                    ++ shiftFrameIndices elseBase elseResult.continues
              }
    | Stmt.while condition body => do
        let conditionCode <- compileFrameExpr slots fns condition
        let bodyResult <- compileFrameBlockFlow slots fns body
        let bodyBase := conditionCode.length + 1
        let exitOffset : Int := Int.ofNat (bodyResult.code.length + 1)
        let backOffset : Int := -Int.ofNat (conditionCode.length + bodyResult.code.length + 2)
        let unpatched :=
          conditionCode
          ++ [FrameOp.jmpIfFalse exitOffset]
          ++ bodyResult.code
          ++ [FrameOp.jmp backOffset]
        let exitTarget := unpatched.length
        let breakSites := shiftFrameIndices bodyBase bodyResult.breaks
        let continueSites := shiftFrameIndices bodyBase bodyResult.continues
        let patchedBreaks := patchFrameJumps unpatched breakSites exitTarget
        let patchedContinues := patchFrameJumps patchedBreaks continueSites 0
        some (FrameCompile.done bodyResult.slots patchedContinues)
    | Stmt.forRange iterator start stop body => do
        let (iteratorSlots, iteratorSlot) := SlotEnv.resolve slots iterator
        let bodyResult <- compileFrameBlockFlow iteratorSlots fns body
        let initCode := [FrameOp.push (Value.num start), FrameOp.store iteratorSlot]
        let conditionCode :=
          [ FrameOp.load iteratorSlot
          , FrameOp.push (Value.num stop)
          , FrameOp.bin BinOp.lt
          ]
        let incrementCode :=
          [ FrameOp.load iteratorSlot
          , FrameOp.push (Value.num 1)
          , FrameOp.bin BinOp.add
          , FrameOp.store iteratorSlot
          ]
        let exitOffset : Int := Int.ofNat (bodyResult.code.length + incrementCode.length + 1)
        let backOffset : Int :=
          -Int.ofNat (conditionCode.length + bodyResult.code.length + incrementCode.length + 2)
        let unpatched :=
          initCode
          ++ conditionCode
          ++ [FrameOp.jmpIfFalse exitOffset]
          ++ bodyResult.code
          ++ incrementCode
          ++ [FrameOp.jmp backOffset]
        let bodyBase := initCode.length + conditionCode.length + 1
        let continueTarget := bodyBase + bodyResult.code.length
        let exitTarget := unpatched.length
        let breakSites := shiftFrameIndices bodyBase bodyResult.breaks
        let continueSites := shiftFrameIndices bodyBase bodyResult.continues
        let patchedBreaks := patchFrameJumps unpatched breakSites exitTarget
        let patchedContinues := patchFrameJumps patchedBreaks continueSites continueTarget
        some (FrameCompile.done bodyResult.slots patchedContinues)
    | Stmt.seal none body => do
        let bodyResult <- compileFrameBlockFlow slots fns body
        let backOffset : Int := -Int.ofNat (bodyResult.code.length + 1)
        let unpatched := bodyResult.code ++ [FrameOp.jmp backOffset]
        let patchedBreaks := patchFrameJumps unpatched bodyResult.breaks unpatched.length
        let patchedContinues := patchFrameJumps patchedBreaks bodyResult.continues 0
        some (FrameCompile.done bodyResult.slots patchedContinues)
    | Stmt.seal (some condition) body => do
        let conditionCode <- compileFrameExpr slots fns condition
        let bodyResult <- compileFrameBlockFlow slots fns body
        let exitOffset : Int := Int.ofNat (bodyResult.code.length + 1)
        let backOffset : Int := -Int.ofNat (conditionCode.length + bodyResult.code.length + 3)
        let unpatched :=
          conditionCode
          ++ [FrameOp.unary UnOp.not, FrameOp.jmpIfFalse exitOffset]
          ++ bodyResult.code
          ++ [FrameOp.jmp backOffset]
        let bodyBase := conditionCode.length + 2
        let breakSites := shiftFrameIndices bodyBase bodyResult.breaks
        let continueSites := shiftFrameIndices bodyBase bodyResult.continues
        let patchedBreaks := patchFrameJumps unpatched breakSites unpatched.length
        let patchedContinues := patchFrameJumps patchedBreaks continueSites 0
        some (FrameCompile.done bodyResult.slots patchedContinues)
    | Stmt.letDecl name expr => do
        let code <- compileFrameExpr slots fns expr
        let (updatedSlots, slot) := SlotEnv.resolve slots name
        some (FrameCompile.done updatedSlots (code ++ [FrameOp.store slot]))
    | Stmt.letDeclTyped name _ expr => do
        let code <- compileFrameExpr slots fns expr
        let (updatedSlots, slot) := SlotEnv.resolve slots name
        some (FrameCompile.done updatedSlots (code ++ [FrameOp.store slot]))
    | Stmt.assign name expr => do
        let slot <- SlotEnv.lookup slots name
        let code <- compileFrameExpr slots fns expr
        some (FrameCompile.done slots (code ++ [FrameOp.store slot]))
    | Stmt.expr expr => do
        let code <- compileFrameExpr slots fns expr
        some (FrameCompile.done slots code)
    | Stmt.ret (some expr) => do
        let code <- compileFrameExpr slots fns expr
        some (FrameCompile.done slots (code ++ [FrameOp.ret]))
    | Stmt.ret none => some (FrameCompile.done slots [FrameOp.ret])
    | Stmt.break => some { slots := slots, code := [FrameOp.jmp 0], breaks := [0], continues := [] }
    | Stmt.continue => some { slots := slots, code := [FrameOp.jmp 0], breaks := [], continues := [0] }
    | _ => none

  def compileFrameBlockFlow (slots : SlotEnv) (fns : FrameFnEnv) : List Stmt -> Option FrameCompile
    | [] => some (FrameCompile.done slots [])
    | stmt :: rest => do
        let stmtResult <- compileFrameStmtFlow slots fns stmt
        let restResult <- compileFrameBlockFlow stmtResult.slots fns rest
        some (appendFrameCompile stmtResult restResult)
end

def compileFrameStmt (slots : SlotEnv) (fns : FrameFnEnv) (stmt : Stmt) : Option (SlotEnv × List FrameOp) := do
  let result <- compileFrameStmtFlow slots fns stmt
  if result.breaks == [] && result.continues == [] then
    some (result.slots, result.code)
  else
    none

def compileFrameBlock (slots : SlotEnv) (fns : FrameFnEnv) (stmts : List Stmt) :
    Option (SlotEnv × List FrameOp) := do
  let result <- compileFrameBlockFlow slots fns stmts
  if result.breaks == [] && result.continues == [] then
    some (result.slots, result.code)
  else
    none

def compileFrameFunction (fns : FrameFnEnv) (fn : FrameFunction) : Option (List FrameOp) := do
  let (_, bodyCode) <- compileFrameBlock (paramsToSlots fn.params) fns fn.body
  some (bodyCode ++ [FrameOp.ret])

def compileFrameFunctions (fns : FrameFnEnv) : List FrameFunction -> Option (List FrameOp)
  | [] => some []
  | fn :: rest => do
      let fnCode <- compileFrameFunction fns fn
      let restCode <- compileFrameFunctions fns rest
      some (fnCode ++ restCode)

def frameTargetsFrom (start : Nat) : List FrameFunction -> FrameFnEnv
  | [] => []
  | fn :: rest =>
      let dummyFns := dummyFrameFns (fn :: rest)
      match compileFrameFunction dummyFns fn with
      | some fnCode =>
          (fn.name, start, fn.params.length) :: frameTargetsFrom (start + fnCode.length) rest
      | none =>
          (fn.name, start, fn.params.length) :: frameTargetsFrom start rest

def compileFrameMain (slots : SlotEnv) (fns : FrameFnEnv) : List Stmt -> Option (SlotEnv × List FrameOp)
  | [] => some (slots, [])
  | Stmt.fnDecl _ _ _ :: rest => compileFrameMain slots fns rest
  | Stmt.fnDeclReturn _ _ _ _ :: rest => compileFrameMain slots fns rest
  | Stmt.fnDeclTyped _ _ _ :: rest => compileFrameMain slots fns rest
  | Stmt.fnDeclTypedReturn _ _ _ _ :: rest => compileFrameMain slots fns rest
  | stmt :: rest => do
      let (slotsAfterStmt, stmtCode) <- compileFrameStmt slots fns stmt
      let (slotsAfterRest, restCode) <- compileFrameMain slotsAfterStmt fns rest
      some (slotsAfterRest, stmtCode ++ restCode)

def compileFrameProgram (stmts : List Stmt) : Option (SlotEnv × FrameFnEnv × List FrameOp) := do
  let initialFunctions := collectFrameFunctions stmts
  let params := collectFrameParamEnv initialFunctions
  let stmts <- normalizeFrameStmtsCalls params stmts
  let functions := collectFrameFunctions stmts
  let dummyFns := dummyFrameFns functions
  let (_, dummyMainCode) <- compileFrameMain [] dummyFns stmts
  let fns := frameTargetsFrom (dummyMainCode.length + 1) functions
  let (mainSlots, mainCode) <- compileFrameMain [] fns stmts
  let functionCode <- compileFrameFunctions fns functions
  some (mainSlots, fns, mainCode ++ [FrameOp.halt] ++ functionCode)

def compileCheckedFrameProgram (stmts : List Stmt) : Option (SlotEnv × FrameFnEnv × List FrameOp) := do
  match Static.checkProgramDetailed stmts with
  | Except.ok _ => compileFrameProgram stmts
  | Except.error _ => none

theorem compileCheckedFrameProgram_static_ok
    {stmts : List Stmt}
    {result : SlotEnv × FrameFnEnv × List FrameOp}
    (h : compileCheckedFrameProgram stmts = some result) :
    ∃ checked, Static.checkProgramDetailed stmts = Except.ok checked := by
  unfold compileCheckedFrameProgram at h
  cases hs : Static.checkProgramDetailed stmts with
  | error found =>
      simp [hs] at h
  | ok checked =>
      exact ⟨checked, rfl⟩

def runCompiledFrameProgram (fuel : Nat) (stmts : List Stmt) : Option (SlotEnv × FrameFnEnv × FrameState) := do
  let (slots, fns, code) <- compileFrameProgram stmts
  let state <- runFrame fuel code
  some (slots, fns, state)

def runCheckedFrameProgram (fuel : Nat) (stmts : List Stmt) :
    Option (SlotEnv × FrameFnEnv × FrameState) := do
  let (slots, fns, code) <- compileCheckedFrameProgram stmts
  let state <- runFrame fuel code
  some (slots, fns, state)

def compiledFrameLocal? (fuel : Nat) (stmts : List Stmt) (slot : Nat) : Option Value := do
  let (_, _, state) <- runCompiledFrameProgram fuel stmts
  listGet? state.locals slot

def checkedFrameLocal? (fuel : Nat) (stmts : List Stmt) (slot : Nat) : Option Value := do
  let (_, _, state) <- runCheckedFrameProgram fuel stmts
  listGet? state.locals slot

def compileCheckedFrameSource (source : String) : Option (SlotEnv × FrameFnEnv × List FrameOp) := do
  let stmts <- Parser.parseProgram source
  compileCheckedFrameProgram stmts

def runCheckedFrameSource (fuel : Nat) (source : String) :
    Option (SlotEnv × FrameFnEnv × FrameState) := do
  let stmts <- Parser.parseProgram source
  runCheckedFrameProgram fuel stmts

def checkedFrameSourceLocal? (fuel : Nat) (source : String) (slot : Nat) : Option Value := do
  let (_, _, state) <- runCheckedFrameSource fuel source
  listGet? state.locals slot

mutual
  def compileExprList : List Expr -> Option (List Op)
    | [] => some []
    | expr :: rest => do
        let exprCode <- compileExpr expr
        let restCode <- compileExprList rest
        some (exprCode ++ restCode)

  def compileArg (arg : Arg) : Option (List Op) :=
    match arg with
    | Arg.positional expr => compileExpr expr
    | Arg.named _ expr => compileExpr expr

  def compileArgList : List Arg -> Option (List Op)
    | [] => some []
    | arg :: rest => do
        let argCode <- compileArg arg
        let restCode <- compileArgList rest
        some (argCode ++ restCode)

  def compileExpr : Expr -> Option (List Op)
    | Expr.num n => some [Op.push (Value.num n)]
    | Expr.float intPart fracMicros => some [Op.push (Value.float intPart fracMicros)]
    | Expr.bool b => some [Op.push (Value.bool b)]
    | Expr.str value => some [Op.push (Value.str value)]
    | Expr.unit => some [Op.push Value.unit]
    | Expr.list exprs => do
        let code <- compileExprList exprs
        some (code ++ [Op.list exprs.length])
    | Expr.unary op expr => do
        let code <- compileExpr expr
        some (code ++ [Op.unary op])
    | Expr.binary left op right => do
        let leftCode <- compileExpr left
        let rightCode <- compileExpr right
        some (leftCode ++ rightCode ++ [Op.bin op])
    | Expr.index target index => do
        let targetCode <- compileExpr target
        let indexCode <- compileExpr index
        some (targetCode ++ indexCode ++ [Op.index])
    | Expr.field target field => do
        let targetCode <- compileExpr target
        some (targetCode ++ [Op.field field])
    | Expr.method target method args => do
        let targetCode <- compileExpr target
        let argCode <- compileArgList args
        some (targetCode ++ argCode ++ [Op.method method args.length])
    | Expr.var _ => none
    | Expr.call _ _ => none
end

mutual
  def compileExprListWithSlots (slots : SlotEnv) : List Expr -> Option (List Op)
    | [] => some []
    | expr :: rest => do
        let exprCode <- compileExprWithSlots slots expr
        let restCode <- compileExprListWithSlots slots rest
        some (exprCode ++ restCode)

  def compileArgWithSlots (slots : SlotEnv) (arg : Arg) : Option (List Op) :=
    match arg with
    | Arg.positional expr => compileExprWithSlots slots expr
    | Arg.named _ expr => compileExprWithSlots slots expr

  def compileArgListWithSlots (slots : SlotEnv) : List Arg -> Option (List Op)
    | [] => some []
    | arg :: rest => do
        let argCode <- compileArgWithSlots slots arg
        let restCode <- compileArgListWithSlots slots rest
        some (argCode ++ restCode)

  def compileExprWithSlots (slots : SlotEnv) : Expr -> Option (List Op)
    | Expr.num n => some [Op.push (Value.num n)]
    | Expr.float intPart fracMicros => some [Op.push (Value.float intPart fracMicros)]
    | Expr.bool b => some [Op.push (Value.bool b)]
    | Expr.str value => some [Op.push (Value.str value)]
    | Expr.unit => some [Op.push Value.unit]
    | Expr.list exprs => do
        let code <- compileExprListWithSlots slots exprs
        some (code ++ [Op.list exprs.length])
    | Expr.var name => do
        let slot <- SlotEnv.lookup slots name
        some [Op.load slot]
    | Expr.unary op expr => do
        let code <- compileExprWithSlots slots expr
        some (code ++ [Op.unary op])
    | Expr.binary left op right => do
        let leftCode <- compileExprWithSlots slots left
        let rightCode <- compileExprWithSlots slots right
        some (leftCode ++ rightCode ++ [Op.bin op])
    | Expr.index target index => do
        let targetCode <- compileExprWithSlots slots target
        let indexCode <- compileExprWithSlots slots index
        some (targetCode ++ indexCode ++ [Op.index])
    | Expr.field target field => do
        let targetCode <- compileExprWithSlots slots target
        some (targetCode ++ [Op.field field])
    | Expr.method target method args => do
        let targetCode <- compileExprWithSlots slots target
        let argCode <- compileArgListWithSlots slots args
        some (targetCode ++ argCode ++ [Op.method method args.length])
    | Expr.call _ _ => none
end

def runCompiledExpr (fuel : Nat) (expr : Expr) : Option State := do
  let code <- compileExpr expr
  run fuel (code ++ [Op.halt])

def compiledExprValue? (fuel : Nat) (expr : Expr) : Option Value := do
  let state <- runCompiledExpr fuel expr
  match state.stack with
  | value :: _ => some value
  | [] => none

def runCompiledExprWithSlots
    (fuel : Nat)
    (slots : SlotEnv)
    (locals : List Value)
    (expr : Expr) : Option State := do
  let code <- compileExprWithSlots slots expr
  runFuel fuel { ip := 0, code := code ++ [Op.halt], stack := [], locals := locals, halted := false }

def compiledExprWithSlotsValue?
    (fuel : Nat)
    (slots : SlotEnv)
    (locals : List Value)
    (expr : Expr) : Option Value := do
  let state <- runCompiledExprWithSlots fuel slots locals expr
  match state.stack with
  | value :: _ => some value
  | [] => none

def compileStmtWithSlots (slots : SlotEnv) : Stmt -> Option (SlotEnv × List Op)
  | Stmt.letDecl name expr => do
      let code <- compileExprWithSlots slots expr
      let (updatedSlots, slot) := SlotEnv.resolve slots name
      some (updatedSlots, code ++ [Op.store slot])
  | Stmt.assign name expr => do
      let slot <- SlotEnv.lookup slots name
      let code <- compileExprWithSlots slots expr
      some (slots, code ++ [Op.store slot])
  | Stmt.expr expr => do
      let code <- compileExprWithSlots slots expr
      some (slots, code)
  | _ => none

def runCompiledStmtWithSlots
    (fuel : Nat)
    (slots : SlotEnv)
    (locals : List Value)
    (stmt : Stmt) : Option (SlotEnv × State) := do
  let (updatedSlots, code) <- compileStmtWithSlots slots stmt
  let state <- runFuel fuel
    { ip := 0, code := code ++ [Op.halt], stack := [], locals := locals, halted := false }
  some (updatedSlots, state)

def compileBlockWithSlots (slots : SlotEnv) : List Stmt -> Option (SlotEnv × List Op)
  | [] => some (slots, [])
  | stmt :: rest => do
      let (slotsAfterStmt, stmtCode) <- compileStmtWithSlots slots stmt
      let (slotsAfterRest, restCode) <- compileBlockWithSlots slotsAfterStmt rest
      some (slotsAfterRest, stmtCode ++ restCode)

def runCompiledBlockWithSlots
    (fuel : Nat)
    (slots : SlotEnv)
    (locals : List Value)
    (stmts : List Stmt) : Option (SlotEnv × State) := do
  let (updatedSlots, code) <- compileBlockWithSlots slots stmts
  let state <- runFuel fuel
    { ip := 0, code := code ++ [Op.halt], stack := [], locals := locals, halted := false }
  some (updatedSlots, state)

mutual
  def compileStmtWithBranches (slots : SlotEnv) : Stmt -> Option (SlotEnv × List Op)
    | Stmt.ifThenElse condition thenBranch elseBranch => do
        let conditionCode <- compileExprWithSlots slots condition
        let (thenSlots, thenCode) <- compileBlockWithBranches slots thenBranch
        match elseBranch with
        | none =>
            let falseOffset : Int := Int.ofNat thenCode.length
            some (thenSlots, conditionCode ++ [Op.jmpIfFalse falseOffset] ++ thenCode)
        | some elseStatements => do
            let (elseSlots, elseCode) <- compileBlockWithBranches thenSlots elseStatements
            let falseOffset : Int := Int.ofNat (thenCode.length + 1)
            let endOffset : Int := Int.ofNat elseCode.length
            some
              ( elseSlots
              , conditionCode
                ++ [Op.jmpIfFalse falseOffset]
                ++ thenCode
                ++ [Op.jmp endOffset]
                ++ elseCode
              )
    | Stmt.while condition body => do
        let conditionCode <- compileExprWithSlots slots condition
        let (bodySlots, bodyCode) <- compileBlockWithBranches slots body
        let exitOffset : Int := Int.ofNat (bodyCode.length + 1)
        let backOffset : Int := -Int.ofNat (conditionCode.length + bodyCode.length + 2)
        some
          ( bodySlots
          , conditionCode
            ++ [Op.jmpIfFalse exitOffset]
            ++ bodyCode
            ++ [Op.jmp backOffset]
          )
    | Stmt.forRange iterator start stop body => do
        let (iteratorSlots, iteratorSlot) := SlotEnv.resolve slots iterator
        let (bodySlots, bodyCode) <- compileBlockWithBranches iteratorSlots body
        let initCode := [Op.push (Value.num start), Op.store iteratorSlot]
        let conditionCode :=
          [ Op.load iteratorSlot
          , Op.push (Value.num stop)
          , Op.bin BinOp.lt
          ]
        let incrementCode :=
          [ Op.load iteratorSlot
          , Op.push (Value.num 1)
          , Op.bin BinOp.add
          , Op.store iteratorSlot
          ]
        let exitOffset : Int := Int.ofNat (bodyCode.length + incrementCode.length + 1)
        let backOffset : Int :=
          -Int.ofNat (conditionCode.length + bodyCode.length + incrementCode.length + 2)
        some
          ( bodySlots
          , initCode
            ++ conditionCode
            ++ [Op.jmpIfFalse exitOffset]
            ++ bodyCode
            ++ incrementCode
            ++ [Op.jmp backOffset]
          )
    | Stmt.seal none body => do
        let (bodySlots, bodyCode) <- compileBlockWithBranches slots body
        let backOffset : Int := -Int.ofNat (bodyCode.length + 1)
        some (bodySlots, bodyCode ++ [Op.jmp backOffset])
    | Stmt.seal (some condition) body => do
        let conditionCode <- compileExprWithSlots slots condition
        let (bodySlots, bodyCode) <- compileBlockWithBranches slots body
        let exitOffset : Int := Int.ofNat (bodyCode.length + 1)
        let backOffset : Int := -Int.ofNat (conditionCode.length + bodyCode.length + 3)
        some
          ( bodySlots
          , conditionCode
            ++ [Op.unary UnOp.not, Op.jmpIfFalse exitOffset]
            ++ bodyCode
            ++ [Op.jmp backOffset]
          )
    | stmt => compileStmtWithSlots slots stmt

  def compileBlockWithBranches (slots : SlotEnv) : List Stmt -> Option (SlotEnv × List Op)
    | [] => some (slots, [])
    | stmt :: rest => do
        let (slotsAfterStmt, stmtCode) <- compileStmtWithBranches slots stmt
        let (slotsAfterRest, restCode) <- compileBlockWithBranches slotsAfterStmt rest
        some (slotsAfterRest, stmtCode ++ restCode)
end

def runCompiledBlockWithBranches
    (fuel : Nat)
    (slots : SlotEnv)
    (locals : List Value)
    (stmts : List Stmt) : Option (SlotEnv × State) := do
  let (updatedSlots, code) <- compileBlockWithBranches slots stmts
  let state <- runFuel fuel
    { ip := 0, code := code ++ [Op.halt], stack := [], locals := locals, halted := false }
  some (updatedSlots, state)

example :
    run 4 [Op.push (Value.num 5), Op.push (Value.num 3), Op.bin BinOp.add, Op.halt]
      = some
        { ip := 4
          code := [Op.push (Value.num 5), Op.push (Value.num 3), Op.bin BinOp.add, Op.halt]
          stack := [Value.num 8]
          locals := []
          halted := true } := by
  native_decide

example :
    run 5 [Op.push (Value.num 7), Op.store 0, Op.load 0, Op.unary UnOp.neg, Op.halt]
      = some
        { ip := 5
          code := [Op.push (Value.num 7), Op.store 0, Op.load 0, Op.unary UnOp.neg, Op.halt]
          stack := [Value.num (-7)]
          locals := [Value.num 7]
          halted := true } := by
  native_decide

example :
    run 5 [Op.push (Value.bool false), Op.jmpIfFalse 1, Op.push (Value.num 99), Op.push (Value.num 1), Op.halt]
      = some
        { ip := 5
          code := [Op.push (Value.bool false), Op.jmpIfFalse 1, Op.push (Value.num 99), Op.push (Value.num 1), Op.halt]
          stack := [Value.num 1]
          locals := []
          halted := true } := by
  native_decide

example :
    runFrame 20
      [ FrameOp.push (Value.num 2)
      , FrameOp.push (Value.num 3)
      , FrameOp.call 4 2
      , FrameOp.halt
      , FrameOp.load 0
      , FrameOp.load 1
      , FrameOp.bin BinOp.add
      , FrameOp.ret
      ]
      =
      some
        { ip := 4
          code :=
            [ FrameOp.push (Value.num 2)
            , FrameOp.push (Value.num 3)
            , FrameOp.call 4 2
            , FrameOp.halt
            , FrameOp.load 0
            , FrameOp.load 1
            , FrameOp.bin BinOp.add
            , FrameOp.ret
            ]
          stack := [Value.num 5]
          locals := []
          frames := []
          halted := true } := by
  native_decide

example :
    runFrame 10
      [ FrameOp.call 2 0
      , FrameOp.halt
      , FrameOp.ret
      ]
      =
      some
        { ip := 2
          code :=
            [ FrameOp.call 2 0
            , FrameOp.halt
            , FrameOp.ret
            ]
          stack := [Value.unit]
          locals := []
          frames := []
          halted := true } := by
  native_decide

example :
    runFrameFuel 20
      { ip := 0
        code :=
          [ FrameOp.push (Value.num 3)
          , FrameOp.call 3 1
          , FrameOp.halt
          , FrameOp.load 0
          , FrameOp.ret
          ]
        stack := []
        locals := [Value.num 10]
        frames := []
        halted := false }
      =
      some
        { ip := 3
          code :=
            [ FrameOp.push (Value.num 3)
            , FrameOp.call 3 1
            , FrameOp.halt
            , FrameOp.load 0
            , FrameOp.ret
            ]
          stack := [Value.num 3]
          locals := [Value.num 10]
          frames := []
          halted := true } := by
  native_decide

example :
    compileFrameProgram
      [ Stmt.fnDecl
          "add"
          ["a", "b"]
          [Stmt.ret (some (Expr.binary (Expr.var "a") BinOp.add (Expr.var "b")))]
      , Stmt.letDecl "y" (Expr.call "add" [Expr.num 2, Expr.num 3])
      ]
      =
      some
        ( [("y", 0)]
        , [("add", 5, 2)]
        , [ FrameOp.push (Value.num 2)
          , FrameOp.push (Value.num 3)
          , FrameOp.call 5 2
          , FrameOp.store 0
          , FrameOp.halt
          , FrameOp.load 0
          , FrameOp.load 1
          , FrameOp.bin BinOp.add
          , FrameOp.ret
          , FrameOp.ret
          ]) := by
  native_decide

example :
    runCompiledFrameProgram
      30
      [ Stmt.fnDecl
          "add"
          ["a", "b"]
          [Stmt.ret (some (Expr.binary (Expr.var "a") BinOp.add (Expr.var "b")))]
      , Stmt.letDecl "y" (Expr.call "add" [Expr.num 2, Expr.num 3])
      ]
      =
      some
        ( [("y", 0)]
        , [("add", 5, 2)]
        , { ip := 5
            code :=
              [ FrameOp.push (Value.num 2)
              , FrameOp.push (Value.num 3)
              , FrameOp.call 5 2
              , FrameOp.store 0
              , FrameOp.halt
              , FrameOp.load 0
              , FrameOp.load 1
              , FrameOp.bin BinOp.add
              , FrameOp.ret
              , FrameOp.ret
              ]
            stack := []
            locals := [Value.num 5]
            frames := []
            halted := true }) := by
  native_decide

example :
    checkedFrameLocal?
      30
      [ Stmt.fnDecl
          "add"
          ["a", "b"]
          [Stmt.ret (some (Expr.binary (Expr.var "a") BinOp.add (Expr.var "b")))]
      , Stmt.letDecl "y" (Expr.call "add" [Expr.num 2, Expr.num 3])
      ]
      0
      =
      some (Value.num 5) := by
  native_decide

example :
    compileFrameProgram
      [Stmt.letDecl "bad" (Expr.binary (Expr.num 2) BinOp.add (Expr.bool true))]
      ≠
      none := by
  native_decide

example :
    compileCheckedFrameProgram
      [Stmt.letDecl "bad" (Expr.binary (Expr.num 2) BinOp.add (Expr.bool true))]
      =
      none := by
  native_decide

example :
    compileCheckedFrameProgram
      [ Stmt.fnDecl
          "id"
          ["x"]
          [Stmt.ret (some (Expr.var "x"))]
      , Stmt.letDecl "bad" (Expr.call "id" [Expr.num 1, Expr.num 2])
      ]
      =
      none := by
  native_decide

example :
    compileCheckedFrameProgram
      [ Stmt.fnDecl "dup" [] [Stmt.ret (some (Expr.num 1))]
      , Stmt.fnDecl "dup" [] [Stmt.ret (some (Expr.num 2))]
      ]
      =
      none := by
  native_decide

example :
    checkedFrameSourceLocal?
      40
      "fn add(a, b) { return a + b }~let y = add(2, 3)"
      0
      =
      some (Value.num 5) := by
  native_decide

example :
    checkedFrameSourceLocal?
      80
      "let x = 0~for i in -2..2 { x = x + i }"
      0
      =
      some (Value.num (-2)) := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let xs = []~let empty = xs.is_empty()"
      1
      =
      some (Value.bool true) := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let name = \"open\"~let empty = name.is_empty()"
      1
      =
      some (Value.bool false) := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let xs = [7, 9]~let first = xs.first()"
      1
      =
      some (Value.num 7) := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let xs = [7, 9]~let last = xs.last()"
      1
      =
      some (Value.num 9) := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let xs = [7, 9]~let picked = xs.at(1)"
      1
      =
      some (Value.num 9) := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let xs = [7, 9]~let has = xs.contains(9)"
      1
      =
      some (Value.bool true) := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let xs = [7, 9]~let rest = xs.tail()"
      1
      =
      some (Value.list [Value.num 9]) := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let xs = [7, 9]~let first = xs.take(1)"
      1
      =
      some (Value.list [Value.num 7]) := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let xs = [7, 9]~let rest = xs.drop(1)"
      1
      =
      some (Value.list [Value.num 9]) := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let xs = [7, 9]~let reversed = xs.reverse()"
      1
      =
      some (Value.list [Value.num 9, Value.num 7]) := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let xs = [7]~let more = xs.append(9)"
      1
      =
      some (Value.list [Value.num 7, Value.num 9]) := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let xs = [9]~let more = xs.prepend(7)"
      1
      =
      some (Value.list [Value.num 7, Value.num 9]) := by
  native_decide

example :
    checkedFrameSourceLocal?
      40
      "let xs = [\"a\", \"b\"]~let text = xs.join(\",\")"
      1
      =
      some (Value.str "a,b") := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let xs = [7]~let more = xs.concat([9])"
      1
      =
      some (Value.list [Value.num 7, Value.num 9]) := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let label = \"open\"~let ch = label.at(1)"
      1
      =
      some (Value.str "p") := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let label = \"open\"~let has = label.contains(\"pe\")"
      1
      =
      some (Value.bool true) := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let label = \"open\"~let has = label.starts_with(\"op\")"
      1
      =
      some (Value.bool true) := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let label = \"open\"~let has = label.ends_with(\"en\")"
      1
      =
      some (Value.bool true) := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let label = \"open\"~let reversed = label.reverse()"
      1
      =
      some (Value.str "nepo") := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let label = \"open\"~let first = label.first()"
      1
      =
      some (Value.str "o") := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let label = \"open\"~let last = label.last()"
      1
      =
      some (Value.str "n") := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let label = \"open\"~let rest = label.tail()"
      1
      =
      some (Value.str "pen") := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let label = \"open\"~let prefix = label.take(2)"
      1
      =
      some (Value.str "op") := by
  native_decide

example :
    checkedFrameSourceLocal?
      30
      "let label = \"open\"~let suffix = label.drop(2)"
      1
      =
      some (Value.str "en") := by
  native_decide

example :
    compileCheckedFrameSource "let bad = 2 + true"
      =
      none := by
  native_decide

example :
    compileCheckedFrameSource "fn id(x) { return x }~let bad = id(1, 2)"
      =
      none := by
  native_decide

example :
    compileCheckedFrameSource "fn bad(x, x) { return x }"
      =
      none := by
  native_decide

example :
    compileCheckedFrameSource "let x = \"unterminated"
      =
      none := by
  native_decide

example :
    runCompiledFrameProgram
      40
      [ Stmt.letDecl "x" (Expr.num 10)
      , Stmt.fnDecl "id" ["x"] [Stmt.ret (some (Expr.var "x"))]
      , Stmt.letDecl "y" (Expr.call "id" [Expr.num 3])
      ]
      =
      some
        ( [("x", 0), ("y", 1)]
        , [("id", 6, 1)]
        , { ip := 6
            code :=
              [ FrameOp.push (Value.num 10)
              , FrameOp.store 0
              , FrameOp.push (Value.num 3)
              , FrameOp.call 6 1
              , FrameOp.store 1
              , FrameOp.halt
              , FrameOp.load 0
              , FrameOp.ret
              , FrameOp.ret
              ]
            stack := []
            locals := [Value.num 10, Value.num 3]
            frames := []
            halted := true }) := by
  native_decide

example :
    runCompiledFrameProgram
      40
      [ Stmt.fnDecl
          "choose"
          ["x"]
          [ Stmt.ifThenElse
              (Expr.binary (Expr.var "x") BinOp.lt (Expr.num 3))
              [Stmt.ret (some (Expr.num 1))]
              (some [Stmt.ret (some (Expr.num 2))])
          ]
      , Stmt.letDecl "y" (Expr.call "choose" [Expr.num 4])
      ]
      =
      some
        ( [("y", 0)]
        , [("choose", 4, 1)]
        , { ip := 4
            code :=
              [ FrameOp.push (Value.num 4)
              , FrameOp.call 4 1
              , FrameOp.store 0
              , FrameOp.halt
              , FrameOp.load 0
              , FrameOp.push (Value.num 3)
              , FrameOp.bin BinOp.lt
              , FrameOp.jmpIfFalse 3
              , FrameOp.push (Value.num 1)
              , FrameOp.ret
              , FrameOp.jmp 2
              , FrameOp.push (Value.num 2)
              , FrameOp.ret
              , FrameOp.ret
              ]
            stack := []
            locals := [Value.num 2]
            frames := []
            halted := true }) := by
  native_decide

example :
    runCompiledFrameProgram
      80
      [ Stmt.fnDecl
          "countTo"
          ["x"]
          [ Stmt.while
              (Expr.binary (Expr.var "x") BinOp.lt (Expr.num 3))
              [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))]
          , Stmt.ret (some (Expr.var "x"))
          ]
      , Stmt.letDecl "y" (Expr.call "countTo" [Expr.num 0])
      ]
      =
      some
        ( [("y", 0)]
        , [("countTo", 4, 1)]
        , { ip := 4
            code :=
              [ FrameOp.push (Value.num 0)
              , FrameOp.call 4 1
              , FrameOp.store 0
              , FrameOp.halt
              , FrameOp.load 0
              , FrameOp.push (Value.num 3)
              , FrameOp.bin BinOp.lt
              , FrameOp.jmpIfFalse 5
              , FrameOp.load 0
              , FrameOp.push (Value.num 1)
              , FrameOp.bin BinOp.add
              , FrameOp.store 0
              , FrameOp.jmp (-9)
              , FrameOp.load 0
              , FrameOp.ret
              , FrameOp.ret
              ]
            stack := []
            locals := [Value.num 3]
            frames := []
            halted := true }) := by
  native_decide

example :
    runCompiledFrameProgram
      80
      [ Stmt.fnDecl
          "sum3"
          []
          [ Stmt.letDecl "sum" (Expr.num 0)
          , Stmt.forRange
              "i"
              0
              3
              [Stmt.assign "sum" (Expr.binary (Expr.var "sum") BinOp.add (Expr.var "i"))]
          , Stmt.ret (some (Expr.var "sum"))
          ]
      , Stmt.letDecl "y" (Expr.call "sum3" [])
      ]
      =
      some
        ( [("y", 0)]
        , [("sum3", 3, 0)]
        , { ip := 3
            code :=
              [ FrameOp.call 3 0
              , FrameOp.store 0
              , FrameOp.halt
              , FrameOp.push (Value.num 0)
              , FrameOp.store 0
              , FrameOp.push (Value.num 0)
              , FrameOp.store 1
              , FrameOp.load 1
              , FrameOp.push (Value.num 3)
              , FrameOp.bin BinOp.lt
              , FrameOp.jmpIfFalse 9
              , FrameOp.load 0
              , FrameOp.load 1
              , FrameOp.bin BinOp.add
              , FrameOp.store 0
              , FrameOp.load 1
              , FrameOp.push (Value.num 1)
              , FrameOp.bin BinOp.add
              , FrameOp.store 1
              , FrameOp.jmp (-13)
              , FrameOp.load 0
              , FrameOp.ret
              , FrameOp.ret
              ]
            stack := []
            locals := [Value.num 3]
            frames := []
            halted := true }) := by
  native_decide

example :
    runCompiledFrameProgram
      80
      [ Stmt.fnDecl
          "sealTo3"
          ["x"]
          [ Stmt.seal
              (some (Expr.binary (Expr.var "x") BinOp.eq (Expr.num 3)))
              [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))]
          , Stmt.ret (some (Expr.var "x"))
          ]
      , Stmt.letDecl "y" (Expr.call "sealTo3" [Expr.num 0])
      ]
      =
      some
        ( [("y", 0)]
        , [("sealTo3", 4, 1)]
        , { ip := 4
            code :=
              [ FrameOp.push (Value.num 0)
              , FrameOp.call 4 1
              , FrameOp.store 0
              , FrameOp.halt
              , FrameOp.load 0
              , FrameOp.push (Value.num 3)
              , FrameOp.bin BinOp.eq
              , FrameOp.unary UnOp.not
              , FrameOp.jmpIfFalse 5
              , FrameOp.load 0
              , FrameOp.push (Value.num 1)
              , FrameOp.bin BinOp.add
              , FrameOp.store 0
              , FrameOp.jmp (-10)
              , FrameOp.load 0
              , FrameOp.ret
              , FrameOp.ret
              ]
            stack := []
            locals := [Value.num 3]
            frames := []
            halted := true }) := by
  native_decide

example :
    compiledFrameLocal?
      120
      [ Stmt.fnDecl
          "firstBreak"
          ["x"]
          [ Stmt.while
              (Expr.binary (Expr.var "x") BinOp.lt (Expr.num 10))
              [ Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))
              , Stmt.ifThenElse
                  (Expr.binary (Expr.var "x") BinOp.eq (Expr.num 4))
                  [Stmt.break]
                  none
              ]
          , Stmt.ret (some (Expr.var "x"))
          ]
      , Stmt.letDecl "y" (Expr.call "firstBreak" [Expr.num 0])
      ]
      0
      =
      some (Value.num 4) := by
  native_decide

example :
    compiledFrameLocal?
      160
      [ Stmt.fnDecl
          "skip3"
          ["i"]
          [ Stmt.letDecl "sum" (Expr.num 0)
          , Stmt.while
              (Expr.binary (Expr.var "i") BinOp.lt (Expr.num 5))
              [ Stmt.assign "i" (Expr.binary (Expr.var "i") BinOp.add (Expr.num 1))
              , Stmt.ifThenElse
                  (Expr.binary (Expr.var "i") BinOp.eq (Expr.num 3))
                  [Stmt.continue]
                  none
              , Stmt.assign "sum" (Expr.binary (Expr.var "sum") BinOp.add (Expr.var "i"))
              ]
          , Stmt.ret (some (Expr.var "sum"))
          ]
      , Stmt.letDecl "y" (Expr.call "skip3" [Expr.num 0])
      ]
      0
      =
      some (Value.num 12) := by
  native_decide

example :
    compileExpr (Expr.binary (Expr.num 2) BinOp.mul (Expr.num 4))
      = some [Op.push (Value.num 2), Op.push (Value.num 4), Op.bin BinOp.mul] := by
  native_decide

example :
    compileExpr (Expr.float 1 500000) = some [Op.push (Value.float 1 500000)] := by
  native_decide

example :
    compiledExprValue? 4 (Expr.binary (Expr.float 1 500000) BinOp.add (Expr.num 2)) =
      some (Value.float 3 500000) := by
  native_decide

example :
    compileFrameExpr [] [] (Expr.float 1 500000) =
      some [FrameOp.push (Value.float 1 500000)] := by
  native_decide

example :
    compiledExprValue? 4 (Expr.binary (Expr.num 2) BinOp.mul (Expr.num 4))
      = evalExpr [] (Expr.binary (Expr.num 2) BinOp.mul (Expr.num 4)) := by
  native_decide

example :
    compiledExprValue? 6
      (Expr.binary
        (Expr.unary UnOp.neg (Expr.num 3))
        BinOp.add
        (Expr.binary (Expr.num 10) BinOp.mod (Expr.num 4)))
      =
      evalExpr []
        (Expr.binary
          (Expr.unary UnOp.neg (Expr.num 3))
          BinOp.add
          (Expr.binary (Expr.num 10) BinOp.mod (Expr.num 4))) := by
  native_decide

example :
    compiledExprValue? 4 (Expr.unary UnOp.not (Expr.bool false))
      = evalExpr [] (Expr.unary UnOp.not (Expr.bool false)) := by
  native_decide

example :
    compileExpr (Expr.str "open") = some [Op.push (Value.str "open")] := by
  native_decide

example :
    compiledExprValue? 2 (Expr.str "open") = some (Value.str "open") := by
  native_decide

example :
    compileFrameExpr [] [] (Expr.str "open") =
      some [FrameOp.push (Value.str "open")] := by
  native_decide

example :
    compileExpr Expr.unit = some [Op.push Value.unit] := by
  native_decide

example :
    compileFrameExpr [] [] Expr.unit = some [FrameOp.push Value.unit] := by
  native_decide

example :
    compileExpr (Expr.list [Expr.num 1, Expr.bool true]) =
      some [Op.push (Value.num 1), Op.push (Value.bool true), Op.list 2] := by
  native_decide

example :
    compiledExprValue? 4 (Expr.list [Expr.num 1, Expr.bool true]) =
      some (Value.list [Value.num 1, Value.bool true]) := by
  native_decide

example :
    compileFrameExpr [("x", 0)] [] (Expr.list [Expr.var "x", Expr.str "open"]) =
      some [FrameOp.load 0, FrameOp.push (Value.str "open"), FrameOp.list 2] := by
  native_decide

example :
    compileExpr (Expr.index (Expr.list [Expr.num 1, Expr.num 2]) (Expr.num 1)) =
      some [Op.push (Value.num 1), Op.push (Value.num 2), Op.list 2, Op.push (Value.num 1), Op.index] := by
  native_decide

example :
    compiledExprValue? 6 (Expr.index (Expr.list [Expr.num 1, Expr.num 2]) (Expr.num 1)) =
      some (Value.num 2) := by
  native_decide

example :
    compiledExprValue? 6 (Expr.index (Expr.str "open") (Expr.num 1)) =
      some (Value.str "p") := by
  native_decide

example :
    compileFrameExpr [("xs", 0)] [] (Expr.index (Expr.var "xs") (Expr.num 0)) =
      some [FrameOp.load 0, FrameOp.push (Value.num 0), FrameOp.index] := by
  native_decide

example :
    compileExpr (Expr.field (Expr.list [Expr.num 1, Expr.num 2]) "length") =
      some [Op.push (Value.num 1), Op.push (Value.num 2), Op.list 2, Op.field "length"] := by
  native_decide

example :
    compiledExprValue? 5 (Expr.field (Expr.list [Expr.num 1, Expr.num 2]) "length") =
      some (Value.num 2) := by
  native_decide

example :
    compileFrameExpr [("xs", 0)] [] (Expr.field (Expr.var "xs") "length") =
      some [FrameOp.load 0, FrameOp.field "length"] := by
  native_decide

example :
    compileExpr (Expr.method (Expr.list [Expr.num 1, Expr.num 2]) "len" []) =
      some [Op.push (Value.num 1), Op.push (Value.num 2), Op.list 2, Op.method "len" 0] := by
  native_decide

example :
    compiledExprValue? 5 (Expr.method (Expr.list [Expr.num 1, Expr.num 2]) "len" []) =
      some (Value.num 2) := by
  native_decide

example :
    compiledExprValue? 4 (Expr.method (Expr.list []) "is_empty" []) =
      some (Value.bool true) := by
  native_decide

example :
    compiledExprValue? 4 (Expr.method (Expr.str "open") "is_empty" []) =
      some (Value.bool false) := by
  native_decide

example :
    compiledExprValue? 5 (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "first" []) =
      some (Value.num 7) := by
  native_decide

example :
    compiledExprValue? 5 (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "last" []) =
      some (Value.num 9) := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "at" [Expr.num 1]) =
      some (Value.num 9) := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "contains" [Expr.num 9]) =
      some (Value.bool true) := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "tail" []) =
      some (Value.list [Value.num 9]) := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "take" [Expr.num 1]) =
      some (Value.list [Value.num 7]) := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "drop" [Expr.num 1]) =
      some (Value.list [Value.num 9]) := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "reverse" []) =
      some (Value.list [Value.num 9, Value.num 7]) := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.list [Expr.num 7]) "append" [Expr.num 9]) =
      some (Value.list [Value.num 7, Value.num 9]) := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.list [Expr.num 9]) "prepend" [Expr.num 7]) =
      some (Value.list [Value.num 7, Value.num 9]) := by
  native_decide

example :
    compiledExprValue? 8 (Expr.method (Expr.list [Expr.str "a", Expr.str "b"]) "join" [Expr.str ","]) =
      some (Value.str "a,b") := by
  native_decide

example :
    compiledExprValue? 8 (Expr.method (Expr.list [Expr.num 7]) "concat" [Expr.list [Expr.num 9]]) =
      some (Value.list [Value.num 7, Value.num 9]) := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.str "open") "at" [Expr.num 1]) =
      some (Value.str "p") := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.str "open") "contains" [Expr.str "pe"]) =
      some (Value.bool true) := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.str "open") "starts_with" [Expr.str "op"]) =
      some (Value.bool true) := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.str "open") "ends_with" [Expr.str "en"]) =
      some (Value.bool true) := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.str "open") "reverse" []) =
      some (Value.str "nepo") := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.str "open") "first" []) =
      some (Value.str "o") := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.str "open") "last" []) =
      some (Value.str "n") := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.str "open") "tail" []) =
      some (Value.str "pen") := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.str "open") "take" [Expr.num 2]) =
      some (Value.str "op") := by
  native_decide

example :
    compiledExprValue? 6 (Expr.method (Expr.str "open") "drop" [Expr.num 2]) =
      some (Value.str "en") := by
  native_decide

example :
    compileFrameExpr [("xs", 0)] [] (Expr.method (Expr.var "xs") "len" []) =
      some [FrameOp.load 0, FrameOp.method "len" 0] := by
  native_decide

example :
    compileFrameExpr [("xs", 0)] [] (Expr.method (Expr.var "xs") "is_empty" []) =
      some [FrameOp.load 0, FrameOp.method "is_empty" 0] := by
  native_decide

example :
    compileFrameExpr [("xs", 0)] [] (Expr.method (Expr.var "xs") "first" []) =
      some [FrameOp.load 0, FrameOp.method "first" 0] := by
  native_decide

example :
    compileFrameExpr [("xs", 0)] [] (Expr.method (Expr.var "xs") "last" []) =
      some [FrameOp.load 0, FrameOp.method "last" 0] := by
  native_decide

example :
    compileFrameExpr [("xs", 0)] [] (Expr.method (Expr.var "xs") "at" [Expr.num 1]) =
      some [FrameOp.load 0, FrameOp.push (Value.num 1), FrameOp.method "at" 1] := by
  native_decide

example :
    compileFrameExpr [("xs", 0)] [] (Expr.method (Expr.var "xs") "contains" [Expr.num 9]) =
      some [FrameOp.load 0, FrameOp.push (Value.num 9), FrameOp.method "contains" 1] := by
  native_decide

example :
    compileFrameExpr [("label", 0)] [] (Expr.method (Expr.var "label") "contains" [Expr.str "pe"]) =
      some [FrameOp.load 0, FrameOp.push (Value.str "pe"), FrameOp.method "contains" 1] := by
  native_decide

example :
    compileFrameExpr [("label", 0)] [] (Expr.method (Expr.var "label") "starts_with" [Expr.str "op"]) =
      some [FrameOp.load 0, FrameOp.push (Value.str "op"), FrameOp.method "starts_with" 1] := by
  native_decide

example :
    compileFrameExpr [("label", 0)] [] (Expr.method (Expr.var "label") "ends_with" [Expr.str "en"]) =
      some [FrameOp.load 0, FrameOp.push (Value.str "en"), FrameOp.method "ends_with" 1] := by
  native_decide

example :
    compileFrameExpr [("label", 0)] [] (Expr.method (Expr.var "label") "reverse" []) =
      some [FrameOp.load 0, FrameOp.method "reverse" 0] := by
  native_decide

example :
    compileFrameExpr [("label", 0)] [] (Expr.method (Expr.var "label") "first" []) =
      some [FrameOp.load 0, FrameOp.method "first" 0] := by
  native_decide

example :
    compileFrameExpr [("label", 0)] [] (Expr.method (Expr.var "label") "last" []) =
      some [FrameOp.load 0, FrameOp.method "last" 0] := by
  native_decide

example :
    compileFrameExpr [("label", 0)] [] (Expr.method (Expr.var "label") "tail" []) =
      some [FrameOp.load 0, FrameOp.method "tail" 0] := by
  native_decide

example :
    compileFrameExpr [("label", 0)] [] (Expr.method (Expr.var "label") "take" [Expr.num 2]) =
      some [FrameOp.load 0, FrameOp.push (Value.num 2), FrameOp.method "take" 1] := by
  native_decide

example :
    compileFrameExpr [("label", 0)] [] (Expr.method (Expr.var "label") "drop" [Expr.num 2]) =
      some [FrameOp.load 0, FrameOp.push (Value.num 2), FrameOp.method "drop" 1] := by
  native_decide

example :
    compileFrameExpr [("xs", 0)] [] (Expr.method (Expr.var "xs") "tail" []) =
      some [FrameOp.load 0, FrameOp.method "tail" 0] := by
  native_decide

example :
    compileFrameExpr [("xs", 0)] [] (Expr.method (Expr.var "xs") "take" [Expr.num 1]) =
      some [FrameOp.load 0, FrameOp.push (Value.num 1), FrameOp.method "take" 1] := by
  native_decide

example :
    compileFrameExpr [("xs", 0)] [] (Expr.method (Expr.var "xs") "drop" [Expr.num 1]) =
      some [FrameOp.load 0, FrameOp.push (Value.num 1), FrameOp.method "drop" 1] := by
  native_decide

example :
    compileFrameExpr [("xs", 0)] [] (Expr.method (Expr.var "xs") "reverse" []) =
      some [FrameOp.load 0, FrameOp.method "reverse" 0] := by
  native_decide

example :
    compileFrameExpr [("xs", 0)] [] (Expr.method (Expr.var "xs") "append" [Expr.num 9]) =
      some [FrameOp.load 0, FrameOp.push (Value.num 9), FrameOp.method "append" 1] := by
  native_decide

example :
    compileFrameExpr [("xs", 0)] [] (Expr.method (Expr.var "xs") "prepend" [Expr.num 7]) =
      some [FrameOp.load 0, FrameOp.push (Value.num 7), FrameOp.method "prepend" 1] := by
  native_decide

example :
    compileFrameExpr [("xs", 0)] [] (Expr.method (Expr.var "xs") "join" [Expr.str ","]) =
      some [FrameOp.load 0, FrameOp.push (Value.str ","), FrameOp.method "join" 1] := by
  native_decide

example :
    compileFrameExpr [("xs", 0)] [] (Expr.method (Expr.var "xs") "concat" [Expr.list [Expr.num 9]]) =
      some [FrameOp.load 0, FrameOp.push (Value.num 9), FrameOp.list 1, FrameOp.method "concat" 1] := by
  native_decide

example :
    compileFrameExpr [("label", 0)] [] (Expr.method (Expr.var "label") "at" [Expr.num 1]) =
      some [FrameOp.load 0, FrameOp.push (Value.num 1), FrameOp.method "at" 1] := by
  native_decide

example :
    compileExprWithSlots [("x", 0)]
      (Expr.binary (Expr.var "x") BinOp.add (Expr.num 2))
      = some [Op.load 0, Op.push (Value.num 2), Op.bin BinOp.add] := by
  native_decide

example :
    compiledExprWithSlotsValue? 4
      [("x", 0)]
      [Value.num 5]
      (Expr.binary (Expr.var "x") BinOp.add (Expr.num 2))
      =
      evalExpr [("x", Value.num 5)]
        (Expr.binary (Expr.var "x") BinOp.add (Expr.num 2)) := by
  native_decide

example :
    compileExprWithSlots [] (Expr.var "missing") = none := by
  native_decide

example :
    compileStmtWithSlots [] (Stmt.letDecl "x" (Expr.num 3))
      = some ([("x", 0)], [Op.push (Value.num 3), Op.store 0]) := by
  native_decide

example :
    runCompiledStmtWithSlots 3 [] [] (Stmt.letDecl "x" (Expr.num 3))
      = some
        ( [("x", 0)]
        , { ip := 3
            code := [Op.push (Value.num 3), Op.store 0, Op.halt]
            stack := []
            locals := [Value.num 3]
            halted := true }) := by
  native_decide

example :
    compileStmtWithSlots [("x", 0)]
      (Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 2)))
      =
      some
        ( [("x", 0)]
        , [Op.load 0, Op.push (Value.num 2), Op.bin BinOp.add, Op.store 0]) := by
  native_decide

example :
    runCompiledStmtWithSlots 5
      [("x", 0)]
      [Value.num 1]
      (Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 2)))
      =
      some
        ( [("x", 0)]
        , { ip := 5
            code := [Op.load 0, Op.push (Value.num 2), Op.bin BinOp.add, Op.store 0, Op.halt]
            stack := []
            locals := [Value.num 3]
            halted := true }) := by
  native_decide

example :
    compileStmtWithSlots [] (Stmt.assign "missing" (Expr.num 1)) = none := by
  native_decide

example :
    compileBlockWithSlots []
      [ Stmt.letDecl "x" (Expr.num 1)
      , Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 2))
      , Stmt.expr (Expr.var "x")
      ]
      =
      some
        ( [("x", 0)]
        , [ Op.push (Value.num 1), Op.store 0
          , Op.load 0, Op.push (Value.num 2), Op.bin BinOp.add, Op.store 0
          , Op.load 0
          ]) := by
  native_decide

example :
    runCompiledBlockWithSlots 8 [] []
      [ Stmt.letDecl "x" (Expr.num 1)
      , Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 2))
      , Stmt.expr (Expr.var "x")
      ]
      =
      some
        ( [("x", 0)]
        , { ip := 8
            code :=
              [ Op.push (Value.num 1), Op.store 0
              , Op.load 0, Op.push (Value.num 2), Op.bin BinOp.add, Op.store 0
              , Op.load 0, Op.halt
              ]
            stack := [Value.num 3]
            locals := [Value.num 3]
            halted := true }) := by
  native_decide

example :
    compileBlockWithSlots [] [Stmt.break] = none := by
  native_decide

example :
    compileBlockWithBranches [("x", 0)]
      [Stmt.ifThenElse
        (Expr.bool true)
        [Stmt.assign "x" (Expr.num 1)]
        (some [Stmt.assign "x" (Expr.num 2)])]
      =
      some
        ( [("x", 0)]
        , [ Op.push (Value.bool true)
          , Op.jmpIfFalse 3
          , Op.push (Value.num 1), Op.store 0
          , Op.jmp 2
          , Op.push (Value.num 2), Op.store 0
          ]) := by
  native_decide

example :
    runCompiledBlockWithBranches 7 [("x", 0)] [Value.num 0]
      [Stmt.ifThenElse
        (Expr.bool true)
        [Stmt.assign "x" (Expr.num 1)]
        (some [Stmt.assign "x" (Expr.num 2)])]
      =
      some
        ( [("x", 0)]
        , { ip := 8
            code :=
              [ Op.push (Value.bool true)
              , Op.jmpIfFalse 3
              , Op.push (Value.num 1), Op.store 0
              , Op.jmp 2
              , Op.push (Value.num 2), Op.store 0
              , Op.halt
              ]
            stack := []
            locals := [Value.num 1]
            halted := true }) := by
  native_decide

example :
    runCompiledBlockWithBranches 6 [("x", 0)] [Value.num 0]
      [Stmt.ifThenElse
        (Expr.bool false)
        [Stmt.assign "x" (Expr.num 1)]
        (some [Stmt.assign "x" (Expr.num 2)])]
      =
      some
        ( [("x", 0)]
        , { ip := 8
            code :=
              [ Op.push (Value.bool false)
              , Op.jmpIfFalse 3
              , Op.push (Value.num 1), Op.store 0
              , Op.jmp 2
              , Op.push (Value.num 2), Op.store 0
              , Op.halt
              ]
            stack := []
            locals := [Value.num 2]
            halted := true }) := by
  native_decide

example :
    compileBlockWithBranches [("x", 0)]
      [Stmt.while
        (Expr.binary (Expr.var "x") BinOp.lt (Expr.num 3))
        [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))]]
      =
      some
        ( [("x", 0)]
        , [ Op.load 0, Op.push (Value.num 3), Op.bin BinOp.lt
          , Op.jmpIfFalse 5
          , Op.load 0, Op.push (Value.num 1), Op.bin BinOp.add, Op.store 0
          , Op.jmp (-9)
          ]) := by
  native_decide

example :
    runCompiledBlockWithBranches 40 [("x", 0)] [Value.num 0]
      [Stmt.while
        (Expr.binary (Expr.var "x") BinOp.lt (Expr.num 3))
        [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))]]
      =
      some
        ( [("x", 0)]
        , { ip := 10
            code :=
              [ Op.load 0, Op.push (Value.num 3), Op.bin BinOp.lt
              , Op.jmpIfFalse 5
              , Op.load 0, Op.push (Value.num 1), Op.bin BinOp.add, Op.store 0
              , Op.jmp (-9)
              , Op.halt
              ]
            stack := []
            locals := [Value.num 3]
            halted := true }) := by
  native_decide

example :
    runCompiledBlockWithBranches 6 [("x", 0)] [Value.num 3]
      [Stmt.while
        (Expr.binary (Expr.var "x") BinOp.lt (Expr.num 3))
        [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))]]
      =
      some
        ( [("x", 0)]
        , { ip := 10
            code :=
              [ Op.load 0, Op.push (Value.num 3), Op.bin BinOp.lt
              , Op.jmpIfFalse 5
              , Op.load 0, Op.push (Value.num 1), Op.bin BinOp.add, Op.store 0
              , Op.jmp (-9)
              , Op.halt
              ]
            stack := []
            locals := [Value.num 3]
            halted := true }) := by
  native_decide

example :
    compileBlockWithBranches [("x", 0)]
      [Stmt.forRange
        "i"
        0
        3
        [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.var "i"))]]
      =
      some
        ( [("x", 0), ("i", 1)]
        , [ Op.push (Value.num 0), Op.store 1
          , Op.load 1, Op.push (Value.num 3), Op.bin BinOp.lt
          , Op.jmpIfFalse 9
          , Op.load 0, Op.load 1, Op.bin BinOp.add, Op.store 0
          , Op.load 1, Op.push (Value.num 1), Op.bin BinOp.add, Op.store 1
          , Op.jmp (-13)
          ]) := by
  native_decide

example :
    runCompiledBlockWithBranches 60 [("x", 0)] [Value.num 0]
      [Stmt.forRange
        "i"
        0
        3
        [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.var "i"))]]
      =
      some
        ( [("x", 0), ("i", 1)]
        , { ip := 16
            code :=
              [ Op.push (Value.num 0), Op.store 1
              , Op.load 1, Op.push (Value.num 3), Op.bin BinOp.lt
              , Op.jmpIfFalse 9
              , Op.load 0, Op.load 1, Op.bin BinOp.add, Op.store 0
              , Op.load 1, Op.push (Value.num 1), Op.bin BinOp.add, Op.store 1
              , Op.jmp (-13)
              , Op.halt
              ]
            stack := []
            locals := [Value.num 3, Value.num 3]
            halted := true }) := by
  native_decide

example :
    compileBlockWithBranches [("x", 0)]
      [Stmt.seal
        (some (Expr.binary (Expr.var "x") BinOp.eq (Expr.num 3)))
        [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))]]
      =
      some
        ( [("x", 0)]
        , [ Op.load 0, Op.push (Value.num 3), Op.bin BinOp.eq
          , Op.unary UnOp.not, Op.jmpIfFalse 5
          , Op.load 0, Op.push (Value.num 1), Op.bin BinOp.add, Op.store 0
          , Op.jmp (-10)
          ]) := by
  native_decide

example :
    runCompiledBlockWithBranches 40 [("x", 0)] [Value.num 0]
      [Stmt.seal
        (some (Expr.binary (Expr.var "x") BinOp.eq (Expr.num 3)))
        [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))]]
      =
      some
        ( [("x", 0)]
        , { ip := 11
            code :=
              [ Op.load 0, Op.push (Value.num 3), Op.bin BinOp.eq
              , Op.unary UnOp.not, Op.jmpIfFalse 5
              , Op.load 0, Op.push (Value.num 1), Op.bin BinOp.add, Op.store 0
              , Op.jmp (-10)
              , Op.halt
              ]
            stack := []
            locals := [Value.num 3]
            halted := true }) := by
  native_decide

example :
    compileBlockWithBranches [("x", 0)]
      [Stmt.seal
        none
        [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))]]
      =
      some
        ( [("x", 0)]
        , [ Op.load 0, Op.push (Value.num 1), Op.bin BinOp.add, Op.store 0
          , Op.jmp (-5)
          ]) := by
  native_decide

end VM
end Aether
