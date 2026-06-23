import Aether.Core

namespace Aether
namespace Static

inductive Ty where
  | num
  | bool
  | str
  | list : Ty -> Ty
  | unit
  | unknown
  deriving Repr, BEq, DecidableEq, Inhabited

abbrev VarEnv := List (Ident × Ty)

structure FnSig where
  arity : Nat
  params : List Ident
  paramTys : List Ty
  result : Ty
  deriving Repr, BEq, DecidableEq

abbrev FnSigEnv := List (Ident × FnSig)

structure Scope where
  loopDepth : Nat
  functionDepth : Nat
  topLevel : Bool
  deriving Repr, BEq, DecidableEq

structure CheckState where
  vars : VarEnv
  deriving Repr, BEq, DecidableEq

inductive CheckError where
  | undeclaredVariable : Ident -> CheckError
  | undeclaredFunction : Ident -> CheckError
  | operandMismatch : BinOp -> Ty -> Ty -> CheckError
  | unaryMismatch : UnOp -> Ty -> CheckError
  | indexMismatch : Ty -> Ty -> CheckError
  | fieldMismatch : Ty -> Ident -> CheckError
  | methodMismatch : Ty -> Ident -> Nat -> CheckError
  | conditionMismatch : Ty -> CheckError
  | assignmentMismatch : Ident -> Ty -> Ty -> CheckError
  | arityMismatch : Ident -> Nat -> Nat -> CheckError
  | functionSignatureMismatch : Ident -> Nat -> Nat -> CheckError
  | unknownNamedArgument : Ident -> Ident -> CheckError
  | duplicateNamedArgument : Ident -> Ident -> CheckError
  | argumentMismatch : Ident -> Ident -> Ty -> Ty -> CheckError
  | returnMismatch : Ident -> Ty -> Ty -> CheckError
  | duplicateFunction : Ident -> CheckError
  | duplicateParameter : Ident -> CheckError
  | nestedFunction : Ident -> CheckError
  | returnOutsideFunction : CheckError
  | breakOutsideLoop : CheckError
  | continueOutsideLoop : CheckError
  deriving Repr, BEq, DecidableEq, Inhabited

instance [BEq ε] [BEq α] : BEq (Except ε α) where
  beq
    | Except.ok left, Except.ok right => left == right
    | Except.error left, Except.error right => left == right
    | _, _ => false

def VarEnv.lookup (env : VarEnv) (name : Ident) : Option Ty :=
  match env with
  | [] => none
  | (key, ty) :: rest =>
      if key == name then some ty else VarEnv.lookup rest name

def VarEnv.bind (env : VarEnv) (name : Ident) (ty : Ty) : VarEnv :=
  (name, ty) :: env

def VarEnv.assign (env : VarEnv) (name : Ident) (ty : Ty) : Option VarEnv :=
  match env with
  | [] => none
  | (key, oldTy) :: rest =>
      if key == name then
        some ((key, ty) :: rest)
      else
        match VarEnv.assign rest name ty with
        | some updated => some ((key, oldTy) :: updated)
        | none => none

def FnSigEnv.lookup (env : FnSigEnv) (name : Ident) : Option FnSig :=
  match env with
  | [] => none
  | (key, sig) :: rest =>
      if key == name then some sig else FnSigEnv.lookup rest name

def FnSigEnv.bind (env : FnSigEnv) (name : Ident) (sig : FnSig) : FnSigEnv :=
  (name, sig) :: env

def bindUnknownParams (env : VarEnv) : List Ident -> VarEnv
  | [] => env
  | name :: rest => bindUnknownParams (VarEnv.bind env name Ty.unknown) rest

def annTyToTy : AnnTy -> Ty
  | AnnTy.num => Ty.num
  | AnnTy.bool => Ty.bool
  | AnnTy.str => Ty.str
  | AnnTy.unit => Ty.unit
  | AnnTy.list elementTy => Ty.list (annTyToTy elementTy)

def bindTypedParams (env : VarEnv) : List (Ident × AnnTy) -> VarEnv
  | [] => env
  | (name, ty) :: rest => bindTypedParams (VarEnv.bind env name (annTyToTy ty)) rest

def typedParamNames : List (Ident × AnnTy) -> List Ident
  | [] => []
  | (name, _) :: rest => name :: typedParamNames rest

def typedParamTys : List (Ident × AnnTy) -> List Ty
  | [] => []
  | (_, ty) :: rest => annTyToTy ty :: typedParamTys rest

def containsIdent (names : List Ident) (name : Ident) : Bool :=
  match names with
  | [] => false
  | first :: rest => first == name || containsIdent rest name

def seenNamedArg (names : List Ident) (name : Ident) : Bool :=
  containsIdent names name

def checkNamedArgs (fnName : Ident) (params : List Ident) (seen : List Ident) :
    List Arg -> Option Unit
  | [] => some ()
  | Arg.positional _ :: rest => checkNamedArgs fnName params seen rest
  | Arg.named name _ :: rest =>
      if !containsIdent params name then
        none
      else if seenNamedArg seen name then
        none
      else
        checkNamedArgs fnName params (name :: seen) rest

def checkNamedArgsDetailed (fnName : Ident) (params : List Ident) (seen : List Ident) :
    List Arg -> Except CheckError Unit
  | [] => Except.ok ()
  | Arg.positional _ :: rest => checkNamedArgsDetailed fnName params seen rest
  | Arg.named name _ :: rest =>
      if !containsIdent params name then
        Except.error (CheckError.unknownNamedArgument fnName name)
      else if seenNamedArg seen name then
        Except.error (CheckError.duplicateNamedArgument fnName name)
      else
        checkNamedArgsDetailed fnName params (name :: seen) rest

def bindUnknownParamsDetailed (env : VarEnv) : List Ident -> Except CheckError VarEnv
  | [] => Except.ok env
  | name :: rest =>
      if containsIdent rest name then
        Except.error (CheckError.duplicateParameter name)
      else
        bindUnknownParamsDetailed (VarEnv.bind env name Ty.unknown) rest

def bindTypedParamsDetailed (env : VarEnv) : List (Ident × AnnTy) -> Except CheckError VarEnv
  | [] => Except.ok env
  | (name, ty) :: rest =>
      if containsIdent (typedParamNames rest) name then
        Except.error (CheckError.duplicateParameter name)
      else
        bindTypedParamsDetailed (VarEnv.bind env name (annTyToTy ty)) rest

def numericLike : Ty -> Bool
  | Ty.num => true
  | Ty.unknown => true
  | _ => false

def boolLike : Ty -> Bool
  | Ty.bool => true
  | Ty.unknown => true
  | _ => false

def listLike : Ty -> Bool
  | Ty.list _ => true
  | Ty.unknown => true
  | _ => false

def compatibleTy : Ty -> Ty -> Bool
  | Ty.unknown, _ => true
  | _, Ty.unknown => true
  | Ty.list left, Ty.list right => compatibleTy left right
  | left, right => left == right

def checkCallArgTys
    (fnName : Ident)
    (params : List Ident)
    (paramTys : List Ty)
    (args : List Arg)
    (values : List Ty) :
    Option Unit :=
  match params, paramTys, args, values with
  | [], [], [], [] => some ()
  | _ :: restParams, expected :: restExpected, _ :: restArgs, actual :: restActual =>
      if compatibleTy expected actual then
        checkCallArgTys fnName restParams restExpected restArgs restActual
      else
        none
  | _, _, _, _ => none

def checkCallArgTysDetailed
    (fnName : Ident)
    (params : List Ident)
    (paramTys : List Ty)
    (args : List Arg)
    (values : List Ty) :
    Except CheckError Unit :=
  match params, paramTys, args, values with
  | [], [], [], [] => Except.ok ()
  | param :: restParams, expected :: restExpected, _ :: restArgs, actual :: restActual =>
      if compatibleTy expected actual then
        checkCallArgTysDetailed fnName restParams restExpected restArgs restActual
      else
        Except.error (CheckError.argumentMismatch fnName param expected actual)
  | _, _, _, _ => Except.error (CheckError.arityMismatch fnName params.length args.length)

def assignable (target value : Ty) : Bool :=
  compatibleTy target value

def refineAssignedTy : Ty -> Ty -> Ty
  | Ty.unknown, value => value
  | Ty.list target, Ty.list value => Ty.list (refineAssignedTy target value)
  | target, _ => target

def mergeElementTy (left right : Ty) : Ty :=
  if left == right then left else Ty.unknown

def joinTy (left right : Ty) : Option Ty :=
  if compatibleTy left right then
    some (mergeElementTy left right)
  else
    none

def joinBranchVar (baseTy thenTy elseTy : Ty) : Ty :=
  match joinTy thenTy elseTy with
  | some joinedTy => refineAssignedTy baseTy joinedTy
  | none => baseTy

def joinNewBranchVars (base thenVars elseVars : VarEnv) : VarEnv :=
  match thenVars with
  | [] => base
  | (name, thenTy) :: rest =>
      let joinedRest := joinNewBranchVars base rest elseVars
      match VarEnv.lookup base name, VarEnv.lookup elseVars name with
      | none, some elseTy =>
          match joinTy thenTy elseTy with
          | some joinedTy => VarEnv.bind joinedRest name joinedTy
          | none => joinedRest
      | some baseTy, some elseTy =>
          match VarEnv.assign joinedRest name (joinBranchVar baseTy thenTy elseTy) with
          | some updated => updated
          | none => joinedRest
      | _, _ => joinedRest

def conditionCompatible : Ty -> Bool :=
  boolLike

def equalityCompatible (left right : Ty) : Bool :=
  compatibleTy left right

def compatibleBinOp (op : BinOp) (left right : Ty) : Option Ty :=
  match op with
  | BinOp.add | BinOp.sub | BinOp.mul | BinOp.div | BinOp.mod =>
      if numericLike left && numericLike right then some Ty.num else none
  | BinOp.lt | BinOp.gt | BinOp.le | BinOp.ge =>
      if numericLike left && numericLike right then some Ty.bool else none
  | BinOp.eq | BinOp.neq =>
      if equalityCompatible left right then some Ty.bool else none
  | BinOp.and | BinOp.or =>
      if boolLike left && boolLike right then some Ty.bool else none

def compatibleUnOp (op : UnOp) (value : Ty) : Option Ty :=
  match op with
  | UnOp.neg => if numericLike value then some Ty.num else none
  | UnOp.not => if boolLike value then some Ty.bool else none

def compatibleIndex (target index : Ty) : Option Ty :=
  match target with
  | Ty.list elem => if numericLike index then some elem else none
  | Ty.str => if numericLike index then some Ty.str else none
  | Ty.unknown => if numericLike index then some Ty.unknown else none
  | _ => none

def compatibleField (target : Ty) (field : Ident) : Option Ty :=
  match target, field with
  | Ty.list _, "length" => some Ty.num
  | Ty.str, "length" => some Ty.num
  | Ty.unknown, _ => some Ty.unknown
  | _, _ => none

def compatibleMethodWithArgs (target : Ty) (method : Ident) (argTys : List Ty) : Option Ty :=
  match target, method, argTys with
  | Ty.list _, "len", [] => some Ty.num
  | Ty.list _, "is_empty", [] => some Ty.bool
  | Ty.list elem, "first", [] => some elem
  | Ty.list elem, "tail", [] => some (Ty.list elem)
  | Ty.list elem, "last", [] => some elem
  | Ty.list elem, "at", [indexTy] => if numericLike indexTy then some elem else none
  | Ty.list elem, "take", [countTy] => if numericLike countTy then some (Ty.list elem) else none
  | Ty.list elem, "drop", [countTy] => if numericLike countTy then some (Ty.list elem) else none
  | Ty.list elem, "reverse", [] => some (Ty.list elem)
  | Ty.list elem, "append", [argTy] => if compatibleTy elem argTy then some (Ty.list elem) else none
  | Ty.list elem, "prepend", [argTy] => if compatibleTy elem argTy then some (Ty.list elem) else none
  | Ty.list elem, "concat", [Ty.list other] => if compatibleTy elem other then some (Ty.list elem) else none
  | Ty.list elem, "join", [argTy] =>
      if compatibleTy Ty.str elem && compatibleTy Ty.str argTy then some Ty.str else none
  | Ty.list elem, "contains", [argTy] => if compatibleTy elem argTy then some Ty.bool else none
  | Ty.str, "len", [] => some Ty.num
  | Ty.str, "is_empty", [] => some Ty.bool
  | Ty.str, "first", [] => some Ty.str
  | Ty.str, "last", [] => some Ty.str
  | Ty.str, "tail", [] => some Ty.str
  | Ty.str, "take", [countTy] => if numericLike countTy then some Ty.str else none
  | Ty.str, "drop", [countTy] => if numericLike countTy then some Ty.str else none
  | Ty.str, "at", [indexTy] => if numericLike indexTy then some Ty.str else none
  | Ty.str, "contains", [argTy] => if compatibleTy Ty.str argTy then some Ty.bool else none
  | Ty.str, "starts_with", [argTy] => if compatibleTy Ty.str argTy then some Ty.bool else none
  | Ty.str, "ends_with", [argTy] => if compatibleTy Ty.str argTy then some Ty.bool else none
  | Ty.str, "reverse", [] => some Ty.str
  | Ty.unknown, _, _ => some Ty.unknown
  | _, _, _ => none

def compatibleMethod (target : Ty) (method : Ident) (arity : Nat) : Option Ty :=
  compatibleMethodWithArgs target method (List.replicate arity Ty.unknown)

mutual
  partial def checkArg (vars : VarEnv) (fns : FnSigEnv) : Arg -> Option Unit
    | Arg.positional expr => do
        let _ <- checkExpr vars fns expr
        some ()
    | Arg.named _ expr => do
        let _ <- checkExpr vars fns expr
        some ()

  partial def checkArgs (vars : VarEnv) (fns : FnSigEnv) : List Arg -> Option Unit
    | [] => some ()
    | arg :: rest => do
        let _ <- checkArg vars fns arg
        checkArgs vars fns rest

  partial def inferArgTys (vars : VarEnv) (fns : FnSigEnv) : List Arg -> Option (List Ty)
    | [] => some []
    | Arg.positional expr :: rest => do
        let ty <- checkExpr vars fns expr
        let tys <- inferArgTys vars fns rest
        some (ty :: tys)
    | Arg.named _ expr :: rest => do
        let ty <- checkExpr vars fns expr
        let tys <- inferArgTys vars fns rest
        some (ty :: tys)

  partial def checkListElems (vars : VarEnv) (fns : FnSigEnv) : List Expr -> Option Ty
    | [] => some Ty.unknown
    | [expr] => checkExpr vars fns expr
    | expr :: rest => do
        let elemTy <- checkExpr vars fns expr
        let restTy <- checkListElems vars fns rest
        some (mergeElementTy elemTy restTy)

  partial def checkExpr (vars : VarEnv) (fns : FnSigEnv) : Expr -> Option Ty
    | Expr.num _ => some Ty.num
    | Expr.float _ _ => some Ty.num
    | Expr.bool _ => some Ty.bool
    | Expr.str _ => some Ty.str
    | Expr.unit => some Ty.unit
    | Expr.list exprs => do
        let elemTy <- checkListElems vars fns exprs
        some (Ty.list elemTy)
    | Expr.var name => VarEnv.lookup vars name
    | Expr.unary op expr => do
        let valueTy <- checkExpr vars fns expr
        compatibleUnOp op valueTy
    | Expr.binary left op right => do
        let leftTy <- checkExpr vars fns left
        let rightTy <- checkExpr vars fns right
        compatibleBinOp op leftTy rightTy
    | Expr.index target index => do
        let targetTy <- checkExpr vars fns target
        let indexTy <- checkExpr vars fns index
        compatibleIndex targetTy indexTy
    | Expr.field target field => do
        let targetTy <- checkExpr vars fns target
        compatibleField targetTy field
    | Expr.method target method args => do
        let targetTy <- checkExpr vars fns target
        let argTys <- inferArgTys vars fns args
        compatibleMethodWithArgs targetTy method argTys
    | Expr.call name args => do
        let sig <- FnSigEnv.lookup fns name
        if sig.arity == args.length then
          let _ <- checkNamedArgs name sig.params [] args
          let argTys <- inferArgTys vars fns args
          let _ <- checkCallArgTys name sig.params sig.paramTys args argTys
          some sig.result
        else
          none

  partial def checkStmt (fns : FnSigEnv) (scope : Scope) (state : CheckState) :
      Stmt -> Option CheckState
    | Stmt.letDecl name expr => do
        let ty <- checkExpr state.vars fns expr
        some { state with vars := VarEnv.bind state.vars name ty }
    | Stmt.letDeclTyped name annTy expr => do
        let expectedTy := annTyToTy annTy
        let valueTy <- checkExpr state.vars fns expr
        if assignable expectedTy valueTy then
          some { state with vars := VarEnv.bind state.vars name (refineAssignedTy expectedTy valueTy) }
        else
          none
    | Stmt.assign name expr => do
        let targetTy <- VarEnv.lookup state.vars name
        let valueTy <- checkExpr state.vars fns expr
        if assignable targetTy valueTy then
          let updatedVars <- VarEnv.assign state.vars name (refineAssignedTy targetTy valueTy)
          some { state with vars := updatedVars }
        else
          none
    | Stmt.ifThenElse condition thenBranch elseBranch => do
        let conditionTy <- checkExpr state.vars fns condition
        if !conditionCompatible conditionTy then
          none
        else
          pure ()
        let blockScope := { scope with topLevel := false }
        let thenState <- checkBlock fns blockScope state thenBranch
        match elseBranch with
        | none => some state
        | some statements => do
            let elseState <- checkBlock fns blockScope state statements
            some { state with vars := joinNewBranchVars state.vars thenState.vars elseState.vars }
    | Stmt.while condition body => do
        let conditionTy <- checkExpr state.vars fns condition
        if !conditionCompatible conditionTy then
          none
        else
          pure ()
        let loopScope := { scope with loopDepth := scope.loopDepth + 1, topLevel := false }
        let _ <- checkBlock fns loopScope state body
        some state
    | Stmt.forRange iterator _ _ body => do
        let loopScope := { scope with loopDepth := scope.loopDepth + 1, topLevel := false }
        let bodyState := { state with vars := VarEnv.bind state.vars iterator Ty.num }
        let _ <- checkBlock fns loopScope bodyState body
        some state
    | Stmt.seal condition body => do
        match condition with
        | none => pure ()
        | some expr =>
            let conditionTy <- checkExpr state.vars fns expr
            if !conditionCompatible conditionTy then
              none
            else
              pure ()
        let loopScope := { scope with loopDepth := scope.loopDepth + 1, topLevel := false }
        let _ <- checkBlock fns loopScope state body
        some state
    | Stmt.fnDecl name params body => do
        if !scope.topLevel then
          none
        else
          let sig <- FnSigEnv.lookup fns name
          if sig.arity == params.length then
            let fnScope := { loopDepth := 0, functionDepth := scope.functionDepth + 1, topLevel := false }
            let fnState := { vars := bindUnknownParams [] params }
            let _ <- checkBlock fns fnScope fnState body
            some state
          else
            none
    | Stmt.fnDeclReturn name params _ body => do
        if !scope.topLevel then
          none
        else
          let sig <- FnSigEnv.lookup fns name
          if sig.arity == params.length then
            let fnScope := { loopDepth := 0, functionDepth := scope.functionDepth + 1, topLevel := false }
            let fnState := { vars := bindUnknownParams [] params }
            let _ <- checkBlock fns fnScope fnState body
            some state
          else
            none
    | Stmt.fnDeclTyped name params body => do
        if !scope.topLevel then
          none
        else
          let sig <- FnSigEnv.lookup fns name
          if sig.arity == params.length then
            let fnScope := { loopDepth := 0, functionDepth := scope.functionDepth + 1, topLevel := false }
            let fnState := { vars := bindTypedParams [] params }
            let _ <- checkBlock fns fnScope fnState body
            some state
          else
            none
    | Stmt.fnDeclTypedReturn name params _ body => do
        if !scope.topLevel then
          none
        else
          let sig <- FnSigEnv.lookup fns name
          if sig.arity == params.length then
            let fnScope := { loopDepth := 0, functionDepth := scope.functionDepth + 1, topLevel := false }
            let fnState := { vars := bindTypedParams [] params }
            let _ <- checkBlock fns fnScope fnState body
            some state
          else
            none
    | Stmt.ret returnValue =>
        if scope.functionDepth == 0 then
          none
        else
          match returnValue with
          | none => some state
          | some expr => do
              let _ <- checkExpr state.vars fns expr
              some state
    | Stmt.break =>
        if scope.loopDepth == 0 then none else some state
    | Stmt.continue =>
        if scope.loopDepth == 0 then none else some state
    | Stmt.expr expr => do
        let _ <- checkExpr state.vars fns expr
        some state

  partial def checkBlock (fns : FnSigEnv) (scope : Scope) (state : CheckState) :
      List Stmt -> Option CheckState
    | [] => some state
    | stmt :: rest => do
        let next <- checkStmt fns scope state stmt
        checkBlock fns scope next rest
end

def collectFnSigs : List Stmt -> FnSigEnv
  | [] => []
  | Stmt.fnDecl name params _ :: rest =>
      FnSigEnv.bind (collectFnSigs rest) name
        { arity := params.length, params := params, paramTys := List.replicate params.length Ty.unknown, result := Ty.unknown }
  | Stmt.fnDeclReturn name params returnTy _ :: rest =>
      FnSigEnv.bind (collectFnSigs rest) name
        { arity := params.length, params := params, paramTys := List.replicate params.length Ty.unknown, result := annTyToTy returnTy }
  | Stmt.fnDeclTyped name params _ :: rest =>
      FnSigEnv.bind (collectFnSigs rest) name
        { arity := params.length, params := typedParamNames params, paramTys := typedParamTys params, result := Ty.unknown }
  | Stmt.fnDeclTypedReturn name params returnTy _ :: rest =>
      FnSigEnv.bind (collectFnSigs rest) name
        { arity := params.length, params := typedParamNames params, paramTys := typedParamTys params, result := annTyToTy returnTy }
  | _ :: rest => collectFnSigs rest

def collectFnSigsDetailed : List Stmt -> FnSigEnv -> Except CheckError FnSigEnv
  | [], fns => Except.ok fns
  | Stmt.fnDecl name params _ :: rest, fns =>
      match FnSigEnv.lookup fns name with
      | some _ => Except.error (CheckError.duplicateFunction name)
      | none =>
          collectFnSigsDetailed rest
            (FnSigEnv.bind fns name
              { arity := params.length, params := params, paramTys := List.replicate params.length Ty.unknown, result := Ty.unknown })
  | Stmt.fnDeclReturn name params returnTy _ :: rest, fns =>
      match FnSigEnv.lookup fns name with
      | some _ => Except.error (CheckError.duplicateFunction name)
      | none =>
          collectFnSigsDetailed rest
            (FnSigEnv.bind fns name
              { arity := params.length, params := params, paramTys := List.replicate params.length Ty.unknown, result := annTyToTy returnTy })
  | Stmt.fnDeclTyped name params _ :: rest, fns =>
      match FnSigEnv.lookup fns name with
      | some _ => Except.error (CheckError.duplicateFunction name)
      | none =>
          collectFnSigsDetailed rest
            (FnSigEnv.bind fns name
              { arity := params.length, params := typedParamNames params, paramTys := typedParamTys params, result := Ty.unknown })
  | Stmt.fnDeclTypedReturn name params returnTy _ :: rest, fns =>
      match FnSigEnv.lookup fns name with
      | some _ => Except.error (CheckError.duplicateFunction name)
      | none =>
          collectFnSigsDetailed rest
            (FnSigEnv.bind fns name
              { arity := params.length, params := typedParamNames params, paramTys := typedParamTys params, result := annTyToTy returnTy })
  | _ :: rest, fns => collectFnSigsDetailed rest fns

def mergeReturnTy? : Option Ty -> Option Ty -> Option Ty
  | none, right => right
  | left, none => left
  | some left, some right => some (mergeElementTy left right)

mutual
  partial def inferStmtReturnTy? (vars : VarEnv) (fns : FnSigEnv) : Stmt -> Option Ty
    | Stmt.ret none => some Ty.unit
    | Stmt.ret (some expr) => checkExpr vars fns expr
    | Stmt.ifThenElse _ thenBranch elseBranch =>
        mergeReturnTy?
          (inferBlockReturnTy? vars fns thenBranch)
          (match elseBranch with
           | none => none
           | some statements => inferBlockReturnTy? vars fns statements)
    | Stmt.while _ body => inferBlockReturnTy? vars fns body
    | Stmt.forRange iterator _ _ body =>
        inferBlockReturnTy? (VarEnv.bind vars iterator Ty.num) fns body
    | Stmt.seal _ body => inferBlockReturnTy? vars fns body
    | _ => none

  partial def inferStmtVars? (vars : VarEnv) (fns : FnSigEnv) : Stmt -> Option VarEnv
    | Stmt.letDecl name expr => do
        let ty <- checkExpr vars fns expr
        some (VarEnv.bind vars name ty)
    | Stmt.letDeclTyped name annTy expr => do
        let expectedTy := annTyToTy annTy
        let valueTy <- checkExpr vars fns expr
        if assignable expectedTy valueTy then
          some (VarEnv.bind vars name (refineAssignedTy expectedTy valueTy))
        else
          none
    | Stmt.assign name expr => do
        let targetTy <- VarEnv.lookup vars name
        let valueTy <- checkExpr vars fns expr
        if assignable targetTy valueTy then
          VarEnv.assign vars name (refineAssignedTy targetTy valueTy)
        else
          none
    | Stmt.ifThenElse condition thenBranch (some elseBranch) => do
        let conditionTy <- checkExpr vars fns condition
        if !conditionCompatible conditionTy then
          none
        else
          let thenVars <- inferBlockVars? vars fns thenBranch
          let elseVars <- inferBlockVars? vars fns elseBranch
          some (joinNewBranchVars vars thenVars elseVars)
    | _ => some vars

  partial def inferBlockVars? (vars : VarEnv) (fns : FnSigEnv) : List Stmt -> Option VarEnv
    | [] => some vars
    | stmt :: rest => do
        let nextVars <- inferStmtVars? vars fns stmt
        inferBlockVars? nextVars fns rest

  partial def inferBlockReturnTy? (vars : VarEnv) (fns : FnSigEnv) : List Stmt -> Option Ty
    | [] => none
    | [Stmt.expr expr] => checkExpr vars fns expr
    | [stmt] => inferStmtReturnTy? vars fns stmt
    | stmt :: rest =>
        let nextVars :=
          match inferStmtVars? vars fns stmt with
          | some updated => updated
          | none => vars
        mergeReturnTy?
          (inferStmtReturnTy? vars fns stmt)
          (inferBlockReturnTy? nextVars fns rest)
end

def inferFnSigFromVars (fns : FnSigEnv) (params : List Ident) (paramTys : List Ty) (vars : VarEnv) (body : List Stmt) : FnSig :=
  let result :=
    match inferBlockReturnTy? vars fns body with
    | some ty => ty
    | none => Ty.unknown
  { arity := params.length, params := params, paramTys := paramTys, result := result }

def inferFnSig (fns : FnSigEnv) (params : List Ident) (body : List Stmt) : FnSig :=
  inferFnSigFromVars fns params (List.replicate params.length Ty.unknown) (bindUnknownParams [] params) body

def inferReturnFnSig (params : List Ident) (returnTy : AnnTy) : FnSig :=
  { arity := params.length
  , params := params
  , paramTys := List.replicate params.length Ty.unknown
  , result := annTyToTy returnTy
  }

def inferTypedFnSig (fns : FnSigEnv) (params : List (Ident × AnnTy)) (body : List Stmt) : FnSig :=
  inferFnSigFromVars fns (typedParamNames params) (typedParamTys params) (bindTypedParams [] params) body

def inferTypedReturnFnSig (params : List (Ident × AnnTy)) (returnTy : AnnTy) : FnSig :=
  { arity := params.length
  , params := typedParamNames params
  , paramTys := typedParamTys params
  , result := annTyToTy returnTy
  }

def inferFnSigResults (stmts : List Stmt) (fns : FnSigEnv) : FnSigEnv :=
  match stmts with
  | [] => fns
  | Stmt.fnDecl name params body :: rest =>
      FnSigEnv.bind (inferFnSigResults rest fns) name (inferFnSig fns params body)
  | Stmt.fnDeclReturn name params returnTy _ :: rest =>
      FnSigEnv.bind (inferFnSigResults rest fns) name (inferReturnFnSig params returnTy)
  | Stmt.fnDeclTyped name params body :: rest =>
      FnSigEnv.bind (inferFnSigResults rest fns) name (inferTypedFnSig fns params body)
  | Stmt.fnDeclTypedReturn name params returnTy _ :: rest =>
      FnSigEnv.bind (inferFnSigResults rest fns) name (inferTypedReturnFnSig params returnTy)
  | _ :: rest => inferFnSigResults rest fns

def checkProgramWithFns (fns : FnSigEnv) (stmts : List Stmt) : Option CheckState :=
  checkBlock fns { loopDepth := 0, functionDepth := 0, topLevel := true } { vars := [] } stmts

def compatibleBinOpDetailed (op : BinOp) (left right : Ty) : Except CheckError Ty :=
  match compatibleBinOp op left right with
  | some ty => Except.ok ty
  | none => Except.error (CheckError.operandMismatch op left right)

def compatibleUnOpDetailed (op : UnOp) (value : Ty) : Except CheckError Ty :=
  match compatibleUnOp op value with
  | some ty => Except.ok ty
  | none => Except.error (CheckError.unaryMismatch op value)

def compatibleIndexDetailed (target index : Ty) : Except CheckError Ty :=
  match compatibleIndex target index with
  | some ty => Except.ok ty
  | none => Except.error (CheckError.indexMismatch target index)

def compatibleFieldDetailed (target : Ty) (field : Ident) : Except CheckError Ty :=
  match compatibleField target field with
  | some ty => Except.ok ty
  | none => Except.error (CheckError.fieldMismatch target field)

def compatibleMethodDetailed (target : Ty) (method : Ident) (arity : Nat) :
    Except CheckError Ty :=
  match compatibleMethod target method arity with
  | some ty => Except.ok ty
  | none => Except.error (CheckError.methodMismatch target method arity)

def checkConditionTyDetailed (ty : Ty) : Except CheckError Unit :=
  if conditionCompatible ty then
    Except.ok ()
  else
    Except.error (CheckError.conditionMismatch ty)

mutual
  partial def checkArgDetailed (vars : VarEnv) (fns : FnSigEnv) :
      Arg -> Except CheckError Unit
    | Arg.positional expr => do
        let _ <- checkExprDetailed vars fns expr
        Except.ok ()
    | Arg.named _ expr => do
        let _ <- checkExprDetailed vars fns expr
        Except.ok ()

  partial def checkArgsDetailed (vars : VarEnv) (fns : FnSigEnv) :
      List Arg -> Except CheckError Unit
    | [] => Except.ok ()
    | arg :: rest => do
        let _ <- checkArgDetailed vars fns arg
        checkArgsDetailed vars fns rest

  partial def inferArgTysDetailed (vars : VarEnv) (fns : FnSigEnv) :
      List Arg -> Except CheckError (List Ty)
    | [] => Except.ok []
    | Arg.positional expr :: rest => do
        let ty <- checkExprDetailed vars fns expr
        let tys <- inferArgTysDetailed vars fns rest
        Except.ok (ty :: tys)
    | Arg.named _ expr :: rest => do
        let ty <- checkExprDetailed vars fns expr
        let tys <- inferArgTysDetailed vars fns rest
        Except.ok (ty :: tys)

  partial def checkListElemsDetailed (vars : VarEnv) (fns : FnSigEnv) :
      List Expr -> Except CheckError Ty
    | [] => Except.ok Ty.unknown
    | [expr] => checkExprDetailed vars fns expr
    | expr :: rest => do
        let elemTy <- checkExprDetailed vars fns expr
        let restTy <- checkListElemsDetailed vars fns rest
        Except.ok (mergeElementTy elemTy restTy)

  partial def checkExprDetailed (vars : VarEnv) (fns : FnSigEnv) :
      Expr -> Except CheckError Ty
    | Expr.num _ => Except.ok Ty.num
    | Expr.float _ _ => Except.ok Ty.num
    | Expr.bool _ => Except.ok Ty.bool
    | Expr.str _ => Except.ok Ty.str
    | Expr.unit => Except.ok Ty.unit
    | Expr.list exprs => do
        let elemTy <- checkListElemsDetailed vars fns exprs
        Except.ok (Ty.list elemTy)
    | Expr.var name =>
        match VarEnv.lookup vars name with
        | some ty => Except.ok ty
        | none => Except.error (CheckError.undeclaredVariable name)
    | Expr.unary op expr => do
        let valueTy <- checkExprDetailed vars fns expr
        compatibleUnOpDetailed op valueTy
    | Expr.binary left op right => do
        let leftTy <- checkExprDetailed vars fns left
        let rightTy <- checkExprDetailed vars fns right
        compatibleBinOpDetailed op leftTy rightTy
    | Expr.index target index => do
        let targetTy <- checkExprDetailed vars fns target
        let indexTy <- checkExprDetailed vars fns index
        compatibleIndexDetailed targetTy indexTy
    | Expr.field target field => do
        let targetTy <- checkExprDetailed vars fns target
        compatibleFieldDetailed targetTy field
    | Expr.method target method args => do
        let targetTy <- checkExprDetailed vars fns target
        let argTys <- inferArgTysDetailed vars fns args
        match compatibleMethodWithArgs targetTy method argTys with
        | some ty => Except.ok ty
        | none => Except.error (CheckError.methodMismatch targetTy method args.length)
    | Expr.call name args =>
        match FnSigEnv.lookup fns name with
        | none => Except.error (CheckError.undeclaredFunction name)
        | some sig =>
            if sig.arity == args.length then do
              let _ <- checkNamedArgsDetailed name sig.params [] args
              let argTys <- inferArgTysDetailed vars fns args
              let _ <- checkCallArgTysDetailed name sig.params sig.paramTys args argTys
              Except.ok sig.result
            else
              Except.error (CheckError.arityMismatch name sig.arity args.length)

  partial def checkDeclaredReturnUnitFallthroughDetailed
      (fnName : Ident)
      (expected : Ty) :
      Except CheckError Unit :=
    if assignable expected Ty.unit then
      Except.ok ()
    else
      Except.error (CheckError.returnMismatch fnName expected Ty.unit)

  partial def checkDeclaredReturnStmtPathsDetailed
      (fnName : Ident)
      (expected : Ty)
      (vars : VarEnv)
      (fns : FnSigEnv) :
      Stmt -> Except CheckError Unit
    | Stmt.ret none =>
        checkDeclaredReturnUnitFallthroughDetailed fnName expected
    | Stmt.ret (some expr) => do
        let actual <- checkExprDetailed vars fns expr
        if assignable expected actual then
          Except.ok ()
        else
          Except.error (CheckError.returnMismatch fnName expected actual)
    | Stmt.ifThenElse _ thenBranch elseBranch => do
        let _ <- checkDeclaredReturnBlockPathsDetailed fnName expected vars fns thenBranch
        match elseBranch with
        | none => Except.ok ()
        | some statements => checkDeclaredReturnBlockPathsDetailed fnName expected vars fns statements
    | Stmt.while _ body => do
        checkDeclaredReturnBlockPathsDetailed fnName expected vars fns body
    | Stmt.forRange iterator _ _ body => do
        checkDeclaredReturnBlockPathsDetailed fnName expected (VarEnv.bind vars iterator Ty.num) fns body
    | Stmt.seal none body =>
        checkDeclaredReturnBlockPathsDetailed fnName expected vars fns body
    | Stmt.seal (some _) body => do
        checkDeclaredReturnBlockPathsDetailed fnName expected vars fns body
    | _ => Except.ok ()

  partial def checkDeclaredReturnBlockPathsDetailed
      (fnName : Ident)
      (expected : Ty)
      (vars : VarEnv)
      (fns : FnSigEnv) :
      List Stmt -> Except CheckError Unit
    | [] => Except.ok ()
    | stmt :: rest => do
        let _ <- checkDeclaredReturnStmtPathsDetailed fnName expected vars fns stmt
        let nextVars :=
          match inferStmtVars? vars fns stmt with
          | some updated => updated
          | none => vars
        checkDeclaredReturnBlockPathsDetailed fnName expected nextVars fns rest

  partial def checkDeclaredReturnBlockDetailed
      (fnName : Ident)
      (expected : Ty)
      (vars : VarEnv)
      (fns : FnSigEnv) :
      List Stmt -> Except CheckError Unit
    | [] => checkDeclaredReturnUnitFallthroughDetailed fnName expected
    | [Stmt.expr expr] => do
        let actual <- checkExprDetailed vars fns expr
        if assignable expected actual then
          Except.ok ()
        else
          Except.error (CheckError.returnMismatch fnName expected actual)
    | [Stmt.ret returnValue] =>
        checkDeclaredReturnStmtPathsDetailed fnName expected vars fns (Stmt.ret returnValue)
    | [Stmt.ifThenElse _ thenBranch elseBranch] => do
        let _ <- checkDeclaredReturnBlockDetailed fnName expected vars fns thenBranch
        match elseBranch with
        | none => checkDeclaredReturnUnitFallthroughDetailed fnName expected
        | some statements => checkDeclaredReturnBlockDetailed fnName expected vars fns statements
    | [Stmt.while _ body] => do
        let _ <- checkDeclaredReturnBlockPathsDetailed fnName expected vars fns body
        checkDeclaredReturnUnitFallthroughDetailed fnName expected
    | [Stmt.forRange iterator _ _ body] => do
        let _ <- checkDeclaredReturnBlockPathsDetailed fnName expected (VarEnv.bind vars iterator Ty.num) fns body
        checkDeclaredReturnUnitFallthroughDetailed fnName expected
    | [Stmt.seal none body] =>
        checkDeclaredReturnBlockDetailed fnName expected vars fns body
    | [Stmt.seal (some _) body] => do
        let _ <- checkDeclaredReturnBlockPathsDetailed fnName expected vars fns body
        checkDeclaredReturnUnitFallthroughDetailed fnName expected
    | [stmt] => do
        let _ <- checkDeclaredReturnStmtPathsDetailed fnName expected vars fns stmt
        checkDeclaredReturnUnitFallthroughDetailed fnName expected
    | stmt :: rest => do
        let _ <- checkDeclaredReturnStmtPathsDetailed fnName expected vars fns stmt
        let nextVars :=
          match inferStmtVars? vars fns stmt with
          | some updated => updated
          | none => vars
        checkDeclaredReturnBlockDetailed fnName expected nextVars fns rest

  partial def checkStmtDetailed (fns : FnSigEnv) (scope : Scope) (state : CheckState) :
      Stmt -> Except CheckError CheckState
    | Stmt.letDecl name expr => do
        let ty <- checkExprDetailed state.vars fns expr
        Except.ok { state with vars := VarEnv.bind state.vars name ty }
    | Stmt.letDeclTyped name annTy expr => do
        let expectedTy := annTyToTy annTy
        let valueTy <- checkExprDetailed state.vars fns expr
        if assignable expectedTy valueTy then
          Except.ok { state with vars := VarEnv.bind state.vars name (refineAssignedTy expectedTy valueTy) }
        else
          Except.error (CheckError.assignmentMismatch name expectedTy valueTy)
    | Stmt.assign name expr =>
        match VarEnv.lookup state.vars name with
        | none => Except.error (CheckError.undeclaredVariable name)
        | some targetTy => do
            let valueTy <- checkExprDetailed state.vars fns expr
            if assignable targetTy valueTy then
              match VarEnv.assign state.vars name (refineAssignedTy targetTy valueTy) with
              | some updatedVars => Except.ok { state with vars := updatedVars }
              | none => Except.error (CheckError.undeclaredVariable name)
            else
              Except.error (CheckError.assignmentMismatch name targetTy valueTy)
    | Stmt.ifThenElse condition thenBranch elseBranch => do
        let conditionTy <- checkExprDetailed state.vars fns condition
        let _ <- checkConditionTyDetailed conditionTy
        let blockScope := { scope with topLevel := false }
        let thenState <- checkBlockDetailed fns blockScope state thenBranch
        match elseBranch with
        | none => Except.ok state
        | some statements => do
            let elseState <- checkBlockDetailed fns blockScope state statements
            Except.ok { state with vars := joinNewBranchVars state.vars thenState.vars elseState.vars }
    | Stmt.while condition body => do
        let conditionTy <- checkExprDetailed state.vars fns condition
        let _ <- checkConditionTyDetailed conditionTy
        let loopScope := { scope with loopDepth := scope.loopDepth + 1, topLevel := false }
        let _ <- checkBlockDetailed fns loopScope state body
        Except.ok state
    | Stmt.forRange iterator _ _ body => do
        let loopScope := { scope with loopDepth := scope.loopDepth + 1, topLevel := false }
        let bodyState := { state with vars := VarEnv.bind state.vars iterator Ty.num }
        let _ <- checkBlockDetailed fns loopScope bodyState body
        Except.ok state
    | Stmt.seal condition body => do
        match condition with
        | none => pure ()
        | some expr =>
            let conditionTy <- checkExprDetailed state.vars fns expr
            let _ <- checkConditionTyDetailed conditionTy
            pure ()
        let loopScope := { scope with loopDepth := scope.loopDepth + 1, topLevel := false }
        let _ <- checkBlockDetailed fns loopScope state body
        Except.ok state
    | Stmt.fnDecl name params body =>
        if !scope.topLevel then
          Except.error (CheckError.nestedFunction name)
        else
          match FnSigEnv.lookup fns name with
          | none => Except.error (CheckError.undeclaredFunction name)
          | some sig =>
              if sig.arity == params.length then do
                let fnScope := { loopDepth := 0, functionDepth := scope.functionDepth + 1, topLevel := false }
                let paramVars <- bindUnknownParamsDetailed [] params
                let fnState := { vars := paramVars }
                let _ <- checkBlockDetailed fns fnScope fnState body
                Except.ok state
              else
                Except.error (CheckError.functionSignatureMismatch name sig.arity params.length)
    | Stmt.fnDeclReturn name params returnTy body =>
        if !scope.topLevel then
          Except.error (CheckError.nestedFunction name)
        else
          match FnSigEnv.lookup fns name with
          | none => Except.error (CheckError.undeclaredFunction name)
          | some sig =>
              if sig.arity == params.length then do
                let fnScope := { loopDepth := 0, functionDepth := scope.functionDepth + 1, topLevel := false }
                let paramVars <- bindUnknownParamsDetailed [] params
                let fnState := { vars := paramVars }
                let _ <- checkBlockDetailed fns fnScope fnState body
                let expected := annTyToTy returnTy
                let _ <- checkDeclaredReturnBlockDetailed name expected fnState.vars fns body
                let actual :=
                  match inferBlockReturnTy? fnState.vars fns body with
                  | some found => found
                  | none => Ty.unit
                if compatibleTy expected actual then
                  Except.ok state
                else
                  Except.error (CheckError.returnMismatch name expected actual)
              else
                Except.error (CheckError.functionSignatureMismatch name sig.arity params.length)
    | Stmt.fnDeclTyped name params body =>
        if !scope.topLevel then
          Except.error (CheckError.nestedFunction name)
        else
          match FnSigEnv.lookup fns name with
          | none => Except.error (CheckError.undeclaredFunction name)
          | some sig =>
              if sig.arity == params.length then do
                let fnScope := { loopDepth := 0, functionDepth := scope.functionDepth + 1, topLevel := false }
                let paramVars <- bindTypedParamsDetailed [] params
                let fnState := { vars := paramVars }
                let _ <- checkBlockDetailed fns fnScope fnState body
                Except.ok state
              else
                Except.error (CheckError.functionSignatureMismatch name sig.arity params.length)
    | Stmt.fnDeclTypedReturn name params returnTy body =>
        if !scope.topLevel then
          Except.error (CheckError.nestedFunction name)
        else
          match FnSigEnv.lookup fns name with
          | none => Except.error (CheckError.undeclaredFunction name)
          | some sig =>
              if sig.arity == params.length then do
                let fnScope := { loopDepth := 0, functionDepth := scope.functionDepth + 1, topLevel := false }
                let paramVars <- bindTypedParamsDetailed [] params
                let fnState := { vars := paramVars }
                let _ <- checkBlockDetailed fns fnScope fnState body
                let expected := annTyToTy returnTy
                let _ <- checkDeclaredReturnBlockDetailed name expected fnState.vars fns body
                let actual :=
                  match inferBlockReturnTy? fnState.vars fns body with
                  | some found => found
                  | none => Ty.unit
                if compatibleTy expected actual then
                  Except.ok state
                else
                  Except.error (CheckError.returnMismatch name expected actual)
              else
                Except.error (CheckError.functionSignatureMismatch name sig.arity params.length)
    | Stmt.ret returnValue =>
        if scope.functionDepth == 0 then
          Except.error CheckError.returnOutsideFunction
        else
          match returnValue with
          | none => Except.ok state
          | some expr => do
              let _ <- checkExprDetailed state.vars fns expr
              Except.ok state
    | Stmt.break =>
        if scope.loopDepth == 0 then
          Except.error CheckError.breakOutsideLoop
        else
          Except.ok state
    | Stmt.continue =>
        if scope.loopDepth == 0 then
          Except.error CheckError.continueOutsideLoop
        else
          Except.ok state
    | Stmt.expr expr => do
        let _ <- checkExprDetailed state.vars fns expr
        Except.ok state

  partial def checkBlockDetailed (fns : FnSigEnv) (scope : Scope) (state : CheckState) :
      List Stmt -> Except CheckError CheckState
    | [] => Except.ok state
    | stmt :: rest => do
        let next <- checkStmtDetailed fns scope state stmt
        checkBlockDetailed fns scope next rest
end

def checkProgramWithFnsDetailed (fns : FnSigEnv) (stmts : List Stmt) :
    Except CheckError CheckState :=
  checkBlockDetailed fns { loopDepth := 0, functionDepth := 0, topLevel := true } { vars := [] } stmts

def checkProgramDetailed (stmts : List Stmt) : Except CheckError CheckState := do
  let fns <- collectFnSigsDetailed stmts []
  checkProgramWithFnsDetailed (inferFnSigResults stmts fns) stmts

def checkProgram (stmts : List Stmt) : Option CheckState :=
  match checkProgramDetailed stmts with
  | Except.ok state => some state
  | Except.error _ => none

example :
    checkExpr [] [] (Expr.binary (Expr.num 2) BinOp.add (Expr.num 3)) = some Ty.num := by
  native_decide

example :
    checkExpr [] [] (Expr.binary (Expr.float 1 500000) BinOp.add (Expr.num 2)) = some Ty.num := by
  native_decide

example :
    checkExpr [] [] (Expr.binary (Expr.num 2) BinOp.add (Expr.bool true)) = none := by
  native_decide

example :
    checkExpr [] [] (Expr.str "open") = some Ty.str := by
  native_decide

example :
    checkExpr [] [] Expr.unit = some Ty.unit := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.binary (Expr.str "open") BinOp.add (Expr.str "done"))
      == Except.error (CheckError.operandMismatch BinOp.add Ty.str Ty.str)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.binary (Expr.str "open") BinOp.eq (Expr.str "open"))
      == Except.ok Ty.bool) := by
  native_decide

example :
    checkExpr [] [] (Expr.list [Expr.num 1, Expr.num 2]) = some (Ty.list Ty.num) := by
  native_decide

example :
    checkExpr [] [] (Expr.list [Expr.str "open", Expr.str "done"]) = some (Ty.list Ty.str) := by
  native_decide

example :
    checkExpr [] [] (Expr.list [Expr.num 1, Expr.bool true]) = some (Ty.list Ty.unknown) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.list [Expr.var "missing"])
      == Except.error (CheckError.undeclaredVariable "missing")) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.binary (Expr.list [Expr.num 1]) BinOp.add (Expr.list [Expr.num 2]))
      == Except.error (CheckError.operandMismatch BinOp.add (Ty.list Ty.num) (Ty.list Ty.num))) := by
  native_decide

example :
    checkExpr [] [] (Expr.index (Expr.list [Expr.num 1]) (Expr.num 0)) = some Ty.num := by
  native_decide

example :
    checkExpr [] [] (Expr.index (Expr.str "open") (Expr.num 1)) = some Ty.str := by
  native_decide

example :
    (checkProgramDetailed
        [ Stmt.letDecl "xs" (Expr.list [Expr.num 1])
        , Stmt.letDecl "x" (Expr.index (Expr.var "xs") (Expr.num 0))
        , Stmt.assign "x" (Expr.str "bad")
        ]
      == Except.error (CheckError.assignmentMismatch "x" Ty.num Ty.str)) := by
  native_decide

example :
    (checkProgramDetailed
        [ Stmt.letDeclTyped "count" AnnTy.num (Expr.bool true) ]
      == Except.error (CheckError.assignmentMismatch "count" Ty.num Ty.bool)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.index (Expr.list [Expr.num 1]) (Expr.bool true))
      == Except.error (CheckError.indexMismatch (Ty.list Ty.num) Ty.bool)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.index (Expr.str "open") (Expr.bool true))
      == Except.error (CheckError.indexMismatch Ty.str Ty.bool)) := by
  native_decide

example :
    checkExpr [] [] (Expr.field (Expr.list [Expr.num 1]) "length") = some Ty.num := by
  native_decide

example :
    checkExpr [] [] (Expr.field (Expr.str "open") "length") = some Ty.num := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.field (Expr.num 1) "length")
      == Except.error (CheckError.fieldMismatch Ty.num "length")) := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.list [Expr.num 1]) "len" []) = some Ty.num := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.str "open") "len" []) = some Ty.num := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.list []) "is_empty" []) = some Ty.bool := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.str "open") "is_empty" []) = some Ty.bool := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.list [Expr.num 1]) "first" []) = some Ty.num := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.list [Expr.str "open"]) "first" []) = some Ty.str := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.list [Expr.num 1]) "last" []) = some Ty.num := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.list [Expr.str "open"]) "last" []) = some Ty.str := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.list [Expr.num 1]) "at" [Expr.num 0]) = some Ty.num := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.list [Expr.num 1]) "at" [Expr.bool true])
      == Except.error (CheckError.methodMismatch (Ty.list Ty.num) "at" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.num 1) "len" [])
      == Except.error (CheckError.methodMismatch Ty.num "len" 0)) := by
  native_decide

example :
    checkProgram [Stmt.assign "x" (Expr.num 1)] = none := by
  native_decide

example :
    checkProgram
      [ Stmt.fnDecl "id" ["x"] [Stmt.ret (some (Expr.var "x"))]
      , Stmt.letDecl "x" (Expr.call "id" [Expr.num 0])
      , Stmt.assign "x" (Expr.num 1)
      ]
      =
      some { vars := [("x", Ty.num)] } := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDecl "id" ["x"] [Stmt.ret (some (Expr.var "x"))]
      , Stmt.letDecl "x" (Expr.call "id" [Expr.num 0])
      , Stmt.assign "x" (Expr.num 1)
      , Stmt.assign "x" (Expr.str "bad")
      ]
      == Except.error (CheckError.assignmentMismatch "x" Ty.num Ty.str)) := by
  native_decide

example :
    checkProgram [Stmt.break] = none := by
  native_decide

example :
    checkProgram
      [ Stmt.letDecl "x" (Expr.num 0)
      , Stmt.while
          (Expr.binary (Expr.var "x") BinOp.lt (Expr.num 10))
          [ Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))
          , Stmt.break
          ]
      ]
      =
      some { vars := [("x", Ty.num)] } := by
  native_decide

example :
    checkProgram
      [ Stmt.fnDecl
          "add"
          ["a", "b"]
          [Stmt.ret (some (Expr.binary (Expr.var "a") BinOp.add (Expr.var "b")))]
      , Stmt.letDecl "x" (Expr.call "add" [Expr.num 2, Expr.num 3])
      ]
      =
      some { vars := [("x", Ty.num)] } := by
  native_decide

example :
    checkExpr [] [("one", { arity := 0, params := [], paramTys := [], result := Ty.num })] (Expr.call "one" []) = some Ty.num := by
  native_decide

example :
    checkProgram
      [ Stmt.fnDecl "one" [] [Stmt.ret (some (Expr.num 1))]
      , Stmt.letDecl "x" (Expr.call "one" [])
      ]
      =
      some { vars := [("x", Ty.num)] } := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDecl "one" [] [Stmt.ret (some (Expr.num 1))]
      , Stmt.letDecl "x" (Expr.call "one" [])
      , Stmt.assign "x" (Expr.str "bad")
      ]
      == Except.error (CheckError.assignmentMismatch "x" Ty.num Ty.str)) := by
  native_decide

example :
    checkProgram
      [ Stmt.fnDecl "implicitOne" [] [Stmt.letDecl "x" (Expr.num 1), Stmt.expr (Expr.var "x")]
      , Stmt.letDecl "x" (Expr.call "implicitOne" [])
      ]
      =
      some { vars := [("x", Ty.num)] } := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDecl "implicitLabel" [] [Stmt.expr (Expr.str "ready")]
      , Stmt.letDecl "label" (Expr.call "implicitLabel" [])
      , Stmt.assign "label" (Expr.num 1)
      ]
      == Except.error (CheckError.assignmentMismatch "label" Ty.str Ty.num)) := by
  native_decide

example :
    checkProgram
      [ Stmt.fnDecl
          "branchValue"
          []
          [ Stmt.ifThenElse
              (Expr.bool true)
              [Stmt.letDecl "x" (Expr.num 1)]
              (some [Stmt.letDecl "x" (Expr.num 2)])
          , Stmt.expr (Expr.var "x")
          ]
      , Stmt.letDecl "x" (Expr.call "branchValue" [])
      ]
      =
      some { vars := [("x", Ty.num)] } := by
  native_decide

example :
    checkProgram
      [ Stmt.fnDecl
          "branchMismatch"
          []
          [ Stmt.ifThenElse
              (Expr.bool true)
              [Stmt.letDecl "x" (Expr.num 1)]
              (some [Stmt.letDecl "x" (Expr.str "bad")])
          , Stmt.expr (Expr.var "x")
          ]
      , Stmt.letDecl "x" (Expr.call "branchMismatch" [])
      ]
      =
      none := by
  native_decide

example :
    checkProgram
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
    checkProgram
      [Stmt.ret (some (Expr.num 1))]
      =
      none := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.binary (Expr.num 2) BinOp.add (Expr.bool true))
      == Except.error (CheckError.operandMismatch BinOp.add Ty.num Ty.bool)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.binary (Expr.num 1) BinOp.and (Expr.bool true))
      == Except.error (CheckError.operandMismatch BinOp.and Ty.num Ty.bool)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.binary (Expr.bool false) BinOp.or (Expr.num 1))
      == Except.error (CheckError.operandMismatch BinOp.or Ty.bool Ty.num)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.binary (Expr.num 1) BinOp.eq (Expr.bool true))
      == Except.error (CheckError.operandMismatch BinOp.eq Ty.num Ty.bool)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.binary (Expr.bool false) BinOp.neq (Expr.num 1))
      == Except.error (CheckError.operandMismatch BinOp.neq Ty.bool Ty.num)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.unary UnOp.not (Expr.num 1))
      == Except.error (CheckError.unaryMismatch UnOp.not Ty.num)) := by
  native_decide

example :
    (checkProgramDetailed [Stmt.assign "x" (Expr.num 1)]
      == Except.error (CheckError.undeclaredVariable "x")) := by
  native_decide

example :
    (checkProgramDetailed [Stmt.break]
      == Except.error CheckError.breakOutsideLoop) := by
  native_decide

example :
    (checkProgramDetailed [Stmt.continue]
      == Except.error CheckError.continueOutsideLoop) := by
  native_decide

example :
    (checkProgramDetailed [Stmt.ret (some (Expr.num 1))]
      == Except.error CheckError.returnOutsideFunction) := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.list [Expr.num 1]) "contains" [Expr.num 1]) =
      some Ty.bool := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.list [Expr.num 1, Expr.num 2]) "tail" []) =
      some (Ty.list Ty.num) := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.list [Expr.num 1, Expr.num 2]) "take" [Expr.num 1]) =
      some (Ty.list Ty.num) := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.list [Expr.num 1, Expr.num 2]) "drop" [Expr.num 1]) =
      some (Ty.list Ty.num) := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.list [Expr.num 1, Expr.num 2]) "reverse" []) =
      some (Ty.list Ty.num) := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.list [Expr.num 1]) "append" [Expr.num 2]) =
      some (Ty.list Ty.num) := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.list [Expr.num 2]) "prepend" [Expr.num 1]) =
      some (Ty.list Ty.num) := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.list [Expr.str "a", Expr.str "b"]) "join" [Expr.str ","]) =
      some Ty.str := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.list [Expr.num 1]) "concat" [Expr.list [Expr.num 2]]) =
      some (Ty.list Ty.num) := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.str "open") "at" [Expr.num 1]) =
      some Ty.str := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.str "open") "contains" [Expr.str "pe"]) =
      some Ty.bool := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.str "open") "starts_with" [Expr.str "op"]) =
      some Ty.bool := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.str "open") "ends_with" [Expr.str "en"]) =
      some Ty.bool := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.str "open") "reverse" []) =
      some Ty.str := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.str "open") "first" []) =
      some Ty.str := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.str "open") "last" []) =
      some Ty.str := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.str "open") "tail" []) =
      some Ty.str := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.str "open") "take" [Expr.num 2]) =
      some Ty.str := by
  native_decide

example :
    checkExpr [] [] (Expr.method (Expr.str "open") "drop" [Expr.num 2]) =
      some Ty.str := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.list [Expr.num 1]) "contains" [Expr.bool true])
      == Except.error (CheckError.methodMismatch (Ty.list Ty.num) "contains" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.list [Expr.num 1]) "tail" [Expr.num 0])
      == Except.error (CheckError.methodMismatch (Ty.list Ty.num) "tail" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.list [Expr.num 1]) "take" [Expr.bool true])
      == Except.error (CheckError.methodMismatch (Ty.list Ty.num) "take" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.list [Expr.num 1]) "drop" [Expr.bool true])
      == Except.error (CheckError.methodMismatch (Ty.list Ty.num) "drop" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.list [Expr.num 1]) "reverse" [Expr.num 0])
      == Except.error (CheckError.methodMismatch (Ty.list Ty.num) "reverse" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.list [Expr.num 1]) "append" [Expr.bool true])
      == Except.error (CheckError.methodMismatch (Ty.list Ty.num) "append" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.list [Expr.num 1]) "prepend" [Expr.bool true])
      == Except.error (CheckError.methodMismatch (Ty.list Ty.num) "prepend" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.list [Expr.num 1]) "join" [Expr.str ","])
      == Except.error (CheckError.methodMismatch (Ty.list Ty.num) "join" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.list [Expr.num 1]) "concat" [Expr.list [Expr.bool true]])
      == Except.error (CheckError.methodMismatch (Ty.list Ty.num) "concat" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.str "open") "at" [Expr.bool true])
      == Except.error (CheckError.methodMismatch Ty.str "at" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.str "open") "contains" [Expr.num 1])
      == Except.error (CheckError.methodMismatch Ty.str "contains" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.str "open") "starts_with" [Expr.num 1])
      == Except.error (CheckError.methodMismatch Ty.str "starts_with" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.str "open") "ends_with" [Expr.num 1])
      == Except.error (CheckError.methodMismatch Ty.str "ends_with" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.str "open") "reverse" [Expr.num 1])
      == Except.error (CheckError.methodMismatch Ty.str "reverse" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.str "open") "first" [Expr.num 0])
      == Except.error (CheckError.methodMismatch Ty.str "first" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.str "open") "last" [Expr.num 0])
      == Except.error (CheckError.methodMismatch Ty.str "last" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.str "open") "tail" [Expr.num 0])
      == Except.error (CheckError.methodMismatch Ty.str "tail" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.str "open") "take" [Expr.bool true])
      == Except.error (CheckError.methodMismatch Ty.str "take" 1)) := by
  native_decide

example :
    (checkExprDetailed [] [] (Expr.method (Expr.str "open") "drop" [Expr.bool true])
      == Except.error (CheckError.methodMismatch Ty.str "drop" 1)) := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDecl
          "id"
          ["x"]
          [Stmt.ret (some (Expr.var "x"))]
      , Stmt.letDecl "bad" (Expr.call "id" [Expr.num 1, Expr.num 2])
      ]
      == Except.error (CheckError.arityMismatch "id" 1 2)) := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDeclTyped
          "id"
          [("x", AnnTy.num)]
          [Stmt.ret (some (Expr.var "x"))]
      , Stmt.letDecl "bad" (Expr.call "id" [Expr.bool true])
      ]
      == Except.error (CheckError.argumentMismatch "id" "x" Ty.num Ty.bool)) := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDeclTypedReturn
          "bad"
          [("x", AnnTy.num)]
          AnnTy.num
          [Stmt.ret (some (Expr.bool true))]
      ]
      == Except.error (CheckError.returnMismatch "bad" Ty.num Ty.bool)) := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDeclReturn
          "bad"
          ["x"]
          AnnTy.num
          [Stmt.ret (some (Expr.bool true))]
      ]
      == Except.error (CheckError.returnMismatch "bad" Ty.num Ty.bool)) := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDeclReturn
          "one"
          []
          AnnTy.num
          [Stmt.ret (some (Expr.num 1))]
      , Stmt.letDeclTyped "bad" AnnTy.bool (Expr.call "one" [])
      ]
      == Except.error (CheckError.assignmentMismatch "bad" Ty.bool Ty.num)) := by
  native_decide

example :
    checkProgram
      [ Stmt.fnDeclTypedReturn
          "bad"
          [("x", AnnTy.num)]
          AnnTy.num
          [Stmt.ret (some (Expr.bool true))]
      ]
      =
      none := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDeclTypedReturn
          "bad"
          []
          AnnTy.num
          [ Stmt.ifThenElse
              (Expr.bool true)
              [Stmt.ret (some (Expr.num 1))]
              (some [Stmt.ret (some (Expr.bool true))])
          ]
      ]
      == Except.error (CheckError.returnMismatch "bad" Ty.num Ty.bool)) := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDeclTypedReturn
          "maybe"
          [("b", AnnTy.bool)]
          AnnTy.num
          [ Stmt.ifThenElse
              (Expr.var "b")
              [Stmt.ret (some (Expr.num 1))]
              none
          ]
      ]
      == Except.error (CheckError.returnMismatch "maybe" Ty.num Ty.unit)) := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDeclTypedReturn
          "loopReturn"
          [("b", AnnTy.bool)]
          AnnTy.num
          [ Stmt.while
              (Expr.var "b")
              [Stmt.ret (some (Expr.num 1))]
          ]
      ]
      == Except.error (CheckError.returnMismatch "loopReturn" Ty.num Ty.unit)) := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDeclTypedReturn
          "rangeReturn"
          []
          AnnTy.num
          [ Stmt.forRange
              "i"
              0
              0
              [Stmt.ret (some (Expr.num 1))]
          ]
      ]
      == Except.error (CheckError.returnMismatch "rangeReturn" Ty.num Ty.unit)) := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDeclTypedReturn
          "sealReturn"
          [("b", AnnTy.bool)]
          AnnTy.num
          [ Stmt.seal
              (some (Expr.var "b"))
              [Stmt.ret (some (Expr.num 1))]
          ]
      ]
      == Except.error (CheckError.returnMismatch "sealReturn" Ty.num Ty.unit)) := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDeclTypedReturn
          "guarded"
          [("b", AnnTy.bool)]
          AnnTy.num
          [ Stmt.ifThenElse
              (Expr.var "b")
              [Stmt.ret (some (Expr.num 1))]
              none
          , Stmt.ret (some (Expr.num 2))
          ]
      ]
      == Except.ok { vars := [] }) := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDeclTypedReturn
          "len"
          [("xs", AnnTy.list AnnTy.num)]
          AnnTy.num
          [Stmt.ret (some (Expr.field (Expr.var "xs") "length"))]
      , Stmt.letDecl "bad" (Expr.call "len" [Expr.list [Expr.bool true]])
      ]
      == Except.error
        (CheckError.argumentMismatch "len" "xs" (Ty.list Ty.num) (Ty.list Ty.bool))) := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDecl
          "pick"
          ["a", "b"]
          [Stmt.ret (some (Expr.var "a"))]
      , Stmt.letDecl "bad" (Expr.call "pick" [Arg.named "c" (Expr.num 1), Arg.named "a" (Expr.num 2)])
      ]
      == Except.error (CheckError.unknownNamedArgument "pick" "c")) := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDecl
          "pick"
          ["a", "b"]
          [Stmt.ret (some (Expr.var "a"))]
      , Stmt.letDecl "bad" (Expr.call "pick" [Arg.named "a" (Expr.num 1), Arg.named "a" (Expr.num 2)])
      ]
      == Except.error (CheckError.duplicateNamedArgument "pick" "a")) := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.fnDecl "dup" [] [Stmt.ret (some (Expr.num 1))]
      , Stmt.fnDecl "dup" [] [Stmt.ret (some (Expr.num 2))]
      ]
      == Except.error (CheckError.duplicateFunction "dup")) := by
  native_decide

example :
    (checkProgramDetailed
      [Stmt.fnDecl "bad" ["x", "x"] [Stmt.ret (some (Expr.var "x"))]]
      == Except.error (CheckError.duplicateParameter "x")) := by
  native_decide

example :
    (checkProgramDetailed [Stmt.ifThenElse (Expr.num 1) [] none]
      == Except.error (CheckError.conditionMismatch Ty.num)) := by
  native_decide

example :
    checkProgram
      [ Stmt.ifThenElse
          (Expr.bool true)
          [Stmt.letDecl "x" (Expr.num 1)]
          (some [Stmt.letDecl "x" (Expr.num 2)])
      , Stmt.letDecl "y" (Expr.var "x")
      ]
      =
      some { vars := [("y", Ty.num), ("x", Ty.num)] } := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.ifThenElse
          (Expr.bool true)
          [Stmt.letDecl "x" (Expr.num 1)]
          (some [Stmt.letDecl "x" (Expr.str "bad")])
      , Stmt.letDecl "y" (Expr.var "x")
      ]
      == Except.error (CheckError.undeclaredVariable "x")) := by
  native_decide

example :
    checkProgram
      [ Stmt.fnDecl "id" ["x"] [Stmt.ret (some (Expr.var "x"))]
      , Stmt.letDecl "x" (Expr.call "id" [Expr.num 0])
      , Stmt.ifThenElse
          (Expr.bool true)
          [Stmt.assign "x" (Expr.num 1)]
          (some [Stmt.assign "x" (Expr.num 2)])
      ]
      =
      some { vars := [("x", Ty.num)] } := by
  native_decide

example :
    checkProgram
      [ Stmt.fnDecl "id" ["x"] [Stmt.ret (some (Expr.var "x"))]
      , Stmt.letDecl "x" (Expr.call "id" [Expr.num 0])
      , Stmt.ifThenElse
          (Expr.bool true)
          [Stmt.assign "x" (Expr.num 1)]
          (some [Stmt.assign "x" (Expr.str "bad")])
      ]
      =
      some { vars := [("x", Ty.unknown)] } := by
  native_decide

example :
    (checkProgramDetailed [Stmt.while (Expr.num 1) []]
      == Except.error (CheckError.conditionMismatch Ty.num)) := by
  native_decide

example :
    (checkProgramDetailed [Stmt.seal (some (Expr.num 1)) []]
      == Except.error (CheckError.conditionMismatch Ty.num)) := by
  native_decide

example :
    (checkProgramDetailed
      [ Stmt.ifThenElse
          (Expr.bool true)
          [Stmt.fnDecl "nested" [] [Stmt.ret (some (Expr.num 1))]]
          none
      ]
      == Except.error (CheckError.nestedFunction "nested")) := by
  native_decide

end Static
end Aether
