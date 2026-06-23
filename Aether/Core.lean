namespace Aether

abbrev Ident := String

inductive BinOp where
  | add
  | sub
  | mul
  | div
  | mod
  | eq
  | neq
  | lt
  | gt
  | le
  | ge
  | and
  | or
  deriving Repr, BEq, DecidableEq

inductive UnOp where
  | neg
  | not
  deriving Repr, BEq, DecidableEq

inductive AnnTy where
  | num
  | bool
  | str
  | unit
  | list : AnnTy -> AnnTy
  deriving Repr, BEq, DecidableEq

mutual
  inductive Expr where
    | num : Int -> Expr
    | float : Int -> Int -> Expr
    | bool : Bool -> Expr
    | str : String -> Expr
    | unit : Expr
    | list : List Expr -> Expr
    | index : Expr -> Expr -> Expr
    | field : Expr -> Ident -> Expr
    | method : Expr -> Ident -> List Arg -> Expr
    | var : Ident -> Expr
    | unary : UnOp -> Expr -> Expr
    | binary : Expr -> BinOp -> Expr -> Expr
    | call : Ident -> List Arg -> Expr
    deriving Repr, BEq

  inductive Arg where
    | positional : Expr -> Arg
    | named : Ident -> Expr -> Arg
    deriving Repr, BEq

  inductive Stmt where
    | letDecl : Ident -> Expr -> Stmt
    | letDeclTyped : Ident -> AnnTy -> Expr -> Stmt
    | assign : Ident -> Expr -> Stmt
    | ifThenElse : Expr -> List Stmt -> Option (List Stmt) -> Stmt
    | while : Expr -> List Stmt -> Stmt
    | forRange : Ident -> Int -> Int -> List Stmt -> Stmt
    | seal : Option Expr -> List Stmt -> Stmt
    | fnDecl : Ident -> List Ident -> List Stmt -> Stmt
    | fnDeclReturn : Ident -> List Ident -> AnnTy -> List Stmt -> Stmt
    | fnDeclTyped : Ident -> List (Ident × AnnTy) -> List Stmt -> Stmt
    | fnDeclTypedReturn : Ident -> List (Ident × AnnTy) -> AnnTy -> List Stmt -> Stmt
    | ret : Option Expr -> Stmt
    | break : Stmt
    | continue : Stmt
    | expr : Expr -> Stmt
    deriving Repr, BEq
end

instance : Coe Expr Arg where
  coe := Arg.positional

inductive Value where
  | num : Int -> Value
  | float : Int -> Int -> Value
  | bool : Bool -> Value
  | str : String -> Value
  | list : List Value -> Value
  | unit : Value
  deriving Repr, BEq

mutual
  def Value.decEq : (left right : Value) -> Decidable (left = right)
    | Value.num left, Value.num right =>
        if h : left = right then isTrue (by cases h; rfl)
        else isFalse (by intro eq; cases eq; exact h rfl)
    | Value.float leftInt leftFrac, Value.float rightInt rightFrac =>
        if hInt : leftInt = rightInt then
          if hFrac : leftFrac = rightFrac then
            isTrue (by cases hInt; cases hFrac; rfl)
          else
            isFalse (by intro eq; cases eq; exact hFrac rfl)
        else
          isFalse (by intro eq; cases eq; exact hInt rfl)
    | Value.bool left, Value.bool right =>
        if h : left = right then isTrue (by cases h; rfl)
        else isFalse (by intro eq; cases eq; exact h rfl)
    | Value.str left, Value.str right =>
        if h : left = right then isTrue (by cases h; rfl)
        else isFalse (by intro eq; cases eq; exact h rfl)
    | Value.list left, Value.list right =>
        match Value.decEqList left right with
        | isTrue h => isTrue (by cases h; rfl)
        | isFalse h => isFalse (by intro eq; cases eq; exact h rfl)
    | Value.unit, Value.unit => isTrue rfl
    | Value.num _, Value.float _ _ => isFalse (by intro h; cases h)
    | Value.num _, Value.bool _ => isFalse (by intro h; cases h)
    | Value.num _, Value.str _ => isFalse (by intro h; cases h)
    | Value.num _, Value.list _ => isFalse (by intro h; cases h)
    | Value.num _, Value.unit => isFalse (by intro h; cases h)
    | Value.float _ _, Value.num _ => isFalse (by intro h; cases h)
    | Value.float _ _, Value.bool _ => isFalse (by intro h; cases h)
    | Value.float _ _, Value.str _ => isFalse (by intro h; cases h)
    | Value.float _ _, Value.list _ => isFalse (by intro h; cases h)
    | Value.float _ _, Value.unit => isFalse (by intro h; cases h)
    | Value.bool _, Value.num _ => isFalse (by intro h; cases h)
    | Value.bool _, Value.float _ _ => isFalse (by intro h; cases h)
    | Value.bool _, Value.str _ => isFalse (by intro h; cases h)
    | Value.bool _, Value.list _ => isFalse (by intro h; cases h)
    | Value.bool _, Value.unit => isFalse (by intro h; cases h)
    | Value.str _, Value.num _ => isFalse (by intro h; cases h)
    | Value.str _, Value.float _ _ => isFalse (by intro h; cases h)
    | Value.str _, Value.bool _ => isFalse (by intro h; cases h)
    | Value.str _, Value.list _ => isFalse (by intro h; cases h)
    | Value.str _, Value.unit => isFalse (by intro h; cases h)
    | Value.list _, Value.num _ => isFalse (by intro h; cases h)
    | Value.list _, Value.float _ _ => isFalse (by intro h; cases h)
    | Value.list _, Value.bool _ => isFalse (by intro h; cases h)
    | Value.list _, Value.str _ => isFalse (by intro h; cases h)
    | Value.list _, Value.unit => isFalse (by intro h; cases h)
    | Value.unit, Value.num _ => isFalse (by intro h; cases h)
    | Value.unit, Value.float _ _ => isFalse (by intro h; cases h)
    | Value.unit, Value.bool _ => isFalse (by intro h; cases h)
    | Value.unit, Value.str _ => isFalse (by intro h; cases h)
    | Value.unit, Value.list _ => isFalse (by intro h; cases h)

  def Value.decEqList : (left right : List Value) -> Decidable (left = right)
    | [], [] => isTrue rfl
    | [], _ :: _ => isFalse (by intro h; cases h)
    | _ :: _, [] => isFalse (by intro h; cases h)
    | left :: leftRest, right :: rightRest =>
        match Value.decEq left right, Value.decEqList leftRest rightRest with
        | isTrue headEq, isTrue tailEq =>
            isTrue (by cases headEq; cases tailEq; rfl)
        | isFalse headNe, _ =>
            isFalse (by intro h; cases h; exact headNe rfl)
        | _, isFalse tailNe =>
            isFalse (by intro h; cases h; exact tailNe rfl)
end

instance : DecidableEq Value := Value.decEq

inductive Flow where
  | value : Value -> Flow
  | return : Value -> Flow
  | break : Flow
  | continue : Flow
  deriving Repr, BEq, DecidableEq

abbrev Env := List (Ident × Value)

structure Function where
  params : List Ident
  body : List Stmt
  deriving Repr, BEq

abbrev FnEnv := List (Ident × Function)

def Env.lookup (env : Env) (name : Ident) : Option Value :=
  match env with
  | [] => none
  | (key, value) :: rest =>
      if key == name then some value else Env.lookup rest name

def Env.bind (env : Env) (name : Ident) (value : Value) : Env :=
  (name, value) :: env

def Env.assign (env : Env) (name : Ident) (value : Value) : Option Env :=
  match env with
  | [] => none
  | (key, old) :: rest =>
      if key == name then
        some ((key, value) :: rest)
      else
        match Env.assign rest name value with
        | some updated => some ((key, old) :: updated)
        | none => none

def FnEnv.lookup (fns : FnEnv) (name : Ident) : Option Function :=
  match fns with
  | [] => none
  | (key, fn) :: rest =>
      if key == name then some fn else FnEnv.lookup rest name

def FnEnv.bind (fns : FnEnv) (name : Ident) (fn : Function) : FnEnv :=
  (name, fn) :: fns

def truthy : Value -> Bool
  | Value.bool b => b
  | Value.num n => n != 0
  | Value.float intPart fracMicros => intPart != 0 || fracMicros != 0
  | Value.str value => value != ""
  | Value.list values => values != []
  | Value.unit => false

def microsPerUnit : Int := 1000000

def numericMicros? : Value -> Option Int
  | Value.num n => some (n * microsPerUnit)
  | Value.float intPart fracMicros => some (intPart * microsPerUnit + fracMicros)
  | _ => none

def floatFromMicros (micros : Int) : Value :=
  Value.float (micros / microsPerUnit) (Int.emod micros microsPerUnit)

def numericResult (hasFloat : Bool) (micros : Int) : Value :=
  if hasFloat then floatFromMicros micros else Value.num (micros / microsPerUnit)

def hasFloat : Value -> Bool
  | Value.float _ _ => true
  | _ => false

def valuesEq : Value -> Value -> Bool
  | Value.num a, Value.num b => a == b
  | Value.num a, Value.float bInt bFrac =>
      numericMicros? (Value.num a) == numericMicros? (Value.float bInt bFrac)
  | Value.float aInt aFrac, Value.num b =>
      numericMicros? (Value.float aInt aFrac) == numericMicros? (Value.num b)
  | Value.float aInt aFrac, Value.float bInt bFrac =>
      numericMicros? (Value.float aInt aFrac) == numericMicros? (Value.float bInt bFrac)
  | Value.bool a, Value.bool b => a == b
  | Value.str a, Value.str b => a == b
  | Value.list a, Value.list b => a == b
  | Value.unit, Value.unit => true
  | _, _ => false

def evalBinOp (op : BinOp) (left right : Value) : Option Value :=
  match op, left, right with
  | BinOp.add, Value.num a, Value.num b => some (Value.num (a + b))
  | BinOp.add, _, _ => do
      let leftMicros <- numericMicros? left
      let rightMicros <- numericMicros? right
      some (numericResult (hasFloat left || hasFloat right) (leftMicros + rightMicros))
  | BinOp.sub, Value.num a, Value.num b => some (Value.num (a - b))
  | BinOp.sub, _, _ => do
      let leftMicros <- numericMicros? left
      let rightMicros <- numericMicros? right
      some (numericResult (hasFloat left || hasFloat right) (leftMicros - rightMicros))
  | BinOp.mul, Value.num a, Value.num b => some (Value.num (a * b))
  | BinOp.mul, _, _ => do
      let leftMicros <- numericMicros? left
      let rightMicros <- numericMicros? right
      some (numericResult (hasFloat left || hasFloat right) ((leftMicros * rightMicros) / microsPerUnit))
  | BinOp.div, Value.num _, Value.num 0 => none
  | BinOp.div, Value.num a, Value.num b => some (Value.num (a / b))
  | BinOp.div, _, _ => do
      let leftMicros <- numericMicros? left
      let rightMicros <- numericMicros? right
      if rightMicros == 0 then none else some (floatFromMicros ((leftMicros * microsPerUnit) / rightMicros))
  | BinOp.mod, Value.num _, Value.num 0 => none
  | BinOp.mod, Value.num a, Value.num b => some (Value.num (Int.emod a b))
  | BinOp.eq, _, _ => some (Value.bool (valuesEq left right))
  | BinOp.neq, _, _ => some (Value.bool (!valuesEq left right))
  | BinOp.lt, Value.num a, Value.num b => some (Value.bool (a < b))
  | BinOp.lt, _, _ => do
      let leftMicros <- numericMicros? left
      let rightMicros <- numericMicros? right
      some (Value.bool (leftMicros < rightMicros))
  | BinOp.gt, Value.num a, Value.num b => some (Value.bool (a > b))
  | BinOp.gt, _, _ => do
      let leftMicros <- numericMicros? left
      let rightMicros <- numericMicros? right
      some (Value.bool (leftMicros > rightMicros))
  | BinOp.le, Value.num a, Value.num b => some (Value.bool (a <= b))
  | BinOp.le, _, _ => do
      let leftMicros <- numericMicros? left
      let rightMicros <- numericMicros? right
      some (Value.bool (leftMicros <= rightMicros))
  | BinOp.ge, Value.num a, Value.num b => some (Value.bool (a >= b))
  | BinOp.ge, _, _ => do
      let leftMicros <- numericMicros? left
      let rightMicros <- numericMicros? right
      some (Value.bool (leftMicros >= rightMicros))
  | BinOp.and, _, _ => some (Value.bool (truthy left && truthy right))
  | BinOp.or, _, _ => some (Value.bool (truthy left || truthy right))
  | _, _, _ => none

def evalUnOp (op : UnOp) (value : Value) : Option Value :=
  match op, value with
  | UnOp.neg, Value.num n => some (Value.num (-n))
  | UnOp.neg, Value.float intPart fracMicros => do
      let micros <- numericMicros? (Value.float intPart fracMicros)
      some (floatFromMicros (-micros))
  | UnOp.not, _ => some (Value.bool (!truthy value))
  | _, _ => none

def evalIndex : Value -> Value -> Option Value
  | Value.list values, Value.num idx =>
      if idx < 0 then none else values[idx.toNat]?
  | Value.str value, Value.num idx =>
      if idx < 0 then none else do
        let char <- value.toList[idx.toNat]?
        some (Value.str (String.ofList [char]))
  | _, _ => none

def evalField : Value -> Ident -> Option Value
  | Value.list values, "length" => some (Value.num values.length)
  | Value.str value, "length" => some (Value.num value.length)
  | _, _ => none

def valueInList (needle : Value) : List Value -> Bool
  | [] => false
  | value :: rest => if value == needle then true else valueInList needle rest

def charPrefix : List Char -> List Char -> Bool
  | [], _ => true
  | _ :: _, [] => false
  | needle :: needleRest, value :: valueRest =>
      needle == value && charPrefix needleRest valueRest

def charListContains (needle : List Char) : List Char -> Bool
  | [] => needle == []
  | value :: rest =>
      charPrefix needle (value :: rest) || charListContains needle rest

def takeValues : Nat -> List Value -> List Value
  | 0, _ => []
  | _ + 1, [] => []
  | n + 1, value :: rest => value :: takeValues n rest

def dropValues : Nat -> List Value -> List Value
  | 0, values => values
  | _ + 1, [] => []
  | n + 1, _ :: rest => dropValues n rest

def takeChars : Nat -> List Char -> List Char
  | 0, _ => []
  | _ + 1, [] => []
  | n + 1, value :: rest => value :: takeChars n rest

def dropChars : Nat -> List Char -> List Char
  | 0, values => values
  | _ + 1, [] => []
  | n + 1, _ :: rest => dropChars n rest

def joinStringValues : List Value -> String -> Option String
  | [], _ => some ""
  | [Value.str value], _ => some value
  | Value.str value :: rest, sep => do
      let joinedRest <- joinStringValues rest sep
      some (value ++ sep ++ joinedRest)
  | _ :: _, _ => none

def evalMethod : Value -> Ident -> List Value -> Option Value
  | Value.list values, "len", [] => some (Value.num values.length)
  | Value.list values, "is_empty", [] => some (Value.bool values.isEmpty)
  | Value.list (value :: _), "first", [] => some value
  | Value.list (_ :: rest), "tail", [] => some (Value.list rest)
  | Value.list values, "last", [] =>
      match values.length with
      | 0 => none
      | n + 1 => values[n]?
  | Value.list values, "at", [Value.num idx] => evalIndex (Value.list values) (Value.num idx)
  | Value.list values, "take", [Value.num idx] =>
      if idx < 0 then none else some (Value.list (takeValues idx.toNat values))
  | Value.list values, "drop", [Value.num idx] =>
      if idx < 0 then none else some (Value.list (dropValues idx.toNat values))
  | Value.list values, "reverse", [] => some (Value.list values.reverse)
  | Value.list values, "append", [value] => some (Value.list (values ++ [value]))
  | Value.list values, "prepend", [value] => some (Value.list (value :: values))
  | Value.list values, "concat", [Value.list suffix] => some (Value.list (values ++ suffix))
  | Value.list values, "join", [Value.str sep] => do
      let joined <- joinStringValues values sep
      some (Value.str joined)
  | Value.list values, "contains", [needle] => some (Value.bool (valueInList needle values))
  | Value.str value, "len", [] => some (Value.num value.length)
  | Value.str value, "is_empty", [] => some (Value.bool value.isEmpty)
  | Value.str value, "first", [] => evalIndex (Value.str value) (Value.num 0)
  | Value.str value, "last", [] =>
      match value.toList.reverse with
      | [] => none
      | char :: _ => some (Value.str (String.ofList [char]))
  | Value.str value, "tail", [] =>
      match value.toList with
      | [] => none
      | _ :: rest => some (Value.str (String.ofList rest))
  | Value.str value, "take", [Value.num idx] =>
      if idx < 0 then none else some (Value.str (String.ofList (takeChars idx.toNat value.toList)))
  | Value.str value, "drop", [Value.num idx] =>
      if idx < 0 then none else some (Value.str (String.ofList (dropChars idx.toNat value.toList)))
  | Value.str value, "at", [Value.num idx] => evalIndex (Value.str value) (Value.num idx)
  | Value.str value, "contains", [Value.str needle] =>
      some (Value.bool (charListContains needle.toList value.toList))
  | Value.str value, "starts_with", [Value.str needle] =>
      some (Value.bool (charPrefix needle.toList value.toList))
  | Value.str value, "ends_with", [Value.str needle] =>
      some (Value.bool (charPrefix needle.toList.reverse value.toList.reverse))
  | Value.str value, "reverse", [] =>
      some (Value.str (String.ofList value.toList.reverse))
  | _, _, _ => none

mutual
  def evalExpr (env : Env) : Expr -> Option Value
    | Expr.num n => some (Value.num n)
    | Expr.float intPart fracMicros => some (Value.float intPart fracMicros)
    | Expr.bool b => some (Value.bool b)
    | Expr.str value => some (Value.str value)
    | Expr.unit => some Value.unit
    | Expr.list exprs => do
        let values <- evalExprs env exprs
        some (Value.list values)
    | Expr.var name => Env.lookup env name
    | Expr.unary op expr => do
        let value <- evalExpr env expr
        evalUnOp op value
    | Expr.binary left op right => do
        let leftValue <- evalExpr env left
        let rightValue <- evalExpr env right
        evalBinOp op leftValue rightValue
    | Expr.index target index => do
        let targetValue <- evalExpr env target
        let indexValue <- evalExpr env index
        evalIndex targetValue indexValue
    | Expr.field target field => do
        let targetValue <- evalExpr env target
        evalField targetValue field
    | Expr.method target method args => do
        let targetValue <- evalExpr env target
        let argValues <- evalArgs env args
        evalMethod targetValue method argValues
    | Expr.call _ _ => none

  def evalExprs (env : Env) : List Expr -> Option (List Value)
    | [] => some []
    | expr :: rest => do
        let value <- evalExpr env expr
        let values <- evalExprs env rest
        some (value :: values)

  def evalArg (env : Env) : Arg -> Option Value
    | Arg.positional expr => evalExpr env expr
    | Arg.named _ expr => evalExpr env expr

  def evalArgs (env : Env) : List Arg -> Option (List Value)
    | [] => some []
    | arg :: rest => do
        let value <- evalArg env arg
        let values <- evalArgs env rest
        some (value :: values)
end

def bindParams : List Ident -> List Value -> Env -> Option Env
  | [], [], env => some env
  | name :: names, value :: values, env => bindParams names values (Env.bind env name value)
  | _, _, _ => none

def identInList (name : Ident) : List Ident -> Bool
  | [] => false
  | first :: rest => first == name || identInList name rest

def argBindingsLookup (bindings : List (Ident × Value)) (name : Ident) : Option Value :=
  match bindings with
  | [] => none
  | (key, value) :: rest =>
      if key == name then some value else argBindingsLookup rest name

def bindArgName
    (params : List Ident)
    (name : Ident)
    (value : Value)
    (bindings : List (Ident × Value)) :
    Option (List (Ident × Value)) :=
  if identInList name params then
    match argBindingsLookup bindings name with
    | none => some ((name, value) :: bindings)
    | some _ => none
  else
    none

def firstUnboundParam (params : List Ident) (bindings : List (Ident × Value)) : Option Ident :=
  match params with
  | [] => none
  | name :: rest =>
      match argBindingsLookup bindings name with
      | none => some name
      | some _ => firstUnboundParam rest bindings

def bindArgValues
    (params : List Ident)
    (args : List Arg)
    (values : List Value)
    (bindings : List (Ident × Value)) :
    Option (List (Ident × Value)) :=
  match args, values with
  | [], [] => some bindings
  | Arg.positional _ :: restArgs, value :: restValues => do
      let name <- firstUnboundParam params bindings
      let updated <- bindArgName params name value bindings
      bindArgValues params restArgs restValues updated
  | Arg.named name _ :: restArgs, value :: restValues => do
      let updated <- bindArgName params name value bindings
      bindArgValues params restArgs restValues updated
  | _, _ => none

def bindBoundParams
    (params : List Ident)
    (bindings : List (Ident × Value))
    (env : Env) :
    Option Env :=
  match params with
  | [] => some env
  | name :: rest => do
      let value <- argBindingsLookup bindings name
      bindBoundParams rest bindings (Env.bind env name value)

def argsAllPositional : List Arg -> Bool
  | [] => true
  | Arg.positional _ :: rest => argsAllPositional rest
  | Arg.named _ _ :: _ => false

def bindCallArgs (params : List Ident) (args : List Arg) (values : List Value) (env : Env) :
    Option Env :=
  if argsAllPositional args then
    bindParams params values env
  else do
    let bindings <- bindArgValues params args values []
    bindBoundParams params bindings env

mutual
  partial def evalExprsWithFns (fuel : Nat) (env : Env) (fns : FnEnv) : List Expr -> Option (List Value)
    | [] => some []
    | expr :: rest => do
        let value <- evalExprWithFns fuel env fns expr
        let values <- evalExprsWithFns fuel env fns rest
        some (value :: values)

  partial def evalArgWithFns (fuel : Nat) (env : Env) (fns : FnEnv) : Arg -> Option Value
    | Arg.positional expr => evalExprWithFns fuel env fns expr
    | Arg.named _ expr => evalExprWithFns fuel env fns expr

  partial def evalArgsWithFns (fuel : Nat) (env : Env) (fns : FnEnv) : List Arg -> Option (List Value)
    | [] => some []
    | arg :: rest => do
        let value <- evalArgWithFns fuel env fns arg
        let values <- evalArgsWithFns fuel env fns rest
        some (value :: values)

  partial def evalExprWithFns (fuel : Nat) (env : Env) (fns : FnEnv) : Expr -> Option Value :=
    match fuel with
    | 0 => fun _ => none
    | fuel' + 1 =>
        fun
          | Expr.num n => some (Value.num n)
          | Expr.float intPart fracMicros => some (Value.float intPart fracMicros)
          | Expr.bool b => some (Value.bool b)
          | Expr.str value => some (Value.str value)
          | Expr.unit => some Value.unit
          | Expr.list exprs => do
              let values <- evalExprsWithFns fuel' env fns exprs
              some (Value.list values)
          | Expr.var name => Env.lookup env name
          | Expr.unary op expr => do
              let value <- evalExprWithFns fuel' env fns expr
              evalUnOp op value
          | Expr.binary left op right => do
              let leftValue <- evalExprWithFns fuel' env fns left
              let rightValue <- evalExprWithFns fuel' env fns right
              evalBinOp op leftValue rightValue
          | Expr.index target index => do
              let targetValue <- evalExprWithFns fuel' env fns target
              let indexValue <- evalExprWithFns fuel' env fns index
              evalIndex targetValue indexValue
          | Expr.field target field => do
              let targetValue <- evalExprWithFns fuel' env fns target
              evalField targetValue field
          | Expr.method target method args => do
              let targetValue <- evalExprWithFns fuel' env fns target
              let argValues <- evalArgsWithFns fuel' env fns args
              evalMethod targetValue method argValues
          | Expr.call name args => do
              let fn <- FnEnv.lookup fns name
              let values <- evalArgsWithFns fuel' env fns args
              let frame <- bindCallArgs fn.params args values env
              let (_, _, flow) <- execBlockWithFns fuel' frame fns fn.body
              match flow with
              | Flow.value value => some value
              | Flow.return value => some value
              | Flow.break => none
              | Flow.continue => none

  partial def execStmtWithFns (fuel : Nat) (env : Env) (fns : FnEnv) : Stmt -> Option (Env × FnEnv × Flow) :=
    match fuel with
    | 0 => fun _ => none
    | fuel' + 1 =>
        fun
          | Stmt.letDecl name expr => do
              let value <- evalExprWithFns fuel' env fns expr
              some (Env.bind env name value, fns, Flow.value value)
          | Stmt.letDeclTyped name _ expr => do
              let value <- evalExprWithFns fuel' env fns expr
              some (Env.bind env name value, fns, Flow.value value)
          | Stmt.assign name expr => do
              let value <- evalExprWithFns fuel' env fns expr
              let updated <- Env.assign env name value
              some (updated, fns, Flow.value value)
          | Stmt.ifThenElse condition thenBranch elseBranch => do
              let value <- evalExprWithFns fuel' env fns condition
              if truthy value then
                execBlockWithFns fuel' env fns thenBranch
              else
                match elseBranch with
                | some branch => execBlockWithFns fuel' env fns branch
                | none => some (env, fns, Flow.value Value.unit)
          | Stmt.while condition body => do
              let value <- evalExprWithFns fuel' env fns condition
              if truthy value then
                let (env1, fns1, flow) <- execBlockWithFns fuel' env fns body
                match flow with
                | Flow.value _ => execStmtWithFns fuel' env1 fns1 (Stmt.while condition body)
                | Flow.return value => some (env1, fns1, Flow.return value)
                | Flow.break => some (env1, fns1, Flow.value Value.unit)
                | Flow.continue => execStmtWithFns fuel' env1 fns1 (Stmt.while condition body)
              else
                some (env, fns, Flow.value Value.unit)
          | Stmt.forRange iterator start stop body => do
              if start < stop then
                let loopEnv := Env.bind env iterator (Value.num start)
                let (env1, fns1, flow) <- execBlockWithFns fuel' loopEnv fns body
                match flow with
                | Flow.value _ =>
                    execStmtWithFns fuel' env1 fns1 (Stmt.forRange iterator (start + 1) stop body)
                | Flow.return value => some (env1, fns1, Flow.return value)
                | Flow.break => some (env1, fns1, Flow.value Value.unit)
                | Flow.continue =>
                    execStmtWithFns fuel' env1 fns1 (Stmt.forRange iterator (start + 1) stop body)
              else
                let finalEnv := Env.bind env iterator (Value.num start)
                some (finalEnv, fns, Flow.value Value.unit)
          | Stmt.seal condition body => do
              match condition with
              | some conditionExpr => do
                  let value <- evalExprWithFns fuel' env fns conditionExpr
                  if truthy value then
                    some (env, fns, Flow.value Value.unit)
                  else
                    let (env1, fns1, flow) <- execBlockWithFns fuel' env fns body
                    match flow with
                    | Flow.value _ => execStmtWithFns fuel' env1 fns1 (Stmt.seal condition body)
                    | Flow.return value => some (env1, fns1, Flow.return value)
                    | Flow.break => some (env1, fns1, Flow.value Value.unit)
                    | Flow.continue => execStmtWithFns fuel' env1 fns1 (Stmt.seal condition body)
              | none => do
                  let (env1, fns1, flow) <- execBlockWithFns fuel' env fns body
                  match flow with
                  | Flow.value _ => execStmtWithFns fuel' env1 fns1 (Stmt.seal none body)
                  | Flow.return value => some (env1, fns1, Flow.return value)
                  | Flow.break => some (env1, fns1, Flow.value Value.unit)
                  | Flow.continue => execStmtWithFns fuel' env1 fns1 (Stmt.seal none body)
          | Stmt.fnDecl name params body =>
              some (env, FnEnv.bind fns name { params := params, body := body }, Flow.value Value.unit)
          | Stmt.fnDeclReturn name params _ body =>
              some (env, FnEnv.bind fns name { params := params, body := body }, Flow.value Value.unit)
          | Stmt.fnDeclTyped name params body =>
              some (env, FnEnv.bind fns name { params := params.map Prod.fst, body := body }, Flow.value Value.unit)
          | Stmt.fnDeclTypedReturn name params _ body =>
              some (env, FnEnv.bind fns name { params := params.map Prod.fst, body := body }, Flow.value Value.unit)
          | Stmt.ret (some expr) => do
              let value <- evalExprWithFns fuel' env fns expr
              some (env, fns, Flow.return value)
          | Stmt.ret none =>
              some (env, fns, Flow.return Value.unit)
          | Stmt.expr expr => do
              let value <- evalExprWithFns fuel' env fns expr
              some (env, fns, Flow.value value)
          | Stmt.break => some (env, fns, Flow.break)
          | Stmt.continue => some (env, fns, Flow.continue)

  partial def execBlockWithFns (fuel : Nat) (env : Env) (fns : FnEnv) : List Stmt -> Option (Env × FnEnv × Flow)
    | [] => some (env, fns, Flow.value Value.unit)
    | [stmt] => execStmtWithFns fuel env fns stmt
    | stmt :: next :: rest => do
        let (env1, fns1, flow) <- execStmtWithFns fuel env fns stmt
        match flow with
        | Flow.value _ => execBlockWithFns fuel env1 fns1 (next :: rest)
        | Flow.return value => some (env1, fns1, Flow.return value)
        | Flow.break => some (env1, fns1, Flow.break)
        | Flow.continue => some (env1, fns1, Flow.continue)
end

mutual
  inductive StepStmt : Env -> Stmt -> Env -> Flow -> Prop where
    | letDecl {env name expr value} :
        evalExpr env expr = some value ->
        StepStmt env (Stmt.letDecl name expr) (Env.bind env name value) (Flow.value value)
    | letDeclTyped {env name ty expr value} :
        evalExpr env expr = some value ->
        StepStmt env (Stmt.letDeclTyped name ty expr) (Env.bind env name value) (Flow.value value)
    | assign {env name expr value updated} :
        evalExpr env expr = some value ->
        Env.assign env name value = some updated ->
        StepStmt env (Stmt.assign name expr) updated (Flow.value value)
    | ifTrue {env env' condition thenBranch elseBranch value flow} :
        evalExpr env condition = some value ->
        truthy value = true ->
        StepBlock env thenBranch env' flow ->
        StepStmt env (Stmt.ifThenElse condition thenBranch elseBranch) env' flow
    | ifFalseSome {env env' condition thenBranch elseBranch value flow} :
        evalExpr env condition = some value ->
        truthy value = false ->
        StepBlock env elseBranch env' flow ->
        StepStmt env (Stmt.ifThenElse condition thenBranch (some elseBranch)) env' flow
    | ifFalseNone {env condition thenBranch value} :
        evalExpr env condition = some value ->
        truthy value = false ->
        StepStmt env (Stmt.ifThenElse condition thenBranch none) env (Flow.value Value.unit)
    | whileFalse {env condition body value} :
        evalExpr env condition = some value ->
        truthy value = false ->
        StepStmt env (Stmt.while condition body) env (Flow.value Value.unit)
    | whileValue {env env1 env2 condition body value bodyValue flow} :
        evalExpr env condition = some value ->
        truthy value = true ->
        StepBlock env body env1 (Flow.value bodyValue) ->
        StepStmt env1 (Stmt.while condition body) env2 flow ->
        StepStmt env (Stmt.while condition body) env2 flow
    | whileReturn {env env1 condition body conditionValue returnValue} :
        evalExpr env condition = some conditionValue ->
        truthy conditionValue = true ->
        StepBlock env body env1 (Flow.return returnValue) ->
        StepStmt env (Stmt.while condition body) env1 (Flow.return returnValue)
    | whileBreak {env env1 condition body value} :
        evalExpr env condition = some value ->
        truthy value = true ->
        StepBlock env body env1 Flow.break ->
        StepStmt env (Stmt.while condition body) env1 (Flow.value Value.unit)
    | whileContinue {env env1 env2 condition body value flow} :
        evalExpr env condition = some value ->
        truthy value = true ->
        StepBlock env body env1 Flow.continue ->
        StepStmt env1 (Stmt.while condition body) env2 flow ->
        StepStmt env (Stmt.while condition body) env2 flow
    | forDone {env iterator start stop body} :
        ¬ start < stop ->
        StepStmt env (Stmt.forRange iterator start stop body)
          (Env.bind env iterator (Value.num start))
          (Flow.value Value.unit)
    | forValue {env env1 env2 iterator start stop body bodyValue flow} :
        start < stop ->
        StepBlock (Env.bind env iterator (Value.num start)) body env1 (Flow.value bodyValue) ->
        StepStmt env1 (Stmt.forRange iterator (start + 1) stop body) env2 flow ->
        StepStmt env (Stmt.forRange iterator start stop body) env2 flow
    | forReturn {env env1 iterator start stop body returnValue} :
        start < stop ->
        StepBlock (Env.bind env iterator (Value.num start)) body env1 (Flow.return returnValue) ->
        StepStmt env (Stmt.forRange iterator start stop body) env1 (Flow.return returnValue)
    | forBreak {env env1 iterator start stop body} :
        start < stop ->
        StepBlock (Env.bind env iterator (Value.num start)) body env1 Flow.break ->
        StepStmt env (Stmt.forRange iterator start stop body) env1 (Flow.value Value.unit)
    | forContinue {env env1 env2 iterator start stop body flow} :
        start < stop ->
        StepBlock (Env.bind env iterator (Value.num start)) body env1 Flow.continue ->
        StepStmt env1 (Stmt.forRange iterator (start + 1) stop body) env2 flow ->
        StepStmt env (Stmt.forRange iterator start stop body) env2 flow
    | sealUntilDone {env condition body value} :
        evalExpr env condition = some value ->
        truthy value = true ->
        StepStmt env (Stmt.seal (some condition) body) env (Flow.value Value.unit)
    | sealUntilValue {env env1 env2 condition body value bodyValue flow} :
        evalExpr env condition = some value ->
        truthy value = false ->
        StepBlock env body env1 (Flow.value bodyValue) ->
        StepStmt env1 (Stmt.seal (some condition) body) env2 flow ->
        StepStmt env (Stmt.seal (some condition) body) env2 flow
    | sealUntilReturn {env env1 condition body conditionValue returnValue} :
        evalExpr env condition = some conditionValue ->
        truthy conditionValue = false ->
        StepBlock env body env1 (Flow.return returnValue) ->
        StepStmt env (Stmt.seal (some condition) body) env1 (Flow.return returnValue)
    | sealUntilBreak {env env1 condition body value} :
        evalExpr env condition = some value ->
        truthy value = false ->
        StepBlock env body env1 Flow.break ->
        StepStmt env (Stmt.seal (some condition) body) env1 (Flow.value Value.unit)
    | sealUntilContinue {env env1 env2 condition body value flow} :
        evalExpr env condition = some value ->
        truthy value = false ->
        StepBlock env body env1 Flow.continue ->
        StepStmt env1 (Stmt.seal (some condition) body) env2 flow ->
        StepStmt env (Stmt.seal (some condition) body) env2 flow
    | sealValue {env env1 env2 body bodyValue flow} :
        StepBlock env body env1 (Flow.value bodyValue) ->
        StepStmt env1 (Stmt.seal none body) env2 flow ->
        StepStmt env (Stmt.seal none body) env2 flow
    | sealReturn {env env1 body returnValue} :
        StepBlock env body env1 (Flow.return returnValue) ->
        StepStmt env (Stmt.seal none body) env1 (Flow.return returnValue)
    | sealBreak {env env1 body} :
        StepBlock env body env1 Flow.break ->
        StepStmt env (Stmt.seal none body) env1 (Flow.value Value.unit)
    | sealContinue {env env1 env2 body flow} :
        StepBlock env body env1 Flow.continue ->
        StepStmt env1 (Stmt.seal none body) env2 flow ->
        StepStmt env (Stmt.seal none body) env2 flow
    | fnDecl {env name params body} :
        StepStmt env (Stmt.fnDecl name params body) env (Flow.value Value.unit)
    | fnDeclReturn {env name params returnTy body} :
        StepStmt env (Stmt.fnDeclReturn name params returnTy body) env (Flow.value Value.unit)
    | fnDeclTyped {env name params body} :
        StepStmt env (Stmt.fnDeclTyped name params body) env (Flow.value Value.unit)
    | fnDeclTypedReturn {env name params returnTy body} :
        StepStmt env (Stmt.fnDeclTypedReturn name params returnTy body) env (Flow.value Value.unit)
    | expr {env expr value} :
        evalExpr env expr = some value ->
        StepStmt env (Stmt.expr expr) env (Flow.value value)
    | retSome {env expr value} :
        evalExpr env expr = some value ->
        StepStmt env (Stmt.ret (some expr)) env (Flow.return value)
    | retNone {env} :
        StepStmt env (Stmt.ret none) env (Flow.return Value.unit)
    | break {env} :
        StepStmt env Stmt.break env Flow.break
    | continue {env} :
        StepStmt env Stmt.continue env Flow.continue

  inductive StepBlock : Env -> List Stmt -> Env -> Flow -> Prop where
    | nil {env} :
        StepBlock env [] env (Flow.value Value.unit)
    | single {env env' stmt flow} :
        StepStmt env stmt env' flow ->
        StepBlock env [stmt] env' flow
    | consValue {env env1 env2 stmt next rest value flow} :
        StepStmt env stmt env1 (Flow.value value) ->
        StepBlock env1 (next :: rest) env2 flow ->
        StepBlock env (stmt :: next :: rest) env2 flow
    | consReturn {env env1 stmt next rest value} :
        StepStmt env stmt env1 (Flow.return value) ->
        StepBlock env (stmt :: next :: rest) env1 (Flow.return value)
    | consBreak {env env1 stmt next rest} :
        StepStmt env stmt env1 Flow.break ->
        StepBlock env (stmt :: next :: rest) env1 Flow.break
    | consContinue {env env1 stmt next rest} :
        StepStmt env stmt env1 Flow.continue ->
        StepBlock env (stmt :: next :: rest) env1 Flow.continue
end

mutual
  inductive EvalExprWithFnsRel : Env -> FnEnv -> Expr -> Value -> Prop where
    | num {env fns n} :
        EvalExprWithFnsRel env fns (Expr.num n) (Value.num n)
    | float {env fns intPart fracMicros} :
        EvalExprWithFnsRel env fns (Expr.float intPart fracMicros) (Value.float intPart fracMicros)
    | bool {env fns b} :
        EvalExprWithFnsRel env fns (Expr.bool b) (Value.bool b)
    | str {env fns value} :
        EvalExprWithFnsRel env fns (Expr.str value) (Value.str value)
    | unit {env fns} :
        EvalExprWithFnsRel env fns Expr.unit Value.unit
    | list {env fns exprs values} :
        EvalExprsWithFnsRel env fns exprs values ->
        EvalExprWithFnsRel env fns (Expr.list exprs) (Value.list values)
    | var {env fns name value} :
        Env.lookup env name = some value ->
        EvalExprWithFnsRel env fns (Expr.var name) value
    | unary {env fns op expr value result} :
        EvalExprWithFnsRel env fns expr value ->
        evalUnOp op value = some result ->
        EvalExprWithFnsRel env fns (Expr.unary op expr) result
    | binary {env fns left op right leftValue rightValue result} :
        EvalExprWithFnsRel env fns left leftValue ->
        EvalExprWithFnsRel env fns right rightValue ->
        evalBinOp op leftValue rightValue = some result ->
        EvalExprWithFnsRel env fns (Expr.binary left op right) result
    | index {env fns target index targetValue indexValue result} :
        EvalExprWithFnsRel env fns target targetValue ->
        EvalExprWithFnsRel env fns index indexValue ->
        evalIndex targetValue indexValue = some result ->
        EvalExprWithFnsRel env fns (Expr.index target index) result
    | field {env fns target field targetValue result} :
        EvalExprWithFnsRel env fns target targetValue ->
        evalField targetValue field = some result ->
        EvalExprWithFnsRel env fns (Expr.field target field) result
    | method {env fns target method args targetValue argValues result} :
        EvalExprWithFnsRel env fns target targetValue ->
        EvalArgsWithFnsRel env fns args argValues ->
        evalMethod targetValue method argValues = some result ->
        EvalExprWithFnsRel env fns (Expr.method target method args) result
    | callValue {env fns name args fn values frame env' fns' value} :
        FnEnv.lookup fns name = some fn ->
        EvalArgsWithFnsRel env fns args values ->
        bindCallArgs fn.params args values env = some frame ->
        StepBlockWithFns frame fns fn.body env' fns' (Flow.value value) ->
        EvalExprWithFnsRel env fns (Expr.call name args) value
    | callReturn {env fns name args fn values frame env' fns' value} :
        FnEnv.lookup fns name = some fn ->
        EvalArgsWithFnsRel env fns args values ->
        bindCallArgs fn.params args values env = some frame ->
        StepBlockWithFns frame fns fn.body env' fns' (Flow.return value) ->
        EvalExprWithFnsRel env fns (Expr.call name args) value

  inductive EvalExprsWithFnsRel : Env -> FnEnv -> List Expr -> List Value -> Prop where
    | nil {env fns} :
        EvalExprsWithFnsRel env fns [] []
    | cons {env fns expr exprs value values} :
        EvalExprWithFnsRel env fns expr value ->
        EvalExprsWithFnsRel env fns exprs values ->
        EvalExprsWithFnsRel env fns (expr :: exprs) (value :: values)

  inductive EvalArgWithFnsRel : Env -> FnEnv -> Arg -> Value -> Prop where
    | positional {env fns expr value} :
        EvalExprWithFnsRel env fns expr value ->
        EvalArgWithFnsRel env fns (Arg.positional expr) value
    | named {env fns name expr value} :
        EvalExprWithFnsRel env fns expr value ->
        EvalArgWithFnsRel env fns (Arg.named name expr) value

  inductive EvalArgsWithFnsRel : Env -> FnEnv -> List Arg -> List Value -> Prop where
    | nil {env fns} :
        EvalArgsWithFnsRel env fns [] []
    | cons {env fns arg args value values} :
        EvalArgWithFnsRel env fns arg value ->
        EvalArgsWithFnsRel env fns args values ->
        EvalArgsWithFnsRel env fns (arg :: args) (value :: values)

  inductive StepStmtWithFns : Env -> FnEnv -> Stmt -> Env -> FnEnv -> Flow -> Prop where
    | letDecl {env fns name expr value} :
        EvalExprWithFnsRel env fns expr value ->
        StepStmtWithFns env fns (Stmt.letDecl name expr)
          (Env.bind env name value) fns (Flow.value value)
    | letDeclTyped {env fns name ty expr value} :
        EvalExprWithFnsRel env fns expr value ->
        StepStmtWithFns env fns (Stmt.letDeclTyped name ty expr)
          (Env.bind env name value) fns (Flow.value value)
    | assign {env fns name expr value updated} :
        EvalExprWithFnsRel env fns expr value ->
        Env.assign env name value = some updated ->
        StepStmtWithFns env fns (Stmt.assign name expr) updated fns (Flow.value value)
    | ifTrue {env fns env' fns' condition thenBranch elseBranch value flow} :
        EvalExprWithFnsRel env fns condition value ->
        truthy value = true ->
        StepBlockWithFns env fns thenBranch env' fns' flow ->
        StepStmtWithFns env fns (Stmt.ifThenElse condition thenBranch elseBranch) env' fns' flow
    | ifFalseSome {env fns env' fns' condition thenBranch elseBranch value flow} :
        EvalExprWithFnsRel env fns condition value ->
        truthy value = false ->
        StepBlockWithFns env fns elseBranch env' fns' flow ->
        StepStmtWithFns env fns (Stmt.ifThenElse condition thenBranch (some elseBranch)) env' fns' flow
    | ifFalseNone {env fns condition thenBranch value} :
        EvalExprWithFnsRel env fns condition value ->
        truthy value = false ->
        StepStmtWithFns env fns (Stmt.ifThenElse condition thenBranch none) env fns (Flow.value Value.unit)
    | whileFalse {env fns condition body value} :
        EvalExprWithFnsRel env fns condition value ->
        truthy value = false ->
        StepStmtWithFns env fns (Stmt.while condition body) env fns (Flow.value Value.unit)
    | whileValue {env fns env1 fns1 env2 fns2 condition body value bodyValue flow} :
        EvalExprWithFnsRel env fns condition value ->
        truthy value = true ->
        StepBlockWithFns env fns body env1 fns1 (Flow.value bodyValue) ->
        StepStmtWithFns env1 fns1 (Stmt.while condition body) env2 fns2 flow ->
        StepStmtWithFns env fns (Stmt.while condition body) env2 fns2 flow
    | whileReturn {env fns env1 fns1 condition body conditionValue returnValue} :
        EvalExprWithFnsRel env fns condition conditionValue ->
        truthy conditionValue = true ->
        StepBlockWithFns env fns body env1 fns1 (Flow.return returnValue) ->
        StepStmtWithFns env fns (Stmt.while condition body) env1 fns1 (Flow.return returnValue)
    | whileBreak {env fns env1 fns1 condition body value} :
        EvalExprWithFnsRel env fns condition value ->
        truthy value = true ->
        StepBlockWithFns env fns body env1 fns1 Flow.break ->
        StepStmtWithFns env fns (Stmt.while condition body) env1 fns1 (Flow.value Value.unit)
    | whileContinue {env fns env1 fns1 env2 fns2 condition body value flow} :
        EvalExprWithFnsRel env fns condition value ->
        truthy value = true ->
        StepBlockWithFns env fns body env1 fns1 Flow.continue ->
        StepStmtWithFns env1 fns1 (Stmt.while condition body) env2 fns2 flow ->
        StepStmtWithFns env fns (Stmt.while condition body) env2 fns2 flow
    | forDone {env fns iterator start stop body} :
        ¬ start < stop ->
        StepStmtWithFns env fns (Stmt.forRange iterator start stop body)
          (Env.bind env iterator (Value.num start)) fns
          (Flow.value Value.unit)
    | forValue {env fns env1 fns1 env2 fns2 iterator start stop body bodyValue flow} :
        start < stop ->
        StepBlockWithFns (Env.bind env iterator (Value.num start)) fns body env1 fns1 (Flow.value bodyValue) ->
        StepStmtWithFns env1 fns1 (Stmt.forRange iterator (start + 1) stop body) env2 fns2 flow ->
        StepStmtWithFns env fns (Stmt.forRange iterator start stop body) env2 fns2 flow
    | forReturn {env fns env1 fns1 iterator start stop body returnValue} :
        start < stop ->
        StepBlockWithFns (Env.bind env iterator (Value.num start)) fns body env1 fns1 (Flow.return returnValue) ->
        StepStmtWithFns env fns (Stmt.forRange iterator start stop body) env1 fns1 (Flow.return returnValue)
    | forBreak {env fns env1 fns1 iterator start stop body} :
        start < stop ->
        StepBlockWithFns (Env.bind env iterator (Value.num start)) fns body env1 fns1 Flow.break ->
        StepStmtWithFns env fns (Stmt.forRange iterator start stop body) env1 fns1 (Flow.value Value.unit)
    | forContinue {env fns env1 fns1 env2 fns2 iterator start stop body flow} :
        start < stop ->
        StepBlockWithFns (Env.bind env iterator (Value.num start)) fns body env1 fns1 Flow.continue ->
        StepStmtWithFns env1 fns1 (Stmt.forRange iterator (start + 1) stop body) env2 fns2 flow ->
        StepStmtWithFns env fns (Stmt.forRange iterator start stop body) env2 fns2 flow
    | sealUntilDone {env fns condition body value} :
        EvalExprWithFnsRel env fns condition value ->
        truthy value = true ->
        StepStmtWithFns env fns (Stmt.seal (some condition) body) env fns (Flow.value Value.unit)
    | sealUntilValue {env fns env1 fns1 env2 fns2 condition body value bodyValue flow} :
        EvalExprWithFnsRel env fns condition value ->
        truthy value = false ->
        StepBlockWithFns env fns body env1 fns1 (Flow.value bodyValue) ->
        StepStmtWithFns env1 fns1 (Stmt.seal (some condition) body) env2 fns2 flow ->
        StepStmtWithFns env fns (Stmt.seal (some condition) body) env2 fns2 flow
    | sealUntilReturn {env fns env1 fns1 condition body conditionValue returnValue} :
        EvalExprWithFnsRel env fns condition conditionValue ->
        truthy conditionValue = false ->
        StepBlockWithFns env fns body env1 fns1 (Flow.return returnValue) ->
        StepStmtWithFns env fns (Stmt.seal (some condition) body) env1 fns1 (Flow.return returnValue)
    | sealUntilBreak {env fns env1 fns1 condition body value} :
        EvalExprWithFnsRel env fns condition value ->
        truthy value = false ->
        StepBlockWithFns env fns body env1 fns1 Flow.break ->
        StepStmtWithFns env fns (Stmt.seal (some condition) body) env1 fns1 (Flow.value Value.unit)
    | sealUntilContinue {env fns env1 fns1 env2 fns2 condition body value flow} :
        EvalExprWithFnsRel env fns condition value ->
        truthy value = false ->
        StepBlockWithFns env fns body env1 fns1 Flow.continue ->
        StepStmtWithFns env1 fns1 (Stmt.seal (some condition) body) env2 fns2 flow ->
        StepStmtWithFns env fns (Stmt.seal (some condition) body) env2 fns2 flow
    | sealValue {env fns env1 fns1 env2 fns2 body bodyValue flow} :
        StepBlockWithFns env fns body env1 fns1 (Flow.value bodyValue) ->
        StepStmtWithFns env1 fns1 (Stmt.seal none body) env2 fns2 flow ->
        StepStmtWithFns env fns (Stmt.seal none body) env2 fns2 flow
    | sealReturn {env fns env1 fns1 body returnValue} :
        StepBlockWithFns env fns body env1 fns1 (Flow.return returnValue) ->
        StepStmtWithFns env fns (Stmt.seal none body) env1 fns1 (Flow.return returnValue)
    | sealBreak {env fns env1 fns1 body} :
        StepBlockWithFns env fns body env1 fns1 Flow.break ->
        StepStmtWithFns env fns (Stmt.seal none body) env1 fns1 (Flow.value Value.unit)
    | sealContinue {env fns env1 fns1 env2 fns2 body flow} :
        StepBlockWithFns env fns body env1 fns1 Flow.continue ->
        StepStmtWithFns env1 fns1 (Stmt.seal none body) env2 fns2 flow ->
        StepStmtWithFns env fns (Stmt.seal none body) env2 fns2 flow
    | fnDecl {env fns name params body} :
        StepStmtWithFns env fns (Stmt.fnDecl name params body)
          env (FnEnv.bind fns name { params := params, body := body })
          (Flow.value Value.unit)
    | fnDeclReturn {env fns name params returnTy body} :
        StepStmtWithFns env fns (Stmt.fnDeclReturn name params returnTy body)
          env (FnEnv.bind fns name { params := params, body := body })
          (Flow.value Value.unit)
    | fnDeclTyped {env fns name params body} :
        StepStmtWithFns env fns (Stmt.fnDeclTyped name params body)
          env (FnEnv.bind fns name { params := params.map Prod.fst, body := body })
          (Flow.value Value.unit)
    | fnDeclTypedReturn {env fns name params returnTy body} :
        StepStmtWithFns env fns (Stmt.fnDeclTypedReturn name params returnTy body)
          env (FnEnv.bind fns name { params := params.map Prod.fst, body := body })
          (Flow.value Value.unit)
    | expr {env fns expr value} :
        EvalExprWithFnsRel env fns expr value ->
        StepStmtWithFns env fns (Stmt.expr expr) env fns (Flow.value value)
    | retSome {env fns expr value} :
        EvalExprWithFnsRel env fns expr value ->
        StepStmtWithFns env fns (Stmt.ret (some expr)) env fns (Flow.return value)
    | retNone {env fns} :
        StepStmtWithFns env fns (Stmt.ret none) env fns (Flow.return Value.unit)
    | break {env fns} :
        StepStmtWithFns env fns Stmt.break env fns Flow.break
    | continue {env fns} :
        StepStmtWithFns env fns Stmt.continue env fns Flow.continue

  inductive StepBlockWithFns : Env -> FnEnv -> List Stmt -> Env -> FnEnv -> Flow -> Prop where
    | nil {env fns} :
        StepBlockWithFns env fns [] env fns (Flow.value Value.unit)
    | single {env fns env' fns' stmt flow} :
        StepStmtWithFns env fns stmt env' fns' flow ->
        StepBlockWithFns env fns [stmt] env' fns' flow
    | consValue {env fns env1 fns1 env2 fns2 stmt next rest value flow} :
        StepStmtWithFns env fns stmt env1 fns1 (Flow.value value) ->
        StepBlockWithFns env1 fns1 (next :: rest) env2 fns2 flow ->
        StepBlockWithFns env fns (stmt :: next :: rest) env2 fns2 flow
    | consReturn {env fns env1 fns1 stmt next rest value} :
        StepStmtWithFns env fns stmt env1 fns1 (Flow.return value) ->
        StepBlockWithFns env fns (stmt :: next :: rest) env1 fns1 (Flow.return value)
    | consBreak {env fns env1 fns1 stmt next rest} :
        StepStmtWithFns env fns stmt env1 fns1 Flow.break ->
        StepBlockWithFns env fns (stmt :: next :: rest) env1 fns1 Flow.break
    | consContinue {env fns env1 fns1 stmt next rest} :
        StepStmtWithFns env fns stmt env1 fns1 Flow.continue ->
        StepBlockWithFns env fns (stmt :: next :: rest) env1 fns1 Flow.continue
end

theorem lookup_bind_same (env : Env) (name : Ident) (value : Value) :
    Env.lookup (Env.bind env name value) name = some value := by
  unfold Env.bind Env.lookup
  simp

theorem eval_bound_var (env : Env) (name : Ident) (value : Value) :
    evalExpr (Env.bind env name value) (Expr.var name) = some value := by
  unfold evalExpr
  exact lookup_bind_same env name value

theorem evalExprWithFnsRel_num_sound :
    EvalExprWithFnsRel [] [] (Expr.num 7) (Value.num 7) ->
    evalExprWithFns 1 [] [] (Expr.num 7) = some (Value.num 7) := by
  intro _
  native_decide

theorem evalExprWithFnsRel_bool_sound :
    EvalExprWithFnsRel [] [] (Expr.bool true) (Value.bool true) ->
    evalExprWithFns 1 [] [] (Expr.bool true) = some (Value.bool true) := by
  intro _
  native_decide

theorem evalExprWithFnsRel_var_sound :
    EvalExprWithFnsRel [("x", Value.str "open")] [] (Expr.var "x") (Value.str "open") ->
    evalExprWithFns 1 [("x", Value.str "open")] [] (Expr.var "x") = some (Value.str "open") := by
  intro _
  native_decide

theorem evalExprWithFnsRel_binary_add_sound :
    EvalExprWithFnsRel [] [] (Expr.binary (Expr.num 2) BinOp.add (Expr.num 5)) (Value.num 7) ->
    evalExprWithFns 2 [] [] (Expr.binary (Expr.num 2) BinOp.add (Expr.num 5)) =
      some (Value.num 7) := by
  intro _
  native_decide

theorem evalExprWithFnsRel_unary_not_sound :
    EvalExprWithFnsRel [] [] (Expr.unary UnOp.not (Expr.bool false)) (Value.bool true) ->
    evalExprWithFns 2 [] [] (Expr.unary UnOp.not (Expr.bool false)) =
      some (Value.bool true) := by
  intro _
  native_decide

theorem evalExprWithFnsRel_list_sound :
    EvalExprWithFnsRel [] [] (Expr.list [Expr.num 1, Expr.bool true])
      (Value.list [Value.num 1, Value.bool true]) ->
    evalExprWithFns 2 [] [] (Expr.list [Expr.num 1, Expr.bool true]) =
      some (Value.list [Value.num 1, Value.bool true]) := by
  intro _
  native_decide

theorem evalExprWithFnsRel_index_sound :
    EvalExprWithFnsRel [] [] (Expr.index (Expr.list [Expr.str "a", Expr.str "b"]) (Expr.num 1))
      (Value.str "b") ->
    evalExprWithFns 3 [] [] (Expr.index (Expr.list [Expr.str "a", Expr.str "b"]) (Expr.num 1)) =
      some (Value.str "b") := by
  intro _
  native_decide

theorem evalExprWithFnsRel_field_length_sound :
    EvalExprWithFnsRel [] [] (Expr.field (Expr.list [Expr.num 1, Expr.num 2]) "length")
      (Value.num 2) ->
    evalExprWithFns 3 [] [] (Expr.field (Expr.list [Expr.num 1, Expr.num 2]) "length") =
      some (Value.num 2) := by
  intro _
  native_decide

theorem evalExprWithFnsRel_method_len_sound :
    EvalExprWithFnsRel [] [] (Expr.method (Expr.str "aether") "len" []) (Value.num 6) ->
    evalExprWithFns 2 [] [] (Expr.method (Expr.str "aether") "len" []) =
      some (Value.num 6) := by
  intro _
  native_decide

theorem evalExprWithFnsRel_call_return_sound :
    EvalExprWithFnsRel []
      [("id", { params := ["x"], body := [Stmt.ret (some (Expr.var "x"))] })]
      (Expr.call "id" [Expr.num 3])
      (Value.num 3) ->
    evalExprWithFns 3 []
      [("id", { params := ["x"], body := [Stmt.ret (some (Expr.var "x"))] })]
      (Expr.call "id" [Expr.num 3]) = some (Value.num 3) := by
  intro _
  native_decide

theorem evalExprWithFnsRel_call_value_sound :
    EvalExprWithFnsRel []
      [("one", { params := [], body := [Stmt.expr (Expr.num 1)] })]
      (Expr.call "one" [])
      (Value.num 1) ->
    evalExprWithFns 3 []
      [("one", { params := [], body := [Stmt.expr (Expr.num 1)] })]
      (Expr.call "one" []) = some (Value.num 1) := by
  intro _
  native_decide

theorem stepStmtWithFns_let_num_exec_sound :
    StepStmtWithFns [] [] (Stmt.letDecl "x" (Expr.num 7))
      [("x", Value.num 7)] [] (Flow.value (Value.num 7)) ->
    (execStmtWithFns 2 [] [] (Stmt.letDecl "x" (Expr.num 7))).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 7)], Flow.value (Value.num 7)) := by
  intro _
  native_decide

theorem stepStmtWithFns_fn_decl_exec_sound :
    StepStmtWithFns [] [] (Stmt.fnDecl "id" ["x"] [Stmt.ret (some (Expr.var "x"))])
      []
      [("id", { params := ["x"], body := [Stmt.ret (some (Expr.var "x"))] })]
      (Flow.value Value.unit) ->
    (execStmtWithFns 1 [] [] (Stmt.fnDecl "id" ["x"] [Stmt.ret (some (Expr.var "x"))])).map
      (fun result =>
        ( result.1
        , result.2.1.length
        , result.2.2)) =
      some ([], 1, Flow.value Value.unit) := by
  intro _
  native_decide

theorem stepStmtWithFns_return_var_exec_sound :
    StepStmtWithFns [("x", Value.num 5)] [] (Stmt.ret (some (Expr.var "x")))
      [("x", Value.num 5)] [] (Flow.return (Value.num 5)) ->
    (execStmtWithFns 2 [("x", Value.num 5)] [] (Stmt.ret (some (Expr.var "x")))).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 5)], Flow.return (Value.num 5)) := by
  intro _
  native_decide

theorem stepStmtWithFns_assign_num_exec_sound :
    StepStmtWithFns [("x", Value.num 1)] [] (Stmt.assign "x" (Expr.num 9))
      [("x", Value.num 9)] [] (Flow.value (Value.num 9)) ->
    (execStmtWithFns 2 [("x", Value.num 1)] [] (Stmt.assign "x" (Expr.num 9))).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 9)], Flow.value (Value.num 9)) := by
  intro _
  native_decide

theorem stepStmtWithFns_expr_bool_exec_sound :
    StepStmtWithFns [] [] (Stmt.expr (Expr.bool true)) [] [] (Flow.value (Value.bool true)) ->
    (execStmtWithFns 2 [] [] (Stmt.expr (Expr.bool true))).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.value (Value.bool true)) := by
  intro _
  native_decide

theorem stepStmtWithFns_return_none_exec_sound :
    StepStmtWithFns [] [] (Stmt.ret none) [] [] (Flow.return Value.unit) ->
    (execStmtWithFns 1 [] [] (Stmt.ret none)).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.return Value.unit) := by
  intro _
  native_decide

theorem stepStmtWithFns_break_exec_sound :
    StepStmtWithFns [] [] Stmt.break [] [] Flow.break ->
    (execStmtWithFns 1 [] [] Stmt.break).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.break) := by
  intro _
  native_decide

theorem stepStmtWithFns_continue_exec_sound :
    StepStmtWithFns [] [] Stmt.continue [] [] Flow.continue ->
    (execStmtWithFns 1 [] [] Stmt.continue).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.continue) := by
  intro _
  native_decide

theorem stepBlockWithFns_nil_exec_sound :
    StepBlockWithFns [] [] [] [] [] (Flow.value Value.unit) ->
    (execBlockWithFns 1 [] [] []).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.value Value.unit) := by
  intro _
  native_decide

theorem stepBlockWithFns_single_expr_exec_sound :
    StepBlockWithFns [] [] [Stmt.expr (Expr.bool true)] [] [] (Flow.value (Value.bool true)) ->
    (execBlockWithFns 2 [] [] [Stmt.expr (Expr.bool true)]).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.value (Value.bool true)) := by
  intro _
  native_decide

theorem stepBlockWithFns_cons_value_exec_sound :
    StepBlockWithFns [] []
      [Stmt.letDecl "x" (Expr.num 7), Stmt.expr (Expr.var "x")]
      [("x", Value.num 7)] [] (Flow.value (Value.num 7)) ->
    (execBlockWithFns 2 [] []
      [Stmt.letDecl "x" (Expr.num 7), Stmt.expr (Expr.var "x")]).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 7)], Flow.value (Value.num 7)) := by
  intro _
  native_decide

theorem stepBlockWithFns_cons_return_exec_sound :
    StepBlockWithFns [] []
      [Stmt.ret (some (Expr.num 1)), Stmt.expr (Expr.num 2)]
      [] [] (Flow.return (Value.num 1)) ->
    (execBlockWithFns 2 [] [] [Stmt.ret (some (Expr.num 1)), Stmt.expr (Expr.num 2)]).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.return (Value.num 1)) := by
  intro _
  native_decide

theorem stepBlockWithFns_cons_break_exec_sound :
    StepBlockWithFns [] [] [Stmt.break, Stmt.expr (Expr.num 2)] [] [] Flow.break ->
    (execBlockWithFns 1 [] [] [Stmt.break, Stmt.expr (Expr.num 2)]).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.break) := by
  intro _
  native_decide

theorem stepBlockWithFns_cons_continue_exec_sound :
    StepBlockWithFns [] [] [Stmt.continue, Stmt.expr (Expr.num 2)] [] [] Flow.continue ->
    (execBlockWithFns 1 [] [] [Stmt.continue, Stmt.expr (Expr.num 2)]).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.continue) := by
  intro _
  native_decide

theorem stepStmtWithFns_if_true_exec_sound :
    StepStmtWithFns [] []
      (Stmt.ifThenElse
        (Expr.bool true)
        [Stmt.letDecl "x" (Expr.num 1)]
        (some [Stmt.letDecl "x" (Expr.num 2)]))
      [("x", Value.num 1)] [] (Flow.value (Value.num 1)) ->
    (execStmtWithFns 3 [] []
      (Stmt.ifThenElse
        (Expr.bool true)
        [Stmt.letDecl "x" (Expr.num 1)]
        (some [Stmt.letDecl "x" (Expr.num 2)]))).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 1)], Flow.value (Value.num 1)) := by
  intro _
  native_decide

theorem stepStmtWithFns_if_false_some_exec_sound :
    StepStmtWithFns [] []
      (Stmt.ifThenElse
        (Expr.bool false)
        [Stmt.letDecl "x" (Expr.num 1)]
        (some [Stmt.letDecl "x" (Expr.num 2)]))
      [("x", Value.num 2)] [] (Flow.value (Value.num 2)) ->
    (execStmtWithFns 3 [] []
      (Stmt.ifThenElse
        (Expr.bool false)
        [Stmt.letDecl "x" (Expr.num 1)]
        (some [Stmt.letDecl "x" (Expr.num 2)]))).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 2)], Flow.value (Value.num 2)) := by
  intro _
  native_decide

theorem stepStmtWithFns_if_false_none_exec_sound :
    StepStmtWithFns [] []
      (Stmt.ifThenElse
        (Expr.bool false)
        [Stmt.letDecl "x" (Expr.num 1)]
        none)
      [] [] (Flow.value Value.unit) ->
    (execStmtWithFns 2 [] []
      (Stmt.ifThenElse
        (Expr.bool false)
        [Stmt.letDecl "x" (Expr.num 1)]
        none)).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.value Value.unit) := by
  intro _
  native_decide

theorem stepStmtWithFns_while_false_exec_sound :
    StepStmtWithFns [("x", Value.num 0)] []
      (Stmt.while (Expr.bool false) [Stmt.assign "x" (Expr.num 1)])
      [("x", Value.num 0)] [] (Flow.value Value.unit) ->
    (execStmtWithFns 2 [("x", Value.num 0)] []
      (Stmt.while (Expr.bool false) [Stmt.assign "x" (Expr.num 1)])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 0)], Flow.value Value.unit) := by
  intro _
  native_decide

theorem stepStmtWithFns_while_return_exec_sound :
    StepStmtWithFns [] []
      (Stmt.while (Expr.bool true) [Stmt.ret (some (Expr.num 4))])
      [] [] (Flow.return (Value.num 4)) ->
    (execStmtWithFns 3 [] []
      (Stmt.while (Expr.bool true) [Stmt.ret (some (Expr.num 4))])).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.return (Value.num 4)) := by
  intro _
  native_decide

theorem stepStmtWithFns_while_break_exec_sound :
    StepStmtWithFns [] []
      (Stmt.while (Expr.bool true) [Stmt.break])
      [] [] (Flow.value Value.unit) ->
    (execStmtWithFns 2 [] []
      (Stmt.while (Expr.bool true) [Stmt.break])).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.value Value.unit) := by
  intro _
  native_decide

theorem stepStmtWithFns_for_done_exec_sound :
    StepStmtWithFns [] []
      (Stmt.forRange "i" 2 2 [Stmt.expr (Expr.num 9)])
      [("i", Value.num 2)] [] (Flow.value Value.unit) ->
    (execStmtWithFns 1 [] []
      (Stmt.forRange "i" 2 2 [Stmt.expr (Expr.num 9)])).map
      (fun result => (result.1, result.2.2)) =
      some ([("i", Value.num 2)], Flow.value Value.unit) := by
  intro _
  native_decide

theorem stepStmtWithFns_for_return_exec_sound :
    StepStmtWithFns [] []
      (Stmt.forRange "i" 0 1 [Stmt.ret (some (Expr.var "i"))])
      [("i", Value.num 0)] [] (Flow.return (Value.num 0)) ->
    (execStmtWithFns 3 [] []
      (Stmt.forRange "i" 0 1 [Stmt.ret (some (Expr.var "i"))])).map
      (fun result => (result.1, result.2.2)) =
      some ([("i", Value.num 0)], Flow.return (Value.num 0)) := by
  intro _
  native_decide

theorem stepStmtWithFns_for_break_exec_sound :
    StepStmtWithFns [] []
      (Stmt.forRange "i" 0 1 [Stmt.break])
      [("i", Value.num 0)] [] (Flow.value Value.unit) ->
    (execStmtWithFns 2 [] []
      (Stmt.forRange "i" 0 1 [Stmt.break])).map
      (fun result => (result.1, result.2.2)) =
      some ([("i", Value.num 0)], Flow.value Value.unit) := by
  intro _
  native_decide

theorem stepStmtWithFns_for_value_exec_sound :
    StepStmtWithFns [("x", Value.num 9)] []
      (Stmt.forRange "i" 0 1 [Stmt.assign "x" (Expr.var "i")])
      [("i", Value.num 1), ("i", Value.num 0), ("x", Value.num 0)] []
      (Flow.value Value.unit) ->
    (execStmtWithFns 3 [("x", Value.num 9)] []
      (Stmt.forRange "i" 0 1 [Stmt.assign "x" (Expr.var "i")])).map
      (fun result => (result.1, result.2.2)) =
      some ([("i", Value.num 1), ("i", Value.num 0), ("x", Value.num 0)],
        Flow.value Value.unit) := by
  intro _
  native_decide

theorem stepStmtWithFns_for_continue_exec_sound :
    StepStmtWithFns [] []
      (Stmt.forRange "i" 0 1 [Stmt.continue])
      [("i", Value.num 1), ("i", Value.num 0)] [] (Flow.value Value.unit) ->
    (execStmtWithFns 3 [] []
      (Stmt.forRange "i" 0 1 [Stmt.continue])).map
      (fun result => (result.1, result.2.2)) =
      some ([("i", Value.num 1), ("i", Value.num 0)], Flow.value Value.unit) := by
  intro _
  native_decide

theorem stepStmtWithFns_seal_until_done_exec_sound :
    StepStmtWithFns [("x", Value.num 0)] []
      (Stmt.seal (some (Expr.bool true)) [Stmt.assign "x" (Expr.num 1)])
      [("x", Value.num 0)] [] (Flow.value Value.unit) ->
    (execStmtWithFns 2 [("x", Value.num 0)] []
      (Stmt.seal (some (Expr.bool true)) [Stmt.assign "x" (Expr.num 1)])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 0)], Flow.value Value.unit) := by
  intro _
  native_decide

theorem stepStmtWithFns_seal_until_value_exec_sound :
    StepStmtWithFns [("x", Value.num 0)] []
      (Stmt.seal
        (some (Expr.var "x"))
        [Stmt.assign "x" (Expr.num 1)])
      [("x", Value.num 1)] [] (Flow.value Value.unit) ->
    (execStmtWithFns 3 [("x", Value.num 0)] []
      (Stmt.seal
        (some (Expr.var "x"))
        [Stmt.assign "x" (Expr.num 1)])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 1)], Flow.value Value.unit) := by
  intro _
  native_decide

theorem stepStmtWithFns_seal_until_break_exec_sound :
    StepStmtWithFns [("x", Value.num 0)] []
      (Stmt.seal (some (Expr.bool false)) [Stmt.break])
      [("x", Value.num 0)] [] (Flow.value Value.unit) ->
    (execStmtWithFns 2 [("x", Value.num 0)] []
      (Stmt.seal (some (Expr.bool false)) [Stmt.break])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 0)], Flow.value Value.unit) := by
  intro _
  native_decide

theorem stepStmtWithFns_seal_until_return_exec_sound :
    StepStmtWithFns [("x", Value.num 0)] []
      (Stmt.seal (some (Expr.bool false)) [Stmt.ret (some (Expr.num 5))])
      [("x", Value.num 0)] [] (Flow.return (Value.num 5)) ->
    (execStmtWithFns 3 [("x", Value.num 0)] []
      (Stmt.seal (some (Expr.bool false)) [Stmt.ret (some (Expr.num 5))])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 0)], Flow.return (Value.num 5)) := by
  intro _
  native_decide

theorem stepStmtWithFns_seal_until_continue_exec_sound :
    StepStmtWithFns [("x", Value.num 0)] []
      (Stmt.seal
        (some (Expr.var "x"))
        [Stmt.assign "x" (Expr.num 1), Stmt.continue])
      [("x", Value.num 1)] [] (Flow.value Value.unit) ->
    (execStmtWithFns 3 [("x", Value.num 0)] []
      (Stmt.seal
        (some (Expr.var "x"))
        [Stmt.assign "x" (Expr.num 1), Stmt.continue])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 1)], Flow.value Value.unit) := by
  intro _
  native_decide

theorem stepStmtWithFns_seal_value_exec_sound :
    StepStmtWithFns [("x", Value.num 0)] []
      (Stmt.seal none
        [ Stmt.ifThenElse
            (Expr.var "x")
            [Stmt.break]
            (some [Stmt.assign "x" (Expr.num 1)])])
      [("x", Value.num 1)] [] (Flow.value Value.unit) ->
    (execStmtWithFns 4 [("x", Value.num 0)] []
      (Stmt.seal none
        [ Stmt.ifThenElse
            (Expr.var "x")
            [Stmt.break]
            (some [Stmt.assign "x" (Expr.num 1)])])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 1)], Flow.value Value.unit) := by
  intro _
  native_decide

theorem stepStmtWithFns_seal_return_exec_sound :
    StepStmtWithFns [("x", Value.num 0)] []
      (Stmt.seal none [Stmt.ret (some (Expr.num 5))])
      [("x", Value.num 0)] [] (Flow.return (Value.num 5)) ->
    (execStmtWithFns 3 [("x", Value.num 0)] []
      (Stmt.seal none [Stmt.ret (some (Expr.num 5))])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 0)], Flow.return (Value.num 5)) := by
  intro _
  native_decide

theorem stepStmtWithFns_seal_break_exec_sound :
    StepStmtWithFns [("x", Value.num 0)] []
      (Stmt.seal none [Stmt.break])
      [("x", Value.num 0)] [] (Flow.value Value.unit) ->
    (execStmtWithFns 2 [("x", Value.num 0)] []
      (Stmt.seal none [Stmt.break])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 0)], Flow.value Value.unit) := by
  intro _
  native_decide

theorem stepStmtWithFns_seal_continue_exec_sound :
    StepStmtWithFns [("x", Value.num 0)] []
      (Stmt.seal none
        [ Stmt.ifThenElse
            (Expr.var "x")
            [Stmt.break]
            (some [Stmt.assign "x" (Expr.num 1), Stmt.continue])])
      [("x", Value.num 1)] [] (Flow.value Value.unit) ->
    (execStmtWithFns 4 [("x", Value.num 0)] []
      (Stmt.seal none
        [ Stmt.ifThenElse
            (Expr.var "x")
            [Stmt.break]
            (some [Stmt.assign "x" (Expr.num 1), Stmt.continue])])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 1)], Flow.value Value.unit) := by
  intro _
  native_decide

example :
    evalExprWithFns 1 [] [] (Expr.num 7) = some (Value.num 7) := by
  apply evalExprWithFnsRel_num_sound
  apply EvalExprWithFnsRel.num

example :
    evalExprWithFns 1 [] [] (Expr.bool true) = some (Value.bool true) := by
  apply evalExprWithFnsRel_bool_sound
  apply EvalExprWithFnsRel.bool

example :
    evalExprWithFns 1 [("x", Value.str "open")] [] (Expr.var "x") =
      some (Value.str "open") := by
  apply evalExprWithFnsRel_var_sound
  apply EvalExprWithFnsRel.var
  rfl

example :
    evalExprWithFns 2 [] [] (Expr.binary (Expr.num 2) BinOp.add (Expr.num 5)) =
      some (Value.num 7) := by
  apply evalExprWithFnsRel_binary_add_sound
  apply EvalExprWithFnsRel.binary
  · apply EvalExprWithFnsRel.num
  · apply EvalExprWithFnsRel.num
  · rfl

example :
    evalExprWithFns 2 [] [] (Expr.unary UnOp.not (Expr.bool false)) =
      some (Value.bool true) := by
  apply evalExprWithFnsRel_unary_not_sound
  apply EvalExprWithFnsRel.unary
  · apply EvalExprWithFnsRel.bool
  · rfl

example :
    evalExprWithFns 2 [] [] (Expr.list [Expr.num 1, Expr.bool true]) =
      some (Value.list [Value.num 1, Value.bool true]) := by
  apply evalExprWithFnsRel_list_sound
  apply EvalExprWithFnsRel.list
  · apply EvalExprsWithFnsRel.cons
    · apply EvalExprWithFnsRel.num
    · apply EvalExprsWithFnsRel.cons
      · apply EvalExprWithFnsRel.bool
      · apply EvalExprsWithFnsRel.nil

example :
    evalExprWithFns 3 [] [] (Expr.index (Expr.list [Expr.str "a", Expr.str "b"]) (Expr.num 1)) =
      some (Value.str "b") := by
  apply evalExprWithFnsRel_index_sound
  apply EvalExprWithFnsRel.index
  · apply EvalExprWithFnsRel.list
    apply EvalExprsWithFnsRel.cons
    · apply EvalExprWithFnsRel.str
    · apply EvalExprsWithFnsRel.cons
      · apply EvalExprWithFnsRel.str
      · apply EvalExprsWithFnsRel.nil
  · apply EvalExprWithFnsRel.num
  · rfl

example :
    evalExprWithFns 3 [] [] (Expr.field (Expr.list [Expr.num 1, Expr.num 2]) "length") =
      some (Value.num 2) := by
  apply evalExprWithFnsRel_field_length_sound
  apply EvalExprWithFnsRel.field
  · apply EvalExprWithFnsRel.list
    apply EvalExprsWithFnsRel.cons
    · apply EvalExprWithFnsRel.num
    · apply EvalExprsWithFnsRel.cons
      · apply EvalExprWithFnsRel.num
      · apply EvalExprsWithFnsRel.nil
  · rfl

example :
    evalExprWithFns 2 [] [] (Expr.method (Expr.str "aether") "len" []) =
      some (Value.num 6) := by
  apply evalExprWithFnsRel_method_len_sound
  apply EvalExprWithFnsRel.method
  · apply EvalExprWithFnsRel.str
  · apply EvalArgsWithFnsRel.nil
  · rfl

example :
    evalExprWithFns 3 []
      [("id", { params := ["x"], body := [Stmt.ret (some (Expr.var "x"))] })]
      (Expr.call "id" [Expr.num 3]) = some (Value.num 3) := by
  apply evalExprWithFnsRel_call_return_sound
  apply EvalExprWithFnsRel.callReturn
  · rfl
  · apply EvalArgsWithFnsRel.cons
    · apply EvalArgWithFnsRel.positional
      apply EvalExprWithFnsRel.num
    · apply EvalArgsWithFnsRel.nil
  · rfl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.retSome
    apply EvalExprWithFnsRel.var
    rfl

example :
    evalExprWithFns 3 []
      [("one", { params := [], body := [Stmt.expr (Expr.num 1)] })]
      (Expr.call "one" []) = some (Value.num 1) := by
  apply evalExprWithFnsRel_call_value_sound
  apply EvalExprWithFnsRel.callValue
  · rfl
  · apply EvalArgsWithFnsRel.nil
  · rfl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.expr
    apply EvalExprWithFnsRel.num

example :
    (execStmtWithFns 2 [] [] (Stmt.letDecl "x" (Expr.num 7))).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 7)], Flow.value (Value.num 7)) := by
  apply stepStmtWithFns_let_num_exec_sound
  apply StepStmtWithFns.letDecl
  apply EvalExprWithFnsRel.num

example :
    (execStmtWithFns 1 [] [] (Stmt.fnDecl "id" ["x"] [Stmt.ret (some (Expr.var "x"))])).map
      (fun result =>
        ( result.1
        , result.2.1.length
        , result.2.2)) =
      some ([], 1, Flow.value Value.unit) := by
  apply stepStmtWithFns_fn_decl_exec_sound
  apply StepStmtWithFns.fnDecl

example :
    (execStmtWithFns 2 [("x", Value.num 5)] [] (Stmt.ret (some (Expr.var "x")))).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 5)], Flow.return (Value.num 5)) := by
  apply stepStmtWithFns_return_var_exec_sound
  apply StepStmtWithFns.retSome
  apply EvalExprWithFnsRel.var
  rfl

example :
    (execStmtWithFns 2 [("x", Value.num 1)] [] (Stmt.assign "x" (Expr.num 9))).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 9)], Flow.value (Value.num 9)) := by
  apply stepStmtWithFns_assign_num_exec_sound
  apply StepStmtWithFns.assign
  · apply EvalExprWithFnsRel.num
  · rfl

example :
    (execStmtWithFns 2 [] [] (Stmt.expr (Expr.bool true))).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.value (Value.bool true)) := by
  apply stepStmtWithFns_expr_bool_exec_sound
  apply StepStmtWithFns.expr
  apply EvalExprWithFnsRel.bool

example :
    (execStmtWithFns 1 [] [] (Stmt.ret none)).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.return Value.unit) := by
  apply stepStmtWithFns_return_none_exec_sound
  apply StepStmtWithFns.retNone

example :
    (execStmtWithFns 1 [] [] Stmt.break).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.break) := by
  apply stepStmtWithFns_break_exec_sound
  apply StepStmtWithFns.break

example :
    (execStmtWithFns 1 [] [] Stmt.continue).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.continue) := by
  apply stepStmtWithFns_continue_exec_sound
  apply StepStmtWithFns.continue

example :
    (execBlockWithFns 1 [] [] []).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.value Value.unit) := by
  apply stepBlockWithFns_nil_exec_sound
  apply StepBlockWithFns.nil

example :
    (execBlockWithFns 2 [] [] [Stmt.expr (Expr.bool true)]).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.value (Value.bool true)) := by
  apply stepBlockWithFns_single_expr_exec_sound
  apply StepBlockWithFns.single
  apply StepStmtWithFns.expr
  apply EvalExprWithFnsRel.bool

example :
    (execBlockWithFns 2 [] []
      [Stmt.letDecl "x" (Expr.num 7), Stmt.expr (Expr.var "x")]).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 7)], Flow.value (Value.num 7)) := by
  apply stepBlockWithFns_cons_value_exec_sound
  apply StepBlockWithFns.consValue
  · apply StepStmtWithFns.letDecl
    apply EvalExprWithFnsRel.num
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.expr
    apply EvalExprWithFnsRel.var
    rfl

example :
    (execBlockWithFns 2 [] [] [Stmt.ret (some (Expr.num 1)), Stmt.expr (Expr.num 2)]).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.return (Value.num 1)) := by
  apply stepBlockWithFns_cons_return_exec_sound
  apply StepBlockWithFns.consReturn
  apply StepStmtWithFns.retSome
  apply EvalExprWithFnsRel.num

example :
    (execBlockWithFns 1 [] [] [Stmt.break, Stmt.expr (Expr.num 2)]).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.break) := by
  apply stepBlockWithFns_cons_break_exec_sound
  apply StepBlockWithFns.consBreak
  apply StepStmtWithFns.break

example :
    (execBlockWithFns 1 [] [] [Stmt.continue, Stmt.expr (Expr.num 2)]).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.continue) := by
  apply stepBlockWithFns_cons_continue_exec_sound
  apply StepBlockWithFns.consContinue
  apply StepStmtWithFns.continue

example :
    (execStmtWithFns 3 [] []
      (Stmt.ifThenElse
        (Expr.bool true)
        [Stmt.letDecl "x" (Expr.num 1)]
        (some [Stmt.letDecl "x" (Expr.num 2)]) )).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 1)], Flow.value (Value.num 1)) := by
  apply stepStmtWithFns_if_true_exec_sound
  apply StepStmtWithFns.ifTrue
  · apply EvalExprWithFnsRel.bool
  · rfl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.letDecl
    apply EvalExprWithFnsRel.num

example :
    (execStmtWithFns 3 [] []
      (Stmt.ifThenElse
        (Expr.bool false)
        [Stmt.letDecl "x" (Expr.num 1)]
        (some [Stmt.letDecl "x" (Expr.num 2)]) )).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 2)], Flow.value (Value.num 2)) := by
  apply stepStmtWithFns_if_false_some_exec_sound
  apply StepStmtWithFns.ifFalseSome
  · apply EvalExprWithFnsRel.bool
  · rfl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.letDecl
    apply EvalExprWithFnsRel.num

example :
    (execStmtWithFns 2 [] []
      (Stmt.ifThenElse
        (Expr.bool false)
        [Stmt.letDecl "x" (Expr.num 1)]
        none)).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.value Value.unit) := by
  apply stepStmtWithFns_if_false_none_exec_sound
  apply StepStmtWithFns.ifFalseNone
  · apply EvalExprWithFnsRel.bool
  · rfl

example :
    (execStmtWithFns 2 [("x", Value.num 0)] []
      (Stmt.while (Expr.bool false) [Stmt.assign "x" (Expr.num 1)])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 0)], Flow.value Value.unit) := by
  apply stepStmtWithFns_while_false_exec_sound
  apply StepStmtWithFns.whileFalse
  · apply EvalExprWithFnsRel.bool
  · rfl

example :
    (execStmtWithFns 3 [] []
      (Stmt.while (Expr.bool true) [Stmt.ret (some (Expr.num 4))])).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.return (Value.num 4)) := by
  apply stepStmtWithFns_while_return_exec_sound
  apply StepStmtWithFns.whileReturn
  · apply EvalExprWithFnsRel.bool
  · rfl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.retSome
    apply EvalExprWithFnsRel.num

example :
    (execStmtWithFns 2 [] []
      (Stmt.while (Expr.bool true) [Stmt.break])).map
      (fun result => (result.1, result.2.2)) =
      some ([], Flow.value Value.unit) := by
  apply stepStmtWithFns_while_break_exec_sound
  apply StepStmtWithFns.whileBreak
  · apply EvalExprWithFnsRel.bool
  · rfl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.break

example :
    (execStmtWithFns 1 [] []
      (Stmt.forRange "i" 2 2 [Stmt.expr (Expr.num 9)])).map
      (fun result => (result.1, result.2.2)) =
      some ([("i", Value.num 2)], Flow.value Value.unit) := by
  apply stepStmtWithFns_for_done_exec_sound
  apply StepStmtWithFns.forDone
  decide

example :
    (execStmtWithFns 3 [] []
      (Stmt.forRange "i" 0 1 [Stmt.ret (some (Expr.var "i"))])).map
      (fun result => (result.1, result.2.2)) =
      some ([("i", Value.num 0)], Flow.return (Value.num 0)) := by
  apply stepStmtWithFns_for_return_exec_sound
  apply StepStmtWithFns.forReturn
  · decide
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.retSome
    apply EvalExprWithFnsRel.var
    rfl

example :
    (execStmtWithFns 2 [] []
      (Stmt.forRange "i" 0 1 [Stmt.break])).map
      (fun result => (result.1, result.2.2)) =
      some ([("i", Value.num 0)], Flow.value Value.unit) := by
  apply stepStmtWithFns_for_break_exec_sound
  apply StepStmtWithFns.forBreak
  · decide
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.break

example :
    (execStmtWithFns 3 [("x", Value.num 9)] []
      (Stmt.forRange "i" 0 1 [Stmt.assign "x" (Expr.var "i")])).map
      (fun result => (result.1, result.2.2)) =
      some ([("i", Value.num 1), ("i", Value.num 0), ("x", Value.num 0)],
        Flow.value Value.unit) := by
  apply stepStmtWithFns_for_value_exec_sound
  apply StepStmtWithFns.forValue
  · decide
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.assign
    · apply EvalExprWithFnsRel.var
      rfl
    · rfl
  · apply StepStmtWithFns.forDone
    decide

example :
    (execStmtWithFns 3 [] []
      (Stmt.forRange "i" 0 1 [Stmt.continue])).map
      (fun result => (result.1, result.2.2)) =
      some ([("i", Value.num 1), ("i", Value.num 0)], Flow.value Value.unit) := by
  apply stepStmtWithFns_for_continue_exec_sound
  apply StepStmtWithFns.forContinue
  · decide
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.continue
  · apply StepStmtWithFns.forDone
    decide

example :
    (execStmtWithFns 2 [("x", Value.num 0)] []
      (Stmt.seal (some (Expr.bool true)) [Stmt.assign "x" (Expr.num 1)])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 0)], Flow.value Value.unit) := by
  apply stepStmtWithFns_seal_until_done_exec_sound
  apply StepStmtWithFns.sealUntilDone
  · apply EvalExprWithFnsRel.bool
  · rfl

example :
    (execStmtWithFns 3 [("x", Value.num 0)] []
      (Stmt.seal
        (some (Expr.var "x"))
        [Stmt.assign "x" (Expr.num 1)])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 1)], Flow.value Value.unit) := by
  apply stepStmtWithFns_seal_until_value_exec_sound
  apply StepStmtWithFns.sealUntilValue
  · apply EvalExprWithFnsRel.var
    rfl
  · rfl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.assign
    · apply EvalExprWithFnsRel.num
    · rfl
  · apply StepStmtWithFns.sealUntilDone
    · apply EvalExprWithFnsRel.var
      rfl
    · rfl

example :
    (execStmtWithFns 2 [("x", Value.num 0)] []
      (Stmt.seal (some (Expr.bool false)) [Stmt.break])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 0)], Flow.value Value.unit) := by
  apply stepStmtWithFns_seal_until_break_exec_sound
  apply StepStmtWithFns.sealUntilBreak
  · apply EvalExprWithFnsRel.bool
  · rfl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.break

example :
    (execStmtWithFns 3 [("x", Value.num 0)] []
      (Stmt.seal (some (Expr.bool false)) [Stmt.ret (some (Expr.num 5))])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 0)], Flow.return (Value.num 5)) := by
  apply stepStmtWithFns_seal_until_return_exec_sound
  apply StepStmtWithFns.sealUntilReturn
  · apply EvalExprWithFnsRel.bool
  · rfl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.retSome
    apply EvalExprWithFnsRel.num

example :
    (execStmtWithFns 3 [("x", Value.num 0)] []
      (Stmt.seal
        (some (Expr.var "x"))
        [Stmt.assign "x" (Expr.num 1), Stmt.continue])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 1)], Flow.value Value.unit) := by
  apply stepStmtWithFns_seal_until_continue_exec_sound
  apply StepStmtWithFns.sealUntilContinue
  · apply EvalExprWithFnsRel.var
    rfl
  · rfl
  · apply StepBlockWithFns.consValue
    · apply StepStmtWithFns.assign
      · apply EvalExprWithFnsRel.num
      · rfl
    · apply StepBlockWithFns.single
      apply StepStmtWithFns.continue
  · apply StepStmtWithFns.sealUntilDone
    · apply EvalExprWithFnsRel.var
      rfl
    · rfl

example :
    (execStmtWithFns 4 [("x", Value.num 0)] []
      (Stmt.seal none
        [ Stmt.ifThenElse
            (Expr.var "x")
            [Stmt.break]
            (some [Stmt.assign "x" (Expr.num 1)])])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 1)], Flow.value Value.unit) := by
  apply stepStmtWithFns_seal_value_exec_sound
  apply StepStmtWithFns.sealValue
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.ifFalseSome
    · apply EvalExprWithFnsRel.var
      rfl
    · rfl
    · apply StepBlockWithFns.single
      apply StepStmtWithFns.assign
      · apply EvalExprWithFnsRel.num
      · rfl
  · apply StepStmtWithFns.sealBreak
    apply StepBlockWithFns.single
    apply StepStmtWithFns.ifTrue
    · apply EvalExprWithFnsRel.var
      rfl
    · rfl
    · apply StepBlockWithFns.single
      apply StepStmtWithFns.break

example :
    (execStmtWithFns 3 [("x", Value.num 0)] []
      (Stmt.seal none [Stmt.ret (some (Expr.num 5))])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 0)], Flow.return (Value.num 5)) := by
  apply stepStmtWithFns_seal_return_exec_sound
  apply StepStmtWithFns.sealReturn
  apply StepBlockWithFns.single
  apply StepStmtWithFns.retSome
  apply EvalExprWithFnsRel.num

example :
    (execStmtWithFns 2 [("x", Value.num 0)] []
      (Stmt.seal none [Stmt.break])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 0)], Flow.value Value.unit) := by
  apply stepStmtWithFns_seal_break_exec_sound
  apply StepStmtWithFns.sealBreak
  apply StepBlockWithFns.single
  apply StepStmtWithFns.break

example :
    (execStmtWithFns 4 [("x", Value.num 0)] []
      (Stmt.seal none
        [ Stmt.ifThenElse
            (Expr.var "x")
            [Stmt.break]
            (some [Stmt.assign "x" (Expr.num 1), Stmt.continue])])).map
      (fun result => (result.1, result.2.2)) =
      some ([("x", Value.num 1)], Flow.value Value.unit) := by
  apply stepStmtWithFns_seal_continue_exec_sound
  apply StepStmtWithFns.sealContinue
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.ifFalseSome
    · apply EvalExprWithFnsRel.var
      rfl
    · rfl
    · apply StepBlockWithFns.consValue
      · apply StepStmtWithFns.assign
        · apply EvalExprWithFnsRel.num
        · rfl
      · apply StepBlockWithFns.single
        apply StepStmtWithFns.continue
  · apply StepStmtWithFns.sealBreak
    apply StepBlockWithFns.single
    apply StepStmtWithFns.ifTrue
    · apply EvalExprWithFnsRel.var
      rfl
    · rfl
    · apply StepBlockWithFns.single
      apply StepStmtWithFns.break

example :
    evalExpr [] (Expr.binary (Expr.num 10) BinOp.mod (Expr.num 4)) = some (Value.num 2) := by
  native_decide

example :
    evalExpr [] (Expr.float 1 500000) = some (Value.float 1 500000) := by
  native_decide

example :
    evalExpr [] (Expr.binary (Expr.float 1 500000) BinOp.add (Expr.num 2)) =
      some (Value.float 3 500000) := by
  native_decide

example :
    evalExpr [] (Expr.binary (Expr.float 1 500000) BinOp.lt (Expr.num 2)) =
      some (Value.bool true) := by
  native_decide

example :
    evalExpr [] (Expr.binary (Expr.bool true) BinOp.and (Expr.num 1)) = some (Value.bool true) := by
  native_decide

example :
    evalExpr [] (Expr.str "open") = some (Value.str "open") := by
  native_decide

example :
    evalExpr [] Expr.unit = some Value.unit := by
  native_decide

example :
    evalExpr [] (Expr.binary (Expr.str "open") BinOp.eq (Expr.str "open")) =
      some (Value.bool true) := by
  native_decide

example :
    evalExpr [] (Expr.list [Expr.num 1, Expr.bool true, Expr.str "open"]) =
      some (Value.list [Value.num 1, Value.bool true, Value.str "open"]) := by
  native_decide

example :
    evalExpr [] (Expr.binary (Expr.list [Expr.num 1]) BinOp.eq (Expr.list [Expr.num 1])) =
      some (Value.bool true) := by
  native_decide

example :
    evalExpr [] (Expr.index (Expr.list [Expr.str "a", Expr.str "b"]) (Expr.num 1)) =
      some (Value.str "b") := by
  native_decide

example :
    evalExpr [] (Expr.index (Expr.list [Expr.num 1]) (Expr.num 3)) = none := by
  native_decide

example :
    evalExpr [] (Expr.index (Expr.str "open") (Expr.num 1)) =
      some (Value.str "p") := by
  native_decide

example :
    evalExpr [] (Expr.index (Expr.str "open") (Expr.num 9)) = none := by
  native_decide

example :
    evalExpr [] (Expr.field (Expr.list [Expr.num 1, Expr.num 2]) "length") =
      some (Value.num 2) := by
  native_decide

example :
    evalExpr [] (Expr.field (Expr.str "open") "length") = some (Value.num 4) := by
  native_decide

example :
    evalExpr [] (Expr.field (Expr.num 1) "length") = none := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 1, Expr.num 2]) "len" []) =
      some (Value.num 2) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "len" []) = some (Value.num 4) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list []) "is_empty" []) = some (Value.bool true) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "is_empty" []) = some (Value.bool false) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "at" [Expr.num 1]) =
      some (Value.str "p") := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "at" [Expr.num 9]) = none := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "contains" [Expr.str "pe"]) =
      some (Value.bool true) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "contains" [Expr.str "zz"]) =
      some (Value.bool false) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "starts_with" [Expr.str "op"]) =
      some (Value.bool true) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "starts_with" [Expr.str "pe"]) =
      some (Value.bool false) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "ends_with" [Expr.str "en"]) =
      some (Value.bool true) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "ends_with" [Expr.str "op"]) =
      some (Value.bool false) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "reverse" []) =
      some (Value.str "nepo") := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "first" []) =
      some (Value.str "o") := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "") "first" []) = none := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "last" []) =
      some (Value.str "n") := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "") "last" []) = none := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "tail" []) =
      some (Value.str "pen") := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "") "tail" []) = none := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "take" [Expr.num 2]) =
      some (Value.str "op") := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "take" [Expr.num (-1)]) = none := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "drop" [Expr.num 2]) =
      some (Value.str "en") := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.str "open") "drop" [Expr.num (-1)]) = none := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "first" []) =
      some (Value.num 7) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list []) "first" []) = none := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "last" []) =
      some (Value.num 9) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list []) "last" []) = none := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "tail" []) =
      some (Value.list [Value.num 9]) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list []) "tail" []) = none := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "at" [Expr.num 1]) =
      some (Value.num 9) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 7]) "at" [Expr.num 3]) = none := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "take" [Expr.num 1]) =
      some (Value.list [Value.num 7]) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "take" [Expr.num 3]) =
      some (Value.list [Value.num 7, Value.num 9]) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 7]) "take" [Expr.num (-1)]) = none := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "drop" [Expr.num 1]) =
      some (Value.list [Value.num 9]) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "drop" [Expr.num 3]) =
      some (Value.list []) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 7]) "drop" [Expr.num (-1)]) = none := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "reverse" []) =
      some (Value.list [Value.num 9, Value.num 7]) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list []) "reverse" []) =
      some (Value.list []) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 7]) "append" [Expr.num 9]) =
      some (Value.list [Value.num 7, Value.num 9]) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list []) "append" [Expr.num 9]) =
      some (Value.list [Value.num 9]) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 9]) "prepend" [Expr.num 7]) =
      some (Value.list [Value.num 7, Value.num 9]) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list []) "prepend" [Expr.num 7]) =
      some (Value.list [Value.num 7]) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.str "a", Expr.str "b"]) "join" [Expr.str ","]) =
      some (Value.str "a,b") := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list []) "join" [Expr.str ","]) =
      some (Value.str "") := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.str "a", Expr.num 1]) "join" [Expr.str ","]) =
      none := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 7]) "concat" [Expr.list [Expr.num 9]]) =
      some (Value.list [Value.num 7, Value.num 9]) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list []) "concat" [Expr.list [Expr.num 9]]) =
      some (Value.list [Value.num 9]) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 7, Expr.num 9]) "contains" [Expr.num 9]) =
      some (Value.bool true) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 7]) "contains" [Expr.num 3]) =
      some (Value.bool false) := by
  native_decide

example :
    evalExpr [] (Expr.method (Expr.list [Expr.num 1]) "len" [Expr.num 0]) = none := by
  native_decide

example :
    evalExprWithFns
      20
      []
      [ ( "add"
        , { params := ["a", "b"]
            body := [Stmt.ret (some (Expr.binary (Expr.var "a") BinOp.add (Expr.var "b")))] })]
      (Expr.call "add" [Expr.num 2, Expr.num 3])
      =
      some (Value.num 5) := by
  native_decide

example :
    evalExprWithFns
      20
      []
      [ ( "pick"
        , { params := ["a", "b"]
            body := [Stmt.ret (some (Expr.var "a"))] })]
      (Expr.call "pick" [Arg.named "b" (Expr.num 2), Arg.named "a" (Expr.num 7)])
      =
      some (Value.num 7) := by
  native_decide

example :
    evalExprWithFns
      20
      []
      [ ( "one"
        , { params := []
            body := [Stmt.letDecl "x" (Expr.num 1), Stmt.expr (Expr.var "x")] })]
      (Expr.call "one" [])
      =
      some (Value.num 1) := by
  native_decide

example :
    evalExprWithFns
      20
      []
      [ ( "id"
        , { params := ["x"]
            body := [Stmt.ret (some (Expr.var "x"))] })]
      (Expr.call "id" [Expr.num 1, Expr.num 2])
      =
      none := by
  native_decide

example :
    (execBlockWithFns
      30
      [("x", Value.num 10)]
      [ ( "id"
        , { params := ["x"]
            body := [Stmt.ret (some (Expr.var "x"))] })]
      [Stmt.letDecl "y" (Expr.call "id" [Expr.num 3])]
      ==
      some
        ( [("y", Value.num 3), ("x", Value.num 10)]
        , [ ( "id"
            , { params := ["x"]
                body := [Stmt.ret (some (Expr.var "x"))] })]
        , Flow.value (Value.num 3)
        )) = true := by
  native_decide

example :
    (execBlockWithFns
      20
      []
      []
      [ Stmt.ifThenElse
          (Expr.bool true)
          [Stmt.letDecl "x" (Expr.num 1)]
          (some [Stmt.letDecl "x" (Expr.num 2)])]
      ==
      some ([("x", Value.num 1)], [], Flow.value (Value.num 1))) = true := by
  native_decide

example :
    (execBlockWithFns
      20
      []
      []
      [ Stmt.ifThenElse
          (Expr.bool false)
          [Stmt.letDecl "x" (Expr.num 1)]
          (some [Stmt.letDecl "x" (Expr.num 2)])]
      ==
      some ([("x", Value.num 2)], [], Flow.value (Value.num 2))) = true := by
  native_decide

example :
    (execBlockWithFns
      20
      []
      []
      [ Stmt.ifThenElse
          (Expr.bool false)
          [Stmt.letDecl "x" (Expr.num 1)]
          none]
      ==
      some ([], [], Flow.value Value.unit)) = true := by
  native_decide

example :
    (execBlockWithFns
      20
      [("x", Value.num 0)]
      []
      [Stmt.while (Expr.bool false) [Stmt.assign "x" (Expr.num 1)]]
      ==
      some ([("x", Value.num 0)], [], Flow.value Value.unit)) = true := by
  native_decide

example :
    (execBlockWithFns
      40
      [("x", Value.num 0)]
      []
      [ Stmt.while
          (Expr.binary (Expr.var "x") BinOp.lt (Expr.num 3))
          [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))]]
      ==
      some ([("x", Value.num 3)], [], Flow.value Value.unit)) = true := by
  native_decide

example :
    (execBlockWithFns
      40
      [("x", Value.num 0)]
      []
      [ Stmt.while
          (Expr.binary (Expr.var "x") BinOp.lt (Expr.num 3))
          [ Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))
          , Stmt.ifThenElse (Expr.binary (Expr.var "x") BinOp.eq (Expr.num 1)) [Stmt.break] none
          , Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 10))]]
      ==
      some ([("x", Value.num 1)], [], Flow.value Value.unit)) = true := by
  native_decide

example :
    (execBlockWithFns
      60
      [("x", Value.num 0), ("sum", Value.num 0)]
      []
      [ Stmt.while
          (Expr.binary (Expr.var "x") BinOp.lt (Expr.num 4))
          [ Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))
          , Stmt.ifThenElse (Expr.binary (Expr.var "x") BinOp.eq (Expr.num 2)) [Stmt.continue] none
          , Stmt.assign "sum" (Expr.binary (Expr.var "sum") BinOp.add (Expr.var "x"))]]
      ==
      some ([("x", Value.num 4), ("sum", Value.num 8)], [], Flow.value Value.unit)) = true := by
  native_decide

example :
    (match execBlockWithFns
      40
      [("sum", Value.num 0)]
      []
      [ Stmt.forRange
          "i"
          0
          3
          [Stmt.assign "sum" (Expr.binary (Expr.var "sum") BinOp.add (Expr.var "i"))]]
    with
    | some (env, _, Flow.value Value.unit) =>
        Env.lookup env "sum" == some (Value.num 3)
        && Env.lookup env "i" == some (Value.num 3)
    | _ => false) = true := by
  native_decide

example :
    (match execBlockWithFns
      20
      []
      []
      [Stmt.forRange "i" 2 2 [Stmt.expr (Expr.num 9)]]
    with
    | some (env, _, Flow.value Value.unit) =>
        Env.lookup env "i" == some (Value.num 2)
    | _ => false) = true := by
  native_decide

example :
    (match execBlockWithFns
      40
      [("sum", Value.num 0)]
      []
      [ Stmt.forRange
          "i"
          0
          5
          [ Stmt.ifThenElse (Expr.binary (Expr.var "i") BinOp.eq (Expr.num 2)) [Stmt.break] none
          , Stmt.assign "sum" (Expr.binary (Expr.var "sum") BinOp.add (Expr.var "i"))]]
    with
    | some (env, _, Flow.value Value.unit) =>
        Env.lookup env "sum" == some (Value.num 1)
        && Env.lookup env "i" == some (Value.num 2)
    | _ => false) = true := by
  native_decide

example :
    (match execBlockWithFns
      60
      [("sum", Value.num 0)]
      []
      [ Stmt.forRange
          "i"
          0
          4
          [ Stmt.ifThenElse (Expr.binary (Expr.var "i") BinOp.eq (Expr.num 1)) [Stmt.continue] none
          , Stmt.assign "sum" (Expr.binary (Expr.var "sum") BinOp.add (Expr.var "i"))]]
    with
    | some (env, _, Flow.value Value.unit) =>
        Env.lookup env "sum" == some (Value.num 5)
        && Env.lookup env "i" == some (Value.num 4)
    | _ => false) = true := by
  native_decide

example :
    (match execBlockWithFns
      20
      [("x", Value.num 0)]
      []
      [Stmt.seal (some (Expr.bool true)) [Stmt.assign "x" (Expr.num 1)]]
    with
    | some (env, _, Flow.value Value.unit) =>
        Env.lookup env "x" == some (Value.num 0)
    | _ => false) = true := by
  native_decide

example :
    (match execBlockWithFns
      60
      [("x", Value.num 0)]
      []
      [ Stmt.seal
          (some (Expr.binary (Expr.var "x") BinOp.ge (Expr.num 3)))
          [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))]]
    with
    | some (env, _, Flow.value Value.unit) =>
        Env.lookup env "x" == some (Value.num 3)
    | _ => false) = true := by
  native_decide

example :
    (match execBlockWithFns
      40
      [("x", Value.num 0)]
      []
      [ Stmt.seal
          none
          [ Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))
          , Stmt.ifThenElse (Expr.binary (Expr.var "x") BinOp.eq (Expr.num 2)) [Stmt.break] none]]
    with
    | some (env, _, Flow.value Value.unit) =>
        Env.lookup env "x" == some (Value.num 2)
    | _ => false) = true := by
  native_decide

example :
    (match execBlockWithFns
      80
      [("x", Value.num 0), ("sum", Value.num 0)]
      []
      [ Stmt.seal
          (some (Expr.binary (Expr.var "x") BinOp.ge (Expr.num 4)))
          [ Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))
          , Stmt.ifThenElse (Expr.binary (Expr.var "x") BinOp.eq (Expr.num 2)) [Stmt.continue] none
          , Stmt.assign "sum" (Expr.binary (Expr.var "sum") BinOp.add (Expr.var "x"))]]
    with
    | some (env, _, Flow.value Value.unit) =>
        Env.lookup env "x" == some (Value.num 4)
        && Env.lookup env "sum" == some (Value.num 8)
    | _ => false) = true := by
  native_decide

example :
    StepBlock []
      [Stmt.letDecl "x" (Expr.num 1), Stmt.expr (Expr.var "x")]
      [("x", Value.num 1)]
      (Flow.value (Value.num 1)) := by
  apply StepBlock.consValue
  · apply StepStmt.letDecl
    rfl
  · apply StepBlock.single
    apply StepStmt.expr
    rfl

example :
    StepBlock
      [("x", Value.num 1)]
      [Stmt.assign "x" (Expr.num 2), Stmt.expr (Expr.var "x")]
      [("x", Value.num 2)]
      (Flow.value (Value.num 2)) := by
  apply StepBlock.consValue
  · apply StepStmt.assign
    · rfl
    · rfl
  · apply StepBlock.single
    apply StepStmt.expr
    rfl

example :
    StepBlock []
      [Stmt.ret (some (Expr.num 7)), Stmt.expr (Expr.num 9)]
      []
      (Flow.return (Value.num 7)) := by
  apply StepBlock.consReturn
  apply StepStmt.retSome
  rfl

example :
    StepBlock []
      [Stmt.break, Stmt.expr (Expr.num 9)]
      []
      Flow.break := by
  apply StepBlock.consBreak
  apply StepStmt.break

example :
    StepBlock []
      [Stmt.continue, Stmt.expr (Expr.num 9)]
      []
      Flow.continue := by
  apply StepBlock.consContinue
  apply StepStmt.continue

example :
    StepBlock []
      [Stmt.ifThenElse
        (Expr.bool true)
        [Stmt.letDecl "x" (Expr.num 1)]
        (some [Stmt.letDecl "x" (Expr.num 2)])]
      [("x", Value.num 1)]
      (Flow.value (Value.num 1)) := by
  apply StepBlock.single
  apply StepStmt.ifTrue
  · rfl
  · rfl
  · apply StepBlock.single
    apply StepStmt.letDecl
    rfl

example :
    StepBlock []
      [Stmt.ifThenElse
        (Expr.bool false)
        [Stmt.letDecl "x" (Expr.num 1)]
        (some [Stmt.letDecl "x" (Expr.num 2)])]
      [("x", Value.num 2)]
      (Flow.value (Value.num 2)) := by
  apply StepBlock.single
  apply StepStmt.ifFalseSome
  · rfl
  · rfl
  · apply StepBlock.single
    apply StepStmt.letDecl
    rfl

example :
    StepBlock []
      [Stmt.ifThenElse
        (Expr.bool false)
        [Stmt.letDecl "x" (Expr.num 1)]
        none]
      []
      (Flow.value Value.unit) := by
  apply StepBlock.single
  apply StepStmt.ifFalseNone
  · rfl
  · rfl

example :
    StepBlock
      [("x", Value.num 0)]
      [Stmt.while (Expr.bool false) [Stmt.assign "x" (Expr.num 1)]]
      [("x", Value.num 0)]
      (Flow.value Value.unit) := by
  apply StepBlock.single
  apply StepStmt.whileFalse
  · rfl
  · rfl

example :
    StepBlock
      [("x", Value.num 0)]
      [ Stmt.while
          (Expr.binary (Expr.var "x") BinOp.lt (Expr.num 1))
          [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))]]
      [("x", Value.num 1)]
      (Flow.value Value.unit) := by
  apply StepBlock.single
  apply StepStmt.whileValue
  · rfl
  · rfl
  · apply StepBlock.single
    apply StepStmt.assign
    · rfl
    · rfl
  · apply StepStmt.whileFalse
    · rfl
    · rfl

example :
    StepBlock
      [("x", Value.num 0)]
      [ Stmt.while
          (Expr.bool true)
          [ Stmt.assign "x" (Expr.num 1)
          , Stmt.break]]
      [("x", Value.num 1)]
      (Flow.value Value.unit) := by
  apply StepBlock.single
  apply StepStmt.whileBreak
  · rfl
  · rfl
  · apply StepBlock.consValue
    · apply StepStmt.assign
      · rfl
      · rfl
    · apply StepBlock.single
      apply StepStmt.break

example :
    StepBlock
      []
      [Stmt.forRange "i" 2 2 [Stmt.expr (Expr.num 9)]]
      [("i", Value.num 2)]
      (Flow.value Value.unit) := by
  apply StepBlock.single
  apply StepStmt.forDone
  decide

example :
    StepBlock
      [("x", Value.num 0)]
      [Stmt.forRange "i" 0 1 [Stmt.assign "x" (Expr.var "i")]]
      [("i", Value.num 1), ("i", Value.num 0), ("x", Value.num 0)]
      (Flow.value Value.unit) := by
  apply StepBlock.single
  apply StepStmt.forValue
  · decide
  · apply StepBlock.single
    apply StepStmt.assign
    · rfl
    · rfl
  · apply StepStmt.forDone
    decide

example :
    StepBlock
      []
      [Stmt.forRange "i" 0 3 [Stmt.break]]
      [("i", Value.num 0)]
      (Flow.value Value.unit) := by
  apply StepBlock.single
  apply StepStmt.forBreak
  · decide
  · apply StepBlock.single
    apply StepStmt.break

example :
    StepBlock
      [("x", Value.num 0)]
      [Stmt.seal (some (Expr.bool true)) [Stmt.assign "x" (Expr.num 1)]]
      [("x", Value.num 0)]
      (Flow.value Value.unit) := by
  apply StepBlock.single
  apply StepStmt.sealUntilDone
  · rfl
  · rfl

example :
    StepBlock
      [("x", Value.num 0)]
      [ Stmt.seal
          (some (Expr.binary (Expr.var "x") BinOp.ge (Expr.num 1)))
          [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))]]
      [("x", Value.num 1)]
      (Flow.value Value.unit) := by
  apply StepBlock.single
  apply StepStmt.sealUntilValue
  · rfl
  · rfl
  · apply StepBlock.single
    apply StepStmt.assign
    · rfl
    · rfl
  · apply StepStmt.sealUntilDone
    · rfl
    · rfl

example :
    StepBlock
      [("x", Value.num 0)]
      [ Stmt.seal
          none
          [ Stmt.assign "x" (Expr.num 1)
          , Stmt.break]]
      [("x", Value.num 1)]
      (Flow.value Value.unit) := by
  apply StepBlock.single
  apply StepStmt.sealBreak
  apply StepBlock.consValue
  · apply StepStmt.assign
    · rfl
    · rfl
  · apply StepBlock.single
    apply StepStmt.break

example :
    StepBlock []
      [Stmt.fnDecl "id" ["x"] [Stmt.ret (some (Expr.var "x"))]]
      []
      (Flow.value Value.unit) := by
  apply StepBlock.single
  apply StepStmt.fnDecl

example :
    StepBlockWithFns []
      []
      [ Stmt.fnDecl "id" ["x"] [Stmt.ret (some (Expr.var "x"))]
      , Stmt.letDecl "y" (Expr.call "id" [Expr.num 3])]
      [("y", Value.num 3)]
      [("id", { params := ["x"], body := [Stmt.ret (some (Expr.var "x"))] })]
      (Flow.value (Value.num 3)) := by
  apply StepBlockWithFns.consValue
  · apply StepStmtWithFns.fnDecl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.letDecl
    apply EvalExprWithFnsRel.callReturn
    · rfl
    · apply EvalArgsWithFnsRel.cons
      · apply EvalArgWithFnsRel.positional
        apply EvalExprWithFnsRel.num
      · apply EvalArgsWithFnsRel.nil
    · rfl
    · apply StepBlockWithFns.single
      apply StepStmtWithFns.retSome
      apply EvalExprWithFnsRel.var
      rfl

example :
    StepBlockWithFns []
      []
      [ Stmt.fnDecl "one" [] [Stmt.expr (Expr.num 1)]
      , Stmt.letDecl "y" (Expr.call "one" [])]
      [("y", Value.num 1)]
      [("one", { params := [], body := [Stmt.expr (Expr.num 1)] })]
      (Flow.value (Value.num 1)) := by
  apply StepBlockWithFns.consValue
  · apply StepStmtWithFns.fnDecl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.letDecl
    apply EvalExprWithFnsRel.callValue
    · rfl
    · apply EvalArgsWithFnsRel.nil
    · rfl
    · apply StepBlockWithFns.single
      apply StepStmtWithFns.expr
      apply EvalExprWithFnsRel.num

example :
    StepBlockWithFns []
      []
      [Stmt.ifThenElse
        (Expr.bool true)
        [Stmt.letDecl "x" (Expr.num 1)]
        (some [Stmt.letDecl "x" (Expr.num 2)])]
      [("x", Value.num 1)]
      []
      (Flow.value (Value.num 1)) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.ifTrue
  · apply EvalExprWithFnsRel.bool
  · rfl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.letDecl
    apply EvalExprWithFnsRel.num

example :
    StepBlockWithFns []
      []
      [Stmt.ifThenElse
        (Expr.bool false)
        [Stmt.letDecl "x" (Expr.num 1)]
        (some [Stmt.letDecl "x" (Expr.num 2)])]
      [("x", Value.num 2)]
      []
      (Flow.value (Value.num 2)) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.ifFalseSome
  · apply EvalExprWithFnsRel.bool
  · rfl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.letDecl
    apply EvalExprWithFnsRel.num

example :
    StepBlockWithFns []
      []
      [Stmt.ifThenElse
        (Expr.bool false)
        [Stmt.letDecl "x" (Expr.num 1)]
        none]
      []
      []
      (Flow.value Value.unit) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.ifFalseNone
  · apply EvalExprWithFnsRel.bool
  · rfl

example :
    StepBlockWithFns
      [("x", Value.num 0)]
      []
      [Stmt.while (Expr.bool false) [Stmt.letDecl "x" (Expr.num 1)]]
      [("x", Value.num 0)]
      []
      (Flow.value Value.unit) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.whileFalse
  · apply EvalExprWithFnsRel.bool
  · rfl

example :
    StepBlockWithFns
      [("go", Value.bool true)]
      []
      [Stmt.while (Expr.var "go") [Stmt.letDecl "go" (Expr.bool false)]]
      [("go", Value.bool false), ("go", Value.bool true)]
      []
      (Flow.value Value.unit) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.whileValue
  · apply EvalExprWithFnsRel.var
    rfl
  · rfl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.letDecl
    apply EvalExprWithFnsRel.bool
  · apply StepStmtWithFns.whileFalse
    · apply EvalExprWithFnsRel.var
      rfl
    · rfl

example :
    StepBlockWithFns
      [("x", Value.num 0)]
      []
      [ Stmt.while
          (Expr.bool true)
          [ Stmt.letDecl "x" (Expr.num 1)
          , Stmt.break]]
      [("x", Value.num 1), ("x", Value.num 0)]
      []
      (Flow.value Value.unit) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.whileBreak
  · apply EvalExprWithFnsRel.bool
  · rfl
  · apply StepBlockWithFns.consValue
    · apply StepStmtWithFns.letDecl
      apply EvalExprWithFnsRel.num
    · apply StepBlockWithFns.single
      apply StepStmtWithFns.break

example :
    StepBlockWithFns
      [("x", Value.num 1)]
      []
      [Stmt.assign "x" (Expr.num 2), Stmt.expr (Expr.var "x")]
      [("x", Value.num 2)]
      []
      (Flow.value (Value.num 2)) := by
  apply StepBlockWithFns.consValue
  · apply StepStmtWithFns.assign
    · apply EvalExprWithFnsRel.num
    · rfl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.expr
    apply EvalExprWithFnsRel.var
    rfl

example :
    StepBlockWithFns
      [("go", Value.bool true)]
      []
      [Stmt.while (Expr.var "go") [Stmt.assign "go" (Expr.bool false)]]
      [("go", Value.bool false)]
      []
      (Flow.value Value.unit) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.whileValue
  · apply EvalExprWithFnsRel.var
    rfl
  · rfl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.assign
    · apply EvalExprWithFnsRel.bool
    · rfl
  · apply StepStmtWithFns.whileFalse
    · apply EvalExprWithFnsRel.var
      rfl
    · rfl

example :
    StepBlockWithFns
      []
      []
      [Stmt.forRange "i" 2 2 [Stmt.expr (Expr.num 9)]]
      [("i", Value.num 2)]
      []
      (Flow.value Value.unit) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.forDone
  decide

example :
    StepBlockWithFns
      [("x", Value.num 0)]
      []
      [Stmt.forRange "i" 0 1 [Stmt.assign "x" (Expr.var "i")]]
      [("i", Value.num 1), ("i", Value.num 0), ("x", Value.num 0)]
      []
      (Flow.value Value.unit) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.forValue
  · decide
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.assign
    · apply EvalExprWithFnsRel.var
      rfl
    · rfl
  · apply StepStmtWithFns.forDone
    decide

example :
    StepBlockWithFns
      []
      []
      [Stmt.forRange "i" 0 3 [Stmt.break]]
      [("i", Value.num 0)]
      []
      (Flow.value Value.unit) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.forBreak
  · decide
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.break

example :
    StepBlockWithFns
      [("x", Value.num 0)]
      []
      [Stmt.seal (some (Expr.bool true)) [Stmt.assign "x" (Expr.num 1)]]
      [("x", Value.num 0)]
      []
      (Flow.value Value.unit) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.sealUntilDone
  · apply EvalExprWithFnsRel.bool
  · rfl

example :
    StepBlockWithFns
      [("x", Value.num 0)]
      []
      [ Stmt.seal
          (some (Expr.var "x"))
          [Stmt.assign "x" (Expr.num 1)]]
      [("x", Value.num 1)]
      []
      (Flow.value Value.unit) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.sealUntilValue
  · apply EvalExprWithFnsRel.var
    rfl
  · rfl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.assign
    · apply EvalExprWithFnsRel.num
    · rfl
  · apply StepStmtWithFns.sealUntilDone
    · apply EvalExprWithFnsRel.var
      rfl
    · rfl

example :
    StepBlockWithFns
      [("x", Value.num 0)]
      []
      [ Stmt.seal
          none
          [ Stmt.assign "x" (Expr.num 1)
          , Stmt.break]]
      [("x", Value.num 1)]
      []
      (Flow.value Value.unit) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.sealBreak
  apply StepBlockWithFns.consValue
  · apply StepStmtWithFns.assign
    · apply EvalExprWithFnsRel.num
    · rfl
  · apply StepBlockWithFns.single
    apply StepStmtWithFns.break

example :
    StepBlockWithFns
      []
      []
      [Stmt.expr (Expr.binary (Expr.num 2) BinOp.add (Expr.num 3))]
      []
      []
      (Flow.value (Value.num 5)) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.expr
  apply EvalExprWithFnsRel.binary
  · apply EvalExprWithFnsRel.num
  · apply EvalExprWithFnsRel.num
  · rfl

example :
    StepBlockWithFns
      []
      []
      [Stmt.expr (Expr.binary (Expr.num 2) BinOp.lt (Expr.num 3))]
      []
      []
      (Flow.value (Value.bool true)) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.expr
  apply EvalExprWithFnsRel.binary
  · apply EvalExprWithFnsRel.num
  · apply EvalExprWithFnsRel.num
  · rfl

example :
    StepBlockWithFns
      []
      []
      [Stmt.expr (Expr.unary UnOp.not (Expr.bool false))]
      []
      []
      (Flow.value (Value.bool true)) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.expr
  apply EvalExprWithFnsRel.unary
  · apply EvalExprWithFnsRel.bool
  · rfl

example :
    StepBlockWithFns
      []
      []
      [Stmt.expr (Expr.list [Expr.num 1, Expr.bool true])]
      []
      []
      (Flow.value (Value.list [Value.num 1, Value.bool true])) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.expr
  apply EvalExprWithFnsRel.list
  apply EvalExprsWithFnsRel.cons
  · apply EvalExprWithFnsRel.num
  · apply EvalExprsWithFnsRel.cons
    · apply EvalExprWithFnsRel.bool
    · apply EvalExprsWithFnsRel.nil

example :
    StepBlockWithFns
      []
      []
      [Stmt.expr (Expr.index (Expr.list [Expr.str "a", Expr.str "b"]) (Expr.num 1))]
      []
      []
      (Flow.value (Value.str "b")) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.expr
  apply EvalExprWithFnsRel.index
  · apply EvalExprWithFnsRel.list
    apply EvalExprsWithFnsRel.cons
    · apply EvalExprWithFnsRel.str
    · apply EvalExprsWithFnsRel.cons
      · apply EvalExprWithFnsRel.str
      · apply EvalExprsWithFnsRel.nil
  · apply EvalExprWithFnsRel.num
  · rfl

example :
    StepBlockWithFns
      []
      []
      [Stmt.expr (Expr.field (Expr.list [Expr.num 1, Expr.num 2]) "length")]
      []
      []
      (Flow.value (Value.num 2)) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.expr
  apply EvalExprWithFnsRel.field
  · apply EvalExprWithFnsRel.list
    apply EvalExprsWithFnsRel.cons
    · apply EvalExprWithFnsRel.num
    · apply EvalExprsWithFnsRel.cons
      · apply EvalExprWithFnsRel.num
      · apply EvalExprsWithFnsRel.nil
  · rfl

example :
    StepBlockWithFns
      []
      []
      [Stmt.expr (Expr.method (Expr.str "open") "len" [])]
      []
      []
      (Flow.value (Value.num 4)) := by
  apply StepBlockWithFns.single
  apply StepStmtWithFns.expr
  apply EvalExprWithFnsRel.method
  · apply EvalExprWithFnsRel.str
  · apply EvalArgsWithFnsRel.nil
  · rfl

end Aether
