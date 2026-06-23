import Aether.Lexer
import Aether.Parser
import Aether.Static
import Aether.VM

namespace Aether
namespace Pipeline

inductive Error where
  | lex : String -> Lexer.SourceSpan -> Error
  | parse : Parser.ParseError -> Lexer.SourceSpan -> Error
  | static : Static.CheckError -> Option Lexer.SourceSpan -> Error
  | compile : Error
  | runtime : Error
  deriving Repr, BEq, DecidableEq

def firstLexError : List Lexer.LocatedToken -> Option (String × Lexer.SourceSpan)
  | [] => none
  | { kind := Lexer.TokenKind.error message, start := start, stop := stop } :: _ =>
      some (message, { start := start, stop := stop })
  | _ :: rest => firstLexError rest

def skipLocatedTerminators : List Lexer.LocatedToken -> List Lexer.LocatedToken
  | { kind := Lexer.TokenKind.newline, start := _, stop := _ } :: rest => skipLocatedTerminators rest
  | { kind := Lexer.TokenKind.tilde, start := _, stop := _ } :: rest => skipLocatedTerminators rest
  | { kind := Lexer.TokenKind.semicolon, start := _, stop := _ } :: rest => skipLocatedTerminators rest
  | tokens => tokens

def tokenKinds (tokens : List Lexer.LocatedToken) : Parser.Tokens :=
  tokens.map (fun token => token.kind)

def tokenMatchesIdent (name : Ident) : Lexer.TokenKind -> Bool
  | Lexer.TokenKind.identifier found => found == name
  | Lexer.TokenKind.self => name == "self"
  | Lexer.TokenKind.embed => name == "embed"
  | Lexer.TokenKind.convergence => name == "convergence"
  | _ => false

def tokenMatchesBinOp (op : BinOp) (token : Lexer.TokenKind) : Bool :=
  match op, token with
  | BinOp.add, Lexer.TokenKind.plus => true
  | BinOp.sub, Lexer.TokenKind.minus => true
  | BinOp.mul, Lexer.TokenKind.star => true
  | BinOp.div, Lexer.TokenKind.slash => true
  | BinOp.mod, Lexer.TokenKind.percent => true
  | BinOp.eq, Lexer.TokenKind.eqEq => true
  | BinOp.neq, Lexer.TokenKind.notEq => true
  | BinOp.lt, Lexer.TokenKind.less => true
  | BinOp.gt, Lexer.TokenKind.greater => true
  | BinOp.le, Lexer.TokenKind.lessEq => true
  | BinOp.ge, Lexer.TokenKind.greaterEq => true
  | BinOp.and, Lexer.TokenKind.and => true
  | BinOp.or, Lexer.TokenKind.or => true
  | _, _ => false

def tokenMatchesUnOp (op : UnOp) (token : Lexer.TokenKind) : Bool :=
  match op, token with
  | UnOp.neg, Lexer.TokenKind.minus => true
  | UnOp.not, Lexer.TokenKind.not => true
  | _, _ => false

def firstTokenSpanWhere
    (p : Lexer.TokenKind -> Bool) : List Lexer.LocatedToken -> Option Lexer.SourceSpan
  | [] => none
  | token :: rest =>
      if p token.kind then
        some token.span
      else
        firstTokenSpanWhere p rest

def lastTokenSpanWhere
    (p : Lexer.TokenKind -> Bool) : List Lexer.LocatedToken -> Option Lexer.SourceSpan
  | [] => none
  | token :: rest =>
      match lastTokenSpanWhere p rest with
      | some span => some span
      | none =>
          if p token.kind then
            some token.span
          else
            none

def secondTokenSpanWhere
    (p : Lexer.TokenKind -> Bool) : List Lexer.LocatedToken -> Option Lexer.SourceSpan
  | [] => none
  | token :: rest =>
      if p token.kind then
        firstTokenSpanWhere p rest
      else
        secondTokenSpanWhere p rest

def fnNameSpanWhere
    (name : Ident) : List Lexer.LocatedToken -> Option Lexer.SourceSpan
  | [] => none
  | { kind := Lexer.TokenKind.fn, start := _, stop := _ }
      :: { kind := Lexer.TokenKind.identifier found, start := start, stop := stop }
      :: rest =>
        if found == name then
          some { start := start, stop := stop }
        else
          fnNameSpanWhere name rest
  | _ :: rest => fnNameSpanWhere name rest

def secondFnNameSpanWhere
    (name : Ident) : List Lexer.LocatedToken -> Option Lexer.SourceSpan
  | [] => none
  | { kind := Lexer.TokenKind.fn, start := _, stop := _ }
      :: { kind := Lexer.TokenKind.identifier found, start := start, stop := stop }
      :: rest =>
        if found == name then
          match fnNameSpanWhere name rest with
          | some span => some span
          | none => some { start := start, stop := stop }
        else
          secondFnNameSpanWhere name rest
  | _ :: rest => secondFnNameSpanWhere name rest

partial def closingBraceSpanFromDepth (depth : Nat) : List Lexer.LocatedToken -> Option Lexer.SourceSpan
  | [] => none
  | { kind := Lexer.TokenKind.lBrace, start := _, stop := _ } :: rest =>
      closingBraceSpanFromDepth (depth + 1) rest
  | { kind := Lexer.TokenKind.rBrace, start := start, stop := stop } :: rest =>
      if depth == 1 then
        some { start := start, stop := stop }
      else
        closingBraceSpanFromDepth (depth - 1) rest
  | _ :: rest => closingBraceSpanFromDepth depth rest

def firstFunctionBodyCloseSpan : List Lexer.LocatedToken -> Option Lexer.SourceSpan
  | [] => none
  | { kind := Lexer.TokenKind.lBrace, start := _, stop := _ } :: rest =>
      closingBraceSpanFromDepth 1 rest
  | _ :: rest => firstFunctionBodyCloseSpan rest

def fnBodyCloseSpanWhere
    (name : Ident) : List Lexer.LocatedToken -> Option Lexer.SourceSpan
  | [] => none
  | { kind := Lexer.TokenKind.fn, start := _, stop := _ }
      :: { kind := Lexer.TokenKind.identifier found, start := _, stop := _ }
      :: rest =>
        if found == name then
          match firstFunctionBodyCloseSpan rest with
          | some span => some span
          | none => fnBodyCloseSpanWhere name rest
        else
          fnBodyCloseSpanWhere name rest
  | _ :: rest => fnBodyCloseSpanWhere name rest

def duplicateParamSpanInList?
    (name : Ident) (seen : Bool) : List Lexer.LocatedToken -> Option Lexer.SourceSpan
  | [] => none
  | { kind := Lexer.TokenKind.rParen, start := _, stop := _ } :: _ => none
  | { kind := Lexer.TokenKind.identifier found, start := start, stop := stop } :: rest =>
      if found == name then
        if seen then
          some { start := start, stop := stop }
        else
          duplicateParamSpanInList? name true rest
      else
        duplicateParamSpanInList? name seen rest
  | _ :: rest => duplicateParamSpanInList? name seen rest

def duplicateParamSpanWhere
    (name : Ident) : List Lexer.LocatedToken -> Option Lexer.SourceSpan
  | [] => none
  | { kind := Lexer.TokenKind.fn, start := _, stop := _ }
      :: { kind := Lexer.TokenKind.identifier _, start := _, stop := _ }
      :: { kind := Lexer.TokenKind.lParen, start := _, stop := _ }
      :: rest =>
        match duplicateParamSpanInList? name false rest with
        | some span => some span
        | none => duplicateParamSpanWhere name rest
  | _ :: rest => duplicateParamSpanWhere name rest

def tokenMatchesTy : Static.Ty -> Lexer.TokenKind -> Bool
  | Static.Ty.num, Lexer.TokenKind.number _ => true
  | Static.Ty.num, Lexer.TokenKind.float _ _ => true
  | Static.Ty.bool, Lexer.TokenKind.true => true
  | Static.Ty.bool, Lexer.TokenKind.false => true
  | Static.Ty.str, Lexer.TokenKind.stringLit _ => true
  | Static.Ty.unit, Lexer.TokenKind.identifier "unit" => true
  | Static.Ty.list _, Lexer.TokenKind.lBracket => true
  | _, _ => false

def conditionTokenSpanWhere
    (actual : Static.Ty) : List Lexer.LocatedToken -> Option Lexer.SourceSpan
  | [] => none
  | { kind := Lexer.TokenKind.if_, start := _, stop := _ } :: token :: rest =>
      if tokenMatchesTy actual token.kind then
        some token.span
      else
        conditionTokenSpanWhere actual rest
  | { kind := Lexer.TokenKind.while, start := _, stop := _ } :: token :: rest =>
      if tokenMatchesTy actual token.kind then
        some token.span
      else
        conditionTokenSpanWhere actual rest
  | { kind := Lexer.TokenKind.seal, start := _, stop := _ }
      :: { kind := Lexer.TokenKind.until, start := _, stop := _ }
      :: token
      :: rest =>
        if tokenMatchesTy actual token.kind then
          some token.span
        else
          conditionTokenSpanWhere actual rest
  | _ :: rest => conditionTokenSpanWhere actual rest

def memberNameSpanWhere
    (name : Ident) : List Lexer.LocatedToken -> Option Lexer.SourceSpan
  | [] => none
  | { kind := Lexer.TokenKind.dot, start := _, stop := _ }
      :: { kind := token, start := start, stop := stop }
      :: rest =>
        if tokenMatchesIdent name token then
          some { start := start, stop := stop }
        else
          memberNameSpanWhere name rest
  | _ :: rest => memberNameSpanWhere name rest

def staticErrorSpan? (tokens : List Lexer.LocatedToken) : Static.CheckError -> Option Lexer.SourceSpan
  | Static.CheckError.undeclaredVariable name =>
      firstTokenSpanWhere (tokenMatchesIdent name) tokens
  | Static.CheckError.undeclaredFunction name =>
      firstTokenSpanWhere (tokenMatchesIdent name) tokens
  | Static.CheckError.operandMismatch op _ _ =>
      firstTokenSpanWhere (tokenMatchesBinOp op) tokens
  | Static.CheckError.unaryMismatch op _ =>
      firstTokenSpanWhere (tokenMatchesUnOp op) tokens
  | Static.CheckError.indexMismatch _ _ =>
      lastTokenSpanWhere (fun token => token == Lexer.TokenKind.lBracket) tokens
  | Static.CheckError.fieldMismatch _ field =>
      match memberNameSpanWhere field tokens with
      | some span => some span
      | none => lastTokenSpanWhere (fun token => token == Lexer.TokenKind.dot) tokens
  | Static.CheckError.methodMismatch _ method _ =>
      match memberNameSpanWhere method tokens with
      | some span => some span
      | none => lastTokenSpanWhere (fun token => token == Lexer.TokenKind.dot) tokens
  | Static.CheckError.conditionMismatch actual =>
      match conditionTokenSpanWhere actual tokens with
      | some span => some span
      | none =>
          firstTokenSpanWhere
            (fun token =>
              token == Lexer.TokenKind.if_
                || token == Lexer.TokenKind.while
                || token == Lexer.TokenKind.seal)
            tokens
  | Static.CheckError.assignmentMismatch name _ _ =>
      firstTokenSpanWhere (tokenMatchesIdent name) tokens
  | Static.CheckError.arityMismatch name _ _ =>
      lastTokenSpanWhere (tokenMatchesIdent name) tokens
  | Static.CheckError.functionSignatureMismatch name _ _ =>
      firstTokenSpanWhere (tokenMatchesIdent name) tokens
  | Static.CheckError.unknownNamedArgument _ name =>
      lastTokenSpanWhere (tokenMatchesIdent name) tokens
  | Static.CheckError.duplicateNamedArgument _ name =>
      lastTokenSpanWhere (tokenMatchesIdent name) tokens
  | Static.CheckError.argumentMismatch _ _ _ actual =>
      lastTokenSpanWhere (tokenMatchesTy actual) tokens
  | Static.CheckError.returnMismatch name _ actual =>
      match lastTokenSpanWhere (tokenMatchesTy actual) tokens with
      | some span => some span
      | none =>
          if actual == Static.Ty.unit then
            match fnBodyCloseSpanWhere name tokens with
            | some span => some span
            | none => fnNameSpanWhere name tokens
          else
            fnNameSpanWhere name tokens
  | Static.CheckError.duplicateFunction name =>
      secondFnNameSpanWhere name tokens
  | Static.CheckError.duplicateParameter name =>
      duplicateParamSpanWhere name tokens
  | Static.CheckError.nestedFunction name =>
      fnNameSpanWhere name tokens
  | Static.CheckError.returnOutsideFunction =>
      firstTokenSpanWhere (fun token => token == Lexer.TokenKind.return) tokens
  | Static.CheckError.breakOutsideLoop =>
      firstTokenSpanWhere (fun token => token == Lexer.TokenKind.break) tokens
  | Static.CheckError.continueOutsideLoop =>
      firstTokenSpanWhere (fun token => token == Lexer.TokenKind.continue) tokens

partial def parseLocatedProgramDetailed :
    List Lexer.LocatedToken -> Except (Parser.ParseError × Lexer.SourceSpan) (List Stmt)
  | [] => Except.ok []
  | { kind := Lexer.TokenKind.eof, start := _, stop := _ } :: _ => Except.ok []
  | tokens =>
      let tokens := skipLocatedTerminators tokens
      match tokens with
      | [] => Except.ok []
      | { kind := Lexer.TokenKind.eof, start := _, stop := _ } :: _ => Except.ok []
      | first :: _ =>
          let kinds := tokenKinds tokens
          match Parser.parseStmt kinds with
          | some (stmt, remainingKinds) =>
              let consumed := kinds.length - remainingKinds.length
              match parseLocatedProgramDetailed (tokens.drop consumed) with
              | Except.ok rest => Except.ok (stmt :: rest)
              | Except.error err => Except.error err
          | none =>
              let offset := Parser.diagnosticOffset kinds
              let diagnosticToken := tokens.drop offset
              let span :=
                match diagnosticToken with
                | token :: _ => token.span
                | [] => first.span
              let err :=
                Parser.ParseError.expected
                  (Parser.classifyStmtStart kinds)
                  (Parser.tokenAt? kinds offset)
              Except.error (err, span)

def parseSource (source : String) : Except Error (List Stmt) :=
  let located := Lexer.tokenizeLocated source
  match firstLexError located with
  | some (message, span) => Except.error (Error.lex message span)
  | none =>
      match parseLocatedProgramDetailed located with
      | Except.ok stmts => Except.ok stmts
      | Except.error (err, span) => Except.error (Error.parse err span)

def checkSource (source : String) : Except Error Static.CheckState := do
  let located := Lexer.tokenizeLocated source
  let stmts <- parseSource source
  match Static.checkProgramDetailed stmts with
  | Except.ok state => Except.ok state
  | Except.error err => Except.error (Error.static err (staticErrorSpan? located err))

def compileSource (source : String) : Except Error (VM.SlotEnv × VM.FrameFnEnv × List VM.FrameOp) := do
  let located := Lexer.tokenizeLocated source
  let stmts <- parseSource source
  match Static.checkProgramDetailed stmts with
  | Except.error err => Except.error (Error.static err (staticErrorSpan? located err))
  | Except.ok _ =>
      match VM.compileFrameProgram stmts with
      | some compiled => Except.ok compiled
      | none => Except.error Error.compile

def runSource (fuel : Nat) (source : String) :
    Except Error (VM.SlotEnv × VM.FrameFnEnv × VM.FrameState) := do
  let (slots, fns, code) <- compileSource source
  match VM.runFrame fuel code with
  | some state => Except.ok (slots, fns, state)
  | none => Except.error Error.runtime

def sourceLocal? (fuel : Nat) (source : String) (slot : Nat) : Except Error Value := do
  let (_, _, state) <- runSource fuel source
  match VM.listGet? state.locals slot with
  | some value => Except.ok value
  | none => Except.error Error.runtime

def posString (pos : Lexer.SourcePos) : String :=
  toString pos.line ++ ":" ++ toString pos.column

def spanString (span : Lexer.SourceSpan) : String :=
  posString span.start ++ "-" ++ posString span.stop

def binOpString : BinOp -> String
  | BinOp.add => "+"
  | BinOp.sub => "-"
  | BinOp.mul => "*"
  | BinOp.div => "/"
  | BinOp.mod => "%"
  | BinOp.eq => "=="
  | BinOp.neq => "!="
  | BinOp.lt => "<"
  | BinOp.gt => ">"
  | BinOp.le => "<="
  | BinOp.ge => ">="
  | BinOp.and => "&&"
  | BinOp.or => "||"

def unOpString : UnOp -> String
  | UnOp.neg => "-"
  | UnOp.not => "!"

def tyString : Static.Ty -> String
  | Static.Ty.num => "num"
  | Static.Ty.bool => "bool"
  | Static.Ty.str => "str"
  | Static.Ty.list elem => "list[" ++ tyString elem ++ "]"
  | Static.Ty.unit => "unit"
  | Static.Ty.unknown => "unknown"

def tokenString : Lexer.TokenKind -> String
  | Lexer.TokenKind.identifier name => "identifier(" ++ name ++ ")"
  | Lexer.TokenKind.number n => "number(" ++ toString n ++ ")"
  | Lexer.TokenKind.float int frac => "float(" ++ toString int ++ "," ++ toString frac ++ ")"
  | Lexer.TokenKind.stringLit value => "string(" ++ value ++ ")"
  | Lexer.TokenKind.error message => "lexer-error(" ++ message ++ ")"
  | Lexer.TokenKind.manifold => "manifold"
  | Lexer.TokenKind.block => "block"
  | Lexer.TokenKind.regress => "regress"
  | Lexer.TokenKind.render => "render"
  | Lexer.TokenKind.embed => "embed"
  | Lexer.TokenKind.escalate => "escalate"
  | Lexer.TokenKind.convergence => "convergence"
  | Lexer.TokenKind.class => "class"
  | Lexer.TokenKind.new => "new"
  | Lexer.TokenKind.self => "self"
  | Lexer.TokenKind.import => "import"
  | Lexer.TokenKind.from => "from"
  | Lexer.TokenKind.as => "as"
  | Lexer.TokenKind.dim => "dim"
  | Lexer.TokenKind.tau => "tau"
  | Lexer.TokenKind.model => "model"
  | Lexer.TokenKind.color => "color"
  | Lexer.TokenKind.axis => "axis"
  | Lexer.TokenKind.project => "project"
  | Lexer.TokenKind.cluster => "cluster"
  | Lexer.TokenKind.center => "center"
  | Lexer.TokenKind.spread => "spread"
  | Lexer.TokenKind.format => "format"
  | Lexer.TokenKind.output => "output"
  | Lexer.TokenKind.let_ => "let"
  | Lexer.TokenKind.fn => "fn"
  | Lexer.TokenKind.for => "for"
  | Lexer.TokenKind.while => "while"
  | Lexer.TokenKind.if_ => "if"
  | Lexer.TokenKind.else_ => "else"
  | Lexer.TokenKind.return => "return"
  | Lexer.TokenKind.break => "break"
  | Lexer.TokenKind.continue => "continue"
  | Lexer.TokenKind.in_ => "in"
  | Lexer.TokenKind.seal => "seal"
  | Lexer.TokenKind.until => "until"
  | Lexer.TokenKind.true => "true"
  | Lexer.TokenKind.false => "false"
  | Lexer.TokenKind.equals => "="
  | Lexer.TokenKind.comma => ","
  | Lexer.TokenKind.dot => "."
  | Lexer.TokenKind.dotDot => ".."
  | Lexer.TokenKind.tilde => "~"
  | Lexer.TokenKind.semicolon => ";"
  | Lexer.TokenKind.newline => "newline"
  | Lexer.TokenKind.eof => "eof"
  | Lexer.TokenKind.lBrace => "{"
  | Lexer.TokenKind.rBrace => "}"
  | Lexer.TokenKind.lParen => "("
  | Lexer.TokenKind.rParen => ")"
  | Lexer.TokenKind.lBracket => "["
  | Lexer.TokenKind.rBracket => "]"
  | Lexer.TokenKind.plus => "+"
  | Lexer.TokenKind.minus => "-"
  | Lexer.TokenKind.star => "*"
  | Lexer.TokenKind.slash => "/"
  | Lexer.TokenKind.percent => "%"
  | Lexer.TokenKind.less => "<"
  | Lexer.TokenKind.greater => ">"
  | Lexer.TokenKind.lessEq => "<="
  | Lexer.TokenKind.greaterEq => ">="
  | Lexer.TokenKind.eqEq => "=="
  | Lexer.TokenKind.notEq => "!="
  | Lexer.TokenKind.and => "&&"
  | Lexer.TokenKind.or => "||"
  | Lexer.TokenKind.not => "!"
  | _ => "token"

def parseContextString : Parser.ParseContext -> String
  | Parser.ParseContext.expression => "expression"
  | Parser.ParseContext.statement => "statement"
  | Parser.ParseContext.ifStatement => "if-statement"
  | Parser.ParseContext.block => "block"
  | Parser.ParseContext.range => "range"
  | Parser.ParseContext.params => "params"
  | Parser.ParseContext.typeAnnotation => "type"
  | Parser.ParseContext.terminator => "terminator"
  | Parser.ParseContext.programEnd => "program-end"

def optionalTokenString : Option Lexer.TokenKind -> String
  | none => "end-of-input"
  | some token => tokenString token

def parseErrorString : Parser.ParseError -> String
  | Parser.ParseError.expected context found =>
      "expected " ++ parseContextString context ++ ", found " ++ optionalTokenString found

def staticErrorString : Static.CheckError -> String
  | Static.CheckError.undeclaredVariable name => "undeclared variable " ++ name
  | Static.CheckError.undeclaredFunction name => "undeclared function " ++ name
  | Static.CheckError.operandMismatch op left right =>
      "operator " ++ binOpString op ++ " cannot accept " ++ tyString left ++ " and " ++ tyString right
  | Static.CheckError.unaryMismatch op ty =>
      "operator " ++ unOpString op ++ " cannot accept " ++ tyString ty
  | Static.CheckError.indexMismatch target index =>
      "index expected list and num, found " ++ tyString target ++ " and " ++ tyString index
  | Static.CheckError.fieldMismatch target field =>
      "field " ++ field ++ " is not available on " ++ tyString target
  | Static.CheckError.methodMismatch target method arity =>
      "method " ++ method ++ "/" ++ toString arity ++ " is not available on " ++ tyString target
  | Static.CheckError.conditionMismatch ty =>
      "condition expected bool, found " ++ tyString ty
  | Static.CheckError.assignmentMismatch name expected actual =>
      "assignment to " ++ name ++ " expected " ++ tyString expected ++ ", found " ++ tyString actual
  | Static.CheckError.arityMismatch name expected actual =>
      "function " ++ name ++ " expected " ++ toString expected ++ " args, found " ++ toString actual
  | Static.CheckError.functionSignatureMismatch name expected actual =>
      "function " ++ name ++ " signature expected " ++ toString expected ++ " params, found " ++ toString actual
  | Static.CheckError.unknownNamedArgument fnName argName =>
      "function " ++ fnName ++ " has no parameter " ++ argName
  | Static.CheckError.duplicateNamedArgument fnName argName =>
      "duplicate named argument " ++ argName ++ " for function " ++ fnName
  | Static.CheckError.argumentMismatch fnName paramName expected actual =>
      "argument " ++ paramName ++ " for function " ++ fnName
        ++ " expected " ++ tyString expected ++ ", found " ++ tyString actual
  | Static.CheckError.returnMismatch fnName expected actual =>
      "function " ++ fnName ++ " expected return " ++ tyString expected
        ++ ", found " ++ tyString actual
  | Static.CheckError.duplicateFunction name => "duplicate function " ++ name
  | Static.CheckError.duplicateParameter name => "duplicate parameter " ++ name
  | Static.CheckError.nestedFunction name => "nested function " ++ name
  | Static.CheckError.returnOutsideFunction => "return outside function"
  | Static.CheckError.breakOutsideLoop => "break outside loop"
  | Static.CheckError.continueOutsideLoop => "continue outside loop"

def errorString : Error -> String
  | Error.lex message span => "lex error at " ++ spanString span ++ ": " ++ message
  | Error.parse err span => "parse error at " ++ spanString span ++ ": " ++ parseErrorString err
  | Error.static err none => "static error: " ++ staticErrorString err
  | Error.static err (some span) =>
      "static error at " ++ spanString span ++ ": " ++ staticErrorString err
  | Error.compile => "compile error"
  | Error.runtime => "runtime error"

def resultErrorString {α : Type} (result : Except Error α) : Option String :=
  match result with
  | Except.ok _ => none
  | Except.error err => some (errorString err)

def parseSourceErrorString (source : String) : Option String :=
  resultErrorString (parseSource source)

def checkSourceErrorString (source : String) : Option String :=
  resultErrorString (checkSource source)

def compileSourceErrorString (source : String) : Option String :=
  resultErrorString (compileSource source)

def runSourceErrorString (fuel : Nat) (source : String) : Option String :=
  resultErrorString (runSource fuel source)

def sourceLocalErrorString (fuel : Nat) (source : String) (slot : Nat) : Option String :=
  resultErrorString (sourceLocal? fuel source slot)

example :
    (sourceLocal? 40 "fn add(a, b) { return a + b }~let y = add(2, 3)" 0
      == Except.ok (Value.num 5)) := by
  native_decide

example :
    (sourceLocal? 40 "fn add(a, b) { return a + b }~let y = add\n(2, 3)" 0
      == Except.ok (Value.num 5)) := by
  native_decide

example :
    (sourceLocal? 40 "fn add\n(a, b) { return a + b }~let y = add(2, 3)" 0
      == Except.ok (Value.num 5)) := by
  native_decide

example :
    (sourceLocal? 40 "fn id(x)\n{ return x }~let y = id(3)" 0
      == Except.ok (Value.num 3)) := by
  native_decide

example :
    (sourceLocal? 40 "fn pick(a, b) { return a }~let y = pick(b=2, a=7)" 0
      == Except.ok (Value.num 7)) := by
  native_decide

example :
    (parseSource "let x = 0~🦭 until x == 3 { x = x + 1 }"
      == Except.ok
        [ Stmt.letDecl "x" (Expr.num 0)
        , Stmt.seal
            (some (Expr.binary (Expr.var "x") BinOp.eq (Expr.num 3)))
            [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))]
        ]) := by
  native_decide

example :
    (sourceLocal? 80 "let x = 0~🦭 until x == 3 { x = x + 1 }" 0
      == Except.ok (Value.num 3)) := by
  native_decide

example :
    (sourceLocal? 100 "let x = 0~if\ntrue { x = x + 1 }~while\nx < 2 { x = x + 1 }~seal until\nx == 3 { x = x + 1 }" 0
      == Except.ok (Value.num 3)) := by
  native_decide

example :
    (sourceLocal? 80 "let x = 0~for i in -2..2 { x = x + i }" 0
      == Except.ok (Value.num (-2))) := by
  native_decide

example :
    (parseSource "let label = \"open\""
      == Except.ok [Stmt.letDecl "label" (Expr.str "open")]) := by
  native_decide

example :
    (sourceLocal? 20 "let label = \"open\"" 0
      == Except.ok (Value.str "open")) := by
  native_decide

example :
    (sourceLocal? 20 "let label = \"line\\nnext\"" 0
      == Except.ok (Value.str "line\nnext")) := by
  native_decide

example :
    (sourceLocal? 20 "let label = \"row\\rnext\"" 0
      == Except.ok (Value.str "row\rnext")) := by
  native_decide

example :
    (sourceLocal? 20 "let x = 1 /* ignored\ncomment */~let y = x + 1" 1
      == Except.ok (Value.num 2)) := by
  native_decide

example :
    (sourceLocal? 20 "let x = 1; let y = x + 1" 1
      == Except.ok (Value.num 2)) := by
  native_decide

example :
    (sourceLocal? 20 "let xs = [1,\n2,]~let count = xs.len()" 1
      == Except.ok (Value.num 2)) := by
  native_decide

example :
    (parseSource "let ratio = 1.5"
      == Except.ok [Stmt.letDecl "ratio" (Expr.float 1 500000)]) := by
  native_decide

example :
    (parseSource "let done: unit = unit"
      == Except.ok [Stmt.letDeclTyped "done" AnnTy.unit Expr.unit]) := by
  native_decide

example :
    (parseSource "fn id(x): num { return x }"
      == Except.ok
        [ Stmt.fnDeclReturn
            "id"
            ["x"]
            AnnTy.num
            [Stmt.ret (some (Expr.var "x"))]
        ]) := by
  native_decide

example :
    (sourceLocal? 60 "fn add(a,\nb,) { return a + b }~let y = add(1, 2)" 0
      == Except.ok (Value.num 3)) := by
  native_decide

example :
    (sourceLocal? 30 "let ratio = 1.5 + 2" 0
      == Except.ok (Value.float 3 500000)) := by
  native_decide

example :
    (sourceLocal? 30 "let total = (\n1 + 2\n)" 0
      == Except.ok (Value.num 3)) := by
  native_decide

example :
    (sourceLocal? 30 "let total = 1 +\n2 * 3" 0
      == Except.ok (Value.num 7)) := by
  native_decide

example :
    (sourceLocal? 30 "let total =\n1 + 2~total =\ntotal * 3" 0
      == Except.ok (Value.num 9)) := by
  native_decide

example :
    (sourceLocal? 30 "let ok = !\nfalse~let n = -\n3" 1
      == Except.ok (Value.num (-3))) := by
  native_decide

example :
    (parseSource "let xs = [1, true, \"open\"]"
      == Except.ok [Stmt.letDecl "xs" (Expr.list [Expr.num 1, Expr.bool true, Expr.str "open"])]) := by
  native_decide

example :
    (sourceLocal? 30 "let xs = [1, true, \"open\"]" 0
      == Except.ok (Value.list [Value.num 1, Value.bool true, Value.str "open"])) := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [1, true, \"open\"]~let first = xs[0]" 1
      == Except.ok (Value.num 1)) := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [1, 2]~let first = xs[\n0\n]" 1
      == Except.ok (Value.num 1)) := by
  native_decide

example :
    (sourceLocal? 40 "let xs: list[\nnum\n] = [1, 2]~let n = xs.len()" 1
      == Except.ok (Value.num 2)) := by
  native_decide

example :
    (sourceLocal? 40 "let label = \"open\"~let second = label[1]" 1
      == Except.ok (Value.str "p")) := by
  native_decide

example :
    checkSourceErrorString "let bad = [1][true]"
      =
      some "static error at 1:14-1:15: index expected list and num, found list[num] and bool" := by
  native_decide

example :
    checkSourceErrorString "let bad = \"open\"[true]"
      =
      some "static error at 1:17-1:18: index expected list and num, found str and bool" := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [1, true, \"open\"]~let n = xs.length" 1
      == Except.ok (Value.num 3)) := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [1, true, \"open\"]~let n = xs.\nlength" 1
      == Except.ok (Value.num 3)) := by
  native_decide

example :
    checkSourceErrorString "let bad = 1.length"
      =
      some "static error at 1:13-1:19: field length is not available on num" := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [1, true, \"open\"]~let n = xs.len()" 1
      == Except.ok (Value.num 3)) := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [1, 2]~let n = xs.\nlen()" 1
      == Except.ok (Value.num 2)) := by
  native_decide

example :
    (sourceLocal? 40 "let xs = []~let empty = xs.is_empty()" 1
      == Except.ok (Value.bool true)) := by
  native_decide

example :
    (sourceLocal? 40 "let label = \"open\"~let empty = label.is_empty()" 1
      == Except.ok (Value.bool false)) := by
  native_decide

example :
    (sourceLocal? 40 "let label = \"open\"~let ch = label.at(1)" 1
      == Except.ok (Value.str "p")) := by
  native_decide

example :
    (sourceLocal? 40 "let label = \"open\"~let ok = label.starts_with(\n\"op\",\n)" 1
      == Except.ok (Value.bool true)) := by
  native_decide

example :
    (sourceLocal? 40 "let label = \"open\"~let has = label.contains(\"pe\")" 1
      == Except.ok (Value.bool true)) := by
  native_decide

example :
    (sourceLocal? 40 "let label = \"open\"~let has = label.starts_with(\"op\")" 1
      == Except.ok (Value.bool true)) := by
  native_decide

example :
    (sourceLocal? 40 "let label = \"open\"~let has = label.ends_with(\"en\")" 1
      == Except.ok (Value.bool true)) := by
  native_decide

example :
    (sourceLocal? 40 "let label = \"open\"~let reversed = label.reverse()" 1
      == Except.ok (Value.str "nepo")) := by
  native_decide

example :
    (sourceLocal? 40 "let label = \"open\"~let first = label.first()" 1
      == Except.ok (Value.str "o")) := by
  native_decide

example :
    (sourceLocal? 40 "let label = \"open\"~let last = label.last()" 1
      == Except.ok (Value.str "n")) := by
  native_decide

example :
    (sourceLocal? 40 "let label = \"open\"~let rest = label.tail()" 1
      == Except.ok (Value.str "pen")) := by
  native_decide

example :
    (sourceLocal? 40 "let label = \"open\"~let prefix = label.take(2)" 1
      == Except.ok (Value.str "op")) := by
  native_decide

example :
    (sourceLocal? 40 "let label = \"open\"~let suffix = label.drop(2)" 1
      == Except.ok (Value.str "en")) := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [7, 9]~let first = xs.first()" 1
      == Except.ok (Value.num 7)) := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [7, 9]~let last = xs.last()" 1
      == Except.ok (Value.num 9)) := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [7, 9]~let picked = xs.at(1)" 1
      == Except.ok (Value.num 9)) := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [7, 9]~let has = xs.contains(9)" 1
      == Except.ok (Value.bool true)) := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [7, 9]~let rest = xs.tail()" 1
      == Except.ok (Value.list [Value.num 9])) := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [7, 9]~let first = xs.take(1)" 1
      == Except.ok (Value.list [Value.num 7])) := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [7, 9]~let rest = xs.drop(1)" 1
      == Except.ok (Value.list [Value.num 9])) := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [7, 9]~let reversed = xs.reverse()" 1
      == Except.ok (Value.list [Value.num 9, Value.num 7])) := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [7]~let more = xs.append(9)" 1
      == Except.ok (Value.list [Value.num 7, Value.num 9])) := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [9]~let more = xs.prepend(7)" 1
      == Except.ok (Value.list [Value.num 7, Value.num 9])) := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [\"a\", \"b\"]~let text = xs.join(\",\")" 1
      == Except.ok (Value.str "a,b")) := by
  native_decide

example :
    (sourceLocal? 40 "let xs = [7]~let more = xs.concat([9])" 1
      == Except.ok (Value.list [Value.num 7, Value.num 9])) := by
  native_decide

example :
    checkSourceErrorString "let bad = 1.len()"
      =
      some "static error at 1:13-1:16: method len/0 is not available on num" := by
  native_decide

example :
    checkSourceErrorString "let bad = [1].at(true)"
      =
      some "static error at 1:15-1:17: method at/1 is not available on list[num]" := by
  native_decide

example :
    checkSourceErrorString "let bad = [1].reverse(0)"
      =
      some "static error at 1:15-1:22: method reverse/1 is not available on list[num]" := by
  native_decide

example :
    checkSourceErrorString "let bad = [1].append(true)"
      =
      some "static error at 1:15-1:21: method append/1 is not available on list[num]" := by
  native_decide

example :
    checkSourceErrorString "let bad = [1].prepend(true)"
      =
      some "static error at 1:15-1:22: method prepend/1 is not available on list[num]" := by
  native_decide

example :
    checkSourceErrorString "let bad = [1].join(\",\")"
      =
      some "static error at 1:15-1:19: method join/1 is not available on list[num]" := by
  native_decide

example :
    checkSourceErrorString "let bad = [1].concat([true])"
      =
      some "static error at 1:15-1:21: method concat/1 is not available on list[num]" := by
  native_decide

example :
    checkSourceErrorString "let bad = \"open\".at(true)"
      =
      some "static error at 1:18-1:20: method at/1 is not available on str" := by
  native_decide

example :
    checkSourceErrorString "let bad = [1].contains(true)"
      =
      some "static error at 1:15-1:23: method contains/1 is not available on list[num]" := by
  native_decide

example :
    checkSourceErrorString "let bad = \"open\".contains(1)"
      =
      some "static error at 1:18-1:26: method contains/1 is not available on str" := by
  native_decide

example :
    checkSourceErrorString "let bad = \"open\".starts_with(1)"
      =
      some "static error at 1:18-1:29: method starts_with/1 is not available on str" := by
  native_decide

example :
    checkSourceErrorString "let bad = \"open\".ends_with(1)"
      =
      some "static error at 1:18-1:27: method ends_with/1 is not available on str" := by
  native_decide

example :
    checkSourceErrorString "let bad = \"open\".reverse(1)"
      =
      some "static error at 1:18-1:25: method reverse/1 is not available on str" := by
  native_decide

example :
    checkSourceErrorString "let bad = \"open\".first(0)"
      =
      some "static error at 1:18-1:23: method first/1 is not available on str" := by
  native_decide

example :
    checkSourceErrorString "let bad = \"open\".last(0)"
      =
      some "static error at 1:18-1:22: method last/1 is not available on str" := by
  native_decide

example :
    checkSourceErrorString "let bad = \"open\".tail(0)"
      =
      some "static error at 1:18-1:22: method tail/1 is not available on str" := by
  native_decide

example :
    checkSourceErrorString "let bad = \"open\".take(true)"
      =
      some "static error at 1:18-1:22: method take/1 is not available on str" := by
  native_decide

example :
    checkSourceErrorString "let bad = \"open\".drop(true)"
      =
      some "static error at 1:18-1:22: method drop/1 is not available on str" := by
  native_decide

example :
    checkSourceErrorString "let bad = [1].tail(0)"
      =
      some "static error at 1:15-1:19: method tail/1 is not available on list[num]" := by
  native_decide

example :
    checkSourceErrorString "let bad = [1].take(true)"
      =
      some "static error at 1:15-1:19: method take/1 is not available on list[num]" := by
  native_decide

example :
    checkSourceErrorString "let bad = [1].drop(true)"
      =
      some "static error at 1:15-1:19: method drop/1 is not available on list[num]" := by
  native_decide

example :
    (compileSource "let x = \"unterminated"
      == Except.error
        (Error.lex
          "unexpected EOF in string"
          { start := { line := 1, column := 9 }, stop := { line := 1, column := 22 } })) := by
  native_decide

example :
    (compileSource "let x = 1 +"
      == Except.error
        (Error.parse
          (Parser.ParseError.expected Parser.ParseContext.expression (some Lexer.TokenKind.plus))
          { start := { line := 1, column := 11 }, stop := { line := 1, column := 12 } })) := by
  native_decide

example :
    (compileSource "\n~for i in 0 { break }"
      == Except.error
        (Error.parse
          (Parser.ParseError.expected Parser.ParseContext.range (some Lexer.TokenKind.for))
          { start := { line := 2, column := 2 }, stop := { line := 2, column := 5 } })) := by
  native_decide

example :
    (compileSource "let x = 1~\nlet y = 2 +"
      == Except.error
        (Error.parse
          (Parser.ParseError.expected Parser.ParseContext.expression (some Lexer.TokenKind.plus))
          { start := { line := 2, column := 11 }, stop := { line := 2, column := 12 } })) := by
  native_decide

example :
    (compileSource "if 1 + { break }"
      == Except.error
        (Error.parse
          (Parser.ParseError.expected Parser.ParseContext.expression (some Lexer.TokenKind.plus))
          { start := { line := 1, column := 6 }, stop := { line := 1, column := 7 } })) := by
  native_decide

example :
    (compileSource "while 1 + { break }"
      == Except.error
        (Error.parse
          (Parser.ParseError.expected Parser.ParseContext.expression (some Lexer.TokenKind.plus))
          { start := { line := 1, column := 9 }, stop := { line := 1, column := 10 } })) := by
  native_decide

example :
    (compileSource "seal until 1 + { break }"
      == Except.error
        (Error.parse
          (Parser.ParseError.expected Parser.ParseContext.expression (some Lexer.TokenKind.plus))
          { start := { line := 1, column := 14 }, stop := { line := 1, column := 15 } })) := by
  native_decide

example :
    (compileSource "let bad = 2 + true"
      == Except.error
        (Error.static
          (Static.CheckError.operandMismatch BinOp.add Static.Ty.num Static.Ty.bool)
          (some { start := { line := 1, column := 13 }, stop := { line := 1, column := 14 } }))) := by
  native_decide

example :
    (compileSource "fn id(x) { return x }~let bad = id(1, 2)"
      == Except.error
        (Error.static
          (Static.CheckError.arityMismatch "id" 1 2)
          (some { start := { line := 1, column := 33 }, stop := { line := 1, column := 35 } }))) := by
  native_decide

example :
    (compileSource "fn dup() { return 1 }~fn dup() { return 2 }"
      == Except.error
        (Error.static
          (Static.CheckError.duplicateFunction "dup")
          (some { start := { line := 1, column := 26 }, stop := { line := 1, column := 29 } }))) := by
  native_decide

example :
    (compileSource "fn dup() { return dup() }~fn dup() { return 2 }"
      == Except.error
        (Error.static
          (Static.CheckError.duplicateFunction "dup")
          (some { start := { line := 1, column := 30 }, stop := { line := 1, column := 33 } }))) := by
  native_decide

example :
    (compileSource "fn bad(x, x) { return x }"
      == Except.error
        (Error.static
          (Static.CheckError.duplicateParameter "x")
          (some { start := { line := 1, column := 11 }, stop := { line := 1, column := 12 } }))) := by
  native_decide

example :
    (compileSource "fn x(x, x) { return x }"
      == Except.error
        (Error.static
          (Static.CheckError.duplicateParameter "x")
          (some { start := { line := 1, column := 9 }, stop := { line := 1, column := 10 } }))) := by
  native_decide

example :
    (runSource 0 "let x = 1"
      ==
      Except.ok
        ( [("x", 0)]
        , []
        , { ip := 0
            code := [VM.FrameOp.push (Value.num 1), VM.FrameOp.store 0, VM.FrameOp.halt]
            stack := []
            locals := []
            frames := []
            halted := false })) := by
  native_decide

example :
    compileSourceErrorString "let x = \"unterminated"
      =
      some "lex error at 1:9-1:22: unexpected EOF in string" := by
  native_decide

example :
    compileSourceErrorString "let x = 1 +"
      =
      some "parse error at 1:11-1:12: expected expression, found +" := by
  native_decide

example :
    compileSourceErrorString "let bad = 2 + true"
      =
      some "static error at 1:13-1:14: operator + cannot accept num and bool" := by
  native_decide

example :
    compileSourceErrorString "fn dup() { return 1 }~fn dup() { return 2 }"
      =
      some "static error at 1:26-1:29: duplicate function dup" := by
  native_decide

example :
    parseSourceErrorString "let x = \"unterminated"
      =
      some "lex error at 1:9-1:22: unexpected EOF in string" := by
  native_decide

example :
    parseSourceErrorString "let x = \"bad\\q\""
      =
      some "lex error at 1:13-1:15: invalid escape: \\q" := by
  native_decide

example :
    parseSourceErrorString "let x = 1 /* never closes"
      =
      some "lex error at 1:11-1:26: unterminated block comment" := by
  native_decide

example :
    parseSourceErrorString "let x = 1 +"
      =
      some "parse error at 1:11-1:12: expected expression, found +" := by
  native_decide

example :
    parseSourceErrorString "let x = 1~\nlet y = 2 +"
      =
      some "parse error at 2:11-2:12: expected expression, found +" := by
  native_decide

example :
    parseSourceErrorString "for i in 0..3"
      =
      some "parse error at 1:1-1:4: expected block, found for" := by
  native_decide

example :
    parseSourceErrorString "for i in -2..2"
      =
      some "parse error at 1:1-1:4: expected block, found for" := by
  native_decide

example :
    parseSourceErrorString "if { break }"
      =
      some "parse error at 1:4-1:5: expected expression, found {" := by
  native_decide

example :
    parseSourceErrorString "if 1 + { break }"
      =
      some "parse error at 1:6-1:7: expected expression, found +" := by
  native_decide

example :
    parseSourceErrorString "else { break }"
      =
      some "parse error at 1:1-1:5: expected if-statement, found else" := by
  native_decide

example :
    parseSourceErrorString "while { break }"
      =
      some "parse error at 1:7-1:8: expected expression, found {" := by
  native_decide

example :
    parseSourceErrorString "while 1 + { break }"
      =
      some "parse error at 1:9-1:10: expected expression, found +" := by
  native_decide

example :
    parseSourceErrorString "seal until { break }"
      =
      some "parse error at 1:12-1:13: expected expression, found {" := by
  native_decide

example :
    parseSourceErrorString "seal until 1 + { break }"
      =
      some "parse error at 1:14-1:15: expected expression, found +" := by
  native_decide

example :
    parseSourceErrorString "fn add(a, b)"
      =
      some "parse error at 1:1-1:3: expected block, found fn" := by
  native_decide

example :
    parseSourceErrorString "let count: = 1"
      =
      some "parse error at 1:12-1:13: expected type, found =" := by
  native_decide

example :
    parseSourceErrorString "let xs: list[ = [1]"
      =
      some "parse error at 1:15-1:16: expected type, found =" := by
  native_decide

example :
    parseSourceErrorString "let xs: list[\n = [1]"
      =
      some "parse error at 2:2-2:3: expected type, found =" := by
  native_decide

example :
    parseSourceErrorString "fn id(x: num): { return x }"
      =
      some "parse error at 1:16-1:17: expected type, found {" := by
  native_decide

example :
    parseSourceErrorString "fn id(x): { return x }"
      =
      some "parse error at 1:11-1:12: expected type, found {" := by
  native_decide

example :
    parseSourceErrorString "fn id(x:) { return x }"
      =
      some "parse error at 1:9-1:10: expected type, found )" := by
  native_decide

example :
    parseSourceErrorString "fn id(x: list[) { return x }"
      =
      some "parse error at 1:15-1:16: expected type, found )" := by
  native_decide

example :
    parseSourceErrorString "fn id(x): list[ { return x }"
      =
      some "parse error at 1:17-1:18: expected type, found {" := by
  native_decide

example :
    parseSourceErrorString "manifold"
      =
      some "parse error at 1:1-1:9: expected statement, found manifold" := by
  native_decide

example :
    parseSourceErrorString "class"
      =
      some "parse error at 1:1-1:6: expected statement, found class" := by
  native_decide

example :
    checkSourceErrorString "let bad = 2 + true"
      =
      some "static error at 1:13-1:14: operator + cannot accept num and bool" := by
  native_decide

example :
    checkSourceErrorString "let bad = 1 && true"
      =
      some "static error at 1:13-1:15: operator && cannot accept num and bool" := by
  native_decide

example :
    checkSourceErrorString "let bad = false || 1"
      =
      some "static error at 1:17-1:19: operator || cannot accept bool and num" := by
  native_decide

example :
    checkSourceErrorString "let bad = 1 == true"
      =
      some "static error at 1:13-1:15: operator == cannot accept num and bool" := by
  native_decide

example :
    checkSourceErrorString "let bad = false != 1"
      =
      some "static error at 1:17-1:19: operator != cannot accept bool and num" := by
  native_decide

example :
    checkSourceErrorString "let bad = !1"
      =
      some "static error at 1:11-1:12: operator ! cannot accept num" := by
  native_decide

example :
    checkSourceErrorString "let count: num = true"
      =
      some "static error at 1:5-1:10: assignment to count expected num, found bool" := by
  native_decide

example :
    checkSourceErrorString "let bad = self"
      =
      some "static error at 1:11-1:15: undeclared variable self" := by
  native_decide

example :
    checkSourceErrorString "let bad = embed(1)"
      =
      some "static error at 1:11-1:16: undeclared function embed" := by
  native_decide

example :
    checkSourceErrorString "fn pick(a, b) { return a }~let bad = pick(c=1, a=2)"
      =
      some "static error at 1:43-1:44: function pick has no parameter c" := by
  native_decide

example :
    checkSourceErrorString "fn pick(a, b) { return a }~let bad = pick(a=1, a=2)"
      =
      some "static error at 1:48-1:49: duplicate named argument a for function pick" := by
  native_decide

example :
    checkSourceErrorString "fn id(x: num) { return x }~let bad = id(true)"
      =
      some "static error at 1:41-1:45: argument x for function id expected num, found bool" := by
  native_decide

example :
    checkSourceErrorString "fn bad(x: num): num { return true }"
      =
      some "static error at 1:30-1:34: function bad expected return num, found bool" := by
  native_decide

example :
    checkSourceErrorString "fn bad(x): num { return true }"
      =
      some "static error at 1:25-1:29: function bad expected return num, found bool" := by
  native_decide

example :
    checkSourceErrorString "fn bad(): num { if true { return 1 } else { return true } }"
      =
      some "static error at 1:52-1:56: function bad expected return num, found bool" := by
  native_decide

example :
    checkSourceErrorString "fn bad(): num { }"
      =
      some "static error at 1:17-1:18: function bad expected return num, found unit" := by
  native_decide

example :
    checkSourceErrorString "fn bad(): num { return unit }"
      =
      some "static error at 1:24-1:28: function bad expected return num, found unit" := by
  native_decide

example :
    checkSourceErrorString "fn len(xs: list[num]): num { return xs.length }~let bad = len([true])"
      =
      some "static error at 1:63-1:64: argument xs for function len expected list[num], found list[bool]" := by
  native_decide

example :
    checkSourceErrorString "break"
      =
      some "static error at 1:1-1:6: break outside loop" := by
  native_decide

example :
    checkSourceErrorString "if 1 { let x = 2 }"
      =
      some "static error at 1:4-1:5: condition expected bool, found num" := by
  native_decide

example :
    checkSourceErrorString "while 1 { break }"
      =
      some "static error at 1:7-1:8: condition expected bool, found num" := by
  native_decide

example :
    checkSourceErrorString "seal until 1 { break }"
      =
      some "static error at 1:12-1:13: condition expected bool, found num" := by
  native_decide

example :
    checkSourceErrorString "if true { fn nested() { return 1 } }"
      =
      some "static error at 1:14-1:20: nested function nested" := by
  native_decide

example :
    runSourceErrorString 0 "let x = 1" = none := by
  native_decide

example :
    sourceLocalErrorString 10 "let x = 1" 9
      =
      some "runtime error" := by
  native_decide

end Pipeline
end Aether
