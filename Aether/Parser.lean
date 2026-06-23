import Aether.Core
import Aether.Lexer

namespace Aether
namespace Parser

abbrev Tokens := List Lexer.TokenKind

inductive ParseContext where
  | expression
  | statement
  | ifStatement
  | block
  | range
  | params
  | typeAnnotation
  | terminator
  | programEnd
  deriving Repr, BEq, DecidableEq, Inhabited

inductive ParseError where
  | expected : ParseContext -> Option Lexer.TokenKind -> ParseError
  deriving Repr, BEq, DecidableEq, Inhabited

def skipTerminators : Tokens -> Tokens
  | Lexer.TokenKind.newline :: rest => skipTerminators rest
  | Lexer.TokenKind.tilde :: rest => skipTerminators rest
  | Lexer.TokenKind.semicolon :: rest => skipTerminators rest
  | tokens => tokens

def skipBinaryRhsNewlines : Tokens -> Tokens
  | Lexer.TokenKind.newline :: rest => skipBinaryRhsNewlines rest
  | tokens => tokens

def skipAssignRhsNewlines : Tokens -> Tokens
  | Lexer.TokenKind.newline :: rest => skipAssignRhsNewlines rest
  | tokens => tokens

def skipConditionNewlines : Tokens -> Tokens
  | Lexer.TokenKind.newline :: rest => skipConditionNewlines rest
  | tokens => tokens

partial def parseLeftAssoc
    (parseAtom : Tokens -> Option (Expr × Tokens))
    (operator : Lexer.TokenKind -> Option BinOp)
    (tokens : Tokens) : Option (Expr × Tokens) := do
  let (first, remaining) <- parseAtom tokens
  let rec loop (left : Expr) : Tokens -> Option (Expr × Tokens)
    | token :: rest =>
        match operator token with
        | some op => do
            let (right, afterRight) <- parseAtom (skipBinaryRhsNewlines rest)
            loop (Expr.binary left op right) afterRight
        | none => some (left, token :: rest)
    | [] => some (left, [])
  loop first remaining

def mulOperator : Lexer.TokenKind -> Option BinOp
  | Lexer.TokenKind.star => some BinOp.mul
  | Lexer.TokenKind.slash => some BinOp.div
  | Lexer.TokenKind.percent => some BinOp.mod
  | _ => none

def addOperator : Lexer.TokenKind -> Option BinOp
  | Lexer.TokenKind.plus => some BinOp.add
  | Lexer.TokenKind.minus => some BinOp.sub
  | _ => none

def comparisonOperator : Lexer.TokenKind -> Option BinOp
  | Lexer.TokenKind.less => some BinOp.lt
  | Lexer.TokenKind.greater => some BinOp.gt
  | Lexer.TokenKind.lessEq => some BinOp.le
  | Lexer.TokenKind.greaterEq => some BinOp.ge
  | _ => none

def equalityOperator : Lexer.TokenKind -> Option BinOp
  | Lexer.TokenKind.eqEq => some BinOp.eq
  | Lexer.TokenKind.notEq => some BinOp.neq
  | _ => none

def andOperator : Lexer.TokenKind -> Option BinOp
  | Lexer.TokenKind.and => some BinOp.and
  | _ => none

def orOperator : Lexer.TokenKind -> Option BinOp
  | Lexer.TokenKind.or => some BinOp.or
  | _ => none

def flexibleIdent? : Lexer.TokenKind -> Option Ident
  | Lexer.TokenKind.identifier name => some name
  | Lexer.TokenKind.dim => some "dim"
  | Lexer.TokenKind.tau => some "tau"
  | Lexer.TokenKind.model => some "model"
  | Lexer.TokenKind.color => some "color"
  | Lexer.TokenKind.axis => some "axis"
  | Lexer.TokenKind.project => some "project"
  | Lexer.TokenKind.cluster => some "cluster"
  | Lexer.TokenKind.center => some "center"
  | Lexer.TokenKind.spread => some "spread"
  | Lexer.TokenKind.format => some "format"
  | Lexer.TokenKind.output => some "output"
  | Lexer.TokenKind.escalate => some "escalate"
  | Lexer.TokenKind.convergence => some "convergence"
  | _ => none

def skipListNewlines : Tokens -> Tokens
  | Lexer.TokenKind.newline :: rest => skipListNewlines rest
  | tokens => tokens

def skipArgNewlines : Tokens -> Tokens
  | Lexer.TokenKind.newline :: rest => skipArgNewlines rest
  | tokens => tokens

def skipParenNewlines : Tokens -> Tokens
  | Lexer.TokenKind.newline :: rest => skipParenNewlines rest
  | tokens => tokens

def skipIndexNewlines : Tokens -> Tokens
  | Lexer.TokenKind.newline :: rest => skipIndexNewlines rest
  | tokens => tokens

def skipMemberNewlines : Tokens -> Tokens
  | Lexer.TokenKind.newline :: rest => skipMemberNewlines rest
  | tokens => tokens

def skipCallOpenNewlines : Tokens -> Tokens
  | Lexer.TokenKind.newline :: rest => skipCallOpenNewlines rest
  | tokens => tokens

def skipFnParamOpenNewlines : Tokens -> Tokens
  | Lexer.TokenKind.newline :: rest => skipFnParamOpenNewlines rest
  | tokens => tokens

def skipFnParamOpenNewlinesWithOffset (offset : Nat) : Tokens -> Nat × Tokens
  | Lexer.TokenKind.newline :: rest => skipFnParamOpenNewlinesWithOffset (offset + 1) rest
  | tokens => (offset, tokens)

def skipUnaryNewlines : Tokens -> Tokens
  | Lexer.TokenKind.newline :: rest => skipUnaryNewlines rest
  | tokens => tokens

mutual
  partial def parseListLiteral : Tokens -> Option (List Expr × Tokens)
    | tokens => do
        match skipListNewlines tokens with
        | Lexer.TokenKind.rBracket :: rest => some ([], rest)
        | tokens => do
            let (item, afterItem) <- parseExpr tokens
            match skipListNewlines afterItem with
            | Lexer.TokenKind.comma :: rest =>
                match skipListNewlines rest with
                | Lexer.TokenKind.rBracket :: afterBracket => some ([item], afterBracket)
                | afterComma => do
                    let (items, afterItems) <- parseListLiteral afterComma
                    some (item :: items, afterItems)
            | Lexer.TokenKind.rBracket :: rest => some ([item], rest)
            | _ => none

  partial def parseArgList : Tokens -> Option (List Arg × Tokens)
    | tokens => do
        match skipArgNewlines tokens with
        | Lexer.TokenKind.rParen :: rest => some ([], rest)
        | nameToken :: Lexer.TokenKind.equals :: rest => do
            let name <- flexibleIdent? nameToken
            let (value, afterValue) <- parseExpr (skipArgNewlines rest)
            match skipArgNewlines afterValue with
            | Lexer.TokenKind.comma :: afterComma =>
                match skipArgNewlines afterComma with
                | Lexer.TokenKind.rParen :: afterParen => some ([Arg.named name value], afterParen)
                | afterComma => do
                    let (args, afterArgs) <- parseArgList afterComma
                    some (Arg.named name value :: args, afterArgs)
            | Lexer.TokenKind.rParen :: afterParen => some ([Arg.named name value], afterParen)
            | _ => none
        | tokens => do
            let (arg, afterArg) <- parseExpr tokens
            match skipArgNewlines afterArg with
            | Lexer.TokenKind.comma :: rest =>
                match skipArgNewlines rest with
                | Lexer.TokenKind.rParen :: afterParen => some ([Arg.positional arg], afterParen)
                | afterComma => do
                    let (args, afterArgs) <- parseArgList afterComma
                    some (Arg.positional arg :: args, afterArgs)
            | Lexer.TokenKind.rParen :: rest => some ([Arg.positional arg], rest)
            | _ => none

  partial def parsePrimary : Tokens -> Option (Expr × Tokens)
    | Lexer.TokenKind.number n :: rest => some (Expr.num n, rest)
    | Lexer.TokenKind.float intPart fracMicros :: rest => some (Expr.float intPart fracMicros, rest)
    | Lexer.TokenKind.true :: rest => some (Expr.bool true, rest)
    | Lexer.TokenKind.false :: rest => some (Expr.bool false, rest)
    | Lexer.TokenKind.stringLit value :: rest => some (Expr.str value, rest)
    | Lexer.TokenKind.identifier "unit" :: rest => some (Expr.unit, rest)
    | Lexer.TokenKind.lBracket :: rest => do
        let (items, afterItems) <- parseListLiteral rest
        some (Expr.list items, afterItems)
    | Lexer.TokenKind.identifier name :: rest =>
        match skipCallOpenNewlines rest with
        | Lexer.TokenKind.lParen :: afterOpen => do
            let (args, afterArgs) <- parseArgList afterOpen
            some (Expr.call name args, afterArgs)
        | _ => some (Expr.var name, rest)
    | Lexer.TokenKind.embed :: rest =>
        match skipCallOpenNewlines rest with
        | Lexer.TokenKind.lParen :: afterOpen => do
            let (args, afterArgs) <- parseArgList afterOpen
            some (Expr.call "embed" args, afterArgs)
        | _ => none
    | Lexer.TokenKind.convergence :: rest =>
        match skipCallOpenNewlines rest with
        | Lexer.TokenKind.lParen :: afterOpen => do
            let (args, afterArgs) <- parseArgList afterOpen
            some (Expr.call "convergence" args, afterArgs)
        | _ => none
    | Lexer.TokenKind.self :: rest => some (Expr.var "self", rest)
    | Lexer.TokenKind.lParen :: rest => do
        let (expr, afterExpr) <- parseExpr (skipParenNewlines rest)
        match skipParenNewlines afterExpr with
        | Lexer.TokenKind.rParen :: afterParen => some (expr, afterParen)
        | _ => none
    | _ => none

  partial def parsePostfixLoop (left : Expr) : Tokens -> Option (Expr × Tokens)
    | Lexer.TokenKind.lBracket :: rest => do
        let (index, afterIndex) <- parseExpr (skipIndexNewlines rest)
        match skipIndexNewlines afterIndex with
        | Lexer.TokenKind.rBracket :: afterBracket =>
            parsePostfixLoop (Expr.index left index) afterBracket
        | _ => none
    | Lexer.TokenKind.dot :: rest =>
        match skipMemberNewlines rest with
        | nameToken :: Lexer.TokenKind.lParen :: afterName => do
            let method <- flexibleIdent? nameToken
            let (args, afterArgs) <- parseArgList afterName
            parsePostfixLoop (Expr.method left method args) afterArgs
        | nameToken :: afterName => do
            let field <- flexibleIdent? nameToken
            parsePostfixLoop (Expr.field left field) afterName
        | _ => none
    | tokens => some (left, tokens)

  partial def parsePostfix (tokens : Tokens) : Option (Expr × Tokens) := do
    let (primary, remaining) <- parsePrimary tokens
    parsePostfixLoop primary remaining

  partial def parseUnary : Tokens -> Option (Expr × Tokens)
    | Lexer.TokenKind.minus :: rest => do
        let (expr, remaining) <- parseUnary (skipUnaryNewlines rest)
        some (Expr.unary UnOp.neg expr, remaining)
    | Lexer.TokenKind.not :: rest => do
        let (expr, remaining) <- parseUnary (skipUnaryNewlines rest)
        some (Expr.unary UnOp.not expr, remaining)
    | tokens => parsePostfix tokens

  partial def parseMul (tokens : Tokens) : Option (Expr × Tokens) :=
    parseLeftAssoc parseUnary mulOperator tokens

  partial def parseAdd (tokens : Tokens) : Option (Expr × Tokens) :=
    parseLeftAssoc parseMul addOperator tokens

  partial def parseComparison (tokens : Tokens) : Option (Expr × Tokens) :=
    parseLeftAssoc parseAdd comparisonOperator tokens

  partial def parseEquality (tokens : Tokens) : Option (Expr × Tokens) :=
    parseLeftAssoc parseComparison equalityOperator tokens

  partial def parseAnd (tokens : Tokens) : Option (Expr × Tokens) :=
    parseLeftAssoc parseEquality andOperator tokens

  partial def parseOr (tokens : Tokens) : Option (Expr × Tokens) :=
    parseLeftAssoc parseAnd orOperator tokens

  partial def parseExpr (tokens : Tokens) : Option (Expr × Tokens) :=
    parseOr tokens
end

def expectTerminator : Tokens -> Option Tokens
  | Lexer.TokenKind.newline :: rest => some (skipTerminators rest)
  | Lexer.TokenKind.tilde :: rest => some (skipTerminators rest)
  | Lexer.TokenKind.semicolon :: rest => some (skipTerminators rest)
  | tokens@(Lexer.TokenKind.rBrace :: _) => some tokens
  | Lexer.TokenKind.eof :: rest => some (Lexer.TokenKind.eof :: rest)
  | [] => some []
  | _ => none

def parseSignedInt : Tokens -> Option (Int × Tokens)
  | Lexer.TokenKind.number value :: rest => some (value, rest)
  | Lexer.TokenKind.minus :: Lexer.TokenKind.number value :: rest => some (-value, rest)
  | _ => none

def parseIntRange (tokens : Tokens) : Option (Int × Int × Tokens) := do
  let (start, afterStart) <- parseSignedInt tokens
  match afterStart with
  | Lexer.TokenKind.dotDot :: afterDot => do
      let (stop, rest) <- parseSignedInt afterDot
      some (start, stop, rest)
  | _ => none

def skipParamNewlines : Tokens -> Tokens
  | Lexer.TokenKind.newline :: rest => skipParamNewlines rest
  | tokens => tokens

def skipTypeNewlines : Tokens -> Tokens
  | Lexer.TokenKind.newline :: rest => skipTypeNewlines rest
  | tokens => tokens

def skipTypeNewlinesWithOffset : Nat -> Tokens -> Nat × Tokens
  | offset, Lexer.TokenKind.newline :: rest => skipTypeNewlinesWithOffset (offset + 1) rest
  | offset, tokens => (offset, tokens)

mutual
  partial def parseParamList : Tokens -> Option (List Ident × Tokens)
    | tokens =>
        match skipParamNewlines tokens with
        | Lexer.TokenKind.rParen :: rest => some ([], rest)
        | Lexer.TokenKind.identifier name :: rest =>
            match skipParamNewlines rest with
            | Lexer.TokenKind.comma :: afterComma =>
                match skipParamNewlines afterComma with
                | Lexer.TokenKind.rParen :: afterParen => some ([name], afterParen)
                | afterComma => do
                    let (params, afterParams) <- parseParamList afterComma
                    some (name :: params, afterParams)
            | Lexer.TokenKind.rParen :: afterParen => some ([name], afterParen)
            | _ => none
        | _ => none

  partial def parseAnnTy : Tokens -> Option (AnnTy × Tokens)
    | tokens =>
        match skipTypeNewlines tokens with
        | Lexer.TokenKind.identifier "num" :: rest => some (AnnTy.num, rest)
        | Lexer.TokenKind.identifier "bool" :: rest => some (AnnTy.bool, rest)
        | Lexer.TokenKind.identifier "str" :: rest => some (AnnTy.str, rest)
        | Lexer.TokenKind.identifier "unit" :: rest => some (AnnTy.unit, rest)
        | Lexer.TokenKind.identifier "list" :: Lexer.TokenKind.lBracket :: rest => do
            let (elementTy, afterElementTy) <- parseAnnTy (skipTypeNewlines rest)
            match skipTypeNewlines afterElementTy with
            | Lexer.TokenKind.rBracket :: afterBracket => some (AnnTy.list elementTy, afterBracket)
            | _ => none
        | _ => none

  partial def parseTypedParamList : Tokens -> Option (List (Ident × AnnTy) × Tokens)
    | tokens =>
        match skipParamNewlines tokens with
        | Lexer.TokenKind.rParen :: rest => some ([], rest)
        | Lexer.TokenKind.identifier name :: Lexer.TokenKind.colon :: rest => do
            let (ty, afterTy) <- parseAnnTy (skipParamNewlines rest)
            match skipParamNewlines afterTy with
            | Lexer.TokenKind.comma :: afterComma =>
                match skipParamNewlines afterComma with
                | Lexer.TokenKind.rParen :: afterParen => some ([(name, ty)], afterParen)
                | afterComma => do
                    let (params, afterParams) <- parseTypedParamList afterComma
                    some ((name, ty) :: params, afterParams)
            | Lexer.TokenKind.rParen :: afterParen => some ([(name, ty)], afterParen)
            | _ => none
        | _ => none

  partial def parseBlock : Tokens -> Option (List Stmt × Tokens)
    | tokens =>
        match skipTerminators tokens with
        | Lexer.TokenKind.lBrace :: rest => parseBlockBody (skipTerminators rest)
        | _ => none

  partial def parseBlockBody : Tokens -> Option (List Stmt × Tokens)
    | [] => none
    | Lexer.TokenKind.eof :: _ => none
    | tokens => do
        let tokens := skipTerminators tokens
        match tokens with
        | Lexer.TokenKind.rBrace :: rest => some ([], skipTerminators rest)
        | [] => none
        | Lexer.TokenKind.eof :: _ => none
        | _ =>
            let (stmt, afterStmt) <- parseStmt tokens
            let (rest, afterBlock) <- parseBlockBody afterStmt
            some (stmt :: rest, afterBlock)

  partial def parseStmt : Tokens -> Option (Stmt × Tokens)
    | Lexer.TokenKind.if_ :: rest => do
        let (condition, afterCondition) <- parseExpr (skipConditionNewlines rest)
        let (thenBranch, afterThen) <- parseBlock afterCondition
        match afterThen with
        | Lexer.TokenKind.else_ :: afterElse => do
            let (elseBranch, afterElseBlock) <- parseBlock afterElse
            some (Stmt.ifThenElse condition thenBranch (some elseBranch), afterElseBlock)
        | _ =>
            some (Stmt.ifThenElse condition thenBranch none, afterThen)
    | Lexer.TokenKind.while :: rest => do
        let (condition, afterCondition) <- parseExpr (skipConditionNewlines rest)
        let (body, afterBody) <- parseBlock afterCondition
        some (Stmt.while condition body, afterBody)
    | Lexer.TokenKind.for :: Lexer.TokenKind.identifier iterator :: Lexer.TokenKind.in_ :: rest => do
        let (start, stop, afterRange) <- parseIntRange rest
        let (body, afterBody) <- parseBlock afterRange
        some (Stmt.forRange iterator start stop body, afterBody)
    | Lexer.TokenKind.seal :: Lexer.TokenKind.until :: rest => do
        let (condition, afterCondition) <- parseExpr (skipConditionNewlines rest)
        let (body, afterBody) <- parseBlock afterCondition
        some (Stmt.seal (some condition) body, afterBody)
    | Lexer.TokenKind.seal :: rest => do
        let (body, afterBody) <- parseBlock rest
        some (Stmt.seal none body, afterBody)
    | Lexer.TokenKind.fn :: Lexer.TokenKind.identifier name :: rest => do
        match skipFnParamOpenNewlines rest with
        | Lexer.TokenKind.lParen :: afterOpen =>
            match parseTypedParamList afterOpen with
            | some (params, afterParams) =>
                match afterParams with
                | Lexer.TokenKind.colon :: afterColon => do
                    let (returnTy, afterReturnTy) <- parseAnnTy afterColon
                    let (body, afterBody) <- parseBlock afterReturnTy
                    some (Stmt.fnDeclTypedReturn name params returnTy body, afterBody)
                | _ => do
                    let (body, afterBody) <- parseBlock afterParams
                    some (Stmt.fnDeclTyped name params body, afterBody)
            | none => do
                let (params, afterParams) <- parseParamList afterOpen
                match afterParams with
                | Lexer.TokenKind.colon :: afterColon => do
                    let (returnTy, afterReturnTy) <- parseAnnTy afterColon
                    let (body, afterBody) <- parseBlock afterReturnTy
                    some (Stmt.fnDeclReturn name params returnTy body, afterBody)
                | _ => do
                    let (body, afterBody) <- parseBlock afterParams
                    some (Stmt.fnDecl name params body, afterBody)
        | _ => none
    | Lexer.TokenKind.let_ :: Lexer.TokenKind.identifier name :: Lexer.TokenKind.colon :: rest => do
        let (ty, afterTy) <- parseAnnTy rest
        match afterTy with
        | Lexer.TokenKind.equals :: afterEquals => do
            let (value, afterExpr) <- parseExpr (skipAssignRhsNewlines afterEquals)
            let remaining <- expectTerminator afterExpr
            some (Stmt.letDeclTyped name ty value, remaining)
        | _ => none
    | Lexer.TokenKind.let_ :: Lexer.TokenKind.identifier name :: Lexer.TokenKind.equals :: rest => do
        let (value, afterExpr) <- parseExpr (skipAssignRhsNewlines rest)
        let remaining <- expectTerminator afterExpr
        some (Stmt.letDecl name value, remaining)
    | Lexer.TokenKind.identifier name :: Lexer.TokenKind.equals :: rest => do
        let (value, afterExpr) <- parseExpr (skipAssignRhsNewlines rest)
        let remaining <- expectTerminator afterExpr
        some (Stmt.assign name value, remaining)
    | Lexer.TokenKind.return :: rest => do
        match rest with
        | Lexer.TokenKind.newline :: _ =>
            let remaining <- expectTerminator rest
            some (Stmt.ret none, remaining)
        | Lexer.TokenKind.tilde :: _ =>
            let remaining <- expectTerminator rest
            some (Stmt.ret none, remaining)
        | Lexer.TokenKind.semicolon :: _ =>
            let remaining <- expectTerminator rest
            some (Stmt.ret none, remaining)
        | Lexer.TokenKind.rBrace :: _ =>
            some (Stmt.ret none, rest)
        | Lexer.TokenKind.eof :: _ =>
            some (Stmt.ret none, rest)
        | [] => some (Stmt.ret none, [])
        | _ => do
            let (value, afterExpr) <- parseExpr rest
            let remaining <- expectTerminator afterExpr
            some (Stmt.ret (some value), remaining)
    | Lexer.TokenKind.break :: rest => do
        let remaining <- expectTerminator rest
        some (Stmt.break, remaining)
    | Lexer.TokenKind.continue :: rest => do
        let remaining <- expectTerminator rest
        some (Stmt.continue, remaining)
    | tokens => do
        let (expr, afterExpr) <- parseExpr tokens
        let remaining <- expectTerminator afterExpr
        some (Stmt.expr expr, remaining)

  partial def parseProgramFromTokens : Tokens -> Option (List Stmt)
    | [] => some []
    | Lexer.TokenKind.eof :: _ => some []
    | tokens => do
        let tokens := skipTerminators tokens
        match tokens with
        | [] => some []
        | Lexer.TokenKind.eof :: _ => some []
        | _ =>
            let (stmt, afterStmt) <- parseStmt tokens
            let rest <- parseProgramFromTokens afterStmt
            some (stmt :: rest)
end

def parseProgram (source : String) : Option (List Stmt) :=
  parseProgramFromTokens (Lexer.tokenize source)

def firstToken? : Tokens -> Option Lexer.TokenKind
  | [] => none
  | token :: _ => some token

def tokenAt? : Tokens -> Nat -> Option Lexer.TokenKind
  | [], _ => none
  | token :: _, 0 => some token
  | _ :: rest, index + 1 => tokenAt? rest index

def startsExpr : Lexer.TokenKind -> Bool
  | Lexer.TokenKind.number _ => true
  | Lexer.TokenKind.float _ _ => true
  | Lexer.TokenKind.true => true
  | Lexer.TokenKind.false => true
  | Lexer.TokenKind.stringLit _ => true
  | Lexer.TokenKind.lBracket => true
  | Lexer.TokenKind.identifier _ => true
  | Lexer.TokenKind.self => true
  | Lexer.TokenKind.lParen => true
  | Lexer.TokenKind.minus => true
  | Lexer.TokenKind.not => true
  | _ => false

def startsExprPrefix : Tokens -> Bool
  | Lexer.TokenKind.embed :: Lexer.TokenKind.lParen :: _ => true
  | Lexer.TokenKind.convergence :: Lexer.TokenKind.lParen :: _ => true
  | token :: _ => startsExpr token
  | [] => false

def invalidExprStartOffset? (offset : Nat) : Tokens -> Option Nat
  | tokens@(_ :: _) => if startsExprPrefix tokens then none else some offset
  | [] => some offset

def isBinOpToken : Lexer.TokenKind -> Bool
  | Lexer.TokenKind.plus => true
  | Lexer.TokenKind.minus => true
  | Lexer.TokenKind.star => true
  | Lexer.TokenKind.slash => true
  | Lexer.TokenKind.percent => true
  | Lexer.TokenKind.less => true
  | Lexer.TokenKind.greater => true
  | Lexer.TokenKind.lessEq => true
  | Lexer.TokenKind.greaterEq => true
  | Lexer.TokenKind.eqEq => true
  | Lexer.TokenKind.notEq => true
  | Lexer.TokenKind.and => true
  | Lexer.TokenKind.or => true
  | _ => false

def isExprTerminator : Lexer.TokenKind -> Bool
  | Lexer.TokenKind.newline => true
  | Lexer.TokenKind.tilde => true
  | Lexer.TokenKind.semicolon => true
  | Lexer.TokenKind.rBrace => true
  | Lexer.TokenKind.eof => true
  | _ => false

def isConditionTerminator : Lexer.TokenKind -> Bool
  | Lexer.TokenKind.lBrace => true
  | token => isExprTerminator token

def trailingBinOpOffset? (offset : Nat) : Tokens -> Option Nat
  | token :: next :: _ =>
      if isBinOpToken token && isExprTerminator next then
        some offset
      else
        none
  | [token] =>
      if isBinOpToken token then some offset else none
  | [] => none

partial def trailingBinOpInExprOffset? (offset : Nat) : Tokens -> Option Nat
  | [] => none
  | token :: rest =>
      match trailingBinOpOffset? offset (token :: rest) with
      | some found => some found
      | none =>
          if isExprTerminator token then
            none
          else
            trailingBinOpInExprOffset? (offset + 1) rest

def trailingBinOpInConditionOffset? (offset : Nat) : Tokens -> Option Nat
  | token :: next :: rest =>
      if isBinOpToken token && isConditionTerminator next then
        some offset
      else if isConditionTerminator token then
        none
      else
        trailingBinOpInConditionOffset? (offset + 1) (next :: rest)
  | [token] =>
      if isBinOpToken token then some offset else none
  | [] => none

def hasParamListPrefix : Tokens -> Bool
  | Lexer.TokenKind.rParen :: _ => true
  | Lexer.TokenKind.identifier _ :: Lexer.TokenKind.rParen :: _ => true
  | Lexer.TokenKind.identifier _ :: Lexer.TokenKind.comma :: rest => hasParamListPrefix rest
  | _ => false

def hasSignedIntPrefix : Tokens -> Option Tokens
  | Lexer.TokenKind.number _ :: rest => some rest
  | Lexer.TokenKind.minus :: Lexer.TokenKind.number _ :: rest => some rest
  | _ => none

def hasIntRangePrefix (tokens : Tokens) : Bool :=
  match hasSignedIntPrefix tokens with
  | some (Lexer.TokenKind.dotDot :: afterDot) =>
      match hasSignedIntPrefix afterDot with
      | some _ => true
      | none => false
  | _ => false

def typedParamListRemainder? (tokens : Tokens) : Option Tokens :=
  match parseTypedParamList tokens with
  | some (_, rest) => some rest
  | none => none

def paramListRemainder? (tokens : Tokens) : Option Tokens :=
  match typedParamListRemainder? tokens with
  | some rest => some rest
  | none =>
      match parseParamList tokens with
      | some (_, rest) => some rest
      | none => none

partial def annTyFailureOffset? (offset : Nat) : Tokens -> Option Nat
  | tokens =>
      let (offset, tokens) := skipTypeNewlinesWithOffset offset tokens
      match tokens with
      | Lexer.TokenKind.identifier "num" :: _ => none
      | Lexer.TokenKind.identifier "bool" :: _ => none
      | Lexer.TokenKind.identifier "str" :: _ => none
      | Lexer.TokenKind.identifier "unit" :: _ => none
      | Lexer.TokenKind.identifier "list" :: Lexer.TokenKind.lBracket :: rest =>
          let (elementOffset, elementTokens) := skipTypeNewlinesWithOffset (offset + 2) rest
          match parseAnnTy elementTokens with
          | some (_, afterElementTy) =>
              let afterElementOffset := elementOffset + (elementTokens.length - afterElementTy.length)
              let (closeOffset, closeTokens) := skipTypeNewlinesWithOffset afterElementOffset afterElementTy
              match closeTokens with
              | Lexer.TokenKind.rBracket :: _ => none
              | _ => some closeOffset
          | none => annTyFailureOffset? elementOffset elementTokens
      | _ :: _ => some offset
      | [] => some offset

def malformedReturnTypeAfterParams? (tokens : Tokens) : Bool :=
  match paramListRemainder? tokens with
  | some (Lexer.TokenKind.colon :: afterColon) =>
      match parseAnnTy afterColon with
      | some _ => false
      | none => true
  | _ => false

partial def malformedParamTypeOffset? (offset : Nat) : Tokens -> Option Nat
  | Lexer.TokenKind.identifier _ :: Lexer.TokenKind.colon :: rest =>
      match parseAnnTy rest with
      | some (_, afterTy@(Lexer.TokenKind.comma :: afterComma)) =>
          malformedParamTypeOffset? (offset + 2 + (rest.length - afterTy.length) + 1) afterComma
      | some (_, Lexer.TokenKind.rParen :: _) => none
      | some _ => none
      | none => annTyFailureOffset? (offset + 2) rest
  | Lexer.TokenKind.identifier _ :: Lexer.TokenKind.comma :: rest =>
      malformedParamTypeOffset? (offset + 2) rest
  | Lexer.TokenKind.identifier _ :: Lexer.TokenKind.rParen :: _ => none
  | Lexer.TokenKind.rParen :: _ => none
  | _ => none

def classifyStmtStart : Tokens -> ParseContext
  | Lexer.TokenKind.if_ :: Lexer.TokenKind.lBrace :: _ => ParseContext.expression
  | Lexer.TokenKind.if_ :: rest =>
      match trailingBinOpInConditionOffset? 1 rest with
      | some _ => ParseContext.expression
      | none => ParseContext.block
  | Lexer.TokenKind.else_ :: _ => ParseContext.ifStatement
  | Lexer.TokenKind.while :: Lexer.TokenKind.lBrace :: _ => ParseContext.expression
  | Lexer.TokenKind.while :: rest =>
      match trailingBinOpInConditionOffset? 1 rest with
      | some _ => ParseContext.expression
      | none => ParseContext.block
  | Lexer.TokenKind.for :: Lexer.TokenKind.identifier _ :: Lexer.TokenKind.in_ :: rest =>
      if hasIntRangePrefix rest then
        ParseContext.block
      else
        ParseContext.range
  | Lexer.TokenKind.for :: _ => ParseContext.range
  | Lexer.TokenKind.seal :: Lexer.TokenKind.until :: _ => ParseContext.expression
  | Lexer.TokenKind.seal :: _ => ParseContext.block
  | Lexer.TokenKind.let_ :: Lexer.TokenKind.identifier _ :: Lexer.TokenKind.colon :: rest =>
      match parseAnnTy rest with
      | some _ => ParseContext.expression
      | none => ParseContext.typeAnnotation
  | Lexer.TokenKind.fn :: Lexer.TokenKind.identifier _ :: rest =>
      match skipFnParamOpenNewlines rest with
      | Lexer.TokenKind.lParen :: afterOpen =>
          if malformedReturnTypeAfterParams? afterOpen then
            ParseContext.typeAnnotation
          else if (malformedParamTypeOffset? 3 afterOpen).isSome then
            ParseContext.typeAnnotation
          else if hasParamListPrefix afterOpen || (typedParamListRemainder? afterOpen).isSome then
            ParseContext.block
          else
            ParseContext.params
      | _ => ParseContext.params
  | Lexer.TokenKind.fn :: _ => ParseContext.params
  | Lexer.TokenKind.let_ :: _ => ParseContext.expression
  | Lexer.TokenKind.identifier _ :: Lexer.TokenKind.equals :: _ => ParseContext.expression
  | Lexer.TokenKind.return :: _ => ParseContext.expression
  | Lexer.TokenKind.break :: _ => ParseContext.terminator
  | Lexer.TokenKind.continue :: _ => ParseContext.terminator
  | _ => ParseContext.statement

def diagnosticOffset : Tokens -> Nat
  | Lexer.TokenKind.if_ :: rest =>
      match trailingBinOpInConditionOffset? 1 rest with
      | some offset => offset
      | none =>
        match invalidExprStartOffset? 1 rest with
        | some offset => offset
        | none => 0
  | Lexer.TokenKind.while :: rest =>
      match trailingBinOpInConditionOffset? 1 rest with
      | some offset => offset
      | none =>
        match invalidExprStartOffset? 1 rest with
        | some offset => offset
        | none => 0
  | Lexer.TokenKind.seal :: Lexer.TokenKind.until :: rest =>
      match trailingBinOpInConditionOffset? 2 rest with
      | some offset => offset
      | none =>
        match invalidExprStartOffset? 2 rest with
        | some offset => offset
        | none => 0
  | Lexer.TokenKind.let_ :: Lexer.TokenKind.identifier _ :: Lexer.TokenKind.colon :: rest =>
      match parseAnnTy rest with
      | some _ => 0
      | none =>
          match annTyFailureOffset? 3 rest with
          | some offset => offset
          | none => 3
  | Lexer.TokenKind.fn :: Lexer.TokenKind.identifier _ :: rest =>
      let (openOffset, afterSkipped) := skipFnParamOpenNewlinesWithOffset 2 rest
      match afterSkipped with
      | Lexer.TokenKind.lParen :: afterOpen =>
          let paramStartOffset := openOffset + 1
          match malformedParamTypeOffset? paramStartOffset afterOpen with
          | some offset => offset
          | none =>
              match paramListRemainder? afterOpen with
              | some (Lexer.TokenKind.colon :: afterColon) =>
                  match parseAnnTy afterColon with
                  | some _ => 0
                  | none =>
                      let afterColonOffset := paramStartOffset + (afterOpen.length - (Lexer.TokenKind.colon :: afterColon).length) + 1
                      match annTyFailureOffset? afterColonOffset afterColon with
                      | some offset => offset
                      | none => afterColonOffset
              | _ => 0
      | _ => 0
  | Lexer.TokenKind.let_ :: Lexer.TokenKind.identifier _ :: Lexer.TokenKind.equals :: rest =>
      match trailingBinOpInExprOffset? 3 rest with
      | some offset => offset
      | none =>
        match invalidExprStartOffset? 3 rest with
        | some offset => offset
        | none => 0
  | Lexer.TokenKind.identifier _ :: Lexer.TokenKind.equals :: rest =>
      match trailingBinOpInExprOffset? 2 rest with
      | some offset => offset
      | none =>
        match invalidExprStartOffset? 2 rest with
        | some offset => offset
        | none => 0
  | Lexer.TokenKind.return :: rest =>
      match trailingBinOpInExprOffset? 1 rest with
      | some offset => offset
      | none =>
        match invalidExprStartOffset? 1 rest with
        | some offset => offset
        | none => 0
  | _ => 0

def parseProgramFromTokensDetailed (tokens : Tokens) : Except ParseError (List Stmt) :=
  match parseProgramFromTokens tokens with
  | some stmts => Except.ok stmts
  | none =>
      let stripped := skipTerminators tokens
      let offset := diagnosticOffset stripped
      Except.error (ParseError.expected (classifyStmtStart stripped) (tokenAt? stripped offset))

def parseProgramDetailed (source : String) : Except ParseError (List Stmt) :=
  parseProgramFromTokensDetailed (Lexer.tokenize source)

def parseFailedAs (result : Except ParseError α) (err : ParseError) : Bool :=
  match result with
  | Except.error found => found == err
  | Except.ok _ => false

example :
    (parseExpr (Lexer.tokenize "1 + 2 * 3")
      ==
      some
        ( Expr.binary
            (Expr.num 1)
            BinOp.add
            (Expr.binary (Expr.num 2) BinOp.mul (Expr.num 3))
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "1 +\n2 * 3")
      ==
      some
        ( Expr.binary
            (Expr.num 1)
            BinOp.add
            (Expr.binary (Expr.num 2) BinOp.mul (Expr.num 3))
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "!(x == 0) || false")
      ==
      some
        ( Expr.binary
            (Expr.unary
              UnOp.not
              (Expr.binary (Expr.var "x") BinOp.eq (Expr.num 0)))
            BinOp.or
            (Expr.bool false)
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "true &&\n!false")
      ==
      some
        ( Expr.binary
            (Expr.bool true)
            BinOp.and
            (Expr.unary UnOp.not (Expr.bool false))
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "!\nfalse")
      ==
      some (Expr.unary UnOp.not (Expr.bool false), [Lexer.TokenKind.eof])) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "-\n3")
      ==
      some (Expr.unary UnOp.neg (Expr.num 3), [Lexer.TokenKind.eof])) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "(\n1 + 2\n)")
      ==
      some
        ( Expr.binary (Expr.num 1) BinOp.add (Expr.num 2)
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "unit")
      ==
      some (Expr.unit, [Lexer.TokenKind.eof])) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "add(1, x * 2)")
      ==
      some
        ( Expr.call
            "add"
            [ Expr.num 1
            , Expr.binary (Expr.var "x") BinOp.mul (Expr.num 2)
            ]
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "add(1,\n2,)")
      ==
      some
        ( Expr.call
            "add"
            [Expr.num 1, Expr.num 2]
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "embed(1)")
      ==
      some
        ( Expr.call
            "embed"
            [Expr.num 1]
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "embed(data, dim=3)")
      ==
      some
        ( Expr.call
            "embed"
            [ Arg.positional (Expr.var "data")
            , Arg.named "dim" (Expr.num 3)
            ]
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "convergence(0.1)")
      ==
      some
        ( Expr.call
            "convergence"
            [Expr.float 0 100000]
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "convergence\n(0.1)")
      ==
      some
        ( Expr.call
            "convergence"
            [Expr.float 0 100000]
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "self.cluster(axis=2)")
      ==
      some
        ( Expr.method
            (Expr.var "self")
            "cluster"
            [Arg.named "axis" (Expr.num 2)]
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "1.5") ==
      some (Expr.float 1 500000, [Lexer.TokenKind.eof])) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "\"open\"") ==
      some (Expr.str "open", [Lexer.TokenKind.eof])) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "self") ==
      some (Expr.var "self", [Lexer.TokenKind.eof])) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "self.length") ==
      some (Expr.field (Expr.var "self") "length", [Lexer.TokenKind.eof])) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "self.\nlength") ==
      some (Expr.field (Expr.var "self") "length", [Lexer.TokenKind.eof])) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "[]") ==
      some (Expr.list [], [Lexer.TokenKind.eof])) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "[1, true, \"open\"]") ==
      some (Expr.list [Expr.num 1, Expr.bool true, Expr.str "open"], [Lexer.TokenKind.eof])) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "[1, 2][0]") ==
      some
        ( Expr.index
            (Expr.list [Expr.num 1, Expr.num 2])
            (Expr.num 0)
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "[1, 2][\n0\n]") ==
      some
        ( Expr.index
            (Expr.list [Expr.num 1, Expr.num 2])
            (Expr.num 0)
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "[1, 2].length") ==
      some
        ( Expr.field
            (Expr.list [Expr.num 1, Expr.num 2])
            "length"
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "self.dim") ==
      some (Expr.field (Expr.var "self") "dim", [Lexer.TokenKind.eof])) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "[1, 2].len()") ==
      some
        ( Expr.method
            (Expr.list [Expr.num 1, Expr.num 2])
            "len"
            []
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "[1, 2].\nlen()") ==
      some
        ( Expr.method
            (Expr.list [Expr.num 1, Expr.num 2])
            "len"
            []
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseExpr (Lexer.tokenize "self.cluster()") ==
      some (Expr.method (Expr.var "self") "cluster" [], [Lexer.TokenKind.eof])) = true := by
  native_decide

example :
    (parseProgram "let x = 1 + 2 * 3~x = x + 1~return x"
      ==
      some
        [ Stmt.letDecl
            "x"
            (Expr.binary
              (Expr.num 1)
              BinOp.add
              (Expr.binary (Expr.num 2) BinOp.mul (Expr.num 3)))
        , Stmt.assign
            "x"
            (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))
        , Stmt.ret (some (Expr.var "x"))
        ]) = true := by
  native_decide

example :
    (parseProgram "let count: num = 1"
      ==
      some [Stmt.letDeclTyped "count" AnnTy.num (Expr.num 1)]) = true := by
  native_decide

example :
    (parseProgram "let total =\n1 + 2"
      ==
      some
        [Stmt.letDecl "total" (Expr.binary (Expr.num 1) BinOp.add (Expr.num 2))]) = true := by
  native_decide

example :
    (parseProgram "let count: num =\n3"
      ==
      some [Stmt.letDeclTyped "count" AnnTy.num (Expr.num 3)]) = true := by
  native_decide

example :
    (parseProgram "let x = 1~x =\nx + 2"
      ==
      some
        [ Stmt.letDecl "x" (Expr.num 1)
        , Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 2))
        ]) = true := by
  native_decide

example :
    (parseProgram "break\ncontinue\nreturn"
      ==
      some
        [ Stmt.break
        , Stmt.continue
        , Stmt.ret none
        ]) = true := by
  native_decide

example :
    (parseBlock (Lexer.tokenize "{ let x = 1~x = x + 1 }")
      ==
      some
        ( [ Stmt.letDecl "x" (Expr.num 1)
          , Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))
          ]
        , [Lexer.TokenKind.eof]
        )) = true := by
  native_decide

example :
    (parseProgram "if x < 3 { x = x + 1 } else { return x }"
      ==
      some
        [ Stmt.ifThenElse
            (Expr.binary (Expr.var "x") BinOp.lt (Expr.num 3))
            [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))]
            (some [Stmt.ret (some (Expr.var "x"))])
        ]) = true := by
  native_decide

example :
    (parseProgram "if true\n{ return 1 }"
      ==
      some
        [ Stmt.ifThenElse
            (Expr.bool true)
            [Stmt.ret (some (Expr.num 1))]
            none
        ]) = true := by
  native_decide

example :
    (parseProgram "if\ntrue { return 1 }"
      ==
      some
        [ Stmt.ifThenElse
            (Expr.bool true)
            [Stmt.ret (some (Expr.num 1))]
            none
        ]) = true := by
  native_decide

example :
    (parseProgram "while x < 3 { x = x + 1~continue }"
      ==
      some
        [ Stmt.while
            (Expr.binary (Expr.var "x") BinOp.lt (Expr.num 3))
            [ Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))
            , Stmt.continue
            ]
        ]) = true := by
  native_decide

example :
    (parseProgram "while\nx < 3 { x = x + 1~continue }"
      ==
      some
        [ Stmt.while
            (Expr.binary (Expr.var "x") BinOp.lt (Expr.num 3))
            [ Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))
            , Stmt.continue
            ]
        ]) = true := by
  native_decide

example :
    (parseProgram "for i in 0..3 { x = x + i }"
      ==
      some
        [ Stmt.forRange
            "i"
            0
            3
            [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.var "i"))]
        ]) = true := by
  native_decide

example :
    (parseProgram "for i in -2..2 { x = x + i }"
      ==
      some
        [ Stmt.forRange
            "i"
            (-2)
            2
            [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.var "i"))]
        ]) = true := by
  native_decide

example :
    (parseProgram "seal until x == 3 { x = x + 1 }"
      ==
      some
        [ Stmt.seal
            (some (Expr.binary (Expr.var "x") BinOp.eq (Expr.num 3)))
            [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))]
        ]) = true := by
  native_decide

example :
    (parseProgram "seal until\nx == 3 { x = x + 1 }"
      ==
      some
        [ Stmt.seal
            (some (Expr.binary (Expr.var "x") BinOp.eq (Expr.num 3)))
            [Stmt.assign "x" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))]
        ]) = true := by
  native_decide

example :
    (parseProgram "seal { break }"
      ==
      some
        [ Stmt.seal none [Stmt.break]
        ]) = true := by
  native_decide

example :
    (parseProgram "fn add(a, b) { return a + b }~let y = add(1, 2)"
      ==
      some
        [ Stmt.fnDecl
            "add"
            ["a", "b"]
            [Stmt.ret (some (Expr.binary (Expr.var "a") BinOp.add (Expr.var "b")))]
        , Stmt.letDecl
            "y"
            (Expr.call "add" [Expr.num 1, Expr.num 2])
        ]) = true := by
  native_decide

example :
    (parseProgram "fn add\n(a, b) { return a + b }"
      ==
      some
        [ Stmt.fnDecl
            "add"
            ["a", "b"]
            [Stmt.ret (some (Expr.binary (Expr.var "a") BinOp.add (Expr.var "b")))]
        ]) = true := by
  native_decide

example :
    (parseProgram "fn add(a,\nb,) { return a + b }"
      ==
      some
        [ Stmt.fnDecl
            "add"
            ["a", "b"]
            [Stmt.ret (some (Expr.binary (Expr.var "a") BinOp.add (Expr.var "b")))]
        ]) = true := by
  native_decide

example :
    (parseProgram "fn id(x: num) { return x }"
      ==
      some
        [ Stmt.fnDeclTyped
            "id"
            [("x", AnnTy.num)]
            [Stmt.ret (some (Expr.var "x"))]
        ]) = true := by
  native_decide

example :
    (parseProgram "fn id(x: num): num { return x }"
      ==
        some
        [ Stmt.fnDeclTypedReturn
            "id"
            [("x", AnnTy.num)]
            AnnTy.num
            [Stmt.ret (some (Expr.var "x"))]
        ]) = true := by
  native_decide

example :
    (parseProgram "fn add(x: num,\ny: num,): num { return x + y }"
      ==
      some
        [ Stmt.fnDeclTypedReturn
            "add"
            [("x", AnnTy.num), ("y", AnnTy.num)]
            AnnTy.num
            [Stmt.ret (some (Expr.binary (Expr.var "x") BinOp.add (Expr.var "y")))]
        ]) = true := by
  native_decide

example :
    (parseProgram "fn id(x): num { return x }"
      ==
      some
        [ Stmt.fnDeclReturn
            "id"
            ["x"]
            AnnTy.num
            [Stmt.ret (some (Expr.var "x"))]
          ]) = true := by
  native_decide

example :
    (parseProgram "fn id\n(x: num): num { return x }"
      ==
      some
        [ Stmt.fnDeclTypedReturn
            "id"
            [("x", AnnTy.num)]
            AnnTy.num
            [Stmt.ret (some (Expr.var "x"))]
          ]) = true := by
  native_decide

example :
    (parseProgram "fn first(xs: list[num]): num { return xs[0] }"
      ==
      some
        [ Stmt.fnDeclTypedReturn
            "first"
            [("xs", AnnTy.list AnnTy.num)]
            AnnTy.num
            [Stmt.ret (some (Expr.index (Expr.var "xs") (Expr.num 0)))]
        ]) = true := by
  native_decide

example :
    (parseProgram "let xs: list[\nnum\n] = [1]"
      ==
      some
        [ Stmt.letDeclTyped
            "xs"
            (AnnTy.list AnnTy.num)
            (Expr.list [Expr.num 1])
        ]) = true := by
  native_decide

example :
    (parseProgram "fn first(xs: list[\nnum\n]): list[\nnum\n] { return xs }"
      ==
      some
        [ Stmt.fnDeclTypedReturn
            "first"
            [("xs", AnnTy.list AnnTy.num)]
            (AnnTy.list AnnTy.num)
            [Stmt.ret (some (Expr.var "xs"))]
        ]) = true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "let x = 1 +")
      (ParseError.expected ParseContext.expression (some Lexer.TokenKind.plus))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "let x = self +")
      (ParseError.expected ParseContext.expression (some Lexer.TokenKind.plus))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "let x = 1.5 +")
      (ParseError.expected ParseContext.expression (some Lexer.TokenKind.plus))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "let x = embed")
      (ParseError.expected ParseContext.expression (some Lexer.TokenKind.embed))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "let x = convergence")
      (ParseError.expected ParseContext.expression (some Lexer.TokenKind.convergence))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "for i in 0 { break }")
      (ParseError.expected ParseContext.range (some Lexer.TokenKind.for))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "for i in 0..3")
      (ParseError.expected ParseContext.block (some Lexer.TokenKind.for))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "for i in -2..2")
      (ParseError.expected ParseContext.block (some Lexer.TokenKind.for))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "if { break }")
      (ParseError.expected ParseContext.expression (some Lexer.TokenKind.lBrace))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "if 1 + { break }")
      (ParseError.expected ParseContext.expression (some Lexer.TokenKind.plus))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "else { break }")
      (ParseError.expected ParseContext.ifStatement (some Lexer.TokenKind.else_))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "while { break }")
      (ParseError.expected ParseContext.expression (some Lexer.TokenKind.lBrace))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "while 1 + { break }")
      (ParseError.expected ParseContext.expression (some Lexer.TokenKind.plus))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "seal until { break }")
      (ParseError.expected ParseContext.expression (some Lexer.TokenKind.lBrace))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "seal until 1 + { break }")
      (ParseError.expected ParseContext.expression (some Lexer.TokenKind.plus))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "fn add(a, { return a }")
      (ParseError.expected ParseContext.params (some Lexer.TokenKind.fn))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "fn add(a, b)")
      (ParseError.expected ParseContext.block (some Lexer.TokenKind.fn))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "let count: = 1")
      (ParseError.expected ParseContext.typeAnnotation (some Lexer.TokenKind.equals))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "let xs: list[ = [1]")
      (ParseError.expected ParseContext.typeAnnotation (some Lexer.TokenKind.equals))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "let xs: list[\n = [1]")
      (ParseError.expected ParseContext.typeAnnotation (some Lexer.TokenKind.equals))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "fn id(x: num): { return x }")
      (ParseError.expected ParseContext.typeAnnotation (some Lexer.TokenKind.lBrace))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "fn id(x): { return x }")
      (ParseError.expected ParseContext.typeAnnotation (some Lexer.TokenKind.lBrace))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "fn id(x:) { return x }")
      (ParseError.expected ParseContext.typeAnnotation (some Lexer.TokenKind.rParen))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "fn id(x: list[) { return x }")
      (ParseError.expected ParseContext.typeAnnotation (some Lexer.TokenKind.rParen))
      =
      true := by
  native_decide

example :
    parseFailedAs
      (parseProgramDetailed "fn id(x): list[ { return x }")
      (ParseError.expected ParseContext.typeAnnotation (some Lexer.TokenKind.lBrace))
      =
      true := by
  native_decide

example :
    (parseProgram "let x = 1; let y = x + 1"
      ==
      some
        [ Stmt.letDecl "x" (Expr.num 1)
        , Stmt.letDecl "y" (Expr.binary (Expr.var "x") BinOp.add (Expr.num 1))
        ]) := by
  native_decide

example :
    (parseProgram "let xs = [1,\n2,]"
      ==
      some [Stmt.letDecl "xs" (Expr.list [Expr.num 1, Expr.num 2])]) := by
  native_decide

end Parser
end Aether
