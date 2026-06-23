namespace Aether
namespace Lexer

inductive TokenKind where
  | manifold
  | block
  | regress
  | render
  | embed
  | until
  | escalate
  | convergence
  | true
  | false
  | class
  | new
  | self
  | import
  | from
  | as
  | seal
  | for
  | while
  | if_
  | else_
  | fn
  | return
  | break
  | continue
  | in_
  | let_
  | dim
  | tau
  | model
  | color
  | axis
  | project
  | cluster
  | center
  | spread
  | format
  | output
  | identifier : String -> TokenKind
  | number : Int -> TokenKind
  | float : Int -> Int -> TokenKind
  | stringLit : String -> TokenKind
  | equals
  | colon
  | comma
  | dot
  | lBrace
  | rBrace
  | lBracket
  | rBracket
  | lParen
  | rParen
  | plus
  | minus
  | star
  | slash
  | percent
  | less
  | greater
  | lessEq
  | greaterEq
  | eqEq
  | notEq
  | and
  | or
  | not
  | dotDot
  | tilde
  | semicolon
  | newline
  | eof
  | error : String -> TokenKind
  deriving Repr, BEq, DecidableEq

structure SourcePos where
  line : Nat
  column : Nat
  deriving Repr, BEq, DecidableEq

structure SourceSpan where
  start : SourcePos
  stop : SourcePos
  deriving Repr, BEq, DecidableEq

structure LocatedToken where
  kind : TokenKind
  start : SourcePos
  stop : SourcePos
  deriving Repr, BEq, DecidableEq

def initialPos : SourcePos := { line := 1, column := 1 }

def advanceNewline (pos : SourcePos) : SourcePos :=
  { line := pos.line + 1, column := 1 }

def advanceChar (pos : SourcePos) (c : Char) : SourcePos :=
  if c == '\n' then
    advanceNewline pos
  else
    { pos with column := pos.column + 1 }

def advanceMany : SourcePos -> List Char -> Nat -> SourcePos
  | pos, _, 0 => pos
  | pos, [], _ + 1 => pos
  | pos, c :: rest, fuel + 1 => advanceMany (advanceChar pos c) rest fuel

def mkLocated (kind : TokenKind) (start : SourcePos) (chars : List Char) (consumed : Nat) :
    LocatedToken :=
  { kind := kind, start := start, stop := advanceMany start chars consumed }

def LocatedToken.span (token : LocatedToken) : SourceSpan :=
  { start := token.start, stop := token.stop }

def isAsciiDigit (c : Char) : Bool :=
  '0'.toNat <= c.toNat && c.toNat <= '9'.toNat

def isAsciiAlpha (c : Char) : Bool :=
  ('a'.toNat <= c.toNat && c.toNat <= 'z'.toNat)
    || ('A'.toNat <= c.toNat && c.toNat <= 'Z'.toNat)

def isIdentStart (c : Char) : Bool :=
  isAsciiAlpha c || c == '_'

def isIdentContinue (c : Char) : Bool :=
  isIdentStart c || isAsciiDigit c

def digitValue? (c : Char) : Option Nat :=
  if isAsciiDigit c then
    some (c.toNat - '0'.toNat)
  else
    none

def parseNatDigits : List Char -> Nat
  | [] => 0
  | c :: rest =>
      match digitValue? c with
      | some digit => digit + 10 * parseNatDigits rest
      | none => parseNatDigits rest

def parseNatDigitsLeft (digits : List Char) : Nat :=
  digits.foldl
    (fun acc c =>
      match digitValue? c with
      | some digit => acc * 10 + digit
      | none => acc)
    0

def readWhile (p : Char -> Bool) : List Char -> List Char -> String × List Char
  | [], acc => (String.ofList acc.reverse, [])
  | c :: rest, acc =>
      if p c then
        readWhile p rest (c :: acc)
      else
        (String.ofList acc.reverse, c :: rest)

def readDigits : List Char -> List Char -> List Char × List Char
  | [], acc => (acc.reverse, [])
  | c :: rest, acc =>
      if isAsciiDigit c then
        readDigits rest (c :: acc)
      else
        (acc.reverse, c :: rest)

def readFracDigits : Nat -> List Char -> Nat -> Nat -> Nat × Nat × List Char
  | 0, chars, acc, digits => (acc, digits, chars)
  | _ + 1, [], acc, digits => (acc, digits, [])
  | fuel + 1, c :: rest, acc, digits =>
      match digitValue? c with
      | some digit => readFracDigits fuel rest (acc * 10 + digit) (digits + 1)
      | none => (acc, digits, c :: rest)

def padFracToMicros : Nat -> Nat -> Nat
  | 0, frac => frac
  | fuel + 1, frac => padFracToMicros fuel (frac * 10)

def keywordOrIdent (name : String) : TokenKind :=
  match name with
  | "manifold" => TokenKind.manifold
  | "block" => TokenKind.block
  | "regress" => TokenKind.regress
  | "render" => TokenKind.render
  | "embed" => TokenKind.embed
  | "until" => TokenKind.until
  | "escalate" => TokenKind.escalate
  | "convergence" => TokenKind.convergence
  | "true" => TokenKind.true
  | "false" => TokenKind.false
  | "class" => TokenKind.class
  | "new" => TokenKind.new
  | "self" => TokenKind.self
  | "import" => TokenKind.import
  | "from" => TokenKind.from
  | "as" => TokenKind.as
  | "seal" => TokenKind.seal
  | "for" => TokenKind.for
  | "while" => TokenKind.while
  | "if" => TokenKind.if_
  | "else" => TokenKind.else_
  | "fn" => TokenKind.fn
  | "return" => TokenKind.return
  | "break" => TokenKind.break
  | "continue" => TokenKind.continue
  | "in" => TokenKind.in_
  | "let" => TokenKind.let_
  | "dim" => TokenKind.dim
  | "tau" => TokenKind.tau
  | "model" => TokenKind.model
  | "color" => TokenKind.color
  | "axis" => TokenKind.axis
  | "project" => TokenKind.project
  | "cluster" => TokenKind.cluster
  | "center" => TokenKind.center
  | "spread" => TokenKind.spread
  | "format" => TokenKind.format
  | "output" => TokenKind.output
  | _ => TokenKind.identifier name

def skipWhitespace : List Char -> List Char
  | [] => []
  | c :: rest =>
      if c == ' ' || c == '\t' then
        skipWhitespace rest
      else
        c :: rest

def skipWhitespaceLocated : List Char -> SourcePos -> List Char × SourcePos
  | [], pos => ([], pos)
  | c :: rest, pos =>
      if c == ' ' || c == '\t' then
        skipWhitespaceLocated rest (advanceChar pos c)
      else
        (c :: rest, pos)

def skipComment : List Char -> List Char
  | [] => []
  | c :: rest =>
      if c == '\n' || c == '\r' then
        c :: rest
      else
        skipComment rest

def skipCommentLocated : List Char -> SourcePos -> List Char × SourcePos
  | [], pos => ([], pos)
  | c :: rest, pos =>
      if c == '\n' || c == '\r' then
        (c :: rest, pos)
      else
        skipCommentLocated rest (advanceChar pos c)

partial def skipBlockCommentDepth (depth : Nat) : List Char -> Option (List Char)
  | [] => none
  | '/' :: '*' :: rest => skipBlockCommentDepth (depth + 1) rest
  | '*' :: '/' :: rest =>
      if depth == 1 then
        some rest
      else
        skipBlockCommentDepth (depth - 1) rest
  | _ :: rest => skipBlockCommentDepth depth rest

def skipBlockComment (chars : List Char) : Option (List Char) :=
  skipBlockCommentDepth 1 chars

partial def skipBlockCommentLocatedDepth
    (depth : Nat) : List Char -> SourcePos -> Except SourcePos (List Char × SourcePos)
  | [], pos => Except.error pos
  | '/' :: '*' :: rest, pos =>
      skipBlockCommentLocatedDepth (depth + 1) rest (advanceChar (advanceChar pos '/') '*')
  | '*' :: '/' :: rest, pos =>
      let nextPos := advanceChar (advanceChar pos '*') '/'
      if depth == 1 then
        Except.ok (rest, nextPos)
      else
        skipBlockCommentLocatedDepth (depth - 1) rest nextPos
  | '\r' :: '\n' :: rest, pos => skipBlockCommentLocatedDepth depth rest (advanceNewline pos)
  | '\r' :: rest, pos => skipBlockCommentLocatedDepth depth rest (advanceNewline pos)
  | c :: rest, pos => skipBlockCommentLocatedDepth depth rest (advanceChar pos c)

def skipBlockCommentLocated (chars : List Char) (pos : SourcePos) :
    Except SourcePos (List Char × SourcePos) :=
  skipBlockCommentLocatedDepth 1 chars pos

def readString : List Char -> List Char -> TokenKind × List Char
  | [], _ => (TokenKind.error "unexpected EOF in string", [])
  | '\\' :: escaped :: rest, acc =>
      match escaped with
      | '"' => readString rest ('"' :: acc)
      | '\\' => readString rest ('\\' :: acc)
      | 'n' => readString rest ('\n' :: acc)
      | 'r' => readString rest ('\r' :: acc)
      | 't' => readString rest ('\t' :: acc)
      | other => (TokenKind.error ("invalid escape: \\" ++ String.ofList [other]), rest)
  | ['\\'], _ => (TokenKind.error "unexpected EOF in string", [])
  | c :: rest, acc =>
      if c == '"' then
        (TokenKind.stringLit (String.ofList acc.reverse), rest)
      else if c == '\n' || c == '\r' then
        (TokenKind.error "unterminated string", c :: rest)
      else
        readString rest (c :: acc)

def readStringLocated
    (openingStart : SourcePos) :
    List Char -> SourcePos -> List Char -> LocatedToken × List Char × SourcePos
  | [], pos, _ =>
      ( { kind := TokenKind.error "unexpected EOF in string"
        , start := openingStart
        , stop := pos }
      , []
      , pos )
  | '\\' :: escaped :: rest, pos, acc =>
      let stop := advanceMany pos ['\\', escaped] 2
      match escaped with
      | '"' => readStringLocated openingStart rest stop ('"' :: acc)
      | '\\' => readStringLocated openingStart rest stop ('\\' :: acc)
      | 'n' => readStringLocated openingStart rest stop ('\n' :: acc)
      | 'r' => readStringLocated openingStart rest stop ('\r' :: acc)
      | 't' => readStringLocated openingStart rest stop ('\t' :: acc)
      | other =>
          ( { kind := TokenKind.error ("invalid escape: \\" ++ String.ofList [other])
            , start := pos
            , stop := stop }
          , rest
          , stop )
  | ['\\'], pos, _ =>
      ( { kind := TokenKind.error "unexpected EOF in string"
        , start := openingStart
        , stop := advanceChar pos '\\' }
      , []
      , advanceChar pos '\\' )
  | c :: rest, pos, acc =>
      if c == '"' then
        let stop := advanceChar pos c
        ( { kind := TokenKind.stringLit (String.ofList acc.reverse)
          , start := openingStart
          , stop := stop }
        , rest
        , stop )
      else if c == '\n' || c == '\r' then
        ( { kind := TokenKind.error "unterminated string"
          , start := openingStart
          , stop := pos }
        , c :: rest
        , pos )
      else
        readStringLocated openingStart rest (advanceChar pos c) (c :: acc)

def readNumber (first : Char) (rest : List Char) : TokenKind × List Char :=
  let (digits, afterDigits) := readDigits rest [first]
  let intPart := parseNatDigitsLeft digits
  match afterDigits with
  | '.' :: next :: afterDot =>
      if isAsciiDigit next then
        let (frac, fracDigits, remaining) := readFracDigits 6 (next :: afterDot) 0 0
        let fracMicros := padFracToMicros (6 - fracDigits) frac
        (TokenKind.float (Int.ofNat intPart) (Int.ofNat fracMicros), remaining)
      else
        (TokenKind.number (Int.ofNat intPart), afterDigits)
  | _ => (TokenKind.number (Int.ofNat intPart), afterDigits)

def scanFuel : Nat -> List Char -> List TokenKind
  | 0, _ => [TokenKind.error "lexer fuel exhausted", TokenKind.eof]
  | fuel + 1, chars =>
      match skipWhitespace chars with
      | [] => [TokenKind.eof]
      | '/' :: '*' :: rest =>
          match skipBlockComment rest with
          | some remaining => scanFuel fuel remaining
          | none => [TokenKind.error "unterminated block comment", TokenKind.eof]
      | '/' :: '/' :: rest => scanFuel fuel (skipComment rest)
      | '/' :: rest => TokenKind.slash :: scanFuel fuel rest
      | '🦭' :: rest => TokenKind.seal :: scanFuel fuel rest
      | '~' :: rest => TokenKind.tilde :: scanFuel fuel rest
      | ';' :: rest => TokenKind.semicolon :: scanFuel fuel rest
      | '=' :: '=' :: rest => TokenKind.eqEq :: scanFuel fuel rest
      | '=' :: rest => TokenKind.equals :: scanFuel fuel rest
      | '!' :: '=' :: rest => TokenKind.notEq :: scanFuel fuel rest
      | '!' :: rest => TokenKind.not :: scanFuel fuel rest
      | '<' :: '=' :: rest => TokenKind.lessEq :: scanFuel fuel rest
      | '<' :: rest => TokenKind.less :: scanFuel fuel rest
      | '>' :: '=' :: rest => TokenKind.greaterEq :: scanFuel fuel rest
      | '>' :: rest => TokenKind.greater :: scanFuel fuel rest
      | '&' :: '&' :: rest => TokenKind.and :: scanFuel fuel rest
      | '|' :: '|' :: rest => TokenKind.or :: scanFuel fuel rest
      | '.' :: '.' :: rest => TokenKind.dotDot :: scanFuel fuel rest
      | ':' :: rest => TokenKind.colon :: scanFuel fuel rest
      | ',' :: rest => TokenKind.comma :: scanFuel fuel rest
      | '.' :: rest => TokenKind.dot :: scanFuel fuel rest
      | '{' :: rest => TokenKind.lBrace :: scanFuel fuel rest
      | '}' :: rest => TokenKind.rBrace :: scanFuel fuel rest
      | '[' :: rest => TokenKind.lBracket :: scanFuel fuel rest
      | ']' :: rest => TokenKind.rBracket :: scanFuel fuel rest
      | '(' :: rest => TokenKind.lParen :: scanFuel fuel rest
      | ')' :: rest => TokenKind.rParen :: scanFuel fuel rest
      | '\r' :: '\n' :: rest => TokenKind.newline :: scanFuel fuel rest
      | '\r' :: rest => TokenKind.newline :: scanFuel fuel rest
      | '\n' :: rest => TokenKind.newline :: scanFuel fuel rest
      | '+' :: rest => TokenKind.plus :: scanFuel fuel rest
      | '-' :: rest => TokenKind.minus :: scanFuel fuel rest
      | '*' :: rest => TokenKind.star :: scanFuel fuel rest
      | '%' :: rest => TokenKind.percent :: scanFuel fuel rest
      | '"' :: rest =>
          let (token, remaining) := readString rest []
          token :: scanFuel fuel remaining
      | c :: rest =>
          if isAsciiDigit c then
            let (token, remaining) := readNumber c rest
            token :: scanFuel fuel remaining
          else if isIdentStart c then
            let (name, remaining) := readWhile isIdentContinue rest [c]
            keywordOrIdent name :: scanFuel fuel remaining
          else
            TokenKind.error ("unexpected char: " ++ String.ofList [c]) :: scanFuel fuel rest

def tokenize (source : String) : List TokenKind :=
  scanFuel (source.toList.length + 1) source.toList

def scanLocatedFuel : Nat -> List Char -> SourcePos -> List LocatedToken
  | 0, _, pos =>
      [ { kind := TokenKind.error "lexer fuel exhausted", start := pos, stop := pos }
      , { kind := TokenKind.eof, start := pos, stop := pos }
      ]
  | fuel + 1, chars, pos =>
      let (chars, pos) := skipWhitespaceLocated chars pos
      match chars with
      | [] => [{ kind := TokenKind.eof, start := pos, stop := pos }]
      | '/' :: '*' :: rest =>
          let commentStart := pos
          let bodyStart := advanceChar (advanceChar pos '/') '*'
          match skipBlockCommentLocated rest bodyStart with
          | Except.ok (remaining, nextPos) => scanLocatedFuel fuel remaining nextPos
          | Except.error stop =>
              [ { kind := TokenKind.error "unterminated block comment", start := commentStart, stop := stop }
              , { kind := TokenKind.eof, start := stop, stop := stop }
              ]
      | '/' :: '/' :: rest =>
          let (remaining, nextPos) := skipCommentLocated rest (advanceMany pos chars 2)
          scanLocatedFuel fuel remaining nextPos
      | '/' :: rest => mkLocated TokenKind.slash pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos '/')
      | '🦭' :: rest => mkLocated TokenKind.seal pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos '🦭')
      | '~' :: rest => mkLocated TokenKind.tilde pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos '~')
      | ';' :: rest => mkLocated TokenKind.semicolon pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos ';')
      | '=' :: '=' :: rest => mkLocated TokenKind.eqEq pos chars 2 :: scanLocatedFuel fuel rest (advanceMany pos chars 2)
      | '=' :: rest => mkLocated TokenKind.equals pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos '=')
      | '!' :: '=' :: rest => mkLocated TokenKind.notEq pos chars 2 :: scanLocatedFuel fuel rest (advanceMany pos chars 2)
      | '!' :: rest => mkLocated TokenKind.not pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos '!')
      | '<' :: '=' :: rest => mkLocated TokenKind.lessEq pos chars 2 :: scanLocatedFuel fuel rest (advanceMany pos chars 2)
      | '<' :: rest => mkLocated TokenKind.less pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos '<')
      | '>' :: '=' :: rest => mkLocated TokenKind.greaterEq pos chars 2 :: scanLocatedFuel fuel rest (advanceMany pos chars 2)
      | '>' :: rest => mkLocated TokenKind.greater pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos '>')
      | '&' :: '&' :: rest => mkLocated TokenKind.and pos chars 2 :: scanLocatedFuel fuel rest (advanceMany pos chars 2)
      | '|' :: '|' :: rest => mkLocated TokenKind.or pos chars 2 :: scanLocatedFuel fuel rest (advanceMany pos chars 2)
      | '.' :: '.' :: rest => mkLocated TokenKind.dotDot pos chars 2 :: scanLocatedFuel fuel rest (advanceMany pos chars 2)
      | ':' :: rest => mkLocated TokenKind.colon pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos ':')
      | ',' :: rest => mkLocated TokenKind.comma pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos ',')
      | '.' :: rest => mkLocated TokenKind.dot pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos '.')
      | '{' :: rest => mkLocated TokenKind.lBrace pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos '{')
      | '}' :: rest => mkLocated TokenKind.rBrace pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos '}')
      | '[' :: rest => mkLocated TokenKind.lBracket pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos '[')
      | ']' :: rest => mkLocated TokenKind.rBracket pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos ']')
      | '(' :: rest => mkLocated TokenKind.lParen pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos '(')
      | ')' :: rest => mkLocated TokenKind.rParen pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos ')')
      | '\r' :: '\n' :: rest =>
          { kind := TokenKind.newline, start := pos, stop := advanceNewline pos }
            :: scanLocatedFuel fuel rest (advanceNewline pos)
      | '\r' :: rest =>
          { kind := TokenKind.newline, start := pos, stop := advanceNewline pos }
            :: scanLocatedFuel fuel rest (advanceNewline pos)
      | '\n' :: rest => mkLocated TokenKind.newline pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos '\n')
      | '+' :: rest => mkLocated TokenKind.plus pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos '+')
      | '-' :: rest => mkLocated TokenKind.minus pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos '-')
      | '*' :: rest => mkLocated TokenKind.star pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos '*')
      | '%' :: rest => mkLocated TokenKind.percent pos chars 1 :: scanLocatedFuel fuel rest (advanceChar pos '%')
      | '"' :: rest =>
          let (token, remaining, nextPos) := readStringLocated pos rest (advanceChar pos '"') []
          token :: scanLocatedFuel fuel remaining nextPos
      | c :: rest =>
          if isAsciiDigit c then
            let (token, remaining) := readNumber c rest
            let consumed := chars.length - remaining.length
            mkLocated token pos chars consumed :: scanLocatedFuel fuel remaining (advanceMany pos chars consumed)
          else if isIdentStart c then
            let (name, remaining) := readWhile isIdentContinue rest [c]
            let consumed := chars.length - remaining.length
            mkLocated (keywordOrIdent name) pos chars consumed :: scanLocatedFuel fuel remaining (advanceMany pos chars consumed)
          else
            mkLocated (TokenKind.error ("unexpected char: " ++ String.ofList [c])) pos chars 1
              :: scanLocatedFuel fuel rest (advanceChar pos c)

def tokenizeLocated (source : String) : List LocatedToken :=
  scanLocatedFuel (source.toList.length + 1) source.toList initialPos

def tokenKinds (tokens : List LocatedToken) : List TokenKind :=
  tokens.map (fun token => token.kind)

example :
    tokenize "let x = 1..10"
      =
      [ TokenKind.let_
      , TokenKind.identifier "x"
      , TokenKind.equals
      , TokenKind.number 1
      , TokenKind.dotDot
      , TokenKind.number 10
      , TokenKind.eof
      ] := by
  native_decide

example :
    tokenize "1.5 1..10"
      =
      [ TokenKind.float 1 500000
      , TokenKind.number 1
      , TokenKind.dotDot
      , TokenKind.number 10
      , TokenKind.eof
      ] := by
  native_decide

example :
    tokenize "if true { x = x + 1 } // done\n~"
      =
      [ TokenKind.if_
      , TokenKind.true
      , TokenKind.lBrace
      , TokenKind.identifier "x"
      , TokenKind.equals
      , TokenKind.identifier "x"
      , TokenKind.plus
      , TokenKind.number 1
      , TokenKind.rBrace
      , TokenKind.newline
      , TokenKind.tilde
      , TokenKind.eof
      ] := by
  native_decide

example :
    tokenize "let x = 1\rlet y = 2"
      =
      [ TokenKind.let_
      , TokenKind.identifier "x"
      , TokenKind.equals
      , TokenKind.number 1
      , TokenKind.newline
      , TokenKind.let_
      , TokenKind.identifier "y"
      , TokenKind.equals
      , TokenKind.number 2
      , TokenKind.eof
      ] := by
  native_decide

example :
    tokenize "let x = 1\r\nlet y = 2"
      =
      [ TokenKind.let_
      , TokenKind.identifier "x"
      , TokenKind.equals
      , TokenKind.number 1
      , TokenKind.newline
      , TokenKind.let_
      , TokenKind.identifier "y"
      , TokenKind.equals
      , TokenKind.number 2
      , TokenKind.eof
      ] := by
  native_decide

example :
    tokenize "let x = 1; let y = 2"
      =
      [ TokenKind.let_
      , TokenKind.identifier "x"
      , TokenKind.equals
      , TokenKind.number 1
      , TokenKind.semicolon
      , TokenKind.let_
      , TokenKind.identifier "y"
      , TokenKind.equals
      , TokenKind.number 2
      , TokenKind.eof
      ] := by
  native_decide

example :
    tokenize "let x = 1 /* ignore tokens + true */ ~"
      =
      [ TokenKind.let_
      , TokenKind.identifier "x"
      , TokenKind.equals
      , TokenKind.number 1
      , TokenKind.tilde
      , TokenKind.eof
      ] := by
  native_decide

example :
    tokenize "let x = 1 /* outer /* inner */ still comment */ ~"
      =
      [ TokenKind.let_
      , TokenKind.identifier "x"
      , TokenKind.equals
      , TokenKind.number 1
      , TokenKind.tilde
      , TokenKind.eof
      ] := by
  native_decide

example :
    tokenize "let x = 1 /* never closes"
      =
      [ TokenKind.let_
      , TokenKind.identifier "x"
      , TokenKind.equals
      , TokenKind.number 1
      , TokenKind.error "unterminated block comment"
      , TokenKind.eof
      ] := by
  native_decide

example :
    tokenize "let x = 1 /* outer /* inner */"
      =
      [ TokenKind.let_
      , TokenKind.identifier "x"
      , TokenKind.equals
      , TokenKind.number 1
      , TokenKind.error "unterminated block comment"
      , TokenKind.eof
      ] := by
  native_decide

example :
    tokenize "🦭 until x == 3 { break }"
      =
      [ TokenKind.seal
      , TokenKind.until
      , TokenKind.identifier "x"
      , TokenKind.eqEq
      , TokenKind.number 3
      , TokenKind.lBrace
      , TokenKind.break
      , TokenKind.rBrace
      , TokenKind.eof
      ] := by
  native_decide

example :
    tokenize "\"open\" \"bad\n"
      =
      [ TokenKind.stringLit "open"
      , TokenKind.error "unterminated string"
      , TokenKind.newline
      , TokenKind.eof
      ] := by
  native_decide

example :
    tokenize "\"say \\\"hi\\\"\" \"path\\\\to\""
      =
      [ TokenKind.stringLit "say \"hi\""
      , TokenKind.stringLit "path\\to"
      , TokenKind.eof
      ] := by
  native_decide

example :
    tokenize "\"line\\nnext\" \"col\\tvalue\""
      =
      [ TokenKind.stringLit "line\nnext"
      , TokenKind.stringLit "col\tvalue"
      , TokenKind.eof
      ] := by
  native_decide

example :
    tokenize "\"row\\rnext\""
      =
      [ TokenKind.stringLit "row\rnext"
      , TokenKind.eof
      ] := by
  native_decide

example :
    tokenize "\"bad\\q\""
      =
      [ TokenKind.error "invalid escape: \\q"
      , TokenKind.error "unexpected EOF in string"
      , TokenKind.eof
      ] := by
  native_decide

example :
    tokenizeLocated "let x = 1\n  @"
      =
      [ { kind := TokenKind.let_, start := { line := 1, column := 1 }, stop := { line := 1, column := 4 } }
      , { kind := TokenKind.identifier "x", start := { line := 1, column := 5 }, stop := { line := 1, column := 6 } }
      , { kind := TokenKind.equals, start := { line := 1, column := 7 }, stop := { line := 1, column := 8 } }
      , { kind := TokenKind.number 1, start := { line := 1, column := 9 }, stop := { line := 1, column := 10 } }
      , { kind := TokenKind.newline, start := { line := 1, column := 10 }, stop := { line := 2, column := 1 } }
      , { kind := TokenKind.error "unexpected char: @", start := { line := 2, column := 3 }, stop := { line := 2, column := 4 } }
      , { kind := TokenKind.eof, start := { line := 2, column := 4 }, stop := { line := 2, column := 4 } }
      ] := by
  native_decide

example :
    tokenizeLocated "let x = 1\r  @"
      =
      [ { kind := TokenKind.let_, start := { line := 1, column := 1 }, stop := { line := 1, column := 4 } }
      , { kind := TokenKind.identifier "x", start := { line := 1, column := 5 }, stop := { line := 1, column := 6 } }
      , { kind := TokenKind.equals, start := { line := 1, column := 7 }, stop := { line := 1, column := 8 } }
      , { kind := TokenKind.number 1, start := { line := 1, column := 9 }, stop := { line := 1, column := 10 } }
      , { kind := TokenKind.newline, start := { line := 1, column := 10 }, stop := { line := 2, column := 1 } }
      , { kind := TokenKind.error "unexpected char: @", start := { line := 2, column := 3 }, stop := { line := 2, column := 4 } }
      , { kind := TokenKind.eof, start := { line := 2, column := 4 }, stop := { line := 2, column := 4 } }
      ] := by
  native_decide

example :
    tokenizeLocated "let x = 1\r\n  @"
      =
      [ { kind := TokenKind.let_, start := { line := 1, column := 1 }, stop := { line := 1, column := 4 } }
      , { kind := TokenKind.identifier "x", start := { line := 1, column := 5 }, stop := { line := 1, column := 6 } }
      , { kind := TokenKind.equals, start := { line := 1, column := 7 }, stop := { line := 1, column := 8 } }
      , { kind := TokenKind.number 1, start := { line := 1, column := 9 }, stop := { line := 1, column := 10 } }
      , { kind := TokenKind.newline, start := { line := 1, column := 10 }, stop := { line := 2, column := 1 } }
      , { kind := TokenKind.error "unexpected char: @", start := { line := 2, column := 3 }, stop := { line := 2, column := 4 } }
      , { kind := TokenKind.eof, start := { line := 2, column := 4 }, stop := { line := 2, column := 4 } }
      ] := by
  native_decide

example :
    tokenizeLocated "let x = 1 /* one\n two */~"
      =
      [ { kind := TokenKind.let_, start := { line := 1, column := 1 }, stop := { line := 1, column := 4 } }
      , { kind := TokenKind.identifier "x", start := { line := 1, column := 5 }, stop := { line := 1, column := 6 } }
      , { kind := TokenKind.equals, start := { line := 1, column := 7 }, stop := { line := 1, column := 8 } }
      , { kind := TokenKind.number 1, start := { line := 1, column := 9 }, stop := { line := 1, column := 10 } }
      , { kind := TokenKind.tilde, start := { line := 2, column := 8 }, stop := { line := 2, column := 9 } }
      , { kind := TokenKind.eof, start := { line := 2, column := 9 }, stop := { line := 2, column := 9 } }
      ] := by
  native_decide

example :
    tokenizeLocated "let x = 1 /* never closes"
      =
      [ { kind := TokenKind.let_, start := { line := 1, column := 1 }, stop := { line := 1, column := 4 } }
      , { kind := TokenKind.identifier "x", start := { line := 1, column := 5 }, stop := { line := 1, column := 6 } }
      , { kind := TokenKind.equals, start := { line := 1, column := 7 }, stop := { line := 1, column := 8 } }
      , { kind := TokenKind.number 1, start := { line := 1, column := 9 }, stop := { line := 1, column := 10 } }
      , { kind := TokenKind.error "unterminated block comment", start := { line := 1, column := 11 }, stop := { line := 1, column := 26 } }
      , { kind := TokenKind.eof, start := { line := 1, column := 26 }, stop := { line := 1, column := 26 } }
      ] := by
  native_decide

example :
    tokenizeLocated "\"bad\n"
      =
      [ { kind := TokenKind.error "unterminated string", start := { line := 1, column := 1 }, stop := { line := 1, column := 5 } }
      , { kind := TokenKind.newline, start := { line := 1, column := 5 }, stop := { line := 2, column := 1 } }
      , { kind := TokenKind.eof, start := { line := 2, column := 1 }, stop := { line := 2, column := 1 } }
      ] := by
  native_decide

example :
    (tokenizeLocated "1..10").map LocatedToken.span
      =
      [ { start := { line := 1, column := 1 }, stop := { line := 1, column := 2 } }
      , { start := { line := 1, column := 2 }, stop := { line := 1, column := 4 } }
      , { start := { line := 1, column := 4 }, stop := { line := 1, column := 6 } }
      , { start := { line := 1, column := 6 }, stop := { line := 1, column := 6 } }
      ] := by
  native_decide

example :
    tokenizeLocated "let x = 1; let y = 2"
      =
      [ { kind := TokenKind.let_, start := { line := 1, column := 1 }, stop := { line := 1, column := 4 } }
      , { kind := TokenKind.identifier "x", start := { line := 1, column := 5 }, stop := { line := 1, column := 6 } }
      , { kind := TokenKind.equals, start := { line := 1, column := 7 }, stop := { line := 1, column := 8 } }
      , { kind := TokenKind.number 1, start := { line := 1, column := 9 }, stop := { line := 1, column := 10 } }
      , { kind := TokenKind.semicolon, start := { line := 1, column := 10 }, stop := { line := 1, column := 11 } }
      , { kind := TokenKind.let_, start := { line := 1, column := 12 }, stop := { line := 1, column := 15 } }
      , { kind := TokenKind.identifier "y", start := { line := 1, column := 16 }, stop := { line := 1, column := 17 } }
      , { kind := TokenKind.equals, start := { line := 1, column := 18 }, stop := { line := 1, column := 19 } }
      , { kind := TokenKind.number 2, start := { line := 1, column := 20 }, stop := { line := 1, column := 21 } }
      , { kind := TokenKind.eof, start := { line := 1, column := 21 }, stop := { line := 1, column := 21 } }
      ] := by
  native_decide

example :
    tokenizeLocated "🦭 until"
      =
      [ { kind := TokenKind.seal, start := { line := 1, column := 1 }, stop := { line := 1, column := 2 } }
      , { kind := TokenKind.until, start := { line := 1, column := 3 }, stop := { line := 1, column := 8 } }
      , { kind := TokenKind.eof, start := { line := 1, column := 8 }, stop := { line := 1, column := 8 } }
      ] := by
  native_decide

example :
    tokenKinds (tokenizeLocated "let x = 1..10")
      =
      tokenize "let x = 1..10" := by
  native_decide

example :
    tokenKinds (tokenizeLocated "if true { x = x + 1 } // done\n~")
      =
      tokenize "if true { x = x + 1 } // done\n~" := by
  native_decide

example :
    tokenKinds (tokenizeLocated "let x = 1 /* one\n two */~")
      =
      tokenize "let x = 1 /* one\n two */~" := by
  native_decide

example :
    tokenKinds (tokenizeLocated "let x = 1 /* outer /* inner */ still comment */ ~")
      =
      tokenize "let x = 1 /* outer /* inner */ still comment */ ~" := by
  native_decide

example :
    tokenKinds (tokenizeLocated "\"open\" \"bad\n")
      =
      tokenize "\"open\" \"bad\n" := by
  native_decide

example :
    tokenKinds (tokenizeLocated "\"line\\nnext\" \"bad\\q\"")
      =
      tokenize "\"line\\nnext\" \"bad\\q\"" := by
  native_decide

example :
    tokenKinds (tokenizeLocated "\"row\\rnext\"")
      =
      tokenize "\"row\\rnext\"" := by
  native_decide

example :
    tokenKinds (tokenizeLocated "🦭 until x == 3 { break }")
      =
      tokenize "🦭 until x == 3 { break }" := by
  native_decide

end Lexer
end Aether
