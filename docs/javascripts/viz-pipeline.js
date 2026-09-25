// viz-pipeline: language pipeline, workspace map, CLI paths, module contracts, index flow.
// Everything here mirrors crates/aether-lang (lexer.rs, parser.rs), crates/*/Cargo.toml,
// crates/aether-cli/src/main.rs and crates/aether-core (manifold.rs, persistence.rs).
(() => {
  // ---------- styles (pipeline blocks + site-wide mermaid palette) ----------
  const style = document.createElement("style");
  style.textContent = `
.ts-viz[data-viz^="pipe-"]{--pa:#2456dc;--pa2:#c2410c;--pg:#15803d;--ph:var(--hair2,#cfcbc1);--pbg:var(--ground2,#f3f1eb)}
[data-md-color-scheme="slate"] .ts-viz[data-viz^="pipe-"]{--pa:#8fb0ff;--pa2:#ffb86b;--pg:#6fd39a;--pbg:#262420}
.ts-viz[data-viz^="pipe-"] .p-mono{font-family:var(--md-code-font-family);font-size:.66rem}
.ts-viz[data-viz^="pipe-"] .p-stages{display:flex;flex-wrap:wrap;gap:.3rem;align-items:center;margin-bottom:.7rem;font-size:.64rem}
.ts-viz[data-viz^="pipe-"] .p-stage{padding:.2rem .55rem;border:1px solid var(--ph);border-radius:999px;color:var(--md-default-fg-color--light);transition:all .35s ease}
.ts-viz[data-viz^="pipe-"] .p-stage.on{border-color:var(--pa);color:var(--pa);background:color-mix(in srgb,var(--pa) 9%,transparent)}
.ts-viz[data-viz^="pipe-"] .p-stage.off{opacity:.45;text-decoration:line-through}
.ts-viz[data-viz^="pipe-"] .p-arrow{color:var(--md-default-fg-color--lighter)}
.ts-viz[data-viz^="pipe-"] textarea{width:100%;box-sizing:border-box;min-height:5.2rem;resize:vertical;font-family:var(--md-code-font-family);font-size:.68rem;line-height:1.5;padding:.55rem .65rem;border:1px solid var(--ph);border-radius:10px;background:var(--pbg);color:var(--md-default-fg-color)}
.ts-viz[data-viz^="pipe-"] textarea:focus-visible,.ts-viz[data-viz^="pipe-"] [tabindex]:focus-visible,.ts-viz[data-viz^="pipe-"] input:focus-visible{outline:2px solid var(--pa);outline-offset:2px}
.ts-viz[data-viz^="pipe-"] .p-label{font-size:.6rem;letter-spacing:.06em;text-transform:uppercase;color:var(--md-default-fg-color--light);margin:.7rem 0 .35rem}
.ts-viz[data-viz^="pipe-"] .p-toks{display:flex;flex-wrap:wrap;gap:.25rem;max-height:9.5rem;overflow:auto}
.ts-viz[data-viz^="pipe-"] .p-tok{font-family:var(--md-code-font-family);font-size:.6rem;padding:.12rem .4rem;border-radius:6px;border:1px solid var(--ph);background:var(--pbg);white-space:nowrap}
.ts-viz[data-viz^="pipe-"] .p-tok.kw{border-color:var(--pa);color:var(--pa)}
.ts-viz[data-viz^="pipe-"] .p-tok.lit{color:var(--pg)}
.ts-viz[data-viz^="pipe-"] .p-tok.sep{color:var(--md-default-fg-color--light);border-style:dashed}
.ts-viz[data-viz^="pipe-"] .p-tok.err{border-color:var(--pa2);color:var(--pa2)}
.ts-viz[data-viz^="pipe-"] .p-in{animation:pIn .35s ease both}
@keyframes pIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
@media (prefers-reduced-motion:reduce){.ts-viz[data-viz^="pipe-"] .p-in{animation:none}}
.ts-viz[data-viz^="pipe-"] .p-tree{font-family:var(--md-code-font-family);font-size:.62rem;overflow:auto;max-height:16rem;padding:.4rem .2rem;border:1px solid var(--ph);border-radius:10px}
.ts-viz[data-viz^="pipe-"] .p-tree ul{list-style:none;margin:0 0 0 .55rem;padding:0 0 0 .7rem;border-left:1px solid var(--ph)}
.ts-viz[data-viz^="pipe-"] .p-tree>ul{border-left:0;margin-left:0}
.ts-viz[data-viz^="pipe-"] .p-tree li{margin:.12rem 0;white-space:nowrap}
.ts-viz[data-viz^="pipe-"] .p-tree b{color:var(--pa);font-weight:600}
.ts-viz[data-viz^="pipe-"] .p-tree i{color:var(--md-default-fg-color--light);font-style:normal}
.ts-viz[data-viz^="pipe-"] .p-err{color:var(--pa2);font-size:.66rem;margin-top:.4rem}
.ts-viz[data-viz^="pipe-"] .p-note{font-size:.64rem;color:var(--md-default-fg-color--light);margin-top:.5rem}
.ts-viz[data-viz^="pipe-"] .p-panel{border:1px solid var(--ph);border-radius:10px;padding:.6rem .75rem;font-size:.66rem;margin-top:.6rem;background:var(--pbg)}
.ts-viz[data-viz^="pipe-"] .p-panel h4{margin:0 0 .3rem;font-size:.72rem}
.ts-viz[data-viz^="pipe-"] .p-panel code{font-size:.62rem}
.ts-viz[data-viz^="pipe-"] svg .edge{fill:none;stroke:var(--ph);stroke-width:1.2;transition:stroke .3s}
.ts-viz[data-viz^="pipe-"] svg .edge.hot{stroke:var(--pa);stroke-width:1.8;stroke-dasharray:5 4;animation:pDash 1s linear infinite}
@keyframes pDash{to{stroke-dashoffset:-9}}
@media (prefers-reduced-motion:reduce){.ts-viz[data-viz^="pipe-"] svg .edge.hot{animation:none}}
.ts-viz[data-viz^="pipe-"] svg .node rect{fill:var(--md-default-bg-color);stroke:var(--ph);transition:all .3s}
.ts-viz[data-viz^="pipe-"] svg .node text{fill:var(--md-default-fg-color);font-size:11px;font-family:var(--md-code-font-family)}
.ts-viz[data-viz^="pipe-"] svg .node{cursor:pointer;outline:none}
.ts-viz[data-viz^="pipe-"] svg .node.sel rect,.ts-viz[data-viz^="pipe-"] svg .node:focus-visible rect{stroke:var(--pa);stroke-width:2}
.ts-viz[data-viz^="pipe-"] svg .node.dim{opacity:.4}
.ts-viz[data-viz^="pipe-"] .p-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:.4rem}
.ts-viz[data-viz^="pipe-"] .p-check{display:flex;gap:.4rem;align-items:center;font-size:.66rem;padding:.35rem .5rem;border:1px solid var(--ph);border-radius:8px}
.ts-viz[data-viz^="pipe-"] .p-badge{display:inline-block;font-size:.58rem;padding:.05rem .45rem;border-radius:999px;border:1px solid currentColor;margin-left:.3rem}
.ts-viz[data-viz^="pipe-"] .p-badge.ok{color:var(--pg)}.ts-viz[data-viz^="pipe-"] .p-badge.gate{color:var(--pa2)}
.ts-viz[data-viz^="pipe-"] .ts-viz-controls button[aria-pressed="true"]{color:var(--pa);border-color:var(--pa)}
.ts-viz[data-viz^="pipe-"] .ts-viz-controls input[type=text]{font-family:var(--md-code-font-family);font-size:.66rem;padding:.25rem .45rem;border:1px solid var(--ph);border-radius:8px;background:var(--pbg);color:var(--md-default-fg-color);width:11rem}
/* Mermaid diagrams, site-wide: palette of the site */
:root>*{--md-mermaid-font-family:"Instrument Sans",var(--md-text-font-family),sans-serif;--md-mermaid-edge-color:#8a857a;--md-mermaid-node-bg-color:#fff;--md-mermaid-node-fg-color:#1c1b19;--md-mermaid-label-bg-color:#fbfaf7;--md-mermaid-label-fg-color:#1c1b19;--md-mermaid-sequence-actor-bg-color:#fff;--md-mermaid-sequence-actor-border-color:#2456dc}
[data-md-color-scheme="slate"]{--md-mermaid-edge-color:#8a857a;--md-mermaid-node-bg-color:#1f1e1b;--md-mermaid-node-fg-color:#ecebe6;--md-mermaid-label-bg-color:#161513;--md-mermaid-label-fg-color:#ecebe6;--md-mermaid-sequence-actor-bg-color:#1f1e1b;--md-mermaid-sequence-actor-border-color:#8fb0ff}
.md-typeset .mermaid{padding:1rem .6rem;border:1px solid var(--hair2,#cfcbc1);border-radius:14px;background:var(--raised,#fff);box-shadow:var(--e2,none);overflow-x:auto}
`;
  document.head.append(style);

  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // ---------- Lexer: mirrors crates/aether-lang/src/lexer.rs ----------
  const KW = { manifold: "Manifold", block: "Block", regress: "Regress", render: "Render", embed: "Embed", until: "Until",
    escalate: "Escalate", convergence: "Convergence", true: "True", false: "False", class: "Class", new: "New", self: "Self_",
    import: "Import", from: "From", as: "As", seal: "Seal", for: "For", while: "While", if: "If", else: "Else", fn: "Fn",
    return: "Return", break: "Break", continue: "Continue", in: "In", let: "Let", dim: "Dim", tau: "Tau", model: "Model",
    color: "Color", axis: "Axis", project: "Project", cluster: "Cluster", center: "Center", spread: "Spread", format: "Format", output: "Output" };
  const KWNAME = Object.fromEntries(Object.entries(KW).map(([k, v]) => [v, k]));
  const alpha = (c) => /\p{Alphabetic}/u.test(c), alnum = (c) => /[\p{Alphabetic}\p{N}]/u.test(c), digit = (c) => c >= "0" && c <= "9";
  const TWO = { "==": "EqEq", "!=": "NotEq", "<=": "LessEq", ">=": "GreaterEq", "&&": "And", "||": "Or", "..": "DotDot" };
  const ONE = { "=": "Equals", ":": "Colon", ",": "Comma", ".": "Dot", "{": "LBrace", "}": "RBrace", "[": "LBracket", "]": "RBracket",
    "(": "LParen", ")": "RParen", "\n": "Newline", "+": "Plus", "-": "Minus", "*": "Star", "%": "Percent", "/": "Slash",
    "!": "Not", "<": "Less", ">": "Greater", "~": "Tilde", "\u{1F9AD}": "Seal" };
  function lex(src) {
    const cs = Array.from(src); let i = 0, line = 1, col = 1, pos = 0; const out = [];
    const adv = () => { const c = cs[i++]; pos += new TextEncoder().encode(c).length; if (c === "\n") { line++; col = 1; } else col++; return c; };
    for (;;) {
      while (cs[i] === " " || cs[i] === "\t" || cs[i] === "\r") adv();
      const t = { line, col, start: pos };
      if (i >= cs.length) { out.push({ ...t, k: "Eof", end: pos }); return out; }
      const c = adv(), n = cs[i];
      if (c === "/" && n === "/") { while (i < cs.length && cs[i] !== "\n") adv(); continue; }
      if (TWO[c + n] && !(c === "&" && n !== "&")) { adv(); t.k = TWO[c + n]; }
      else if (ONE[c]) t.k = ONE[c];
      else if (c === '"') {
        let s = ""; t.k = "Error"; t.v = "unexpected EOF in string";
        while (i < cs.length) { if (cs[i] === '"') { adv(); t.k = "StringLit"; t.v = s; break; } if (cs[i] === "\n") { t.v = "unterminated string"; break; } s += adv(); }
      } else if (digit(c)) {
        let ip = c; while (digit(cs[i] || "")) ip += adv();
        if (cs[i] === "." && digit(cs[i + 1] || "")) { adv(); let f = ""; while (digit(cs[i] || "") && f.length < 6) f += adv(); t.k = "Float"; t.v = [+ip, +f.padEnd(6, "0")]; }
        else { t.k = "Number"; t.v = +ip; }
      } else if (alpha(c) || c === "_") {
        let s = c; while (i < cs.length && (alnum(cs[i]) || cs[i] === "_")) s += adv();
        if (KW[s]) t.k = KW[s]; else { t.k = "Identifier"; t.v = s; }
      } else { t.k = "Error"; t.v = "unexpected char: " + c; }
      t.end = pos; out.push(t);
    }
  }
  const tokText = (t) => t.k === "Identifier" || t.k === "StringLit" ? `${t.k}("${t.v}")` : t.k === "Number" ? `Number(${t.v})`
    : t.k === "Float" ? `Float(${t.v[0]}, ${t.v[1]})` : t.k === "Error" ? `Error("${t.v}")` : t.k;
  const tokClass = (t) => t.k === "Error" ? "err" : KWNAME[t.k] ? "kw" : /Number|Float|StringLit|True|False/.test(t.k) ? "lit"
    : /Tilde|Newline|Eof/.test(t.k) ? "sep" : "";

  // ---------- Parser: mirrors crates/aether-lang/src/parser.rs (recursive descent) ----------
  // Node = [label, ...children]; children are nodes or strings.
  function parse(toks) {
    let p = 0;
    const peek = () => toks[Math.min(p, toks.length - 1)], next = () => toks[Math.min(p++, toks.length - 1)];
    const check = (k) => peek().k === k, end = () => check("Eof");
    const fail = (msg, t = peek()) => { throw new Error(`${msg} at ${t.line}:${t.col}`); };
    const expect = (k) => check(k) ? next() : fail(`expected ${k}, found ${peek().k}`);
    const ident = () => check("Identifier") ? next().v : fail(`expected identifier, found ${peek().k}`);
    const flexOk = () => check("Identifier") || !!KWNAME[peek().k];
    const flex = () => { const t = next(); return t.k === "Identifier" ? t.v : KWNAME[t.k] || fail("expected identifier", t); };
    const seps = () => { while (check("Tilde") || check("Newline")) next(); };
    const block = () => { expect("LBrace"); const s = []; while (!check("RBrace") && !end()) { seps(); if (check("RBrace")) break; const st = stmt(); if (st) s.push(st); } expect("RBrace"); return ["Block", ...s]; };
    const args = () => {
      expect("LParen"); const a = [];
      while (!check("RParen") && !end()) {
        if (flexOk()) { const save = p, nm = flex(); if (check("Equals")) { next(); a.push(["Named " + nm, expr()]); } else { p = save; a.push(expr()); } }
        else a.push(expr());
        if (check("Comma")) next();
      }
      expect("RParen"); return a;
    };
    const num = () => { const t = next(); return t.k === "Number" ? t.v : t.k === "Float" ? t.v[0] + t.v[1] / 1e6 : fail("expected number", t); };
    const identCont = (name) => {
      if (check("Dot")) { next(); const m = flex(); return check("LParen") ? ["MethodCall " + name + "." + m, ...args()] : ["FieldAccess " + name + "." + m]; }
      if (check("LParen")) return ["Call " + name, ...args()];
      if (check("LBracket")) { next(); const a = num(); expect("Colon"); const b = num(); expect("RBracket"); return ["Index " + name + "[" + a + ":" + b + "]"]; }
      return ["Ident " + name];
    };
    const bin = (sub, ops) => () => { let l = sub(); while (ops[peek().k]) { const o = ops[next().k]; l = ["BinaryOp " + o, l, sub()]; } return l; };
    const primary = () => {
      const t = next();
      switch (t.k) {
        case "Number": return ["Literal Num(" + t.v + ")"];
        case "Float": return ["Literal Num(" + (t.v[0] + t.v[1] / 1e6) + ")"];
        case "True": case "False": return ["Literal Bool(" + t.k.toLowerCase() + ")"];
        case "StringLit": return ['Literal Str("' + t.v + '")'];
        case "Self_": return ["Ident self"];
        case "Identifier": return identCont(t.v);
        case "New": { const c = ident(); expect("LParen"); const a = []; while (!check("RParen") && !end()) { a.push(expr()); if (check("Comma")) next(); } expect("RParen"); return ["New " + c, ...a]; }
        case "LBracket": { const e = []; while (!check("RBracket") && !end()) { if (check("Newline")) { next(); continue; } e.push(expr()); if (check("Comma")) next(); } expect("RBracket"); return ["List", ...e]; }
        case "Embed": case "Convergence": return ["Call " + KWNAME[t.k], ...args()];
        case "Error": return fail("lexer error: " + t.v, t);
        default: return fail("expected expression, found " + t.k, t);
      }
    };
    const unary = () => check("Minus") || check("Not") ? ["UnaryOp " + (next().k === "Minus" ? "Neg" : "Not"), unary()] : primary();
    const term = bin(unary, { Star: "Mul", Slash: "Div", Percent: "Mod" });
    const arith = bin(term, { Plus: "Add", Minus: "Sub" });
    const range = () => { const l = arith(); if (check("Colon") || check("DotDot")) { next(); const r = arith(); const v = (e) => /^Literal Num/.test(e[0]) ? e[0].slice(12, -1) : fail("expected number"); return ["Range " + v(l) + ".." + v(r)]; } return l; };
    const cmp = bin(range, { Less: "Lt", Greater: "Gt", LessEq: "Le", GreaterEq: "Ge" });
    const eq = bin(cmp, { EqEq: "Eq", NotEq: "Neq" });
    const and = bin(eq, { And: "And" });
    const expr = bin(and, { Or: "Or" });
    function stmt() {
      const t = peek(); let n;
      switch (t.k) {
        case "Error": fail("lexer error: " + t.v); break;
        case "Manifold": next(); n = ["Manifold " + ident()]; expect("Equals"); n.push(expr()); break;
        case "Block": next(); n = ["BlockDecl " + ident()]; expect("Equals"); n.push(expr()); break;
        case "Regress": case "Render": fail(t.k.toLowerCase() + " config blocks are parsed by parser.rs but not modelled in this viewer"); break;
        case "Identifier": {
          const a = ident();
          if (check("Identifier") && toks[p + 1] && toks[p + 1].k === "Equals") { const b = ident(); next(); n = ["Var " + b + " (type_hint " + a + ")", expr()]; }
          else if (check("Equals")) { next(); n = ["Assign " + a, expr()]; }
          else n = ["Expr", identCont(a)];
          break;
        }
        case "Class": {
          next(); n = ["Class " + ident()]; expect("LBrace");
          while (!check("RBrace") && !end()) {
            if (check("Newline") || check("Tilde")) { next(); continue; }
            if (check("Fn")) n.push(stmt());
            else { const f = ident(); const v = check("Equals") ? (next(), expr()) : ["Literal Bool(false)"]; if (check("Comma")) next(); n.push(["Field " + f, v]); }
          }
          expect("RBrace"); break;
        }
        case "Import": next(); n = ["Import " + ident()]; break;
        case "From": { next(); const m = ident(); expect("Import"); n = ["Import " + m + " { symbol: " + ident() + " }"]; break; }
        case "If": { next(); n = ["If", ["condition", expr()], ["then", block()]]; if (check("Else")) { next(); n.push(["else", block()]); } break; }
        case "While": next(); n = ["While", ["condition", expr()], block()]; break;
        case "For": { next(); const it = ident(); expect("In"); const r = expr(); if (!/^Range/.test(r[0])) fail("expected range in for loop"); n = ["For " + it, r, block()]; break; }
        case "Seal": next(); n = ["Loop (seal)"]; if (check("Until")) { next(); n.push(["until", expr()]); } n.push(block()); break;
        case "Fn": { next(); const nm = ident(); expect("LParen"); const ps = []; while (!check("RParen") && !end()) { ps.push(ident()); if (check("Comma")) next(); } expect("RParen"); n = ["Fn " + nm + "(" + ps.join(", ") + ")", block()]; break; }
        case "Return": next(); n = ["Return"]; if (!check("Newline") && !check("Tilde") && !check("RBrace")) n.push(expr()); break;
        case "Break": next(); n = ["Break"]; break;
        case "Continue": next(); n = ["Continue"]; break;
        case "Let": next(); n = ["Var " + ident()]; expect("Equals"); n.push(expr()); break;
        case "Newline": case "Eof": next(); return null;
        default: fail("unexpected token " + t.k + " (expected statement)");
      }
      seps(); return n;
    }
    const prog = ["Program"];
    while (!end()) { seps(); if (end()) break; const s = stmt(); if (s) prog.push(s); }
    return prog;
  }

  // Minimal self-check (runs once, console only on failure): lexer.rs unit tests replayed.
  (() => {
    const k = (s) => lex(s).map((t) => t.k).join(" ");
    const ok = k("manifold M = embed(data, dim=3)").startsWith("Manifold Identifier Equals Embed")
      && tokText(lex("1.5")[0]) === "Float(1, 500000)" && k("~") === "Tilde Eof" && k("1..10") === "Number DotDot Number Eof"
      && k("🦭 until") === "Seal Until Eof" && parse(lex("for i in 0..4 { t = t + i~ }"))[1][0] === "For i";
    if (!ok) console.warn("viz-pipeline: lexer/parser self-check failed");
  })();

  const treeHTML = (n) => {
    const [label, ...kids] = n; const sp = label.indexOf(" ");
    const head = sp < 0 ? `<b>${esc(label)}</b>` : `<b>${esc(label.slice(0, sp))}</b> <i>${esc(label.slice(sp + 1))}</i>`;
    return `<li>${head}${kids.length ? "<ul>" + kids.map(treeHTML).join("") + "</ul>" : ""}</li>`;
  };

  // Snippets are copied from the page the block sits on.
  const PRESETS = {
    pipeline: {
      "seal until": "let count = 0~\n\n🦭 until count >= 3 {\n  count = count + 1~\n}",
      manifold: "let data = [1.0, 2.0, 3.0, 4.0]~\nmanifold M = embed(data, tau=1)~\nblock B = M.cluster(0:2)~",
      fn: "fn add(a, b) {\n  return a + b~\n}\n\nlet result = add(2, 3)~",
    },
    syntax: {
      literals: 'let x = 10~\nlet y = 3.14~\nlet ok = true~\nlet name = "aether"~\nlet values = [1.0, 2.0, 3.0]~',
      "type hint": "point C = [1.0, 2.0, 3.0]~",
      "if / else": 'if count == 0 {\n  print("empty")~\n} else {\n  print("nonempty")~\n}',
      for: "for i in 0..4 {\n  total = total + i~\n}",
      seal: "seal until count >= 3 {\n  count = count + 1~\n}",
      manifold: "let data = [1.0, 2.0, 3.0, 4.0]~\nmanifold M = embed(data, tau=1)~\nblock B = M.cluster(0:2)~",
    },
    examples: {
      runtime: 'let data = [1.0, 2.0, 3.0, 4.0]~\nmanifold M = embed(data, tau=1)~\nprint("embedded")~',
      topology: 'import topology~\nlet data = [1.0, 1.0, 1.0, 1.0, 1.0]~\nmanifold M = embed(data, tau=1)~\nlet diagram = topology.ph(M, max_dim=2, mode="vr", max_points=16)~\nlet b = topology.betti(diagram, radius=0.0)~',
    },
    seal: {
      "seal": "seal until count >= 3 {\n  count = count + 1~\n}",
      "🦭": "🦭 until count >= 3 {\n  count = count + 1~\n}",
    },
  };

  // ---------- pipe-lexer: source -> tokens -> AST ----------
  TSViz.register("pipe-lexer", (stage, api) => {
    const box = stage.parentElement, presets = PRESETS[box.dataset.preset] || PRESETS.pipeline;
    const stages = el("div", "p-stages"); stages.setAttribute("aria-hidden", "true");
    const names = ["Source", "Lexer", "Tokens", "Parser", "AST with spans", "Interpreter / Titan VM"];
    const chips = names.map((n, i) => { if (i) stages.append(el("span", "p-arrow", "→")); const c = el("span", "p-stage", n); stages.append(c); return c; });
    const ta = el("textarea"); ta.spellcheck = false; ta.setAttribute("aria-label", "Aether source to tokenize and parse");
    const tl = el("div", "p-label", "Token stream · lexer.rs"), toks = el("div", "p-toks");
    const al = el("div", "p-label", "AST · parser.rs"), tree = el("div", "p-tree"); tree.tabIndex = 0; tree.setAttribute("aria-label", "Parsed AST");
    const err = el("div", "p-err"); err.setAttribute("role", "status");
    const note = el("div", "p-note", "The last stage is not simulated here: aether run hands this AST to Interpreter::execute, and --mode titan hands it to the Titan Compiler and TitanVM.");
    stage.append(stages, ta, tl, toks, al, tree, err, note);
    let timers = [];
    const light = (n) => chips.forEach((c, i) => c.classList.toggle("on", i < n));
    function run() {
      timers.forEach(clearTimeout); timers = [];
      const ts = lex(ta.value); toks.textContent = ""; err.textContent = ""; tree.innerHTML = "";
      const anim = !api.still, gap = Math.min(40, 900 / ts.length);
      ts.forEach((t, i) => {
        const c = el("span", "p-tok " + tokClass(t) + (anim ? " p-in" : ""), tokText(t));
        if (anim) c.style.animationDelay = (i * gap) + "ms";
        c.title = `line ${t.line}, col ${t.col}, bytes ${t.start}..${t.end}`; toks.append(c);
      });
      light(3);
      const done = () => {
        try { const ast = parse(ts); tree.innerHTML = "<ul>" + treeHTML(ast) + "</ul>"; if (anim) tree.firstChild.classList.add("p-in"); light(5); }
        catch (e) { err.textContent = "ParseError: " + e.message; light(3); }
      };
      anim ? timers.push(setTimeout(done, ts.length * gap + 150)) : done();
    }
    const c = api.controls(), btns = [];
    for (const [name, src] of Object.entries(presets)) {
      const b = c.button(name, () => { ta.value = src; btns.forEach((x) => x.setAttribute("aria-pressed", x === b)); run(); }); btns.push(b);
    }
    ta.value = Object.values(presets)[0]; btns[0].setAttribute("aria-pressed", "true");
    let t; ta.addEventListener("input", () => { clearTimeout(t); t = setTimeout(run, 250); btns.forEach((x) => x.setAttribute("aria-pressed", "false")); });
    run();
  });

  // ---------- pipe-workspace: Cargo workspace dependency map ----------
  const CRATES = {
    "aether-cli": { x: 60, y: 30, role: "REPL, script runner, syntax checker (binary `aether`)", deps: ["aether-core", "aether-lang"], ext: "clap, rustyline" },
    "aegis-cli": { x: 180, y: 30, role: "Legacy compatibility binary", deps: ["aegis-core", "aether-lang"], ext: "clap, rustyline", note: "depends on aether-lang under the alias aegis-lang" },
    "aether-kernel": { x: 300, y: 30, role: "no_std sparse scheduler, loader, allocator, boot scaffolding", deps: ["aether-core", "aether-lang"], ext: "x86_64, spin, volatile, heapless, libm, getrandom, multiboot2, bootloader (optional)", note: "both path deps use default-features = false, features = [\"no_std\"]" },
    "aether-gpu": { x: 300, y: 130, role: "Workspace member not listed in the Runtime Surface table; GPU claims are gated", deps: ["aether-core"], ext: "wgpu 27.0, pollster, bytemuck" },
    "aether-lang": { x: 120, y: 130, role: "Lexer, parser, AST, interpreter, Titan VM, exporters", deps: ["aether-core"], ext: "libm, heapless, rand; optional candle-core, candle-transformers, tokenizers, hf-hub, pyo3" },
    "aether-core": { x: 180, y: 230, role: "Manifolds, topology, ML primitives, governors, state", deps: [], ext: "libm, nalgebra, heapless" },
    "aegis-core": { x: 50, y: 230, role: "Legacy compatibility surface", deps: [], ext: "libm; optional serde, nalgebra" },
  };
  TSViz.register("pipe-workspace", (stage, api) => {
    const NS = "http://www.w3.org/2000/svg", W = 110, H = 26;
    const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", "0 0 360 256"); svg.setAttribute("role", "group");
    svg.setAttribute("aria-label", "Crate dependency map of the Cargo workspace"); svg.style.maxWidth = "560px"; svg.style.margin = "0 auto";
    const edges = [], nodes = {};
    for (const [n, c] of Object.entries(CRATES)) for (const d of c.deps) {
      const t = CRATES[d], path = document.createElementNS(NS, "path"), x1 = c.x, y1 = c.y + H / 2, x2 = t.x, y2 = t.y - H / 2;
      path.setAttribute("d", `M${x1} ${y1} C${x1} ${(y1 + y2) / 2} ${x2} ${(y1 + y2) / 2} ${x2} ${y2}`); path.setAttribute("class", "edge");
      path.setAttribute("marker-end", "url(#pipe-arr)"); svg.append(path); edges.push({ path, from: n, to: d });
    }
    svg.insertAdjacentHTML("afterbegin", `<defs><marker id="pipe-arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L8 4L0 8z" fill="#8a857a"/></marker></defs>`);
    const panel = el("div", "p-panel"); panel.setAttribute("aria-live", "polite");
    function select(name) {
      const c = CRATES[name], near = new Set([name, ...c.deps, ...Object.keys(CRATES).filter((k) => CRATES[k].deps.includes(name))]);
      edges.forEach((e) => e.path.classList.toggle("hot", e.from === name || e.to === name));
      Object.entries(nodes).forEach(([k, g]) => { g.classList.toggle("sel", k === name); g.classList.toggle("dim", !near.has(k)); g.setAttribute("aria-pressed", k === name); });
      const users = Object.keys(CRATES).filter((k) => CRATES[k].deps.includes(name));
      panel.innerHTML = `<h4>crates/${name}</h4><div>${esc(c.role)}</div>` +
        `<div class="p-note">Path deps: <code>${c.deps.join("</code>, <code>") || "none"}</code> · Used by: <code>${users.join("</code>, <code>") || "none"}</code></div>` +
        `<div class="p-note">External deps (Cargo.toml): ${esc(c.ext)}${c.note ? " · " + esc(c.note) : ""}</div>`;
    }
    for (const [n, c] of Object.entries(CRATES)) {
      const g = document.createElementNS(NS, "g"); g.setAttribute("class", "node"); g.setAttribute("tabindex", "0"); g.setAttribute("role", "button");
      g.setAttribute("aria-label", n);
      g.innerHTML = `<rect x="${c.x - W / 2}" y="${c.y - H / 2}" width="${W}" height="${H}" rx="8"/><text x="${c.x}" y="${c.y + 4}" text-anchor="middle">${n}</text>`;
      g.addEventListener("click", () => select(n)); g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(n); } });
      svg.append(g); nodes[n] = g;
    }
    stage.append(svg, panel); select("aether-lang");
  });

  // ---------- pipe-extension: the six-stage extension boundary ----------
  TSViz.register("pipe-extension", (stage) => {
    const steps = ["token kind", "AST node", "parser rule", "interpreter behavior or VM opcode", "tests", "documentation status update"];
    const grid = el("div", "p-grid"), out = el("div", "p-panel"); out.setAttribute("aria-live", "polite");
    const boxes = steps.map((s, i) => { const l = el("label", "p-check"); const b = el("input"); b.type = "checkbox"; b.checked = true; l.append(b, `${i + 1}. ${s}`); grid.append(l); b.addEventListener("change", upd); return b; });
    function upd() {
      const on = boxes.map((b) => b.checked), front = on[0] && on[1] && on[2], runtime = on[3];
      let v;
      if (front && !runtime) v = ["Parsed-only feature", "gate", "The parser accepts it, but nothing executes it."];
      else if (!front && runtime) v = ["Runtime-only feature", "gate", "The runtime can execute it, but source text cannot reach it."];
      else if (!front && !runtime) v = ["Not a language feature yet", "gate", "Neither the front end nor the runtime carries it."];
      else if (!on[4]) v = ["Implemented, not evidenced", "gate", "Without a test the evidence policy does not allow calling it active."];
      else if (!on[5]) v = ["Tested, status not updated", "gate", "The status matrix still has to record it."];
      else v = ["Complete path", "ok", "Token, AST, parser, runtime, test and docs agree."];
      out.innerHTML = `<h4>${v[0]}<span class="p-badge ${v[1]}">${v[1] === "ok" ? "active" : "gated"}</span></h4>${v[2]}`;
    }
    stage.append(grid, out); upd();
  });

  // ---------- pipe-cli: what each aether command runs (aether-cli/src/main.rs) ----------
  TSViz.register("pipe-cli", (stage, api) => {
    const S = ["read file", "Parser::new (lexes)", "parser.parse → AST", "Interpreter::execute", "Compiler::compile", "TitanVM::run"];
    const CMD = {
      "check FILE": { on: [0, 1, 2], out: "✓ Syntax OK, or the parse error with its position; exit 1 on failure." },
      "run FILE": { on: [0, 1, 2, 3], out: "Default --mode bio: the tree-walking interpreter executes the AST. Warns if the extension is not .aether or .ae." },
      "run FILE --mode titan": { on: [0, 1, 2, 4, 5], out: "The AST is compiled to stack bytecode and run on the Titan VM." },
      "repl": { on: [1, 2, 3], out: "Each line is parsed and executed against one long-lived Interpreter." },
    };
    const row = el("div", "p-stages"), chips = S.map((s, i) => { if (i) row.append(el("span", "p-arrow", "→")); const c = el("span", "p-stage", s); row.append(c); return c; });
    const out = el("div", "p-panel"); out.setAttribute("aria-live", "polite"); stage.append(row, out);
    const c = api.controls(), btns = []; let timers = [];
    const pick = (k, b) => {
      btns.forEach((x) => x.setAttribute("aria-pressed", x === b)); timers.forEach(clearTimeout); timers = [];
      chips.forEach((ch) => { ch.classList.remove("on"); ch.classList.toggle("off", !CMD[k].on.includes(chips.indexOf(ch))); });
      CMD[k].on.forEach((i, n) => api.still ? chips[i].classList.add("on") : timers.push(setTimeout(() => chips[i].classList.add("on"), n * 220)));
      out.innerHTML = `<h4 class="p-mono">aether ${esc(k)}</h4>${esc(CMD[k].out)}`;
    };
    for (const k of Object.keys(CMD)) { const b = c.button("aether " + k, () => pick(k, b)); btns.push(b); }
    pick("check FILE", btns[0]);
  });

  // ---------- pipe-modules: module contracts (modules.md, interpreter.rs import_*) ----------
  TSViz.register("pipe-modules", (stage, api) => {
    const M = {
      math: { names: ["sin", "cos", "sqrt", "exp", "pi"], status: "ok", note: "Imported names bind as native functions; pi binds as a number." },
      topology: { names: ["topology.ph(manifold, ...)", "topology.betti(diagram_or_manifold, radius=...)", "topology.intervals(diagram)", "topology.Betti(...) alias"], status: "ok", note: "Dispatches to NativeFunction::TopoPh, TopoBetti, TopoIntervals." },
      Ml: { names: ["Ml.MLP(...)", "Ml.KMeans(...)", "Ml.Conv2D(...)", "tensor helpers: matmul, add, ReLU, softmax"], status: "gate", note: "Constructors are active (MlpNew, KMeansNew, Conv2DNew). Individual methods count only where a test documents them." },
      Seal: { names: ["Seal.train"], status: "gate", note: "Native entrypoint exists; training quality and topological stop are gated until a test or benchmark covers the call path." },
    };
    const flow = el("div", "p-stages"), panel = el("div", "p-panel"); panel.setAttribute("aria-live", "polite");
    stage.append(flow, panel);
    const c = api.controls(), btns = [];
    const pick = (k, b) => {
      btns.forEach((x) => x.setAttribute("aria-pressed", x === b));
      flow.innerHTML = ["import " + k + "~", "Value::Module(\"" + k + "\")", "Value::NativeFn", "execute_native_fn"].map((s, i) => `${i ? '<span class="p-arrow">→</span>' : ""}<span class="p-stage on p-mono${api.still ? "" : " p-in"}" style="animation-delay:${i * 120}ms">${esc(s)}</span>`).join("");
      const m = M[k];
      panel.innerHTML = `<h4>${k}<span class="p-badge ${m.status}">${m.status === "ok" ? "active" : "gated"}</span></h4><ul style="margin:.2rem 0 .4rem 1rem">${m.names.map((n) => `<li><code>${esc(n)}</code></li>`).join("")}</ul><div class="p-note">${esc(m.note)}</div>`;
    };
    for (const k of Object.keys(M)) { const b = c.button(k, () => pick(k, b)); btns.push(b); }
    pick("math", btns[0]);
  });

  // ---------- pipe-flow: index.md flow with a concrete numeric example ----------
  // embed: ManifoldWorkspace::embed_data + TimeDelayEmbedder<3> (point = [x(t), x(t-τ), x(t-2τ)], needs 3τ samples).
  // topology: Vietoris-Rips, edge enters at its Euclidean length; betti_at(r) counts birth <= r < death.
  TSViz.register("pipe-flow", (stage, api) => {
    const NS = "http://www.w3.org/2000/svg";
    const titles = ["Source", "Typed runtime object", "Embedding", "Topology", "Decision"];
    const row = el("div", "p-stages"), chips = titles.map((s, i) => { if (i) row.append(el("span", "p-arrow", "→")); const c = el("span", "p-stage", s); row.append(c); return c; });
    const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", "0 0 320 150"); svg.setAttribute("aria-hidden", "true"); svg.style.maxWidth = "520px"; svg.style.margin = ".2rem auto";
    const panel = el("div", "p-panel p-mono"); panel.setAttribute("aria-live", "polite");
    stage.append(row, svg, panel);
    let data = [1, 2, 3, 4], tau = 1, r = 1, step = 0;
    const embed = () => { const pts = []; for (let t = 3 * tau - 1; t < data.length; t++) pts.push([data[t], data[t - tau], data[t - 2 * tau]]); return pts; };
    const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    const b0 = (pts) => { const par = pts.map((_, i) => i), f = (i) => par[i] === i ? i : (par[i] = f(par[i]));
      pts.forEach((a, i) => pts.forEach((b, j) => { if (j > i && dist(a, b) <= r) par[f(i)] = f(j); })); return new Set(pts.map((_, i) => f(i))).size; };
    const fmt = (v) => +v.toFixed(3);
    function draw() {
      chips.forEach((c, i) => c.classList.toggle("on", i <= step));
      const pts = embed(), th = api.theme(), all = pts.flat(), lo = Math.min(...all, 0), hi = Math.max(...all, 1), sc = 110 / (hi - lo || 1);
      const P = (p) => [30 + (p[0] - lo) * sc + (p[2] - lo) * sc * 0.35, 135 - (p[1] - lo) * sc - (p[2] - lo) * sc * 0.18];
      let s = "";
      if (step === 0 || step === 1) data.forEach((v, i) => { const x = 20 + i * (280 / Math.max(data.length, 1)), h = (v - Math.min(...data, 0)) / ((Math.max(...data) - Math.min(...data, 0)) || 1) * 100;
        s += `<rect x="${x}" y="${135 - h}" width="${Math.max(6, 200 / data.length)}" height="${h}" rx="3" fill="${step ? th.accent : th.hair}"/><text x="${x}" y="148" font-size="9" fill="${th.muted}">${v}</text>`; });
      else {
        if (step >= 3) pts.forEach((a, i) => pts.forEach((b, j) => { if (j > i && dist(a, b) <= r) { const [x1, y1] = P(a), [x2, y2] = P(b); s += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${th.accent}" stroke-width="1.5"/>`; } }));
        pts.forEach((p) => { const [x, y] = P(p); if (step >= 3) s += `<circle cx="${x}" cy="${y}" r="${r / 2 * sc}" fill="${th.accent}" fill-opacity=".08" stroke="${th.accent}" stroke-opacity=".35"/>`;
          s += `<circle cx="${x}" cy="${y}" r="4" fill="${th.accent}"/><text x="${x + 6}" y="${y - 6}" font-size="9" fill="${th.muted}">(${p.map(fmt).join(", ")})</text>`; });
      }
      svg.innerHTML = s;
      const lst = `[${data.join(", ")}]`;
      const txt = [
        `let data = ${lst}~\nmanifold M = embed(data, tau=${tau})~\nlet diagram = topology.ph(M)~\nlet b = topology.betti(diagram, radius=${r})~`,
        `Value::List of ${data.length} Value::Num → manifold M = Value::Manifold(handle)\nhandle → manifolds[handle] (interpreter-owned arena)`,
        pts.length ? `TimeDelayEmbedder<3>, τ=${tau}: point(t) = [x(t), x(t−${tau}), x(t−${2 * tau})]\nneeds 3τ = ${3 * tau} samples → ${pts.length} point(s):\n${pts.map((p) => "(" + p.map(fmt).join(", ") + ")").join("  ")}`
          : `TimeDelayEmbedder<3>, τ=${tau} needs ${3 * tau} samples; ${data.length} given → no points.`,
        `Vietoris–Rips: edge enters at its length.\n${pts.length > 1 ? "pairwise distances: " + pts.flatMap((a, i) => pts.slice(i + 1).map((b) => fmt(dist(a, b)))).join(", ") : "fewer than two points"}\nbetti_at(r=${r}): β0 = ${pts.length ? b0(pts) : 0}`,
        pts.length ? `β0 = ${b0(pts)} at radius ${r}: ${b0(pts) === 1 ? "one connected component. A threshold of β0 = 1 is met: converge / proceed." : b0(pts) + " components. The shape has not joined, so a β0 = 1 threshold is not met yet: keep looping or prune."}`
          : "No points, so there is no shape to decide on: this is a rejection case.",
      ];
      panel.textContent = `${step + 1}/5 · ${titles[step]}\n` + txt[step]; panel.style.whiteSpace = "pre-wrap";
    }
    const c = api.controls();
    c.button("← Back", () => { step = Math.max(0, step - 1); draw(); });
    c.button("Next →", () => { step = Math.min(4, step + 1); draw(); });
    const lab = el("label", null, "data "), inp = el("input"); inp.type = "text"; inp.value = data.join(", "); lab.append(inp);
    stage.parentElement.querySelector(".ts-viz-controls").append(lab);
    inp.addEventListener("input", () => { const v = inp.value.split(/[,\s]+/).filter(Boolean).map(Number); if (v.length && v.every(isFinite)) { data = v.slice(0, 32); draw(); } });
    c.slider("τ", 1, 3, 1, 1, (v) => { tau = v; draw(); });
    c.slider("radius", 0, 4, 0.05, 1, (v) => { r = v; draw(); });
    api.onTheme(draw);
    // Auto-advance once when first seen; manual after that.
    if (!api.still) { let t = 0; const id = setInterval(() => { if (++t > 4) return clearInterval(id); step = t; draw(); }, 1600); }
    draw();
  });
})();
