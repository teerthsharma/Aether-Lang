// viz-formal: visualisations for docs/FORMAL_CORE.md, grounded in Aether/*.lean.
(() => {
  if (!window.TSViz) return;
  const R = TSViz.register;

  const style = document.createElement("style");
  style.textContent = `
.ts-viz[data-viz^="formal-"] .fv { font-size:.7rem; color:var(--fv-ink); }
.ts-viz[data-viz^="formal-"] .fv code, .ts-viz[data-viz^="formal-"] .fv pre { font-family:var(--md-code-font-family); font-size:.66rem; }
.ts-viz[data-viz^="formal-"] .fv pre { margin:0; padding:.55rem .7rem; border:1px solid var(--fv-hair); border-radius:8px; background:var(--fv-ground); white-space:pre-wrap; word-break:break-word; line-height:1.5; }
.ts-viz[data-viz^="formal-"] .fv-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:.6rem; margin-top:.6rem; }
.ts-viz[data-viz^="formal-"] .fv-k { font-size:.6rem; letter-spacing:.06em; text-transform:uppercase; color:var(--fv-muted); margin:0 0 .25rem; }
.ts-viz[data-viz^="formal-"] .fv-steps { display:flex; flex-wrap:wrap; gap:.35rem; margin:.2rem 0 .6rem; }
.ts-viz[data-viz^="formal-"] .fv-steps span { padding:.15rem .5rem; border:1px solid var(--fv-hair); border-radius:999px; color:var(--fv-muted); font-family:var(--md-code-font-family); font-size:.62rem; transition:all .25s; }
.ts-viz[data-viz^="formal-"] .fv-steps span.on { border-color:var(--fv-accent); color:var(--fv-accent); background:color-mix(in srgb,var(--fv-accent) 10%,transparent); }
.ts-viz[data-viz^="formal-"] .fv-steps span.done { color:var(--fv-ink); }
.ts-viz[data-viz^="formal-"] .fv-chip { display:inline-block; padding:.1rem .5rem; border-radius:999px; font-weight:600; font-size:.62rem; }
.ts-viz[data-viz^="formal-"] .fv-ok { color:var(--fv-good); border:1px solid var(--fv-good); }
.ts-viz[data-viz^="formal-"] .fv-bad { color:var(--fv-warn); border:1px solid var(--fv-warn); }
.ts-viz[data-viz^="formal-"] .fv-trace { max-height:180px; overflow:auto; }
.ts-viz[data-viz^="formal-"] .fv-trace div { white-space:pre; font-family:var(--md-code-font-family); font-size:.62rem; color:var(--fv-muted); animation:fvIn .3s both; }
.ts-viz[data-viz^="formal-"] .fv-cells { display:flex; flex-wrap:wrap; gap:.3rem; min-height:2rem; align-items:center; }
.ts-viz[data-viz^="formal-"] .fv-cells span { padding:.25rem .5rem; border:1px solid var(--fv-hair); border-radius:6px; font-family:var(--md-code-font-family); font-size:.64rem; transition:all .25s; }
.ts-viz[data-viz^="formal-"] .fv-cells span.scan { border-color:var(--fv-muted); }
.ts-viz[data-viz^="formal-"] .fv-cells span.hit { border-color:var(--fv-accent); color:var(--fv-accent); box-shadow:0 0 0 3px color-mix(in srgb,var(--fv-accent) 18%,transparent); }
.ts-viz[data-viz^="formal-"] .fv-cells span.new { animation:fvIn .35s both; }
.ts-viz[data-viz^="formal-"] select { font:inherit; font-size:.66rem; max-width:100%; padding:.2rem .3rem; border:1px solid var(--fv-hair); border-radius:6px; background:var(--fv-ground); color:var(--fv-ink); }
.ts-viz[data-viz^="formal-"] mark { background:color-mix(in srgb,var(--fv-accent) 16%,transparent); color:inherit; border-radius:3px; padding:0 .1rem; }
.ts-viz[data-viz="formal-deps"] svg .n rect { transition:stroke .2s, fill .2s; }
.ts-viz[data-viz="formal-deps"] svg .n { cursor:pointer; outline:none; }
.ts-viz[data-viz="formal-deps"] svg .e { transition:stroke .2s, opacity .2s; }
.ts-viz[data-viz="formal-deps"] svg .e.hot { stroke-dasharray:5 4; animation:fvDash .8s linear infinite; }
@keyframes fvDash { to { stroke-dashoffset:-18; } }
@keyframes fvIn { from { opacity:0; transform:translateY(3px); } }
@media (prefers-reduced-motion: reduce) { .ts-viz[data-viz^="formal-"] * { animation:none !important; transition:none !important; } }
`;
  document.head.append(style);

  const h = (tag, attrs = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) k === "class" ? (e.className = v) : k === "html" ? (e.innerHTML = v) : e.setAttribute(k, v);
    e.append(...kids); return e;
  };
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
  function root(stage, api) {
    const el = h("div", { class: "fv" }); stage.append(el);
    const paint = (t) => { const s = el.style; s.setProperty("--fv-ink", t.ink); s.setProperty("--fv-muted", t.muted); s.setProperty("--fv-hair", t.hair);
      s.setProperty("--fv-ground", t.ground); s.setProperty("--fv-accent", t.accent); s.setProperty("--fv-good", t.good); s.setProperty("--fv-warn", t.accent2); };
    paint(api.theme()); api.onTheme(paint); return el;
  }
  // Step-through proof explorer: steps = [{tac, goal, note}]
  function prover(el, api, steps) {
    const strip = h("div", { class: "fv-steps" }); const goal = h("pre"); const note = h("p", { class: "fv-k", style: "margin-top:.4rem;text-transform:none;letter-spacing:0" });
    steps.forEach((s) => strip.append(h("span", {}, s.tac)));
    el.append(h("p", { class: "fv-k" }, "Lean proof, step by step"), strip, goal, note);
    let i = 0;
    const show = () => { [...strip.children].forEach((c, j) => { c.className = j === i ? "on" : j < i ? "done" : ""; });
      goal.innerHTML = steps[i].goal; note.textContent = `Step ${i + 1}/${steps.length}. ${steps[i].note}`; };
    const c = api.controls();
    c.button("Previous step", () => { i = Math.max(0, i - 1); show(); });
    c.button("Next step", () => { i = Math.min(steps.length - 1, i + 1); show(); });
    show(); return c;
  }

  // ---------- Lean-faithful mini executor (Aether/Core.lean 533-682) ----------
  const N = (n) => ["num", n], Bo = (b) => ["bool", b], St = (s) => ["str", s], Vr = (x) => ["var", x], Li = (...a) => ["list", a];
  const Bin = (l, op, r) => ["bin", l, op, r], Not = (e) => ["not", e], Ix = (t, i) => ["index", t, i], Fd = (t, f) => ["field", t, f], Me = (t, m) => ["method", t, m], Ca = (f, a) => ["call", f, a];
  const Let = (x, e) => ["let", x, e], As = (x, e) => ["assign", x, e], If = (c, t, e) => ["if", c, t, e], Wh = (c, b) => ["while", c, b], For = (i, a, b, body) => ["for", i, a, b, body],
    Seal = (c, b) => ["seal", c, b], Fn = (n, p, b) => ["fn", n, p, b], Ret = (e) => ["ret", e], Ex = (e) => ["expr", e], Brk = ["break"], Cnt = ["continue"];
  const vNum = (n) => ({ k: "num", v: n }), vBool = (b) => ({ k: "bool", v: b }), vStr = (s) => ({ k: "str", v: s }), vList = (a) => ({ k: "list", v: a }), vUnit = { k: "unit" };
  const fv = (v) => v.k === "unit" ? "Value.unit" : v.k === "list" ? `Value.list [${v.v.map(fv).join(", ")}]` : v.k === "str" ? `Value.${v.k} "${v.v}"` : `Value.${v.k} ${v.v}`;
  const fenv = (env) => `[${env.map(([k, v]) => `("${k}", ${fv(v)})`).join(", ")}]`;
  const fflow = (f) => f.k === "value" ? `Flow.value (${fv(f.v)})` : f.k === "return" ? `Flow.return (${fv(f.v)})` : `Flow.${f.k}`;
  const fe = (e) => { switch (e[0]) {
    case "num": return `Expr.num ${e[1]}`; case "bool": return `Expr.bool ${e[1]}`; case "str": return `Expr.str "${e[1]}"`; case "var": return `Expr.var "${e[1]}"`;
    case "list": return `Expr.list [${e[1].map(fe).join(", ")}]`; case "bin": return `Expr.binary (${fe(e[1])}) BinOp.${e[2]} (${fe(e[3])})`;
    case "not": return `Expr.unary UnOp.not (${fe(e[1])})`; case "index": return `Expr.index (${fe(e[1])}) (${fe(e[2])})`;
    case "field": return `Expr.field (${fe(e[1])}) "${e[2]}"`; case "method": return `Expr.method (${fe(e[1])}) "${e[2]}" []`;
    case "call": return `Expr.call "${e[1]}" [${e[2].map(fe).join(", ")}]`; } };
  const fb = (b) => `[${b.map(fs).join(", ")}]`;
  const fs = (s) => { switch (s[0]) {
    case "let": return `Stmt.letDecl "${s[1]}" (${fe(s[2])})`; case "assign": return `Stmt.assign "${s[1]}" (${fe(s[2])})`;
    case "if": return `Stmt.ifThenElse (${fe(s[1])}) ${fb(s[2])} ${s[3] ? `(some ${fb(s[3])})` : "none"}`;
    case "while": return `Stmt.while (${fe(s[1])}) ${fb(s[2])}`; case "for": return `Stmt.forRange "${s[1]}" ${s[2]} ${s[3]} ${fb(s[4])}`;
    case "seal": return `Stmt.seal ${s[1] ? `(some (${fe(s[1])}))` : "none"} ${fb(s[2])}`; case "fn": return `Stmt.fnDecl "${s[1]}" [${s[2].map((p) => `"${p}"`).join(", ")}] ${fb(s[3])}`;
    case "ret": return `Stmt.ret ${s[1] ? `(some (${fe(s[1])}))` : "none"}`; case "expr": return `Stmt.expr (${fe(s[1])})`; default: return `Stmt.${s[0]}`; } };
  const lookup = (env, x) => { for (const [k, v] of env) if (k === x) return v; return null; };
  const assign = (env, x, v) => { const i = env.findIndex(([k]) => k === x); if (i < 0) return null; const c = env.slice(); c[i] = [x, v]; return c; };
  const truthy = (v) => v.k === "bool" ? v.v : v.k === "num" ? v.v !== 0 : v.k === "str" ? v.v !== "" : v.k === "list" ? v.v.length > 0 : false;

  function run(fuel, env, fns, node, isExpr, trace) {
    const T = (d, s) => trace.push("  ".repeat(Math.min(d, 8)) + s);
    function evalE(f, env, fns, e, d) {
      if (f === 0) { T(d, `evalExprWithFns 0 … = none (fuel exhausted)`); return null; }
      const g = f - 1; T(d, `evalExprWithFns ${f}  ${fe(e).slice(0, 60)}`);
      switch (e[0]) {
        case "num": return vNum(e[1]); case "bool": return vBool(e[1]); case "str": return vStr(e[1]);
        case "var": return lookup(env, e[1]);
        case "list": { const vs = []; for (const x of e[1]) { const v = evalE(g, env, fns, x, d + 1); if (!v) return null; vs.push(v); } return vList(vs); }
        case "not": { const v = evalE(g, env, fns, e[1], d + 1); return v && v.k === "bool" ? vBool(!v.v) : null; }
        case "bin": { const l = evalE(g, env, fns, e[1], d + 1); if (!l) return null; const r = evalE(g, env, fns, e[3], d + 1); if (!r) return null;
          return e[2] === "add" && l.k === "num" && r.k === "num" ? vNum(l.v + r.v) : null; }
        case "index": { const t = evalE(g, env, fns, e[1], d + 1); if (!t) return null; const i = evalE(g, env, fns, e[2], d + 1); if (!i) return null;
          return t.k === "list" && i.k === "num" ? t.v[i.v] || null : null; }
        case "field": { const t = evalE(g, env, fns, e[1], d + 1); return t && e[2] === "length" && t.k === "list" ? vNum(t.v.length) : null; }
        case "method": { const t = evalE(g, env, fns, e[1], d + 1); return t && e[2] === "len" && t.k === "str" ? vNum([...t.v].length) : null; }
        case "call": { const fn = lookup(fns, e[1]); if (!fn) return null; const vs = [];
          for (const x of e[2]) { const v = evalE(g, env, fns, x, d + 1); if (!v) return null; vs.push(v); }
          if (vs.length !== fn.params.length) return null;
          let frame = env; fn.params.forEach((p, i) => { frame = [[p, vs[i]], ...frame]; });
          const r = execB(g, frame, fns, fn.body, d + 1); if (!r) return null;
          return r[2].k === "value" || r[2].k === "return" ? r[2].v : null; }
      } return null;
    }
    function execS(f, env, fns, s, d) {
      if (f === 0) { T(d, `execStmtWithFns 0 … = none (fuel exhausted)`); return null; }
      const g = f - 1; T(d, `execStmtWithFns ${f}  ${fs(s).slice(0, 60)}`);
      const loop = (env1, fns1, flow, again) => flow.k === "value" || flow.k === "continue" ? again(env1, fns1)
        : flow.k === "return" ? [env1, fns1, flow] : [env1, fns1, { k: "value", v: vUnit }];
      switch (s[0]) {
        case "let": { const v = evalE(g, env, fns, s[2], d + 1); return v && [[[s[1], v], ...env], fns, { k: "value", v }]; }
        case "assign": { const v = evalE(g, env, fns, s[2], d + 1); if (!v) return null; const u = assign(env, s[1], v); return u && [u, fns, { k: "value", v }]; }
        case "if": { const v = evalE(g, env, fns, s[1], d + 1); if (!v) return null;
          return truthy(v) ? execB(g, env, fns, s[2], d + 1) : s[3] ? execB(g, env, fns, s[3], d + 1) : [env, fns, { k: "value", v: vUnit }]; }
        case "while": { const v = evalE(g, env, fns, s[1], d + 1); if (!v) return null;
          if (!truthy(v)) return [env, fns, { k: "value", v: vUnit }];
          const r = execB(g, env, fns, s[2], d + 1); return r && loop(...r, (e1, f1) => execS(g, e1, f1, s, d + 1)); }
        case "for": { const [, it, a, b, body] = s;
          if (!(a < b)) return [[[it, vNum(a)], ...env], fns, { k: "value", v: vUnit }];
          const r = execB(g, [[it, vNum(a)], ...env], fns, body, d + 1);
          return r && loop(...r, (e1, f1) => execS(g, e1, f1, For(it, a + 1, b, body), d + 1)); }
        case "seal": { if (s[1]) { const v = evalE(g, env, fns, s[1], d + 1); if (!v) return null; if (truthy(v)) return [env, fns, { k: "value", v: vUnit }]; }
          const r = execB(g, env, fns, s[2], d + 1); return r && loop(...r, (e1, f1) => execS(g, e1, f1, s, d + 1)); }
        case "fn": return [env, [[s[1], { params: s[2], body: s[3] }], ...fns], { k: "value", v: vUnit }];
        case "ret": { if (!s[1]) return [env, fns, { k: "return", v: vUnit }]; const v = evalE(g, env, fns, s[1], d + 1); return v && [env, fns, { k: "return", v }]; }
        case "expr": { const v = evalE(g, env, fns, s[1], d + 1); return v && [env, fns, { k: "value", v }]; }
        case "break": return [env, fns, { k: "break" }]; case "continue": return [env, fns, { k: "continue" }];
      } return null;
    }
    function execB(f, env, fns, b, d) {
      if (b.length === 0) return [env, fns, { k: "value", v: vUnit }];
      if (b.length === 1) return execS(f, env, fns, b[0], d);
      const r = execS(f, env, fns, b[0], d); if (!r) return null;
      return r[2].k === "value" ? execB(f, r[0], r[1], b.slice(1), d) : r;
    }
    return isExpr ? evalE(fuel, env, fns, node, 0) : node[0] === "block" ? execB(fuel, env, fns, node[1], 0) : execS(fuel, env, fns, node, 0);
  }

  // The 45 witness theorems (Aether/Core.lean 1106-1580). [name, line, fuel, env, fns, node, expected]
  const E = (n, v) => [["x", vNum(n)]], idFn = [["id", { params: ["x"], body: [Ret(Vr("x"))] }]];
  const W = {
    expr: [
      ["evalExprWithFnsRel_num_sound", 1106, 1, [], [], N(7), vNum(7)],
      ["evalExprWithFnsRel_bool_sound", 1112, 1, [], [], Bo(true), vBool(true)],
      ["evalExprWithFnsRel_var_sound", 1118, 1, [["x", vStr("open")]], [], Vr("x"), vStr("open")],
      ["evalExprWithFnsRel_binary_add_sound", 1124, 2, [], [], Bin(N(2), "add", N(5)), vNum(7)],
      ["evalExprWithFnsRel_unary_not_sound", 1131, 2, [], [], Not(Bo(false)), vBool(true)],
      ["evalExprWithFnsRel_list_sound", 1138, 2, [], [], Li(N(1), Bo(true)), vList([vNum(1), vBool(true)])],
      ["evalExprWithFnsRel_index_sound", 1146, 3, [], [], Ix(Li(St("a"), St("b")), N(1)), vStr("b")],
      ["evalExprWithFnsRel_field_length_sound", 1154, 3, [], [], Fd(Li(N(1), N(2)), "length"), vNum(2)],
      ["evalExprWithFnsRel_method_len_sound", 1162, 2, [], [], Me(St("aether"), "len"), vNum(6)],
      ["evalExprWithFnsRel_call_return_sound", 1169, 3, [], idFn, Ca("id", [N(3)]), vNum(3)],
      ["evalExprWithFnsRel_call_value_sound", 1180, 3, [], [["one", { params: [], body: [Ex(N(1))] }]], Ca("one", []), vNum(1)],
    ],
    stmt: [
      ["stepStmtWithFns_let_num_exec_sound", 1191, 2, [], [], Let("x", N(7)), [E(7), "value", vNum(7)]],
      ["stepStmtWithFns_fn_decl_exec_sound", 1200, 1, [], [], Fn("id", ["x"], [Ret(Vr("x"))]), [[], "value", vUnit, 1]],
      ["stepStmtWithFns_return_var_exec_sound", 1214, 2, E(5), [], Ret(Vr("x")), [E(5), "return", vNum(5)]],
      ["stepStmtWithFns_assign_num_exec_sound", 1223, 2, E(1), [], As("x", N(9)), [E(9), "value", vNum(9)]],
      ["stepStmtWithFns_expr_bool_exec_sound", 1232, 2, [], [], Ex(Bo(true)), [[], "value", vBool(true)]],
      ["stepStmtWithFns_return_none_exec_sound", 1240, 1, [], [], Ret(null), [[], "return", vUnit]],
      ["stepStmtWithFns_break_exec_sound", 1248, 1, [], [], Brk, [[], "break"]],
      ["stepStmtWithFns_continue_exec_sound", 1256, 1, [], [], Cnt, [[], "continue"]],
    ],
    block: [
      ["stepBlockWithFns_nil_exec_sound", 1264, 1, [], [], ["block", []], [[], "value", vUnit]],
      ["stepBlockWithFns_single_expr_exec_sound", 1272, 2, [], [], ["block", [Ex(Bo(true))]], [[], "value", vBool(true)]],
      ["stepBlockWithFns_cons_value_exec_sound", 1280, 2, [], [], ["block", [Let("x", N(7)), Ex(Vr("x"))]], [E(7), "value", vNum(7)]],
      ["stepBlockWithFns_cons_return_exec_sound", 1291, 2, [], [], ["block", [Ret(N(1)), Ex(N(2))]], [[], "return", vNum(1)]],
      ["stepBlockWithFns_cons_break_exec_sound", 1301, 1, [], [], ["block", [Brk, Ex(N(2))]], [[], "break"]],
      ["stepBlockWithFns_cons_continue_exec_sound", 1309, 1, [], [], ["block", [Cnt, Ex(N(2))]], [[], "continue"]],
    ],
    if: [
      ["stepStmtWithFns_if_true_exec_sound", 1317, 3, [], [], If(Bo(true), [Let("x", N(1))], [Let("x", N(2))]), [E(1), "value", vNum(1)]],
      ["stepStmtWithFns_if_false_some_exec_sound", 1334, 3, [], [], If(Bo(false), [Let("x", N(1))], [Let("x", N(2))]), [E(2), "value", vNum(2)]],
      ["stepStmtWithFns_if_false_none_exec_sound", 1351, 2, [], [], If(Bo(false), [Let("x", N(1))], null), [[], "value", vUnit]],
    ],
    loop: [
      ["stepStmtWithFns_while_false_exec_sound", 1368, 2, E(0), [], Wh(Bo(false), [As("x", N(1))]), [E(0), "value", vUnit]],
      ["stepStmtWithFns_while_return_exec_sound", 1379, 3, [], [], Wh(Bo(true), [Ret(N(4))]), [[], "return", vNum(4)]],
      ["stepStmtWithFns_while_break_exec_sound", 1390, 2, [], [], Wh(Bo(true), [Brk]), [[], "value", vUnit]],
      ["stepStmtWithFns_for_done_exec_sound", 1401, 1, [], [], For("i", 2, 2, [Ex(N(9))]), [[["i", vNum(2)]], "value", vUnit]],
      ["stepStmtWithFns_for_return_exec_sound", 1412, 3, [], [], For("i", 0, 1, [Ret(Vr("i"))]), [[["i", vNum(0)]], "return", vNum(0)]],
      ["stepStmtWithFns_for_break_exec_sound", 1423, 2, [], [], For("i", 0, 1, [Brk]), [[["i", vNum(0)]], "value", vUnit]],
      ["stepStmtWithFns_for_value_exec_sound", 1434, 3, E(9), [], For("i", 0, 1, [As("x", Vr("i"))]), [[["i", vNum(1)], ["i", vNum(0)], ["x", vNum(0)]], "value", vUnit]],
      ["stepStmtWithFns_for_continue_exec_sound", 1447, 3, [], [], For("i", 0, 1, [Cnt]), [[["i", vNum(1)], ["i", vNum(0)]], "value", vUnit]],
      ["stepStmtWithFns_seal_until_done_exec_sound", 1458, 2, E(0), [], Seal(Bo(true), [As("x", N(1))]), [E(0), "value", vUnit]],
      ["stepStmtWithFns_seal_until_value_exec_sound", 1469, 3, E(0), [], Seal(Vr("x"), [As("x", N(1))]), [E(1), "value", vUnit]],
      ["stepStmtWithFns_seal_until_break_exec_sound", 1484, 2, E(0), [], Seal(Bo(false), [Brk]), [E(0), "value", vUnit]],
      ["stepStmtWithFns_seal_until_return_exec_sound", 1495, 3, E(0), [], Seal(Bo(false), [Ret(N(5))]), [E(0), "return", vNum(5)]],
      ["stepStmtWithFns_seal_until_continue_exec_sound", 1506, 3, E(0), [], Seal(Vr("x"), [As("x", N(1)), Cnt]), [E(1), "value", vUnit]],
      ["stepStmtWithFns_seal_value_exec_sound", 1521, 4, E(0), [], Seal(null, [If(Vr("x"), [Brk], [As("x", N(1))])]), [E(1), "value", vUnit]],
      ["stepStmtWithFns_seal_return_exec_sound", 1540, 3, E(0), [], Seal(null, [Ret(N(5))]), [E(0), "return", vNum(5)]],
      ["stepStmtWithFns_seal_break_exec_sound", 1551, 2, E(0), [], Seal(null, [Brk]), [E(0), "value", vUnit]],
      ["stepStmtWithFns_seal_continue_exec_sound", 1562, 4, E(0), [], Seal(null, [If(Vr("x"), [Brk], [As("x", N(1)), Cnt])]), [E(1), "value", vUnit]],
    ],
  };
  // Lean encodes Stmt.ret / Stmt.seal payloads with `some`; the helpers above take the bare expression.
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

  R("formal-witness", (stage, api) => {
    const el = root(stage, api);
    const group = W[stage.parentElement.dataset.group] || W.expr;
    const isExpr = group === W.expr;
    const sel = h("select", { "aria-label": "Witness theorem" });
    group.forEach(([n, line], i) => sel.append(h("option", { value: i }, `${n}  (Core.lean:${line})`)));
    const hyp = h("pre"), got = h("pre"), want = h("pre"), trace = h("div", { class: "fv-trace" }), verdict = h("p", { style: "margin:.5rem 0 0" });
    el.append(h("p", { class: "fv-k" }, "Theorem"), sel,
      h("div", { class: "fv-steps", style: "margin-top:.5rem" }, h("span", { class: "done" }, "intro _"), h("span", { class: "on" }, "native_decide")),
      h("div", { class: "fv-row" }, h("div", {}, h("p", { class: "fv-k" }, isExpr ? "Program and environment" : "Statement and starting env"), hyp),
        h("div", {}, h("p", { class: "fv-k" }, "Lean's right-hand side"), want),
        h("div", {}, h("p", { class: "fv-k" }, "This page's replay at chosen fuel"), got)),
      verdict, h("p", { class: "fv-k", style: "margin-top:.6rem" }, "Evaluation trace (replayed from Core.lean 533-682)"), trace);
    let fuel = 1, slider;
    const show = () => {
      const [, , f0, env, fns, node, exp] = group[+sel.value];
      hyp.textContent = `${isExpr ? fe(node) : node[0] === "block" ? fb(node[1]) : fs(node)}\nenv = ${fenv(env)}${fns.length ? `\nfns = [${fns.map(([n]) => `"${n}"`).join(", ")}]` : ""}`;
      want.textContent = `fuel ${f0}  ⟹  some (` + (isExpr ? fv(exp) : `${fenv(exp[0])}, ${exp.length > 3 ? exp[3] + ", " : ""}${exp[1] === "value" || exp[1] === "return" ? fflow({ k: exp[1], v: exp[2] }) : fflow({ k: exp[1] })}`) + ")";
      const tr = []; const r = run(fuel, env, fns, node, isExpr, tr);
      let proj = null;
      if (r) proj = isExpr ? r : [r[0], r[2].k, r[2].v, ...(exp.length > 3 ? [r[1].length] : [])].filter((x) => x !== undefined);
      const expN = isExpr ? exp : exp.filter((x) => x !== undefined);
      got.textContent = `fuel ${fuel}  ⟹  ` + (r ? `some (${isExpr ? fv(r) : `${fenv(r[0])}, ${exp.length > 3 ? r[1].length + ", " : ""}${fflow(r[2])}`})` : "none");
      const ok = r && same(proj, expN);
      verdict.innerHTML = ok ? `<span class="fv-chip fv-ok">matches Lean</span> <code>native_decide</code> compiles and runs this same executable computation and checks the equality (the hypothesis is discarded by <code>intro _</code>).`
        : `<span class="fv-chip fv-bad">${r ? "different result" : "none"}</span> At fuel ${fuel} the bounded executor ${r ? "stops early" : "runs out of fuel and returns <code>none</code>"}; Lean states the theorem at fuel ${f0}.`;
      trace.replaceChildren(...tr.slice(0, 80).map((t) => h("div", {}, t)));
    };
    sel.addEventListener("change", () => { fuel = group[+sel.value][2]; slider.value = fuel; slider.dispatchEvent(new Event("input")); });
    const c = api.controls();
    c.button("Previous", () => { sel.value = Math.max(0, +sel.value - 1); sel.dispatchEvent(new Event("change")); });
    c.button("Next", () => { sel.value = Math.min(group.length - 1, +sel.value + 1); sel.dispatchEvent(new Event("change")); });
    slider = c.slider("fuel", 0, 6, 1, group[0][2], (v) => { fuel = v; show(); });
  });

  // ---------- lookup_bind_same / eval_bound_var (Core.lean 1096-1104; defs 174-181, 417) ----------
  R("formal-lookup", (stage, api) => {
    const el = root(stage, api);
    let env = [["y", vNum(2)], ["x", vNum(1)]], name = "x", val = 5;
    const cells = h("div", { class: "fv-cells", "aria-live": "polite" }), res = h("p", { style: "margin:.4rem 0 0" });
    el.append(h("p", { class: "fv-k" }, "Concrete instance: Env.lookup (Env.bind env name value) name"), cells, res);
    const draw = (hitNew) => {
      cells.replaceChildren(...env.map(([k, v], i) => h("span", { class: i === 0 && hitNew ? "new" : "" }, `("${k}", ${fv(v)})`)));
      const spans = [...cells.children]; let i = 0;
      const step = () => { spans.forEach((s, j) => s.className = j < i ? "scan" : ""); if (i < spans.length) spans[i].className = env[i][0] === name ? "hit" : "scan";
        if (i < spans.length && env[i][0] !== name) { i++; setTimeout(step, api.still ? 0 : 280); } };
      step();
      const r = lookup(env, name);
      res.innerHTML = `Env.lookup … "${name}" = <code>${r ? "some (" + fv(r) + ")" : "none"}</code>` + (hitNew ? ` <span class="fv-chip fv-ok">head cell, first comparison</span>` : "");
    };
    el.append(h("div", { style: "height:.5rem" }));
    prover(el, api, [
      { tac: "lookup_bind_same", goal: esc(`Env.lookup (Env.bind env name value) name = some value`), note: "Core.lean:1096. Env.bind prepends (name, value) (line 180)." },
      { tac: "unfold Env.bind Env.lookup", goal: esc(`(match (name, value) :: env with\n | [] => none\n | (key, value) :: rest =>\n     if key == name then some value else Env.lookup rest name) = some value`), note: "Both definitions unfolded; the list is now visibly a cons cell whose key is name." },
      { tac: "simp", goal: "<mark>no goals</mark>", note: "simp reduces the match and the test name == name, closing the goal. Proof ends." },
      { tac: "eval_bound_var", goal: esc(`evalExpr (Env.bind env name value) (Expr.var name) = some value`), note: "Core.lean:1101." },
      { tac: "unfold evalExpr", goal: esc(`Env.lookup (Env.bind env name value) name = some value`), note: "evalExpr's var case is Env.lookup env name (line 417)." },
      { tac: "exact lookup_bind_same env name value", goal: "<mark>no goals</mark>", note: "The remaining goal is exactly the previous theorem." },
    ]).button("bind name value", () => { env = [[name, vNum(val)], ...env]; draw(true); });
    const c = api.controls();
    const pick = h("select", { "aria-label": "name" }); ["x", "y", "z"].forEach((n) => pick.append(h("option", {}, n)));
    pick.addEventListener("change", () => { name = pick.value; draw(false); });
    c.slider("value", -9, 9, 1, 5, (v) => { val = v; });
    el.parentElement.parentElement.querySelector(".ts-viz-controls").prepend(h("label", {}, "name ", pick));
    c.button("reset env", () => { env = [["y", vNum(2)], ["x", vNum(1)]]; draw(false); });
    draw(false);
  });

  // ---------- truthy (Core.lean 203-209) ----------
  R("formal-truthy", (stage, api) => {
    const el = root(stage, api);
    const src = ["def truthy : Value -> Bool", "  | Value.bool b => b", "  | Value.num n => n != 0", "  | Value.float intPart fracMicros => intPart != 0 || fracMicros != 0",
      "  | Value.str value => value != \"\"", "  | Value.list values => values != []", "  | Value.unit => false"];
    const pre = h("pre"), out = h("p", { style: "margin:.5rem 0 0" });
    el.append(h("p", { class: "fv-k" }, "Aether/Core.lean 203-209, the arm that fires is marked"), pre, out);
    let kind = "num", n = 0, b = false;
    const show = () => {
      const v = { bool: [vBool(b), 1, b], num: [vNum(n), 2, n !== 0], float: [{ k: "float" }, 3, n !== 0], str: [vStr("a".repeat(Math.max(0, n))), 4, n > 0],
        list: [vList(Array.from({ length: Math.max(0, n) }, (_, i) => vNum(i))), 5, n > 0], unit: [vUnit, 6, false] }[kind];
      pre.innerHTML = src.map((l, i) => i === v[1] ? `<mark>${esc(l)}</mark>` : esc(l)).join("\n");
      const shown = kind === "float" ? `Value.float ${n} 0` : fv(v[0]);
      out.innerHTML = `truthy (<code>${esc(shown)}</code>) = <span class="fv-chip ${v[2] ? "fv-ok" : "fv-bad"}">${v[2]}</span>`;
    };
    const c = api.controls();
    const pick = h("select", { "aria-label": "Value constructor" }); ["num", "float", "bool", "str", "list", "unit"].forEach((k) => pick.append(h("option", {}, k)));
    pick.addEventListener("change", () => { kind = pick.value; show(); });
    el.parentElement.parentElement.querySelector(".ts-viz-controls").append(h("label", {}, "constructor ", pick));
    c.slider("n / length", -3, 3, 1, 0, (v) => { n = v; show(); });
    c.button("toggle bool", () => { b = !b; show(); });
  });

  // ---------- compileCheckedFrameProgram_static_ok (VM.lean 799-814) ----------
  R("formal-checked", (stage, api) => {
    const el = root(stage, api);
    let ok = true;
    const flow = h("pre"); el.append(h("p", { class: "fv-k" }, "Concrete case: what compileCheckedFrameProgram returns"), flow, h("div", { style: "height:.5rem" }));
    const show = () => {
      flow.innerHTML = esc(`Static.checkProgramDetailed stmts = `) + (ok ? `<mark>Except.ok checked</mark>` : `<mark>Except.error found</mark>`) +
        esc(`\n  ⟹ compileCheckedFrameProgram stmts = `) + (ok ? esc("compileFrameProgram stmts") : "<mark>none</mark>") +
        (ok ? esc("\n  so h can hold, and the witness is ⟨checked, rfl⟩.") : esc("\n  so h : none = some result is impossible; simp [hs] at h closes this branch."));
    };
    prover(el, api, [
      { tac: "theorem …_static_ok", goal: esc(`h : compileCheckedFrameProgram stmts = some result\n⊢ ∃ checked, Static.checkProgramDetailed stmts = Except.ok checked`), note: "VM.lean:804. The definition (799-802) matches on the static checker first." },
      { tac: "unfold compileCheckedFrameProgram at h", goal: esc(`h : (match Static.checkProgramDetailed stmts with\n     | Except.ok _ => compileFrameProgram stmts\n     | Except.error _ => none) = some result`), note: "The hypothesis now exposes the match on the checker." },
      { tac: "cases hs : Static.checkProgramDetailed stmts", goal: esc(`case error:  hs : … = Except.error found\ncase ok:     hs : … = Except.ok checked`), note: "Split on the checker's result, remembering it as hs." },
      { tac: "error => simp [hs] at h", goal: esc(`h : none = some result   -- contradiction\n`) + "<mark>branch closed</mark>", note: "simp rewrites with hs; the error arm yields none, which cannot equal some result." },
      { tac: "ok => exact ⟨checked, rfl⟩", goal: "<mark>no goals</mark>", note: "In the ok branch the witness is checked itself, by reflexivity." },
    ]).button("toggle checker result", () => { ok = !ok; show(); });
    show();
  });

  // ---------- dependency graph (imports + theorem uses) ----------
  R("formal-deps", (stage, api) => {
    const NS = "http://www.w3.org/2000/svg";
    const nodes = {
      Lexer: [60, 40, "Aether/Lexer.lean"], Core: [60, 130, "Aether/Core.lean"], Static: [220, 130, "Aether/Static.lean"], Parser: [220, 40, "Aether/Parser.lean"],
      VM: [380, 90, "Aether/VM.lean"], Pipeline: [540, 60, "Aether/Pipeline.lean"], Aether: [540, 150, "Aether.lean (root)"],
      lbs: [60, 230, "lookup_bind_same"], ebv: [60, 300, "eval_bound_var"], wit: [250, 265, "45 witnesses (native_decide)"],
      sok: [470, 265, "…FrameProgram_static_ok"],
    };
    const edges = [["Parser", "Core"], ["Parser", "Lexer"], ["Static", "Core"], ["VM", "Core"], ["VM", "Static"], ["VM", "Parser"],
      ["Pipeline", "Lexer"], ["Pipeline", "Parser"], ["Pipeline", "Static"], ["Pipeline", "VM"], ["Aether", "Lexer"], ["Aether", "Core"], ["Aether", "Static"],
      ["Aether", "Parser"], ["Aether", "VM"], ["Aether", "Pipeline"], ["lbs", "Core"], ["ebv", "lbs"], ["ebv", "Core"], ["wit", "Core"], ["sok", "VM"], ["sok", "Static"]];
    const why = { lbs: "Uses Env.bind, Env.lookup (Core.lean 174-181). Proof: unfold; simp.", ebv: "Uses evalExpr and lookup_bind_same (exact). Core.lean 1101.",
      wit: "Each relates a Prop-level relation fact to the bounded executor at fixed fuel. Core.lean 1106-1580.",
      sok: "Uses Static.checkProgramDetailed via compileCheckedFrameProgram. VM.lean 804.",
      Aether: "Root module: imports all six.", Lexer: "No Aether imports.", Core: "No Aether imports." };
    const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", "0 0 640 340"); svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Module import and theorem dependency graph"); const wrap = h("div", { style: "overflow-x:auto" }); svg.style.minWidth = "560px"; wrap.append(svg); stage.append(wrap);
    const info = h("p", { class: "ts-viz-caption", style: "margin:.4rem 0 0", "aria-live": "polite" }, "Hover or focus a node to trace what it depends on."); stage.append(info);
    const mk = (t, a) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); return e; };
    const defs = mk("defs", {}); defs.innerHTML = `<marker id="fvArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="context-stroke"/></marker>`; svg.append(defs);
    const W2 = (k) => Math.max(92, nodes[k][2].length * 5.6 + 16) / 2;
    const eg = edges.map(([a, b]) => { const [x1, y1] = nodes[a], [x2, y2] = nodes[b];
      const l = mk("line", { class: "e", x1, y1, x2, y2, "stroke-width": 1.2, "marker-end": "url(#fvArrow)" }); svg.append(l);
      // shorten to box edge
      const dx = x2 - x1, dy = y2 - y1, t = Math.min(W2(b) / Math.abs(dx || 1e-9), 14 / Math.abs(dy || 1e-9), 1);
      l.setAttribute("x2", x2 - dx * t); l.setAttribute("y2", y2 - dy * t); return { a, b, l }; });
    const ng = {};
    for (const k in nodes) { const [x, y, label] = nodes[k], w = W2(k) * 2;
      const g = mk("g", { class: "n", tabindex: 0, role: "button", "aria-label": label }); const r = mk("rect", { x: x - w / 2, y: y - 13, width: w, height: 26, rx: 7, "stroke-width": 1.2 });
      const t = mk("text", { x, y: y + 4, "text-anchor": "middle", "font-size": 10.5, "font-family": "var(--md-code-font-family)" }); t.textContent = label;
      g.append(r, t); svg.append(g); ng[k] = { g, r, t }; }
    const deps = (k, s = new Set()) => { edges.forEach(({ 0: a, 1: b }) => { if (a === k && !s.has(b)) { s.add(b); deps(b, s); } }); return s; };
    let focus = null;
    const paint = () => { const th = api.theme(); const d = focus ? deps(focus) : null;
      for (const k in ng) { const on = focus && (k === focus || d.has(k)); ng[k].r.setAttribute("fill", k === focus ? th.accent : th.ground);
        ng[k].r.setAttribute("stroke", on ? th.accent : th.hair); ng[k].t.setAttribute("fill", k === focus ? th.ground : on ? th.accent : th.ink);
        ng[k].g.style.opacity = focus && !on ? .45 : 1; }
      eg.forEach(({ a, b, l }) => { const hot = focus && (a === focus || d.has(a)) && (b === focus || d.has(b));
        l.classList.toggle("hot", !!hot); l.setAttribute("stroke", hot ? th.accent : th.hair); l.style.opacity = focus && !hot ? .35 : 1; });
      info.textContent = focus ? `${nodes[focus][2]}: depends on ${[...d].map((k) => nodes[k][2]).join(", ") || "nothing inside Aether"}. ${why[focus] || ""}` : "Hover or focus a node to trace what it depends on."; };
    for (const k in ng) { const on = () => { focus = k; paint(); }, off = () => { focus = null; paint(); };
      ng[k].g.addEventListener("mouseenter", on); ng[k].g.addEventListener("focus", on); ng[k].g.addEventListener("mouseleave", off); ng[k].g.addEventListener("blur", off); }
    paint(); api.onTheme(paint);
  });
})();
