// viz-topology: visualisations for topology/*.md and MATHEMATICS.md.
// Every plot is computed here from the formula the page states; the Rust it
// mirrors is named beside each routine.
(() => {
  if (!window.TSViz) return;
  const reg = TSViz.register;

  // ---------- style (scoped to topo-* blocks) ----------
  const st = document.createElement("style");
  st.textContent = `
.ts-viz[data-viz^="topo-"] .topo-split{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:.8rem}
.ts-viz[data-viz^="topo-"] .topo-cell{position:relative;min-width:0}
.ts-viz[data-viz^="topo-"] .topo-cell canvas{display:block;width:100%;touch-action:none}
.ts-viz[data-viz^="topo-"] .topo-read{font-family:var(--md-code-font-family);font-size:.64rem;line-height:1.55;margin-top:.55rem;color:var(--md-default-fg-color--light);overflow-wrap:anywhere}
.ts-viz[data-viz^="topo-"] .topo-read b{color:var(--md-default-fg-color);font-weight:600}
.ts-viz[data-viz^="topo-"] .topo-read .on{color:var(--blue-500)}
.ts-viz[data-viz^="topo-"] .topo-read .bad{color:#c2410c}
[data-md-color-scheme=slate] .ts-viz[data-viz^="topo-"] .topo-read .bad{color:#ffb86b}
.ts-viz[data-viz^="topo-"] .ts-viz-controls textarea,.ts-viz[data-viz^="topo-"] .ts-viz-controls input[type=text]{font:inherit;font-family:var(--md-code-font-family);font-size:.62rem;width:100%;box-sizing:border-box;padding:.35rem .5rem;border-radius:8px;border:1px solid var(--hair2);background:var(--raised);color:var(--md-default-fg-color)}
.ts-viz[data-viz^="topo-"] .ts-viz-controls .topo-wide{flex:1 1 100%;display:flex;flex-direction:column;gap:.25rem}
.ts-viz[data-viz^="topo-"] .ts-viz-controls button[aria-pressed=true]{color:var(--blue-500);border-color:var(--blue-500)}`;
  document.head.append(st);

  // ---------- small DOM / canvas helpers ----------
  const el = (tag, cls, parent) => { const e = document.createElement(tag); if (cls) e.className = cls; if (parent) parent.append(e); return e; };
  function panel(parent, h, draw) {
    const d = el("div", "topo-cell", parent), cv = el("canvas", "", d);
    cv.style.height = h + "px";
    const ctx = cv.getContext("2d");
    const o = { cv, ctx, w: 0, h };
    o.paint = () => { if (!o.w) return; ctx.clearRect(0, 0, o.w, h); draw(ctx, o.w, h); };
    new ResizeObserver(() => {
      const dpr = Math.min(devicePixelRatio || 1, 2);
      o.w = d.clientWidth; cv.width = o.w * dpr; cv.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); o.paint();
    }).observe(d);
    return o;
  }
  // Readout sits under the canvases: re-appended after the mount code has added them.
  const readout = (stage) => { const r = el("div", "topo-read", stage); r.setAttribute("aria-live", "polite"); queueMicrotask(() => stage.append(r)); return r; };
  const f = (x, n = 3) => (x == null ? "∞" : !isFinite(x) ? "∞" : (+x).toFixed(n));
  const font = (ctx, px = 11) => { ctx.font = `${px}px ${getComputedStyle(document.body).getPropertyValue("--md-code-font-family") || "monospace"}`; };
  function toggle(ctrl, label, on, cb) {
    const b = ctrl.button(label, () => { on = !on; b.setAttribute("aria-pressed", on); cb(on); });
    b.setAttribute("aria-pressed", on); return b;
  }
  const mulberry = (s) => () => { s |= 0; s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const noisyCircle = (n, seed, noise = 0.05, cx = 0.5, cy = 0.5, rad = 0.32) => {
    const r = mulberry(seed);
    return Array.from({ length: n }, (_, i) => { const a = (i / n) * 2 * Math.PI + (r() - 0.5) * 0.4; const q = rad + (r() - 0.5) * 2 * noise; return [cx + q * Math.cos(a), cy + q * Math.sin(a)]; });
  };
  const dist = (a, b) => Math.hypot(...a.map((x, i) => x - b[i]));

  // Unit square [0,1]^2 <-> centred square in a panel.
  const frame = (w, h, pad = 12) => { const s = Math.min(w, h) - 2 * pad, ox = (w - s) / 2, oy = (h - s) / 2; return { s, px: (p) => [ox + p[0] * s, oy + (1 - p[1]) * s], un: (x, y) => [(x - ox) / s, 1 - (y - oy) / s] }; };
  // Drag points of pts (unit coords) inside panel pn; onMove after every move.
  function draggable(pn, pts, onMove, getFrame) {
    let k = -1;
    const at = (e) => { const r = pn.cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    pn.cv.addEventListener("pointerdown", (e) => {
      const [x, y] = at(e), F = getFrame(); let best = 18;
      pts.forEach((p, i) => { const [a, b] = F.px(p), d = Math.hypot(a - x, b - y); if (d < best) { best = d; k = i; } });
      if (k >= 0) { pn.cv.setPointerCapture(e.pointerId); e.preventDefault(); }
    });
    pn.cv.addEventListener("pointermove", (e) => {
      const [x, y] = at(e), F = getFrame();
      if (k < 0) { pn.cv.style.cursor = pts.some((p) => { const [a, b] = F.px(p); return Math.hypot(a - x, b - y) < 18; }) ? "grab" : "default"; return; }
      const u = F.un(x, y); pts[k] = [Math.min(1, Math.max(0, u[0])), Math.min(1, Math.max(0, u[1]))]; onMove(k);
    });
    const up = () => { k = -1; };
    pn.cv.addEventListener("pointerup", up); pn.cv.addEventListener("pointercancel", up);
  }

  // ---------- persistent homology (mirrors crates/aether-core/src/persistence.rs) ----------
  // Simplices up to dimension maxDim+1 (tetrahedra for max_dim=2), sorted by (filtration, dimension).
  function filtration(n, filt, maxDim = 2) {
    const S = [];
    for (let i = 0; i < n; i++) S.push({ v: [i], d: 0, f: 0 });
    const add = (v) => { const x = filt(v); if (x != null && isFinite(x)) S.push({ v, d: v.length - 1, f: x }); };
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      add([i, j]);
      if (maxDim >= 1) for (let k = j + 1; k < n; k++) {
        add([i, j, k]);
        if (maxDim >= 2) for (let l = k + 1; l < n; l++) add([i, j, k, l]);
      }
    }
    S.sort((a, b) => a.f - b.f || a.d - b.d);
    return S;
  }
  const vrFilt = (pts) => (v) => { let m = 0; for (let a = 0; a < v.length; a++) for (let b = a + 1; b < v.length; b++) m = Math.max(m, dist(pts[v[a]], pts[v[b]])); return m; };
  function xor(a, b) { const o = []; let i = 0, j = 0; while (i < a.length || j < b.length) { if (j >= b.length || (i < a.length && a[i] < b[j])) o.push(a[i++]); else if (i >= a.length || b[j] < a[i]) o.push(b[j++]); else { i++; j++; } } return o; }
  // Standard Z2 column reduction. Returns boundary, reduced columns, additions per column, and pairs.
  function reduce(S) {
    const key = new Map(S.map((s, i) => [s.v.join(), i]));
    const bd = [], cols = [], adds = [], owner = new Map(), killed = new Set(), pairs = [];
    for (let j = 0; j < S.length; j++) {
      const v = S[j].v; let c = [];
      if (v.length > 1) for (let k = 0; k < v.length; k++) c.push(key.get(v.filter((_, t) => t !== k).join()));
      c.sort((a, b) => a - b); bd[j] = c; adds[j] = [];
      while (c.length && owner.has(c[c.length - 1])) { const o = owner.get(c[c.length - 1]); adds[j].push(o); c = xor(c, cols[o]); }
      cols[j] = c;
      if (c.length) { const i = c[c.length - 1]; owner.set(i, j); killed.add(i); if (S[j].f > S[i].f) pairs.push({ dim: S[i].d, birth: S[i].f, death: S[j].f, i, j }); }
    }
    S.forEach((s, j) => { if (!cols[j].length && !killed.has(j) && s.d <= 2) pairs.push({ dim: s.d, birth: s.f, death: null, i: j }); });
    return { bd, cols, adds, pairs, owner };
  }
  const vrPH = (pts, maxDim = 2) => { const S = filtration(pts.length, vrFilt(pts), maxDim); return { S, ...reduce(S) }; };
  // PersistenceDiagram::betti_at: birth <= r && (death is None || r < death)
  const bettiAt = (pairs, r) => [0, 1, 2].map((k) => pairs.filter((p) => p.dim === k && p.birth <= r && (p.death == null || r < p.death)).length);

  // Bottleneck distance per dimension (binary search + Kuhn matching with diagonal copies).
  function bottleneck(P, Q) {
    let out = 0;
    for (const k of [0, 1, 2]) {
      const A = P.filter((p) => p.dim === k), B = Q.filter((p) => p.dim === k);
      const ea = A.filter((p) => p.death == null).map((p) => p.birth).sort((a, b) => a - b), eb = B.filter((p) => p.death == null).map((p) => p.birth).sort((a, b) => a - b);
      if (ea.length !== eb.length) return Infinity;
      ea.forEach((b, i) => { out = Math.max(out, Math.abs(b - eb[i])); });
      const a = A.filter((p) => p.death != null), b = B.filter((p) => p.death != null), m = a.length, n = b.length;
      if (!m && !n) continue;
      const li = (p, q) => Math.max(Math.abs(p.birth - q.birth), Math.abs(p.death - q.death)), dg = (p) => (p.death - p.birth) / 2;
      const cand = [0]; a.forEach((p) => { cand.push(dg(p)); b.forEach((q) => cand.push(li(p, q))); }); b.forEach((q) => cand.push(dg(q)));
      cand.sort((x, y) => x - y);
      const ok = (e) => {
        const adj = []; for (let i = 0; i < m + n; i++) { const r = []; if (i < m) { for (let j = 0; j < n; j++) if (li(a[i], b[j]) <= e + 1e-12) r.push(j); if (dg(a[i]) <= e + 1e-12) r.push(n + i); } else { const j = i - m; if (dg(b[j]) <= e + 1e-12) r.push(j); for (let t = 0; t < m; t++) r.push(n + t); } adj.push(r); }
        const mt = new Array(m + n).fill(-1);
        const aug = (u, seen) => { for (const v of adj[u]) { if (seen[v]) continue; seen[v] = 1; if (mt[v] < 0 || aug(mt[v], seen)) { mt[v] = u; return true; } } return false; };
        for (let u = 0; u < m + n; u++) if (!aug(u, [])) return false;
        return true;
      };
      let lo = 0, hi = cand.length - 1; while (lo < hi) { const mid = (lo + hi) >> 1; if (ok(cand[mid])) hi = mid; else lo = mid + 1; }
      out = Math.max(out, cand[lo]);
    }
    return out;
  }

  // ---------- shared drawing ----------
  const dimCol = (T, d) => [T.accent, T.accent2, T.good][d];
  function drawComplex(ctx, F, pts, S, r, T, opts = {}) {
    ctx.save();
    if (opts.balls) { ctx.fillStyle = T.accent; ctx.globalAlpha = T.dark ? 0.07 : 0.06; pts.forEach((p) => { const [x, y] = F.px(p); ctx.beginPath(); ctx.arc(x, y, (r / 2) * F.s, 0, 7); ctx.fill(); }); }
    ctx.globalAlpha = T.dark ? 0.22 : 0.16; ctx.fillStyle = T.accent;
    S.forEach((s) => { if (s.d === 2 && s.f <= r) { ctx.beginPath(); s.v.forEach((i, k) => { const [x, y] = F.px(pts[i]); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.closePath(); ctx.fill(); } });
    ctx.globalAlpha = 0.85; ctx.strokeStyle = T.accent; ctx.lineWidth = 1.2;
    S.forEach((s) => { if (s.d === 1 && s.f <= r) { const [a, b] = F.px(pts[s.v[0]]), [c, d] = F.px(pts[s.v[1]]); ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke(); } });
    ctx.globalAlpha = 1;
    pts.forEach((p, i) => { const [x, y] = F.px(p); ctx.beginPath(); ctx.arc(x, y, opts.big && opts.big(i) ? 5.5 : 3.6, 0, 7); ctx.fillStyle = opts.color ? opts.color(i) : T.ink; ctx.fill(); ctx.strokeStyle = T.ground; ctx.lineWidth = 1.5; ctx.stroke(); });
    ctx.restore();
  }
  function drawBarcode(ctx, w, h, pairs, rmax, r, T, label = true) {
    const bars = pairs.slice().sort((a, b) => a.dim - b.dim || a.birth - b.birth || (a.death ?? 9) - (b.death ?? 9));
    const L = 30, R = w - 10, top = 8, bot = h - 20, n = Math.max(bars.length, 1), gap = Math.min(10, (bot - top) / n);
    const X = (v) => L + (Math.min(v, rmax) / rmax) * (R - L);
    font(ctx, 10); ctx.fillStyle = T.muted; ctx.strokeStyle = T.hair; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, bot + 2); ctx.lineTo(R, bot + 2); ctx.stroke();
    for (let t = 0; t <= 4; t++) { const v = (rmax * t) / 4; ctx.fillText(v.toFixed(2), Math.min(X(v) - 10, w - 30), h - 5); }
    let prev = -1;
    bars.forEach((p, i) => {
      const y = top + i * gap + gap / 2, live = p.birth <= r && (p.death == null || r < p.death);
      if (label && p.dim !== prev) { ctx.fillStyle = dimCol(T, p.dim); ctx.fillText("H" + p.dim, 2, y + 3); prev = p.dim; }
      ctx.strokeStyle = dimCol(T, p.dim); ctx.globalAlpha = live ? 1 : 0.35; ctx.lineWidth = Math.max(1.5, Math.min(4, gap - 3));
      const x2 = p.death == null ? R : X(p.death);
      ctx.beginPath(); ctx.moveTo(X(p.birth), y); ctx.lineTo(x2, y); ctx.stroke();
      if (p.death == null) { ctx.beginPath(); ctx.moveTo(R, y); ctx.lineTo(R - 5, y - 3); ctx.moveTo(R, y); ctx.lineTo(R - 5, y + 3); ctx.lineWidth = 1.2; ctx.stroke(); }
    });
    ctx.globalAlpha = 1;
    if (r != null) { ctx.strokeStyle = T.ink; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(r), top - 4); ctx.lineTo(X(r), bot + 2); ctx.stroke(); ctx.setLineDash([]); }
  }
  function drawDiagram(ctx, w, h, sets, rmax, T) {
    const s = Math.min(w, h) - 36, L = (w - s) / 2 + 10, B = h - 22, X = (v) => L + (Math.min(v, rmax) / rmax) * s, Y = (v) => B - (Math.min(v, rmax) / rmax) * s;
    font(ctx, 10); ctx.strokeStyle = T.hair; ctx.lineWidth = 1;
    ctx.strokeRect(L, B - s, s, s); ctx.beginPath(); ctx.moveTo(L, B); ctx.lineTo(L + s, B - s); ctx.stroke();
    ctx.fillStyle = T.muted; ctx.fillText("birth →", L + s - 44, B + 13); ctx.save(); ctx.translate(L - 6, B - s + 44); ctx.rotate(-Math.PI / 2); ctx.fillText("death →", 0, 0); ctx.restore();
    ctx.fillText("∞", L - 12, B - s + 4);
    sets.forEach(({ pairs, hollow }) => pairs.forEach((p) => {
      const x = X(p.birth), y = p.death == null ? B - s : Y(p.death);
      ctx.beginPath(); ctx.arc(x, y, 3.8, 0, 7);
      if (hollow) { ctx.strokeStyle = dimCol(T, p.dim); ctx.lineWidth = 1.3; ctx.stroke(); } else { ctx.fillStyle = dimCol(T, p.dim); ctx.fill(); }
    }));
  }

  // ======================= persistent-homology.md =======================

  // Input: point cloud X.
  reg("topo-cloud", (stage, api) => {
    const pts = noisyCircle(12, 7), out = readout(stage);
    let F;
    const pn = panel(stage, 240, (ctx, w, h) => { const T = api.theme(); F = frame(w, h); drawComplex(ctx, F, pts, [], 0, T); pts.forEach((p, i) => { const [x, y] = F.px(p); font(ctx, 10); ctx.fillStyle = T.muted; ctx.fillText("x" + (i + 1), x + 6, y - 6); }); });
    const upd = () => { pn.paint(); out.innerHTML = `X = {x<sub>1</sub>…x<sub>${pts.length}</sub>}, each a ManifoldPoint&lt;2&gt;: ` + pts.slice(0, 3).map((p, i) => `x${i + 1}=(${f(p[0], 2)}, ${f(p[1], 2)})`).join(" ") + " …"; };
    draggable(pn, pts, upd, () => F); api.onTheme(upd); upd();
  });

  // Vietoris-Rips complex at radius r.
  reg("topo-rips", (stage, api) => {
    const pts = noisyCircle(12, 3, 0.06);
    let r = 0.3, F, S = filtration(pts.length, vrFilt(pts), 2);
    const out = readout(stage);
    const pn = panel(stage, 280, (ctx, w, h) => { F = frame(w, h); drawComplex(ctx, F, pts, S, r, api.theme(), { balls: true }); });
    const upd = () => {
      pn.paint(); const c = [0, 0, 0, 0]; S.forEach((s) => { if (s.f <= r) c[s.d]++; });
      out.innerHTML = `VR<sub>${f(r, 2)}</sub>(X): <b>${c[0]}</b> vertices · <b>${c[1]}</b> edges (d ≤ r) · <b>${c[2]}</b> triangles (all 3 edges ≤ r) · <b>${c[3]}</b> tetrahedra (all 6 edges ≤ r). Discs have radius r/2: an edge exists exactly when two discs touch.`;
    };
    draggable(pn, pts, () => { S = filtration(pts.length, vrFilt(pts), 2); upd(); }, () => F);
    const c = api.controls(); const sl = c.slider("radius r", 0, 0.7, 0.01, r, (v) => { r = v; upd(); });
    let sweep = false, t0 = 0;
    if (!api.still) toggle(c, "Sweep filtration", false, (on) => { sweep = on; t0 = performance.now(); });
    api.loop(() => { if (!sweep) return; r = ((performance.now() - t0) / 6000 % 1) * 0.7; sl.value = r; sl.dispatchEvent(new Event("input")); });
    api.onTheme(upd);
  });

  // Lazy witness vs Vietoris-Rips.
  reg("topo-witness", (stage, api) => {
    const X = noisyCircle(40, 11, 0.05);
    let nL = 8, r = 0.15, L = [], WS = [], VS = [];
    const maxmin = (k) => { const sel = [0]; while (sel.length < Math.min(k, X.length)) { let bi = -1, bd = -1; X.forEach((p, i) => { if (sel.includes(i)) return; const d = Math.min(...sel.map((s) => dist(p, X[s]))); if (d > bd) { bd = d; bi = i; } }); sel.push(bi); } return sel; };
    const build = () => {
      L = maxmin(nL); const Lp = L.map((i) => X[i]);
      const dWL = X.map((w) => Lp.map((l) => dist(w, l))), near = dWL.map((row) => Math.min(...row));
      // witness_filtration: min_w max(max_{l in sigma} d(w,l) - d(w,L), 0)
      const wf = (v) => { let best = Infinity; dWL.forEach((row, wi) => { let m = 0; v.forEach((i) => { m = Math.max(m, row[i]); }); best = Math.min(best, Math.max(m - near[wi], 0)); }); return best; };
      WS = filtration(nL, wf, 2); VS = filtration(X.length, vrFilt(X), 1);
    };
    const out = readout(stage), g = el("div", "topo-split", stage);
    let F1, F2;
    const p1 = panel(g, 240, (ctx, w, h) => { const T = api.theme(); F1 = frame(w, h); drawComplex(ctx, F1, X, VS, r, T); font(ctx); ctx.fillStyle = T.muted; ctx.fillText("Vietoris–Rips on all " + X.length + " points", 6, 14); });
    const p2 = panel(g, 240, (ctx, w, h) => {
      const T = api.theme(); F2 = frame(w, h); const Lp = L.map((i) => X[i]);
      X.forEach((p) => { const [x, y] = F2.px(p); ctx.beginPath(); ctx.arc(x, y, 2, 0, 7); ctx.fillStyle = T.hair; ctx.fill(); });
      drawComplex(ctx, F2, Lp, WS, r, T, { color: () => T.accent2, big: () => true });
      font(ctx, 10); ctx.fillStyle = T.muted; Lp.forEach((p, i) => { const [x, y] = F2.px(p); ctx.fillText(i + 1, x + 7, y - 6); });
      font(ctx); ctx.fillText(`witness complex on ${nL} maxmin landmarks`, 6, 14);
    });
    const upd = () => {
      p1.paint(); p2.paint();
      const cv = [0, 0, 0], cw = [0, 0, 0, 0]; VS.forEach((s) => { if (s.f <= r) cv[s.d]++; }); WS.forEach((s) => { if (s.f <= r) cw[s.d]++; });
      const bw = bettiAt(reduce(WS).pairs, r);
      out.innerHTML = `r = ${f(r, 2)} · Rips: <b>${cv[0] + cv[1] + cv[2]}</b> simplices (${cv.join(" / ")} of dim 0/1/2) · Witness: <b>${cw[0] + cw[1] + cw[2] + cw[3]}</b> simplices (${cw.join(" / ")} of dim 0–3), β = (${bw.join(", ")}). Grey dots are witnesses only; landmark numbers give maxmin order.`;
    };
    const c = api.controls();
    c.slider("landmarks |L|", 3, 12, 1, nL, (v) => { nL = v; build(); upd(); });
    c.slider("radius r", 0, 0.5, 0.01, r, (v) => { r = v; upd(); });
    api.onTheme(upd);
  });

  // Boundary-matrix reduction over Z2, stepped column by column.
  reg("topo-matrix", (stage, api) => {
    const pts = [[0.2, 0.5], [0.38, 0.8], [0.66, 0.78], [0.82, 0.48], [0.62, 0.2], [0.34, 0.22]];
    const S = filtration(pts.length, vrFilt(pts), 1), R = reduce(S), N = S.length;
    let k = 0;
    const out = readout(stage), g = el("div", "topo-split", stage);
    const name = (s) => (["v", "e", "t"][s.d]) + s.v.map((x) => x + 1).join("");
    const pm = panel(g, 300, (ctx, w, h) => {
      const T = api.theme(), s = Math.min(w - 20, h - 20) / N, ox = (w - s * N) / 2, oy = 10;
      ctx.strokeStyle = T.hair; ctx.lineWidth = 1; ctx.strokeRect(ox, oy, s * N, s * N);
      for (let j = 0; j < N; j++) {
        const col = j < k ? R.cols[j] : R.bd[j];
        col.forEach((i, t) => { ctx.fillStyle = j < k && t === col.length - 1 ? T.accent2 : j === k - 1 ? T.accent : T.muted; ctx.globalAlpha = j < k || j === k - 1 ? 1 : 0.45; ctx.fillRect(ox + j * s + 0.5, oy + i * s + 0.5, s - 1, s - 1); });
      }
      ctx.globalAlpha = 1;
      if (k > 0) { ctx.strokeStyle = T.accent; ctx.lineWidth = 1.5; ctx.strokeRect(ox + (k - 1) * s, oy, s, N * s); }
    });
    let F;
    const pc = panel(g, 300, (ctx, w, h) => {
      const T = api.theme(); F = frame(w, h, 24); const r = k ? S[k - 1].f : 0;
      drawComplex(ctx, F, pts, S.slice(0, k), Infinity, T);
      font(ctx, 10); ctx.fillStyle = T.muted; pts.forEach((p, i) => { const [x, y] = F.px(p); ctx.fillText(i + 1, x + 6, y - 6); });
      if (k) { ctx.fillText(`filtration value ${f(r)}`, 6, 14); }
    });
    const upd = () => {
      pm.paint(); pc.paint();
      if (!k) { out.innerHTML = `${N} columns (6 vertices, 15 edges, 20 triangles) sorted by (filtration, dimension). Grey = unreduced boundary ∂; blue = current column; orange = low pivot of a reduced column.`; return; }
      const j = k - 1, s = S[j], col = R.cols[j];
      let msg = `column ${k}/${N}: <b>${name(s)}</b> at f=${f(s.f)}` + (R.adds[j].length ? ` · added columns ${R.adds[j].map((a) => name(S[a])).join(", ")}` : "");
      if (!col.length) msg += ` · reduced column is <span class="on">empty → births an H${s.d} feature</span>`;
      else { const i = col[col.length - 1]; msg += ` · low pivot ${name(S[i])} → <span class="bad">kills the H${S[i].d} feature born at ${f(S[i].f)}</span>` + (S[i].f === s.f ? " (zero-length, dropped)" : ""); }
      const done = R.pairs.filter((p) => p.death != null && p.j < k).length;
      out.innerHTML = msg + ` · finite pairs so far: ${done}`;
    };
    const c = api.controls();
    c.button("Step", () => { k = Math.min(N, k + 1); upd(); });
    c.button("Back", () => { k = Math.max(0, k - 1); upd(); });
    c.button("Reset", () => { k = 0; upd(); });
    c.button("Run all", () => { k = N; upd(); });
    api.onTheme(upd); upd();
  });

  // PersistenceDiagram output + stability under perturbation.
  reg("topo-diagram", (stage, api) => {
    const base = noisyCircle(10, 5, 0.05), pts = base.map((p) => p.slice());
    const rmax = 0.8;
    let r = 0.25, P0 = vrPH(base).pairs, cur = vrPH(pts);
    const out = readout(stage), g = el("div", "topo-split", stage);
    let F;
    const pc = panel(g, 250, (ctx, w, h) => {
      const T = api.theme(); F = frame(w, h);
      ctx.save(); ctx.strokeStyle = T.hair; base.forEach((p, i) => { const [a, b] = F.px(p), [c2, d] = F.px(pts[i]); ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c2, d); ctx.stroke(); ctx.beginPath(); ctx.arc(a, b, 2.5, 0, 7); ctx.stroke(); }); ctx.restore();
      drawComplex(ctx, F, pts, cur.S, r, T);
    });
    const pd = panel(g, 250, (ctx, w, h) => drawDiagram(ctx, w, h, [{ pairs: P0, hollow: true }, { pairs: cur.pairs }], rmax, api.theme()));
    const pb = panel(stage, 150, (ctx, w, h) => drawBarcode(ctx, w, h, cur.pairs, rmax, r, api.theme()));
    const upd = () => {
      pc.paint(); pd.paint(); pb.paint();
      const delta = Math.max(...pts.map((p, i) => dist(p, base[i]))), dB = bottleneck(P0, cur.pairs);
      const fin = cur.pairs.filter((p) => p.death != null).length;
      out.innerHTML = `PersistencePair count: <b>${cur.pairs.length}</b> (${fin} finite, ${cur.pairs.length - fin} essential with death = None) · bottleneck distance to the original diagram d<sub>B</sub> = <b>${f(dB)}</b> ≤ 2·max‖x<sub>i</sub>−x′<sub>i</sub>‖ = ${f(2 * delta)} <span class="${dB <= 2 * delta + 1e-9 ? "on" : "bad"}">${dB <= 2 * delta + 1e-9 ? "✓ stability holds" : "✗"}</span>. Hollow = original, filled = current.`;
    };
    const recompute = () => { cur = vrPH(pts); upd(); };
    draggable(pc, pts, recompute, () => F);
    const c = api.controls(), rnd = mulberry(99);
    c.slider("radius r", 0, rmax, 0.01, r, (v) => { r = v; pb.paint(); pc.paint(); });
    c.button("Jitter points", () => { pts.forEach((p, i) => { p[0] = base[i][0] + (rnd() - 0.5) * 0.08; p[1] = base[i][1] + (rnd() - 0.5) * 0.08; }); recompute(); });
    c.button("Reset", () => { base.forEach((p, i) => { pts[i] = p.slice(); }); recompute(); });
    api.onTheme(upd); upd();
  });

  // Betti query beta_k(r).
  reg("topo-betti", (stage, api) => {
    const pts = noisyCircle(8, 21, 0.05, 0.3, 0.5, 0.2).concat(noisyCircle(5, 4, 0.02, 0.78, 0.5, 0.1));
    const { pairs } = vrPH(pts), rmax = 0.7;
    let r = 0.2;
    const out = readout(stage);
    const pn = panel(stage, 230, (ctx, w, h) => {
      const T = api.theme(), L = 30, R = w - 10, top = 10, n = 3, bh = (h - 30) / n, X = (v) => L + (v / rmax) * (R - L);
      const steps = 400;
      for (let k = 0; k < 3; k++) {
        const vals = []; for (let s = 0; s <= steps; s++) vals.push(bettiAt(pairs, (s / steps) * rmax)[k]);
        const mx = Math.max(2, ...vals), y0 = top + (k + 1) * bh - 6, Y = (v) => y0 - (v / mx) * (bh - 14);
        ctx.strokeStyle = T.hair; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L, y0); ctx.lineTo(R, y0); ctx.stroke();
        font(ctx, 10); ctx.fillStyle = dimCol(T, k); ctx.fillText("β" + k, 4, y0 - bh / 3);
        ctx.strokeStyle = dimCol(T, k); ctx.lineWidth = 1.8; ctx.beginPath();
        vals.forEach((v, s) => { const x = X((s / steps) * rmax); s ? (ctx.lineTo(x, Y(vals[s - 1])), ctx.lineTo(x, Y(v))) : ctx.moveTo(x, Y(v)); }); ctx.stroke();
        ctx.fillStyle = T.muted; ctx.fillText(mx, L - 14, Y(mx) + 4);
      }
      font(ctx, 10); ctx.fillStyle = T.muted; for (let t = 0; t <= 4; t++) ctx.fillText(((rmax * t) / 4).toFixed(2), Math.min(X((rmax * t) / 4) - 10, w - 30), h - 4);
      ctx.strokeStyle = T.ink; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(r), top); ctx.lineTo(X(r), h - 16); ctx.stroke(); ctx.setLineDash([]);
    });
    const pb = panel(stage, 150, (ctx, w, h) => drawBarcode(ctx, w, h, pairs, rmax, r, api.theme()));
    const upd = () => {
      pn.paint(); pb.paint(); const b = bettiAt(pairs, r);
      out.innerHTML = `β(${f(r, 2)}) = (<b>${b[0]}</b>, <b>${b[1]}</b>, <b>${b[2]}</b>): each count is the number of bars of that dimension with b<sub>i</sub> ≤ r &lt; d<sub>i</sub> (bold bars). Arrowed bars are essential (death = None) and stay live after birth.`;
    };
    api.controls().slider("radius r", 0, rmax, 0.005, r, (v) => { r = v; upd(); });
    api.onTheme(upd);
  });

  // ======================= derivations.md =======================

  // Time-delay embedding (TimeDelayEmbedder<3>): emits once buffer_len >= D*tau.
  reg("topo-delay", (stage, api) => {
    const N = 240, D = 3, x = Array.from({ length: N }, (_, n) => Math.sin((2 * Math.PI * n) / 48) + 0.45 * Math.sin((2 * Math.PI * n) / 17 + 1));
    let tauIn = 12, tau = 12, t = N - 1, yaw = 0.6, pitch = 0.35, spin = !api.still;
    const emb = () => { const P = []; for (let n = 0; n < N; n++) if (n + 1 >= D * tau) P.push({ n, p: [x[n], x[n - tau], x[n - 2 * tau]] }); return P; };
    let P = emb();
    const out = readout(stage);
    const ps = panel(stage, 110, (ctx, w, h) => {
      const T = api.theme(), X = (n) => 8 + (n / (N - 1)) * (w - 16), Y = (v) => h / 2 - v * (h / 3.4);
      ctx.fillStyle = T.hair; ctx.globalAlpha = 0.5; ctx.fillRect(X(0), 4, X(D * tau - 1) - X(0), h - 8); ctx.globalAlpha = 1;
      ctx.strokeStyle = T.ink; ctx.lineWidth = 1.3; ctx.beginPath(); x.forEach((v, n) => (n ? ctx.lineTo(X(n), Y(v)) : ctx.moveTo(X(n), Y(v)))); ctx.stroke();
      if (t + 1 >= D * tau) [0, 1, 2].forEach((i) => { const n = t - i * tau; ctx.strokeStyle = T.accent; ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.moveTo(X(n), 4); ctx.lineTo(X(n), h - 4); ctx.stroke(); ctx.globalAlpha = 1; ctx.beginPath(); ctx.arc(X(n), Y(x[n]), 4, 0, 7); ctx.fillStyle = T.accent; ctx.fill(); });
      font(ctx, 10); ctx.fillStyle = T.muted; ctx.fillText("x(t); shaded samples come before the first emitted point", 8, 14);
    });
    const p3 = panel(stage, 280, (ctx, w, h) => {
      const T = api.theme(), s = Math.min(w, h) * 0.26, cx = w / 2, cy = h / 2;
      const pr = (p) => { const [a, b, c] = p, x1 = a * Math.cos(yaw) - c * Math.sin(yaw), z1 = a * Math.sin(yaw) + c * Math.cos(yaw), y1 = b * Math.cos(pitch) - z1 * Math.sin(pitch), z2 = b * Math.sin(pitch) + z1 * Math.cos(pitch); return [cx + x1 * s, cy - y1 * s, z2]; };
      ctx.strokeStyle = T.hair; ctx.lineWidth = 1; font(ctx, 10);
      [[1.6, 0, 0, "x(t)"], [0, 1.6, 0, "x(t−τ)"], [0, 0, 1.6, "x(t−2τ)"]].forEach(([a, b, c, l]) => { const o = pr([0, 0, 0]), e = pr([a, b, c]); ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(e[0], e[1]); ctx.stroke(); ctx.fillStyle = T.muted; ctx.fillText(l, e[0] + 3, e[1]); });
      ctx.strokeStyle = T.accent; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.8; ctx.beginPath();
      P.forEach((q, i) => { const [a, b] = pr(q.p); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); }); ctx.stroke(); ctx.globalAlpha = 1;
      const cur = P.find((q) => q.n === t); if (cur) { const [a, b] = pr(cur.p); ctx.beginPath(); ctx.arc(a, b, 5, 0, 7); ctx.fillStyle = T.accent2; ctx.fill(); }
    });
    const upd = () => {
      ps.paint(); p3.paint();
      const cur = P.find((q) => q.n === t);
      out.innerHTML = `τ = ${tauIn}${tauIn === 0 ? " → normalised to 1" : ""}, D = 3 · ${N} samples → <b>${P.length}</b> points (first after ${D * tau} samples, the buffer_len ≥ D·τ check in TimeDelayEmbedder::embed) · ` + (cur ? `Φ(${t}) = [${cur.p.map((v) => f(v, 2)).join(", ")}]` : `t=${t}: not enough samples yet`);
    };
    // drag to rotate
    let drag = null;
    p3.cv.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY]; p3.cv.setPointerCapture(e.pointerId); spin = false; });
    p3.cv.addEventListener("pointermove", (e) => { if (!drag) return; yaw += (e.clientX - drag[0]) * 0.01; pitch = Math.max(-1.4, Math.min(1.4, pitch + (e.clientY - drag[1]) * 0.01)); drag = [e.clientX, e.clientY]; p3.paint(); });
    p3.cv.addEventListener("pointerup", () => { drag = null; }); p3.cv.style.cursor = "grab";
    const c = api.controls();
    c.slider("delay τ", 0, 24, 1, tauIn, (v) => { tauIn = v; tau = v === 0 ? 1 : v; P = emb(); upd(); });
    const ts = c.slider("time t", 0, N - 1, 1, t, (v) => { t = v; upd(); });
    c.button("Rotate ←", () => { yaw -= 0.3; p3.paint(); }); c.button("Rotate →", () => { yaw += 0.3; p3.paint(); });
    let last = 0;
    api.loop((s) => { if (!spin) return; yaw += (s - last) * 0.25; last = s; p3.paint(); });
    api.onTheme(upd);
  });

  // Euclidean distance.
  reg("topo-distance", (stage, api) => {
    const pts = [[0.22, 0.25], [0.74, 0.7]], out = readout(stage); let F;
    const pn = panel(stage, 220, (ctx, w, h) => {
      const T = api.theme(); F = frame(w, h, 16); const [a, b] = F.px(pts[0]), [c, d] = F.px(pts[1]);
      ctx.strokeStyle = T.hair; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, b); ctx.lineTo(c, d); ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = T.accent; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke();
      font(ctx, 11); ctx.fillStyle = T.muted; ctx.fillText("p₁−q₁", (a + c) / 2 - 14, b + (d < b ? 14 : -6)); ctx.fillText("p₂−q₂", c + 6, (b + d) / 2);
      [[a, b, "p"], [c, d, "q"]].forEach(([x, y, l]) => { ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); ctx.fillStyle = T.ink; ctx.fill(); ctx.fillText(l, x - 14, y - 6); });
    });
    const upd = () => { pn.paint(); const [p, q] = pts, dx = p[0] - q[0], dy = p[1] - q[1]; out.innerHTML = `D = 2: d(p,q) = √((${f(dx)})² + (${f(dy)})²) = <b>${f(Math.hypot(dx, dy))}</b>. Drag p or q.`; };
    draggable(pn, pts, upd, () => F);
    const c = api.controls(); c.button("Swap p and q", () => { pts.reverse(); upd(); });
    api.onTheme(upd); upd();
  });

  // BlockMetadata<D>::from_points: centroid, radius, variance, concentration. data-focus picks the highlighted quantity.
  function blockStats(P) {
    const n = P.length, mu = [0, 0]; P.forEach((p) => { mu[0] += p[0] / n; mu[1] += p[1] / n; });
    const ds = P.map((p) => dist(p, mu)), nm = Math.hypot(...mu);
    const mean = ds.reduce((a, b) => a + b, 0) / n, var_ = ds.reduce((a, b) => a + b * b, 0) / n - mean * mean;
    let cs = 0; P.forEach((p) => { const np = Math.hypot(...p); if (np > 0 && nm > 0) cs += (p[0] * mu[0] + p[1] * mu[1]) / (np * nm); });
    return { mu, r: Math.max(...ds), ds, mean, var: Math.max(0, var_), conc: cs / n, nm };
  }
  // Points live in [-1,1]^2 with the origin at the panel centre (cosines and norms need an origin).
  const cframe = (w, h, pad = 14) => { const s = (Math.min(w, h) - 2 * pad) / 2, cx = w / 2, cy = h / 2; return { s, px: (p) => [cx + p[0] * s, cy - p[1] * s], un: (x, y) => [(x - cx) / s, (cy - y) / s] }; };
  function cdrag(pn, pts, cb, getF) { // wraps draggable for centred coords
    const proxy = { get length() { return pts.length; } };
    draggable(pn, new Proxy(pts, { set(t, k, v) { t[k] = [v[0] * 2 - 1, v[1] * 2 - 1]; return true; } }), cb, () => { const F = getF(); return { px: F.px, un: (x, y) => { const u = F.un(x, y); return [(u[0] + 1) / 2, (u[1] + 1) / 2]; } }; });
    return proxy;
  }
  reg("topo-block", (stage, api) => {
    const focus = stage.parentElement.dataset.focus || "centroid";
    const pts = [[0.35, 0.55], [0.6, 0.72], [0.52, 0.3], [0.78, 0.45], [0.2, 0.28], [0.45, 0.85], [0.7, 0.15]], out = readout(stage); let F;
    const pn = panel(stage, 260, (ctx, w, h) => {
      const T = api.theme(); F = cframe(w, h); const B = blockStats(pts), m = F.px(B.mu), o = F.px([0, 0]);
      ctx.strokeStyle = T.hair; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(o[0] - F.s, o[1]); ctx.lineTo(o[0] + F.s, o[1]); ctx.moveTo(o[0], o[1] - F.s); ctx.lineTo(o[0], o[1] + F.s); ctx.stroke();
      if (focus === "concentration") { pts.forEach((p) => { const q = F.px(p); ctx.strokeStyle = T.accent2; ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); }); ctx.globalAlpha = 1; ctx.strokeStyle = T.accent; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(m[0], m[1]); ctx.stroke(); }
      else pts.forEach((p, i) => { const q = F.px(p), far = B.ds[i] === B.r; ctx.strokeStyle = focus === "radius" && far ? T.accent2 : T.accent; ctx.globalAlpha = focus === "centroid" ? 0.3 : 0.6; ctx.lineWidth = far && focus === "radius" ? 2 : 1; ctx.beginPath(); ctx.moveTo(m[0], m[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); });
      ctx.globalAlpha = 1;
      if (focus !== "centroid") { ctx.strokeStyle = T.accent; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.arc(m[0], m[1], B.r * F.s, 0, 7); ctx.stroke(); if (focus === "variance") { ctx.strokeStyle = T.good; ctx.beginPath(); ctx.arc(m[0], m[1], B.mean * F.s, 0, 7); ctx.stroke(); } ctx.setLineDash([]); }
      pts.forEach((p) => { const q = F.px(p); ctx.beginPath(); ctx.arc(q[0], q[1], 4.5, 0, 7); ctx.fillStyle = T.ink; ctx.fill(); });
      ctx.strokeStyle = T.accent2; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(m[0] - 7, m[1] - 7); ctx.lineTo(m[0] + 7, m[1] + 7); ctx.moveTo(m[0] + 7, m[1] - 7); ctx.lineTo(m[0] - 7, m[1] + 7); ctx.stroke();
      font(ctx, 10); ctx.fillStyle = T.muted; ctx.fillText("μ_B", m[0] + 9, m[1] - 8); ctx.fillText("0", o[0] + 4, o[1] + 12);
    });
    const upd = () => {
      pn.paint(); const B = blockStats(pts), row = (k, s) => (k === focus ? `<b class="on">${s}</b>` : s);
      out.innerHTML = [row("centroid", `μ_B = (${f(B.mu[0])}, ${f(B.mu[1])})`), row("radius", `r_B = max d(x_i, μ_B) = ${f(B.r)}`), row("variance", `d̄ = ${f(B.mean)}, σ²_B = mean(d²) − d̄² = ${f(B.var, 4)}`), row("concentration", `c_B = mean cos(x_i, μ_B) = ${f(B.conc)}`)].join(" · ") + ` · n = ${pts.length}`;
    };
    cdrag(pn, pts, upd, () => F);
    const c = api.controls(), rnd = mulberry(3);
    c.button("Scatter", () => { pts.forEach((p, i) => { pts[i] = [rnd() * 1.8 - 0.9, rnd() * 1.8 - 0.9]; }); upd(); });
    c.button("Tighten", () => { const B = blockStats(pts); pts.forEach((p, i) => { pts[i] = [B.mu[0] + (p[0] - B.mu[0]) * 0.6, B.mu[1] + (p[1] - B.mu[1]) * 0.6]; }); upd(); });
    api.onTheme(upd); upd();
  });

  // Cauchy-Schwarz upper bound: q.x <= |q||x| <= |q|(|mu_B| + r_B).
  reg("topo-bound", (stage, api) => {
    const pts = [[0.35, 0.55], [0.6, 0.72], [0.52, 0.3], [0.78, 0.45], [0.45, 0.62], [0.66, 0.35]], q = [[-0.4, 0.5]];
    let thr = 0.6, F;
    const out = readout(stage), g = el("div", "topo-split", stage);
    const pn = panel(g, 250, (ctx, w, h) => {
      const T = api.theme(); F = cframe(w, h); const B = blockStats(pts), m = F.px(B.mu), o = F.px([0, 0]), qp = F.px(q[0]);
      ctx.strokeStyle = T.hair; ctx.beginPath(); ctx.moveTo(o[0] - F.s, o[1]); ctx.lineTo(o[0] + F.s, o[1]); ctx.moveTo(o[0], o[1] - F.s); ctx.lineTo(o[0], o[1] + F.s); ctx.stroke();
      ctx.strokeStyle = T.accent; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.arc(m[0], m[1], B.r * F.s, 0, 7); ctx.stroke(); ctx.setLineDash([]);
      pts.forEach((p) => { const a = F.px(p); ctx.beginPath(); ctx.arc(a[0], a[1], 4, 0, 7); ctx.fillStyle = T.ink; ctx.fill(); });
      ctx.fillStyle = T.accent; ctx.beginPath(); ctx.arc(m[0], m[1], 3, 0, 7); ctx.fill();
      ctx.strokeStyle = T.accent2; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(qp[0], qp[1]); ctx.stroke(); ctx.beginPath(); ctx.arc(qp[0], qp[1], 6, 0, 7); ctx.fillStyle = T.accent2; ctx.fill();
      font(ctx, 10); ctx.fillStyle = T.muted; ctx.fillText("q (drag)", qp[0] + 8, qp[1] - 6); ctx.fillText("block B", m[0] + B.r * F.s * 0.7, m[1] - B.r * F.s * 0.8);
    });
    const pbars = panel(g, 250, (ctx, w, h) => {
      const T = api.theme(), B = blockStats(pts), sc = pts.map((p) => p[0] * q[0][0] + p[1] * q[0][1]), ub = Math.hypot(...q[0]) * (B.nm + B.r);
      const vmax = Math.max(ub, thr, ...sc.map(Math.abs)) * 1.1 || 1, L = 10, R = w - 10, y0 = h / 2 + 20, Y = (v) => y0 - (v / vmax) * (h / 2 - 10), bw = (R - L - 60) / pts.length;
      ctx.strokeStyle = T.hair; ctx.beginPath(); ctx.moveTo(L, y0); ctx.lineTo(R, y0); ctx.stroke();
      sc.forEach((v, i) => { ctx.fillStyle = T.ink; ctx.globalAlpha = 0.7; ctx.fillRect(L + i * bw + 3, Math.min(Y(v), y0), bw - 6, Math.abs(Y(v) - y0)); }); ctx.globalAlpha = 1;
      font(ctx, 10);
      [[ub, T.accent, "bound"], [thr, T.accent2, "threshold"]].forEach(([v, col, l]) => { ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(R, Y(v)); ctx.stroke(); ctx.fillStyle = col; ctx.fillText(l, R - 56, Y(v) - 4); });
      ctx.fillStyle = T.muted; ctx.fillText("q·x_i for each point", L, 12);
    });
    const upd = () => {
      pn.paint(); pbars.paint(); const B = blockStats(pts), sc = Math.max(...pts.map((p) => p[0] * q[0][0] + p[1] * q[0][1])), ub = Math.hypot(...q[0]) * (B.nm + B.r), prune = ub < thr;
      out.innerHTML = `max q·x_i = ${f(sc)} ≤ ‖q‖(‖μ_B‖ + r_B) = ${f(Math.hypot(...q[0]))}·(${f(B.nm)} + ${f(B.r)}) = <b>${f(ub)}</b> · ` + (prune ? `<span class="on">bound &lt; threshold ${f(thr, 2)} → block pruned without scanning its points</span>` : `<span class="bad">bound ≥ threshold ${f(thr, 2)} → block must be scanned</span>`);
    };
    cdrag(pn, q, upd, () => F);
    api.controls().slider("threshold", 0, 2, 0.01, thr, (v) => { thr = v; upd(); });
    api.onTheme(upd);
  });

  // Sparse event trigger: wake iff |mu(t) - mu(t_last)| >= eps.
  reg("topo-wake", (stage, api) => {
    const N = 260, rnd = mulberry(8), mu = [[0, 0]];
    for (let t = 1; t < N; t++) { const p = mu[t - 1], burst = t > 120 && t < 160 ? 4 : 1; mu.push([p[0] * 0.985 + (rnd() - 0.5) * 0.05 * burst, p[1] * 0.985 + (rnd() - 0.5) * 0.05 * burst]); }
    let eps = 0.12, shown = api.still ? N : 0, run = [];
    const sim = () => { run = []; let last = 0; for (let t = 0; t < N; t++) { const d = dist(mu[t], mu[last]), w = t > 0 && d >= eps; run.push({ d, w, last }); if (w) last = t; } };
    sim();
    const out = readout(stage), g = el("div", "topo-split", stage);
    const pp = panel(g, 220, (ctx, w, h) => {
      const T = api.theme(), ext = Math.max(...mu.flat().map(Math.abs)) || 1, s = (Math.min(w, h) / 2 - 16) / ext, cx = w / 2, cy = h / 2 + 6, P = (p) => [cx + p[0] * s, cy - p[1] * s], k = Math.max(0, shown - 1);
      ctx.strokeStyle = T.hair; ctx.beginPath(); for (let t = 0; t <= k; t++) { const [x, y] = P(mu[t]); t ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke();
      for (let t = 0; t <= k; t++) if (run[t].w) { const [x, y] = P(mu[t]); ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 7); ctx.fillStyle = T.accent2; ctx.fill(); }
      const L = P(mu[run[k].last]), C = P(mu[k]); ctx.strokeStyle = T.accent; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.arc(L[0], L[1], eps * s, 0, 7); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(C[0], C[1], 4, 0, 7); ctx.fillStyle = T.ink; ctx.fill();
      font(ctx, 10); ctx.fillStyle = T.muted; ctx.fillText("μ(t); dashed circle = ε around μ(t_last)", 6, 12);
    });
    const pt = panel(g, 220, (ctx, w, h) => {
      const T = api.theme(), k = Math.max(0, shown - 1), vmax = Math.max(eps * 1.6, ...run.map((r) => r.d)), X = (t) => 6 + (t / (N - 1)) * (w - 12), Y = (v) => h - 18 - (v / vmax) * (h - 34);
      ctx.strokeStyle = T.ink; ctx.lineWidth = 1.2; ctx.beginPath(); for (let t = 0; t <= k; t++) t ? ctx.lineTo(X(t), Y(run[t].d)) : ctx.moveTo(X(t), Y(run[t].d)); ctx.stroke();
      ctx.strokeStyle = T.accent; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(X(0), Y(eps)); ctx.lineTo(X(N - 1), Y(eps)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = T.accent2; for (let t = 0; t <= k; t++) if (run[t].w) ctx.fillRect(X(t) - 1, h - 14, 2, 8);
      font(ctx, 10); ctx.fillStyle = T.muted; ctx.fillText("Δ(t) vs ε; ticks = wake", 6, 12);
    });
    const upd = () => { pp.paint(); pt.paint(); const k = Math.max(0, shown - 1), wakes = run.slice(0, k + 1).filter((r) => r.w).length; out.innerHTML = `t = ${k}: Δ(t) = ${f(run[k].d)} ${run[k].d >= eps && k ? "≥" : "&lt;"} ε = ${f(eps, 2)} → ${run[k].w ? '<span class="bad">wake</span>' : "sleep"} · ${wakes} wakes in ${k + 1} ticks (${f((100 * wakes) / (k + 1), 1)}%)`; };
    api.controls().slider("threshold ε", 0.02, 0.4, 0.01, eps, (v) => { eps = v; sim(); upd(); });
    if (!api.still) api.loop((s) => { const n = Math.min(N, Math.floor((s * 40) % (N + 60))); if (n !== shown) { shown = n; upd(); } });
    api.onTheme(upd); upd();
  });

  // GeometricGovernor::adapt, constants from crates/aether-core/src/governor.rs.
  reg("topo-governor", (stage, api) => {
    const TARGET = 1000, A = 0.25, B = 0.05, EMIN = 0.001, EMAX = 10, E0 = 0.1, STEPS = 40;
    let delta = 50, dt = 1;
    const sim = () => { let e = E0, le = 0; const o = [{ e, rate: delta / e, err: 0 }]; for (let i = 0; i < STEPS; i++) { const rate = delta / e, err = Math.max(-1, Math.min(1, 1 - rate / TARGET)), de = err - le; e = Math.min(EMAX, Math.max(EMIN, e * Math.exp(-(A * err + B * de)))); le = err; o.push({ e, rate, err }); } return o; };
    const out = readout(stage);
    let run = sim();
    const pn = panel(stage, 240, (ctx, w, h) => {
      const T = api.theme(), L = 44, R = w - 10, top = 10, bot = h - 22, X = (i) => L + (i / STEPS) * (R - L), Y = (e) => bot - ((Math.log10(e) + 3) / 4) * (bot - top);
      font(ctx, 10); ctx.strokeStyle = T.hair; ctx.fillStyle = T.muted;
      [0.001, 0.01, 0.1, 1, 10].forEach((v) => { ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(R, Y(v)); ctx.stroke(); ctx.fillText(v, 4, Y(v) + 3); });
      ctx.fillText("step →", R - 40, h - 6); ctx.fillText("ε (log)", L + 4, top + 10);
      const eq = delta / TARGET; if (eq >= EMIN && eq <= EMAX) { ctx.strokeStyle = T.good; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(L, Y(eq)); ctx.lineTo(R, Y(eq)); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = T.good; ctx.fillText("Δ/R_target", R - 64, Y(eq) - 4); }
      ctx.strokeStyle = T.accent; ctx.lineWidth = 1.8; ctx.beginPath(); run.forEach((r, i) => (i ? ctx.lineTo(X(i), Y(r.e)) : ctx.moveTo(X(i), Y(r.e)))); ctx.stroke();
      run.forEach((r, i) => { ctx.beginPath(); ctx.arc(X(i), Y(r.e), 2.2, 0, 7); ctx.fillStyle = r.e <= EMIN || r.e >= EMAX ? T.accent2 : T.accent; ctx.fill(); });
    });
    const upd = () => {
      run = sim(); pn.paint(); const last = run[run.length - 1], clamps = run.filter((r) => r.e <= EMIN || r.e >= EMAX).length;
      out.innerHTML = `R_target = ${TARGET}, α = ${A}, β = ${B}, ε₀ = ${E0}, clamp [${EMIN}, ${EMAX}] (governor.rs constants) · Δ = ${f(delta, 3)}, dt = ${f(dt, 2)} · after ${STEPS} steps ε = <b>${f(last.e, 4)}</b>, R_actual = ${f(last.rate, 1)} · <span class="${clamps ? "bad" : "on"}">${clamps} of ${STEPS + 1} steps at a clamp</span> (orange). Green dashed line: ε where R_actual = R_target.`;
    };
    const c = api.controls();
    c.slider("log₁₀ Δ", -3, 2, 0.05, Math.log10(delta), (v) => { delta = 10 ** v; upd(); });
    c.slider("dt", 0.25, 4, 0.25, dt, (v) => { dt = v; upd(); });
    api.onTheme(upd);
  });

  // Binary shape gate: crates/aether-core/src/topology.rs, byte for byte.
  const SG = { CLUSTER: 15, DMIN: 0.1, DMAX: 0.6, MAXB1: 10, TOL: 5 };
  function betti0(d) { if (d.length < 2) return d.length ? 1 : 0; let c = 0, inC = false; const gaps = []; for (let i = 0; i + 1 < d.length; i++) { if (Math.abs(d[i] - d[i + 1]) > SG.CLUSTER) { gaps.push(i); if (!inC) { c++; inC = true; } } else inC = false; } return { c, gaps }; }
  function betti1(d) { const loops = []; for (let i = 0; i + 3 < d.length; i++) { const a = d[i]; if (Math.abs(a - d[i + 3]) <= SG.TOL && (Math.abs(a - d[i + 1]) > SG.TOL || Math.abs(a - d[i + 2]) > SG.TOL)) loops.push(i); } return loops; }
  function shape(d) { const b0 = betti0(d), c = typeof b0 === "number" ? b0 : b0.c, loops = betti1(d); return { b0: c, gaps: b0.gaps || [], b1: loops.length, loops, density: d.length ? c / d.length : 0 }; }
  const PRESETS = {
    "Prologue (test bytes)": [0x55, 0x48, 0x89, 0xe5, 0x48, 0x83, 0xec, 0x20, 0x89, 0x7d, 0xec, 0x89, 0x75, 0xe8, 0x48, 0x89, 0x55, 0xe0, 0x48, 0x89, 0x4d, 0xd8, 0x44, 0x89, 0x45, 0xd4, 0x44, 0x89, 0x4d, 0xd0, 0x8b, 0x45],
    "NOP sled": Array(64).fill(0x90),
    "Random bytes": (() => { const r = mulberry(42); return Array.from({ length: 64 }, () => Math.floor(r() * 256)); })(),
    "Repeating a,b,c,a": Array.from({ length: 48 }, (_, i) => [0x10, 0x80, 0xf0][i % 3]),
  };
  reg("topo-shape", (stage, api) => {
    const withRef = stage.parentElement.dataset.mode === "reference", ref = shape(PRESETS["Prologue (test bytes)"]);
    let data = PRESETS["Prologue (test bytes)"].slice(), thr = 3;
    const out = readout(stage);
    const pn = panel(stage, 200, (ctx, w, h) => {
      const T = api.theme(), S = shape(data), n = Math.max(data.length, 1), bw = (w - 12) / n, Y = (v) => h - 40 - (v / 255) * (h - 60);
      S.loops.forEach((i) => { ctx.fillStyle = T.accent2; ctx.globalAlpha = 0.12; ctx.fillRect(6 + i * bw, 6, bw * 4, h - 46); }); ctx.globalAlpha = 1;
      data.forEach((v, i) => { ctx.fillStyle = T.ink; ctx.globalAlpha = 0.75; ctx.fillRect(6 + i * bw + bw * 0.15, Y(v), bw * 0.7, h - 40 - Y(v)); }); ctx.globalAlpha = 1;
      S.gaps.forEach((i) => { ctx.strokeStyle = T.accent; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(6 + (i + 1) * bw, h - 38); ctx.lineTo(6 + (i + 1) * bw, h - 30); ctx.stroke(); });
      font(ctx, 10); ctx.fillStyle = T.muted; ctx.fillText("blue: |Δ| > 15 gap · orange: loop window", 6, h - 14);
      const gx = (v) => 6 + (v / 1) * (w - 12); ctx.strokeStyle = T.hair; ctx.beginPath(); ctx.moveTo(gx(0), h - 4); ctx.lineTo(gx(1), h - 4); ctx.stroke();
      ctx.fillStyle = T.good; ctx.globalAlpha = 0.35; ctx.fillRect(gx(SG.DMIN), h - 7, gx(SG.DMAX) - gx(SG.DMIN), 6); ctx.globalAlpha = 1;
      ctx.fillStyle = T.accent2; ctx.fillRect(gx(Math.min(1, S.density)) - 1.5, h - 9, 3, 10);
    });
    const upd = () => {
      pn.paint(); const S = shape(data);
      let verdict;
      const dist_ = Math.sqrt((S.b0 - ref.b0) ** 2 + (S.b1 - ref.b1) ** 2 + (S.density - ref.density) ** 2);
      if (withRef && dist_ > thr) verdict = `<span class="bad">ShapeMismatch: distance ${f(dist_)} &gt; ${f(thr, 1)}</span>`;
      else if (S.density < SG.DMIN || S.density > SG.DMAX) verdict = `<span class="bad">InvalidDensity: ${f(S.density)} outside [${SG.DMIN}, ${SG.DMAX}]</span>`;
      else if (S.b1 > SG.MAXB1) verdict = `<span class="bad">ExcessiveLoops: ${S.b1} &gt; ${SG.MAXB1}</span>`;
      else verdict = `<span class="on">Pass</span>`;
      out.innerHTML = `|B| = ${data.length} · β₀ ≈ ${S.b0} · β₁ ≈ ${S.b1} · density = β₀/|B| = <b>${f(S.density)}</b> (green band [0.1, 0.6]) · ` + (withRef ? `reference = prologue (β₀ ${ref.b0}, β₁ ${ref.b1}), distance ${f(dist_)} · ` : "") + verdict;
    };
    const c = api.controls();
    Object.keys(PRESETS).forEach((k) => c.button(k, () => { data = PRESETS[k].slice(); ta.value = hex(); upd(); }));
    if (withRef) c.slider("mismatch threshold", 0, 20, 0.5, thr, (v) => { thr = v; upd(); });
    const hex = () => data.map((b) => b.toString(16).padStart(2, "0")).join(" ");
    const lab = el("label", "topo-wide", stage.parentElement.querySelector(".ts-viz-controls")); lab.textContent = "bytes (hex, editable)";
    const ta = el("textarea", "", lab); ta.rows = 2; ta.value = hex(); ta.spellcheck = false;
    ta.addEventListener("input", () => { data = (ta.value.match(/[0-9a-f]{1,2}/gi) || []).slice(0, 256).map((s) => parseInt(s, 16)); upd(); });
    api.onTheme(upd); upd();
  });

  // ======================= shape-gates.md =======================

  // The DSL example: embed(data, tau) with D=3, then topology.ph(..., mode="vr", max_points) and betti(radius).
  reg("topo-dsl", (stage, api) => {
    let text = "1.0, 1.0, 1.0, 1.0, 1.0", tau = 1, radius = 0, maxPts = 16;
    const out = readout(stage);
    let P = [], res = null, err = "";
    const run = () => {
      const data = (text.match(/-?\d+(\.\d+)?(e-?\d+)?/gi) || []).map(Number), t = tau === 0 ? 1 : tau; P = [];
      for (let n = 0; n < data.length; n++) if (n + 1 >= 3 * t) P.push([data[n], data[n - t], data[n - 2 * t]]);
      err = P.length > maxPts ? `TooManyPoints { actual: ${P.length}, max: ${maxPts} }` : "";
      res = err || !P.length ? null : vrPH(P);
    };
    const pn = panel(stage, 200, (ctx, w, h) => {
      const T = api.theme(); if (!P.length) return;
      const all = P.flat(), lo = Math.min(...all), hi = Math.max(...all), sp = hi - lo || 1;
      const pr = (p) => { const a = (p[0] - lo) / sp - 0.5, b = (p[1] - lo) / sp - 0.5, c = (p[2] - lo) / sp - 0.5; return [w / 2 + (a - c * 0.5) * h * 0.7, h / 2 - (b - c * 0.35) * h * 0.7]; };
      if (res) drawComplex(ctx, { px: pr, s: 0 }, P, res.S.filter((s) => s.d > 0), radius, T);
      else P.forEach((p) => { const [x, y] = pr(p); ctx.beginPath(); ctx.arc(x, y, 3.6, 0, 7); ctx.fillStyle = T.ink; ctx.fill(); });
      font(ctx, 10); ctx.fillStyle = T.muted; ctx.fillText(`${P.length} embedded points (oblique 3D view)`, 6, 12);
    });
    const upd = () => {
      run(); pn.paint();
      if (err) { out.innerHTML = `<span class="bad">topology.ph → ${err}</span>`; return; }
      if (!res) { out.innerHTML = `embed(data, tau=${tau}) emitted no points: needs at least 3·τ = ${3 * (tau || 1)} samples.`; return; }
      const b = bettiAt(res.pairs, radius);
      out.innerHTML = `manifold M: ${P.length} points in ℝ³ · diagram: ${res.pairs.length} pairs · topology.betti(diagram, radius=${f(radius, 2)}) = <b>[${b.join(", ")}]</b>` + (P.every((p) => dist(p, P[0]) === 0) ? " (all points coincide, so every edge enters at 0 and one component survives)" : "");
    };
    const c = api.controls();
    const lab = el("label", "topo-wide", stage.parentElement.querySelector(".ts-viz-controls")); lab.textContent = "let data = [ … ]";
    const inp = el("input", "", lab); inp.type = "text"; inp.value = text; inp.addEventListener("input", () => { text = inp.value; upd(); });
    c.slider("tau", 0, 4, 1, tau, (v) => { tau = v; upd(); });
    c.slider("radius", 0, 2, 0.01, radius, (v) => { radius = v; upd(); });
    c.slider("max_points", 4, 32, 1, maxPts, (v) => { maxPts = v; upd(); });
    c.button("Load a sine", () => { inp.value = text = Array.from({ length: 14 }, (_, i) => Math.sin(i * 0.9).toFixed(2)).join(", "); upd(); });
    api.onTheme(upd);
  });
})();
