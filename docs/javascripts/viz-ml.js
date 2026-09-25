// viz-ml: live ML visualisations. Every curve is computed here by a JS port of
// the Rust it names (crates/aether-core/src/ml, aether-lang/src/interpreter.rs).
(() => {
  if (!window.TSViz) return;
  const st = document.createElement("style");
  st.textContent = `
.ts-viz[data-viz^="ml-"] .ml-read { font-family: var(--md-code-font-family); font-size: .64rem;
  color: var(--md-default-fg-color--light); margin-top: .45rem; min-height: 1.2em; overflow-wrap: anywhere; }
.ts-viz[data-viz^="ml-"] .ml-read b { color: var(--md-default-fg-color); font-weight: 600; }
.ts-viz[data-viz^="ml-"] .ts-viz-controls input[type=range] { max-width: 42vw; }`;
  document.head.append(st);

  // ---------- shared helpers ----------
  const M64 = (1n << 64n) - 1n;
  // The crate's LCG: rng = rng * 6364136223846793005 + 1 (wrapping u64).
  const lcg = (s) => (s * 6364136223846793005n + 1n) & M64;
  const U64MAX = 18446744073709551615;
  // Seeded data generator (data only, never algorithm state).
  const prng = (seed) => () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const gauss = (r) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
  const readout = (stage) => { const d = document.createElement("div"); d.className = "ml-read"; d.setAttribute("aria-live", "polite"); stage.after(d); return d; };
  const font = (ctx, px = 11) => { ctx.font = `${px}px "JetBrains Mono", ui-monospace, monospace`; };
  const clear = (ctx, w, h, T) => { ctx.fillStyle = T.ground; ctx.fillRect(0, 0, w, h); };
  function axes(ctx, x, y, w, h, T, label) {
    ctx.strokeStyle = T.hair; ctx.lineWidth = 1; ctx.strokeRect(x + .5, y + .5, w, h);
    if (label) { font(ctx, 10); ctx.fillStyle = T.muted; ctx.fillText(label, x + 6, y + 13); }
  }
  function polyline(ctx, pts, color, width = 1.6) {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
  }
  const mix = (a, b, t) => { // hex mix
    const p = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
    const A = p(a), B = p(b); return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(",")})`;
  };

  // ---------- convergence.rs: ResidualAnalyzer + ConvergenceDetector ----------
  function residualBetti(r) { // ResidualAnalyzer::compute_betti
    if (!r.length) return [0, 0];
    let b0 = 1, prev = r[0] >= 0;
    for (let i = 1; i < r.length; i++) { const s = r[i] >= 0; if (s !== prev) b0++; prev = s; }
    b0 = Math.ceil((b0 + 1) / 2);
    let b1 = 0, inc = true;
    for (let i = 1; i < r.length; i++) { const c = r[i] - r[i - 1] >= 0; if (c !== inc) b1++; inc = c; }
    return [b0, Math.floor(b1 / 4)];
  }
  function residualDrift(r) { // ResidualAnalyzer::compute_drift
    if (r.length < 2) return 0;
    const m = r.reduce((s, v) => s + Math.abs(v), 0) / r.length;
    return Math.sqrt(r.reduce((s, v) => s + (Math.abs(v) - m) ** 2, 0) / r.length);
  }
  function detector(eps, win) { // ConvergenceDetector::new(epsilon, stability_window)
    win = Math.max(3, win);
    const B = [], D = [], E = [];
    return {
      record(b, d, e) { if (B.length >= 32) { B.shift(); D.shift(); E.shift(); } B.push(b); D.push(d); E.push(e); },
      check() {
        if (B.length < win) return null;
        if (E[E.length - 1] < eps) return "error";
        const w = B.slice(-win), bs = w.every((b) => b[0] === w[0][0] && b[1] === w[0][1]);
        const dw = D.slice(-win); let prev = Infinity, ds = true;
        for (const d of dw) { if (d > prev * 1.5) { ds = false; break; } prev = d; }
        ds = ds && dw[dw.length - 1] < 0.01;
        return bs && ds ? "topology" : null;
      },
    };
  }

  // ===== 1. seal loop over a training run =====
  TSViz.register("ml-seal-detector", (stage, api) => {
    const out = readout(stage);
    const S = { noise: 0.004, win: 5, lr: 0.5 };
    let run = null, shown = 0;
    const N = 48, MAX = 600;
    function train() {
      const r = prng(7), xs = [], ys = [];
      for (let i = 0; i < N; i++) { const x = -1 + (2 * i) / (N - 1); xs.push(x); ys.push(0.6 * x ** 3 - 0.4 * x + 0.1 + S.noise * gauss(r)); }
      let w = [0, 0, 0, 0];
      const det = detector(1e-3, S.win), trace = [];
      let halt = null;
      for (let ep = 0; ep < MAX; ep++) { // seal until det.is_converged()
        const g = [0, 0, 0, 0], res = [];
        for (let i = 0; i < N; i++) {
          const f = [1, xs[i], xs[i] ** 2, xs[i] ** 3], p = f.reduce((s, v, j) => s + v * w[j], 0), e = p - ys[i];
          res.push(ys[i] - p); f.forEach((v, j) => (g[j] += (2 * e * v) / N));
        }
        const rmse = Math.sqrt(res.reduce((s, v) => s + v * v, 0) / N);
        const b = residualBetti(res), d = residualDrift(res);
        det.record(b, d, rmse);
        trace.push({ w: w.slice(), rmse, b, d });
        const why = det.check();
        if (why) { halt = { ep, why }; break; }
        w = w.map((v, j) => v - S.lr * g[j]);
      }
      run = { xs, ys, trace, halt }; shown = api.still ? trace.length : 1;
    }
    const c = api.canvas(300, (ctx, w, h) => {
      const T = api.theme(); clear(ctx, w, h, T); if (!run) return;
      const narrow = w < 520, pw = narrow ? w : w * 0.38, ph = narrow ? 120 : h;
      const t = run.trace[Math.min(shown, run.trace.length) - 1];
      // fit panel
      axes(ctx, 0, 0, pw - 8, ph - 1, T, "fit y = w·[1,x,x²,x³]");
      const X = (x) => 4 + ((x + 1) / 2) * (pw - 16), Y = (y) => ph / 2 - y * (ph * 0.55);
      ctx.fillStyle = T.muted; run.xs.forEach((x, i) => { ctx.beginPath(); ctx.arc(X(x), Y(run.ys[i]), 2, 0, 7); ctx.fill(); });
      const fit = []; for (let k = 0; k <= 60; k++) { const x = -1 + k / 30; fit.push([X(x), Y(t.w[0] + t.w[1] * x + t.w[2] * x * x + t.w[3] * x ** 3)]); }
      polyline(ctx, fit, T.accent, 2);
      // history panels
      const hx = narrow ? 0 : pw + 4, hw = w - hx - 1, top = narrow ? ph + 8 : 0, avail = h - top, h1 = avail * 0.55 - 6, h2 = avail - h1 - 8;
      axes(ctx, hx, top, hw, h1, T, "log10 RMSE and drift");
      const n = run.trace.length, EX = (e) => hx + 6 + (e / Math.max(n - 1, 1)) * (hw - 12);
      const LY = (v) => top + 18 + ((0.5 - Math.log10(Math.max(v, 1e-5))) / 5.5) * (h1 - 24);
      const vis = run.trace.slice(0, shown);
      polyline(ctx, vis.map((q, e) => [EX(e), LY(q.rmse)]), T.accent);
      polyline(ctx, vis.map((q, e) => [EX(e), LY(q.d)]), T.accent2, 1.2);
      ctx.setLineDash([3, 3]); polyline(ctx, [[hx, LY(0.01)], [hx + hw, LY(0.01)]], T.hair, 1); ctx.setLineDash([]);
      font(ctx, 9); ctx.fillStyle = T.muted; ctx.fillText("drift 0.01", hx + hw - 60, LY(0.01) - 3);
      const t2 = top + h1 + 8; axes(ctx, hx, t2, hw, h2, T, "β₀ (blue)  β₁ (orange)");
      const bmax = Math.max(4, ...run.trace.map((q) => q.b[0])), BY = (v) => t2 + h2 - 6 - (v / bmax) * (h2 - 22);
      polyline(ctx, vis.map((q, e) => [EX(e), BY(q.b[0])]), T.accent, 1.2);
      polyline(ctx, vis.map((q, e) => [EX(e), BY(q.b[1])]), T.accent2, 1.2);
      if (run.halt && shown >= n) {
        const x = EX(run.halt.ep); ctx.strokeStyle = T.good; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, t2 + h2); ctx.stroke();
      }
      const q = t; out.innerHTML = `epoch <b>${Math.min(shown, n) - 1}</b> · RMSE <b>${q.rmse.toExponential(2)}</b> · β=(<b>${q.b}</b>) · drift <b>${q.d.toFixed(4)}</b> · ` +
        (shown < n ? "running" : run.halt ? `<b>sealed at epoch ${run.halt.ep}</b> (${run.halt.why === "error" ? "error < ε" : `β stable ${S.win} epochs and drift < 0.01`})` : `no halt in ${MAX} epochs (budget exhausted)`);
    });
    api.onTheme(() => c.redraw());
    const restart = () => { train(); c.redraw(); };
    const ui = api.controls();
    let ready = false;
    ui.slider("noise σ", 0, 0.03, 0.001, S.noise, (v) => { S.noise = v; ready && restart(); });
    ui.slider("window", 3, 10, 1, S.win, (v) => { S.win = v; ready && restart(); });
    ui.slider("lr", 0.05, 0.9, 0.05, S.lr, (v) => { S.lr = v; ready && restart(); });
    ui.button("Replay", restart);
    ready = true; restart();
    api.loop(() => { if (run && shown < run.trace.length) { shown = Math.min(run.trace.length, shown + 3); c.redraw(); } });
  });

  // ===== 2. interpreter escalating regressor residual heuristic =====
  TSViz.register("ml-residual-shape", (stage, api) => {
    const out = readout(stage);
    const S = { freq: 3, n: 40, ep: 0 };
    let run;
    const models = ["Linear", "Polynomial{2}", "Polynomial{3}", "Polynomial{4}", "Rbf{0.4}", "Rbf{0.5}", "Rbf{0.6}", "Rbf{1.0}"];
    const modelAt = (e) => (e === 0 ? ["lin"] : e <= 3 ? ["poly", e + 1] : ["rbf"]);
    function fitLinear(xs, ys) {
      const n = xs.length; let sx = 0, sy = 0, sxy = 0, sxx = 0;
      xs.forEach((x, i) => { sx += x; sy += ys[i]; sxy += x * ys[i]; sxx += x * x; });
      const c = new Array(8).fill(0), den = n * sxx - sx * sx;
      if (Math.abs(den) > 1e-10) { c[1] = (n * sxy - sx * sy) / den; c[0] = (sy - c[1] * sx) / n; }
      return c;
    }
    const fit = (xs, ys, m) => { if (m[0] === "lin") return fitLinear(xs, ys); const c = fitLinear(xs, ys); c[m[0] === "poly" ? m[1] : 3] = 0.01; return c; };
    const predict = (x, c, m) => { if (m[0] === "lin") return c[0] + c[1] * x; const d = m[0] === "poly" ? m[1] : 3;
      let y = c[0], xp = x; for (let k = 1; k <= Math.min(d, 7); k++) { y += c[k] * xp; xp *= x; } return y; };
    function betti(res) { // compute_residual_betti, including its prev_delta = prev_residual comparison
      let sc = 0, osc = 0, pr = 0, ps = true;
      res.forEach((r, i) => { const s = r >= 0; if (i > 0 && s !== ps) sc++;
        if (i > 1 && ((r - pr) > 0) !== (pr > 0)) osc++; pr = r; ps = s; });
      return [Math.floor(sc / 2) + 1, Math.floor(osc / 4)];
    }
    function go() {
      const r = prng(11), xs = [], ys = [];
      for (let i = 0; i < S.n; i++) { const x = i / (S.n - 1); xs.push(x); ys.push(Math.sin(2 * Math.PI * S.freq * x / 2) + 0.05 * gauss(r)); }
      const hist = [], epochs = []; let stop = null;
      for (let e = 0; e < 10; e++) { // max_epochs = 10 without `escalate`
        const m = modelAt(e), c = fit(xs, ys, m), res = ys.map((y, i) => y - predict(xs[i], c, m));
        const err = Math.sqrt(res.reduce((s, v) => s + v * v, 0) / res.length), b = betti(res);
        hist.push(b); if (hist.length > 10) hist.shift();
        epochs.push({ m: models[Math.min(e, 7)], c, res, err, b, mm: m });
        const conv = err < 1e-6 || (hist.length >= 3 && hist.slice(-3).every((q) => q[0] === b[0] && q[1] === b[1]));
        if (conv) { stop = e; break; }
      }
      run = { xs, ys, epochs, stop };
      epSl.max = epochs.length - 1; if (S.ep > epochs.length - 1) { S.ep = epochs.length - 1; epSl.value = S.ep; }
      c.redraw();
    }
    const c = api.canvas(230, (ctx, w, h) => {
      const T = api.theme(); clear(ctx, w, h, T); if (!run) return;
      const E = run.epochs[S.ep], n = run.xs.length, h1 = h * 0.5;
      axes(ctx, 0, 0, w - 1, h1 - 4, T, `epoch ${S.ep}: ${E.m}`);
      const X = (i) => 8 + (i / (n - 1)) * (w - 16), Y = (y) => h1 / 2 - y * h1 * 0.36;
      ctx.fillStyle = T.muted; run.ys.forEach((y, i) => { ctx.beginPath(); ctx.arc(X(i), Y(y), 2, 0, 7); ctx.fill(); });
      polyline(ctx, run.xs.map((x, i) => [X(i), Y(predict(x, E.c, E.mm))]), T.accent, 2);
      const t = h1 + 4, rh = h - t - 1, z = t + rh / 2, rs = Math.max(...E.res.map(Math.abs), 1e-9);
      axes(ctx, 0, t, w - 1, rh, T, "residuals rᵢ = yᵢ − ŷᵢ, sign changes marked");
      const bw = Math.max(1, (w - 16) / n - 2);
      E.res.forEach((r, i) => { ctx.fillStyle = r >= 0 ? T.accent : T.accent2; const y = (r / rs) * (rh / 2 - 16); ctx.fillRect(X(i) - bw / 2, y >= 0 ? z - y : z, bw, Math.abs(y)); });
      ctx.strokeStyle = T.good; ctx.lineWidth = 1.5;
      E.res.forEach((r, i) => { if (i && (r >= 0) !== (E.res[i - 1] >= 0)) { const x = (X(i) + X(i - 1)) / 2; ctx.beginPath(); ctx.moveTo(x, z - 8); ctx.lineTo(x, z + 8); ctx.stroke(); } });
      out.innerHTML = `RMSE <b>${E.err.toFixed(4)}</b> · β=(sign_changes/2+1, oscillations/4) = (<b>${E.b}</b>) · history ${run.epochs.slice(0, S.ep + 1).slice(-3).map((q) => `(${q.b})`).join(" ")}` +
        (run.stop === S.ep ? ` · <b>converged: last 3 equal</b>` : run.stop == null && S.ep === run.epochs.length - 1 ? " · 10-epoch budget exhausted" : "");
    });
    api.onTheme(() => c.redraw());
    const ui = api.controls();
    let epSl = { max: 0 };
    ui.slider("frequency", 1, 8, 1, S.freq, (v) => { S.freq = v; run && go(); });
    epSl = ui.slider("epoch", 0, 9, 1, 0, (v) => { S.ep = v; run && c.redraw(); });
    go();
  });

  // ===== 3. KMeans::fit =====
  TSViz.register("ml-kmeans", (stage, api) => {
    const out = readout(stage);
    const S = { k: 3 };
    let data = [], cents = [], labels = [], it = 0, done = null, prevIn = Infinity, inertia = 0, timer = 0;
    const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
    const r = prng(3);
    for (let b = 0; b < 4; b++) { const cx = 0.2 + 0.6 * r(), cy = 0.2 + 0.6 * r(); for (let i = 0; i < 22; i++) data.push([cx + 0.07 * gauss(r), cy + 0.07 * gauss(r)]); }
    function init() { // init_centroids_plusplus, seed 42
      const n = data.length; let rng = lcg(42n); cents = [data[Number((rng >> 33n) % BigInt(n))].slice()];
      for (let c = 1; c < S.k; c++) {
        let md = 0, mi = 0;
        data.forEach((p, i) => { const m = Math.min(...cents.map((q) => d2(p, q)));
          rng = lcg(rng); const wd = m * (Number((rng >> 33n) % 1000n) / 1000 + 0.5); if (wd > md) { md = wd; mi = i; } });
        cents.push(data[mi].slice());
      }
      labels = data.map(() => 0); it = 0; done = null; prevIn = Infinity; inertia = 0;
    }
    function step() {
      if (done || it >= 100) return;
      let changed = 0;
      data.forEach((p, i) => { let m = Infinity, j = 0; cents.forEach((q, k) => { const d = d2(p, q); if (d < m) { m = d; j = k; } }); if (j !== labels[i]) { labels[i] = j; changed++; } });
      const sum = cents.map(() => [0, 0, 0]);
      data.forEach((p, i) => { const s = sum[labels[i]]; s[0] += p[0]; s[1] += p[1]; s[2]++; });
      cents = cents.map((q, k) => (sum[k][2] ? [sum[k][0] / sum[k][2], sum[k][1] / sum[k][2]] : [0, 0]));
      inertia = data.reduce((s, p, i) => s + d2(p, cents[labels[i]]), 0); it++;
      if (changed === 0 || Math.abs(prevIn - inertia) < 1e-4) done = changed === 0 ? "no label changed" : "|Δinertia| < 1e-4";
      prevIn = inertia;
      out.innerHTML = `iteration <b>${it}</b> · changed <b>${changed}</b> · inertia <b>${inertia.toFixed(4)}</b>${done ? ` · <b>stopped: ${done}</b>` : ""}`;
    }
    const pal = (T, k) => [T.accent, T.accent2, T.good, T.muted, T.ink, mix("#2456dc", "#c2410c", 0.5)][k % 6];
    const c = api.canvas(260, (ctx, w, h) => {
      const T = api.theme(); clear(ctx, w, h, T); axes(ctx, 0, 0, w - 1, h - 1, T);
      const s = Math.min(w, h) - 12, ox = (w - s) / 2, P = (p) => [ox + p[0] * s, 6 + p[1] * s];
      data.forEach((p, i) => { const [x, y] = P(p); ctx.fillStyle = it ? pal(T, labels[i]) : T.muted; ctx.globalAlpha = 0.75; ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill(); });
      ctx.globalAlpha = 1;
      cents.forEach((q, k) => { const [x, y] = P(q); ctx.strokeStyle = T.ground; ctx.lineWidth = 3; ctx.fillStyle = pal(T, k);
        ctx.beginPath(); ctx.rect(x - 6, y - 6, 12, 12); ctx.stroke(); ctx.fill(); });
    });
    api.onTheme(() => c.redraw());
    const reset = () => { clearInterval(timer); init(); out.textContent = `k-means++ seeds placed (seed 42) · k = ${S.k}`; c.redraw(); };
    const ui = api.controls();
    let ready = false;
    ui.slider("k", 2, 6, 1, S.k, (v) => { S.k = v; ready && reset(); });
    ui.button("Step", () => { step(); c.redraw(); });
    ui.button("Run", () => { clearInterval(timer); timer = setInterval(() => { step(); c.redraw(); if (done || it >= 100) clearInterval(timer); }, api.still ? 0 : 450); });
    ui.button("Reset", reset);
    ready = true; reset();
  });

  // ===== 4. LogisticRegression::fit / Perceptron::fit =====
  TSViz.register("ml-classify", (stage, api) => {
    const out = readout(stage);
    const S = { model: "logistic", sep: 0.9 };
    let X = [], Y = [], w = [0, 0], b = 0, hist = [], stopAt = null, epoch = 0, playing = true;
    function data() {
      const r = prng(5); X = []; Y = [];
      for (let i = 0; i < 60; i++) { const y = i % 2; X.push([(y ? S.sep : -S.sep) + 0.6 * gauss(r), (y ? 0.4 : -0.4) + 0.6 * gauss(r)]); Y.push(y); }
      w = [0, 0]; b = 0; hist = []; stopAt = null; epoch = 0;
    }
    function epochLogistic() { // one iteration of LogisticRegression::fit (lr 0.1, tol 1e-4)
      const n = X.length, g = [0, 0]; let gb = 0, loss = 0;
      X.forEach((x, i) => { const p = 1 / (1 + Math.exp(-(b + w[0] * x[0] + w[1] * x[1]))), e = p - Y[i];
        g[0] += e * x[0]; g[1] += e * x[1]; gb += e; const pc = Math.min(Math.max(p, 1e-7), 1 - 1e-7);
        loss -= Y[i] * Math.log(pc) + (1 - Y[i]) * Math.log(1 - pc); });
      w = w.map((v, j) => v - (0.1 * g[j]) / n); b -= (0.1 * gb) / n; loss /= n;
      if (hist.length && Math.abs(hist[hist.length - 1] - loss) < 1e-4) stopAt = epoch;
      hist.push(loss);
    }
    function epochPerceptron() { // one epoch of Perceptron::fit (lr 1.0, labels ±1)
      let errs = 0;
      X.forEach((x, i) => { const t = Y[i] ? 1 : -1, pred = b + w[0] * x[0] + w[1] * x[1] >= 0 ? 1 : -1, e = t - pred;
        if (e) { errs++; w[0] += e * x[0]; w[1] += e * x[1]; b += e; } });
      hist.push(errs); if (!errs) stopAt = epoch;
    }
    const budget = () => (S.model === "logistic" ? 100 : 50);
    function tick() { if (stopAt != null || epoch >= budget()) return false; S.model === "logistic" ? epochLogistic() : epochPerceptron(); epoch++; return true; }
    const c = api.canvas(250, (ctx, W, H) => {
      const T = api.theme(); clear(ctx, W, H, T);
      const narrow = W < 520, pw = narrow ? W : W * 0.55, ph = narrow ? 150 : H;
      axes(ctx, 0, 0, pw - 6, ph - 1, T);
      const sc = Math.min(pw - 6, ph) / 6.4, P = (x) => [(pw - 6) / 2 + x[0] * sc, ph / 2 - x[1] * sc];
      // decision surface
      const cell = 8;
      for (let px = 0; px < pw - 6; px += cell) for (let py = 0; py < ph; py += cell) {
        const x = [(px + cell / 2 - (pw - 6) / 2) / sc, (ph / 2 - py - cell / 2) / sc], z = b + w[0] * x[0] + w[1] * x[1];
        const p = S.model === "logistic" ? 1 / (1 + Math.exp(-z)) : z >= 0 ? 0.8 : 0.2;
        ctx.fillStyle = p >= 0.5 ? T.accent : T.accent2; ctx.globalAlpha = 0.06 + 0.16 * Math.abs(p - 0.5) * 2; ctx.fillRect(px, py, cell, cell);
      }
      ctx.globalAlpha = 1;
      X.forEach((x, i) => { const [a, q] = P(x); ctx.fillStyle = Y[i] ? T.accent : T.accent2; ctx.beginPath(); ctx.arc(a, q, 3, 0, 7); ctx.fill(); });
      if (Math.abs(w[1]) > 1e-9) { const f = (x0) => -(b + w[0] * x0) / w[1]; polyline(ctx, [P([-5, f(-5)]), P([5, f(5)])], T.ink, 1.5); }
      const hx = narrow ? 0 : pw + 4, ht = narrow ? ph + 8 : 0, hw = W - hx - 1, hh = H - ht - 1;
      ctx.fillStyle = T.ground; ctx.fillRect(hx, ht, hw, hh);
      axes(ctx, hx, ht, hw, hh, T, S.model === "logistic" ? "binary cross-entropy" : "misclassified per epoch");
      if (hist.length) { const mx = Math.max(...hist, 1e-9), n = budget();
        polyline(ctx, hist.map((v, i) => [hx + 6 + (i / (n - 1)) * (hw - 12), ht + hh - 6 - (v / mx) * (hh - 24)]), T.accent); }
      out.innerHTML = `${S.model} · epoch <b>${epoch}</b> · w=(<b>${w[0].toFixed(3)}, ${w[1].toFixed(3)}</b>) b=<b>${b.toFixed(3)}</b> · ` +
        (S.model === "logistic" ? `loss <b>${(hist[hist.length - 1] ?? NaN).toFixed(4)}</b>` : `errors <b>${hist[hist.length - 1] ?? "-"}</b>`) +
        (stopAt != null ? ` · <b>stopped at iteration ${stopAt}</b> (${S.model === "logistic" ? "|Δloss| < 1e-4" : "zero errors"})` : epoch >= budget() ? ` · budget of ${budget()} exhausted` : "");
    });
    api.onTheme(() => c.redraw());
    const ui = api.controls();
    let ready = false;
    const restart = () => { data(); playing = true; if (api.still) while (tick()); c.redraw(); };
    const mb = ui.button("Model: logistic", () => { S.model = S.model === "logistic" ? "perceptron" : "logistic"; mb.textContent = "Model: " + S.model; restart(); });
    ui.slider("class separation", 0, 2, 0.1, S.sep, (v) => { S.sep = v; ready && restart(); });
    ui.button("Replay", restart);
    ready = true; restart();
    let acc = 0;
    api.loop((t) => { if (!playing) return; if (t - acc < (S.model === "logistic" ? 0.03 : 0.25)) return; acc = t; if (tick()) c.redraw(); else playing = false; });
  });

  // ===== 5. MLP XOR, test_mlp_xor configuration =====
  TSViz.register("ml-mlp-xor", (stage, api) => {
    const out = readout(stage);
    const S = { lr: 0.1, mom: 0.9 };
    const Xs = [[0, 0], [0, 1], [1, 0], [1, 1]], Ys = [0, 1, 1, 0];
    let L, hist, epoch, playing;
    function layer(inp, outp, act, seed) { // DenseLayer::new, Xavier scale, crate LCG
      const s = Math.sqrt(2 / (inp + outp)); let rng = BigInt(seed); const W = [];
      for (let o = 0; o < outp; o++) { W.push([]); for (let i = 0; i < inp; i++) { rng = lcg(rng); W[o].push((Number(rng) / U64MAX * 2 - 1) * s); } }
      return { W, b: new Array(outp).fill(0), act, vW: W.map((r) => r.map(() => 0)), vb: new Array(outp).fill(0) };
    }
    const f = (a, z) => (a === "tanh" ? Math.tanh(z) : 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z)))));
    const df = (a, z) => { const v = f(a, z); return a === "tanh" ? 1 - v * v : v * (1 - v); };
    function forward(x) { let a = x; for (const l of L) { l.x = a; l.z = l.W.map((r, o) => r.reduce((s, w, i) => s + w * a[i], l.b[o])); a = l.z.map((z) => f(l.act, z)); } return a; }
    function trainStep(x, y) { // MLP::train_step, MSE, SGD with momentum
      const p = forward(x)[0], loss = (p - y) ** 2; let g = [2 * (p - y)];
      for (let k = L.length - 1; k >= 0; k--) {
        const l = L[k], d = g.map((gi, o) => gi * df(l.act, l.z[o]));
        const gin = l.x.map((_, i) => l.W.reduce((s, r, o) => s + r[i] * d[o], 0));
        l.W.forEach((r, o) => r.forEach((_, i) => { l.vW[o][i] = l.vW[o][i] * S.mom - S.lr * d[o] * l.x[i]; r[i] += l.vW[o][i]; }));
        l.b.forEach((_, o) => { l.vb[o] = l.vb[o] * S.mom - S.lr * d[o]; l.b[o] += l.vb[o]; });
        g = gin;
      }
      return loss;
    }
    const reset = () => { L = [layer(2, 8, "tanh", 42), layer(8, 1, "sigmoid", 43)]; hist = []; epoch = 0; playing = true; if (api.still) while (ep()); };
    const ep = () => { if (epoch >= 500) return false; let t = 0; Xs.forEach((x, i) => (t += trainStep(x, Ys[i]))); hist.push(t / 4); epoch++; return true; };
    const c = api.canvas(240, (ctx, W, H) => {
      const T = api.theme(); clear(ctx, W, H, T); if (!L) return;
      const s = Math.min(H, W * 0.45) - 2, n = 24, cs = s / n;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const p = forward([(i + 0.5) / n * 1.4 - 0.2, 1.2 - (j + 0.5) / n * 1.4])[0];
        ctx.fillStyle = mix(api.theme().dark ? "#1f1e1b" : "#ffffff", api.theme().dark ? "#8fb0ff" : "#2456dc", p * 0.85); ctx.fillRect(i * cs, j * cs, cs + 0.5, cs + 0.5);
      }
      Xs.forEach((x, i) => { const px = ((x[0] + 0.2) / 1.4) * s, py = ((1.2 - x[1]) / 1.4) * s; ctx.fillStyle = Ys[i] ? T.accent : T.accent2;
        ctx.strokeStyle = T.ground; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(px, py, 6, 0, 7); ctx.fill(); ctx.stroke(); });
      axes(ctx, 0, 0, s, s, T);
      const hx = s + 8, hw = W - hx - 1; axes(ctx, hx, 0, hw, H - 1, T, "MSE per epoch (500 epochs)");
      if (hist.length) { const mx = Math.max(...hist); polyline(ctx, hist.map((v, i) => [hx + 6 + (i / 499) * (hw - 12), H - 8 - (v / mx) * (H - 30)]), T.accent); }
      const preds = Xs.map((x) => forward(x)[0].toFixed(2));
      out.innerHTML = `epoch <b>${epoch}</b>/500 · loss <b>${(hist[hist.length - 1] ?? NaN).toFixed(4)}</b> · ŷ(00,01,10,11) = <b>${preds.join(", ")}</b>`;
    });
    api.onTheme(() => c.redraw());
    const ui = api.controls(); let ready = false;
    ui.slider("learning rate", 0.01, 1, 0.01, S.lr, (v) => { S.lr = v; ready && (reset(), c.redraw()); });
    ui.slider("momentum", 0, 0.95, 0.05, S.mom, (v) => { S.mom = v; ready && (reset(), c.redraw()); });
    ui.button("Retrain", () => { reset(); c.redraw(); });
    ready = true; reset(); c.redraw();
    api.loop(() => { if (!playing) return; for (let k = 0; k < 4; k++) if (!ep()) { playing = false; break; } c.redraw(); });
  });

  // ===== 6. Conv2D::forward =====
  TSViz.register("ml-conv2d", (stage, api) => {
    const out = readout(stage);
    const K = { "edge": [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], "vertical": [[1, 0, -1], [2, 0, -2], [1, 0, -1]], "blur": [[1, 2, 1], [2, 4, 2], [1, 2, 1]].map((r) => r.map((v) => v / 16)) };
    const names = Object.keys(K); let ki = 0;
    const S = { stride: 1, pad: 1 }, N = 12;
    const img = []; for (let y = 0; y < N; y++) { img.push([]); for (let x = 0; x < N; x++) img[y].push(((x - 5.5) ** 2 + (y - 5.5) ** 2 < 14) || (x > 8 && y < 3) ? 1 : 0); }
    let pos = 0, outImg, oh, ow;
    function conv() { // output = ReLU(b + Σ input[y*s-p+ky][x*s-p+kx] * W[ky][kx]), zero outside
      const k = K[names[ki]]; oh = Math.floor((N + 2 * S.pad - 3) / S.stride) + 1; ow = oh; outImg = [];
      for (let y = 0; y < oh; y++) { outImg.push([]); for (let x = 0; x < ow; x++) { let s = 0;
        for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) { const iy = y * S.stride - S.pad + ky, ix = x * S.stride - S.pad + kx;
          if (iy >= 0 && iy < N && ix >= 0 && ix < N) s += img[iy][ix] * k[ky][kx]; }
        outImg[y].push(Math.max(0, s)); } }
    }
    const c = api.canvas(230, (ctx, W, H) => {
      const T = api.theme(); clear(ctx, W, H, T); conv();
      const gap = 16, cell = Math.min((W - gap) / (N + 2 + oh + 2), (H - 20) / (N + 2)), ox = 0, oy = 18;
      font(ctx, 10); ctx.fillStyle = T.muted; ctx.fillText(`input ${N}×${N} (pad ${S.pad})`, 0, 11);
      const o2 = ox + (N + 2) * cell + gap; ctx.fillText(`ReLU output ${oh}×${ow}`, o2, 11);
      for (let y = -1; y <= N; y++) for (let x = -1; x <= N; x++) {
        const v = y >= 0 && y < N && x >= 0 && x < N ? img[y][x] : null;
        if (v === null && (S.pad === 0)) continue;
        ctx.fillStyle = v === null ? T.ground : v ? T.ink : mix(T.dark ? "#1f1e1b" : "#ffffff", T.dark ? "#3a3833" : "#cfcbc1", 0.35);
        ctx.fillRect(ox + (x + 1) * cell, oy + (y + 1) * cell, cell - 1, cell - 1);
        if (v === null) { ctx.strokeStyle = T.hair; ctx.strokeRect(ox + (x + 1) * cell + .5, oy + (y + 1) * cell + .5, cell - 2, cell - 2); }
      }
      const total = oh * ow, p = pos % total, py = Math.floor(p / ow), px = p % ow;
      const mx = Math.max(1e-9, ...outImg.flat());
      outImg.forEach((row, y) => row.forEach((v, x) => { const idx = y * ow + x;
        ctx.fillStyle = idx > p ? T.ground : mix(T.dark ? "#1f1e1b" : "#ffffff", T.dark ? "#8fb0ff" : "#2456dc", v / mx);
        ctx.fillRect(o2 + x * cell, oy + cell + y * cell, cell - 1, cell - 1); }));
      ctx.strokeStyle = T.accent2; ctx.lineWidth = 2;
      ctx.strokeRect(ox + (px * S.stride - S.pad + 1) * cell, oy + (py * S.stride - S.pad + 1) * cell, 3 * cell - 1, 3 * cell - 1);
      ctx.strokeRect(o2 + px * cell, oy + cell + py * cell, cell - 1, cell - 1);
      out.innerHTML = `kernel <b>${names[ki]}</b> · out[${py}][${px}] = ReLU(Σ in·W) = <b>${outImg[py][px].toFixed(3)}</b> · output size ⌊(${N}+2·${S.pad}−3)/${S.stride}⌋+1 = <b>${oh}</b>`;
    });
    api.onTheme(() => c.redraw());
    const ui = api.controls(); let ready = false;
    const kb = ui.button("Kernel: edge", () => { ki = (ki + 1) % names.length; kb.textContent = "Kernel: " + names[ki]; c.redraw(); });
    ui.slider("stride", 1, 3, 1, 1, (v) => { S.stride = v; pos = 0; ready && c.redraw(); });
    ui.slider("padding", 0, 1, 1, 1, (v) => { S.pad = v; pos = 0; ready && c.redraw(); });
    ready = true;
    if (api.still) { pos = 1e9; }
    let last = 0; api.loop((t) => { if (t - last > 0.12) { last = t; pos++; c.redraw(); } });
  });

  // ===== 7. GossipRing::tick / converge =====
  TSViz.register("ml-gossip", (stage, api) => {
    const out = readout(stage);
    const S = { n: 8, tol: 0.01 };
    let est, hist, iter, done, mean0;
    function reset() {
      const r = prng(9); est = [];
      for (let i = 0; i < S.n; i++) { let s = 0; const m = 10 + Math.floor(r() * 20), mu = r() * 10; for (let k = 0; k < m; k++) s += mu + gauss(r); est.push(s / m); } // compute_local_centroid
      mean0 = est.reduce((a, b) => a + b, 0) / S.n; hist = [est.slice()]; iter = 0; done = null;
    }
    const conv = () => { const m = est.reduce((a, b) => a + b, 0) / est.length; return est.every((v) => (v - m) ** 2 <= S.tol * S.tol); };
    function tick() { // node i mixes with i-1 (ring), alpha 0.5, on a snapshot
      if (done != null || iter >= 200) return false;
      const snap = est.slice(); est = est.map((v, i) => v * 0.5 + snap[i === 0 ? est.length - 1 : i - 1] * 0.5);
      iter++; hist.push(est.slice()); if (conv()) done = iter; return true;
    }
    const c = api.canvas(250, (ctx, W, H) => {
      const T = api.theme(); clear(ctx, W, H, T); if (!est) return;
      const all = hist.flat(), lo = Math.min(...all), hi = Math.max(...all), col = (v) => mix(T.dark ? "#ffb86b" : "#c2410c", T.dark ? "#8fb0ff" : "#2456dc", (v - lo) / (hi - lo || 1));
      const R = Math.min(H, W * 0.42) / 2 - 18, cx = R + 18, cy = H / 2, P = (i) => [cx + R * Math.cos((2 * Math.PI * i) / S.n - Math.PI / 2), cy + R * Math.sin((2 * Math.PI * i) / S.n - Math.PI / 2)];
      ctx.strokeStyle = T.hair; ctx.lineWidth = 1.2;
      for (let i = 0; i < S.n; i++) { const [a, b] = P(i === 0 ? S.n - 1 : i - 1), [x, y] = P(i), mx = (a + x) / 2, my = (b + y) / 2;
        ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(x, y); ctx.stroke(); const an = Math.atan2(y - b, x - a);
        ctx.beginPath(); ctx.moveTo(mx + 5 * Math.cos(an), my + 5 * Math.sin(an)); ctx.lineTo(mx - 4 * Math.cos(an - 0.6), my - 4 * Math.sin(an - 0.6)); ctx.lineTo(mx - 4 * Math.cos(an + 0.6), my - 4 * Math.sin(an + 0.6)); ctx.fillStyle = T.hair; ctx.fill(); }
      font(ctx, 9);
      est.forEach((v, i) => { const [x, y] = P(i); ctx.fillStyle = col(v); ctx.beginPath(); ctx.arc(x, y, 12, 0, 7); ctx.fill(); ctx.fillStyle = T.ground; ctx.textAlign = "center"; ctx.fillText(v.toFixed(1), x, y + 3); });
      ctx.textAlign = "left";
      const hx = cx + R + 30, hw = W - hx - 1; if (hw < 60) return;
      axes(ctx, hx, 0, hw, H - 1, T, "global_estimate per node");
      const Y = (v) => H - 8 - ((v - lo) / (hi - lo || 1)) * (H - 30), X = (k) => hx + 6 + (k / Math.max(hist.length - 1, 12)) * (hw - 12);
      for (let i = 0; i < S.n; i++) polyline(ctx, hist.map((h, k) => [X(k), Y(h[i])]), col(hist[0][i]), 1.2);
      ctx.setLineDash([3, 3]); polyline(ctx, [[hx, Y(mean0)], [hx + hw, Y(mean0)]], T.good, 1); ctx.setLineDash([]);
      const spread = Math.max(...est) - Math.min(...est);
      out.innerHTML = `tick <b>${iter}</b> · spread <b>${spread.toFixed(4)}</b> · mean <b>${(est.reduce((a, b) => a + b, 0) / S.n).toFixed(4)}</b> (start ${mean0.toFixed(4)})` + (done ? ` · <b>converged in ${done} ticks</b> (every node within tol of the mean)` : "");
    });
    api.onTheme(() => c.redraw());
    const ui = api.controls(); let ready = false;
    ui.slider("nodes", 2, 16, 1, S.n, (v) => { S.n = v; ready && (reset(), c.redraw()); });
    ui.slider("tolerance", 0.001, 0.5, 0.001, S.tol, (v) => { S.tol = v; ready && (reset(), c.redraw()); });
    ui.button("Tick", () => { tick(); c.redraw(); });
    ui.button("Restart", () => { reset(); c.redraw(); });
    ready = true; reset(); if (api.still) while (tick()); c.redraw();
    let last = 0; api.loop((t) => { if (t - last > 0.35) { last = t; if (tick()) c.redraw(); } });
  });

  // ===== 8. sparse trigger gate: Δ = ||μ(t) − μ(t_last)||₂ ≥ ε =====
  TSViz.register("ml-sparse-gate", (stage, api) => {
    const out = readout(stage);
    const S = { eps: 0.1 };
    const T0 = 240, r = prng(21), path = []; let p = [0, 0];
    for (let t = 0; t < T0; t++) { const burst = (t > 60 && t < 80) || (t > 160 && t < 170); p = [p[0] + (burst ? 0.06 : 0.008) * gauss(r), p[1] + (burst ? 0.06 : 0.008) * gauss(r)]; path.push(p); }
    let upto = 0;
    function evalGate() { let ref = path[0], wakes = [0], devs = [0];
      for (let t = 1; t < T0; t++) { const d = Math.hypot(path[t][0] - ref[0], path[t][1] - ref[1]); devs.push(d); if (d >= S.eps) { wakes.push(t); ref = path[t]; } } // should_trigger, then handle_event
      return { wakes, devs }; }
    const c = api.canvas(200, (ctx, W, H) => {
      const T = api.theme(); clear(ctx, W, H, T); const g = evalGate(), n = Math.min(upto, T0 - 1);
      axes(ctx, 0, 0, W - 1, H - 1, T, "Δ(t) since last wake, ε dashed, wakes marked");
      const mx = Math.max(S.eps * 1.6, ...g.devs), X = (t) => 6 + (t / (T0 - 1)) * (W - 12), Y = (v) => H - 8 - (v / mx) * (H - 30);
      polyline(ctx, g.devs.slice(0, n + 1).map((v, t) => [X(t), Y(v)]), T.accent, 1.3);
      ctx.setLineDash([4, 3]); polyline(ctx, [[0, Y(S.eps)], [W, Y(S.eps)]], T.accent2, 1); ctx.setLineDash([]);
      ctx.fillStyle = T.good; const wk = g.wakes.filter((t) => t <= n); wk.forEach((t) => ctx.fillRect(X(t) - 1, H - 7, 2, 6));
      out.innerHTML = `t = <b>${n}</b> · wakes <b>${wk.length}</b> · skips <b>${n + 1 - wk.length}</b> · work done on <b>${((100 * wk.length) / (n + 1)).toFixed(1)}%</b> of ticks at ε = ${S.eps}`;
    });
    api.onTheme(() => c.redraw());
    const ui = api.controls(); let ready = false;
    ui.slider("ε", 0.01, 0.5, 0.01, S.eps, (v) => { S.eps = v; ready && c.redraw(); });
    ui.button("Replay", () => { upto = 0; });
    ready = true; if (api.still) upto = T0; c.redraw();
    api.loop(() => { if (upto < T0) { upto += 2; c.redraw(); } });
  });
})();
