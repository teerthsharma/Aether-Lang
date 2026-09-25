// viz-kernel: sparse-event scheduler, governor, attention selectors, CSR schedule,
// and the measured ledgers on the kernel / status / evidence pages.
// Algorithms are line-for-line ports of crates/aether-kernel/src/scheduler.rs,
// crates/aether-core/src/{governor,attention,scheduled}.rs and the test RNGs, so
// the live numbers reproduce the Rust fixtures (seeded exactly as the tests seed).
(() => {
  if (!window.TSViz) return;
  const R = TSViz.register;

  // ── style (scoped) ─────────────────────────────────────────────────────────
  const st = document.createElement("style");
  st.textContent = `
  .ts-viz[data-viz^="k-"] .k-read{display:flex;flex-wrap:wrap;gap:.25rem 1rem;font-size:.66rem;margin-top:.5rem;font-family:var(--md-code-font-family);color:var(--md-default-fg-color--light)}
  .ts-viz[data-viz^="k-"] .k-read b{color:var(--md-default-fg-color);font-weight:600}
  .ts-viz[data-viz^="k-"] .k-read .bad{color:var(--k-bad)}
  .ts-viz[data-viz^="k-"] .k-read .ok{color:var(--k-good)}
  .ts-viz[data-viz^="k-"] button[aria-pressed="true"]{color:var(--blue-500);border-color:var(--blue-500)}
  .ts-viz[data-viz^="k-"] .k-csr{font-family:var(--md-code-font-family);font-size:.62rem;margin-top:.45rem;overflow-wrap:anywhere;color:var(--md-default-fg-color--light)}
  `;
  document.head.append(st);

  // ── shared helpers ─────────────────────────────────────────────────────────
  const FONT = (px, w = 400) => `${w} ${px}px "Instrument Sans", system-ui, sans-serif`;
  const MONO = (px) => `${px}px ui-monospace, "JetBrains Mono", monospace`;
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const alpha = (hex, a) => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
  };
  function setVars(stage, t) {
    const box = stage.parentElement;
    box.style.setProperty("--k-bad", t.accent2);
    box.style.setProperty("--k-good", t.good);
  }
  function readout(stage) {
    const d = document.createElement("div");
    d.className = "k-read";
    d.setAttribute("aria-live", "polite");
    stage.parentElement.insertBefore(d, stage.nextSibling);
    return d;
  }
  // A mounted canvas that also redraws on theme change.
  function cvs(stage, api, h, draw) {
    const c = api.canvas(h, (ctx, w, hh) => { setVars(stage, api.theme()); draw(ctx, w, hh, api.theme()); });
    api.onTheme(() => c.redraw());
    return c;
  }
  // Segmented toggle built from the shared button control; returns setter.
  function seg(ctl, labels, initial, onPick) {
    const bs = labels.map((l, i) => ctl.button(l, () => pick(i)));
    function pick(i) { bs.forEach((b, j) => b.setAttribute("aria-pressed", String(i === j))); onPick(i); }
    bs.forEach((b, j) => b.setAttribute("aria-pressed", String(j === initial)));
    return pick;
  }
  // Plot frame: linear or log axes, hairline grid, tick labels.
  function frame(ctx, w, h, t, o) {
    const m = Object.assign({ l: 44, r: 12, t: 10, b: 30 }, o.m || {});
    const tx = (v) => (o.xlog ? Math.log10(v) : v);
    const ty = (v) => (o.ylog ? Math.log10(v) : v);
    const [x0, x1] = o.x.map(tx), [y0, y1] = o.y.map(ty);
    const X = (v) => m.l + ((tx(v) - x0) / (x1 - x0)) * (w - m.l - m.r);
    const Y = (v) => h - m.b - ((ty(v) - y0) / (y1 - y0)) * (h - m.t - m.b);
    ctx.clearRect(0, 0, w, h);
    ctx.font = FONT(10);
    ctx.lineWidth = 1;
    ctx.strokeStyle = t.hair; ctx.fillStyle = t.muted;
    (o.yt || []).forEach((v) => {
      const y = Math.round(Y(v)) + 0.5;
      ctx.globalAlpha = 0.6; ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(w - m.r, y); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(o.yf ? o.yf(v) : String(v), m.l - 5, y);
    });
    (o.xt || []).forEach((v) => {
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText(o.xf ? o.xf(v) : String(v), X(v), h - m.b + 5);
    });
    ctx.beginPath(); ctx.moveTo(m.l + 0.5, m.t); ctx.lineTo(m.l + 0.5, h - m.b + 0.5); ctx.lineTo(w - m.r, h - m.b + 0.5);
    ctx.stroke();
    if (o.xl) { ctx.textAlign = "right"; ctx.textBaseline = "bottom"; ctx.fillText(o.xl, w - m.r, h - 1); }
    if (o.yl) { ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(o.yl, m.l + 4, m.t); }
    return { X, Y, m };
  }
  function line(ctx, pts, color, width = 1.6, dash) {
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; if (dash) ctx.setLineDash(dash);
    ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke(); ctx.restore();
  }
  function dot(ctx, x, y, r, fill, stroke) {
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.4; ctx.stroke(); }
  }
  // Horizontal labelled bars: items [{label, value, color, note}].
  function hbars(ctx, w, h, t, items, o) {
    ctx.clearRect(0, 0, w, h);
    const lw = Math.min(o.lw || 150, w * 0.42), rw = o.rw || 46;
    const rowH = (h - 8) / items.length, bh = Math.min(16, rowH * 0.62);
    const max = o.max;
    const X = (v) => lw + (v / max) * (w - lw - rw);
    if (o.ref != null) {
      const x = Math.round(X(o.ref)) + 0.5;
      line(ctx, [[x, 0], [x, h]], t.muted, 1, [3, 3]);
      ctx.font = FONT(9); ctx.fillStyle = t.muted; ctx.textAlign = "left"; ctx.textBaseline = "top";
      if (o.refLabel) ctx.fillText(o.refLabel, x + 3, 0);
    }
    items.forEach((it, i) => {
      const y = 4 + i * rowH + rowH / 2;
      ctx.font = FONT(10.5); ctx.fillStyle = t.ink; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      let lab = it.label; while (ctx.measureText(lab).width > lw - 8 && lab.length > 4) lab = lab.slice(0, -2) + "…";
      ctx.fillText(lab, lw - 8, y);
      if (it.value == null) {
        ctx.save(); ctx.strokeStyle = t.accent2; ctx.setLineDash([3, 3]);
        ctx.strokeRect(lw + 0.5, y - bh / 2 + 0.5, (w - lw - rw) * 0.25, bh - 1); ctx.restore();
        ctx.font = FONT(9.5); ctx.fillStyle = t.accent2; ctx.textAlign = "left";
        ctx.fillText(it.note || "not run", lw + 6, y);
        return;
      }
      ctx.fillStyle = it.color || t.accent;
      const x0 = X(Math.min(0, it.value)), x1 = X(Math.max(0, it.value));
      ctx.fillRect(x0, y - bh / 2, Math.max(1.5, x1 - x0), bh);
      ctx.font = MONO(10); ctx.fillStyle = it.color || t.ink; ctx.textAlign = "left";
      ctx.fillText(it.text || String(it.value), x1 + 5, y);
    });
  }

  // ── RNGs, bit-exact with the Rust tests ────────────────────────────────────
  const M64 = (1n << 64n) - 1n;
  function XorShift(seed) { // tests/*: Rng::new(seed) = seed | 1; xorshift64* output
    let s = BigInt(seed) | 1n;
    const next = () => {
      let x = s; x ^= x >> 12n; x ^= (x << 25n) & M64; x ^= x >> 27n; s = x;
      return (x * 0x2545F4914F6CDD1Dn) & M64;
    };
    const unit = () => Number(next() >> 11n) / 2 ** 53;
    return { next, unit, signed: () => unit() * 2 - 1 };
  }
  function splitmix(z) { // attention.rs splitmix
    z = (z + 0x9E3779B97F4A7C15n) & M64;
    z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & M64;
    z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & M64;
    return z ^ (z >> 31n);
  }

  // ── attention.rs port ──────────────────────────────────────────────────────
  const dotQK = (q, k, i, j, d) => { let s = 0; for (let c = 0; c < d; c++) s += q[i * d + c] * k[j * d + c]; return s; };
  const distQK = (q, k, i, j, d) => { let s = 0; for (let c = 0; c < d; c++) { const e = q[i * d + c] - k[j * d + c]; s += e * e; } return Math.sqrt(s); };
  function find(p, x) { let r = x; while (p[r] !== r) r = p[r]; while (p[x] !== r) { const n = p[x]; p[x] = r; x = n; } return r; }
  function singleLinkage(pts, count, dim, clusters, normalize) {
    const P = new Float64Array(count * dim);
    for (let t = 0; t < count; t++) {
      let n = 0; for (let d = 0; d < dim; d++) n += pts[t * dim + d] ** 2;
      n = Math.sqrt(n); const inv = normalize ? (n > 0 ? 1 / n : 0) : 1;
      for (let d = 0; d < dim; d++) P[t * dim + d] = pts[t * dim + d] * inv;
    }
    const E = [];
    for (let i = 0; i < count; i++) for (let j = i + 1; j < count; j++) {
      let s = 0; for (let d = 0; d < dim; d++) { const e = P[i * dim + d] - P[j * dim + d]; s += e * e; }
      E.push([Math.sqrt(s), i, j]);
    }
    E.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
    const parent = [...Array(count).keys()], heights = [];
    let comps = count; const target = clamp(clusters, 1, Math.max(count, 1));
    for (const [h, i, j] of E) {
      const ri = find(parent, i), rj = find(parent, j);
      if (ri === rj) continue;
      heights.push(h);
      if (comps > target) { parent[ri] = rj; comps--; }
    }
    const labelOf = new Map(), asg = new Array(count);
    for (let t = 0; t < count; t++) { const r = find(parent, t); if (!labelOf.has(r)) labelOf.set(r, labelOf.size); asg[t] = labelOf.get(r); }
    return { asg, heights };
  }
  const ROUTING_COST_THRESHOLD = 0.6;
  function routedCost(k, seq, hd, budget, clusters, causal) { // selection_dot_cost, TopologicalRouted arm
    const { asg } = singleLinkage(k, seq, hd, clusters, true);
    const cc = Math.max(...asg) + 1, sizes = new Array(cc).fill(0);
    asg.forEach((l) => sizes[l]++);
    const labels = [...Array(cc).keys()].sort((a, b) => sizes[b] - sizes[a] || a - b);
    let tot = 0;
    for (let i = 0; i < seq; i++) {
      const end = causal ? i + 1 : seq; let cand = 0;
      for (const l of labels) { if (cand >= budget) break; for (let j = 0; j < end; j++) if (asg[j] === l) cand++; }
      tot += cc + cand;
    }
    return tot / seq;
  }
  const denseCost = (seq, causal) => { let t = 0; for (let i = 0; i < seq; i++) t += causal ? i + 1 : seq; return t / seq; };
  function routingPlan(k, seq, hd, clusters, budget, causal) {
    const { asg, heights } = singleLinkage(k, seq, hd, clusters, true);
    const cc = Math.max(...asg) + 1, sizes = new Array(cc).fill(0);
    asg.forEach((l) => sizes[l]++);
    const cost = routedCost(k, seq, hd, budget, clusters, causal) / denseCost(seq, causal);
    const taken = Math.max(0, seq - cc), below = heights[taken - 1], above = heights[taken];
    const gap = taken >= 1 && below != null && above != null && below > 0 ? above / below : 1;
    return { cost, sizes, largest: Math.max(...sizes) / seq, gap, worth: cost < ROUTING_COST_THRESHOLD };
  }
  // select_mask; sel = {kind, budget, seed, clusters}
  function selectMask(sel, q, k, seq, hd, causal) {
    if (sel.kind === "adaptive")
      sel = routingPlan(k, seq, hd, sel.clusters, sel.budget, causal).worth ? { ...sel, kind: "routed" } : { kind: "dense" };
    const mask = new Uint8Array(seq * seq);
    let routing = null;
    if (sel.kind === "routed") {
      const { asg } = singleLinkage(k, seq, hd, sel.clusters, true);
      const cc = Math.max(...asg) + 1, cen = new Float64Array(cc * hd), mem = new Array(cc).fill(0);
      asg.forEach((l, t) => {
        let n = 0; for (let d = 0; d < hd; d++) n += k[t * hd + d] ** 2; n = Math.sqrt(n);
        const inv = n > 0 ? 1 / n : 0;
        for (let d = 0; d < hd; d++) cen[l * hd + d] += k[t * hd + d] * inv;
        mem[l]++;
      });
      for (let l = 0; l < cc; l++) for (let d = 0; d < hd; d++) cen[l * hd + d] /= Math.max(1, mem[l]);
      routing = { asg, cen, cc };
    }
    for (let i = 0; i < seq; i++) {
      const end = causal ? i + 1 : seq, legal = [...Array(end).keys()];
      let chosen;
      if (sel.kind === "dense") chosen = legal;
      else if (sel.kind === "random") {
        const pool = legal.slice();
        let s = splitmix(BigInt(sel.seed) ^ ((BigInt(i) * 0x9E3779B97F4A7C15n) & M64));
        const take = Math.min(sel.budget, pool.length);
        for (let slot = 0; slot < take; slot++) {
          s = splitmix(s);
          const pick = slot + Number(s % BigInt(pool.length - slot));
          [pool[slot], pool[pick]] = [pool[pick], pool[slot]];
        }
        chosen = pool.slice(0, take);
      } else if (sel.kind === "oracle") {
        chosen = legal.map((j) => [j, dotQK(q, k, i, j, hd)]).sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, sel.budget).map((e) => e[0]);
      } else if (sel.kind === "nearest") { // Topological, radius_scale = INFINITY
        chosen = legal.map((j) => [j, distQK(q, k, i, j, hd)]).sort((a, b) => a[1] - b[1] || a[0] - b[0]).slice(0, sel.budget).map((e) => e[0]);
        if (!chosen.length) chosen = [i];
      } else { // routed
        const { asg, cen, cc } = routing;
        const ranked = [...Array(cc).keys()].map((l) => { let s = 0; for (let d = 0; d < hd; d++) s += q[i * hd + d] * cen[l * hd + d]; return [l, s]; })
          .sort((a, b) => b[1] - a[1] || a[0] - b[0]);
        const cand = [];
        for (const [l] of ranked) { if (cand.length >= sel.budget) break; for (const j of legal) if (asg[j] === l) cand.push(j); }
        chosen = cand.map((j) => [j, dotQK(q, k, i, j, hd)]).sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, sel.budget).map((e) => e[0]);
        if (!chosen.length) chosen = [i];
      }
      for (const j of chosen) mask[i * seq + j] = 1;
    }
    return mask;
  }
  function softmaxRows(q, k, seq, hd, causal) {
    const W = new Float64Array(seq * seq), sc = 1 / Math.sqrt(hd);
    for (let i = 0; i < seq; i++) {
      const end = causal ? i + 1 : seq; let mx = -Infinity;
      for (let j = 0; j < end; j++) { W[i * seq + j] = dotQK(q, k, i, j, hd) * sc; mx = Math.max(mx, W[i * seq + j]); }
      let den = 0; for (let j = 0; j < end; j++) den += Math.exp(W[i * seq + j] - mx);
      for (let j = 0; j < end; j++) W[i * seq + j] = Math.exp(W[i * seq + j] - mx) / den;
    }
    return W;
  }
  const massOf = (mask, W, seq) => { let t = 0; for (let x = 0; x < seq * seq; x++) if (mask[x]) t += W[x]; return t / seq; };

  // ── scheduled.rs port ──────────────────────────────────────────────────────
  function blockCentroids(keys, nb, bs, dim) {
    const c = new Float64Array(nb * dim);
    for (let b = 0; b < nb; b++) { for (let t = 0; t < bs; t++) for (let d = 0; d < dim; d++) c[b * dim + d] += keys[(b * bs + t) * dim + d]; for (let d = 0; d < dim; d++) c[b * dim + d] /= bs; }
    return c;
  }
  function blockSalience(keys, nb, bs, dim) { // elder rule over centroids; also returns merge log
    if (nb === 1) return { sal: [1], merges: [] };
    const c = blockCentroids(keys, nb, bs, dim), E = [];
    for (let i = 0; i < nb; i++) for (let j = i + 1; j < nb; j++) {
      let s = 0; for (let d = 0; d < dim; d++) s += (c[i * dim + d] - c[j * dim + d]) ** 2; E.push([Math.sqrt(s), i, j]);
    }
    E.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
    const parent = [...Array(nb).keys()], members = parent.map((i) => [i]), sal = new Array(nb).fill(0), merges = [];
    for (const [dist, l, r] of E) {
      let a = find(parent, l), b = find(parent, r);
      if (a === b) continue;
      if (members[a].length > members[b].length) [a, b] = [b, a];
      for (const blk of members[a]) sal[blk] = dist;
      merges.push([dist, l, r, members[a].slice()]);
      parent[a] = b; members[b].push(...members[a]); members[a] = [];
    }
    return { sal, merges, c };
  }
  function topologySchedule(keys, nb, bs, dim, cfg) { // topology_block_schedule
    const { sal } = blockSalience(keys, nb, bs, dim);
    const topk = Math.min(cfg.topk, nb);
    const ranked = [...Array(nb).keys()].sort((a, b) => sal[b] - sal[a] || a - b);
    const salient = new Array(nb).fill(false); ranked.slice(0, topk).forEach((b) => (salient[b] = true));
    const offsets = [0], indices = [], src = [];
    for (let q = 0; q < nb; q++) {
      const allowed = new Array(q + 1).fill(0);
      for (let b = 0; b < Math.min(cfg.sink, nb, q + 1); b++) allowed[b] |= 1;
      for (let b = Math.max(0, q - cfg.radius); b <= q; b++) allowed[b] |= 2;
      allowed.forEach((_, b) => { if (salient[b]) allowed[b] |= 4; });
      if (!allowed.some(Boolean)) allowed[q] = 2;
      for (let b = 0; b <= q; b++) if (allowed[b]) { indices.push(b); src.push(allowed[b]); }
      offsets.push(indices.length);
    }
    return { offsets, indices, src, sal, salient };
  }
  function blockMassTable(q, k, seq, hd, bs) {
    const nb = seq / bs, T = new Float64Array(nb * nb), W = softmaxRows(q, k, seq, hd, true);
    for (let r = 0; r < seq; r++) for (let c = 0; c <= r; c++) T[((r / bs) | 0) * nb + ((c / bs) | 0)] += W[r * seq + c];
    for (let x = 0; x < T.length; x++) T[x] /= bs;
    return T;
  }
  const rowsOf = (s) => s.offsets.slice(0, -1).map((o, q) => s.indices.slice(o, s.offsets[q + 1]));
  const recovered = (rows, T, nb) => rows.reduce((t, row, q) => t + row.reduce((u, b) => u + T[q * nb + b], 0), 0) / nb;
  function randomSchedule(budget, seed) { // random_block_schedule
    let s = (BigInt(seed) * 6364136223846793005n + 1442695040888963407n) & M64;
    const next = () => { s = (s * 6364136223846793005n + 1442695040888963407n) & M64; return s >> 33n; };
    return budget.map((want, q) => {
      const cands = q + 1, take = Math.min(want, cands), pool = [...Array(cands).keys()];
      for (let i = 0; i < take; i++) { const j = i + Number(next() % BigInt(cands - i)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
      return pool.slice(0, take).sort((a, b) => a - b);
    });
  }
  const oracleSchedule = (T, nb, budget) => budget.map((want, q) =>
    [...Array(q + 1).keys()].sort((a, b) => T[q * nb + b] - T[q * nb + a] || a - b).slice(0, Math.min(want, q + 1)).sort((a, b) => a - b));
  function blockyKeys(nb, bs, dim, seed) { // tests/scheduled_attention.rs blocky_keys
    const rng = XorShift(seed);
    const centers = [...Array(nb)].map(() => [...Array(dim)].map(() => 4 * rng.signed()));
    const keys = new Float64Array(nb * bs * dim);
    for (let b = 0; b < nb; b++) for (let t = 0; t < bs; t++) for (let d = 0; d < dim; d++) keys[(b * bs + t) * dim + d] = centers[b][d] + 0.05 * rng.signed();
    return keys;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // sparse-events.md
  // ════════════════════════════════════════════════════════════════════════════
  // governor.rs constants, verbatim.
  const GOV = { TARGET_TICK_RATE: 1000, ALPHA: 0.01, BETA: 0.05, EPSILON_MIN: 0.001, EPSILON_MAX: 10, EPSILON_INITIAL: 0.1 };
  function Governor() {
    const g = { eps: GOV.EPSILON_INITIAL, lastErr: 0, n: 0 };
    g.adapt = (dev, dt) => {
      if (dt <= 0 || g.eps <= 0) return g.eps;
      const rate = dev / g.eps, err = GOV.TARGET_TICK_RATE - rate, dErr = (err - g.lastErr) / dt;
      g.eps -= GOV.ALPHA * err + GOV.BETA * dErr;
      g.lastErr = err; g.n++;
      g.eps = clamp(g.eps, GOV.EPSILON_MIN, GOV.EPSILON_MAX);
      return g.eps;
    };
    return g;
  }

  R("k-wake", (stage, api) => {
    // SparseScheduler<2>: one interrupt per 1 ms tick (timestamps in µs, as SystemState).
    const N = 240, hist = [];
    let sch, mu, t, rngState = 7, amp = 0.05, burst = 0.6, useGov = true;
    const rnd = () => ((rngState = (rngState * 1103515245 + 12345) % 2147483648) / 2147483648) * 2 - 1;
    function reset() {
      sch = { gov: Governor(), last: { v: [0, 0], ts: 0 }, pool: 0n, events: 0, skips: 0, fixedEps: 0.1 };
      mu = [0, 0]; t = 0; hist.length = 0;
    }
    function tick() {
      t += 1000;
      mu = [mu[0] + amp * rnd() * 0.3, mu[1] + amp * rnd() * 0.3];
      if (t % 60000 === 0) mu = [mu[0] + burst, mu[1] - burst * 0.5]; // periodic burst every 60 ticks
      const dev = Math.hypot(mu[0] - sch.last.v[0], mu[1] - sch.last.v[1]); // state.rs deviation (L2)
      const eps = useGov ? sch.gov.eps : sch.fixedEps;
      const wake = dev >= eps;                                              // governor.should_trigger
      if (wake) {
        const dt = t > sch.last.ts ? (t - sch.last.ts) / 1e6 : 0.001;       // handle_event, DEFAULT_DT
        if (useGov) sch.gov.adapt(dev, dt);
        sch.last = { v: mu.slice(), ts: t }; sch.events++;
      } else {
        sch.pool = (sch.pool * 6364136223846793005n + 1n) & M64;            // accumulate_entropy
        sch.skips++;
      }
      hist.push({ dev, eps, wake });
      if (hist.length > N) hist.shift();
    }
    reset();
    for (let i = 0; i < N; i++) tick();
    const lo = 1e-4, hi = 30;
    const c = cvs(stage, api, 230, (ctx, w, h, th) => {
      const { X, Y } = frame(ctx, w, h, th, { x: [0, N - 1], y: [lo, hi], ylog: true, yt: [1e-3, 1e-2, 0.1, 1, 10],
        yf: (v) => (v >= 1 ? v : v.toExponential(0)), xl: "last 240 ticks (1 ms each)", m: { l: 38 } });
      // clamp band
      ctx.fillStyle = alpha(th.accent, 0.06);
      ctx.fillRect(X(0), Y(GOV.EPSILON_MAX), X(N - 1) - X(0), Y(GOV.EPSILON_MIN) - Y(GOV.EPSILON_MAX));
      const cl = (v) => clamp(v, lo, hi);
      line(ctx, hist.map((p, i) => [X(i), Y(cl(p.dev))]), th.muted, 1.1);
      line(ctx, hist.map((p, i) => [X(i), Y(cl(p.eps))]), th.accent, 2);
      hist.forEach((p, i) => p.wake && dot(ctx, X(i), Y(cl(p.dev)), 2.6, th.good));
      ctx.font = FONT(10); ctx.textAlign = "left"; ctx.textBaseline = "bottom";
      ctx.fillStyle = th.accent; ctx.fillText("ε(t)", X(N - 1) - 26, Y(cl(hist[hist.length - 1].eps)) - 3);
      ctx.fillStyle = th.muted; ctx.fillText("Δ(t) = ‖μ(t) − μ(t_last)‖₂   ● wake", 42, 22);
    });
    const out = readout(stage);
    const show = () => {
      const tot = sch.events + sch.skips;
      const eps = useGov ? sch.gov.eps : sch.fixedEps;
      const pinned = useGov && (eps === GOV.EPSILON_MIN || eps === GOV.EPSILON_MAX);
      out.innerHTML = `<span>events <b>${sch.events}</b></span><span>skips <b>${sch.skips}</b></span>` +
        `<span>event_ratio <b>${(tot ? sch.events / tot : 0).toFixed(3)}</b></span>` +
        `<span>ε <b class="${pinned ? "bad" : ""}">${eps.toPrecision(3)}${pinned ? " (clamped)" : ""}</b></span>` +
        `<span>entropy_pool <b>0x${sch.pool.toString(16).padStart(16, "0").slice(0, 10)}…</b></span>`;
    };
    show();
    const ctl = api.controls();
    ctl.slider("noise", 0.005, 0.2, 0.005, amp, (v) => (amp = v));
    ctl.slider("burst", 0, 2, 0.05, burst, (v) => (burst = v));
    seg(ctl, ["adaptive ε (governor.rs)", "fixed ε = 0.1"], 0, (i) => { useGov = i === 0; reset(); });
    ctl.button("reset", reset);
    let acc = 0, prev = 0;
    api.loop((s) => {
      acc += Math.min(0.1, s - prev); prev = s;
      while (acc > 1 / 60) { tick(); acc -= 1 / 60; }
      c.redraw(); show();
    });
  });

  R("k-governor", (stage, api) => {
    let dev = 0.05, dt = 0.001;
    const STEPS = 40;
    const out = readout(stage);
    const c = cvs(stage, api, 210, (ctx, w, h, th) => {
      const g = Governor(), eps = [g.eps];
      for (let i = 0; i < STEPS; i++) eps.push(g.adapt(dev, dt));
      const { X, Y } = frame(ctx, w, h, th, { x: [0, STEPS], y: [5e-4, 20], ylog: true, yt: [1e-3, 1e-2, 0.1, 1, 10],
        yf: (v) => (v >= 1 ? v : v.toExponential(0)), xt: [0, 10, 20, 30, 40], xl: "adapt() call", m: { l: 38 } });
      [GOV.EPSILON_MIN, GOV.EPSILON_MAX].forEach((v) => line(ctx, [[X(0), Y(v)], [X(STEPS), Y(v)]], th.accent2, 1, [4, 3]));
      const eq = dev / GOV.TARGET_TICK_RATE; // rate = Δ/ε = TARGET ⇒ error 0
      if (eq > 5e-4 && eq < 20) line(ctx, [[X(0), Y(eq)], [X(STEPS), Y(eq)]], th.good, 1, [2, 3]);
      line(ctx, eps.map((e, i) => [X(i), Y(e)]), th.accent, 2);
      eps.forEach((e, i) => dot(ctx, X(i), Y(e), 2.2, th.accent));
      ctx.font = FONT(9.5); ctx.fillStyle = th.accent2; ctx.textAlign = "right"; ctx.textBaseline = "bottom";
      ctx.fillText("EPSILON_MAX 10", w - 14, Y(10) - 2);
      ctx.textBaseline = "top"; ctx.fillText("EPSILON_MIN 0.001", w - 14, Y(1e-3) + 2);
      if (eq > 5e-4 && eq < 20) { ctx.fillStyle = th.good; ctx.textAlign = "left"; ctx.fillText("error = 0 at ε = Δ/1000", 44, Y(eq) + 2); }
      const tail = eps.slice(-6), pinned = tail.every((e) => e === GOV.EPSILON_MIN || e === GOV.EPSILON_MAX);
      out.innerHTML = `<span>ε after ${STEPS} calls <b class="${pinned ? "bad" : ""}">${eps[STEPS].toPrecision(3)}</b></span>` +
        `<span>last 6 ${pinned ? '<b class="bad">sit on a clamp</b>' : "<b>interior</b>"}</span>` +
        `<span>adjustment = α·e + β·de/dt, α=0.01, β=0.05, target 1000</span>`;
    });
    const ctl = api.controls();
    ctl.slider("log₁₀ Δ", -4, 3, 0.1, Math.log10(dev), (v) => { dev = 10 ** v; c.redraw(); });
    ctl.slider("dt ms", 1, 100, 1, 1, (v) => { dt = v / 1000; c.redraw(); });
  });

  // ════════════════════════════════════════════════════════════════════════════
  // hardware-boundary.md / evidence-gates.md — evidence ledgers
  // ════════════════════════════════════════════════════════════════════════════
  function matrix(stage, api, rows, cols, cell, note) {
    const c = cvs(stage, api, 30 + rows.length * 26, (ctx, w, h, th) => {
      ctx.clearRect(0, 0, w, h);
      const lw = Math.min(170, w * 0.42), cw = (w - lw) / cols.length;
      ctx.font = FONT(9.5); ctx.fillStyle = th.muted; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      cols.forEach((cl, j) => ctx.fillText(w < 520 ? cl.short : cl.label, lw + cw * (j + 0.5), 22));
      rows.forEach((r, i) => {
        const y = 30 + i * 26 + 13;
        ctx.font = FONT(10.5); ctx.fillStyle = th.ink; ctx.textAlign = "right"; ctx.textBaseline = "middle";
        let lab = r; while (ctx.measureText(lab).width > lw - 8 && lab.length > 4) lab = lab.slice(0, -2) + "…";
        ctx.fillText(lab, lw - 8, y);
        cols.forEach((_, j) => {
          const v = cell(i, j), x = lw + cw * (j + 0.5);
          if (v === 1) dot(ctx, x, y, 5, th.good);
          else if (v === 0) dot(ctx, x, y, 5, null, th.accent2);
          else { ctx.fillStyle = th.hair; ctx.fillRect(x - 4, y - 0.5, 8, 1); }
        });
      });
    });
    const o = readout(stage); o.innerHTML = note;
    return c;
  }
  R("k-hw-gates", (stage, api) => {
    const claims = ["general boot support", "binary authentication", "measured power reduction", "real-time guarantees", "verified hw isolation"];
    const need = [{ label: "hw config", short: "cfg" }, { label: "boot steps", short: "boot" }, { label: "logs", short: "logs" },
      { label: "test artifacts", short: "tests" }, { label: "failure modes", short: "fail" }];
    matrix(stage, api, claims, need, () => 0,
      `<span><b class="bad">○</b> required artifact, not committed</span><span><b>0 / 25</b> present — every gated claim stays gated</span>`);
  });
  R("k-evidence-gates", (stage, api) => {
    // Row → evidence exists on the status page (1) or listed under "Current Gaps" (0).
    const rows = [
      ["Parser support", 1], ["Interpreter behavior", 1], ["Titan VM behavior", 0], ["Topology correctness", 1],
      ["Witness-mode behavior", 1], ["ML primitive behavior", 1], ["CLI behavior", 1], ["no_std compatibility", 1],
      ["Speed claim", 0], ["Security claim", 0], ["Hardware claim", 0]];
    matrix(stage, api, rows.map((r) => r[0]), [{ label: "evidence committed", short: "evidence" }], (i) => rows[i][1],
      `<span><b class="ok">●</b> 7 claim types have committed evidence</span><span><b class="bad">○</b> 4 map to a Current Gap (Titan parity, E2E/speed, security corpus, hardware)</span>`);
  });

  // ════════════════════════════════════════════════════════════════════════════
  // reference/status.md
  // ════════════════════════════════════════════════════════════════════════════
  R("k-ledger", (stage, api) => {
    // Active table rows that state an explicit test count, plus the Partial "Sparse scheduler" row.
    const rows = [
      ["attention_contracts.rs", 29], ["diagram_distance.rs", 17], ["scheduled_attention.rs", 16],
      ["persistence_invariants.rs", 12], ["interpreter.rs", 11], ["persistence.rs", 9], ["parser.rs", 7],
      ["manifold.rs", 7], ["persistence_scale.rs", 7], ["governor.rs", 5], ["lexer.rs", 4], ["aether.rs (blocks)", 4],
      ["scheduler.rs — never runs", 4, 1]];
    const c = cvs(stage, api, rows.length * 20 + 10, (ctx, w, h, th) =>
      hbars(ctx, w, h, th, rows.map(([l, v, bad]) => ({ label: l, value: v, color: bad ? th.accent2 : th.accent, text: bad ? v + " · 0 executed" : String(v) })), { max: 29, lw: 170, rw: 80 }));
  });

  R("k-mutation", (stage, api) => {
    let mode = 0;
    const c = cvs(stage, api, 150, (ctx, w, h, th) => {
      if (mode === 0) {
        // "caught by 4, 4, and 7 of the 11 tests … The six pre-existing example tests caught 0, 0, and 1."
        const defs = [["dropped triangle edge", 4, 0], ["hardcoded +0.001 epsilon", 4, 0], ["reduction stops after 1 column", 7, 1]];
        const items = [];
        defs.forEach(([l, a, b]) => { items.push({ label: l + " · invariants", value: a, text: `${a} / 11` }); items.push({ label: "pre-existing examples", value: b, color: th.accent2, text: `${b} / 6` }); });
        hbars(ctx, w, h, th, items, { max: 11, lw: 190, rw: 44 });
      } else {
        const items = [["skip descending sort", 2], ["drop persistence weight", 1], ["max instead of sum", 1], ["hardcode Gaussian σ", 1], ["forbid diagonal projection", null]]
          .map(([l, v]) => ({ label: l, value: v, text: v + " / 17", note: "not run — ∞ costs diverge" }));
        hbars(ctx, w, h, th, items, { max: 17, lw: 170, rw: 44 });
      }
    });
    seg(api.controls(), ["invariant suite", "diagram suite"], 0, (i) => { mode = i; c.redraw(); });
  });

  R("k-circle", (stage, api) => {
    // Circle ground truth: H1 death = 2r·sin(π·⌈n/3⌉/n) → √3·r.
    const tested = [9, 10, 11, 12, 13, 17, 24, 48];
    let hover = 12;
    const f = (n) => 2 * Math.sin((Math.PI * Math.ceil(n / 3)) / n);
    const c = cvs(stage, api, 200, (ctx, w, h, th) => {
      const { X, Y } = frame(ctx, w, h, th, { x: [3, 48], y: [1.55, 2.02], yt: [1.6, 1.7, Math.sqrt(3), 1.8, 1.9, 2], yf: (v) => (v === Math.sqrt(3) ? "√3" : v.toFixed(1)), xt: [3, 9, 12, 24, 36, 48], xl: "n points on the circle (r = 1)" });
      line(ctx, [[X(3), Y(Math.sqrt(3))], [X(48), Y(Math.sqrt(3))]], th.good, 1, [4, 3]);
      for (let n = 3; n <= 48; n++) {
        const t = tested.includes(n);
        dot(ctx, X(n), Y(f(n)), t ? 3.6 : 2, t ? th.accent : th.hair, t ? null : null);
      }
      dot(ctx, X(hover), Y(f(hover)), 6, null, th.ink);
      ctx.font = MONO(10); ctx.fillStyle = th.ink; ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText(`n=${hover}: 2·sin(π·${Math.ceil(hover / 3)}/${hover}) = ${f(hover).toFixed(12)}`, 48, 12);
      ctx.fillStyle = th.muted; ctx.fillText(`${tested.includes(hover) ? "asserted to 1e-12" : "not in the tested set"}${hover % 3 === 0 ? " · 3 | n ⇒ exactly √3" : ""}`, 48, 26);
    });
    api.controls().slider("n", 3, 48, 1, hover, (v) => { hover = v; c.redraw(); });
  });

  R("k-diagram", (stage, api) => {
    // Fixtures from tests/diagram_distance.rs; distances computed here by exhaustive matching.
    const FX = [
      { name: "pairing", A: [[0, 1]], B: [[0.2, 1.1]] },
      { name: "diagonal", A: [[0, 1]], B: [[0, 1], [0.5, 0.9]] },
      { name: "bottleneck vs W₁", A: [[0, 1], [2, 3]], B: [[0.2, 1.2], [2.2, 3.2]] },
      { name: "landscape", L: [[0, 1.2], [0.4, 3]] },
      { name: "tent [0,2]", L: [[0, 2]] }];
    let fi = 0, tt = 1;
    const linf = (p, q) => Math.max(Math.abs(p[0] - q[0]), Math.abs(p[1] - q[1]));
    const diag = (p) => (p[1] - p[0]) / 2;
    function match(A, B) { // augmented assignment, brute force (≤ 4! permutations)
      const n = A.length + B.length;
      const P = [...A, ...B.map((b) => ({ d: b }))], Q = [...B, ...A.map((a) => ({ d: a }))];
      const cost = (i, j) => { const p = P[i], q = Q[j];
        if (Array.isArray(p) && Array.isArray(q)) return i < A.length && j < B.length ? linf(p, q) : Infinity;
        if (Array.isArray(p)) return j >= B.length && Q[j].d === p ? diag(p) : Infinity;
        if (Array.isArray(q)) return i >= A.length && P[i].d === q ? diag(q) : Infinity;
        return 0; };
      let best = { b: Infinity, w: Infinity, perm: null }, bestW = Infinity;
      const perm = (arr, k) => { if (k === n) {
          const cs = arr.map((j, i) => cost(i, j)); const b = Math.max(...cs), s = cs.reduce((x, y) => x + y, 0);
          if (b < best.b || (b === best.b && s < best.w)) best = { b, w: s, perm: arr.slice() };
          bestW = Math.min(bestW, s); return; }
        for (let i = k; i < n; i++) { [arr[k], arr[i]] = [arr[i], arr[k]]; perm(arr, k + 1); [arr[k], arr[i]] = [arr[i], arr[k]]; } };
      perm([...Array(n).keys()], 0);
      return { ...best, w1: bestW, P, Q, nA: A.length, nB: B.length };
    }
    const out = readout(stage);
    const c = cvs(stage, api, 230, (ctx, w, h, th) => {
      const fx = FX[fi];
      if (fx.L) {
        const xmax = 3.2;
        const { X, Y } = frame(ctx, w, h, th, { x: [0, xmax], y: [0, 1.2], xt: [0, 0.5, 1, 1.5, 2, 2.5, 3], yt: [0, 0.2, 0.4, 0.6, 0.8, 1], xl: "t", yl: "λ_k(t)" });
        const tent = (b, t) => Math.max(0, Math.min(t - b[0], b[1] - t));
        const cols = [th.accent, th.accent2 === th.accent ? th.ink : th.muted];
        fx.L.forEach((b, i) => { const pts = []; for (let t = 0; t <= xmax; t += 0.01) pts.push([X(t), Y(tent(b, t))]); line(ctx, pts, cols[i], 1.2, [3, 3]); });
        if (fx.L.length > 1) { const pts1 = [], pts2 = [];
          for (let t = 0; t <= xmax; t += 0.01) { const v = fx.L.map((b) => tent(b, t)).sort((a, b) => b - a); pts1.push([X(t), Y(v[0])]); pts2.push([X(t), Y(v[1])]); }
          line(ctx, pts1, th.accent, 2.2); line(ctx, pts2, th.good, 2.2); }
        else { const pts = []; for (let t = 0; t <= xmax; t += 0.01) pts.push([X(t), Y(tent(fx.L[0], t))]); line(ctx, pts, th.accent, 2.2);
          [0, 0.5, 1, 1.5, 2].forEach((t) => dot(ctx, X(t), Y(tent(fx.L[0], t)), 3.5, th.accent)); }
        const vals = fx.L.map((b) => tent(b, tt)).sort((a, b) => b - a);
        line(ctx, [[X(tt), Y(0)], [X(tt), Y(1.2)]], th.hair, 1);
        vals.forEach((v, i) => dot(ctx, X(tt), Y(v), 4.5, i ? th.good : th.accent));
        out.innerHTML = fx.L.length > 1
          ? `<span>A=[0,1.2] B=[0.4,3.0]</span><span>λ₁(${tt.toFixed(2)}) <b>${vals[0].toFixed(3)}</b></span><span>λ₂ <b>${vals[1].toFixed(3)}</b></span><span>test: λ₁(1.0)=0.6, λ₂(1.0)=0.2</span>`
          : `<span>samples at 0, 0.5, 1, 1.5, 2 → <b>${[0, 0.5, 1, 1.5, 2].map((t) => tent(fx.L[0], t)).join(", ")}</b></span>`;
        return;
      }
      const r = match(fx.A, fx.B), lim = 3.4;
      const { X, Y } = frame(ctx, w, h, th, { x: [0, lim], y: [0, lim], xt: [0, 1, 2, 3], yt: [0, 1, 2, 3], xl: "birth", yl: "death", m: { l: 30 } });
      line(ctx, [[X(0), Y(0)], [X(lim), Y(lim)]], th.hair, 1);
      r.perm.forEach((j, i) => { const p = r.P[i], q = r.Q[j];
        if (Array.isArray(p) && Array.isArray(q)) line(ctx, [[X(p[0]), Y(p[1])], [X(q[0]), Y(q[1])]], th.muted, 1.2);
        else if (Array.isArray(p) || Array.isArray(q)) { const s = Array.isArray(p) ? p : q, m = (s[0] + s[1]) / 2; line(ctx, [[X(s[0]), Y(s[1])], [X(m), Y(m)]], th.accent2, 1.2, [3, 2]); } });
      fx.A.forEach((p) => dot(ctx, X(p[0]), Y(p[1]), 4.5, th.accent));
      fx.B.forEach((p) => dot(ctx, X(p[0]), Y(p[1]), 4.5, null, th.ink));
      out.innerHTML = `<span>● A &nbsp;○ B</span><span>bottleneck <b>${r.b.toFixed(3)}</b></span><span>1-Wasserstein <b>${r.w1.toFixed(3)}</b></span>` +
        (fi === 1 ? `<span class="bad">unmatched bar (0.5,0.9): persistence 0.4 → cost 0.2</span>` : "");
    });
    const ctl = api.controls();
    seg(ctl, FX.map((f) => f.name), 0, (i) => { fi = i; c.redraw(); });
    ctl.slider("t", 0, 3, 0.05, 1, (v) => { tt = v; c.redraw(); });
  });

  // Nearest-neighbour vs routed: exact reproduction of the test fixture.
  function spreadPlacement(spread) {
    const seq = 32, hd = 8, budget = 6, trials = 8;
    let near = 0, routed = 0; const mass = { r: 0, n: 0, o: 0, t: 0 };
    for (let trial = 0; trial < trials; trial++) {
      const rng = XorShift(trial * 2 + 67), n = seq * hd;
      const q = Array.from({ length: n }, () => rng.signed());
      for (let i = 0; i < n; i++) rng.signed(); // v, unused by the mask
      const k = new Float64Array(n);
      for (let t = 0; t < seq; t++) { const gain = 1 + spread * (rng.signed() * 0.5 + 0.5); for (let d = 0; d < hd; d++) k[t * hd + d] = rng.signed() * gain; }
      const W = softmaxRows(q, k, seq, hd, true), m = (s) => massOf(selectMask(s, q, k, seq, hd, true), W, seq);
      const r = m({ kind: "random", budget, seed: 1 }), o = m({ kind: "oracle", budget }), nn = m({ kind: "nearest", budget }), ro = m({ kind: "routed", budget, clusters: 6 });
      near += (nn - r) / (o - r); routed += (ro - r) / (o - r);
      mass.r += r / trials; mass.o += o / trials; mass.n += nn / trials; mass.t += ro / trials;
    }
    return { near: near / trials, routed: routed / trials, mass };
  }
  R("k-spread", (stage, api) => {
    // Page tables, verbatim: nearest-neighbour ablation (8 trials) and routed placement.
    const NN = [[0, 0.884], [0.5, 0.732], [1, 0.533], [2, 0.202], [4, -0.109], [8, -0.285]];
    const RT = [[0, 0.898], [2, 0.885], [4, 0.874], [8, 0.866]];
    let live = null, sp = 4;
    const c = cvs(stage, api, 240, (ctx, w, h, th) => {
      const { X, Y } = frame(ctx, w, h, th, { x: [0, 8], y: [-0.4, 1], xt: [0, 1, 2, 4, 8], yt: [-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1], yf: (v) => (v > 0 ? "+" : "") + v.toFixed(1), xl: "key-norm spread", yl: "placement", m: { l: 40 } });
      line(ctx, [[X(0), Y(0)], [X(8), Y(0)]], th.muted, 1, [4, 3]);
      ctx.font = FONT(9.5); ctx.fillStyle = th.muted; ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.fillText("random = 0", X(0) + 4, Y(0) - 2);
      line(ctx, NN.map(([x, y]) => [X(x), Y(y)]), th.accent2, 1.8);
      NN.forEach(([x, y]) => dot(ctx, X(x), Y(y), 3.4, th.accent2));
      line(ctx, RT.map(([x, y]) => [X(x), Y(y)]), th.accent, 2);
      RT.forEach(([x, y]) => dot(ctx, X(x), Y(y), 3.4, th.accent));
      ctx.fillStyle = th.accent; ctx.fillText("routed", X(8) - 38, Y(0.866) - 5);
      ctx.fillStyle = th.accent2; ctx.fillText("nearest-neighbour", X(4.3), Y(-0.15) - 4);
      if (live) {
        line(ctx, [[X(sp), Y(-0.4)], [X(sp), Y(1)]], th.hair, 1);
        dot(ctx, X(sp), Y(live.near), 6.5, null, th.accent2); dot(ctx, X(sp), Y(live.routed), 6.5, null, th.accent);
      }
    });
    const out = readout(stage);
    const run = () => {
      out.innerHTML = "<span>running 8 seeded trials…</span>";
      setTimeout(() => {
        live = spreadPlacement(sp); c.redraw();
        const row = NN.find((r) => r[0] === sp), rrow = RT.find((r) => r[0] === sp);
        const f = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3);
        out.innerHTML = `<span>live, spread ${sp}: nearest <b class="${live.near < 0 ? "bad" : ""}">${f(live.near)}</b>${row ? ` (table ${f(row[1])})` : ""}</span>` +
          `<span>routed <b>${f(live.routed)}</b>${rrow ? ` (table ${f(rrow[1])})` : ""}</span>` +
          `<span>mass: random ${live.mass.r.toFixed(4)} · nearest ${live.mass.n.toFixed(4)} · oracle ${live.mass.o.toFixed(4)}</span>`;
      }, 20);
    };
    const ctl = api.controls();
    const s = ctl.slider("spread", 0, 8, 0.5, sp, (v) => (sp = v));
    s.addEventListener("change", run);
    ctl.button("run the test fixture", run);
    run();
  });

  R("k-seeds", (stage, api) => {
    // seq 32, head_dim 8, budget 6, uniform random q/k — the tautological rows.
    const rows = [[67, 0.4875, 0.5789, 0.5879, 0.910], [71, 0.4813, 0.5648, 0.5743, 0.898], [73, 0.4882, 0.5619, 0.5742, 0.857], [79, 0.4947, 0.5795, 0.5850, 0.939]];
    cvs(stage, api, 170, (ctx, w, h, th) => {
      const { X, Y } = frame(ctx, w, h, th, { x: [0.47, 0.6], y: [0, 4], xt: [0.48, 0.52, 0.56, 0.6], xf: (v) => v.toFixed(2), xl: "attention mass recovered", m: { l: 56 } });
      rows.forEach(([seed, r, t, o, p], i) => {
        const y = Y(i + 0.5);
        ctx.font = FONT(10); ctx.fillStyle = th.ink; ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText("seed " + seed, 50, y);
        line(ctx, [[X(r), y], [X(o), y]], th.hair, 3);
        dot(ctx, X(r), y, 4, th.muted); dot(ctx, X(o), y, 4, null, th.ink); dot(ctx, X(t), y, 4.2, th.accent2);
        ctx.font = MONO(9.5); ctx.fillStyle = th.muted; ctx.textAlign = "left"; ctx.fillText("+" + p.toFixed(3), X(o) + 7, y);
      });
      ctx.save(); ctx.translate(w - 70, 26); ctx.rotate(-0.12); ctx.strokeStyle = th.accent2; ctx.lineWidth = 1.4;
      ctx.strokeRect(-50, -10, 100, 20); ctx.fillStyle = th.accent2; ctx.font = FONT(10, 650); ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("TAUTOLOGICAL", 0, 1); ctx.restore();
    });
    readout(stage).innerHTML = `<span>● random &nbsp;<b class="bad">●</b> nearest-neighbour &nbsp;○ oracle</span><span>equal key norms ⇒ ‖q−k‖² ranks like q·k</span>`;
  });

  R("k-routing-cost", (stage, api) => {
    let mode = 0;
    const c = cvs(stage, api, 190, (ctx, w, h, th) => {
      if (mode === 0) {
        // routing_cost example table (64 keys, budget 8).
        const rows = [["uniform random [61,1,1,1]", 0.999, "+0.942"], ["4 clusters [16,16,16,16]", 0.449, "+0.990"], ["8 clusters [8,8,8,8]", 0.528, "+0.995"], ["16 clusters [4,4,4,4]", 0.733, "+0.989"]];
        hbars(ctx, w, h, th, rows.map(([l, v, p]) => ({ label: l, value: v, color: v >= ROUTING_COST_THRESHOLD ? th.accent2 : th.accent, text: `${v.toFixed(3)} · ${p}` })),
          { max: 1.1, ref: ROUTING_COST_THRESHOLD, refLabel: "worth_routing: cost < 0.6", lw: 170, rw: 92 });
      } else if (mode === 1) {
        // Adaptive table + the rejected fallback and the misleading +7.6.
        const rows = [["structured → route", 0.449, "cost 0.449 · placement +0.980", th.accent], ["unstructured → dense", 1, "cost 1.000 · mass 1.000", th.good],
          ["rejected: budget-6 window", 0.014, "placement +0.014", th.accent2]];
        hbars(ctx, w, h, th, rows.map(([l, v, t, col]) => ({ label: l, value: v, color: col, text: t })), { max: 1.25, lw: 160, rw: 150 });
      } else {
        // gap_ratio, 6 trials each at seq 48.
        const { X } = frame(ctx, w, h, th, { x: [0.8, 3.2], y: [0, 1], xt: [1, 1.04, 2, 2.7, 3], xf: (v) => v.toFixed(2).replace(/\.00$/, ""), xl: "gap_ratio" });
        const y = h / 2;
        ctx.fillStyle = alpha(th.accent2, 0.18); ctx.fillRect(X(0.8), y - 22, X(1.04) - X(0.8), 18);
        ctx.fillStyle = alpha(th.accent, 0.18); ctx.fillRect(X(2.7), y + 4, X(3.2) - X(2.7), 18);
        line(ctx, [[X(1.04), y - 30], [X(1.04), y + 30]], th.accent2, 1.6);
        line(ctx, [[X(2.7), y - 30], [X(2.7), y + 30]], th.accent, 1.6);
        ctx.font = FONT(10); ctx.textBaseline = "middle"; ctx.textAlign = "right";
        ctx.fillStyle = th.accent2; ctx.fillText("uniform: max 1.04", X(1.04) - 4, y - 13);
        ctx.textAlign = "left"; ctx.fillStyle = th.accent; ctx.fillText("6 clusters: min 2.70", X(2.7) + 4, y + 13);
        ctx.fillStyle = th.muted; ctx.textAlign = "center"; ctx.fillText("no overlap", (X(1.04) + X(2.7)) / 2, y);
      }
    });
    seg(api.controls(), ["cost vs dense", "Adaptive", "gap_ratio"], 0, (i) => { mode = i; c.redraw(); });
  });

  R("k-router", (stage, api) => {
    // Live select_mask on 64 keys, head_dim 8, budget 8 — routing_cost's generator.
    const seq = 64, hd = 8, budget = 8;
    let groups = 4, spread = 0, seed = 67, si = 0, S = null;
    const SEL = [{ kind: "routed", name: "routed" }, { kind: "nearest", name: "nearest-nbr" }, { kind: "oracle", name: "oracle top-k" }, { kind: "random", name: "random" }, { kind: "adaptive", name: "Adaptive" }];
    function build() {
      const rng = XorShift(seed | 1), n = seq * hd;
      const q = Array.from({ length: n }, () => rng.signed()); for (let i = 0; i < n; i++) rng.signed();
      let k;
      if (groups === 0) k = Float64Array.from({ length: n }, () => rng.signed());
      else { // clustered_keys(seq, hd, groups, tightness 0.05, spread)
        const cen = [...Array(groups)].map(() => [...Array(hd)].map(() => rng.signed()));
        k = new Float64Array(n);
        for (let t = 0; t < seq; t++) { const g = 1 + spread * rng.unit(); for (let d = 0; d < hd; d++) k[t * hd + d] = (cen[t % groups][d] + 0.05 * rng.signed()) * g; }
      }
      const W = softmaxRows(q, k, seq, hd, true), clusters = groups || 4;
      const plan = routingPlan(k, seq, hd, clusters, budget, true);
      const r = massOf(selectMask({ kind: "random", budget, seed: 1 }, q, k, seq, hd, true), W, seq);
      const o = massOf(selectMask({ kind: "oracle", budget }, q, k, seq, hd, true), W, seq);
      S = { q, k, W, plan, r, o, clusters };
      pick();
    }
    function pick() {
      const sel = { ...SEL[si], budget, seed: 1, clusters: S.clusters };
      S.mask = selectMask(sel, S.q, S.k, seq, hd, true);
      S.m = massOf(S.mask, S.W, seq);
      c.redraw(); report();
    }
    const c = cvs(stage, api, 300, (ctx, w, h, th) => {
      if (!S) return;
      ctx.clearRect(0, 0, w, h);
      const side = Math.min(h - 6, w), cell = side / seq, ox = (w - side) / 2;
      for (let i = 0; i < seq; i++) for (let j = 0; j <= i; j++) {
        const wt = S.W[i * seq + j], sel = S.mask[i * seq + j];
        ctx.fillStyle = sel ? alpha(th.accent, 0.35 + 0.65 * Math.min(1, Math.sqrt(wt * 4))) : alpha(th.ink, Math.min(0.5, Math.sqrt(wt) * 0.9) + 0.03);
        ctx.fillRect(ox + j * cell, 3 + i * cell, Math.ceil(cell), Math.ceil(cell));
      }
      ctx.strokeStyle = th.hair; ctx.strokeRect(ox + 0.5, 3.5, side - 1, side - 1);
    });
    const out = readout(stage);
    function report() {
      const p = S.plan, place = (S.m - S.r) / (S.o - S.r);
      const top = S.plan.sizes.slice().sort((a, b) => b - a).slice(0, 4);
      const cost = SEL[si].kind === "routed" ? p.cost : SEL[si].kind === "adaptive" ? (p.worth ? p.cost : 1) : SEL[si].kind === "random" ? 0 : 1;
      out.innerHTML = `<span>H0 sizes <b>[${top.join(",")}]</b></span><span>cost_ratio <b class="${p.worth ? "ok" : "bad"}">${p.cost.toFixed(3)}</b></span>` +
        `<span>gap_ratio <b>${p.gap.toFixed(2)}</b></span><span>worth_routing <b class="${p.worth ? "ok" : "bad"}">${p.worth}</b></span>` +
        `<span>${SEL[si].name}: mass <b>${S.m.toFixed(3)}</b>, selection cost ${cost.toFixed(3)}× dense</span>` +
        `<span>placement <b class="${place < 0 ? "bad" : ""}">${SEL[si].kind === "adaptive" && !p.worth ? "n/a (dense)" : (place >= 0 ? "+" : "") + place.toFixed(3)}</b></span>`;
    }
    const ctl = api.controls();
    seg(ctl, SEL.map((s) => s.name), 0, (i) => { si = i; pick(); });
    ctl.slider("key groups (0 = uniform)", 0, 16, 1, groups, (v) => (groups = v)).addEventListener("change", build);
    ctl.slider("norm spread", 0, 8, 0.5, spread, (v) => (spread = v)).addEventListener("change", build);
    ctl.button("new draw", () => { seed += 2; build(); });
    build();
  });

  R("k-csr", (stage, api) => {
    // tests/scheduled_attention.rs: blocky_keys(16 blocks, block 4, dim 8, seed 13), radius 1, sink 1, topk 2 → 56 / 136.
    const bs = 4, dim = 8;
    let nb = 16, cfg = { radius: 1, sink: 1, topk: 2 }, seed = 13, S, qb = 0;
    function build() {
      const keys = blockyKeys(nb, bs, dim, seed), seq = nb * bs;
      S = topologySchedule(keys, nb, bs, dim, cfg);
      const rng = XorShift(seed + 1000), q = Float64Array.from({ length: seq * dim }, () => rng.signed());
      const T = blockMassTable(q, keys, seq, dim, bs), rows = rowsOf(S), budget = rows.map((r) => r.length);
      S.mass = { topo: recovered(rows, T, nb), rand: recovered(randomSchedule(budget, 7), T, nb), orc: recovered(oracleSchedule(T, nb, budget), T, nb) };
      c.redraw(); report();
    }
    const c = cvs(stage, api, 280, (ctx, w, h, th) => {
      if (!S) return;
      ctx.clearRect(0, 0, w, h);
      const side = Math.min(h - 20, w - 20), cell = side / nb, ox = (w - side) / 2 + 8, oy = 4;
      const col = (s) => (s & 4 ? th.accent2 : s & 1 ? th.good : th.accent);
      for (let q = 0; q < nb; q++) for (let b = 0; b <= q; b++) {
        ctx.fillStyle = alpha(th.ink, 0.05); ctx.fillRect(ox + b * cell + 1, oy + q * cell + 1, cell - 2, cell - 2);
      }
      for (let q = 0; q < nb; q++) for (let x = S.offsets[q]; x < S.offsets[q + 1]; x++) {
        const b = S.indices[x];
        ctx.fillStyle = alpha(col(S.src[x]), q === qb ? 1 : 0.55);
        ctx.fillRect(ox + b * cell + 1, oy + q * cell + 1, cell - 2, cell - 2);
      }
      ctx.strokeStyle = th.ink; ctx.lineWidth = 1.5; ctx.strokeRect(ox + 0.5, oy + qb * cell + 0.5, (qb + 1) * cell - 1, cell - 1);
      ctx.font = FONT(9); ctx.fillStyle = th.muted; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText("q" + qb, ox - 3, oy + (qb + 0.5) * cell);
      ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText("key block →", ox + side / 2, oy + side + 3);
    });
    const out = readout(stage);
    const arr = document.createElement("div"); arr.className = "k-csr"; out.after(arr);
    function report() {
      const dense = (nb * (nb + 1)) / 2, red = 1 - S.indices.length / dense;
      const row = S.indices.slice(S.offsets[qb], S.offsets[qb + 1]);
      out.innerHTML = `<span><b class="ok">■</b> sink <b style="color:var(--blue-500)">■</b> local <b class="bad">■</b> top-k salient</span>` +
        `<span>scheduled <b>${S.indices.length} / ${dense}</b> · <b>${(red * 100).toFixed(1)}%</b> reduction</span>` +
        `<span>mass at equal budget: topology ${S.mass.topo.toFixed(3)} · random ${S.mass.rand.toFixed(3)} · oracle ${S.mass.orc.toFixed(3)}</span>`;
      const off = S.offsets.length > 18 ? S.offsets.slice(0, 17).join(", ") + ", …" : S.offsets.join(", ");
      arr.textContent = `row q${qb} visits [${row.join(", ")}]   offsets [${off}]   salient blocks {${S.salient.map((s, i) => (s ? i : -1)).filter((i) => i >= 0).join(", ")}}`;
    }
    const ctl = api.controls();
    ctl.slider("blocks", 4, 16, 1, nb, (v) => { nb = v; qb = Math.min(qb, nb - 1); if (S) build(); });
    ctl.slider("local_radius", 0, 4, 1, cfg.radius, (v) => { cfg.radius = v; if (S) build(); });
    ctl.slider("sink", 0, 3, 1, cfg.sink, (v) => { cfg.sink = v; if (S) build(); });
    ctl.slider("topk", 0, 6, 1, cfg.topk, (v) => { cfg.topk = v; if (S) build(); });
    ctl.button("reset to test fixture", resetFx);
    function resetFx() { const box = stage.parentElement; box.querySelectorAll("input[type=range]").forEach((i, j) => { i.value = [16, 1, 1, 2][j]; i.dispatchEvent(new Event("input")); }); }
    build();
    let last = -1;
    api.loop((s) => { const k = Math.floor(s / 0.7) % nb; if (k !== last) { last = k; qb = k; c.redraw(); report(); } });
  });

  R("k-elder", (stage, api) => {
    // block_salience on 2-D centroids; shuffle block order to see the tie-break caveat.
    const nb = 12, bs = 4, dim = 2;
    let order = [...Array(nb).keys()], base, seed = 13;
    function gen() { base = blockyKeys(nb, bs, dim, seed); }
    function keysInOrder() { const k = new Float64Array(nb * bs * dim); order.forEach((src, dst) => k.set(base.subarray(src * bs * dim, (src + 1) * bs * dim), dst * bs * dim)); return k; }
    gen();
    const out = readout(stage);
    const c = cvs(stage, api, 250, (ctx, w, h, th) => {
      ctx.clearRect(0, 0, w, h);
      const { sal, merges, c: cen } = blockSalience(keysInOrder(), nb, bs, dim);
      const split = w < 520 ? w : w * 0.55, ph = w < 520 ? 150 : h;
      const X = (v) => 12 + ((v + 4.4) / 8.8) * (split - 24), Y = (v) => ph - 10 - ((v + 4.4) / 8.8) * (ph - 20);
      merges.forEach(([d, l, r]) => line(ctx, [[X(cen[l * 2]), Y(cen[l * 2 + 1])], [X(cen[r * 2]), Y(cen[r * 2 + 1])]], th.hair, 1));
      for (let b = 0; b < nb; b++) {
        dot(ctx, X(cen[b * 2]), Y(cen[b * 2 + 1]), 5, sal[b] === 0 ? th.accent2 : th.accent);
        ctx.font = FONT(9); ctx.fillStyle = th.muted; ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.fillText(b, X(cen[b * 2]) + 6, Y(cen[b * 2 + 1]));
      }
      // salience bars per block index + sorted multiset
      const bx = w < 520 ? 0 : split + 10, by = w < 520 ? ph + 6 : 10, bw = w - bx - 4, bh = (w < 520 ? h - ph - 10 : h - 20) / 2 - 8;
      const mx = Math.max(...sal) || 1, sw = bw / nb;
      const bars = (vals, y0, lab) => {
        ctx.font = FONT(9); ctx.fillStyle = th.muted; ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(lab, bx, y0);
        vals.forEach((v, i) => { const hh = (v / mx) * (bh - 12); ctx.fillStyle = v === 0 ? th.accent2 : th.accent; ctx.fillRect(bx + i * sw + 1, y0 + bh - hh, sw - 2, Math.max(1, hh)); });
      };
      bars(sal, by, "salience by block index");
      bars(sal.slice().sort((a, b) => b - a), by + bh + 8, "sorted multiset (the H0 barcode)");
      out.innerHTML = `<span>zero-salience block <b class="bad">${sal.indexOf(0)}</b> (never absorbed)</span><span>multiset <b>${sal.slice().sort((a, b) => b - a).map((v) => v.toFixed(2)).join(" ")}</b></span>`;
    });
    const ctl = api.controls();
    ctl.button("shuffle block order", () => { for (let i = nb - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; } c.redraw(); });
    ctl.button("original order", () => { order = [...Array(nb).keys()]; c.redraw(); });
  });

  R("k-reduction", (stage, api) => {
    cvs(stage, api, 110, (ctx, w, h, th) => hbars(ctx, w, h, th, [
      { label: "this port · 16 blocks (unit test)", value: 58.8, text: "58.8%  (56 / 136)" },
      { label: "Triton PR · seq 1024, RTX 4060", value: 56.6, color: th.muted, text: "56.6%  external" },
      { label: "Triton PR · seq 4096, RTX 4060", value: 80.9, color: th.muted, text: "80.9%  external" }],
      { max: 100, lw: 190, rw: 110 }));
    readout(stage).innerHTML = `<span>block reduction only — the port does not reproduce the Triton wall-clock (1.04x–3.48x), measured on hardware this workspace lacks</span>`;
  });

  R("k-scale", (stage, api) => {
    // scale_probe table verbatim: [dim, n, pairs, seconds]
    const T = [[0, 200, 200, 0.049], [0, 1000, 1000, 5.781], [0, 4000, 4000, 335.049], [1, 60, 1771, 0.117], [1, 120, 7141, 2.202],
      [1, 200, 19901, 20.728], [1, 300, 44851, 131.343], [2, 30, 4090, 0.1], [2, 50, 19650, 1.859], [2, 70, 54810, 15.338]];
    let xAxis = 0;
    const c = cvs(stage, api, 240, (ctx, w, h, th) => {
      const xi = xAxis ? 2 : 1, xr = xAxis ? [150, 1e5] : [20, 6000];
      const { X, Y } = frame(ctx, w, h, th, { x: xr, y: [0.03, 600], xlog: true, ylog: true, xt: xAxis ? [200, 1e3, 1e4, 1e5] : [30, 100, 300, 1000, 4000],
        yt: [0.1, 1, 10, 100], xl: xAxis ? "pairs (log)" : "points n (log)", yl: "seconds (log)", xf: (v) => (v >= 1e4 ? v / 1e3 + "k" : v) });
      const cols = [th.accent, th.good, th.accent2];
      [0, 1, 2].forEach((d) => {
        const pts = T.filter((r) => r[0] === d);
        line(ctx, pts.map((r) => [X(r[xi]), Y(r[3])]), cols[d], 1.8);
        pts.forEach((r) => dot(ctx, X(r[xi]), Y(r[3]), 3.4, cols[d]));
        const L = pts[pts.length - 1];
        ctx.font = FONT(10, 600); ctx.fillStyle = cols[d]; ctx.textAlign = "right"; ctx.textBaseline = "bottom"; ctx.fillText("H" + d, X(L[xi]) - 5, Y(L[3]) - 3);
      });
      // local slope for H0 between n=1000 and 4000: log(335.049/5.781)/log(4)
    });
    const s0 = Math.log(335.049 / 5.781) / Math.log(4);
    readout(stage).innerHTML = `<span>H0 n 1000→4000: slope <b>${s0.toFixed(2)}</b> in n (from the table rows)</span><span>face lookup indexed: <b>29.07 s → 1.10 s</b>, 26x on identical assertions</span><span>defaults: h2 48 · h1 128 · h0 512 points</span>`;
    seg(api.controls(), ["x = points", "x = pairs"], 0, (i) => { xAxis = i; c.redraw(); });
  });
})();
