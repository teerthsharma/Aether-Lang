// viz-integrated: visualisations for integrated/*.md.
// int-link mirrors crates/aether-core/src/linking.rs (gauss_sum, omega, direction,
// triangle, GaussLinking::certify) on the twisted_band generator of tests/linking.rs.
(() => {
  if (!window.TSViz) return;

  const st = document.createElement("style");
  st.textContent = `
.ts-viz[data-viz^="int-"] .int-read{font-family:var(--md-code-font-family);font-size:.64rem;line-height:1.55;margin-top:.55rem;color:var(--md-default-fg-color--light);overflow-wrap:anywhere}
.ts-viz[data-viz^="int-"] .int-read b{color:var(--md-default-fg-color);font-weight:600}
.ts-viz[data-viz^="int-"] .ts-viz-controls button[aria-pressed=true]{color:var(--blue-500);border-color:var(--blue-500)}`;
  document.head.append(st);

  // ---------- linking.rs, line for line ----------
  const U = 2 ** -53, K = 128 * U, TAU = 2 * Math.PI;
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  // direction(): a zero or non-normal squared length is a refusal, not a direction.
  const direction = (v) => {
    const s = dot(v, v);
    if (!(s >= 2 ** -1022) || !isFinite(s)) return null;
    const n = Math.sqrt(s);
    return [v[0] / n, v[1] / n, v[2] / n];
  };
  // triangle(): Van Oosterom–Strackee solid angle and its error, refusing near the atan2 branch cut.
  function triangle(a, b, c) {
    const num = dot(a, cross(b, c)), den = 1 + dot(a, b) + dot(a, c) + dot(b, c), rho = Math.hypot(num, den);
    if ((Math.abs(num) <= K && den <= K) || rho <= 2 * K) return null;
    return [2 * Math.atan2(num, den), 2 * (Math.SQRT2 * K / (rho - Math.SQRT2 * K) + 8 * U)];
  }
  // gauss_sum(a, b, false): { value, bound } or null for a refused (intersecting) pair.
  function linkingNumber(A, B) {
    let sum = 0, abs = 0, err = 0, terms = 0;
    for (let i = 0; i < A.length; i++) {
      const p1 = A[i], p2 = A[(i + 1) % A.length];
      for (let j = 0; j < B.length; j++) {
        const p3 = B[j], p4 = B[(j + 1) % B.length];
        const r13 = direction(sub(p3, p1)), r14 = direction(sub(p4, p1));
        const r24 = direction(sub(p4, p2)), r23 = direction(sub(p3, p2));
        if (!r13 || !r14 || !r24 || !r23) return null;
        const t1 = triangle(r13, r14, r24), t2 = triangle(r13, r24, r23);
        if (!t1 || !t2) return null;
        const w = t1[0] + t2[0];
        sum += -w; abs += Math.abs(w); err += t1[1] + t2[1] + U * Math.abs(w); terms++;
      }
    }
    const nu = terms * U, value = sum / (4 * Math.PI);
    return { value, bound: (err + (nu / (1 - nu)) * abs) / (4 * Math.PI) + 2 * U * Math.abs(value) };
  }
  // GaussLinking::certify.
  function certify(g) {
    const n = Math.round(g.value);
    if (Math.abs(g.value - n) + g.bound < 0.5) return n === 0 ? ["ZeroLinking", 0] : [`Linked { lk: ${n} }`, n];
    return ["Undetermined", null];
  }
  // tests/linking.rs twisted_band(n_twist, r = 1, a, m): the (2, 2n) torus link.
  function twistedBand(nt, m, a = 0.3) {
    const A = [], B = [];
    for (let k = 0; k < m; k++) {
      const t = (TAU * k) / m, c = Math.cos(t), s = Math.sin(t), ct = Math.cos(nt * t), zt = Math.sin(nt * t);
      const u = [c * ct, s * ct, zt];
      A.push([c + a * u[0], s + a * u[1], a * u[2]]);
      B.push([c - a * u[0], s - a * u[1], -a * u[2]]);
    }
    return [A, B];
  }

  // ---------- int-link ----------
  TSViz.register("int-link", (stage, api) => {
    let twists = 1, m = 96, reversed = false, phi = 0.6, A = [], B = [], result = null;
    const tilt = 1.05;
    // Orthographic view: rotate about z by phi, then tilt about x.
    const view = (p) => {
      const x = Math.cos(phi) * p[0] - Math.sin(phi) * p[1], y = Math.sin(phi) * p[0] + Math.cos(phi) * p[1];
      return [x, Math.cos(tilt) * y - Math.sin(tilt) * p[2], Math.sin(tilt) * y + Math.cos(tilt) * p[2]];
    };
    const draw = (ctx, w, h) => {
      const th = api.theme(), s = Math.min(w, h) * 0.34, cx = w / 2, cy = h / 2;
      const segs = [];
      [[A, th.accent], [B, th.accent2]].forEach(([C, col]) => C.forEach((p, i) => {
        const a = view(p), b = view(C[(i + 1) % C.length]);
        segs.push({ a, b, z: (a[2] + b[2]) / 2, col });
      }));
      // Painter's order with a ground-coloured halo, so every crossing reads as over or under.
      segs.sort((u, v) => u.z - v.z);
      ctx.lineCap = "round";
      for (const g of segs) {
        const x0 = cx + s * g.a[0], y0 = cy + s * g.a[1], x1 = cx + s * g.b[0], y1 = cy + s * g.b[1];
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
        ctx.strokeStyle = th.ground; ctx.lineWidth = 7; ctx.stroke();
        ctx.strokeStyle = g.col; ctx.lineWidth = 2.6; ctx.stroke();
      }
    };
    const c = api.canvas(260, draw);
    const paint = () => { c.ctx.clearRect(0, 0, c.w, c.h); draw(c.ctx, c.w, c.h); };
    api.onTheme(paint);

    const read = document.createElement("div");
    read.className = "int-read"; read.setAttribute("aria-live", "polite");
    queueMicrotask(() => stage.append(read));
    const fmt = (x) => (x === 0 ? "0" : Math.abs(x) < 1e-3 ? x.toExponential(2) : x.toFixed(15));
    function compute() {
      [A, B] = twistedBand(twists, m);
      if (reversed) B.reverse();
      result = linkingNumber(A, B);
      if (!result) { read.innerHTML = `<b>refused</b>: two segments meet within rounding (LinkingError::Intersecting)`; paint(); return; }
      const [verdict] = certify(result);
      read.innerHTML = `(2, ${2 * twists}) torus link, ${m} vertices per curve, B ${reversed ? "reversed" : "as generated"}<br>` +
        `Lk&#770; = <b>${fmt(result.value)}</b> &nbsp; B = <b>${fmt(result.bound)}</b><br>` +
        `|Lk&#770; &minus; n| + B &lt; 1/2 &rArr; <b>${verdict}</b>`;
      paint();
    }
    const ctl = api.controls();
    ctl.slider("twists", 0, 4, 1, twists, (v) => { twists = v; compute(); });
    ctl.slider("vertices", 12, 192, 12, m, (v) => { m = v; compute(); });
    const rev = ctl.button("Reverse B", () => { reversed = !reversed; rev.setAttribute("aria-pressed", reversed); compute(); });
    rev.setAttribute("aria-pressed", reversed);
    api.loop((t) => { phi = 0.6 + t * 0.25; paint(); });
  });
})();
