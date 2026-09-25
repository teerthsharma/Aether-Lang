// Shared runtime for page visualisations. A page embeds
//   <div class="ts-viz" data-viz="name" data-title="..." data-caption="..."></div>
// and a viz-<area>.js file calls TSViz.register("name", mount).
// mount(stage, api) runs once, when the block first scrolls near the viewport.
(() => {
  const registry = new Map();
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const css = (n) => getComputedStyle(document.body).getPropertyValue(n).trim();
  const theme = () => {
    const dark = document.body.getAttribute("data-md-color-scheme") === "slate";
    return {
      dark,
      ink: dark ? "#ecebe6" : "#1c1b19",
      muted: dark ? "#a5a198" : "#5f5b53",
      hair: dark ? "#3a3833" : "#cfcbc1",
      ground: dark ? "#1f1e1b" : "#ffffff",
      accent: dark ? "#8fb0ff" : "#2456dc",
      accent2: dark ? "#ffb86b" : "#c2410c",   // second series / "rejected"
      good: dark ? "#6fd39a" : "#15803d",
      world: dark ? "#1b2a3c" : "#bcd6ee",
    };
  };
  const listeners = new Set();
  new MutationObserver(() => listeners.forEach((f) => f(theme())))
    .observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });

  // Canvas sized to its box at devicePixelRatio; calls draw(ctx, w, h) on resize.
  function canvas(stage, height, draw) {
    const cv = document.createElement("canvas");
    cv.style.height = height + "px";
    stage.append(cv);
    const ctx = cv.getContext("2d");
    let w = 0;
    const fit = () => {
      const dpr = Math.min(devicePixelRatio || 1, 2);
      w = stage.clientWidth;
      cv.width = w * dpr; cv.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw && draw(ctx, w, height);
    };
    new ResizeObserver(fit).observe(stage);
    fit();
    return { cv, ctx, get w() { return w; }, h: height, redraw: fit };
  }

  // requestAnimationFrame loop that pauses offscreen; one static frame under reduced motion.
  function loop(el, tick) {
    let on = false, id = 0, t0 = performance.now();
    const step = (t) => { tick((t - t0) / 1000); if (on && !still) id = requestAnimationFrame(step); };
    new IntersectionObserver(([e]) => {
      on = e.isIntersecting;
      cancelAnimationFrame(id);
      if (on) id = requestAnimationFrame(step);
    }).observe(el);
  }

  // Labelled slider / button controls under the stage.
  function controls(box) {
    const bar = document.createElement("div");
    bar.className = "ts-viz-controls";
    box.append(bar);
    return {
      slider(label, min, max, step, value, onInput) {
        const l = document.createElement("label");
        const out = document.createElement("output");
        const i = Object.assign(document.createElement("input"), { type: "range", min, max, step, value });
        const set = () => { out.textContent = (+i.value).toFixed(step < 1 ? 2 : 0); onInput(+i.value); };
        i.addEventListener("input", set);
        l.append(label + " ", i, out); bar.append(l); set();
        return i;
      },
      button(label, onClick) {
        const b = Object.assign(document.createElement("button"), { type: "button", textContent: label });
        b.addEventListener("click", onClick); bar.append(b); return b;
      },
    };
  }

  function mount(box) {
    const fn = registry.get(box.dataset.viz);
    if (!fn || box.dataset.mounted) return;
    box.dataset.mounted = "1";
    if (box.dataset.title) {
      const h = document.createElement("div"); h.className = "ts-viz-title"; h.textContent = box.dataset.title; box.prepend(h);
    }
    const stage = document.createElement("div"); stage.className = "ts-viz-stage"; box.append(stage);
    const api = { theme, onTheme: (f) => listeners.add(f), still, canvas: (h, d) => canvas(stage, h, d),
                  loop: (tick) => loop(stage, tick), controls: () => controls(box), css };
    try { fn(stage, api); } catch (e) { stage.textContent = "Visualisation failed to load."; console.error(e); }
    if (box.dataset.caption) {
      const c = document.createElement("p"); c.className = "ts-viz-caption"; c.textContent = box.dataset.caption; box.append(c);
    }
  }

  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) { io.unobserve(e.target); mount(e.target); }
  }), { rootMargin: "200px" });
  const scan = () => document.querySelectorAll(".ts-viz:not([data-mounted])").forEach((b) => io.observe(b));

  window.TSViz = { register(name, fn) { registry.set(name, fn); scan(); }, theme };
  document.addEventListener("DOMContentLoaded", scan);
})();
