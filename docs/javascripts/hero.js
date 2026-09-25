// Hero scene: a drifting point cloud whose Rips radius breathes, and a seal on the ice.
(() => {
  const hero = document.querySelector(".ts-hero");
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Section reveal on every page.
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) { e.target.classList.add("ts-in"); io.unobserve(e.target); }
  }), { rootMargin: "0px 0px -8% 0px" });
  document.querySelectorAll(".md-content__inner > :not(.ts-hero)").forEach((el) => {
    el.classList.add("ts-reveal"); io.observe(el);
  });

  if (!hero) return;
  const cv = document.createElement("canvas");
  cv.className = "ts-scene"; cv.setAttribute("aria-hidden", "true");
  hero.prepend(cv);
  const seal = document.createElement("div");
  seal.className = "ts-seal"; seal.setAttribute("aria-hidden", "true");
  seal.innerHTML = `<svg viewBox="0 0 64 36" width="64" height="36"><path d="M6 28c0-9 9-16 22-16 7 0 11-5 17-5 6 0 9 4 9 8 0 3-2 5-5 6 5 1 9 4 9 7H6z" fill="#5b6675"/><path d="M12 28c3-4 10-6 18-6s14 2 18 6" fill="#8793a3"/><circle cx="48" cy="13" r="1.6" fill="#1c1b19"/><circle cx="53.5" cy="16" r="1" fill="#1c1b19"/><path d="M2 30c3-3 6-2 8-1" stroke="#5b6675" stroke-width="3" fill="none" stroke-linecap="round"/></svg>`;
  hero.append(seal);

  const ctx = cv.getContext("2d");
  let w = 0, h = 0, dpr = 1, pts = [];
  const size = () => {
    dpr = Math.min(devicePixelRatio || 1, 2);
    w = hero.clientWidth; h = hero.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.round(Math.min(70, w * h / 9000));
    pts = Array.from({ length: n }, () => ({
      x: Math.random() * w, y: Math.random() * h * 0.85,
      vx: (Math.random() - .5) * .18, vy: (Math.random() - .5) * .12,
    }));
  };
  new ResizeObserver(size).observe(hero);
  size();

  const dark = () => document.body.getAttribute("data-md-color-scheme") === "slate";
  const frame = (t) => {
    ctx.clearRect(0, 0, w, h);
    const ink = dark() ? "143,176,255" : "36,86,220";
    const r = 60 + 40 * Math.sin(t / 2600); // filtration radius breathing
    for (const p of pts) {
      if (!still) { p.x += p.vx; p.y += p.vy; }
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h * .85) p.vy *= -1;
    }
    // ponytail: O(n²) edge scan, n ≤ 70 so ~2.4k checks/frame; grid-bucket if n grows.
    for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
      const a = pts[i], b = pts[j], d = Math.hypot(a.x - b.x, a.y - b.y);
      if (d < r) {
        ctx.strokeStyle = `rgba(${ink},${.28 * (1 - d / r)})`;
        ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
    }
    ctx.fillStyle = `rgba(${ink},.55)`;
    for (const p of pts) { ctx.beginPath(); ctx.arc(p.x, p.y, 2, 0, 7); ctx.fill(); }
    if (!still) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
})();
