"""Draw the README figures from numbers Aether itself prints.

    python scripts/make_readme_assets.py

Runs the `aether` CLI on the opening program of the README (the shape half of
examples/tour.aegis, plus one line printing the persistence intervals), and
draws three SVGs into assets/:

  hero.svg       the delay-embedded signal, its barcode, and the Betti vector
  filtration.svg the same cloud at four radii, each labelled with the Betti
                 numbers Aether reports there
  refuse.svg     the tour's threshold loop: three refusals, then a proof

The cloud is recomputed here from the same formula `manifold.rs` uses
(x_t, x_{t-2}, x_{t-4}); every bar, Betti number, radius and verdict is parsed
from the CLI's output, never typed in.
"""

import json
import math
import os
import pathlib
import subprocess
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

BG, PANEL, TEXT, MUTED = "#0d1117", "#161b22", "#e6edf3", "#8b949e"
BLUE, AMBER, GREEN, RED = "#58a6ff", "#d29922", "#3fb950", "#f85149"
SANS = "system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

SHAPE_PROGRAM = """import topology~
import math~
let signal = []~
let t = 0~
while t < 18 {
    signal.push(sin(t * 0.7))~
    t = t + 1~
}
manifold M = embed(signal, dim=3, tau=2)~
let shape = topology.ph(M, max_dim=1, mode="vr")~
print(topology.intervals(shape))~
"""


def aether(path):
    exe = ROOT / "target" / "debug" / ("aether.exe" if os.name == "nt" else "aether")
    cmd = [str(exe), "run", str(path)] if exe.exists() else [
        "cargo", "run", "-q", "-p", "aether-cli", "--", "run", str(path)]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return [line for line in out.splitlines() if line.startswith("[")]


def text(x, y, s, size, fill=TEXT, anchor="start", family=SANS, weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-family="{family}" font-weight="{weight}">{s}</text>')


def cloud():
    x = [math.sin(0.7 * t) for t in range(18)]
    pts = np.array([[x[t], x[t - 2], x[t - 4]] for t in range(4, 18)])
    # embed() needs D*tau = 6 samples before its first point: t = 5..17.
    pts = pts[1:]
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return pts, centred @ vt[:2].T


def betti(intervals, r):
    b = [0, 0, 0]
    for dim, birth, death in intervals:
        if birth <= r and (death == -1 or r < death):
            b[int(dim)] += 1
    return b


def draw_cloud(pts3, pts2, r, x0, y0, w, h):
    lo, hi = pts2.min(axis=0), pts2.max(axis=0)
    scale = min(w / (hi[0] - lo[0]), h / (hi[1] - lo[1])) * 0.82
    cx, cy = x0 + w / 2, y0 + h / 2
    mid = (lo + hi) / 2
    xy = [(cx + (p[0] - mid[0]) * scale, cy - (p[1] - mid[1]) * scale) for p in pts2]
    out = []
    n = len(pts3)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pts3[i] - pts3[j]) <= r:
                (a, b), (c, d) = xy[i], xy[j]
                out.append(f'<line x1="{a:.1f}" y1="{b:.1f}" x2="{c:.1f}" y2="{d:.1f}" '
                           f'stroke="{BLUE}" stroke-opacity="0.45" stroke-width="2"/>')
    for a, b in xy:
        out.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="6" fill="{BLUE}"/>')
    return out


def hero(intervals, pts3, pts2):
    W, H = 1600, 480
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f'<rect width="{W}" height="{H}" fill="{BG}"/>',
         text(56, 44, "sin(0.7 t), 18 samples, delay-embedded: dim 3, tau 2", 22, MUTED, family=MONO),
         text(W - 56, 44, "exact F2 persistence, Vietoris-Rips", 22, MUTED, "end", MONO)]
    s += draw_cloud(pts3, pts2, 1.0, 40, 70, 460, 380)
    s.append(text(270, 452, "the cloud at radius 1", 20, MUTED, "middle", MONO))

    # Barcode: every bar of positive length, H0 blue then H1 amber.
    bars = [(d, b, e) for d, b, e in intervals if e == -1 or e - b > 1e-9]
    bars.sort(key=lambda z: (z[0], z[1], z[2] if z[2] != -1 else 9e9))
    x0, x1, top, rmax = 560, 1080, 100, 2.4
    sx = lambda r: x0 + (x1 - x0) * min(r, rmax) / rmax
    step = 300 / len(bars)
    for k, (dim, b, e) in enumerate(bars):
        y = top + k * step
        end = rmax if e == -1 else e
        colour = BLUE if dim == 0 else AMBER
        s.append(f'<rect x="{sx(b):.1f}" y="{y:.1f}" width="{max(sx(end) - sx(b), 2):.1f}" '
                 f'height="{step * 0.62:.1f}" rx="3" fill="{colour}"/>')
        if e == -1:
            s.append(text(sx(rmax) + 6, y + step * 0.6, "&#8734;", 20, MUTED))
    s.append(f'<line x1="{sx(1.0):.1f}" y1="{top - 10}" x2="{sx(1.0):.1f}" y2="{top + 310}" '
             f'stroke="{GREEN}" stroke-width="2" stroke-dasharray="6 6"/>')
    for r in (0, 1, 2):
        s.append(text(sx(r), top + 336, str(r), 18, MUTED, "middle", MONO))
    s.append(text(x1 + 44, top + 336, "r", 18, MUTED, "middle", MONO))
    s.append(text(sx(1.0), 452, "sealed at r = 1", 22, GREEN, "middle", MONO, 600))
    s.append(text(x0, 82, "H0 pieces", 20, BLUE, family=MONO))
    s.append(text(x0 + 140, 82, "H1 loops", 20, AMBER, family=MONO))

    b = betti(intervals, 1.0)
    s.append(text(1340, 210, f"&#946; = [{b[0]}, {b[1]}, {b[2]}]", 64, TEXT, "middle", weight=600))
    s.append(text(1340, 262, "one piece, one loop", 30, MUTED, "middle"))
    s.append(text(1340, 330, "nobody asked whether", 24, MUTED, "middle"))
    s.append(text(1340, 362, "the signal was periodic", 24, MUTED, "middle"))
    s.append(text(1340, 452, "examples/tour.aegis", 20, MUTED, "middle", MONO))
    s.append("</svg>")
    return "\n".join(s)


def filtration(intervals, pts3, pts2):
    W, H = 1600, 420
    radii = [0.5, 0.8, 1.0, 2.1]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f'<rect width="{W}" height="{H}" fill="{BG}"/>']
    for k, r in enumerate(radii):
        x0 = 20 + k * 395
        s.append(f'<rect x="{x0}" y="20" width="375" height="380" rx="10" fill="{PANEL}"/>')
        s += draw_cloud(pts3, pts2, r, x0, 40, 375, 270)
        b = betti(intervals, r)
        s.append(text(x0 + 187, 346, f"r = {r}", 24, MUTED, "middle", MONO))
        colour = GREEN if b[:2] == [1, 1] else TEXT
        s.append(text(x0 + 187, 382, f"&#946; = [{b[0]}, {b[1]}, {b[2]}]", 28, colour, "middle", MONO, 600))
    s.append("</svg>")
    return "\n".join(s)


def refuse(rows):
    W, H = 1600, 360
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f'<rect width="{W}" height="{H}" fill="{BG}"/>',
         text(56, 44, "is 1.0 below 1.5, when the score may be off by r?", 24, MUTED, family=MONO)]
    x0, x1, lo, hi = 360, 1500, -1.2, 3.2
    sx = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)
    s.append(f'<line x1="{sx(1.5):.1f}" y1="70" x2="{sx(1.5):.1f}" y2="{H - 30}" stroke="{TEXT}" stroke-width="2"/>')
    s.append(text(sx(1.5) + 8, 84, "1.5", 20, TEXT, family=MONO))
    for k, (radius, verdict) in enumerate(rows):
        y = 110 + k * 62
        ok = verdict == "below"
        colour = GREEN if ok else RED
        s.append(text(56, y + 8, f"r = {radius:g}", 24, TEXT, family=MONO))
        s.append(text(200, y + 8, verdict, 24, colour, family=MONO, weight=600))
        s.append(f'<rect x="{sx(1 - radius):.1f}" y="{y - 14}" width="{sx(1 + radius) - sx(1 - radius):.1f}" '
                 f'height="26" rx="13" fill="{colour}" fill-opacity="0.28" stroke="{colour}" stroke-width="2"/>')
        s.append(f'<circle cx="{sx(1.0):.1f}" cy="{y - 1}" r="7" fill="{TEXT}"/>')
    s.append("</svg>")
    return "\n".join(s)


def main():
    ASSETS.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".aegis", delete=False, encoding="utf-8") as f:
        f.write(SHAPE_PROGRAM)
    try:
        intervals = json.loads(aether(f.name)[-1])
    finally:
        os.unlink(f.name)

    # The tour's refusal loop prints [radius, r, verdict, v] per pass and
    # [proven at radius, r] at the end.
    rows = []
    for line in aether(ROOT / "examples" / "tour.aegis"):
        parts = [p.strip() for p in line.strip("[]").split(",")]
        if parts[0] == "radius":
            rows.append((float(parts[1]), parts[3]))
        elif parts[0] == "proven at radius":
            rows.append((float(parts[1]), "below"))

    pts3, pts2 = cloud()
    (ASSETS / "hero.svg").write_text(hero(intervals, pts3, pts2), encoding="utf-8")
    (ASSETS / "filtration.svg").write_text(filtration(intervals, pts3, pts2), encoding="utf-8")
    (ASSETS / "refuse.svg").write_text(refuse(rows), encoding="utf-8")
    print("betti at 0.5, 0.8, 1.0, 2.1:", [betti(intervals, r) for r in (0.5, 0.8, 1.0, 2.1)])
    print("refusal rows:", rows)


if __name__ == "__main__":
    main()
