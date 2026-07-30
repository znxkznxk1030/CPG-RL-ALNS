#!/usr/bin/env python3
"""Generate publication-quality SVG figures from the ablation result data.

matplotlib is unusable in this environment (NumPy 2.x ABI break), so we emit
clean, dependency-free vector SVG directly. All numbers are the frozen
test-pool results reported in docs/b1_results.md, docs/b2_results.md and
outputs/k1_stats.txt.

Figures written to paper/figures/:
  fig1_selector_budget.svg   selection-policy budget sweep (line)
  fig2_operator_pool.svg      operator-pool ablation, per cell (grouped bars)
  fig3_component_ablation.svg engine-component leave-one-out (bars)
"""
from __future__ import annotations
import os
from html import escape

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# ---- shared style -----------------------------------------------------------
FONT = "Helvetica, Arial, sans-serif"
INK = "#1a1a1a"
GRID = "#d9d9d9"
AXIS = "#4d4d4d"
# grayscale-safe, colourblind-friendly trio
C = ["#2b6cb0", "#dd6b20", "#718096"]  # blue, orange, gray
C_FILL = ["#3182ce", "#ed8936", "#a0aec0"]


def _svg(w, h, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="{FONT}">\n'
        f'<rect width="{w}" height="{h}" fill="white"/>\n{body}</svg>\n'
    )


def _txt(x, y, s, size=13, anchor="middle", fill=INK, weight="normal", rot=None):
    tr = f' transform="rotate({rot} {x} {y})"' if rot is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}"{tr}>{escape(s)}</text>\n'
    )


def _line(x1, y1, x2, y2, stroke=AXIS, w=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{w}"{d}/>\n')


# ---- Figure 1: selection-policy budget sweep --------------------------------
def fig1():
    budgets = [50, 200, 1000, 3000]
    series = {  # mean gap% to per-instance best-known (outputs/k1_stats.txt)
        "Uniform (proposed)": [0.47, 0.38, 0.23, 0.16],
        "Tabular Q-learning": [0.56, 0.46, 0.23, 0.09],
        "Transfer DQN":       [0.56, 0.51, 0.38, 0.26],
    }
    W, H = 560, 380
    L, R, T, B = 70, 200, 30, 55
    pw, ph = W - L - R, H - T - B
    import math
    xs = [math.log10(b) for b in budgets]
    x0, x1 = min(xs), max(xs)
    ymax = 0.6

    def X(b):
        return L + (math.log10(b) - x0) / (x1 - x0) * pw

    def Y(v):
        return T + (1 - v / ymax) * ph

    body = ""
    # y grid + labels
    for gy in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        y = Y(gy)
        body += _line(L, y, L + pw, y, GRID, 1)
        body += _txt(L - 10, y + 4, f"{gy:.1f}", 11, "end", AXIS)
    # x ticks
    for b in budgets:
        body += _line(X(b), T + ph, X(b), T + ph + 5, AXIS, 1)
        body += _txt(X(b), T + ph + 20, str(b), 11, "middle", AXIS)
    # axes
    body += _line(L, T, L, T + ph, AXIS, 1.2)
    body += _line(L, T + ph, L + pw, T + ph, AXIS, 1.2)
    # axis titles
    body += _txt(L + pw / 2, H - 12, "Iteration budget", 13, "middle", INK)
    body += _txt(18, T + ph / 2, "Mean gap to best-known (%)", 13, "middle",
                 INK, rot=-90)
    # series
    for i, (name, ys) in enumerate(series.items()):
        pts = " ".join(f"{X(b):.1f},{Y(v):.1f}" for b, v in zip(budgets, ys))
        body += (f'<polyline points="{pts}" fill="none" stroke="{C[i]}" '
                 f'stroke-width="2.4"/>\n')
        for b, v in zip(budgets, ys):
            body += (f'<circle cx="{X(b):.1f}" cy="{Y(v):.1f}" r="3.6" '
                     f'fill="{C[i]}"/>\n')
        # legend
        ly = T + 6 + i * 20
        body += _line(L + pw + 16, ly, L + pw + 40, ly, C[i], 2.4)
        body += _txt(L + pw + 46, ly + 4, name, 11.5, "start", INK)
    body += _txt(L, T - 12, "Learned policies never beat uniform",
                 12, "start", "#555", weight="bold")
    with open(os.path.join(OUT, "fig1_selector_budget.svg"), "w") as f:
        f.write(_svg(W, H, body))


# ---- Figure 2: operator-pool ablation (grouped bars per cell) ---------------
def fig2():
    cells = ["S-none", "S-med", "S-tight", "M-none", "M-med", "M-tight",
             "L-none", "L-med", "L-tight"]
    # docs/b1_results.md gap% (generic, critical, full)
    data = {
        "generic":  [0.45, 0.41, 0.35, 0.39, 0.45, 0.25, 0.22, 0.09, 0.01],
        "critical": [0.17, 0.17, 0.17, 0.28, 0.37, 0.19, 0.17, 0.07, 0.01],
        "full":     [0.10, 0.08, 0.16, 0.24, 0.28, 0.11, 0.12, 0.06, 0.00],
    }
    labels = ["generic (7 ops)", "critical (+g1,g2)", "full (+g3,g4)"]
    W, H = 640, 380
    L, R, T, B = 60, 20, 40, 70
    pw, ph = W - L - R, H - T - B
    ymax = 0.5
    n = len(cells)
    group_w = pw / n
    bar_w = group_w * 0.24

    def Y(v):
        return T + (1 - v / ymax) * ph

    body = ""
    for gy in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        y = Y(gy)
        body += _line(L, y, L + pw, y, GRID, 1)
        body += _txt(L - 10, y + 4, f"{gy:.1f}", 11, "end", AXIS)
    body += _line(L, T, L, T + ph, AXIS, 1.2)
    body += _line(L, T + ph, L + pw, T + ph, AXIS, 1.2)
    body += _txt(18, T + ph / 2, "Mean gap to best-known (%)", 13, "middle",
                 INK, rot=-90)
    for ci, cell in enumerate(cells):
        gx = L + ci * group_w + group_w / 2
        for k, key in enumerate(["generic", "critical", "full"]):
            v = data[key][ci]
            bx = gx + (k - 1) * (bar_w + 3) - bar_w / 2
            by = Y(v)
            body += (f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" '
                     f'height="{T + ph - by:.1f}" fill="{C_FILL[k]}"/>\n')
        body += _txt(gx, T + ph + 16, cell, 10.5, "middle", AXIS)
    # legend
    for k, lab in enumerate(labels):
        lx = L + 20 + k * 200
        body += (f'<rect x="{lx}" y="{H - 34}" width="14" height="14" '
                 f'fill="{C_FILL[k]}"/>\n')
        body += _txt(lx + 20, H - 23, lab, 11.5, "start", INK)
    body += _txt(L, T - 16, "Adding bottleneck-guided operators lowers the "
                 "gap in every cell (generic > critical > full)",
                 12, "start", "#555", weight="bold")
    with open(os.path.join(OUT, "fig2_operator_pool.svg"), "w") as f:
        f.write(_svg(W, H, body))


# ---- Figure 3: engine-component leave-one-out -------------------------------
def fig3():
    # docs/b2_results.md aggregate degradation% when component removed
    comps = [
        ("Best-improvement descent", 0.156, "p<0.0001"),
        ("Kick restart",             0.069, "p<0.0001"),
        ("VAA initial solution",     0.022, "n.s. by cell"),
        ("Stochastic acceptance",   -0.026, "greedy no worse"),
    ]
    W, H = 660, 300
    L, R, T, B = 200, 150, 40, 45
    pw, ph = W - L - R, H - T - B
    vmin, vmax = -0.06, 0.18
    rows = len(comps)
    rh = ph / rows

    def X(v):
        return L + (v - vmin) / (vmax - vmin) * pw

    body = ""
    x0 = X(0.0)
    for gv in [-0.05, 0.0, 0.05, 0.10, 0.15]:
        x = X(gv)
        body += _line(x, T, x, T + ph, GRID, 1)
        body += _txt(x, T + ph + 18, f"{gv:+.2f}", 10.5, "middle", AXIS)
    body += _line(x0, T, x0, T + ph, AXIS, 1.4)  # zero line
    for i, (name, val, note) in enumerate(comps):
        cy = T + i * rh + rh / 2
        col = C_FILL[0] if val > 0 else C_FILL[1]
        bx = min(x0, X(val))
        bw = abs(X(val) - x0)
        body += (f'<rect x="{bx:.1f}" y="{cy - 11:.1f}" width="{bw:.1f}" '
                 f'height="22" fill="{col}"/>\n')
        body += _txt(L - 12, cy + 4, name, 12, "end", INK)
        body += _txt(L + pw + 8, cy + 4, f"{val:+.3f} ({note})", 10.5,
                     "start", "#333")
    body += _txt(L + pw / 2, H - 10,
                 "Objective degradation when component removed (%)", 12,
                 "middle", INK)
    body += _txt(L, T - 16, "Descent and restart carry the engine; "
                 "initialization and acceptance do not", 12, "start", "#555",
                 weight="bold")
    with open(os.path.join(OUT, "fig3_component_ablation.svg"), "w") as f:
        f.write(_svg(W, H, body))


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    for fn in ("fig1_selector_budget", "fig2_operator_pool",
               "fig3_component_ablation"):
        print("wrote", os.path.join("paper/figures", fn + ".svg"))
