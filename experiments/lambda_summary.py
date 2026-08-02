"""Summarise the lambda-sensitivity sweep into a makespan-tardiness trade-off.

Reads outputs/lambda_sensitivity.jsonl and produces:
  - a scale-invariant trade-off curve: per (instance, rep) each metric is
    min-max normalised across the lambda grid, then averaged per lambda, so all
    sizes contribute on a common [0, 1] scale;
  - per-size raw mean makespan and total tardiness tables (absolute magnitudes);
  - outputs/lambda_curve.json for the figure generator (experiments/make_figures.py).

Run:
    python experiments/lambda_summary.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "outputs" / "lambda_sensitivity.jsonl"
CURVE_JSON = ROOT / "outputs" / "lambda_curve.json"
SUMMARY_TXT = ROOT / "outputs" / "lambda_summary.txt"


def _load():
    with IN.open() as f:
        return [json.loads(line) for line in f]


def _norm(value, lo, hi):
    return 0.0 if hi <= lo else (value - lo) / (hi - lo)


def main() -> None:
    recs = _load()
    lambdas = sorted({r["lam"] for r in recs})

    # group by (cell, index, rep) -> {lam: (makespan, tardiness)}
    groups: dict[tuple, dict[float, tuple[float, float]]] = defaultdict(dict)
    for r in recs:
        key = (r["cell"], r["index"], r["rep"])
        groups[key][r["lam"]] = (r["makespan"], r["total_tardiness"])

    # scale-invariant normalised curve
    norm_ms: dict[float, list[float]] = {lam: [] for lam in lambdas}
    norm_td: dict[float, list[float]] = {lam: [] for lam in lambdas}
    for key, by_lam in groups.items():
        ms = {lam: by_lam[lam][0] for lam in lambdas if lam in by_lam}
        td = {lam: by_lam[lam][1] for lam in lambdas if lam in by_lam}
        if len(ms) < len(lambdas):
            continue
        m_lo, m_hi = min(ms.values()), max(ms.values())
        t_lo, t_hi = min(td.values()), max(td.values())
        for lam in lambdas:
            norm_ms[lam].append(_norm(ms[lam], m_lo, m_hi))
            norm_td[lam].append(_norm(td[lam], t_lo, t_hi))

    curve = [
        {
            "lam": lam,
            "makespan_norm": statistics.mean(norm_ms[lam]),
            "tardiness_norm": statistics.mean(norm_td[lam]),
        }
        for lam in lambdas
    ]

    # per-size raw means
    per_size: dict[str, dict[float, dict[str, float]]] = defaultdict(dict)
    by_size_lam: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: {"ms": [], "td": []}
    )
    for r in recs:
        b = by_size_lam[(r["size"], r["lam"])]
        b["ms"].append(r["makespan"])
        b["td"].append(r["total_tardiness"])
    for (size, lam), b in by_size_lam.items():
        per_size[size][lam] = {
            "makespan": statistics.mean(b["ms"]),
            "tardiness": statistics.mean(b["td"]),
        }

    CURVE_JSON.write_text(json.dumps({"curve": curve, "lambdas": lambdas}, indent=1))

    lines = []
    lines.append("=== Scale-invariant makespan-tardiness trade-off (mean over all cells) ===")
    lines.append(f"{'lambda':>8}{'makespan_norm':>16}{'tardiness_norm':>16}")
    for pt in curve:
        lines.append(
            f"{pt['lam']:>8}{pt['makespan_norm']:>16.3f}{pt['tardiness_norm']:>16.3f}"
        )
    lines.append("  (0 = best/min over the lambda grid per instance, 1 = worst/max)")
    lines.append("")
    for size in ("S", "M", "L"):
        lines.append(f"=== {size}: raw mean makespan / total tardiness by lambda ===")
        lines.append(f"{'lambda':>8}{'makespan':>14}{'tardiness':>14}")
        for lam in lambdas:
            d = per_size[size][lam]
            lines.append(f"{lam:>8}{d['makespan']:>14.1f}{d['tardiness']:>14.1f}")
        lines.append("")
    text = "\n".join(lines)
    SUMMARY_TXT.write_text(text)
    print(text)
    print(f"wrote {CURVE_JSON}\nwrote {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
