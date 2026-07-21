"""B1 summary: guided-operator ablation (generic / critical / full).

Reports, per cell and pooled, the mean objective and the gap to the
per-instance best-known solution for each operator pool, plus paired Wilcoxon
tests for the three contrasts:

  - critical vs generic : value of the makespan-critical guided moves (g1,g2)
  - full     vs critical: value of the tardiness-guided moves (g3,g4)
  - full     vs generic : total guided contribution

The tardiness-guided contrast is also split by time-window presence, since
g3/g4 fall back to g1/g2 when no truck is late (so they can only help on TW
cells). The point of the ablation is to show the guided pool changes quality by
much more than the selection policy did (<=0.17%p in K1).

Usage:
    python experiments/b1_summary.py
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.k1_stats import wilcoxon_signed_rank

RESULTS = ROOT / "outputs" / "b1_guided_ablation.jsonl"
POOLS = ("generic", "critical", "full")
METHOD = {pool: f"GILS-{pool}-1000" for pool in POOLS}


def load() -> dict:
    by_run: dict[tuple, float] = {}
    for line in RESULTS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        obj = r.get("objective", r.get("makespan"))
        if obj is None:
            continue
        key = (r["method"], r["size_class"], r["tw_tightness"], r["index"], r["rep"])
        by_run[key] = obj
    return by_run


def paired_diffs(by_run, pool_a, pool_b, *, tw_filter=None) -> list[float]:
    """Per-(instance, rep) relative diff (b - a)/a * 100 for two pools."""

    method_a, method_b = METHOD[pool_a], METHOD[pool_b]
    diffs = []
    for (method, size, tw, index, rep), value_a in by_run.items():
        if method != method_a:
            continue
        if tw_filter is not None and tw not in tw_filter:
            continue
        key_b = (method_b, size, tw, index, rep)
        if key_b in by_run:
            diffs.append(100.0 * (by_run[key_b] - value_a) / value_a)
    return diffs


def main() -> None:
    by_run = load()

    # Per-instance best-known across the three pools (all reps).
    best_known: dict[tuple, float] = {}
    for (method, size, tw, index, rep), value in by_run.items():
        key = (size, tw, index)
        if key not in best_known or value < best_known[key]:
            best_known[key] = value

    # Mean gap to best-known, per cell per pool.
    cell_gap: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (method, size, tw, index, rep), value in by_run.items():
        pool = next((p for p, m in METHOD.items() if m == method), None)
        if pool is None:
            continue
        base = best_known[(size, tw, index)]
        cell_gap[(size, tw)][pool].append(100.0 * (value - base) / base)

    print("=== Mean gap%% to per-instance best-known, by cell and pool ===")
    print(f"{'cell':<12}{'generic':>10}{'critical':>10}{'full':>10}")
    for (size, tw) in sorted(cell_gap, key=str):
        row = cell_gap[(size, tw)]
        cells = "".join(
            f"{statistics.mean(row[p]):>10.2f}" if row[p] else f"{'—':>10}"
            for p in POOLS
        )
        label = f"{size}-{tw or 'none'}"
        print(f"{label:<12}{cells}")

    print()
    print("=== Paired Wilcoxon (two-sided); mean%% = improvement from adding ops ===")
    print("(pool_b -> pool_a where pool_a is the richer pool; +mean%% = richer is better)")
    print(f"{'contrast':<34}{'scope':<10}{'n':>5}{'mean%':>9}{'p':>10}{'verdict':>14}")

    tw_all = None
    tw_none = {None: True}
    tw_windows = {"medium": True, "tight": True}
    contrasts = [
        ("critical", "generic", tw_all, "all"),
        ("full", "critical", tw_all, "all"),
        ("full", "critical", tw_windows, "TW-only"),
        ("full", "critical", tw_none, "none-only"),
        ("full", "generic", tw_all, "all"),
    ]
    for pool_a, pool_b, tw_filter, scope in contrasts:
        diffs = paired_diffs(by_run, pool_a, pool_b, tw_filter=tw_filter)
        if not diffs:
            continue
        p, n = wilcoxon_signed_rank(diffs)
        mean_diff = statistics.mean(diffs)
        verdict = "significant" if p < 0.05 else "no difference"
        label = f"{pool_b} -> {pool_a}"
        print(f"{label:<34}{scope:<10}{n:>5}{mean_diff:>+9.3f}{p:>10.4f}{verdict:>14}")

    print()
    print("Reading: mean%% = 100*(objective_pool_b - objective_pool_a)/pool_a, so a")
    print("positive value means the richer pool (pool_a: adds the guided operators)")
    print("has the lower objective, i.e., the guided operators help.")


if __name__ == "__main__":
    main()
