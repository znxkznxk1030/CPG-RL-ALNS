"""B2 summary: engine-component ablation (leave-one-out).

For each removed component, reports the relative objective degradation vs the
full engine ("none"), paired per (instance, rep), pooled and split by
time-window presence, with a Wilcoxon signed-rank test. A positive degradation
means removing the component made the objective worse, i.e., the component
contributes.

Usage:
    python experiments/b2_summary.py
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

RESULTS = ROOT / "outputs" / "b2_component_ablation.jsonl"
DROPS = ("init", "descent", "sa", "restart")
REFERENCE = "GILS-ablate-none-1000"
METHOD = {drop: f"GILS-ablate-{drop}-1000" for drop in (*DROPS, "none")}


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


def degradation(by_run, drop, *, tw_filter=None) -> list[float]:
    """Per-(instance, rep) relative degradation (drop - none)/none * 100."""

    method = METHOD[drop]
    diffs = []
    for (m, size, tw, index, rep), value in by_run.items():
        if m != REFERENCE:
            continue
        if tw_filter is not None and tw not in tw_filter:
            continue
        drop_value = by_run.get((method, size, tw, index, rep))
        if drop_value is not None:
            diffs.append(100.0 * (drop_value - value) / value)
    return diffs


def main() -> None:
    by_run = load()

    # Per-cell mean degradation for each dropped component.
    cell_deg: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (m, size, tw, index, rep), value in by_run.items():
        if m != REFERENCE:
            continue
        for drop in DROPS:
            dv = by_run.get((METHOD[drop], size, tw, index, rep))
            if dv is not None:
                cell_deg[(size, tw)][drop].append(100.0 * (dv - value) / value)

    print("=== Mean objective degradation%% when a component is removed, by cell ===")
    print(f"{'cell':<12}" + "".join(f"{d:>10}" for d in DROPS))
    for (size, tw) in sorted(cell_deg, key=str):
        row = cell_deg[(size, tw)]
        cells = "".join(
            f"{statistics.mean(row[d]):>10.2f}" if row[d] else f"{'—':>10}" for d in DROPS
        )
        print(f"{f'{size}-{tw or 'none'}':<12}{cells}")

    print()
    print("=== Paired Wilcoxon: degradation from removing each component ===")
    print(f"{'component removed':<20}{'scope':<10}{'n':>5}{'degrade%':>10}{'p':>10}{'verdict':>14}")
    tw_none = {None: True}
    tw_windows = {"medium": True, "tight": True}
    for drop in DROPS:
        for scope, tw_filter in (("all", None), ("none", tw_none), ("TW", tw_windows)):
            diffs = degradation(by_run, drop, tw_filter=tw_filter)
            if not diffs:
                continue
            p, n = wilcoxon_signed_rank(diffs)
            mean_diff = statistics.mean(diffs)
            verdict = "significant" if p < 0.05 else "no difference"
            print(f"{drop:<20}{scope:<10}{n:>5}{mean_diff:>+10.3f}{p:>10.4f}{verdict:>14}")

    print()
    print("Reading: degrade%% > 0 means removing the component worsens the objective,")
    print("so the component contributes. Larger = more important.")


if __name__ == "__main__":
    main()
