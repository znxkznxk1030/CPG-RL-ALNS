"""Statistical analysis for the APIEMS paper: Wilcoxon tests + budget curves.

scipy is unusable in this environment (NumPy-1.x-built binaries), so the
Wilcoxon signed-rank test is implemented here: exact null distribution for
n <= 25 without ties in |d|, otherwise normal approximation with tie
correction and continuity correction.

Usage:
    python experiments/k1_stats.py
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "outputs" / "k1_results.jsonl"


def wilcoxon_signed_rank(diffs: list[float]) -> tuple[float, int]:
    """Two-sided Wilcoxon signed-rank test. Returns (p_value, n_used)."""

    nonzero = [d for d in diffs if abs(d) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0, 0

    magnitudes = sorted((abs(d), i) for i, d in enumerate(nonzero))
    ranks = [0.0] * n
    i = 0
    has_ties = False
    while i < n:
        j = i
        while j + 1 < n and abs(magnitudes[j + 1][0] - magnitudes[i][0]) < 1e-12:
            j += 1
        if j > i:
            has_ties = True
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[magnitudes[k][1]] = average_rank
        i = j + 1

    w_plus = sum(rank for rank, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(rank for rank, d in zip(ranks, nonzero) if d < 0)
    t_stat = min(w_plus, w_minus)

    if n <= 25 and not has_ties:
        # Exact: distribution of rank-sum over all 2^n sign assignments.
        total = n * (n + 1) // 2
        counts = [0] * (total + 1)
        counts[0] = 1
        for rank in range(1, n + 1):
            for s in range(total, rank - 1, -1):
                counts[s] += counts[s - rank]
        threshold = int(round(t_stat))
        tail = sum(counts[: threshold + 1])
        p = min(1.0, 2.0 * tail / (2**n))
        return p, n

    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    # tie correction
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(magnitudes[j + 1][0] - magnitudes[i][0]) < 1e-12:
            j += 1
        t = j - i + 1
        if t > 1:
            variance -= (t**3 - t) / 48.0
        i = j + 1
    if variance <= 0:
        return 1.0, n
    z = (t_stat - mean + 0.5) / math.sqrt(variance)
    p = min(1.0, math.erfc(-z / math.sqrt(2.0)))  # 2 * Phi(z) for z < 0
    return p, n


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


def paired_diffs(by_run, method_a, method_b, *, tw_filter=None) -> list[float]:
    """Per-(instance, rep) paired relative differences (b - a) / a * 100."""

    diffs = []
    for (method, size, tw, index, rep), value_a in by_run.items():
        if method != method_a:
            continue
        if tw_filter is not None and tw not in tw_filter:
            continue
        key_b = (method_b, size, tw, index, rep)
        if key_b in by_run:
            value_b = by_run[key_b]
            diffs.append(100.0 * (value_b - value_a) / value_a)
    return diffs


def instance_mean_diffs(by_run, method_a, method_b, *, tw_filter=None) -> list[float]:
    """Per-instance paired relative differences using rep means (for VAA pairing)."""

    values: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (method, size, tw, index, rep), value in by_run.items():
        if tw_filter is not None and tw not in tw_filter:
            continue
        values[(size, tw, index)][method].append(value)

    diffs = []
    for _, methods in sorted(values.items(), key=str):
        if method_a in methods and method_b in methods:
            mean_a = statistics.mean(methods[method_a])
            mean_b = statistics.mean(methods[method_b])
            diffs.append(100.0 * (mean_b - mean_a) / mean_a)
    return diffs


def main() -> None:
    by_run = load()

    print("=== Wilcoxon signed-rank (two-sided), objective ===")
    print(f"{'comparison':<42}{'n':>5}{'mean diff%':>11}{'p-value':>10}{'verdict':>14}")

    comparisons = [
        ("GILS-1000", "VAA", None, "instance"),
        ("GILS-1000", "Paper-SA-RL5-1000", (None,), "run"),
        ("GILS-1000", "GILS-uniform-1000", None, "run"),
        ("GILS-1000", "GILS-dqn-1000", None, "run"),
        ("GILS-uniform-1000", "GILS-dqn-1000", None, "run"),
    ]
    for method_a, method_b, tw_filter, mode in comparisons:
        if mode == "run":
            diffs = paired_diffs(by_run, method_a, method_b, tw_filter=tw_filter)
        else:
            diffs = instance_mean_diffs(by_run, method_a, method_b, tw_filter=tw_filter)
        if not diffs:
            continue
        p, n = wilcoxon_signed_rank(diffs)
        mean_diff = statistics.mean(diffs)
        verdict = "significant" if p < 0.05 else "no difference"
        print(
            f"{method_a} vs {method_b:<20}{n:>5}{mean_diff:>+11.3f}{p:>10.4f}{verdict:>14}"
        )

    print()
    print("=== Budget sensitivity: mean gap% to per-instance best-known ===")
    best_known: dict[tuple, float] = {}
    for (method, size, tw, index, rep), value in by_run.items():
        key = (size, tw, index)
        if key not in best_known or value < best_known[key]:
            best_known[key] = value

    budgets = (50, 200, 1000, 3000)
    selectors = {"uniform": "GILS-uniform-", "tabular": "GILS-", "dqn": "GILS-dqn-"}
    print(f"{'selector':<10}" + "".join(f"{b:>10}" for b in budgets))
    for selector_label, prefix in selectors.items():
        row = []
        for budget in budgets:
            method = f"{prefix}{budget}"
            gaps = []
            for (m, size, tw, index, rep), value in by_run.items():
                if m != method:
                    continue
                base = best_known[(size, tw, index)]
                gaps.append(100.0 * (value - base) / base)
            row.append(statistics.mean(gaps) if gaps else float("nan"))
        print(f"{selector_label:<10}" + "".join(f"{v:>10.2f}" for v in row))

    print()
    print("=== Budget sensitivity: selector spread per budget (max-min of the three) ===")
    for budget in budgets:
        means = []
        for prefix in selectors.values():
            method = f"{prefix}{budget}"
            values = [
                100.0 * (v - best_known[(s, t, i)]) / best_known[(s, t, i)]
                for (m, s, t, i, rep), v in by_run.items()
                if m == method
            ]
            if values:
                means.append(statistics.mean(values))
        if means:
            print(f"budget {budget:>5}: spread {max(means) - min(means):.3f}%p")


if __name__ == "__main__":
    main()
