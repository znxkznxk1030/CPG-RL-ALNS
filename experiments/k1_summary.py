"""Summarize K1 results into the paper's main table.

Reads outputs/k1_results.jsonl and prints, per cell (size x TW level):
mean +- std objective per method, gap vs best lower bound (CP-SAT bound or
combinatorial bound, whichever is tighter), and CP-SAT incumbent comparison.

Usage:
    python experiments/k1_summary.py
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

from crossdock_solver.exact.lower_bounds import combinatorial_objective_lower_bound
from experiments.protocol import BenchmarkCell, cell_instance


RESULTS = ROOT / "outputs" / "k1_results.jsonl"
SEARCH_METHODS = ("VAA", "Paper-SA-RL5-1000", "GILS-uniform-1000", "GILS-1000", "GILS-dqn-1000")


def load_records() -> list[dict]:
    records = []
    with RESULTS.open() as source:
        for line in source:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    records = load_records()

    by_instance: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = (record["size_class"], record["tw_tightness"], record["index"])
        by_instance[key][record["method"]].append(record)

    cells = sorted({(k[0], k[1]) for k in by_instance}, key=str)
    print(
        f"{'cell':<12}{'method':<20}{'mean obj':>12}{'std':>8}{'gap_lb%':>9}"
        f"{'vs CP-SAT%':>11}{'n':>4}"
    )
    for size, tw in cells:
        instance_keys = [k for k in by_instance if k[0] == size and k[1] == tw]

        lb: dict[tuple, float] = {}
        cp_incumbent: dict[tuple, float] = {}
        weight = 1.0 if tw is not None else 0.0
        for key in instance_keys:
            cell = BenchmarkCell(key[0], "uniform", key[1])
            instance = cell_instance("test", cell, key[2])
            bound = combinatorial_objective_lower_bound(instance, weight)
            for record in by_instance[key].get("CPSAT-300", []) + by_instance[key].get("CPSAT-600", []):
                if record.get("lower_bound"):
                    bound = max(bound, record["lower_bound"])
                if record.get("objective") is not None:
                    cp_incumbent[key] = record["objective"]
            lb[key] = bound

        for method in SEARCH_METHODS:
            objectives = []
            gaps = []
            vs_cp = []
            for key in instance_keys:
                for record in by_instance[key].get(method, []):
                    objective = record.get("objective", record.get("makespan"))
                    objectives.append(objective)
                    if lb[key] > 0:
                        gaps.append(100.0 * (objective - lb[key]) / lb[key])
                    if key in cp_incumbent and cp_incumbent[key] > 0:
                        vs_cp.append(
                            100.0 * (objective - cp_incumbent[key]) / cp_incumbent[key]
                        )
            if not objectives:
                continue
            std = statistics.pstdev(objectives) if len(objectives) > 1 else 0.0
            gap_text = f"{statistics.mean(gaps):>8.1f}" if gaps else "        -"
            cp_text = f"{statistics.mean(vs_cp):>+10.2f}" if vs_cp else "         -"
            print(
                f"{size}-{tw or 'none':<9}{method:<20}"
                f"{statistics.mean(objectives):>12.1f}{std:>8.1f}{gap_text}{cp_text}"
                f"{len(objectives):>4}"
            )
        print()


if __name__ == "__main__":
    main()
