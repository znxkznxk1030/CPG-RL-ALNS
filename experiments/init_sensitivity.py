"""Direction-B kill test: does the initial solution survive the improvement engine?

If the guided-ILS engine converges to the same final objective regardless of the
starting solution, a learned construction policy cannot help (direction B dies).
If better starts lead to better finals, construction learning has room, and the
init-vs-final correlation tells us what training signal to use.

Run:
    python experiments/init_sensitivity.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossdock_solver.baselines.vaa import vaa_solution
from crossdock_solver.baselines.vaa_qrl import VaaQRLConfig, run_vaa_qrl
from crossdock_solver.core.fast_evaluator import FastEvaluator
from crossdock_solver.initial.random_init import random_feasible_solution
from experiments.protocol import BenchmarkCell, cell_instance


def _best_of_random(instance, fast, rng, samples: int = 30):
    best = None
    best_objective = float("inf")
    for _ in range(samples):
        candidate = random_feasible_solution(instance, rng)
        objective = fast.evaluate(candidate).objective
        if objective < best_objective:
            best = candidate
            best_objective = objective
    return best


def main(
    *,
    size_classes=("S", "M", "L"),
    tw_levels=(None, "medium"),
    indices=(0, 1, 2),
    reps=(0, 1, 2),
    iterations: int = 300,
) -> None:
    results = defaultdict(lambda: defaultdict(lambda: {"init": [], "final": []}))

    for size in size_classes:
        for tw in tw_levels:
            cell = BenchmarkCell(size, "uniform", tw)
            weight = 1.0 if tw is not None else 0.0
            for index in indices:
                instance = cell_instance("tuning", cell, index)
                fast = FastEvaluator(instance, tardiness_weight=weight)
                for rep in reps:
                    rng = random.Random(7_000 + index * 10 + rep)
                    starts = {
                        "vaa": vaa_solution(instance),
                        "random": random_feasible_solution(instance, rng),
                        "random30": _best_of_random(instance, fast, rng),
                    }
                    for name, start in starts.items():
                        init_objective = fast.evaluate(start).objective
                        run = run_vaa_qrl(
                            instance,
                            VaaQRLConfig(
                                max_iterations=iterations,
                                tardiness_weight=weight,
                                seed=80_000 + index * 100 + rep,
                            ),
                            initial_solution=start,
                        )
                        final_objective = (
                            run.result.makespan + weight * run.result.total_tardiness
                        )
                        bucket = results[cell.name][name]
                        bucket["init"].append(init_objective)
                        bucket["final"].append(final_objective)

    print(f"{'cell':<14}{'start':<10}{'mean init':>12}{'mean final':>12}{'final vs vaa':>14}")
    for cell_name, by_start in results.items():
        vaa_final = statistics.mean(by_start["vaa"]["final"])
        for name in ("vaa", "random30", "random"):
            bucket = by_start[name]
            mean_init = statistics.mean(bucket["init"])
            mean_final = statistics.mean(bucket["final"])
            delta = 100.0 * (mean_final - vaa_final) / vaa_final
            print(
                f"{cell_name:<14}{name:<10}{mean_init:>12.1f}{mean_final:>12.1f}{delta:>+13.2f}%"
            )


if __name__ == "__main__":
    main()
