"""Phase C3: tardiness-weight (lambda) sensitivity.

Justifies the modelling choice lambda = 1 by tracing the makespan-vs-tardiness
trade-off as lambda varies. The engine minimises makespan + lambda * total
tardiness; sweeping lambda and recording the *physical* makespan and total
tardiness of the returned schedule yields a Pareto-style trade-off curve.

Methodology: run on the TUNING pool, never the test pool. Lambda is a modelling
parameter, so a curve justifying its value belongs to model design, not final
reporting. Seeds depend only on (index, rep) so the sole difference across
lambda values on a given (instance, rep) is the objective weight (paired design).

Run:
    python experiments/lambda_sensitivity.py
writes outputs/lambda_sensitivity.jsonl and prints a summary.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossdock_solver.baselines.vaa_qrl import VaaQRLConfig, run_vaa_qrl
from experiments.protocol import BenchmarkCell, cell_instance

LAMBDAS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SIZES: tuple[str, ...] = ("S", "M", "L")
TIGHTNESS: tuple[str, ...] = ("medium", "tight")  # none cells have due=inf
INDICES = range(5)
REPS = range(3)
ITERATIONS = 1000
OUT = ROOT / "outputs" / "lambda_sensitivity.jsonl"


def main() -> None:
    records = []
    total = len(SIZES) * len(TIGHTNESS) * len(INDICES) * len(REPS) * len(LAMBDAS)
    done = 0
    for size in SIZES:
        for tw in TIGHTNESS:
            cell = BenchmarkCell(size, "uniform", tw)
            for index in INDICES:
                instance = cell_instance("tuning", cell, index)
                for rep in REPS:
                    seed = 90_000 + index * 100 + rep  # shared across lambdas
                    for lam in LAMBDAS:
                        run = run_vaa_qrl(
                            instance,
                            VaaQRLConfig(
                                max_iterations=ITERATIONS,
                                tardiness_weight=lam,
                                seed=seed,
                            ),
                        )
                        rec = {
                            "cell": cell.name,
                            "size": size,
                            "tightness": tw,
                            "index": index,
                            "rep": rep,
                            "lam": lam,
                            "makespan": run.result.makespan,
                            "total_tardiness": run.result.total_tardiness,
                            "runtime_sec": run.runtime_sec,
                        }
                        records.append(rec)
                        done += 1
                    if done % 70 == 0 or done == total:
                        print(f"  ... {done}/{total} runs", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {OUT} ({len(records)} records)")


if __name__ == "__main__":
    main()
