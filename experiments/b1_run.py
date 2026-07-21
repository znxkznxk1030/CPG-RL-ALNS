"""B1: guided-operator ablation for the thesis (advisor-directed reframing).

Holds the selection policy fixed (uniform random) and varies only the operator
pool: generic (7 paper neighborhoods) / critical (+g1,g2) / full (+g3,g4 ==
ACTIONS_TW). This isolates the contribution of the bottleneck- and
tardiness-guided operators, which the selector ablation (uniform/tabular/DQN)
cannot show. It directly supports the claim that performance comes from the
guided search structure, not from a learned selection policy.

Test-pool measurement, same cells as K1. The three arms are structurally
pre-specified (no tuning), so this is analysis, not hyperparameter selection;
the `full` arm reproduces GILS-uniform-1000 exactly (a consistency check).

Usage:
    python experiments/b1_run.py            # run the ablation grid
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runner import Job, run_jobs


OUTPUT = ROOT / "outputs" / "b1_guided_ablation.jsonl"

SIZES = ("S", "M", "L")
TW_LEVELS = (None, "medium", "tight")
INDICES = (0, 1, 2, 3, 4)
REPS = (0, 1, 2, 3, 4)
POOLS = ("generic", "critical", "full")


def ablation_jobs() -> list[Job]:
    jobs: list[Job] = []
    for size in SIZES:
        for tw in TW_LEVELS:
            for index in INDICES:
                common = dict(
                    pool="test",
                    size_class=size,
                    flow_pattern="uniform",
                    tw_tightness=tw,
                    index=index,
                )
                for pool in POOLS:
                    for rep in REPS:
                        jobs.append(Job(method=f"GILS-{pool}-1000", rep=rep, **common))
    return jobs


def main() -> None:
    executed = run_jobs(ablation_jobs(), OUTPUT, workers=6)
    print(f"b1: executed {executed} new jobs -> {OUTPUT}")


if __name__ == "__main__":
    main()
