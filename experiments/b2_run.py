"""B2: engine-component ablation (leave-one-out) for the thesis.

Holds the selection policy fixed (uniform) and the operator pool fixed (full),
and removes one engine component at a time to measure its marginal contribution
to the near-optimal attractor:

    none    -> full engine (reference, == GILS-uniform-1000);
    init    -> VAA construction replaced by a random feasible start;
    descent -> best-improvement descent removed;
    sa      -> SA acceptance replaced by greedy;
    restart -> kick-restart on stagnation removed.

Together with B1 (operator-pool ablation) this decomposes where the quality
comes from: engine skeleton (this file) vs guided operators (B1) vs learned
selection (K1, negligible).

Test-pool measurement, same cells as K1. The `none` arm reproduces
GILS-uniform-1000 exactly (consistency check).

Usage:
    python experiments/b2_run.py
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runner import Job, run_jobs


OUTPUT = ROOT / "outputs" / "b2_component_ablation.jsonl"

SIZES = ("S", "M", "L")
TW_LEVELS = (None, "medium", "tight")
INDICES = (0, 1, 2, 3, 4)
REPS = (0, 1, 2, 3, 4)
DROPS = ("none", "init", "descent", "sa", "restart")


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
                for drop in DROPS:
                    for rep in REPS:
                        jobs.append(Job(method=f"GILS-ablate-{drop}-1000", rep=rep, **common))
    return jobs


def main() -> None:
    executed = run_jobs(ablation_jobs(), OUTPUT, workers=6)
    print(f"b2: executed {executed} new jobs -> {OUTPUT}")


if __name__ == "__main__":
    main()
