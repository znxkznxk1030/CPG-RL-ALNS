"""K1: main experiment sweep for the APIEMS 2026 full paper (KCI-track plan).

Test-pool measurement. Methods and hyperparameters were frozen on the tuning
pool; this is the first and only test-pool run for these methods.

Grid: sizes S/M/L x uniform flow x TW in (none, medium, tight), 5 instances per
cell. Stochastic methods get 5 replications; VAA and CP-SAT are deterministic.
CP-SAT runs on all S instances (300s) and on 2 instances per M/L cell (600s).

Usage:
    python experiments/k1_run.py search   # fast batch (minutes)
    python experiments/k1_run.py cpsat    # long batch (hours, run in background)
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runner import Job, run_jobs


OUTPUT = ROOT / "outputs" / "k1_results.jsonl"

SIZES = ("S", "M", "L")
TW_LEVELS = (None, "medium", "tight")
INDICES = (0, 1, 2, 3, 4)
REPS = (0, 1, 2, 3, 4)


def search_jobs() -> list[Job]:
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
                jobs.append(Job(method="VAA", rep=0, **common))
                for rep in REPS:
                    jobs.append(Job(method="GILS-1000", rep=rep, **common))
                    jobs.append(Job(method="GILS-uniform-1000", rep=rep, **common))
                    jobs.append(Job(method="GILS-dqn-1000", rep=rep, **common))
                    if tw is None:
                        jobs.append(Job(method="Paper-SA-RL5-1000", rep=rep, **common))
    return jobs


def budget_jobs() -> list[Job]:
    """Budget-sensitivity sweep: selector x iteration budget (50/200/3000).

    The 1000-iteration points come from the main search batch.
    """

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
                for budget in (50, 200, 3000):
                    for selector in ("", "uniform-", "dqn-"):
                        for rep in REPS:
                            jobs.append(
                                Job(method=f"GILS-{selector}{budget}", rep=rep, **common)
                            )
    return jobs


def cpsat_jobs() -> list[Job]:
    jobs: list[Job] = []
    for tw in TW_LEVELS:
        for index in INDICES:
            jobs.append(
                Job(
                    method="CPSAT-300",
                    pool="test",
                    size_class="S",
                    flow_pattern="uniform",
                    tw_tightness=tw,
                    index=index,
                    rep=0,
                )
            )
        for size in ("M", "L"):
            for index in (0, 1):
                jobs.append(
                    Job(
                        method="CPSAT-600",
                        pool="test",
                        size_class=size,
                        flow_pattern="uniform",
                        tw_tightness=tw,
                        index=index,
                        rep=0,
                    )
                )
    return jobs


def main() -> None:
    batch = sys.argv[1] if len(sys.argv) > 1 else "search"
    if batch == "search":
        executed = run_jobs(search_jobs(), OUTPUT, workers=6)
    elif batch == "budget":
        executed = run_jobs(budget_jobs(), OUTPUT, workers=6)
    elif batch == "cpsat":
        executed = run_jobs(cpsat_jobs(), OUTPUT, workers=2)
    else:
        raise SystemExit(f"unknown batch {batch!r} (use: search | budget | cpsat)")
    print(f"{batch}: executed {executed} new jobs -> {OUTPUT}")


if __name__ == "__main__":
    main()
