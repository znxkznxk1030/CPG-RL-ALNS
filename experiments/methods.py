"""Method registry for the experiment runner.

All methods must be defined at module import time so multiprocessing workers
(spawn start method on macOS) can resolve them after re-import. A method is a
callable `(instance, seed, budget_sec) -> dict` returning at least `makespan`
and `runtime_sec`.
"""

from __future__ import annotations

from typing import Callable

from crossdock_solver.baselines.paper_sa_rl import PaperSARLConfig, run_paper_sa_rl
from crossdock_solver.baselines.vaa import run_vaa
from crossdock_solver.baselines.vaa_qrl import VaaQRLConfig, run_vaa_qrl
from crossdock_solver.data.instance import CrossDockInstance


MethodFn = Callable[[CrossDockInstance, int, float | None], dict]


def _vaa(instance: CrossDockInstance, seed: int, budget_sec: float | None) -> dict:
    run = run_vaa(instance)
    return {"makespan": run.result.makespan, "runtime_sec": run.runtime_sec}


def _paper_sa_rl(iterations: int) -> MethodFn:
    def method(instance: CrossDockInstance, seed: int, budget_sec: float | None) -> dict:
        run = run_paper_sa_rl(
            instance,
            PaperSARLConfig(max_iterations=iterations, seed=seed),
        )
        return {"makespan": run.result.makespan, "runtime_sec": run.runtime_sec}

    return method


def _vaa_qrl(iterations: int) -> MethodFn:
    def method(instance: CrossDockInstance, seed: int, budget_sec: float | None) -> dict:
        run = run_vaa_qrl(
            instance,
            VaaQRLConfig(
                max_iterations=iterations,
                time_budget_sec=budget_sec,
                seed=seed,
            ),
        )
        return {"makespan": run.result.makespan, "runtime_sec": run.runtime_sec}

    return method


METHOD_REGISTRY: dict[str, MethodFn] = {
    "VAA": _vaa,
    "Paper-SA-RL5-300": _paper_sa_rl(300),
    "VAA-QRL-50": _vaa_qrl(50),
    "VAA-QRL-300": _vaa_qrl(300),
    "VAA-QRL-1000": _vaa_qrl(1000),
}
