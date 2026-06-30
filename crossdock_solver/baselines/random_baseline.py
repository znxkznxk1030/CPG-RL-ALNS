from __future__ import annotations

from dataclasses import dataclass
import random
import time

from crossdock_solver.core.evaluator import ScheduleResult, evaluate_solution
from crossdock_solver.core.solution import Solution
from crossdock_solver.data.instance import CrossDockInstance
from crossdock_solver.initial.random_init import random_feasible_solution


@dataclass
class BaselineRun:
    name: str
    solution: Solution
    result: ScheduleResult
    runtime_sec: float
    samples: int = 1


def random_one_solution(
    instance: CrossDockInstance,
    *,
    seed: int | None = None,
) -> BaselineRun:
    start = time.perf_counter()
    solution = random_feasible_solution(instance, random.Random(seed))
    result = evaluate_solution(instance, solution)
    return BaselineRun(
        name="Random-1",
        solution=solution,
        result=result,
        runtime_sec=time.perf_counter() - start,
        samples=1,
    )


def random_best_of(
    instance: CrossDockInstance,
    *,
    samples: int = 30,
    seed: int | None = None,
) -> BaselineRun:
    if samples < 1:
        raise ValueError("samples must be positive")

    start = time.perf_counter()
    rng = random.Random(seed)
    best_solution: Solution | None = None
    best_result: ScheduleResult | None = None

    for _ in range(samples):
        solution = random_feasible_solution(instance, rng)
        result = evaluate_solution(instance, solution)
        if best_result is None or result.makespan < best_result.makespan:
            best_solution = solution
            best_result = result

    if best_solution is None or best_result is None:
        raise RuntimeError("random baseline failed to generate a solution")

    return BaselineRun(
        name=f"Random-{samples}",
        solution=best_solution,
        result=best_result,
        runtime_sec=time.perf_counter() - start,
        samples=samples,
    )

