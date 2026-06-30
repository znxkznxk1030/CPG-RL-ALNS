from __future__ import annotations

from dataclasses import dataclass, field
import random

from crossdock_solver.alns.acceptance import accept_by_sa
from crossdock_solver.alns.destroy import critical_door_destroy
from crossdock_solver.alns.repair import greedy_repair, regret_k_repair
from crossdock_solver.core.evaluator import ScheduleResult, evaluate_solution
from crossdock_solver.core.solution import Solution
from crossdock_solver.data.instance import CrossDockInstance
from crossdock_solver.initial.random_init import random_feasible_solution


@dataclass
class ALNSConfig:
    max_iterations: int = 200
    destroy_size: str = "small"
    repair_name: str = "regret"
    regret_k: int = 2
    initial_temperature: float | None = None
    cooling_rate: float = 0.995
    seed: int | None = None


@dataclass
class IterationLog:
    iteration: int
    current_makespan: float
    candidate_makespan: float
    best_makespan: float
    accepted: bool
    critical_door: int
    removed_count: int
    temperature: float


@dataclass
class ALNSRunResult:
    best_solution: Solution
    best_result: ScheduleResult
    final_solution: Solution
    final_result: ScheduleResult
    logs: list[IterationLog] = field(default_factory=list)


def simple_alns(
    instance: CrossDockInstance,
    config: ALNSConfig | None = None,
    *,
    initial_solution: Solution | None = None,
) -> ALNSRunResult:
    config = config or ALNSConfig()
    rng = random.Random(config.seed)

    current = initial_solution.copy() if initial_solution is not None else random_feasible_solution(instance, rng)
    current_result = evaluate_solution(instance, current)
    best = current.copy()
    best_result = current_result

    temperature = (
        config.initial_temperature
        if config.initial_temperature is not None
        else max(1.0, 0.05 * current_result.makespan)
    )

    logs: list[IterationLog] = []

    for iteration in range(config.max_iterations):
        iteration_result = evaluate_solution(instance, current)
        destroyed = critical_door_destroy(
            instance,
            current,
            iteration_result,
            size=config.destroy_size,
            rng=rng,
        )

        if config.repair_name == "greedy":
            candidate = greedy_repair(instance, destroyed, rng=rng)
        elif config.repair_name == "regret":
            candidate = regret_k_repair(instance, destroyed, k=config.regret_k, rng=rng)
        else:
            raise ValueError(f"unknown repair_name {config.repair_name!r}")

        candidate_result = evaluate_solution(instance, candidate)
        accepted = accept_by_sa(
            iteration_result.makespan,
            candidate_result.makespan,
            temperature,
            rng,
        )

        if accepted:
            current = candidate
            current_result = candidate_result
        else:
            current_result = iteration_result

        if candidate_result.makespan < best_result.makespan:
            best = candidate.copy()
            best_result = candidate_result

        logs.append(
            IterationLog(
                iteration=iteration,
                current_makespan=current_result.makespan,
                candidate_makespan=candidate_result.makespan,
                best_makespan=best_result.makespan,
                accepted=accepted,
                critical_door=iteration_result.critical_door,
                removed_count=len(destroyed.removed_trucks),
                temperature=temperature,
            )
        )

        temperature *= config.cooling_rate

    final_result = evaluate_solution(instance, current)
    return ALNSRunResult(
        best_solution=best,
        best_result=best_result,
        final_solution=current,
        final_result=final_result,
        logs=logs,
    )

