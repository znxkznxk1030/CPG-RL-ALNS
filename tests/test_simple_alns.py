from __future__ import annotations

from crossdock_solver.alns.destroy import critical_door_destroy
from crossdock_solver.alns.loop import ALNSConfig, simple_alns
from crossdock_solver.alns.repair import greedy_repair, regret_k_repair
from crossdock_solver.core.evaluator import evaluate_solution
from crossdock_solver.core.feasibility import check_feasible
from crossdock_solver.initial.random_init import random_feasible_solution
from tests.conftest import make_toy_instance, make_toy_solution


def test_random_initial_solution_is_feasible() -> None:
    instance = make_toy_instance()
    solution = random_feasible_solution(instance)
    check_feasible(instance, solution)


def test_critical_door_destroy_and_repairs_return_feasible_solution() -> None:
    instance = make_toy_instance()
    solution = make_toy_solution()
    result = evaluate_solution(instance, solution)
    destroyed = critical_door_destroy(instance, solution, result, size="small")

    assert destroyed.removed_trucks
    assert len(destroyed.removed_trucks) == len(destroyed.removed_destinations)

    greedy = greedy_repair(instance, destroyed)
    regret = regret_k_repair(instance, destroyed, k=2)

    check_feasible(instance, greedy)
    check_feasible(instance, regret)


def test_simple_alns_runs_and_keeps_best_solution() -> None:
    instance = make_toy_instance()
    initial = make_toy_solution()
    initial_result = evaluate_solution(instance, initial)

    run = simple_alns(
        instance,
        ALNSConfig(max_iterations=15, repair_name="regret", seed=7),
        initial_solution=initial,
    )

    check_feasible(instance, run.best_solution)
    assert len(run.logs) == 15
    assert run.best_result.makespan <= initial_result.makespan

