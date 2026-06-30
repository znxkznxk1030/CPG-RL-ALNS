from __future__ import annotations

from crossdock_solver.baselines.random_baseline import random_best_of, random_one_solution
from crossdock_solver.baselines.paper_sa_rl import PaperSARLConfig, paper_sa_rl5, paper_sa_rl6, run_paper_sa_rl
from crossdock_solver.baselines.vaa import run_vaa, vaa_solution, vva_solution
from crossdock_solver.core.evaluator import evaluate_solution
from crossdock_solver.core.feasibility import check_feasible
from tests.conftest import make_toy_instance


def test_vaa_solution_is_feasible_and_evaluable() -> None:
    instance = make_toy_instance()
    solution = vaa_solution(instance)
    check_feasible(instance, solution)
    assert evaluate_solution(instance, solution).makespan > 0


def test_vva_alias_matches_vaa_solution() -> None:
    instance = make_toy_instance()
    assert vva_solution(instance) == vaa_solution(instance)


def test_random_baselines_return_feasible_results() -> None:
    instance = make_toy_instance()
    one = random_one_solution(instance, seed=1)
    best = random_best_of(instance, samples=5, seed=1)

    check_feasible(instance, one.solution)
    check_feasible(instance, best.solution)
    assert best.result.makespan <= max(best.result.makespan, one.result.makespan)
    assert best.samples == 5


def test_run_vaa_returns_baseline_run() -> None:
    instance = make_toy_instance()
    run = run_vaa(instance)
    assert run.name == "VAA"
    assert run.result.makespan > 0


def test_paper_sa_rl5_baseline_runs() -> None:
    instance = make_toy_instance()
    run = run_paper_sa_rl(
        instance,
        PaperSARLConfig(max_iterations=25, seed=123),
    )
    check_feasible(instance, run.solution)
    assert run.name == "Paper-SA-RL5-25"
    assert run.result.makespan > 0


def test_paper_sa_rl_aliases_run() -> None:
    instance = make_toy_instance()
    rl5 = paper_sa_rl5(instance, seed=1)
    rl6 = paper_sa_rl6(instance, seed=1)
    check_feasible(instance, rl5.solution)
    check_feasible(instance, rl6.solution)
    assert rl5.name.startswith("Paper-SA-RL5")
    assert rl6.name.startswith("Paper-SA-RL6")
