from __future__ import annotations

from crossdock_solver.baselines.cargo_matrix_rl import (
    CargoMatrixRLConfig,
    _top_load_destination_order,
    cargo_matrix_rl_solution,
    run_cargo_matrix_rl,
    run_topload_cargo_matrix_rl,
    topload_cargo_matrix_rl_solution,
)
from crossdock_solver.baselines.destination_agent_rl import (
    DestinationAgentRLConfig,
    destination_agent_rl_solution,
    run_destination_agent_rl,
)
from crossdock_solver.baselines.random_baseline import random_best_of, random_one_solution
from crossdock_solver.baselines.paper_sa_rl import PaperSARLConfig, paper_sa_rl5, paper_sa_rl6, run_paper_sa_rl
from crossdock_solver.baselines.vaa import (
    _compound_destination_cost,
    _destination_load,
    run_vaa,
    vaa_solution,
    vva_solution,
)
from crossdock_solver.core.evaluator import evaluate_solution
from crossdock_solver.core.feasibility import check_feasible
from tests.conftest import make_toy_instance


def test_vaa_solution_is_feasible_and_evaluable() -> None:
    instance = make_toy_instance()
    solution = vaa_solution(instance)
    check_feasible(instance, solution)
    assert evaluate_solution(instance, solution).makespan > 0


def test_vaa_cost_uses_paper_eq_23() -> None:
    instance = make_toy_instance()

    # If C1 keeps D1, it unloads D2 and D3: 5 + 2.
    # It also loads D1 demand initially held by C2: 3.
    assert _compound_destination_cost(instance, "C1", "D1") == 10.0


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


def test_destination_agent_rl_baseline_runs() -> None:
    instance = make_toy_instance()
    result = run_destination_agent_rl(
        instance,
        DestinationAgentRLConfig(episodes=8, batch_size=4, warmup=4, seed=7),
    )

    check_feasible(instance, result.run.solution)
    assert result.run.name == "DestAgent-RL-8"
    assert result.run.result.makespan > 0
    assert len(result.training_rewards) == 8


def test_destination_agent_rl_solution_helper_returns_solution() -> None:
    instance = make_toy_instance()
    solution = destination_agent_rl_solution(instance, seed=7, episodes=4)

    check_feasible(instance, solution)


def test_cargo_matrix_rl_baseline_runs() -> None:
    instance = make_toy_instance()
    result = run_cargo_matrix_rl(
        instance,
        CargoMatrixRLConfig(episodes=8, batch_size=4, warmup=4, seed=11),
    )

    check_feasible(instance, result.run.solution)
    assert result.run.name == "CargoMatrix-RL-8"
    assert result.run.result.makespan > 0
    assert len(result.training_rewards) == 8


def test_cargo_matrix_rl_solution_helper_returns_solution() -> None:
    instance = make_toy_instance()
    solution = cargo_matrix_rl_solution(instance, seed=11, episodes=4)

    check_feasible(instance, solution)


def test_topload_cargo_matrix_order_uses_destination_load() -> None:
    instance = make_toy_instance()
    solution = vaa_solution(instance)

    order = _top_load_destination_order(instance, solution)
    loads = [_destination_load(instance, destination) for destination in order]

    assert all(left >= right for left, right in zip(loads, loads[1:]))


def test_topload_cargo_matrix_rl_baseline_runs() -> None:
    instance = make_toy_instance()
    result = run_topload_cargo_matrix_rl(
        instance,
        CargoMatrixRLConfig(episodes=8, batch_size=4, warmup=4, seed=11),
    )

    check_feasible(instance, result.run.solution)
    assert result.run.name == "TopLoad-CargoMatrix-RL-8"
    assert result.run.result.makespan > 0
    assert len(result.training_rewards) == 8


def test_topload_cargo_matrix_rl_solution_helper_returns_solution() -> None:
    instance = make_toy_instance()
    solution = topload_cargo_matrix_rl_solution(instance, seed=11, episodes=4)

    check_feasible(instance, solution)
