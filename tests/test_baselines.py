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
from crossdock_solver.baselines.graph_cargo_rl import (
    DOOR_FEATURES,
    GRAPH_OBS_SIZE,
    GraphCargoRLConfig,
    _build_graph_state,
    _encode_graph_state,
    graph_cargo_rl_solution,
    run_graph_cargo_rl,
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
from crossdock_solver.data.generator import generate_random_instance
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


def test_graph_cargo_state_is_variable_size_with_fixed_encoding() -> None:
    small = make_toy_instance()
    medium = generate_random_instance(
        num_compounds=4,
        num_outbounds=6,
        num_doors=5,
        num_products=3,
        seed=301,
    )

    small_state = _build_graph_state(
        small,
        current_destination=small.destinations[0],
        remaining_destinations=list(small.destinations),
        available_trucks=set(small.all_trucks),
        assigned_count=0,
        reference_solution=vaa_solution(small),
    )
    medium_state = _build_graph_state(
        medium,
        current_destination=medium.destinations[0],
        remaining_destinations=list(medium.destinations),
        available_trucks=set(medium.all_trucks),
        assigned_count=0,
        reference_solution=vaa_solution(medium),
    )

    assert small_state.truck_nodes.shape[0] != medium_state.truck_nodes.shape[0]
    assert small_state.destination_nodes.shape[0] != medium_state.destination_nodes.shape[0]
    assert small_state.door_nodes.shape[0] != medium_state.door_nodes.shape[0]
    assert small_state.door_nodes.shape[1] == DOOR_FEATURES
    assert _encode_graph_state(small_state).shape == (GRAPH_OBS_SIZE,)
    assert _encode_graph_state(medium_state).shape == (GRAPH_OBS_SIZE,)


def test_graph_cargo_state_includes_door_release_and_workload() -> None:
    instance = make_toy_instance()
    reference_solution = vaa_solution(instance)
    state = _build_graph_state(
        instance,
        current_destination=instance.destinations[0],
        remaining_destinations=list(instance.destinations),
        available_trucks=set(instance.all_trucks),
        assigned_count=0,
        reference_solution=reference_solution,
    )

    release_feature = state.door_nodes[:, 3]
    workload_feature = state.door_nodes[:, 4]
    utilization_feature = state.door_nodes[:, 5]

    assert release_feature.max() > 0.0
    assert workload_feature.max() > 0.0
    assert utilization_feature.max() > 0.0


def test_graph_cargo_rl_baseline_runs() -> None:
    instance = make_toy_instance()
    result = run_graph_cargo_rl(
        instance,
        GraphCargoRLConfig(episodes=8, batch_size=4, warmup=4, seed=13),
    )

    check_feasible(instance, result.run.solution)
    assert result.run.name == "GraphCargoMatrix-RL-8"
    assert result.run.result.makespan > 0
    assert len(result.training_rewards) == 8


def test_graph_cargo_rl_solution_helper_returns_solution() -> None:
    instance = make_toy_instance()
    solution = graph_cargo_rl_solution(instance, seed=13, episodes=4)

    check_feasible(instance, solution)
