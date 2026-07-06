from __future__ import annotations

import random

import numpy as np
import pytest

from crossdock_solver.baselines.vaa_qrl import (
    ACTIONS,
    ACTIONS_TW,
    VaaQRLConfig,
    run_vaa_qrl,
)
from crossdock_solver.core.fast_evaluator import FastEvaluator
from crossdock_solver.baselines.vaa import vaa_solution
from crossdock_solver.data.generator import generate_random_instance
from crossdock_solver.core.feasibility import check_feasible
from crossdock_solver.rl.dqn import DQNAgent
from crossdock_solver.rl.features import feature_dim
from crossdock_solver.rl.selectors import (
    DQNSelector,
    SelectionContext,
    TabularQSelector,
    UniformSelector,
)
from tests.conftest import make_toy_instance


def test_default_selector_reproduces_tabular_run() -> None:
    # Passing an explicitly constructed TabularQSelector must give the exact
    # same result as the default (None) selector path.
    instance = make_toy_instance()
    config = VaaQRLConfig(max_iterations=40, seed=11)

    default_run = run_vaa_qrl(instance, config)
    explicit = TabularQSelector(
        ACTIONS,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        random_prob=config.random_selection_prob,
        roulette_prob=config.roulette_selection_prob,
    )
    explicit_run = run_vaa_qrl(instance, VaaQRLConfig(max_iterations=40, seed=11), selector=explicit)

    assert default_run.result.makespan == pytest.approx(explicit_run.result.makespan)


def test_uniform_selector_runs_and_is_feasible() -> None:
    instance = make_toy_instance()
    run = run_vaa_qrl(
        instance,
        VaaQRLConfig(max_iterations=30, seed=5),
        selector=UniformSelector(ACTIONS),
    )
    check_feasible(instance, run.solution)
    assert run.result.makespan > 0


def test_dqn_selector_trains_and_runs_zero_shot(tmp_path) -> None:
    instance = make_toy_instance()
    agent = DQNAgent(
        input_dim=feature_dim(len(ACTIONS)),
        num_actions=len(ACTIONS),
        warmup=8,
        batch_size=8,
        seed=0,
    )

    train_run = run_vaa_qrl(
        instance,
        VaaQRLConfig(max_iterations=30, seed=3),
        selector=DQNSelector(agent, ACTIONS, epsilon=0.5, train=True),
    )
    check_feasible(instance, train_run.solution)
    assert agent.steps == 30

    path = tmp_path / "agent.pt"
    agent.save(path)
    loaded = DQNAgent.load(path)

    zero_shot = run_vaa_qrl(
        instance,
        VaaQRLConfig(max_iterations=30, seed=4),
        selector=DQNSelector(loaded, ACTIONS, epsilon=0.0, train=False),
    )
    check_feasible(instance, zero_shot.solution)
    assert zero_shot.result.makespan > 0


def test_tw_action_pool_runs_on_time_window_instance() -> None:
    instance = generate_random_instance(
        seed=41, num_compounds=3, num_outbounds=4, num_doors=4,
        num_products=3, tw_tightness="tight",
    )
    run = run_vaa_qrl(
        instance,
        VaaQRLConfig(max_iterations=60, seed=2, tardiness_weight=1.0),
        selector=UniformSelector(ACTIONS_TW),
    )
    check_feasible(instance, run.solution)

    fast = FastEvaluator(instance, tardiness_weight=1.0)
    vaa_objective = fast.evaluate(vaa_solution(instance)).objective
    final = run.result.makespan + run.result.total_tardiness
    assert final <= vaa_objective + 1e-9


def test_most_tardy_truck_matches_reference_evaluator() -> None:
    from crossdock_solver.core.evaluator import evaluate_solution

    instance = generate_random_instance(
        seed=42, num_compounds=3, num_outbounds=5, num_doors=4,
        num_products=3, tw_tightness="tight",
    )
    solution = vaa_solution(instance)
    fast_result = FastEvaluator(instance, tardiness_weight=1.0).evaluate(solution)
    slow = evaluate_solution(instance, solution)

    tardiness = {
        truck: max(0.0, slow.truck_finish[truck] - instance.due_time[truck])
        for truck in instance.all_trucks
        if instance.due_time[truck] != float("inf")
    }
    if max(tardiness.values(), default=0.0) > 0.0:
        worst = max(tardiness, key=tardiness.get)
        assert fast_result.most_tardy_truck == worst
    else:
        assert fast_result.most_tardy_truck is None


def test_dqn_selector_greedy_choice_follows_q_values() -> None:
    class FakeAgent:
        def q_values(self, features):
            q = np.zeros(len(ACTIONS), dtype=np.float32)
            q[3] = 10.0
            return q

    selector = DQNSelector(FakeAgent(), ACTIONS, epsilon=0.0)
    context = SelectionContext(state_bin=1, features=np.zeros(feature_dim(len(ACTIONS))))
    assert selector.select(context, random.Random(0)) == ACTIONS[3]
