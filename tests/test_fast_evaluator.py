from __future__ import annotations

import random

import pytest

from crossdock_solver.baselines.paper_sa_rl import NEIGHBORHOODS
from crossdock_solver.baselines.vaa import vaa_solution
from crossdock_solver.core.evaluator import evaluate_solution
from crossdock_solver.core.fast_evaluator import FastEvaluator
from crossdock_solver.data.generator import generate_random_instance
from crossdock_solver.initial.random_init import random_feasible_solution
from tests.conftest import make_toy_instance, make_toy_solution


def _assert_equivalent(instance, solution, fast: FastEvaluator) -> None:
    slow = evaluate_solution(instance, solution)
    fast_result = fast.evaluate(solution)

    assert fast_result.makespan == pytest.approx(slow.makespan, abs=1e-9)
    assert fast_result.total_tardiness == pytest.approx(slow.total_tardiness, abs=1e-9)
    assert fast_result.objective == pytest.approx(
        slow.makespan + fast.tardiness_weight * slow.total_tardiness, abs=1e-9
    )
    assert slow.truck_finish[fast_result.critical_truck] == pytest.approx(
        slow.makespan, abs=1e-9
    )
    assert slow.door_finish[fast_result.critical_door] == pytest.approx(
        max(slow.door_finish.values()), abs=1e-9
    )


def test_fast_evaluator_matches_on_toy_instance() -> None:
    instance = make_toy_instance()
    _assert_equivalent(instance, make_toy_solution(), FastEvaluator(instance))


def test_fast_evaluator_matches_on_vaa_solutions() -> None:
    for seed, shape in [
        (11, dict(num_compounds=2, num_outbounds=4, num_doors=3, num_products=2)),
        (12, dict(num_compounds=4, num_outbounds=6, num_doors=5, num_products=3)),
        (13, dict(num_compounds=8, num_outbounds=22, num_doors=10, num_products=3)),
    ]:
        instance = generate_random_instance(seed=seed, **shape)
        _assert_equivalent(instance, vaa_solution(instance), FastEvaluator(instance))


def test_fast_evaluator_fuzz_against_reference() -> None:
    shapes = [
        dict(num_compounds=2, num_outbounds=4, num_doors=3, num_products=2),
        dict(num_compounds=3, num_outbounds=5, num_doors=4, num_products=3),
        dict(num_compounds=6, num_outbounds=14, num_doors=8, num_products=3),
    ]
    operators = tuple(NEIGHBORHOODS.values())

    for shape_seed, shape in enumerate(shapes):
        for tightness in (None, "medium", "tight"):
            instance = generate_random_instance(
                seed=1000 + shape_seed, tw_tightness=tightness, **shape
            )
            fast = FastEvaluator(instance, tardiness_weight=0.5)
            rng = random.Random(2000 + shape_seed)

            for _ in range(5):
                solution = random_feasible_solution(instance, rng)
                _assert_equivalent(instance, solution, fast)

                for _ in range(20):
                    operator = rng.choice(operators)
                    solution = operator(instance, solution, rng)
                    _assert_equivalent(instance, solution, fast)


def test_time_windows_delay_starts_and_create_tardiness() -> None:
    shape = dict(num_compounds=3, num_outbounds=5, num_doors=4, num_products=3)
    base = generate_random_instance(seed=77, **shape)
    windowed = generate_random_instance(seed=77, tw_tightness="tight", **shape)

    solution = vaa_solution(base)
    base_result = evaluate_solution(base, solution)
    windowed_result = evaluate_solution(windowed, solution)

    assert windowed_result.makespan >= base_result.makespan
    assert windowed_result.total_tardiness >= 0.0
    assert base_result.total_tardiness == 0.0


def test_unconstrained_instance_keeps_previous_results() -> None:
    instance = generate_random_instance(
        seed=301, num_compounds=4, num_outbounds=6, num_doors=5, num_products=3
    )
    solution = vaa_solution(instance)
    result = evaluate_solution(instance, solution)

    assert result.total_tardiness == 0.0
    assert result.truck_start[instance.compound_trucks[0]] == 0.0
