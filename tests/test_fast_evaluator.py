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
        instance = generate_random_instance(seed=1000 + shape_seed, **shape)
        fast = FastEvaluator(instance)
        rng = random.Random(2000 + shape_seed)

        for _ in range(5):
            solution = random_feasible_solution(instance, rng)
            _assert_equivalent(instance, solution, fast)

            for _ in range(20):
                operator = rng.choice(operators)
                solution = operator(instance, solution, rng)
                _assert_equivalent(instance, solution, fast)
