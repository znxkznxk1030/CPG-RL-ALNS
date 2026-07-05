from __future__ import annotations

import pytest

from crossdock_solver.core.feasibility import check_feasible
from crossdock_solver.data.generator import generate_random_instance
from crossdock_solver.exact.milp import ExactMILPConfig, solve_exact_milp
from tests.conftest import make_toy_instance


def test_exact_milp_solves_toy_instance() -> None:
    instance = make_toy_instance()

    result = solve_exact_milp(
        instance,
        ExactMILPConfig(time_limit_sec=30, msg=False),
    )

    assert result.status == "Optimal"
    assert result.proven_optimal
    assert result.run is not None
    check_feasible(instance, result.run.solution)
    assert result.run.result.makespan == 24.0


def test_exact_milp_objective_matches_evaluator_with_time_windows() -> None:
    instance = generate_random_instance(
        seed=55,
        num_compounds=2,
        num_outbounds=3,
        num_doors=3,
        num_products=2,
        tw_tightness="tight",
    )
    weight = 1.0

    result = solve_exact_milp(
        instance,
        ExactMILPConfig(time_limit_sec=60, msg=False, tardiness_weight=weight),
    )

    assert result.status == "Optimal"
    assert result.run is not None
    check_feasible(instance, result.run.solution)

    evaluator_objective = (
        result.run.result.makespan + weight * result.run.result.total_tardiness
    )
    assert result.objective_bound == pytest.approx(evaluator_objective, abs=1e-4)

