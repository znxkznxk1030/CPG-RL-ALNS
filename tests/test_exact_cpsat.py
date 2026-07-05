from __future__ import annotations

import pytest

from crossdock_solver.core.feasibility import check_feasible
from crossdock_solver.data.generator import generate_random_instance
from crossdock_solver.exact.cpsat import ExactCPSATConfig, solve_exact_cpsat
from crossdock_solver.exact.milp import ExactMILPConfig, solve_exact_milp
from tests.conftest import make_toy_instance


def test_cpsat_solves_toy_instance() -> None:
    instance = make_toy_instance()

    result = solve_exact_cpsat(instance, ExactCPSATConfig(time_limit_sec=30))

    assert result.proven_optimal
    assert result.run is not None
    check_feasible(instance, result.run.solution)
    assert result.run.result.makespan == 24.0
    assert result.lower_bound == pytest.approx(24.0, abs=1e-6)


def test_cpsat_matches_milp_optimum() -> None:
    instance = generate_random_instance(
        seed=55, num_compounds=2, num_outbounds=3, num_doors=3, num_products=2
    )

    cpsat = solve_exact_cpsat(instance, ExactCPSATConfig(time_limit_sec=60))
    milp = solve_exact_milp(instance, ExactMILPConfig(time_limit_sec=120))

    assert cpsat.proven_optimal
    assert milp.status == "Optimal"
    assert cpsat.objective_value == pytest.approx(milp.run.result.makespan, abs=1e-4)


def test_cpsat_objective_matches_evaluator_with_time_windows() -> None:
    instance = generate_random_instance(
        seed=55,
        num_compounds=2,
        num_outbounds=3,
        num_doors=3,
        num_products=2,
        tw_tightness="tight",
    )
    weight = 1.0

    result = solve_exact_cpsat(
        instance, ExactCPSATConfig(time_limit_sec=60, tardiness_weight=weight)
    )

    assert result.proven_optimal
    assert result.run is not None
    check_feasible(instance, result.run.solution)

    evaluator_objective = (
        result.run.result.makespan + weight * result.run.result.total_tardiness
    )
    assert result.objective_value == pytest.approx(evaluator_objective, abs=1e-4)
    assert result.lower_bound == pytest.approx(evaluator_objective, abs=1e-4)


def test_cpsat_lower_bound_never_exceeds_incumbent() -> None:
    instance = generate_random_instance(
        seed=7, num_compounds=3, num_outbounds=3, num_doors=4, num_products=3
    )

    result = solve_exact_cpsat(instance, ExactCPSATConfig(time_limit_sec=30))

    assert result.run is not None
    assert result.lower_bound <= result.objective_value + 1e-6
