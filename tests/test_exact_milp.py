from __future__ import annotations

from crossdock_solver.core.feasibility import check_feasible
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

