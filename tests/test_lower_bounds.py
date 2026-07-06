from __future__ import annotations

import pytest

from crossdock_solver.data.generator import generate_random_instance
from crossdock_solver.exact.cpsat import ExactCPSATConfig, solve_exact_cpsat
from crossdock_solver.exact.lower_bounds import combinatorial_lower_bound
from tests.conftest import make_toy_instance


def test_lower_bound_positive_on_toy() -> None:
    assert combinatorial_lower_bound(make_toy_instance()) > 0.0


def test_lower_bound_never_exceeds_optimum() -> None:
    # A valid lower bound must not exceed the proven optimal makespan.
    instances = [make_toy_instance()]
    for seed in (1, 2, 3):
        instances.append(
            generate_random_instance(
                seed=seed, num_compounds=2, num_outbounds=3, num_doors=3, num_products=2
            )
        )

    for instance in instances:
        result = solve_exact_cpsat(instance, ExactCPSATConfig(time_limit_sec=60))
        assert result.proven_optimal
        assert combinatorial_lower_bound(instance) <= result.objective_value + 1e-6


def test_objective_lower_bound_never_exceeds_optimum_with_tw() -> None:
    from crossdock_solver.exact.lower_bounds import combinatorial_objective_lower_bound

    for seed in (1, 2):
        instance = generate_random_instance(
            seed=seed, num_compounds=2, num_outbounds=3, num_doors=3,
            num_products=2, tw_tightness="tight",
        )
        result = solve_exact_cpsat(
            instance, ExactCPSATConfig(time_limit_sec=60, tardiness_weight=1.0)
        )
        assert result.proven_optimal
        bound = combinatorial_objective_lower_bound(instance, 1.0)
        assert bound <= result.objective_value + 1e-6
        assert bound >= combinatorial_lower_bound(instance) - 1e-9


def test_lower_bound_respects_time_windows() -> None:
    base = generate_random_instance(
        seed=9, num_compounds=3, num_outbounds=4, num_doors=4, num_products=3
    )
    windowed = generate_random_instance(
        seed=9, num_compounds=3, num_outbounds=4, num_doors=4, num_products=3,
        tw_tightness="tight",
    )
    # Non-zero release times only push the critical chain later.
    assert combinatorial_lower_bound(windowed) >= combinatorial_lower_bound(base) - 1e-9
