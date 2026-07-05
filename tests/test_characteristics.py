from __future__ import annotations

import numpy as np
import pytest

from crossdock_solver.data.characteristics import (
    compound_fraction,
    dbpr,
    mean_dbpr,
    paper_time_budget,
)
from crossdock_solver.data.generator import generate_random_instance


def _paper_size(num_compounds, num_destinations, num_doors, **kw):
    return generate_random_instance(
        num_compounds=num_compounds,
        num_outbounds=num_destinations - num_compounds,
        num_doors=num_doors,
        param_profile="paper",
        **kw,
    )


def test_paper_time_budget_matches_published_values() -> None:
    # Shahmardan & Sajadieh (2020) Table 5: (I, D, M) -> seconds.
    assert paper_time_budget(_paper_size(8, 10, 8)) == pytest.approx(50.4)
    assert paper_time_budget(_paper_size(20, 30, 20)) == pytest.approx(350.0)
    assert paper_time_budget(_paper_size(2, 3, 2)) == pytest.approx(3.5)


def test_paper_profile_uses_i_shape_travel_and_wider_ranges() -> None:
    instance = _paper_size(3, 6, 4, seed=1)

    assert instance.travel(1, 4) == pytest.approx(3.0)
    assert instance.travel(2, 2) == pytest.approx(0.0)
    assert max(instance.enter_time.values()) > 5.0


def test_dbpr_is_high_for_clustered_and_bounded() -> None:
    clustered = generate_random_instance(
        num_compounds=4, num_outbounds=8, num_doors=6, seed=5, flow_pattern="clustered"
    )
    uniform = generate_random_instance(
        num_compounds=4, num_outbounds=8, num_doors=6, seed=5, flow_pattern="uniform"
    )

    for value in dbpr(clustered).values():
        assert 0.0 <= value <= 1.0
    assert mean_dbpr(clustered) > mean_dbpr(uniform)


def test_compound_fraction() -> None:
    instance = generate_random_instance(num_compounds=8, num_outbounds=2, num_doors=8, seed=1)
    assert compound_fraction(instance) == pytest.approx(0.8)


def test_invalid_param_profile_raises() -> None:
    with pytest.raises(ValueError):
        generate_random_instance(
            num_compounds=2, num_outbounds=3, num_doors=3, param_profile="real", seed=1
        )
