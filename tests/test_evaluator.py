from __future__ import annotations

import pytest

from crossdock_solver.core.evaluator import evaluate_solution
from crossdock_solver.core.feasibility import InfeasibleSolution, check_feasible
from tests.conftest import make_toy_instance, make_toy_solution


def test_evaluator_respects_partial_unloading_and_sequences() -> None:
    instance = make_toy_instance()
    solution = make_toy_solution()

    result = evaluate_solution(instance, solution)

    assert result.metadata["compound_unload_finish"] == {
        "C1": pytest.approx(8.0),
        "C2": pytest.approx(12.0),
    }
    assert result.metadata["destination_ready"] == {
        "D1": pytest.approx(22.0),
        "D2": pytest.approx(18.0),
        "D3": pytest.approx(28.0),
        "D4": pytest.approx(17.0),
    }
    assert result.truck_finish["C1"] == pytest.approx(26.0)
    assert result.truck_finish["C2"] == pytest.approx(24.0)
    assert result.truck_finish["O1"] == pytest.approx(32.0)
    assert result.truck_finish["O2"] == pytest.approx(42.0)
    assert result.makespan == pytest.approx(42.0)
    assert result.critical_door == 3
    assert result.critical_truck == "O2"
    assert result.total_transfer_time == pytest.approx(160.0)
    assert result.max_transfer_edge_time == pytest.approx(50.0)


def test_feasibility_rejects_duplicate_destination() -> None:
    instance = make_toy_instance()
    solution = make_toy_solution()
    solution.outbound_assignment["O2"] = ("D3", 3)

    with pytest.raises(InfeasibleSolution):
        check_feasible(instance, solution)


def test_feasibility_rejects_compound_door_overload() -> None:
    instance = make_toy_instance()
    solution = make_toy_solution()
    solution.compound_assignment["C2"] = ("D2", 1)

    with pytest.raises(InfeasibleSolution):
        check_feasible(instance, solution)

