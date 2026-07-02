from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import time

import pulp

from crossdock_solver.baselines.random_baseline import BaselineRun
from crossdock_solver.core.evaluator import evaluate_solution
from crossdock_solver.core.solution import Solution
from crossdock_solver.data.instance import CrossDockInstance, DestinationId, DoorId, TruckId


@dataclass(frozen=True)
class ExactMILPConfig:
    time_limit_sec: int | None = 120
    msg: bool = False
    name: str = "Exact-MILP"


@dataclass(frozen=True)
class ExactMILPResult:
    run: BaselineRun | None
    status: str
    proven_optimal: bool
    objective_bound: float | None
    runtime_sec: float


def solve_exact_milp(
    instance: CrossDockInstance,
    config: ExactMILPConfig | None = None,
) -> ExactMILPResult:
    """Solve the current MVP scheduling model exactly with PuLP/CBC.

    The formulation targets the repository's MVP evaluator, not the full paper
    MILP verbatim. It optimizes the same decisions represented by `Solution`:
    destination-carrier assignment, compound door assignment, outbound door
    assignment, and outbound sequencing on each door.
    """

    config = config or ExactMILPConfig()
    start_time = time.perf_counter()

    compounds = tuple(instance.compound_trucks)
    outbounds = tuple(instance.outbound_trucks)
    destinations = tuple(instance.destinations)
    doors = tuple(instance.doors)

    handling = {
        (compound, destination): instance.handling_time(compound, destination)
        for compound in compounds
        for destination in destinations
    }
    total_handling = {
        compound: sum(handling[(compound, destination)] for destination in destinations)
        for compound in compounds
    }
    compound_load = {
        (compound, destination): sum(
            handling[(source, destination)]
            for source in compounds
            if source != compound
        )
        for compound in compounds
        for destination in destinations
    }
    outbound_load = {
        destination: sum(handling[(source, destination)] for source in compounds)
        for destination in destinations
    }
    big_m = _big_m(instance)

    model = pulp.LpProblem("crossdock_mvp_exact", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "x_compound",
        (compounds, destinations, doors),
        lowBound=0,
        upBound=1,
        cat="Binary",
    )
    z = pulp.LpVariable.dicts(
        "z_outbound",
        (outbounds, destinations, doors),
        lowBound=0,
        upBound=1,
        cat="Binary",
    )
    before = pulp.LpVariable.dicts(
        "before",
        (outbounds, outbounds, doors),
        lowBound=0,
        upBound=1,
        cat="Binary",
    )
    unload_finish = pulp.LpVariable.dicts("unload_finish", list(compounds), lowBound=0)
    compound_finish = pulp.LpVariable.dicts("compound_finish", list(compounds), lowBound=0)
    outbound_start = pulp.LpVariable.dicts("outbound_start", list(outbounds), lowBound=0)
    outbound_finish = pulp.LpVariable.dicts("outbound_finish", list(outbounds), lowBound=0)
    cmax = pulp.LpVariable("makespan", lowBound=0)

    model += cmax

    compound_door = {
        (compound, door): pulp.lpSum(x[compound][destination][door] for destination in destinations)
        for compound in compounds
        for door in doors
    }
    outbound_door = {
        (outbound, door): pulp.lpSum(z[outbound][destination][door] for destination in destinations)
        for outbound in outbounds
        for door in doors
    }
    outbound_destination = {
        (outbound, destination): pulp.lpSum(z[outbound][destination][door] for door in doors)
        for outbound in outbounds
        for destination in destinations
    }

    for compound in compounds:
        model += (
            pulp.lpSum(
                x[compound][destination][door]
                for destination in destinations
                for door in doors
            )
            == 1
        )
        model += unload_finish[compound] == (
            instance.enter_time[compound]
            + total_handling[compound]
            - pulp.lpSum(
                handling[(compound, destination)] * x[compound][destination][door]
                for destination in destinations
                for door in doors
            )
        )
        model += cmax >= compound_finish[compound]

    for door in doors:
        model += (
            pulp.lpSum(
                x[compound][destination][door]
                for compound in compounds
                for destination in destinations
            )
            <= 1
        )

    for outbound in outbounds:
        model += (
            pulp.lpSum(
                z[outbound][destination][door]
                for destination in destinations
                for door in doors
            )
            == 1
        )
        model += cmax >= outbound_finish[outbound]

    for destination in destinations:
        model += (
            pulp.lpSum(
                x[compound][destination][door]
                for compound in compounds
                for door in doors
            )
            + pulp.lpSum(
                z[outbound][destination][door]
                for outbound in outbounds
                for door in doors
            )
            == 1
        )

    for compound in compounds:
        for destination in destinations:
            for door in doors:
                model += compound_finish[compound] >= (
                    unload_finish[compound]
                    + compound_load[(compound, destination)]
                    + instance.leave_time[compound]
                    - big_m * (1 - x[compound][destination][door])
                )

                for source in compounds:
                    if source == compound or instance.unit_amount(source, destination) <= 0:
                        continue
                    for source_door in doors:
                        model += compound_finish[compound] >= (
                            unload_finish[source]
                            + instance.travel(source_door, door)
                            + compound_load[(compound, destination)]
                            + instance.leave_time[compound]
                            - big_m
                            * (
                                2
                                - x[compound][destination][door]
                                - compound_door[(source, source_door)]
                            )
                        )

    for outbound in outbounds:
        for destination in destinations:
            model += outbound_finish[outbound] >= (
                outbound_start[outbound]
                + instance.enter_time[outbound]
                + outbound_load[destination]
                + instance.leave_time[outbound]
                - big_m * (1 - outbound_destination[(outbound, destination)])
            )

            for door in doors:
                for source in compounds:
                    if instance.unit_amount(source, destination) <= 0:
                        continue
                    for source_door in doors:
                        model += outbound_start[outbound] >= (
                            unload_finish[source]
                            + instance.travel(source_door, door)
                            - big_m
                            * (
                                2
                                - z[outbound][destination][door]
                                - compound_door[(source, source_door)]
                            )
                        )

        for door in doors:
            for compound in compounds:
                model += outbound_start[outbound] >= (
                    compound_finish[compound]
                    - big_m
                    * (
                        2
                        - outbound_door[(outbound, door)]
                        - compound_door[(compound, door)]
                    )
                )

    for first in outbounds:
        for second in outbounds:
            if first == second:
                for door in doors:
                    model += before[first][second][door] == 0
                continue
            for door in doors:
                model += before[first][second][door] <= outbound_door[(first, door)]
                model += before[first][second][door] <= outbound_door[(second, door)]
                model += before[first][second][door] + before[second][first][door] >= (
                    outbound_door[(first, door)] + outbound_door[(second, door)] - 1
                )
                model += outbound_start[second] >= (
                    outbound_finish[first]
                    - big_m * (1 - before[first][second][door])
                )

    with tempfile.NamedTemporaryFile(prefix="crossdock_cbc_", suffix=".log", delete=False) as log_file:
        log_path = Path(log_file.name)

    solver = pulp.PULP_CBC_CMD(
        msg=config.msg,
        timeLimit=config.time_limit_sec,
        gapRel=0.0,
        logPath=str(log_path),
    )
    status_code = model.solve(solver)
    runtime_sec = time.perf_counter() - start_time
    status = pulp.LpStatus[status_code]
    objective_bound = pulp.value(model.objective)
    proven_optimal = _cbc_proven_optimal(log_path)
    try:
        log_path.unlink()
    except OSError:
        pass

    if status != "Optimal":
        return ExactMILPResult(
            run=None,
            status=status,
            proven_optimal=proven_optimal,
            objective_bound=objective_bound,
            runtime_sec=runtime_sec,
        )

    solution = _extract_solution(instance, x, z, outbound_start)
    result = evaluate_solution(instance, solution)
    return ExactMILPResult(
        run=BaselineRun(
            name=config.name,
            solution=solution,
            result=result,
            runtime_sec=runtime_sec,
            samples=1,
        ),
        status=status,
        proven_optimal=proven_optimal,
        objective_bound=objective_bound,
        runtime_sec=runtime_sec,
    )


def _extract_solution(
    instance: CrossDockInstance,
    x: dict[TruckId, dict[DestinationId, dict[DoorId, pulp.LpVariable]]],
    z: dict[TruckId, dict[DestinationId, dict[DoorId, pulp.LpVariable]]],
    outbound_start: dict[TruckId, pulp.LpVariable],
) -> Solution:
    compound_assignment: dict[TruckId, tuple[DestinationId, DoorId]] = {}
    outbound_assignment: dict[TruckId, tuple[DestinationId, DoorId]] = {}
    door_sequences: dict[DoorId, list[TruckId]] = {door: [] for door in instance.doors}

    for compound in instance.compound_trucks:
        destination, door = max(
            (
                (destination, door)
                for destination in instance.destinations
                for door in instance.doors
            ),
            key=lambda item: pulp.value(x[compound][item[0]][item[1]]) or 0.0,
        )
        compound_assignment[compound] = (destination, door)

    for outbound in instance.outbound_trucks:
        destination, door = max(
            (
                (destination, door)
                for destination in instance.destinations
                for door in instance.doors
            ),
            key=lambda item: pulp.value(z[outbound][item[0]][item[1]]) or 0.0,
        )
        outbound_assignment[outbound] = (destination, door)
        door_sequences[door].append(outbound)

    for door, sequence in door_sequences.items():
        sequence.sort(
            key=lambda outbound: (
                pulp.value(outbound_start[outbound]) or 0.0,
                outbound,
            )
        )

    return Solution(
        compound_assignment=compound_assignment,
        outbound_assignment=outbound_assignment,
        door_sequences=door_sequences,
    )


def _big_m(instance: CrossDockInstance) -> float:
    total_handling = sum(
        instance.handling_time(compound, destination)
        for compound in instance.compound_trucks
        for destination in instance.destinations
    )
    max_enter_leave = sum(instance.enter_time.values()) + sum(instance.leave_time.values())
    max_travel = max(
        instance.travel(source, target)
        for source in instance.doors
        for target in instance.doors
    )
    return 10.0 + max_enter_leave + 3.0 * total_handling + len(instance.destinations) * max_travel


def _cbc_proven_optimal(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "Result - Optimal solution found" in text
