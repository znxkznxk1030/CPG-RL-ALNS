from __future__ import annotations

from dataclasses import dataclass
import time

from crossdock_solver.baselines.random_baseline import BaselineRun
from crossdock_solver.core.evaluator import evaluate_solution
from crossdock_solver.core.feasibility import check_feasible
from crossdock_solver.core.solution import Solution
from crossdock_solver.data.instance import CrossDockInstance, DestinationId, DoorId, TruckId


@dataclass(frozen=True)
class RegretChoice:
    regret: float
    best_cost: float
    compound: TruckId
    destination: DestinationId


def vaa_solution(instance: CrossDockInstance) -> Solution:
    """Build the paper-style VAA constructive heuristic.

    The paper uses VAA only for compound-truck destination assignment
    with Eq. (23), then completes the heuristic with outbound-destination
    assignment, central-door assignment, FAT construction, and FT_m based
    outbound insertion.
    """

    compound_to_destination = _assign_compound_destinations_by_regret(instance)
    outbound_destination_by_truck = _assign_outbound_destinations_by_paper_priority(
        instance,
        compound_to_destination,
    )
    destination_to_truck = _destination_to_truck(
        compound_to_destination,
        outbound_destination_by_truck,
    )
    truck_to_destination = {
        **compound_to_destination,
        **outbound_destination_by_truck,
    }
    destination_priority = {
        destination: _destination_completion_priority(
            instance,
            destination,
            destination_to_truck,
            truck_to_destination,
        )
        for destination in instance.destinations
    }
    solution = Solution(
        compound_assignment={},
        outbound_assignment={},
        door_sequences={door: [] for door in instance.doors},
    )

    door_finish = _assign_first_trucks_to_doors(
        instance,
        solution,
        compound_to_destination,
        outbound_destination_by_truck,
        destination_priority,
    )
    _assign_remaining_outbounds_by_ftm(
        instance,
        solution,
        outbound_destination_by_truck,
        destination_priority,
        door_finish,
    )
    check_feasible(instance, solution)
    return solution


def run_vaa(instance: CrossDockInstance) -> BaselineRun:
    start = time.perf_counter()
    solution = vaa_solution(instance)
    result = evaluate_solution(instance, solution)
    return BaselineRun(
        name="VAA",
        solution=solution,
        result=result,
        runtime_sec=time.perf_counter() - start,
        samples=1,
    )


def vva_solution(instance: CrossDockInstance) -> Solution:
    """Alias for the common VVA typo in experiment scripts."""

    return vaa_solution(instance)


def _assign_compound_destinations_by_regret(
    instance: CrossDockInstance,
) -> dict[TruckId, DestinationId]:
    unassigned_compounds = set(instance.compound_trucks)
    available_destinations = set(instance.destinations)
    assignment: dict[TruckId, DestinationId] = {}
    cost_cache = {
        (compound, destination): _compound_destination_cost(instance, compound, destination)
        for compound in instance.compound_trucks
        for destination in instance.destinations
    }

    while unassigned_compounds:
        choices: list[RegretChoice] = []

        for compound in sorted(unassigned_compounds):
            costs = sorted(
                (
                    (cost_cache[(compound, destination)], destination)
                    for destination in available_destinations
                ),
                key=lambda item: item[0],
            )
            best_cost, best_destination = costs[0]
            second_cost = costs[1][0] if len(costs) > 1 else best_cost
            choices.append(
                RegretChoice(
                    regret=second_cost - best_cost,
                    best_cost=best_cost,
                    compound=compound,
                    destination=best_destination,
                )
            )

        for destination in sorted(available_destinations):
            costs = sorted(
                (
                    (cost_cache[(compound, destination)], compound)
                    for compound in unassigned_compounds
                ),
                key=lambda item: item[0],
            )
            best_cost, best_compound = costs[0]
            second_cost = costs[1][0] if len(costs) > 1 else best_cost
            choices.append(
                RegretChoice(
                    regret=second_cost - best_cost,
                    best_cost=best_cost,
                    compound=best_compound,
                    destination=destination,
                )
            )

        choice = max(
            choices,
            key=lambda item: (item.regret, -item.best_cost, item.compound, item.destination),
        )
        assignment[choice.compound] = choice.destination
        unassigned_compounds.remove(choice.compound)
        available_destinations.remove(choice.destination)

    return assignment


def _compound_destination_cost(
    instance: CrossDockInstance,
    compound: TruckId,
    destination: DestinationId,
) -> float:
    total_compound_handling = sum(
        instance.handling_time(compound, d)
        for d in instance.destinations
    )
    retained_handling = instance.handling_time(compound, destination)
    unload_time = total_compound_handling - retained_handling
    compound_loading_time = _destination_load(instance, destination) - retained_handling

    # Paper Eq. (23): partial unloading time of compound i plus loading
    # time of destination d initially loaded in other compound trucks.
    return unload_time + compound_loading_time


def _assign_outbound_destinations_by_paper_priority(
    instance: CrossDockInstance,
    compound_to_destination: dict[TruckId, DestinationId],
) -> dict[TruckId, DestinationId]:
    """Paper Steps 3-5: assign remaining destinations to outbound trucks."""

    assigned_destinations = set(compound_to_destination.values())
    unassigned_destinations = {
        destination for destination in instance.destinations if destination not in assigned_destinations
    }
    outbound_destination_by_truck: dict[TruckId, DestinationId] = {}

    for outbound in sorted(
        instance.outbound_trucks,
        key=lambda truck: (instance.enter_time[truck] + instance.leave_time[truck], truck),
    ):
        destination = max(
            unassigned_destinations,
            key=lambda d: (_outbound_destination_priority(instance, d, compound_to_destination), d),
        )
        outbound_destination_by_truck[outbound] = destination
        unassigned_destinations.remove(destination)

    return outbound_destination_by_truck


def _assign_first_trucks_to_doors(
    instance: CrossDockInstance,
    solution: Solution,
    compound_to_destination: dict[TruckId, DestinationId],
    outbound_destination_by_truck: dict[TruckId, DestinationId],
    destination_priority: dict[DestinationId, float],
) -> dict[DoorId, float]:
    """Paper Steps 6-10: construct FAT and assign it to central doors."""

    central_doors = sorted(
        instance.doors,
        key=lambda door: (
            sum(instance.travel(door, other) for other in instance.doors),
            door,
        ),
    )

    first_trucks = _first_assigned_trucks(
        instance,
        compound_to_destination,
        outbound_destination_by_truck,
        destination_priority,
    )

    for truck, door in zip(first_trucks, central_doors):
        if truck in instance.compound_index:
            solution.compound_assignment[truck] = (compound_to_destination[truck], door)
        else:
            solution.outbound_assignment[truck] = (outbound_destination_by_truck[truck], door)
            solution.door_sequences[door].append(truck)

    # All compound trucks are included in FAT by construction. The defensive
    # fallback keeps the solution feasible if an unusual instance has more
    # compound trucks than FAT slots.
    used_compound_doors = {door for _, door in solution.compound_assignment.values()}
    for compound in instance.compound_trucks:
        if compound in solution.compound_assignment:
            continue
        available_doors = [door for door in central_doors if door not in used_compound_doors]
        door = available_doors[0]
        solution.compound_assignment[compound] = (compound_to_destination[compound], door)
        used_compound_doors.add(door)

    return _door_finish_after_first_trucks(instance, solution)


def _assign_remaining_outbounds_by_ftm(
    instance: CrossDockInstance,
    solution: Solution,
    outbound_destination_by_truck: dict[TruckId, DestinationId],
    destination_priority: dict[DestinationId, float],
    door_finish: dict[DoorId, float],
) -> None:
    """Paper Step 11: insert remaining outbound trucks by highest T_d and lowest FT_m."""

    remaining = [
        truck for truck in instance.outbound_trucks if truck not in solution.outbound_assignment
    ]
    remaining.sort(
        key=lambda truck: (
            destination_priority[outbound_destination_by_truck[truck]],
            outbound_destination_by_truck[truck],
            truck,
        ),
        reverse=True,
    )

    for truck in remaining:
        destination = outbound_destination_by_truck[truck]
        door = min(instance.doors, key=lambda m: (door_finish[m], m))
        solution.outbound_assignment[truck] = (destination, door)
        solution.door_sequences[door].append(truck)
        door_finish[door] = _outbound_finish_on_door(
            instance,
            solution,
            truck,
            door,
            previous_finish=door_finish[door],
        )


def _first_assigned_trucks(
    instance: CrossDockInstance,
    compound_to_destination: dict[TruckId, DestinationId],
    outbound_destination_by_truck: dict[TruckId, DestinationId],
    destination_priority: dict[DestinationId, float],
) -> list[TruckId]:
    """Paper Step 7: FAT contains all compounds plus longest outbound jobs if needed."""

    first_trucks: list[TruckId] = list(instance.compound_trucks)
    remaining_slots = max(0, min(len(instance.doors), len(instance.all_trucks)) - len(first_trucks))
    if remaining_slots > 0:
        outbound_by_loading = sorted(
            instance.outbound_trucks,
            key=lambda truck: (
                _destination_load(instance, outbound_destination_by_truck[truck]),
                outbound_destination_by_truck[truck],
                truck,
            ),
            reverse=True,
        )
        first_trucks.extend(outbound_by_loading[:remaining_slots])

    destination_by_truck = {
        **compound_to_destination,
        **outbound_destination_by_truck,
    }
    return sorted(
        first_trucks,
        key=lambda truck: (
            destination_priority[destination_by_truck[truck]],
            destination_by_truck[truck],
            truck,
        ),
        reverse=True,
    )


def _door_finish_after_first_trucks(
    instance: CrossDockInstance,
    solution: Solution,
) -> dict[DoorId, float]:
    door_finish = {door: 0.0 for door in instance.doors}

    for compound, (_, door) in solution.compound_assignment.items():
        door_finish[door] = _compound_finish(instance, solution, compound)

    for outbound, (_, door) in solution.outbound_assignment.items():
        door_finish[door] = _outbound_finish_on_door(
            instance,
            solution,
            outbound,
            door,
            previous_finish=door_finish[door],
        )

    return door_finish


def _destination_to_truck(
    compound_to_destination: dict[TruckId, DestinationId],
    outbound_destination_by_truck: dict[TruckId, DestinationId],
) -> dict[DestinationId, TruckId]:
    carriers: dict[DestinationId, TruckId] = {}
    for truck, destination in compound_to_destination.items():
        carriers[destination] = truck
    for truck, destination in outbound_destination_by_truck.items():
        carriers[destination] = truck
    return carriers


def _destination_completion_priority(
    instance: CrossDockInstance,
    destination: DestinationId,
    destination_to_truck: dict[DestinationId, TruckId],
    truck_to_destination: dict[TruckId, DestinationId],
) -> float:
    carrier = destination_to_truck[destination]
    if carrier in instance.compound_index:
        ready = max(
            [_compound_unload_time(instance, carrier, truck_to_destination[carrier])]
            + [
                _compound_unload_time(instance, source, truck_to_destination[source])
                for source in instance.compound_trucks
                if source != carrier and instance.unit_amount(source, destination) > 0
            ],
            default=0.0,
        )
        load = sum(
            instance.handling_time(source, destination)
            for source in instance.compound_trucks
            if source != carrier
        )
        return ready + load

    compound_to_destination = {
        truck: truck_to_destination[truck]
        for truck in instance.compound_trucks
    }
    return _outbound_destination_priority(instance, destination, compound_to_destination)


def _outbound_destination_priority(
    instance: CrossDockInstance,
    destination: DestinationId,
    compound_to_destination: dict[TruckId, DestinationId],
) -> float:
    ready = max(
        (
            _compound_unload_time(instance, source, compound_to_destination[source])
            for source in instance.compound_trucks
            if instance.unit_amount(source, destination) > 0
        ),
        default=0.0,
    )
    return ready + _destination_load(instance, destination)


def _compound_finish(
    instance: CrossDockInstance,
    solution: Solution,
    compound: TruckId,
) -> float:
    destination, target_door = solution.compound_assignment[compound]
    own_unload_finish = (
        instance.enter_time[compound]
        + _compound_unload_time(instance, compound, destination)
    )
    destination_ready = _destination_ready_at_door(instance, solution, destination, target_door, carrier=compound)
    load_time = sum(
        instance.handling_time(source, destination)
        for source in instance.compound_trucks
        if source != compound
    )
    return max(own_unload_finish, destination_ready) + load_time + instance.leave_time[compound]


def _outbound_finish_on_door(
    instance: CrossDockInstance,
    solution: Solution,
    outbound: TruckId,
    door: DoorId,
    *,
    previous_finish: float,
) -> float:
    destination = solution.outbound_assignment[outbound][0]
    destination_ready = _destination_ready_at_door(instance, solution, destination, door, carrier=outbound)
    load_time = _destination_load(instance, destination)
    start = max(previous_finish, destination_ready)
    return start + instance.enter_time[outbound] + load_time + instance.leave_time[outbound]


def _destination_ready_at_door(
    instance: CrossDockInstance,
    solution: Solution,
    destination: DestinationId,
    target_door: DoorId,
    *,
    carrier: TruckId,
) -> float:
    ready = 0.0
    for source in instance.compound_trucks:
        if source == carrier:
            continue
        if instance.unit_amount(source, destination) <= 0:
            continue
        source_door = solution.compound_assignment[source][1]
        retained_destination = solution.compound_assignment[source][0]
        source_unload_finish = (
            instance.enter_time[source]
            + _compound_unload_time(instance, source, retained_destination)
        )
        ready = max(ready, source_unload_finish + instance.travel(source_door, target_door))
    return ready


def _destination_load(instance: CrossDockInstance, destination: DestinationId) -> float:
    return sum(
        instance.handling_time(compound, destination)
        for compound in instance.compound_trucks
    )


def _compound_unload_time(
    instance: CrossDockInstance,
    compound: TruckId,
    retained_destination: DestinationId,
) -> float:
    return sum(
        instance.handling_time(compound, destination)
        for destination in instance.destinations
        if destination != retained_destination
    )
