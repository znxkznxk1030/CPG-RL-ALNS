from __future__ import annotations

from dataclasses import dataclass
import math
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
    """Build a deterministic VAA-style constructive baseline.

    This is a MVP-compatible adaptation of Vogel's Approximation Algorithm. It
    uses row/column regrets over compound-destination costs to decide which
    destinations are retained by compound trucks, then greedily assigns doors
    and outbound sequences.
    """

    compound_to_destination = _assign_compound_destinations_by_regret(instance)
    compound_assignment = _assign_compound_doors(instance, compound_to_destination)

    assigned_destinations = set(compound_to_destination.values())
    outbound_destinations = [
        destination
        for destination in sorted(instance.destinations, key=lambda d: _destination_load(instance, d), reverse=True)
        if destination not in assigned_destinations
    ]
    outbound_trucks = sorted(
        instance.outbound_trucks,
        key=lambda truck: instance.enter_time[truck] + instance.leave_time[truck],
    )
    outbound_destination_by_truck = {
        truck: destination
        for truck, destination in zip(outbound_trucks, outbound_destinations, strict=True)
    }

    solution = Solution(
        compound_assignment=compound_assignment,
        outbound_assignment={},
        door_sequences={door: [] for door in instance.doors},
    )
    _insert_outbounds_greedily(instance, solution, outbound_destination_by_truck)
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

    while unassigned_compounds:
        choices: list[RegretChoice] = []

        for compound in sorted(unassigned_compounds):
            costs = sorted(
                (
                    (_compound_destination_cost(instance, compound, destination), destination)
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
                    (_compound_destination_cost(instance, compound, destination), compound)
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

    # Lower is better. The first term rewards retaining high own-destination
    # volume; the second avoids assigning a compound to a destination with heavy
    # inbound loading from other compounds.
    return unload_time + 0.5 * compound_loading_time


def _assign_compound_doors(
    instance: CrossDockInstance,
    compound_to_destination: dict[TruckId, DestinationId],
) -> dict[TruckId, tuple[DestinationId, DoorId]]:
    central_doors = sorted(
        instance.doors,
        key=lambda door: (
            sum(instance.travel(door, other) for other in instance.doors),
            door,
        ),
    )
    compounds_by_workload = sorted(
        compound_to_destination,
        key=lambda compound: _compound_workload(instance, compound, compound_to_destination[compound]),
        reverse=True,
    )

    assignment: dict[TruckId, tuple[DestinationId, DoorId]] = {}
    for compound, door in zip(compounds_by_workload, central_doors[: len(compounds_by_workload)], strict=True):
        assignment[compound] = (compound_to_destination[compound], door)
    return assignment


def _insert_outbounds_greedily(
    instance: CrossDockInstance,
    solution: Solution,
    destination_by_truck: dict[TruckId, DestinationId],
) -> None:
    ordered_trucks = sorted(
        destination_by_truck,
        key=lambda truck: (
            _destination_load(instance, destination_by_truck[truck])
            + instance.enter_time[truck]
            + instance.leave_time[truck]
        ),
        reverse=True,
    )

    for truck in ordered_trucks:
        destination = destination_by_truck[truck]
        best_solution: Solution | None = None
        best_makespan = math.inf

        for door in instance.doors:
            sequence = solution.door_sequences.get(door, [])
            for position in range(len(sequence) + 1):
                candidate = solution.copy()
                candidate.outbound_assignment[truck] = (destination, door)
                candidate.door_sequences.setdefault(door, [])
                candidate.door_sequences[door].insert(position, truck)

                completed = _complete_remaining_outbounds(instance, candidate, destination_by_truck)
                makespan = evaluate_solution(instance, completed).makespan
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_solution = candidate

        if best_solution is None:
            raise RuntimeError(f"failed to insert outbound truck {truck}")
        solution.outbound_assignment = best_solution.outbound_assignment
        solution.door_sequences = best_solution.door_sequences


def _complete_remaining_outbounds(
    instance: CrossDockInstance,
    partial: Solution,
    destination_by_truck: dict[TruckId, DestinationId],
) -> Solution:
    completed = partial.copy()
    for truck in sorted(destination_by_truck):
        if truck in completed.outbound_assignment:
            continue
        destination = destination_by_truck[truck]
        door = min(
            instance.doors,
            key=lambda d: (
                len(completed.door_sequences.get(d, [])),
                sum(
                    instance.enter_time[t]
                    + _destination_load(instance, completed.outbound_assignment[t][0])
                    + instance.leave_time[t]
                    for t in completed.door_sequences.get(d, [])
                ),
                d,
            ),
        )
        completed.outbound_assignment[truck] = (destination, door)
        completed.door_sequences.setdefault(door, []).append(truck)
    return completed


def _destination_load(instance: CrossDockInstance, destination: DestinationId) -> float:
    return sum(
        instance.handling_time(compound, destination)
        for compound in instance.compound_trucks
    )


def _compound_workload(
    instance: CrossDockInstance,
    compound: TruckId,
    retained_destination: DestinationId,
) -> float:
    unload_time = sum(
        instance.handling_time(compound, destination)
        for destination in instance.destinations
        if destination != retained_destination
    )
    load_time = sum(
        instance.handling_time(source, retained_destination)
        for source in instance.compound_trucks
        if source != compound
    )
    return unload_time + load_time + instance.enter_time[compound] + instance.leave_time[compound]
