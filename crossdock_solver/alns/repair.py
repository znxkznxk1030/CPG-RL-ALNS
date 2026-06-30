from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable

from crossdock_solver.alns.destroy import DestroyResult
from crossdock_solver.core.evaluator import evaluate_solution
from crossdock_solver.core.feasibility import check_feasible
from crossdock_solver.core.solution import Solution
from crossdock_solver.data.instance import CrossDockInstance, DestinationId, DoorId, TruckId


@dataclass
class InsertionOption:
    truck: TruckId
    destination: DestinationId
    door: DoorId
    sequence_position: int | None
    score: float
    solution: Solution


def greedy_repair(
    instance: CrossDockInstance,
    destroyed: DestroyResult,
    *,
    rng: random.Random | None = None,
) -> Solution:
    rng = rng or random.Random()
    solution = destroyed.partial_solution.copy()
    unassigned_compounds = set(destroyed.removed_compounds)
    unassigned_outbounds = set(destroyed.removed_outbounds)
    unassigned_destinations = set(destroyed.removed_destinations)

    while unassigned_compounds or unassigned_outbounds:
        options = _scored_insertion_options(
            instance,
            solution,
            unassigned_compounds,
            unassigned_outbounds,
            unassigned_destinations,
            rng=rng,
        )
        if not options:
            raise RuntimeError("no feasible insertion option found during greedy repair")

        best = min(options, key=lambda option: option.score)
        solution = best.solution
        _mark_assigned(best, unassigned_compounds, unassigned_outbounds, unassigned_destinations)

    check_feasible(instance, solution)
    return solution


def regret_k_repair(
    instance: CrossDockInstance,
    destroyed: DestroyResult,
    *,
    k: int = 2,
    rng: random.Random | None = None,
) -> Solution:
    rng = rng or random.Random()
    if k < 2:
        raise ValueError("k must be at least 2 for regret-k repair")

    solution = destroyed.partial_solution.copy()
    unassigned_compounds = set(destroyed.removed_compounds)
    unassigned_outbounds = set(destroyed.removed_outbounds)
    unassigned_destinations = set(destroyed.removed_destinations)

    while unassigned_compounds or unassigned_outbounds:
        options_by_truck: dict[TruckId, list[InsertionOption]] = {}
        for option in _scored_insertion_options(
            instance,
            solution,
            unassigned_compounds,
            unassigned_outbounds,
            unassigned_destinations,
            rng=rng,
        ):
            options_by_truck.setdefault(option.truck, []).append(option)

        if not options_by_truck:
            raise RuntimeError("no feasible insertion option found during regret repair")

        best_choice: InsertionOption | None = None
        best_regret = -math.inf
        for options in options_by_truck.values():
            options.sort(key=lambda option: option.score)
            best = options[0]
            kth = options[min(k - 1, len(options) - 1)]
            regret = kth.score - best.score
            if regret > best_regret or (
                math.isclose(regret, best_regret) and best_choice is not None and best.score < best_choice.score
            ):
                best_regret = regret
                best_choice = best

        if best_choice is None:
            raise RuntimeError("regret repair failed to select an insertion")

        solution = best_choice.solution
        _mark_assigned(best_choice, unassigned_compounds, unassigned_outbounds, unassigned_destinations)

    check_feasible(instance, solution)
    return solution


def _scored_insertion_options(
    instance: CrossDockInstance,
    solution: Solution,
    unassigned_compounds: set[TruckId],
    unassigned_outbounds: set[TruckId],
    unassigned_destinations: set[DestinationId],
    *,
    rng: random.Random,
) -> list[InsertionOption]:
    options: list[InsertionOption] = []
    trucks = [*sorted(unassigned_compounds), *sorted(unassigned_outbounds)]
    destinations = sorted(unassigned_destinations)
    rng.shuffle(trucks)
    rng.shuffle(destinations)

    for truck in trucks:
        for destination in destinations:
            for candidate in _raw_insertion_candidates(instance, solution, truck, destination):
                remaining_compounds = set(unassigned_compounds)
                remaining_outbounds = set(unassigned_outbounds)
                remaining_destinations = set(unassigned_destinations)
                if truck in remaining_compounds:
                    remaining_compounds.remove(truck)
                else:
                    remaining_outbounds.remove(truck)
                remaining_destinations.remove(destination)

                completed = _complete_partial_solution(
                    instance,
                    candidate,
                    remaining_compounds,
                    remaining_outbounds,
                    remaining_destinations,
                )
                score = evaluate_solution(instance, completed).makespan
                options.append(
                    InsertionOption(
                        truck=truck,
                        destination=destination,
                        door=candidate.truck_door(truck),
                        sequence_position=_sequence_position(candidate, truck),
                        score=score,
                        solution=candidate,
                    )
                )

    return options


def _raw_insertion_candidates(
    instance: CrossDockInstance,
    solution: Solution,
    truck: TruckId,
    destination: DestinationId,
) -> Iterable[Solution]:
    if truck in instance.compound_index:
        used_compound_doors = {door for _, door in solution.compound_assignment.values()}
        for door in instance.doors:
            if door in used_compound_doors:
                continue
            candidate = solution.copy()
            candidate.compound_assignment[truck] = (destination, door)
            yield candidate
        return

    for door in instance.doors:
        sequence = solution.door_sequences.get(door, [])
        for position in range(len(sequence) + 1):
            candidate = solution.copy()
            candidate.outbound_assignment[truck] = (destination, door)
            candidate.door_sequences.setdefault(door, [])
            candidate.door_sequences[door].insert(position, truck)
            yield candidate


def _complete_partial_solution(
    instance: CrossDockInstance,
    solution: Solution,
    unassigned_compounds: set[TruckId],
    unassigned_outbounds: set[TruckId],
    unassigned_destinations: set[DestinationId],
) -> Solution:
    completed = solution.copy()
    destinations = sorted(unassigned_destinations)
    trucks = [*sorted(unassigned_compounds), *sorted(unassigned_outbounds)]

    if len(destinations) != len(trucks):
        raise RuntimeError("partial solution has mismatched truck and destination counts")

    for truck, destination in zip(trucks, destinations, strict=True):
        if truck in unassigned_compounds:
            used_compound_doors = {door for _, door in completed.compound_assignment.values()}
            available_doors = [door for door in instance.doors if door not in used_compound_doors]
            if not available_doors:
                raise RuntimeError("no available compound door while completing partial solution")
            completed.compound_assignment[truck] = (destination, available_doors[0])
        else:
            best_door = min(
                instance.doors,
                key=lambda door: len(completed.door_sequences.get(door, [])),
            )
            completed.outbound_assignment[truck] = (destination, best_door)
            completed.door_sequences.setdefault(best_door, []).append(truck)

    return completed


def _sequence_position(solution: Solution, truck: TruckId) -> int | None:
    if truck in solution.compound_assignment:
        return None
    door = solution.outbound_assignment[truck][1]
    return solution.door_sequences[door].index(truck)


def _mark_assigned(
    option: InsertionOption,
    unassigned_compounds: set[TruckId],
    unassigned_outbounds: set[TruckId],
    unassigned_destinations: set[DestinationId],
) -> None:
    unassigned_compounds.discard(option.truck)
    unassigned_outbounds.discard(option.truck)
    unassigned_destinations.remove(option.destination)

