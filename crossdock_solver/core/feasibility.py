from __future__ import annotations

from collections import Counter

from crossdock_solver.core.solution import Solution
from crossdock_solver.data.instance import CrossDockInstance


class InfeasibleSolution(ValueError):
    """Raised when a solution violates the MVP representation constraints."""


def check_feasible(instance: CrossDockInstance, solution: Solution) -> None:
    compound_set = set(instance.compound_trucks)
    outbound_set = set(instance.outbound_trucks)
    door_set = set(instance.doors)

    if set(solution.compound_assignment) != compound_set:
        missing = compound_set - set(solution.compound_assignment)
        extra = set(solution.compound_assignment) - compound_set
        raise InfeasibleSolution(f"invalid compound assignment keys, missing={missing}, extra={extra}")

    if set(solution.outbound_assignment) != outbound_set:
        missing = outbound_set - set(solution.outbound_assignment)
        extra = set(solution.outbound_assignment) - outbound_set
        raise InfeasibleSolution(f"invalid outbound assignment keys, missing={missing}, extra={extra}")

    covered: list[str] = []
    for truck, (destination, door) in solution.compound_assignment.items():
        if destination not in instance.destination_index:
            raise InfeasibleSolution(f"compound {truck} has unknown destination {destination}")
        if door not in door_set:
            raise InfeasibleSolution(f"compound {truck} has unknown door {door}")
        covered.append(destination)

    for truck, (destination, door) in solution.outbound_assignment.items():
        if destination not in instance.destination_index:
            raise InfeasibleSolution(f"outbound {truck} has unknown destination {destination}")
        if door not in door_set:
            raise InfeasibleSolution(f"outbound {truck} has unknown door {door}")
        covered.append(destination)

    destination_counts = Counter(covered)
    bad_destinations = {
        destination: count
        for destination, count in destination_counts.items()
        if count != 1
    }
    missing_destinations = set(instance.destinations) - set(destination_counts)
    if bad_destinations or missing_destinations:
        raise InfeasibleSolution(
            "each destination must be covered exactly once; "
            f"bad={bad_destinations}, missing={sorted(missing_destinations)}"
        )

    compound_door_counts = Counter(door for _, door in solution.compound_assignment.values())
    overloaded = {door: count for door, count in compound_door_counts.items() if count > 1}
    if overloaded:
        raise InfeasibleSolution(f"compound first-stage door capacity violated: {overloaded}")

    expected_sequences = {door: [] for door in instance.doors}
    for truck, (_, door) in solution.outbound_assignment.items():
        expected_sequences[door].append(truck)

    sequenced: list[str] = []
    for door, sequence in solution.door_sequences.items():
        if door not in door_set:
            raise InfeasibleSolution(f"door sequence for unknown door {door}")
        for truck in sequence:
            if truck not in outbound_set:
                raise InfeasibleSolution(f"door sequence includes non-outbound truck {truck}")
            if solution.outbound_assignment[truck][1] != door:
                raise InfeasibleSolution(
                    f"outbound {truck} assigned to door {solution.outbound_assignment[truck][1]} "
                    f"but appears in sequence for door {door}"
                )
        sequenced.extend(sequence)

    sequence_counts = Counter(sequenced)
    duplicated = {truck: count for truck, count in sequence_counts.items() if count != 1}
    missing_sequence = outbound_set - set(sequence_counts)
    if duplicated or missing_sequence:
        raise InfeasibleSolution(
            "each outbound truck must appear exactly once in door sequences; "
            f"duplicated={duplicated}, missing={sorted(missing_sequence)}"
        )

    for door in instance.doors:
        solution.door_sequences.setdefault(door, [])

