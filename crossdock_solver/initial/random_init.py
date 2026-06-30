from __future__ import annotations

import random

from crossdock_solver.core.feasibility import check_feasible
from crossdock_solver.core.solution import Solution
from crossdock_solver.data.instance import CrossDockInstance


def random_feasible_solution(
    instance: CrossDockInstance,
    rng: random.Random | None = None,
) -> Solution:
    """Build a random feasible solution for the MVP representation."""

    rng = rng or random.Random()
    destinations = list(instance.destinations)
    rng.shuffle(destinations)

    compound_assignment = {}
    outbound_assignment = {}

    compound_doors = rng.sample(instance.doors, k=len(instance.compound_trucks))
    for truck, destination, door in zip(
        instance.compound_trucks,
        destinations[: len(instance.compound_trucks)],
        compound_doors,
        strict=True,
    ):
        compound_assignment[truck] = (destination, door)

    remaining_destinations = destinations[len(instance.compound_trucks) :]
    door_sequences = {door: [] for door in instance.doors}

    for truck, destination in zip(instance.outbound_trucks, remaining_destinations, strict=True):
        door = rng.choice(instance.doors)
        outbound_assignment[truck] = (destination, door)
        door_sequences[door].append(truck)

    for sequence in door_sequences.values():
        rng.shuffle(sequence)

    solution = Solution(
        compound_assignment=compound_assignment,
        outbound_assignment=outbound_assignment,
        door_sequences=door_sequences,
    )
    check_feasible(instance, solution)
    return solution

