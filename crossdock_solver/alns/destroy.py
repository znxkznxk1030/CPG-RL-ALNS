from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from crossdock_solver.core.evaluator import ScheduleResult, TransferEdge
from crossdock_solver.core.solution import Solution
from crossdock_solver.data.instance import CrossDockInstance, DestinationId, DoorId, TruckId


@dataclass
class DestroyResult:
    partial_solution: Solution
    removed_compounds: set[TruckId]
    removed_outbounds: set[TruckId]
    removed_destinations: set[DestinationId]
    candidate_doors: set[DoorId]
    metadata: dict = field(default_factory=dict)

    @property
    def removed_trucks(self) -> set[TruckId]:
        return set(self.removed_compounds) | set(self.removed_outbounds)


def critical_door_destroy(
    instance: CrossDockInstance,
    solution: Solution,
    result: ScheduleResult,
    *,
    size: str = "small",
    rng: random.Random | None = None,
) -> DestroyResult:
    """Remove trucks around the current critical door.

    The first layer removes trucks assigned to the critical door. If the target
    destroy size is larger, high weighted transfer edges incident to that door
    add adjacent source/carrier trucks.
    """

    rng = rng or random.Random()
    target_count = _target_destroy_count(
        total=len(instance.compound_trucks) + len(instance.outbound_trucks),
        size=size,
    )
    critical_door = result.critical_door

    removed_compounds = {
        truck
        for truck, (_, door) in solution.compound_assignment.items()
        if door == critical_door
    }
    removed_outbounds = {
        truck
        for truck, (_, door) in solution.outbound_assignment.items()
        if door == critical_door
    }

    transfer_edges = sorted(
        result.metadata.get("transfer_edges", []),
        key=lambda edge: edge.weighted_time,
        reverse=True,
    )
    for edge in transfer_edges:
        if len(removed_compounds) + len(removed_outbounds) >= target_count:
            break
        if not _touches_door(edge, critical_door):
            continue
        _add_truck(edge.source_compound, instance, removed_compounds, removed_outbounds)
        if len(removed_compounds) + len(removed_outbounds) >= target_count:
            break
        _add_truck(edge.carrier, instance, removed_compounds, removed_outbounds)

    if not removed_compounds and not removed_outbounds:
        # Defensive fallback for degenerate schedules.
        all_trucks = [*instance.compound_trucks, *instance.outbound_trucks]
        chosen = rng.choice(all_trucks)
        _add_truck(chosen, instance, removed_compounds, removed_outbounds)

    removed_destinations = {
        solution.compound_assignment[truck][0] for truck in removed_compounds
    } | {
        solution.outbound_assignment[truck][0] for truck in removed_outbounds
    }

    partial = solution.copy()
    for truck in removed_compounds:
        del partial.compound_assignment[truck]
    for truck in removed_outbounds:
        del partial.outbound_assignment[truck]
    for door in instance.doors:
        partial.door_sequences[door] = [
            truck for truck in partial.door_sequences.get(door, []) if truck not in removed_outbounds
        ]

    fixed_compound_doors = {door for _, door in partial.compound_assignment.values()}
    candidate_doors = set(instance.doors) - fixed_compound_doors

    return DestroyResult(
        partial_solution=partial,
        removed_compounds=removed_compounds,
        removed_outbounds=removed_outbounds,
        removed_destinations=removed_destinations,
        candidate_doors=candidate_doors,
        metadata={
            "destroy_name": "critical_door",
            "critical_door": critical_door,
            "target_count": target_count,
        },
    )


def _target_destroy_count(total: int, size: str) -> int:
    ratio_by_size = {
        "small": 0.20,
        "medium": 0.35,
        "large": 0.50,
    }
    if size not in ratio_by_size:
        raise ValueError(f"unknown destroy size {size!r}")
    return max(1, min(total, math.ceil(total * ratio_by_size[size])))


def _touches_door(edge: TransferEdge, door: DoorId) -> bool:
    return edge.source_door == door or edge.target_door == door


def _add_truck(
    truck: TruckId,
    instance: CrossDockInstance,
    removed_compounds: set[TruckId],
    removed_outbounds: set[TruckId],
) -> None:
    if truck in instance.compound_index:
        removed_compounds.add(truck)
    elif truck in instance.outbound_trucks:
        removed_outbounds.add(truck)

