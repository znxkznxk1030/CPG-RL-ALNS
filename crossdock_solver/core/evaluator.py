from __future__ import annotations

from dataclasses import dataclass, field
from statistics import pstdev
from typing import Any

from crossdock_solver.core.feasibility import check_feasible
from crossdock_solver.core.solution import Solution
from crossdock_solver.data.instance import CrossDockInstance, DestinationId, DoorId, TruckId


@dataclass
class TransferEdge:
    source_compound: TruckId
    carrier: TruckId
    destination: DestinationId
    source_door: DoorId
    target_door: DoorId
    amount: float
    travel_time: float
    weighted_time: float


@dataclass
class ScheduleResult:
    makespan: float
    truck_start: dict[TruckId, float]
    truck_finish: dict[TruckId, float]
    door_finish: dict[DoorId, float]
    door_load: dict[DoorId, float]
    door_utilization: dict[DoorId, float]
    total_transfer_time: float
    max_transfer_edge_time: float
    waiting_time: dict[TruckId, float]
    critical_door: DoorId
    critical_truck: TruckId
    total_tardiness: float = 0.0
    critical_path: list[str] = field(default_factory=list)
    precedence_graph: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def evaluate_solution(
    instance: CrossDockInstance,
    solution: Solution,
    *,
    build_graph: bool = False,
) -> ScheduleResult:
    """Evaluate a feasible MVP solution.

    The MVP keeps the timing model intentionally conservative: a carrier starts
    loading a destination only after all transferred products for that
    destination have reached the carrier door.
    """

    if build_graph:
        raise NotImplementedError("full precedence graph is reserved for the research extension")

    check_feasible(instance, solution)

    truck_start: dict[TruckId, float] = {}
    truck_finish: dict[TruckId, float] = {}
    waiting_time: dict[TruckId, float] = {}
    compound_unload_finish: dict[TruckId, float] = {}
    compound_load_start: dict[TruckId, float] = {}
    door_finish: dict[DoorId, float] = {door: 0.0 for door in instance.doors}
    door_load: dict[DoorId, float] = {door: 0.0 for door in instance.doors}

    for compound in instance.compound_trucks:
        destination, door = solution.compound_assignment[compound]
        unload_time = _compound_unload_time(instance, compound, retained_destination=destination)
        release = instance.release_time[compound]
        truck_start[compound] = release
        compound_unload_finish[compound] = release + instance.enter_time[compound] + unload_time
        door_load[door] += instance.enter_time[compound] + unload_time

    carrier_by_destination = solution.destination_carriers()
    carrier_door_by_destination = {
        destination: solution.truck_door(carrier)
        for destination, carrier in carrier_by_destination.items()
    }

    destination_ready: dict[DestinationId, float] = {}
    transfer_edges: list[TransferEdge] = []
    total_transfer_time = 0.0
    max_transfer_edge_time = 0.0

    for destination in instance.destinations:
        carrier = carrier_by_destination[destination]
        target_door = carrier_door_by_destination[destination]
        ready_time = 0.0

        for source in instance.compound_trucks:
            if source == carrier:
                continue

            amount = instance.unit_amount(source, destination)
            if amount <= 0:
                continue

            source_door = solution.compound_assignment[source][1]
            travel_time = instance.travel(source_door, target_door)
            ready_time = max(ready_time, compound_unload_finish[source] + travel_time)

            weighted_time = amount * travel_time
            total_transfer_time += weighted_time
            max_transfer_edge_time = max(max_transfer_edge_time, weighted_time)
            transfer_edges.append(
                TransferEdge(
                    source_compound=source,
                    carrier=carrier,
                    destination=destination,
                    source_door=source_door,
                    target_door=target_door,
                    amount=amount,
                    travel_time=travel_time,
                    weighted_time=weighted_time,
                )
            )

        destination_ready[destination] = ready_time

    for compound in instance.compound_trucks:
        destination, door = solution.compound_assignment[compound]
        load_time = _compound_load_time(instance, compound, destination)
        load_start = max(compound_unload_finish[compound], destination_ready[destination])
        finish = load_start + load_time + instance.leave_time[compound]

        compound_load_start[compound] = load_start
        waiting_time[compound] = max(0.0, load_start - compound_unload_finish[compound])
        truck_finish[compound] = finish
        door_finish[door] = max(door_finish[door], finish)
        door_load[door] += load_time + instance.leave_time[compound]

    outbound_load_start: dict[TruckId, float] = {}
    for door in instance.doors:
        previous_finish = door_finish[door]
        for outbound in solution.door_sequences.get(door, []):
            destination, assigned_door = solution.outbound_assignment[outbound]
            if assigned_door != door:
                raise AssertionError("feasibility checker should reject mismatched door sequences")

            load_time = _outbound_load_time(instance, destination)
            start = max(
                previous_finish,
                destination_ready[destination],
                instance.release_time[outbound],
            )
            finish = start + instance.enter_time[outbound] + load_time + instance.leave_time[outbound]

            truck_start[outbound] = start
            outbound_load_start[outbound] = start + instance.enter_time[outbound]
            waiting_time[outbound] = max(0.0, start - destination_ready[destination])
            truck_finish[outbound] = finish
            previous_finish = finish
            door_load[door] += instance.enter_time[outbound] + load_time + instance.leave_time[outbound]

        door_finish[door] = previous_finish

    makespan = max(truck_finish.values(), default=0.0)
    total_tardiness = sum(
        max(0.0, finish - instance.due_time[truck])
        for truck, finish in truck_finish.items()
        if instance.due_time[truck] != float("inf")
    )
    door_utilization = {
        door: (door_load[door] / makespan if makespan > 0 else 0.0)
        for door in instance.doors
    }
    critical_door = max(door_finish, key=door_finish.get)
    critical_truck = max(truck_finish, key=truck_finish.get)
    critical_path = [f"Door_{critical_door}", critical_truck]

    return ScheduleResult(
        makespan=makespan,
        truck_start=truck_start,
        truck_finish=truck_finish,
        door_finish=door_finish,
        door_load=door_load,
        door_utilization=door_utilization,
        total_transfer_time=total_transfer_time,
        max_transfer_edge_time=max_transfer_edge_time,
        waiting_time=waiting_time,
        critical_door=critical_door,
        critical_truck=critical_truck,
        total_tardiness=total_tardiness,
        critical_path=critical_path,
        precedence_graph=None,
        metadata={
            "compound_unload_finish": compound_unload_finish,
            "compound_load_start": compound_load_start,
            "outbound_load_start": outbound_load_start,
            "destination_ready": destination_ready,
            "transfer_edges": transfer_edges,
            "door_load_std": pstdev(door_load.values()) if len(door_load) > 1 else 0.0,
        },
    )


def _compound_unload_time(
    instance: CrossDockInstance,
    compound: TruckId,
    *,
    retained_destination: DestinationId,
) -> float:
    return sum(
        instance.handling_time(compound, destination)
        for destination in instance.destinations
        if destination != retained_destination
    )


def _compound_load_time(
    instance: CrossDockInstance,
    carrier: TruckId,
    destination: DestinationId,
) -> float:
    return sum(
        instance.handling_time(source, destination)
        for source in instance.compound_trucks
        if source != carrier
    )


def _outbound_load_time(instance: CrossDockInstance, destination: DestinationId) -> float:
    return sum(
        instance.handling_time(source, destination)
        for source in instance.compound_trucks
    )

