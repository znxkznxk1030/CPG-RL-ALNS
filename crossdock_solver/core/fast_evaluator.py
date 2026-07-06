from __future__ import annotations

from dataclasses import dataclass

from crossdock_solver.core.solution import Solution
from crossdock_solver.data.instance import CrossDockInstance, DoorId, TruckId


@dataclass(frozen=True)
class FastResult:
    """Minimal schedule summary for search hot paths.

    `door_finish` follows `instance.doors` order; `tardy_count` is the number
    of trucks finishing after their due time. Both feed the search-state
    features of the transferable policy.
    """

    makespan: float
    critical_door: DoorId
    critical_truck: TruckId
    total_tardiness: float = 0.0
    objective: float = 0.0
    door_finish: tuple[float, ...] = ()
    tardy_count: int = 0
    most_tardy_truck: TruckId | None = None


class FastEvaluator:
    """Search-oriented evaluator with instance-static tables precomputed once.

    Produces the same makespan and total tardiness as `evaluate_solution` but
    skips the feasibility check, per-call numpy reductions, and result
    metadata. `objective = makespan + tardiness_weight * total_tardiness` is
    the scalar the search should compare. Callers are responsible for
    feasibility.
    """

    def __init__(self, instance: CrossDockInstance, *, tardiness_weight: float = 0.0) -> None:
        self.instance = instance
        self.tardiness_weight = tardiness_weight
        compounds = instance.compound_trucks
        destinations = instance.destinations
        doors = instance.doors

        self._compound_idx = {truck: idx for idx, truck in enumerate(compounds)}
        self._dest_idx = {dest: idx for idx, dest in enumerate(destinations)}
        self._door_idx = {door: idx for idx, door in enumerate(doors)}

        self._handling = [
            [instance.handling_time(compound, dest) for dest in destinations]
            for compound in compounds
        ]
        self._total_handling = [sum(row) for row in self._handling]
        self._dest_load = [
            sum(self._handling[c][d] for c in range(len(compounds)))
            for d in range(len(destinations))
        ]
        self._sources_by_dest = [
            [
                c
                for c, compound in enumerate(compounds)
                if instance.unit_amount(compound, destinations[d]) > 0
            ]
            for d in range(len(destinations))
        ]
        self._travel = [
            [instance.travel(a, b) for b in doors]
            for a in doors
        ]
        self._enter = dict(instance.enter_time)
        self._leave = dict(instance.leave_time)
        self._release = dict(instance.release_time)
        self._due = {
            truck: due
            for truck, due in instance.due_time.items()
            if due != float("inf")
        }

    def makespan(self, solution: Solution) -> float:
        return self.evaluate(solution).makespan

    def evaluate(self, solution: Solution) -> FastResult:
        instance = self.instance
        compound_idx = self._compound_idx
        dest_idx = self._dest_idx
        door_idx = self._door_idx
        handling = self._handling
        travel = self._travel

        num_compounds = len(instance.compound_trucks)
        num_dests = len(instance.destinations)

        unload_finish = [0.0] * num_compounds
        compound_door = [0] * num_compounds
        carrier_compound = [-1] * num_dests
        carrier_door = [0] * num_dests

        for truck, (destination, door) in solution.compound_assignment.items():
            c = compound_idx[truck]
            d = dest_idx[destination]
            m = door_idx[door]
            unload_finish[c] = (
                self._release[truck] + self._enter[truck] + self._total_handling[c] - handling[c][d]
            )
            compound_door[c] = m
            carrier_compound[d] = c
            carrier_door[d] = m

        for truck, (destination, door) in solution.outbound_assignment.items():
            carrier_door[dest_idx[destination]] = door_idx[door]

        destination_ready = [0.0] * num_dests
        for d in range(num_dests):
            target = carrier_door[d]
            carrier = carrier_compound[d]
            travel_from_target = travel[target]
            ready = 0.0
            for c in self._sources_by_dest[d]:
                if c == carrier:
                    continue
                arrival = unload_finish[c] + travel_from_target[compound_door[c]]
                if arrival > ready:
                    ready = arrival
            destination_ready[d] = ready

        best_makespan = 0.0
        total_tardiness = 0.0
        tardy_count = 0
        worst_tardiness = 0.0
        most_tardy_truck: TruckId | None = None
        due = self._due
        critical_truck: TruckId = instance.all_trucks[0]
        door_finish = [0.0] * len(instance.doors)

        for truck, (destination, _) in solution.compound_assignment.items():
            c = compound_idx[truck]
            d = dest_idx[destination]
            load_start = unload_finish[c]
            if destination_ready[d] > load_start:
                load_start = destination_ready[d]
            finish = (
                load_start
                + self._dest_load[d]
                - handling[c][d]
                + self._leave[truck]
            )
            door_finish[compound_door[c]] = finish
            if finish > best_makespan:
                best_makespan = finish
                critical_truck = truck
            truck_due = due.get(truck)
            if truck_due is not None and finish > truck_due:
                tardiness = finish - truck_due
                total_tardiness += tardiness
                tardy_count += 1
                if tardiness > worst_tardiness:
                    worst_tardiness = tardiness
                    most_tardy_truck = truck

        for door, sequence in solution.door_sequences.items():
            m = door_idx[door]
            previous = door_finish[m]
            for truck in sequence:
                d = dest_idx[solution.outbound_assignment[truck][0]]
                start = previous if previous > destination_ready[d] else destination_ready[d]
                release = self._release[truck]
                if release > start:
                    start = release
                finish = start + self._enter[truck] + self._dest_load[d] + self._leave[truck]
                previous = finish
                if finish > best_makespan:
                    best_makespan = finish
                    critical_truck = truck
                truck_due = due.get(truck)
                if truck_due is not None and finish > truck_due:
                    tardiness = finish - truck_due
                    total_tardiness += tardiness
                    tardy_count += 1
                    if tardiness > worst_tardiness:
                        worst_tardiness = tardiness
                        most_tardy_truck = truck
            door_finish[m] = previous

        critical_door = instance.doors[max(range(len(door_finish)), key=door_finish.__getitem__)]
        return FastResult(
            makespan=best_makespan,
            critical_door=critical_door,
            critical_truck=critical_truck,
            total_tardiness=total_tardiness,
            objective=best_makespan + self.tardiness_weight * total_tardiness,
            door_finish=tuple(door_finish),
            tardy_count=tardy_count,
            most_tardy_truck=most_tardy_truck,
        )
