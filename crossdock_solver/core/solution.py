from __future__ import annotations

from dataclasses import dataclass

from crossdock_solver.data.instance import DestinationId, DoorId, TruckId


@dataclass
class Solution:
    """Assignment and outbound door sequences."""

    compound_assignment: dict[TruckId, tuple[DestinationId, DoorId]]
    outbound_assignment: dict[TruckId, tuple[DestinationId, DoorId]]
    door_sequences: dict[DoorId, list[TruckId]]

    def copy(self) -> "Solution":
        return Solution(
            compound_assignment=dict(self.compound_assignment),
            outbound_assignment=dict(self.outbound_assignment),
            door_sequences={door: list(seq) for door, seq in self.door_sequences.items()},
        )

    def destination_carriers(self) -> dict[DestinationId, TruckId]:
        carriers: dict[DestinationId, TruckId] = {}
        for truck, (destination, _) in self.compound_assignment.items():
            carriers[destination] = truck
        for truck, (destination, _) in self.outbound_assignment.items():
            carriers[destination] = truck
        return carriers

    def truck_door(self, truck: TruckId) -> DoorId:
        if truck in self.compound_assignment:
            return self.compound_assignment[truck][1]
        return self.outbound_assignment[truck][1]

    def truck_destination(self, truck: TruckId) -> DestinationId:
        if truck in self.compound_assignment:
            return self.compound_assignment[truck][0]
        return self.outbound_assignment[truck][0]

