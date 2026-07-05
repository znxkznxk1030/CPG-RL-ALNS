from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np


TruckId = str
DestinationId = str
ProductId = str
DoorId = int


@dataclass(frozen=True)
class CrossDockInstance:
    """Problem data for the cross-docking scheduling MVP.

    The MVP assumes every truck is a destination carrier exactly once and every
    destination is covered exactly once. Therefore `len(compound_trucks) +
    len(outbound_trucks)` must equal `len(destinations)`.
    """

    compound_trucks: list[TruckId]
    outbound_trucks: list[TruckId]
    destinations: list[DestinationId]
    doors: list[DoorId]
    product_types: list[ProductId]
    flow: np.ndarray
    product_time: np.ndarray
    travel_time: np.ndarray
    enter_time: dict[TruckId, float]
    leave_time: dict[TruckId, float]
    release_time: dict[TruckId, float] | None = None
    due_time: dict[TruckId, float] | None = None
    compound_index: dict[TruckId, int] = field(init=False)
    destination_index: dict[DestinationId, int] = field(init=False)
    product_index: dict[ProductId, int] = field(init=False)
    door_index: dict[DoorId, int] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "compound_index", {truck: idx for idx, truck in enumerate(self.compound_trucks)}
        )
        object.__setattr__(
            self, "destination_index", {dest: idx for idx, dest in enumerate(self.destinations)}
        )
        object.__setattr__(
            self, "product_index", {prod: idx for idx, prod in enumerate(self.product_types)}
        )
        object.__setattr__(
            self, "door_index", {door: idx for idx, door in enumerate(self.doors)}
        )
        if self.release_time is None:
            object.__setattr__(
                self, "release_time", {truck: 0.0 for truck in self.all_trucks}
            )
        if self.due_time is None:
            object.__setattr__(
                self, "due_time", {truck: float("inf") for truck in self.all_trucks}
            )
        self.validate()
        handling = self.flow @ self.product_time
        units = self.flow.sum(axis=2)
        object.__setattr__(
            self,
            "_handling_cache",
            {
                (truck, dest): float(handling[c, d])
                for truck, c in self.compound_index.items()
                for dest, d in self.destination_index.items()
            },
        )
        object.__setattr__(
            self,
            "_unit_cache",
            {
                (truck, dest): float(units[c, d])
                for truck, c in self.compound_index.items()
                for dest, d in self.destination_index.items()
            },
        )

    @property
    def all_trucks(self) -> list[TruckId]:
        return [*self.compound_trucks, *self.outbound_trucks]

    def validate(self) -> None:
        _ensure_unique("compound_trucks", self.compound_trucks)
        _ensure_unique("outbound_trucks", self.outbound_trucks)
        _ensure_unique("destinations", self.destinations)
        _ensure_unique("doors", self.doors)
        _ensure_unique("product_types", self.product_types)

        overlap = set(self.compound_trucks) & set(self.outbound_trucks)
        if overlap:
            raise ValueError(f"truck ids must be unique across types: {sorted(overlap)}")

        expected_flow_shape = (
            len(self.compound_trucks),
            len(self.destinations),
            len(self.product_types),
        )
        if tuple(self.flow.shape) != expected_flow_shape:
            raise ValueError(
                f"flow shape must be {expected_flow_shape}, got {tuple(self.flow.shape)}"
            )

        if tuple(self.product_time.shape) != (len(self.product_types),):
            raise ValueError(
                "product_time shape must be "
                f"{(len(self.product_types),)}, got {tuple(self.product_time.shape)}"
            )

        expected_travel_shape = (len(self.doors), len(self.doors))
        if tuple(self.travel_time.shape) != expected_travel_shape:
            raise ValueError(
                f"travel_time shape must be {expected_travel_shape}, "
                f"got {tuple(self.travel_time.shape)}"
            )

        if np.any(self.flow < 0):
            raise ValueError("flow must be non-negative")
        if np.any(self.product_time < 0):
            raise ValueError("product_time must be non-negative")
        if np.any(self.travel_time < 0):
            raise ValueError("travel_time must be non-negative")

        missing_enter = set(self.all_trucks) - set(self.enter_time)
        missing_leave = set(self.all_trucks) - set(self.leave_time)
        if missing_enter:
            raise ValueError(f"enter_time missing trucks: {sorted(missing_enter)}")
        if missing_leave:
            raise ValueError(f"leave_time missing trucks: {sorted(missing_leave)}")

        missing_release = set(self.all_trucks) - set(self.release_time)
        missing_due = set(self.all_trucks) - set(self.due_time)
        if missing_release:
            raise ValueError(f"release_time missing trucks: {sorted(missing_release)}")
        if missing_due:
            raise ValueError(f"due_time missing trucks: {sorted(missing_due)}")
        for truck in self.all_trucks:
            if self.release_time[truck] < 0:
                raise ValueError(f"release_time must be non-negative for {truck}")
            if self.due_time[truck] < self.release_time[truck]:
                raise ValueError(f"due_time must be at least release_time for {truck}")

        if len(self.compound_trucks) + len(self.outbound_trucks) != len(self.destinations):
            raise ValueError(
                "MVP requires one destination per truck and one carrier per destination: "
                "len(compound_trucks) + len(outbound_trucks) must equal len(destinations)"
            )

        if len(self.compound_trucks) > len(self.doors):
            raise ValueError("MVP compound first-stage capacity requires at least one door per compound")

    def flow_vector(self, compound: TruckId, destination: DestinationId) -> np.ndarray:
        return self.flow[self.compound_index[compound], self.destination_index[destination], :]

    def handling_time(self, compound: TruckId, destination: DestinationId) -> float:
        return self._handling_cache[(compound, destination)]

    def unit_amount(self, compound: TruckId, destination: DestinationId) -> float:
        return self._unit_cache[(compound, destination)]

    def travel(self, source_door: DoorId, target_door: DoorId) -> float:
        return float(self.travel_time[self.door_index[source_door], self.door_index[target_door]])


def _ensure_unique(name: str, values: list[Hashable]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
