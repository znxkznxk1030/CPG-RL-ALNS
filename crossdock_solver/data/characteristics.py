"""Instance characteristics used for benchmark design and reporting.

See `docs/benchmark_design.md` for the paper mapping.
"""

from __future__ import annotations

from crossdock_solver.data.instance import CrossDockInstance, DestinationId


def paper_time_budget(instance: CrossDockInstance, *, factor: float = 0.7) -> float:
    """Metaheuristic time budget from Shahmardan & Sajadieh (2020).

    T = ((|I| + |D|) / 2) * |M| * factor  seconds, with |I| compound trucks,
    |D| destinations, |M| dock doors. Validated against the paper's Table 5:
    (8,10,8) -> 50.4s, (20,30,20) -> 350s.
    """

    num_compounds = len(instance.compound_trucks)
    num_destinations = len(instance.destinations)
    num_doors = len(instance.doors)
    return (num_compounds + num_destinations) / 2.0 * num_doors * factor


def dbpr(instance: CrossDockInstance) -> dict[DestinationId, float]:
    """Destination bound product ratio per destination (paper Eq. 32).

    For each destination d, the maximum over compound trucks of the fraction of
    that truck's total handling work devoted to d. High DBPR means a
    destination's demand is concentrated in one compound truck, where partial
    unloading helps most.
    """

    totals = {
        compound: sum(
            instance.handling_time(compound, destination)
            for destination in instance.destinations
        )
        for compound in instance.compound_trucks
    }
    result: dict[DestinationId, float] = {}
    for destination in instance.destinations:
        ratios = [
            instance.handling_time(compound, destination) / totals[compound]
            for compound in instance.compound_trucks
            if totals[compound] > 0
        ]
        result[destination] = max(ratios, default=0.0)
    return result


def mean_dbpr(instance: CrossDockInstance) -> float:
    values = dbpr(instance).values()
    return sum(values) / len(values) if values else 0.0


def compound_fraction(instance: CrossDockInstance) -> float:
    """Share of trucks that are compound trucks (paper regime is 0.67-0.80)."""

    return len(instance.compound_trucks) / max(1, len(instance.all_trucks))
