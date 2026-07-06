"""Combinatorial lower bounds on makespan.

Fallback for the G1 gate: CP-SAT proves optimality only on small instances
(S), so for M/L/XL we report an optimality gap against a provably valid
combinatorial lower bound. Each bound below is a relaxation of the evaluator's
timing model, so it never exceeds the true optimum.
"""

from __future__ import annotations

from crossdock_solver.data.instance import CrossDockInstance


def combinatorial_lower_bound(instance: CrossDockInstance) -> float:
    """Return max of valid combinatorial lower bounds on the optimal makespan."""

    return max(
        _critical_truck_bound(instance),
        _door_area_bound(instance),
    )


def combinatorial_objective_lower_bound(
    instance: CrossDockInstance, tardiness_weight: float
) -> float:
    """Valid lower bound on makespan + weight * total tardiness.

    For any feasible solution, makespan >= the makespan bound and total
    tardiness >= the structural tardiness bound (each truck finishes no earlier
    than its fastest possible chain), so their weighted sum bounds the
    objective.
    """

    return combinatorial_lower_bound(instance) + tardiness_weight * _tardiness_bound(
        instance
    )


def _earliest_finish(instance: CrossDockInstance, truck) -> float:
    """Earliest possible finish of a truck, relaxing all door contention."""

    compounds = instance.compound_trucks
    destinations = instance.destinations
    if truck in instance.compound_index:
        total_handling = sum(instance.handling_time(truck, d) for d in destinations)
        chain = min(
            total_handling
            - instance.handling_time(truck, d)
            + sum(
                instance.handling_time(source, d)
                for source in compounds
                if source != truck
            )
            for d in destinations
        )
    else:
        chain = min(
            sum(instance.handling_time(source, d) for source in compounds)
            for d in destinations
        )
    return (
        instance.release_time[truck]
        + instance.enter_time[truck]
        + chain
        + instance.leave_time[truck]
    )


def _tardiness_bound(instance: CrossDockInstance) -> float:
    """Structural lower bound on total tardiness (per-truck earliest finish)."""

    total = 0.0
    for truck in instance.all_trucks:
        due = instance.due_time[truck]
        if due == float("inf"):
            continue
        total += max(0.0, _earliest_finish(instance, truck) - due)
    return total


def _critical_truck_bound(instance: CrossDockInstance) -> float:
    """Longest unavoidable single-truck chain.

    A compound truck c retaining destination d finishes no earlier than
    release + enter + (unload all but d) + (load d from other compounds) + leave.
    Taking the best d for c gives a per-truck lower bound; the makespan is at
    least the largest such value over compounds. Outbound trucks give an
    analogous bound via their destination load.
    """

    compounds = instance.compound_trucks
    destinations = instance.destinations
    best = 0.0

    for compound in compounds:
        total_handling = sum(instance.handling_time(compound, d) for d in destinations)
        chain = min(
            total_handling
            - instance.handling_time(compound, d)
            + sum(
                instance.handling_time(source, d)
                for source in compounds
                if source != compound
            )
            for d in destinations
        )
        finish = (
            instance.release_time[compound]
            + instance.enter_time[compound]
            + chain
            + instance.leave_time[compound]
        )
        best = max(best, finish)

    for outbound in instance.outbound_trucks:
        min_load = min(
            sum(instance.handling_time(source, d) for source in compounds)
            for d in destinations
        )
        finish = (
            instance.release_time[outbound]
            + instance.enter_time[outbound]
            + min_load
            + instance.leave_time[outbound]
        )
        best = max(best, finish)

    return best


def _door_area_bound(instance: CrossDockInstance) -> float:
    """Total forced door-occupancy spread over the doors.

    Every truck occupies some door for at least (enter + minimal unload + minimal
    load + leave). Doors work in parallel, so the makespan is at least the sum of
    these minimal occupancies divided by the number of doors.
    """

    compounds = instance.compound_trucks
    destinations = instance.destinations
    num_doors = len(instance.doors)
    if num_doors == 0:
        return 0.0

    total = 0.0
    for compound in compounds:
        total_handling = sum(instance.handling_time(compound, d) for d in destinations)
        min_unload = total_handling - max(
            instance.handling_time(compound, d) for d in destinations
        )
        min_load = min(
            sum(
                instance.handling_time(source, d)
                for source in compounds
                if source != compound
            )
            for d in destinations
        )
        total += (
            instance.enter_time[compound]
            + min_unload
            + min_load
            + instance.leave_time[compound]
        )

    for outbound in instance.outbound_trucks:
        min_load = min(
            sum(instance.handling_time(source, d) for source in compounds)
            for d in destinations
        )
        total += instance.enter_time[outbound] + min_load + instance.leave_time[outbound]

    return total / num_doors
