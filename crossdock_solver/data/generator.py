from __future__ import annotations

import random

import numpy as np

from crossdock_solver.data.instance import CrossDockInstance


def generate_random_instance(
    *,
    num_compounds: int,
    num_outbounds: int,
    num_doors: int,
    num_products: int = 3,
    seed: int | None = None,
    flow_low: int = 0,
    flow_high: int = 20,
) -> CrossDockInstance:
    """Generate a random MVP-compatible instance.

    The number of destinations is set to `num_compounds + num_outbounds` so the
    MVP one-carrier-per-destination representation is feasible.
    """

    if num_doors < num_compounds:
        raise ValueError("num_doors must be at least num_compounds")

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    compound_trucks = [f"C{i + 1}" for i in range(num_compounds)]
    outbound_trucks = [f"O{i + 1}" for i in range(num_outbounds)]
    destinations = [f"D{i + 1}" for i in range(num_compounds + num_outbounds)]
    doors = list(range(1, num_doors + 1))
    product_types = [f"K{i + 1}" for i in range(num_products)]

    flow = np_rng.integers(
        flow_low,
        flow_high + 1,
        size=(num_compounds, len(destinations), num_products),
    ).astype(float)
    product_time = np_rng.integers(1, 5, size=num_products).astype(float)

    coordinates = np_rng.uniform(0, 100, size=(num_doors, 2))
    travel_time = np.zeros((num_doors, num_doors), dtype=float)
    for a in range(num_doors):
        for b in range(num_doors):
            travel_time[a, b] = round(float(np.linalg.norm(coordinates[a] - coordinates[b]) / 10), 2)

    all_trucks = [*compound_trucks, *outbound_trucks]
    enter_time = {truck: float(rng.randint(1, 5)) for truck in all_trucks}
    leave_time = {truck: float(rng.randint(1, 5)) for truck in all_trucks}

    return CrossDockInstance(
        compound_trucks=compound_trucks,
        outbound_trucks=outbound_trucks,
        destinations=destinations,
        doors=doors,
        product_types=product_types,
        flow=flow,
        product_time=product_time,
        travel_time=travel_time,
        enter_time=enter_time,
        leave_time=leave_time,
    )

