from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from crossdock_solver.data.instance import CrossDockInstance


FLOW_PATTERNS = ("uniform", "skewed", "clustered")


@dataclass(frozen=True)
class InstanceShape:
    num_compounds: int
    num_outbounds: int
    num_doors: int
    num_products: int


SIZE_CLASSES: dict[str, InstanceShape] = {
    "S": InstanceShape(num_compounds=3, num_outbounds=7, num_doors=5, num_products=3),
    "M": InstanceShape(num_compounds=8, num_outbounds=22, num_doors=10, num_products=3),
    "L": InstanceShape(num_compounds=15, num_outbounds=45, num_doors=20, num_products=5),
    "XL": InstanceShape(num_compounds=25, num_outbounds=75, num_doors=30, num_products=5),
}


def generate_random_instance(
    *,
    num_compounds: int,
    num_outbounds: int,
    num_doors: int,
    num_products: int = 3,
    seed: int | None = None,
    flow_low: int = 0,
    flow_high: int = 20,
    flow_pattern: str = "uniform",
) -> CrossDockInstance:
    """Generate a random MVP-compatible instance.

    The number of destinations is set to `num_compounds + num_outbounds` so the
    MVP one-carrier-per-destination representation is feasible.

    Flow patterns:
    - `uniform`: independent uniform integer flow per (compound, destination, product).
    - `skewed`: heavy-tailed destination weights; few destinations carry most flow.
    - `clustered`: destinations are grouped per compound; each compound ships
      mostly to its home cluster.
    """

    if num_doors < num_compounds:
        raise ValueError("num_doors must be at least num_compounds")
    if flow_pattern not in FLOW_PATTERNS:
        raise ValueError(f"flow_pattern must be one of {FLOW_PATTERNS}, got {flow_pattern!r}")

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    compound_trucks = [f"C{i + 1}" for i in range(num_compounds)]
    outbound_trucks = [f"O{i + 1}" for i in range(num_outbounds)]
    destinations = [f"D{i + 1}" for i in range(num_compounds + num_outbounds)]
    doors = list(range(1, num_doors + 1))
    product_types = [f"K{i + 1}" for i in range(num_products)]

    flow = _generate_flow(
        np_rng,
        num_compounds=num_compounds,
        num_destinations=len(destinations),
        num_products=num_products,
        flow_low=flow_low,
        flow_high=flow_high,
        pattern=flow_pattern,
    )
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


def generate_benchmark_instance(
    size_class: str,
    flow_pattern: str = "uniform",
    *,
    seed: int,
) -> CrossDockInstance:
    """Generate one instance of a named benchmark cell (size class x flow pattern)."""

    if size_class not in SIZE_CLASSES:
        raise ValueError(f"size_class must be one of {tuple(SIZE_CLASSES)}, got {size_class!r}")
    shape = SIZE_CLASSES[size_class]
    return generate_random_instance(
        num_compounds=shape.num_compounds,
        num_outbounds=shape.num_outbounds,
        num_doors=shape.num_doors,
        num_products=shape.num_products,
        seed=seed,
        flow_pattern=flow_pattern,
    )


def _generate_flow(
    np_rng: np.random.Generator,
    *,
    num_compounds: int,
    num_destinations: int,
    num_products: int,
    flow_low: int,
    flow_high: int,
    pattern: str,
) -> np.ndarray:
    shape = (num_compounds, num_destinations, num_products)
    base = np_rng.integers(flow_low, flow_high + 1, size=shape).astype(float)

    if pattern == "uniform":
        return base

    if pattern == "skewed":
        weights = 1.0 + np_rng.pareto(1.5, size=num_destinations)
        weights = weights * (num_destinations / weights.sum())
        return np.floor(base * weights[None, :, None])

    cluster_of_destination = np_rng.integers(0, num_compounds, size=num_destinations)
    home_cluster = np.arange(num_compounds)
    is_home = home_cluster[:, None] == cluster_of_destination[None, :]
    off_home_scale = 0.2
    scale = np.where(is_home, 1.0, off_home_scale)
    return np.floor(base * scale[:, :, None])

