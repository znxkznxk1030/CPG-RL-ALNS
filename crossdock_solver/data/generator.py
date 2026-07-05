from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from crossdock_solver.data.instance import CrossDockInstance


FLOW_PATTERNS = ("uniform", "skewed", "clustered")

PARAM_PROFILES = ("synthetic", "paper")

TW_TIGHTNESS: dict[str, tuple[float, float]] = {
    "loose": (0.10, 1.00),
    "medium": (0.25, 0.60),
    "tight": (0.50, 0.35),
}


@dataclass(frozen=True)
class InstanceShape:
    num_compounds: int
    num_outbounds: int
    num_doors: int
    num_products: int


# Paper-matching regime (Shahmardan & Sajadieh 2020): compound-heavy, |I| = |M|,
# |D| = 1.5|I| so compound fraction = 2/3. Sizes chosen as (I, D, M):
# S = (6,9,6) is the paper's largest exact-solvable small-scale size (Table 2);
# L = (20,30,20) is the paper's largest large-scale size (Table 4);
# XL extends beyond the paper. See docs/benchmark_design.md.
SIZE_CLASSES: dict[str, InstanceShape] = {
    "S": InstanceShape(num_compounds=6, num_outbounds=3, num_doors=6, num_products=3),
    "M": InstanceShape(num_compounds=12, num_outbounds=6, num_doors=12, num_products=3),
    "L": InstanceShape(num_compounds=20, num_outbounds=10, num_doors=20, num_products=3),
    "XL": InstanceShape(num_compounds=30, num_outbounds=15, num_doors=30, num_products=3),
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
    tw_tightness: str | None = None,
    param_profile: str = "synthetic",
) -> CrossDockInstance:
    """Generate a random MVP-compatible instance.

    The number of destinations is set to `num_compounds + num_outbounds` so the
    MVP one-carrier-per-destination representation is feasible.

    Flow patterns:
    - `uniform`: independent uniform integer flow per (compound, destination, product).
    - `skewed`: heavy-tailed destination weights; few destinations carry most flow.
    - `clustered`: destinations are grouped per compound; each compound ships
      mostly to its home cluster.

    Parameter profiles (see `docs/benchmark_design.md`):
    - `synthetic`: DE/DL ~ U(1,5), t_k ~ U(1,5), Euclidean door travel on a
      100x100 plane.
    - `paper`: Shahmardan & Sajadieh (2020) setup, DE/DL ~ U(0,20),
      t_k ~ U(3,10), I-shape linear doors with adjacent travel = 1 (|m-n|).

    Time windows: `tw_tightness` in {loose, medium, tight} draws truck release
    times in [0, rho * H] and sets due times to release + delta * H, where H is
    an unconstrained-makespan estimate. `None` keeps the classic unconstrained
    instance (release 0, due infinity).
    """

    if num_doors < num_compounds:
        raise ValueError("num_doors must be at least num_compounds")
    if flow_pattern not in FLOW_PATTERNS:
        raise ValueError(f"flow_pattern must be one of {FLOW_PATTERNS}, got {flow_pattern!r}")
    if tw_tightness is not None and tw_tightness not in TW_TIGHTNESS:
        raise ValueError(
            f"tw_tightness must be one of {tuple(TW_TIGHTNESS)} or None, got {tw_tightness!r}"
        )
    if param_profile not in PARAM_PROFILES:
        raise ValueError(f"param_profile must be one of {PARAM_PROFILES}, got {param_profile!r}")

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

    if param_profile == "paper":
        product_time = np_rng.integers(3, 11, size=num_products).astype(float)
        travel_time = np.array(
            [[float(abs(a - b)) for b in range(num_doors)] for a in range(num_doors)]
        )
        enter_low, enter_high = 0, 20
    else:
        product_time = np_rng.integers(1, 5, size=num_products).astype(float)
        coordinates = np_rng.uniform(0, 100, size=(num_doors, 2))
        travel_time = np.zeros((num_doors, num_doors), dtype=float)
        for a in range(num_doors):
            for b in range(num_doors):
                travel_time[a, b] = round(
                    float(np.linalg.norm(coordinates[a] - coordinates[b]) / 10), 2
                )
        enter_low, enter_high = 1, 5

    all_trucks = [*compound_trucks, *outbound_trucks]
    enter_time = {truck: float(rng.randint(enter_low, enter_high)) for truck in all_trucks}
    leave_time = {truck: float(rng.randint(enter_low, enter_high)) for truck in all_trucks}

    release_time = None
    due_time = None
    if tw_tightness is not None:
        release_ratio, width_ratio = TW_TIGHTNESS[tw_tightness]
        horizon = _makespan_estimate(flow, product_time, num_doors)
        release_time = {
            truck: round(rng.uniform(0.0, release_ratio * horizon), 1)
            for truck in all_trucks
        }
        due_time = {
            truck: round(release_time[truck] + width_ratio * horizon, 1)
            for truck in all_trucks
        }

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
        release_time=release_time,
        due_time=due_time,
    )


def generate_benchmark_instance(
    size_class: str,
    flow_pattern: str = "uniform",
    *,
    seed: int,
    tw_tightness: str | None = None,
) -> CrossDockInstance:
    """Generate one instance of a named benchmark cell."""

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
        tw_tightness=tw_tightness,
    )


def _makespan_estimate(flow: np.ndarray, product_time: np.ndarray, num_doors: int) -> float:
    """Rough unconstrained makespan scale: total unload+load work spread over doors."""

    total_handling = float((flow @ product_time).sum())
    return max(1.0, 2.0 * total_handling / num_doors)


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

