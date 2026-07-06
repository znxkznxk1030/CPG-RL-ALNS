"""Scale-invariant features for the transferable operator-selection policy.

All features are ratios or normalized quantities so a policy trained on S/M
instances can be applied to L/XL instances (plan phase 2a). The feature vector
is `instance features (static per run) + search-state features (per iteration)
+ per-operator recent success rates`.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev

import numpy as np

from crossdock_solver.core.fast_evaluator import FastResult
from crossdock_solver.data.characteristics import compound_fraction, mean_dbpr
from crossdock_solver.data.instance import CrossDockInstance


INSTANCE_FEATURE_DIM = 10
SEARCH_FEATURE_DIM = 8


def feature_dim(num_operators: int) -> int:
    return INSTANCE_FEATURE_DIM + SEARCH_FEATURE_DIM + num_operators


def instance_features(instance: CrossDockInstance) -> np.ndarray:
    """Static, scale-invariant description of the problem instance."""

    compounds = instance.compound_trucks
    destinations = instance.destinations
    doors = instance.doors
    num_trucks = len(instance.all_trucks)

    dest_loads = [
        sum(instance.handling_time(c, d) for c in compounds) for d in destinations
    ]
    total_load = sum(dest_loads) + 1e-9
    mean_load = total_load / len(destinations)
    load_cv = pstdev(dest_loads) / mean_load if mean_load > 0 else 0.0

    nonzero = sum(
        1
        for c in compounds
        for d in destinations
        if instance.unit_amount(c, d) > 0
    )
    flow_density = nonzero / (len(compounds) * len(destinations))

    mean_travel = float(np.mean(instance.travel_time))
    travel_intensity = mean_travel / (mean_load + 1e-9)

    horizon = max(1.0, 2.0 * total_load / len(doors))
    finite_windows = [
        (instance.release_time[t], instance.due_time[t])
        for t in instance.all_trucks
        if instance.due_time[t] != float("inf")
    ]
    has_tw = 1.0 if finite_windows else 0.0
    mean_release = (
        sum(r for r, _ in finite_windows) / len(finite_windows) / horizon
        if finite_windows
        else 0.0
    )
    mean_width = (
        sum(d - r for r, d in finite_windows) / len(finite_windows) / horizon
        if finite_windows
        else 2.0
    )

    return np.array(
        [
            compound_fraction(instance),
            len(doors) / num_trucks,
            len(destinations) / len(doors),
            flow_density,
            min(2.0, load_cv),
            mean_dbpr(instance),
            min(2.0, travel_intensity),
            has_tw,
            min(2.0, mean_release),
            min(2.0, mean_width),
        ],
        dtype=np.float32,
    )


@dataclass
class SearchState:
    """Dynamic search-loop quantities needed for the state features."""

    iteration: int
    max_iterations: int
    temperature: float
    initial_temperature: float
    current: FastResult
    best: FastResult
    no_improvement: int
    no_improvement_cap: int
    since_best: int
    restart_after: int
    num_trucks: int


def search_features(state: SearchState) -> np.ndarray:
    best_obj = max(1e-9, state.best.objective)
    gap = (state.current.objective - state.best.objective) / best_obj

    door_finish = state.current.door_finish
    if door_finish:
        mean_finish = sum(door_finish) / len(door_finish) + 1e-9
        door_cv = pstdev(door_finish) / mean_finish
        concentration = max(door_finish) / mean_finish
    else:
        door_cv = 0.0
        concentration = 1.0

    tardy_ratio = (
        state.current.objective - state.current.makespan
    ) / max(1e-9, state.current.objective)

    return np.array(
        [
            state.iteration / max(1, state.max_iterations),
            min(2.0, state.temperature / max(1e-9, state.initial_temperature)),
            min(2.0, gap),
            min(2.0, state.no_improvement / max(1, state.no_improvement_cap)),
            min(2.0, state.since_best / max(1, state.restart_after)),
            min(2.0, door_cv),
            min(4.0, concentration),
            min(1.0, tardy_ratio),
        ],
        dtype=np.float32,
    )


class OperatorSuccessTracker:
    """Exponential moving average of each operator's recent success."""

    def __init__(self, actions: tuple[str, ...], *, decay: float = 0.9) -> None:
        self.actions = actions
        self.decay = decay
        self._index = {action: i for i, action in enumerate(actions)}
        self._ema = np.full(len(actions), 0.5, dtype=np.float32)

    def update(self, action: str, success: bool) -> None:
        i = self._index[action]
        self._ema[i] = self.decay * self._ema[i] + (1.0 - self.decay) * (1.0 if success else 0.0)

    def vector(self) -> np.ndarray:
        return self._ema.copy()


def build_feature_vector(
    instance_feats: np.ndarray,
    state: SearchState,
    tracker: OperatorSuccessTracker,
) -> np.ndarray:
    return np.concatenate([instance_feats, search_features(state), tracker.vector()])
