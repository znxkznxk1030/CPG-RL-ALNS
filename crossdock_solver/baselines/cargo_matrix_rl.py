from __future__ import annotations

from dataclasses import dataclass, field
import random
import time

import numpy as np

from crossdock_solver.baselines.destination_agent_rl import _build_solution_from_destination_carriers
from crossdock_solver.baselines.random_baseline import BaselineRun
from crossdock_solver.baselines.vaa import _destination_load, vaa_solution
from crossdock_solver.core.evaluator import evaluate_solution
from crossdock_solver.core.solution import Solution
from crossdock_solver.data.instance import CrossDockInstance, DestinationId, TruckId
from crossdock_solver.rl.networks import NumpyMLP
from crossdock_solver.rl.replay_buffer import ReplayBuffer


MAX_COMPOUND_SLOTS = 9
DESTINATION_WINDOW = 3
EXTRA_FEATURES = 7
OBS_SIZE = MAX_COMPOUND_SLOTS * DESTINATION_WINDOW + MAX_COMPOUND_SLOTS + DESTINATION_WINDOW + EXTRA_FEATURES


@dataclass(frozen=True)
class CargoMatrixRLConfig:
    """VAA-ordered destination-agent RL with explicit inbound cargo matrix state."""

    episodes: int = 150
    hidden: int = 64
    lr: float = 1e-3
    batch_size: int = 32
    buffer_capacity: int = 10_000
    warmup: int = 16
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.97
    seed: int | None = None
    name: str = "CargoMatrix-RL"
    window_strategy: str = "vaa"


@dataclass
class CargoMatrixRLResult:
    run: BaselineRun
    training_rewards: list[float] = field(default_factory=list)
    best_by_episode: list[float] = field(default_factory=list)


def run_cargo_matrix_rl(
    instance: CrossDockInstance,
    config: CargoMatrixRLConfig | None = None,
    *,
    initial_solution: Solution | None = None,
) -> CargoMatrixRLResult:
    """Train and evaluate a cargo-matrix destination-agent RL baseline.

    The state exposes a fixed-size inbound cargo matrix:
    up to 9 compound trucks by the current 3-destination VAA window. Each
    destination agent still chooses a carrier truck; the learned carrier
    assignment is completed by the same release-time greedy scheduler used by
    the first destination-agent RL baseline.
    """

    if len(instance.compound_trucks) > MAX_COMPOUND_SLOTS:
        raise ValueError(
            f"CargoMatrix-RL supports at most {MAX_COMPOUND_SLOTS} compound trucks"
        )

    config = config or CargoMatrixRLConfig()
    rng = random.Random(config.seed)
    np_seed = 0 if config.seed is None else config.seed
    start = time.perf_counter()

    reference_solution = initial_solution.copy() if initial_solution is not None else vaa_solution(instance)
    reference_result = evaluate_solution(instance, reference_solution)
    destination_order = _destination_order_for_strategy(
        instance,
        reference_solution,
        strategy=config.window_strategy,
    )

    best = reference_solution.copy()
    best_result = reference_result

    net = NumpyMLP(
        obs_size=OBS_SIZE,
        hidden=config.hidden,
        n_actions=len(instance.all_trucks),
        lr=config.lr,
        seed=np_seed,
    )
    buffer = ReplayBuffer(config.buffer_capacity, OBS_SIZE, seed=np_seed)

    epsilon = config.epsilon_start
    training_rewards: list[float] = []
    best_by_episode: list[float] = []

    for _ in range(config.episodes):
        carrier_by_destination, transitions = _rollout_cargo_matrix_agents(
            instance,
            destination_order,
            net,
            rng,
            epsilon=epsilon,
        )
        candidate = _build_solution_from_destination_carriers(instance, carrier_by_destination)
        candidate_result = evaluate_solution(instance, candidate)
        reward = (reference_result.makespan - candidate_result.makespan) / reference_result.makespan

        for obs, action in transitions:
            buffer.push(obs, action, reward)

        if len(buffer) >= config.warmup:
            batch_size = min(config.batch_size, len(buffer))
            obs_batch, action_batch, reward_batch = buffer.sample(batch_size)
            net.update(obs_batch, action_batch, reward_batch)

        if candidate_result.makespan < best_result.makespan:
            best = candidate.copy()
            best_result = candidate_result

        training_rewards.append(float(reward))
        best_by_episode.append(float(best_result.makespan))
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

    run = BaselineRun(
        name=f"{config.name}-{config.episodes}",
        solution=best,
        result=best_result,
        runtime_sec=time.perf_counter() - start,
        samples=config.episodes,
    )
    return CargoMatrixRLResult(
        run=run,
        training_rewards=training_rewards,
        best_by_episode=best_by_episode,
    )


def cargo_matrix_rl_solution(
    instance: CrossDockInstance,
    *,
    seed: int | None = None,
    episodes: int = 150,
) -> Solution:
    return run_cargo_matrix_rl(
        instance,
        CargoMatrixRLConfig(episodes=episodes, seed=seed),
    ).run.solution


def run_topload_cargo_matrix_rl(
    instance: CrossDockInstance,
    config: CargoMatrixRLConfig | None = None,
    *,
    initial_solution: Solution | None = None,
) -> CargoMatrixRLResult:
    config = config or CargoMatrixRLConfig()
    top_load_config = CargoMatrixRLConfig(
        episodes=config.episodes,
        hidden=config.hidden,
        lr=config.lr,
        batch_size=config.batch_size,
        buffer_capacity=config.buffer_capacity,
        warmup=config.warmup,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.epsilon_decay,
        seed=config.seed,
        name="TopLoad-CargoMatrix-RL",
        window_strategy="top_load",
    )
    return run_cargo_matrix_rl(
        instance,
        top_load_config,
        initial_solution=initial_solution,
    )


def topload_cargo_matrix_rl_solution(
    instance: CrossDockInstance,
    *,
    seed: int | None = None,
    episodes: int = 150,
) -> Solution:
    return run_topload_cargo_matrix_rl(
        instance,
        CargoMatrixRLConfig(episodes=episodes, seed=seed),
    ).run.solution


def _rollout_cargo_matrix_agents(
    instance: CrossDockInstance,
    destination_order: list[DestinationId],
    net: NumpyMLP,
    rng: random.Random,
    *,
    epsilon: float,
) -> tuple[dict[DestinationId, TruckId], list[tuple[np.ndarray, int]]]:
    available_trucks = set(instance.all_trucks)
    carrier_by_destination: dict[DestinationId, TruckId] = {}
    transitions: list[tuple[np.ndarray, int]] = []

    for position, destination in enumerate(destination_order):
        window = destination_order[position : position + DESTINATION_WINDOW]
        obs = _cargo_matrix_observation(
            instance,
            current_destination=destination,
            destination_window=window,
            available_trucks=available_trucks,
            assigned_count=len(carrier_by_destination),
        )
        valid_actions = [
            idx
            for idx, truck in enumerate(instance.all_trucks)
            if truck in available_trucks
        ]
        action = _select_masked_action(net, obs, valid_actions, rng, epsilon=epsilon)
        truck = instance.all_trucks[action]

        carrier_by_destination[destination] = truck
        available_trucks.remove(truck)
        transitions.append((obs, action))

    return carrier_by_destination, transitions


def _select_masked_action(
    net: NumpyMLP,
    obs: np.ndarray,
    valid_actions: list[int],
    rng: random.Random,
    *,
    epsilon: float,
) -> int:
    if rng.random() < epsilon:
        return rng.choice(valid_actions)

    q_values = net.forward(obs)
    return max(valid_actions, key=lambda action: (q_values[action], -action))


def _cargo_matrix_observation(
    instance: CrossDockInstance,
    *,
    current_destination: DestinationId,
    destination_window: list[DestinationId],
    available_trucks: set[TruckId],
    assigned_count: int,
) -> np.ndarray:
    max_units = max(
        (
            instance.unit_amount(compound, destination)
            for compound in instance.compound_trucks
            for destination in instance.destinations
        ),
        default=0.0,
    ) + 1e-9
    max_load = max(_destination_load(instance, d) for d in instance.destinations) + 1e-9
    total_load = sum(_destination_load(instance, d) for d in instance.destinations) + 1e-9

    cargo_features: list[float] = []
    for row in range(MAX_COMPOUND_SLOTS):
        compound = instance.compound_trucks[row] if row < len(instance.compound_trucks) else None
        for col in range(DESTINATION_WINDOW):
            if compound is None or col >= len(destination_window):
                cargo_features.append(0.0)
            else:
                amount = instance.unit_amount(compound, destination_window[col])
                cargo_features.append(amount / max_units)

    compound_available_mask = [
        1.0 if row < len(instance.compound_trucks) and instance.compound_trucks[row] in available_trucks else 0.0
        for row in range(MAX_COMPOUND_SLOTS)
    ]
    destination_window_mask = [
        1.0 if col < len(destination_window) else 0.0
        for col in range(DESTINATION_WINDOW)
    ]
    min_outbound_tee = min(
        (
            instance.enter_time[truck] + instance.leave_time[truck]
            for truck in available_trucks
            if truck in instance.outbound_trucks
        ),
        default=0.0,
    )
    max_tee = max(
        instance.enter_time[truck] + instance.leave_time[truck]
        for truck in instance.all_trucks
    ) + 1e-9
    best_centrality = min(
        sum(instance.travel(door, other) for other in instance.doors)
        for door in instance.doors
    )
    worst_centrality = max(
        sum(instance.travel(door, other) for other in instance.doors)
        for door in instance.doors
    ) + 1e-9

    extra = [
        instance.destination_index[current_destination] / max(1, len(instance.destinations) - 1),
        _destination_load(instance, current_destination) / max_load,
        _destination_load(instance, current_destination) / total_load,
        sum(1 for truck in instance.compound_trucks if truck in available_trucks)
        / max(1, len(instance.compound_trucks)),
        sum(1 for truck in instance.outbound_trucks if truck in available_trucks)
        / max(1, len(instance.outbound_trucks)),
        assigned_count / max(1, len(instance.destinations)),
        1.0 - (min_outbound_tee / max_tee) + (1.0 - best_centrality / worst_centrality),
    ]

    obs = np.array(
        [
            *cargo_features,
            *compound_available_mask,
            *destination_window_mask,
            *extra,
        ],
        dtype=np.float32,
    )
    if obs.shape != (OBS_SIZE,):
        raise AssertionError(f"CargoMatrix-RL obs size mismatch: {obs.shape}, expected {OBS_SIZE}")
    return obs


def _destination_order_for_strategy(
    instance: CrossDockInstance,
    solution: Solution,
    *,
    strategy: str,
) -> list[DestinationId]:
    if strategy == "vaa":
        return _vaa_destination_order(instance, solution)
    if strategy == "top_load":
        return _top_load_destination_order(instance, solution)
    raise ValueError(f"unknown cargo matrix window_strategy {strategy!r}")


def _vaa_destination_order(
    instance: CrossDockInstance,
    solution: Solution,
) -> list[DestinationId]:
    seen: set[DestinationId] = set()
    order: list[DestinationId] = []
    for truck in instance.all_trucks:
        destination = solution.truck_destination(truck)
        if destination not in seen:
            order.append(destination)
            seen.add(destination)

    for destination in instance.destinations:
        if destination not in seen:
            order.append(destination)
    return order


def _top_load_destination_order(
    instance: CrossDockInstance,
    solution: Solution,
) -> list[DestinationId]:
    vaa_order = _vaa_destination_order(instance, solution)
    vaa_rank = {destination: idx for idx, destination in enumerate(vaa_order)}
    return sorted(
        instance.destinations,
        key=lambda destination: (
            -_destination_load(instance, destination),
            vaa_rank[destination],
        ),
    )
