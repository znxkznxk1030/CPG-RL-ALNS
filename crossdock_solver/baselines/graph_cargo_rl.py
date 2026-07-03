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
from crossdock_solver.data.instance import CrossDockInstance, DestinationId, DoorId, TruckId
from crossdock_solver.rl.networks import NumpyMLP
from crossdock_solver.rl.replay_buffer import ReplayBuffer


TRUCK_FEATURES = 8
DESTINATION_FEATURES = 6
DOOR_FEATURES = 8
CARGO_EDGE_FEATURES = 5
TRAVEL_EDGE_FEATURES = 3
GLOBAL_FEATURES = 13
POOL_STATS = 4
GRAPH_OBS_SIZE = (
    POOL_STATS
    * (
        TRUCK_FEATURES
        + DESTINATION_FEATURES
        + DOOR_FEATURES
        + CARGO_EDGE_FEATURES
        + TRAVEL_EDGE_FEATURES
    )
    + GLOBAL_FEATURES
)


@dataclass(frozen=True)
class GraphCargoRLConfig:
    """Variable-size graph-state destination-agent RL configuration."""

    episodes: int = 150
    hidden: int = 96
    lr: float = 1e-3
    batch_size: int = 32
    buffer_capacity: int = 10_000
    warmup: int = 16
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.97
    seed: int | None = None
    name: str = "GraphCargoMatrix-RL"


@dataclass(frozen=True)
class GraphState:
    truck_nodes: np.ndarray
    destination_nodes: np.ndarray
    door_nodes: np.ndarray
    cargo_edges: np.ndarray
    travel_edges: np.ndarray
    global_features: np.ndarray


@dataclass(frozen=True)
class DoorProfile:
    finish: dict[DoorId, float]
    load: dict[DoorId, float]
    utilization: dict[DoorId, float]
    assigned_count: dict[DoorId, int]
    critical_door: DoorId


@dataclass
class GraphCargoRLResult:
    run: BaselineRun
    training_rewards: list[float] = field(default_factory=list)
    best_by_episode: list[float] = field(default_factory=list)


def run_graph_cargo_rl(
    instance: CrossDockInstance,
    config: GraphCargoRLConfig | None = None,
    *,
    initial_solution: Solution | None = None,
) -> GraphCargoRLResult:
    """Train a variable-size graph-state destination-agent RL baseline.

    The graph state has truck, destination, and door nodes plus cargo and
    door-travel edges. Node/edge counts vary with the instance and remaining
    destinations. A shared pooling encoder converts the graph to a fixed-size
    embedding for the lightweight NumPy MLP policy.
    """

    config = config or GraphCargoRLConfig()
    rng = random.Random(config.seed)
    np_seed = 0 if config.seed is None else config.seed
    start = time.perf_counter()

    reference_solution = initial_solution.copy() if initial_solution is not None else vaa_solution(instance)
    reference_result = evaluate_solution(instance, reference_solution)
    destination_order = _graph_destination_order(instance, reference_solution)

    best = reference_solution.copy()
    best_result = reference_result

    net = NumpyMLP(
        obs_size=GRAPH_OBS_SIZE,
        hidden=config.hidden,
        n_actions=len(instance.all_trucks),
        lr=config.lr,
        seed=np_seed,
    )
    buffer = ReplayBuffer(config.buffer_capacity, GRAPH_OBS_SIZE, seed=np_seed)

    epsilon = config.epsilon_start
    training_rewards: list[float] = []
    best_by_episode: list[float] = []

    for _ in range(config.episodes):
        carrier_by_destination, transitions = _rollout_graph_agents(
            instance,
            destination_order,
            reference_solution,
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
    return GraphCargoRLResult(
        run=run,
        training_rewards=training_rewards,
        best_by_episode=best_by_episode,
    )


def graph_cargo_rl_solution(
    instance: CrossDockInstance,
    *,
    seed: int | None = None,
    episodes: int = 150,
) -> Solution:
    return run_graph_cargo_rl(
        instance,
        GraphCargoRLConfig(episodes=episodes, seed=seed),
    ).run.solution


def _rollout_graph_agents(
    instance: CrossDockInstance,
    destination_order: list[DestinationId],
    reference_solution: Solution,
    net: NumpyMLP,
    rng: random.Random,
    *,
    epsilon: float,
) -> tuple[dict[DestinationId, TruckId], list[tuple[np.ndarray, int]]]:
    available_trucks = set(instance.all_trucks)
    carrier_by_destination: dict[DestinationId, TruckId] = {}
    transitions: list[tuple[np.ndarray, int]] = []

    for position, destination in enumerate(destination_order):
        remaining_destinations = destination_order[position:]
        graph_state = _build_graph_state(
            instance,
            current_destination=destination,
            remaining_destinations=remaining_destinations,
            available_trucks=available_trucks,
            assigned_count=len(carrier_by_destination),
            reference_solution=reference_solution,
            carrier_by_destination=carrier_by_destination,
        )
        obs = _encode_graph_state(graph_state)
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


def _build_graph_state(
    instance: CrossDockInstance,
    *,
    current_destination: DestinationId,
    remaining_destinations: list[DestinationId],
    available_trucks: set[TruckId],
    assigned_count: int,
    reference_solution: Solution | None = None,
    carrier_by_destination: dict[DestinationId, TruckId] | None = None,
) -> GraphState:
    max_tee = max(
        instance.enter_time[truck] + instance.leave_time[truck]
        for truck in instance.all_trucks
    ) + 1e-9
    max_enter = max(instance.enter_time[truck] for truck in instance.all_trucks) + 1e-9
    max_leave = max(instance.leave_time[truck] for truck in instance.all_trucks) + 1e-9
    max_units = max(
        (
            instance.unit_amount(compound, destination)
            for compound in instance.compound_trucks
            for destination in instance.destinations
        ),
        default=0.0,
    ) + 1e-9
    max_handling = max(
        (
            instance.handling_time(compound, destination)
            for compound in instance.compound_trucks
            for destination in instance.destinations
        ),
        default=0.0,
    ) + 1e-9
    max_load = max(_destination_load(instance, d) for d in instance.destinations) + 1e-9
    total_load = sum(_destination_load(instance, d) for d in instance.destinations) + 1e-9

    centrality = {
        door: sum(instance.travel(door, other) for other in instance.doors)
        for door in instance.doors
    }
    worst_centrality = max(centrality.values()) + 1e-9
    max_travel = max(
        instance.travel(source, target)
        for source in instance.doors
        for target in instance.doors
    ) + 1e-9
    reference_solution = reference_solution or vaa_solution(instance)
    carrier_by_destination = carrier_by_destination or {}
    door_profile = _estimate_door_profile(
        instance,
        reference_solution=reference_solution,
        carrier_by_destination=carrier_by_destination,
    )
    max_door_finish = max(door_profile.finish.values(), default=0.0)
    max_door_load = max(door_profile.load.values(), default=0.0)
    door_time_scale = max(max_door_finish, max_door_load, max_tee, 1e-9)
    max_assigned_count = max(1, len(instance.all_trucks))

    remaining_set = set(remaining_destinations)

    truck_nodes = np.array(
        [
            _truck_node_features(
                instance,
                truck,
                current_destination,
                remaining_set,
                available_trucks,
                max_tee=max_tee,
                max_enter=max_enter,
                max_leave=max_leave,
                max_units=max_units,
                max_handling=max_handling,
            )
            for truck in instance.all_trucks
        ],
        dtype=np.float32,
    )

    destination_nodes = np.array(
        [
            _destination_node_features(
                instance,
                destination,
                current_destination,
                max_load=max_load,
                total_load=total_load,
            )
            for destination in remaining_destinations
        ],
        dtype=np.float32,
    )

    door_nodes = np.array(
        [
            [
                instance.door_index[door] / max(1, len(instance.doors) - 1),
                centrality[door] / worst_centrality,
                1.0 - centrality[door] / worst_centrality,
                door_profile.finish[door] / door_time_scale,
                door_profile.load[door] / door_time_scale,
                door_profile.utilization[door],
                door_profile.assigned_count[door] / max_assigned_count,
                1.0 if door == door_profile.critical_door else 0.0,
            ]
            for door in instance.doors
        ],
        dtype=np.float32,
    )

    cargo_edges = np.array(
        [
            _cargo_edge_features(
                instance,
                compound,
                destination,
                current_destination,
                available_trucks,
                max_units=max_units,
                max_handling=max_handling,
                max_load=max_load,
            )
            for compound in instance.compound_trucks
            for destination in remaining_destinations
        ],
        dtype=np.float32,
    )

    travel_edges = np.array(
        [
            [
                centrality[source] / worst_centrality,
                centrality[target] / worst_centrality,
                instance.travel(source, target) / max_travel,
            ]
            for source in instance.doors
            for target in instance.doors
        ],
        dtype=np.float32,
    )

    min_outbound_tee = min(
        (
            instance.enter_time[truck] + instance.leave_time[truck]
            for truck in available_trucks
            if truck in instance.outbound_trucks
        ),
        default=0.0,
    )
    finish_values = np.array(list(door_profile.finish.values()), dtype=np.float32)
    load_values = np.array(list(door_profile.load.values()), dtype=np.float32)
    global_features = np.array(
        [
            len(instance.compound_trucks) / max(1, len(instance.all_trucks)),
            len(instance.outbound_trucks) / max(1, len(instance.all_trucks)),
            len(instance.doors) / max(1, len(instance.all_trucks)),
            assigned_count / max(1, len(instance.destinations)),
            len(remaining_destinations) / max(1, len(instance.destinations)),
            sum(1 for truck in instance.compound_trucks if truck in available_trucks)
            / max(1, len(instance.compound_trucks)),
            sum(1 for truck in instance.outbound_trucks if truck in available_trucks)
            / max(1, len(instance.outbound_trucks)),
            1.0 - min_outbound_tee / max_tee,
            float(np.max(finish_values)) / door_time_scale,
            float(np.mean(finish_values)) / door_time_scale,
            float(np.std(finish_values)) / door_time_scale,
            float(np.max(load_values)) / door_time_scale,
            float(np.max(load_values) - np.min(load_values)) / door_time_scale,
        ],
        dtype=np.float32,
    )

    return GraphState(
        truck_nodes=truck_nodes,
        destination_nodes=destination_nodes,
        door_nodes=door_nodes,
        cargo_edges=cargo_edges,
        travel_edges=travel_edges,
        global_features=global_features,
    )


def _truck_node_features(
    instance: CrossDockInstance,
    truck: TruckId,
    current_destination: DestinationId,
    remaining_destinations: set[DestinationId],
    available_trucks: set[TruckId],
    *,
    max_tee: float,
    max_enter: float,
    max_leave: float,
    max_units: float,
    max_handling: float,
) -> list[float]:
    is_compound = truck in instance.compound_index
    current_units = instance.unit_amount(truck, current_destination) if is_compound else 0.0
    current_handling = instance.handling_time(truck, current_destination) if is_compound else 0.0
    remaining_units = (
        sum(instance.unit_amount(truck, destination) for destination in remaining_destinations)
        if is_compound
        else 0.0
    )
    return [
        1.0 if is_compound else 0.0,
        0.0 if is_compound else 1.0,
        1.0 if truck in available_trucks else 0.0,
        (instance.enter_time[truck] + instance.leave_time[truck]) / max_tee,
        instance.enter_time[truck] / max_enter,
        instance.leave_time[truck] / max_leave,
        current_units / max_units,
        (current_handling / max_handling + remaining_units / max_units) / 2.0,
    ]


def _destination_node_features(
    instance: CrossDockInstance,
    destination: DestinationId,
    current_destination: DestinationId,
    *,
    max_load: float,
    total_load: float,
) -> list[float]:
    load = _destination_load(instance, destination)
    source_units = [
        instance.unit_amount(compound, destination)
        for compound in instance.compound_trucks
    ]
    total_units = sum(source_units) + 1e-9
    return [
        1.0 if destination == current_destination else 0.0,
        instance.destination_index[destination] / max(1, len(instance.destinations) - 1),
        load / max_load,
        load / total_load,
        sum(1 for units in source_units if units > 0.0) / max(1, len(instance.compound_trucks)),
        max(source_units, default=0.0) / total_units,
    ]


def _cargo_edge_features(
    instance: CrossDockInstance,
    compound: TruckId,
    destination: DestinationId,
    current_destination: DestinationId,
    available_trucks: set[TruckId],
    *,
    max_units: float,
    max_handling: float,
    max_load: float,
) -> list[float]:
    return [
        1.0 if compound in available_trucks else 0.0,
        1.0 if destination == current_destination else 0.0,
        instance.unit_amount(compound, destination) / max_units,
        instance.handling_time(compound, destination) / max_handling,
        _destination_load(instance, destination) / max_load,
    ]


def _encode_graph_state(graph_state: GraphState) -> np.ndarray:
    obs = np.array(
        [
            *_pool(graph_state.truck_nodes, TRUCK_FEATURES),
            *_pool(graph_state.destination_nodes, DESTINATION_FEATURES),
            *_pool(graph_state.door_nodes, DOOR_FEATURES),
            *_pool(graph_state.cargo_edges, CARGO_EDGE_FEATURES),
            *_pool(graph_state.travel_edges, TRAVEL_EDGE_FEATURES),
            *graph_state.global_features,
        ],
        dtype=np.float32,
    )
    if obs.shape != (GRAPH_OBS_SIZE,):
        raise AssertionError(f"GraphCargoMatrix-RL obs size mismatch: {obs.shape}, expected {GRAPH_OBS_SIZE}")
    return obs


def _estimate_door_profile(
    instance: CrossDockInstance,
    *,
    reference_solution: Solution,
    carrier_by_destination: dict[DestinationId, TruckId],
) -> DoorProfile:
    projected_carriers = _project_carrier_assignment(
        instance,
        reference_solution=reference_solution,
        carrier_by_destination=carrier_by_destination,
    )
    projected_solution = _build_solution_from_destination_carriers(
        instance,
        projected_carriers,
    )
    result = evaluate_solution(instance, projected_solution)
    assigned_count = {
        door: sum(
            1
            for truck in instance.all_trucks
            if projected_solution.truck_door(truck) == door
            and projected_solution.truck_destination(truck) in carrier_by_destination
        )
        for door in instance.doors
    }
    return DoorProfile(
        finish=result.door_finish,
        load=result.door_load,
        utilization=result.door_utilization,
        assigned_count=assigned_count,
        critical_door=result.critical_door,
    )


def _project_carrier_assignment(
    instance: CrossDockInstance,
    *,
    reference_solution: Solution,
    carrier_by_destination: dict[DestinationId, TruckId],
) -> dict[DestinationId, TruckId]:
    reference_carriers = reference_solution.destination_carriers()
    projected: dict[DestinationId, TruckId] = dict(carrier_by_destination)
    used_trucks = set(projected.values())

    for destination in instance.destinations:
        if destination in projected:
            continue
        reference_truck = reference_carriers[destination]
        if reference_truck in used_trucks:
            continue
        projected[destination] = reference_truck
        used_trucks.add(reference_truck)

    remaining_destinations = [
        destination
        for destination in instance.destinations
        if destination not in projected
    ]
    remaining_trucks = [
        truck
        for truck in instance.all_trucks
        if truck not in used_trucks
    ]
    for destination, truck in zip(remaining_destinations, remaining_trucks, strict=True):
        projected[destination] = truck

    return projected


def _pool(values: np.ndarray, feature_count: int) -> list[float]:
    if values.size == 0:
        return [0.0] * (POOL_STATS * feature_count)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return [
        *np.mean(values, axis=0).tolist(),
        *np.max(values, axis=0).tolist(),
        *np.min(values, axis=0).tolist(),
        *np.std(values, axis=0).tolist(),
    ]


def _graph_destination_order(
    instance: CrossDockInstance,
    solution: Solution,
) -> list[DestinationId]:
    carriers = solution.destination_carriers()
    carrier_rank = {
        destination: instance.all_trucks.index(carrier)
        for destination, carrier in carriers.items()
    }
    return sorted(
        instance.destinations,
        key=lambda destination: (
            -_destination_load(instance, destination),
            carrier_rank.get(destination, len(instance.all_trucks)),
            destination,
        ),
    )
