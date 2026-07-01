from __future__ import annotations

from dataclasses import dataclass, field
import random
import time

import numpy as np

from crossdock_solver.baselines.random_baseline import BaselineRun
from crossdock_solver.baselines.vaa import (
    _compound_destination_cost,
    _destination_load,
    _destination_ready_at_door,
    vaa_solution,
)
from crossdock_solver.core.evaluator import evaluate_solution
from crossdock_solver.core.feasibility import check_feasible
from crossdock_solver.core.solution import Solution
from crossdock_solver.data.instance import CrossDockInstance, DestinationId, DoorId, TruckId
from crossdock_solver.rl.networks import NumpyMLP
from crossdock_solver.rl.replay_buffer import ReplayBuffer


OBS_SIZE = 11


@dataclass(frozen=True)
class DestinationAgentRLConfig:
    """Shared-network destination-agent RL configuration.

    Each destination acts as one agent. Agents share a small NumPy MLP, choose
    one still-available carrier truck, and receive the final team reward.
    """

    episodes: int = 150
    hidden: int = 48
    lr: float = 1e-3
    batch_size: int = 32
    buffer_capacity: int = 10_000
    warmup: int = 16
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.97
    seed: int | None = None
    name: str = "DestAgent-RL"


@dataclass
class DestinationAgentRLResult:
    run: BaselineRun
    training_rewards: list[float] = field(default_factory=list)
    best_by_episode: list[float] = field(default_factory=list)


def run_destination_agent_rl(
    instance: CrossDockInstance,
    config: DestinationAgentRLConfig | None = None,
    *,
    initial_solution: Solution | None = None,
) -> DestinationAgentRLResult:
    """Train and evaluate a destination-agent RL constructive baseline.

    This is not the paper's SA-RL operator selector. It is a separate baseline
    where each destination-agent learns which carrier truck should serve it.
    Door assignment and outbound sequencing are completed by a deterministic
    release-time greedy scheduler.
    """

    config = config or DestinationAgentRLConfig()
    rng = random.Random(config.seed)
    np_seed = 0 if config.seed is None else config.seed
    start = time.perf_counter()

    reference_solution = initial_solution.copy() if initial_solution is not None else vaa_solution(instance)
    reference_result = evaluate_solution(instance, reference_solution)
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
        carrier_by_destination, transitions = _rollout_destination_agents(
            instance,
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
    return DestinationAgentRLResult(
        run=run,
        training_rewards=training_rewards,
        best_by_episode=best_by_episode,
    )


def destination_agent_rl_solution(
    instance: CrossDockInstance,
    *,
    seed: int | None = None,
    episodes: int = 150,
) -> Solution:
    return run_destination_agent_rl(
        instance,
        DestinationAgentRLConfig(episodes=episodes, seed=seed),
    ).run.solution


def _rollout_destination_agents(
    instance: CrossDockInstance,
    net: NumpyMLP,
    rng: random.Random,
    *,
    epsilon: float,
) -> tuple[dict[DestinationId, TruckId], list[tuple[np.ndarray, int]]]:
    available_trucks = set(instance.all_trucks)
    carrier_by_destination: dict[DestinationId, TruckId] = {}
    transitions: list[tuple[np.ndarray, int]] = []

    ordered_destinations = sorted(
        instance.destinations,
        key=lambda destination: (_destination_load(instance, destination), destination),
        reverse=True,
    )

    for destination in ordered_destinations:
        obs = _destination_observation(
            instance,
            destination,
            available_trucks,
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


def _destination_observation(
    instance: CrossDockInstance,
    destination: DestinationId,
    available_trucks: set[TruckId],
    *,
    assigned_count: int,
) -> np.ndarray:
    all_trucks = instance.all_trucks
    destination_load = _destination_load(instance, destination)
    total_load = sum(_destination_load(instance, d) for d in instance.destinations) + 1e-9
    max_destination_load = max(_destination_load(instance, d) for d in instance.destinations) + 1e-9
    compound_costs = [
        _compound_destination_cost(instance, compound, destination)
        for compound in instance.compound_trucks
    ]
    total_compound_work = sum(
        instance.handling_time(compound, d)
        for compound in instance.compound_trucks
        for d in instance.destinations
    ) + 1e-9
    dest_units = [
        instance.unit_amount(compound, destination)
        for compound in instance.compound_trucks
    ]
    total_units = sum(dest_units) + 1e-9
    nonzero_sources = sum(1 for amount in dest_units if amount > 0)
    available_compounds = sum(1 for truck in instance.compound_trucks if truck in available_trucks)
    available_outbounds = sum(1 for truck in instance.outbound_trucks if truck in available_trucks)
    min_tee = min(
        (
            instance.enter_time[truck] + instance.leave_time[truck]
            for truck in available_trucks
            if truck in instance.outbound_trucks
        ),
        default=0.0,
    )
    max_tee = max(
        instance.enter_time[truck] + instance.leave_time[truck]
        for truck in all_trucks
    ) + 1e-9
    best_centrality = min(
        sum(instance.travel(door, other) for other in instance.doors)
        for door in instance.doors
    )
    worst_centrality = max(
        sum(instance.travel(door, other) for other in instance.doors)
        for door in instance.doors
    ) + 1e-9

    obs = np.array(
        [
            instance.destination_index[destination] / max(1, len(instance.destinations) - 1),
            destination_load / max_destination_load,
            destination_load / total_load,
            min(compound_costs) / total_compound_work,
            (sum(compound_costs) / len(compound_costs)) / total_compound_work,
            max(dest_units) / total_units,
            nonzero_sources / max(1, len(instance.compound_trucks)),
            available_compounds / max(1, len(instance.compound_trucks)),
            available_outbounds / max(1, len(instance.outbound_trucks)),
            assigned_count / max(1, len(instance.destinations)),
            1.0 - (min_tee / max_tee) + (1.0 - best_centrality / worst_centrality),
        ],
        dtype=np.float32,
    )
    return obs


def _build_solution_from_destination_carriers(
    instance: CrossDockInstance,
    carrier_by_destination: dict[DestinationId, TruckId],
) -> Solution:
    destination_by_truck = {
        truck: destination
        for destination, truck in carrier_by_destination.items()
    }
    solution = Solution(
        compound_assignment={},
        outbound_assignment={},
        door_sequences={door: [] for door in instance.doors},
    )

    _assign_compound_doors(instance, solution, destination_by_truck)
    _assign_outbounds_by_release_time(instance, solution, destination_by_truck)
    check_feasible(instance, solution)
    return solution


def _assign_compound_doors(
    instance: CrossDockInstance,
    solution: Solution,
    destination_by_truck: dict[TruckId, DestinationId],
) -> None:
    central_doors = sorted(
        instance.doors,
        key=lambda door: (
            sum(instance.travel(door, other) for other in instance.doors),
            door,
        ),
    )
    compounds = sorted(
        instance.compound_trucks,
        key=lambda truck: (
            _compound_destination_cost(instance, truck, destination_by_truck[truck]),
            destination_by_truck[truck],
            truck,
        ),
        reverse=True,
    )
    for truck, door in zip(compounds, central_doors):
        solution.compound_assignment[truck] = (destination_by_truck[truck], door)


def _assign_outbounds_by_release_time(
    instance: CrossDockInstance,
    solution: Solution,
    destination_by_truck: dict[TruckId, DestinationId],
) -> None:
    door_finish = {door: 0.0 for door in instance.doors}
    for compound, (destination, door) in solution.compound_assignment.items():
        own_unload = (
            instance.enter_time[compound]
            + sum(
                instance.handling_time(compound, other)
                for other in instance.destinations
                if other != destination
            )
        )
        ready = _destination_ready_at_door(
            instance,
            solution,
            destination,
            door,
            carrier=compound,
        )
        load = sum(
            instance.handling_time(source, destination)
            for source in instance.compound_trucks
            if source != compound
        )
        door_finish[door] = max(own_unload, ready) + load + instance.leave_time[compound]

    outbounds = sorted(
        instance.outbound_trucks,
        key=lambda truck: (_destination_load(instance, destination_by_truck[truck]), truck),
        reverse=True,
    )
    for truck in outbounds:
        destination = destination_by_truck[truck]
        best_door = min(
            instance.doors,
            key=lambda door: (
                _outbound_finish_if_assigned(instance, solution, truck, destination, door, door_finish[door]),
                door_finish[door],
                door,
            ),
        )
        solution.outbound_assignment[truck] = (destination, best_door)
        solution.door_sequences[best_door].append(truck)
        door_finish[best_door] = _outbound_finish_if_assigned(
            instance,
            solution,
            truck,
            destination,
            best_door,
            door_finish[best_door],
        )


def _outbound_finish_if_assigned(
    instance: CrossDockInstance,
    solution: Solution,
    truck: TruckId,
    destination: DestinationId,
    door: DoorId,
    previous_finish: float,
) -> float:
    ready = _destination_ready_at_door(instance, solution, destination, door, carrier=truck)
    load = _destination_load(instance, destination)
    start = max(previous_finish, ready)
    return start + instance.enter_time[truck] + load + instance.leave_time[truck]

