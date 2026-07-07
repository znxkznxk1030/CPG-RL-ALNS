from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Callable, TypeVar

from crossdock_solver.alns.acceptance import accept_by_sa
from crossdock_solver.baselines.random_baseline import BaselineRun
from crossdock_solver.baselines.vaa import vaa_solution
from crossdock_solver.core.evaluator import ScheduleResult, evaluate_solution
from crossdock_solver.core.feasibility import check_feasible
from crossdock_solver.core.solution import Solution
from crossdock_solver.data.instance import CrossDockInstance, DestinationId, DoorId, TruckId


Neighborhood = Callable[[CrossDockInstance, Solution, random.Random], Solution]
T = TypeVar("T")


NEIGHBORHOODS: dict[str, Neighborhood] = {}


@dataclass
class PaperSARLConfig:
    """Paper-style Q-learning SA configuration.

    The default threshold tuple matches the paper's SA-RL5 setting:
    (lambda_1, lambda_2, lambda_3, lambda_4) = (5, 10, 15, 20).
    """

    max_iterations: int = 300
    thresholds: tuple[int, int, int, int] = (5, 10, 15, 20)
    learning_rate: float = 0.3
    discount_factor: float = 0.9
    random_selection_prob: float = 0.10
    roulette_selection_prob: float = 0.20
    initial_temperature: float | None = None
    cooling_rate: float = 0.995
    tardiness_weight: float = 0.0
    seed: int | None = None
    name: str = "Paper-SA-RL5"


def run_paper_sa_rl(
    instance: CrossDockInstance,
    config: PaperSARLConfig | None = None,
    *,
    initial_solution: Solution | None = None,
) -> BaselineRun:
    """Run a paper-style RL-SA baseline.

    The implementation follows the paper's main mechanism: VAA initialization,
    SA acceptance, reward 1 when the generated solution is no worse than the
    current solution, and Q-learning over no-improvement states.
    """

    config = config or PaperSARLConfig()
    rng = random.Random(config.seed)
    start_time = time.perf_counter()

    current = initial_solution.copy() if initial_solution is not None else vaa_solution(instance)
    current_result = evaluate_solution(instance, current)
    best = current.copy()
    best_result = current_result
    temperature = (
        config.initial_temperature
        if config.initial_temperature is not None
        else max(1.0, 0.05 * current_result.makespan)
    )

    actions = tuple(NEIGHBORHOODS)
    q_values = {
        (state, action): 0.0
        for state in range(1, 6)
        for action in actions
    }
    no_improvement_count = 0

    for _ in range(config.max_iterations):
        state = _state_from_no_improvement(no_improvement_count, config.thresholds)
        action = _select_action(
            state,
            actions,
            q_values,
            rng,
            random_prob=config.random_selection_prob,
            roulette_prob=config.roulette_selection_prob,
        )

        candidate = NEIGHBORHOODS[action](instance, current, rng)
        candidate_result = evaluate_solution(instance, candidate)

        reward = 1.0 if candidate_result.makespan <= current_result.makespan else 0.0
        next_no_improvement = 0 if reward > 0 else no_improvement_count + 1
        next_state = _state_from_no_improvement(next_no_improvement, config.thresholds)
        q_values[(state, action)] += config.learning_rate * (
            reward
            + config.discount_factor * max(q_values[(next_state, next_action)] for next_action in actions)
            - q_values[(state, action)]
        )

        accepted = accept_by_sa(
            current_result.makespan,
            candidate_result.makespan,
            temperature,
            rng,
        )
        if accepted:
            current = candidate
            current_result = candidate_result

        no_improvement_count = next_no_improvement
        if candidate_result.makespan < best_result.makespan:
            best = candidate.copy()
            best_result = candidate_result

        temperature *= config.cooling_rate

    return BaselineRun(
        name=f"{config.name}-{config.max_iterations}",
        solution=best,
        result=best_result,
        runtime_sec=time.perf_counter() - start_time,
        samples=config.max_iterations,
    )


def paper_sa_rl5(instance: CrossDockInstance, *, seed: int | None = None) -> BaselineRun:
    return run_paper_sa_rl(instance, PaperSARLConfig(seed=seed, thresholds=(5, 10, 15, 20)))


def paper_sa_rl6(instance: CrossDockInstance, *, seed: int | None = None) -> BaselineRun:
    return run_paper_sa_rl(
        instance,
        PaperSARLConfig(seed=seed, thresholds=(10, 20, 50, 100), name="Paper-SA-RL6"),
    )


def _state_from_no_improvement(
    no_improvement_count: int,
    thresholds: tuple[int, int, int, int],
) -> int:
    first, second, third, fourth = thresholds
    if no_improvement_count <= first:
        return 1
    if no_improvement_count <= second:
        return 2
    if no_improvement_count <= third:
        return 3
    if no_improvement_count <= fourth:
        return 4
    return 5


def _select_action(
    state: int,
    actions: tuple[str, ...],
    q_values: dict[tuple[int, str], float],
    rng: random.Random,
    *,
    random_prob: float,
    roulette_prob: float,
) -> str:
    draw = rng.random()
    if draw < random_prob:
        return rng.choice(actions)
    if draw < random_prob + roulette_prob:
        weights = [max(0.0, q_values[(state, action)]) + 1e-9 for action in actions]
        return _weighted_choice(actions, weights, rng)

    best_value = max(q_values[(state, action)] for action in actions)
    best_actions = [action for action in actions if math.isclose(q_values[(state, action)], best_value)]
    return rng.choice(best_actions)


def _swap_destinations(
    instance: CrossDockInstance,
    solution: Solution,
    rng: random.Random,
) -> Solution:
    candidate = solution.copy()
    trucks = [*instance.compound_trucks, *instance.outbound_trucks]
    first, second = rng.sample(trucks, 2)
    first_destination = candidate.truck_destination(first)
    second_destination = candidate.truck_destination(second)
    _set_destination(candidate, first, second_destination)
    _set_destination(candidate, second, first_destination)
    check_feasible(instance, candidate)
    return candidate


def _swap_compound_doors(
    instance: CrossDockInstance,
    solution: Solution,
    rng: random.Random,
) -> Solution:
    if len(instance.compound_trucks) < 2:
        return solution.copy()
    candidate = solution.copy()
    first, second = rng.sample(instance.compound_trucks, 2)
    first_destination, first_door = candidate.compound_assignment[first]
    second_destination, second_door = candidate.compound_assignment[second]
    candidate.compound_assignment[first] = (first_destination, second_door)
    candidate.compound_assignment[second] = (second_destination, first_door)
    check_feasible(instance, candidate)
    return candidate


def _swap_outbound_doors(
    instance: CrossDockInstance,
    solution: Solution,
    rng: random.Random,
) -> Solution:
    if len(instance.outbound_trucks) < 2:
        return solution.copy()

    candidate = solution.copy()
    first, second = rng.sample(instance.outbound_trucks, 2)
    first_destination, first_door = candidate.outbound_assignment[first]
    second_destination, second_door = candidate.outbound_assignment[second]

    if first_door == second_door:
        sequence = candidate.door_sequences[first_door]
        first_pos = sequence.index(first)
        second_pos = sequence.index(second)
        sequence[first_pos], sequence[second_pos] = sequence[second_pos], sequence[first_pos]
        check_feasible(instance, candidate)
        return candidate

    first_pos = candidate.door_sequences[first_door].index(first)
    second_pos = candidate.door_sequences[second_door].index(second)
    _remove_outbound_from_sequence(candidate, first)
    _remove_outbound_from_sequence(candidate, second)

    candidate.outbound_assignment[first] = (first_destination, second_door)
    candidate.outbound_assignment[second] = (second_destination, first_door)
    candidate.door_sequences[second_door].insert(
        min(second_pos, len(candidate.door_sequences[second_door])),
        first,
    )
    candidate.door_sequences[first_door].insert(
        min(first_pos, len(candidate.door_sequences[first_door])),
        second,
    )
    check_feasible(instance, candidate)
    return candidate


def _insert_outbound(
    instance: CrossDockInstance,
    solution: Solution,
    rng: random.Random,
) -> Solution:
    if not instance.outbound_trucks:
        return solution.copy()
    truck = rng.choice(instance.outbound_trucks)
    target_door = rng.choice(instance.doors)
    return _move_outbound_to_door(instance, solution, truck, target_door, rng)


def _insert_compound_to_door_weighted(
    instance: CrossDockInstance,
    solution: Solution,
    rng: random.Random,
) -> Solution:
    truck_weights = [
        _compound_unload_workload(instance, solution, truck)
        for truck in instance.compound_trucks
    ]
    truck = _weighted_choice(tuple(instance.compound_trucks), truck_weights, rng)
    door_weights = _centrality_weights(instance)
    target_door = _weighted_choice(tuple(instance.doors), door_weights, rng)
    return _move_compound_to_door(instance, solution, truck, target_door, rng)


def _insert_outbound_to_door_weighted(
    instance: CrossDockInstance,
    solution: Solution,
    rng: random.Random,
) -> Solution:
    if not instance.outbound_trucks:
        return solution.copy()
    truck_weights = [
        _destination_load(instance, solution.outbound_assignment[truck][0])
        for truck in instance.outbound_trucks
    ]
    truck = _weighted_choice(tuple(instance.outbound_trucks), truck_weights, rng)
    door_weights = _centrality_weights(instance)
    target_door = _weighted_choice(tuple(instance.doors), door_weights, rng)
    return _move_outbound_to_door(instance, solution, truck, target_door, rng)


def _insert_compound_to_destination_weighted(
    instance: CrossDockInstance,
    solution: Solution,
    rng: random.Random,
) -> Solution:
    compound_weights = [
        max(_compound_destination_cost(instance, truck, destination) for destination in instance.destinations)
        for truck in instance.compound_trucks
    ]
    compound = _weighted_choice(tuple(instance.compound_trucks), compound_weights, rng)
    destination_costs = [
        _compound_destination_cost(instance, compound, destination)
        for destination in instance.destinations
    ]
    max_cost = max(destination_costs) if destination_costs else 0.0
    destination_weights = [max_cost - cost + 1e-9 for cost in destination_costs]
    destination = _weighted_choice(tuple(instance.destinations), destination_weights, rng)

    current_destination = solution.compound_assignment[compound][0]
    if destination == current_destination:
        return solution.copy()

    carrier = solution.destination_carriers()[destination]
    candidate = solution.copy()
    _set_destination(candidate, compound, destination)
    _set_destination(candidate, carrier, current_destination)
    check_feasible(instance, candidate)
    return candidate


def _move_outbound_to_door(
    instance: CrossDockInstance,
    solution: Solution,
    truck: TruckId,
    target_door: DoorId,
    rng: random.Random,
) -> Solution:
    candidate = solution.copy()
    destination, _ = candidate.outbound_assignment[truck]
    _remove_outbound_from_sequence(candidate, truck)
    candidate.outbound_assignment[truck] = (destination, target_door)
    sequence = candidate.door_sequences.setdefault(target_door, [])
    position = rng.randrange(len(sequence) + 1)
    sequence.insert(position, truck)
    check_feasible(instance, candidate)
    return candidate


def _move_compound_to_door(
    instance: CrossDockInstance,
    solution: Solution,
    truck: TruckId,
    target_door: DoorId,
    rng: random.Random,
) -> Solution:
    candidate = solution.copy()
    destination, current_door = candidate.compound_assignment[truck]
    if target_door == current_door:
        return candidate

    holder = next(
        (
            other
            for other, (_, door) in candidate.compound_assignment.items()
            if other != truck and door == target_door
        ),
        None,
    )
    candidate.compound_assignment[truck] = (destination, target_door)
    if holder is not None:
        holder_destination, _ = candidate.compound_assignment[holder]
        candidate.compound_assignment[holder] = (holder_destination, current_door)
    check_feasible(instance, candidate)
    return candidate


def _remove_outbound_from_sequence(solution: Solution, truck: TruckId) -> None:
    for sequence in solution.door_sequences.values():
        if truck in sequence:
            sequence.remove(truck)
            return


def _set_destination(solution: Solution, truck: TruckId, destination: DestinationId) -> None:
    if truck in solution.compound_assignment:
        _, door = solution.compound_assignment[truck]
        solution.compound_assignment[truck] = (destination, door)
    else:
        _, door = solution.outbound_assignment[truck]
        solution.outbound_assignment[truck] = (destination, door)


def _compound_unload_workload(
    instance: CrossDockInstance,
    solution: Solution,
    compound: TruckId,
) -> float:
    retained_destination = solution.compound_assignment[compound][0]
    return sum(
        instance.handling_time(compound, destination)
        for destination in instance.destinations
        if destination != retained_destination
    )


def _compound_destination_cost(
    instance: CrossDockInstance,
    compound: TruckId,
    destination: DestinationId,
) -> float:
    unload_time = sum(
        instance.handling_time(compound, other_destination)
        for other_destination in instance.destinations
        if other_destination != destination
    )
    load_time = sum(
        instance.handling_time(source, destination)
        for source in instance.compound_trucks
        if source != compound
    )
    return unload_time + 0.5 * load_time


def _destination_load(instance: CrossDockInstance, destination: DestinationId) -> float:
    return sum(
        instance.handling_time(compound, destination)
        for compound in instance.compound_trucks
    )


def _centrality_weights(instance: CrossDockInstance) -> list[float]:
    return [
        1.0 / (sum(instance.travel(door, other) for other in instance.doors) + 1e-9)
        for door in instance.doors
    ]


def _weighted_choice(
    values: tuple[T, ...],
    weights: list[float],
    rng: random.Random,
) -> T:
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    positive_weights = [max(0.0, weight) for weight in weights]
    total = sum(positive_weights)
    if total <= 0:
        return rng.choice(values)

    draw = rng.random() * total
    cumulative = 0.0
    for value, weight in zip(values, positive_weights, strict=True):
        cumulative += weight
        if draw <= cumulative:
            return value
    return values[-1]


NEIGHBORHOODS.update(
    {
        "k1_destination_swap": _swap_destinations,
        "k2_compound_door_swap": _swap_compound_doors,
        "k3_outbound_door_swap": _swap_outbound_doors,
        "k4_outbound_insertion": _insert_outbound,
        "k6_compound_door_insertion": _insert_compound_to_door_weighted,
        "k7_outbound_door_insertion": _insert_outbound_to_door_weighted,
        "k8_compound_destination_insertion": _insert_compound_to_destination_weighted,
    }
)
