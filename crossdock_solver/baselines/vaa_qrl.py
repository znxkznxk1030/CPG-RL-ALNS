from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Iterator

from crossdock_solver.alns.acceptance import accept_by_sa
from crossdock_solver.baselines.paper_sa_rl import (
    NEIGHBORHOODS,
    _state_from_no_improvement,
)
from crossdock_solver.rl.features import (
    OperatorSuccessTracker,
    SearchState,
    build_feature_vector,
    instance_features,
)
from crossdock_solver.rl.selectors import SelectionContext, TabularQSelector
from crossdock_solver.baselines.random_baseline import BaselineRun
from crossdock_solver.baselines.vaa import vaa_solution
from crossdock_solver.core.evaluator import evaluate_solution
from crossdock_solver.core.fast_evaluator import FastEvaluator, FastResult
from crossdock_solver.core.solution import Solution
from crossdock_solver.data.instance import CrossDockInstance, DestinationId, DoorId, TruckId


EPSILON = 1e-9

GUIDED_ACTIONS = (
    "g1_critical_outbound_relocate",
    "g2_critical_destination_swap",
)
TW_GUIDED_ACTIONS = (
    "g3_tardy_truck_relocate",
    "g4_tardy_destination_swap",
)
ACTIONS = (*NEIGHBORHOODS, *GUIDED_ACTIONS)
ACTIONS_TW = (*ACTIONS, *TW_GUIDED_ACTIONS)


@dataclass
class VaaQRLConfig:
    """VAA-initialized Q-learning iterated local search configuration.

    The model keeps the paper's Q-learning operator-selection frame and extends
    it with critical-door guided operators, a new-best shaped reward, restart
    with reheating on stagnation, and a deterministic descent polish applied to
    every new best solution.
    """

    max_iterations: int = 300
    time_budget_sec: float | None = None
    tardiness_weight: float = 0.0
    thresholds: tuple[int, int, int, int] = (5, 10, 15, 20)
    learning_rate: float = 0.3
    discount_factor: float = 0.9
    random_selection_prob: float = 0.10
    roulette_selection_prob: float = 0.20
    initial_temperature: float | None = None
    cooling_rate: float = 0.995
    restart_after: int = 30
    restart_kick: int = 3
    reheat_ratio: float = 0.02
    new_best_reward: float = 2.0
    # Component toggles for the engine ablation (Phase B2). All True reproduces
    # the standard engine byte-for-byte. use_sa_acceptance=False makes acceptance
    # greedy (accept a candidate only when it improves the current solution).
    use_descent: bool = True
    use_sa_acceptance: bool = True
    use_restart: bool = True
    seed: int | None = None
    name: str = "VAA-QRL"


def run_vaa_qrl(
    instance: CrossDockInstance,
    config: VaaQRLConfig | None = None,
    *,
    initial_solution: Solution | None = None,
    selector=None,
) -> BaselineRun:
    """Run the VAA + Q-learning guided iterated local search model.

    Structure:
    1. VAA constructive initial solution, polished by best-improvement descent.
    2. A selector policy picks a move operator among the paper neighborhoods
       plus critical-door guided operators. The default is the paper-style
       tabular Q-learning over no-improvement states; alternative selectors
       (uniform random, pretrained DQN) plug in via `selector`.
    3. SA acceptance with restart-from-best and reheating on stagnation.
    4. Every new best solution is polished by the same descent.
    """

    config = config or VaaQRLConfig()
    rng = random.Random(config.seed)
    start_time = time.perf_counter()
    deadline = (
        start_time + config.time_budget_sec if config.time_budget_sec is not None else None
    )
    fast = FastEvaluator(instance, tardiness_weight=config.tardiness_weight)

    current = initial_solution.copy() if initial_solution is not None else vaa_solution(instance)
    current_result = fast.evaluate(current)
    if config.use_descent:
        current, current_result = _descent(instance, current, current_result, fast, deadline)
    best = current.copy()
    best_result = current_result
    temperature = (
        config.initial_temperature
        if config.initial_temperature is not None
        else max(1.0, 0.05 * current_result.objective)
    )

    def _guided_relocate(inst, solution, move_rng):
        return _critical_outbound_relocate(inst, solution, move_rng, fast)

    def _guided_destination_swap(inst, solution, move_rng):
        return _critical_destination_swap(inst, solution, move_rng, fast)

    def _guided_tardy_relocate(inst, solution, move_rng):
        return _tardy_truck_relocate(inst, solution, move_rng, fast)

    def _guided_tardy_destination_swap(inst, solution, move_rng):
        return _tardy_destination_swap(inst, solution, move_rng, fast)

    guided: dict[str, object] = {
        "g1_critical_outbound_relocate": _guided_relocate,
        "g2_critical_destination_swap": _guided_destination_swap,
        "g3_tardy_truck_relocate": _guided_tardy_relocate,
        "g4_tardy_destination_swap": _guided_tardy_destination_swap,
    }
    actions = (
        tuple(selector.actions) if selector is not None and hasattr(selector, "actions")
        else ACTIONS
    )
    if selector is None:
        selector = TabularQSelector(
            actions,
            learning_rate=config.learning_rate,
            discount_factor=config.discount_factor,
            random_prob=config.random_selection_prob,
            roulette_prob=config.roulette_selection_prob,
        )
    needs_features = getattr(selector, "needs_features", False)
    if needs_features:
        static_features = instance_features(instance)
        tracker = OperatorSuccessTracker(actions)
    initial_temperature = temperature
    no_improvement_count = 0
    since_best = 0

    def _context(iteration: int, state_bin: int) -> SelectionContext:
        if not needs_features:
            return SelectionContext(state_bin=state_bin)
        search_state = SearchState(
            iteration=iteration,
            max_iterations=config.max_iterations,
            temperature=temperature,
            initial_temperature=initial_temperature,
            current=current_result,
            best=best_result,
            no_improvement=no_improvement_count,
            no_improvement_cap=config.thresholds[-1],
            since_best=since_best,
            restart_after=config.restart_after,
            num_trucks=len(instance.all_trucks),
        )
        return SelectionContext(
            state_bin=state_bin,
            features=build_feature_vector(static_features, search_state, tracker),
        )

    for iteration in range(config.max_iterations):
        if (
            config.time_budget_sec is not None
            and time.perf_counter() - start_time >= config.time_budget_sec
        ):
            break
        state = _state_from_no_improvement(no_improvement_count, config.thresholds)
        context = _context(iteration, state)
        action = selector.select(context, rng)

        operator = NEIGHBORHOODS.get(action) or guided[action]
        candidate = operator(instance, current, rng)
        candidate_result = fast.evaluate(candidate)

        if candidate_result.objective < best_result.objective - EPSILON:
            if config.use_descent:
                candidate, candidate_result = _descent(
                    instance, candidate, candidate_result, fast, deadline
                )
            best = candidate.copy()
            best_result = candidate_result
            reward = config.new_best_reward
            since_best = 0
        else:
            reward = 1.0 if candidate_result.objective <= current_result.objective + EPSILON else 0.0
            since_best += 1

        next_no_improvement = (
            0 if candidate_result.objective <= current_result.objective + EPSILON else no_improvement_count + 1
        )
        next_state = _state_from_no_improvement(next_no_improvement, config.thresholds)

        if config.use_sa_acceptance:
            accepted = accept_by_sa(
                current_result.objective, candidate_result.objective, temperature, rng
            )
        else:
            accepted = candidate_result.objective < current_result.objective - EPSILON
        if accepted:
            current = candidate
            current_result = candidate_result

        no_improvement_count = next_no_improvement
        temperature *= config.cooling_rate

        if config.use_restart and since_best >= config.restart_after:
            current = best.copy()
            for _ in range(config.restart_kick):
                kick = NEIGHBORHOODS[rng.choice(tuple(NEIGHBORHOODS))]
                current = kick(instance, current, rng)
            current_result = fast.evaluate(current)
            temperature = max(temperature, config.reheat_ratio * best_result.objective)
            since_best = 0
            no_improvement_count = 0

        if needs_features:
            tracker.update(action, reward > 0.0)
        selector.observe(context, action, reward, _context(iteration + 1, next_state))

    if config.use_descent:
        best, _ = _descent(instance, best, best_result, fast, deadline)

    return BaselineRun(
        name=f"{config.name}-{config.max_iterations}",
        solution=best,
        result=evaluate_solution(instance, best),
        runtime_sec=time.perf_counter() - start_time,
        samples=config.max_iterations,
    )


def vaa_qrl_solution(
    instance: CrossDockInstance,
    *,
    seed: int | None = None,
    max_iterations: int = 300,
) -> Solution:
    return run_vaa_qrl(
        instance,
        VaaQRLConfig(max_iterations=max_iterations, seed=seed),
    ).solution


def _descent(
    instance: CrossDockInstance,
    solution: Solution,
    result: FastResult,
    fast: FastEvaluator,
    deadline: float | None = None,
) -> tuple[Solution, FastResult]:
    """Best-improvement descent over relocation and swap moves."""

    current = solution.copy()
    current_result = result

    while True:
        best_move: Solution | None = None
        best_move_result = current_result
        for move in _descent_moves(instance, current):
            if deadline is not None and time.perf_counter() >= deadline:
                if best_move is not None and best_move_result.objective < current_result.objective:
                    return best_move, best_move_result
                return current, current_result
            move_result = fast.evaluate(move)
            if move_result.objective < best_move_result.objective - EPSILON:
                best_move = move
                best_move_result = move_result
        if best_move is None:
            return current, current_result
        current = best_move
        current_result = best_move_result


def _descent_moves(instance: CrossDockInstance, solution: Solution) -> Iterator[Solution]:
    for truck in instance.outbound_trucks:
        yield from _relocations(instance, solution, truck)

    for first_idx, first in enumerate(instance.compound_trucks):
        for second in instance.compound_trucks[first_idx + 1:]:
            yield _compound_doors_swapped(solution, first, second)

    occupied = {door for _, door in solution.compound_assignment.values()}
    free_doors = [door for door in instance.doors if door not in occupied]
    for compound in instance.compound_trucks:
        destination, _ = solution.compound_assignment[compound]
        for door in free_doors:
            candidate = solution.copy()
            candidate.compound_assignment[compound] = (destination, door)
            yield candidate

    all_trucks = instance.all_trucks
    for first_idx, first in enumerate(all_trucks):
        for second in all_trucks[first_idx + 1:]:
            yield _destinations_swapped(solution, first, second)


def _relocations(
    instance: CrossDockInstance,
    solution: Solution,
    truck: TruckId,
) -> Iterator[Solution]:
    current_door = solution.outbound_assignment[truck][1]
    current_position = solution.door_sequences[current_door].index(truck)
    for door in instance.doors:
        slots = len(solution.door_sequences.get(door, []))
        if door == current_door:
            positions = [p for p in range(slots) if p != current_position]
        else:
            positions = list(range(slots + 1))
        for position in positions:
            yield _relocated(solution, truck, door, position)


def _relocated(solution: Solution, truck: TruckId, door: DoorId, position: int) -> Solution:
    candidate = solution.copy()
    destination, current_door = candidate.outbound_assignment[truck]
    candidate.door_sequences[current_door].remove(truck)
    candidate.outbound_assignment[truck] = (destination, door)
    candidate.door_sequences.setdefault(door, []).insert(position, truck)
    return candidate


def _compound_doors_swapped(solution: Solution, first: TruckId, second: TruckId) -> Solution:
    candidate = solution.copy()
    first_destination, first_door = candidate.compound_assignment[first]
    second_destination, second_door = candidate.compound_assignment[second]
    candidate.compound_assignment[first] = (first_destination, second_door)
    candidate.compound_assignment[second] = (second_destination, first_door)
    return candidate


def _destinations_swapped(solution: Solution, first: TruckId, second: TruckId) -> Solution:
    candidate = solution.copy()
    first_destination = candidate.truck_destination(first)
    second_destination = candidate.truck_destination(second)
    _set_destination(candidate, first, second_destination)
    _set_destination(candidate, second, first_destination)
    return candidate


def _set_destination(solution: Solution, truck: TruckId, destination: DestinationId) -> None:
    if truck in solution.compound_assignment:
        _, door = solution.compound_assignment[truck]
        solution.compound_assignment[truck] = (destination, door)
    else:
        _, door = solution.outbound_assignment[truck]
        solution.outbound_assignment[truck] = (destination, door)


def _critical_outbound_relocate(
    instance: CrossDockInstance,
    solution: Solution,
    rng: random.Random,
    fast: FastEvaluator,
) -> Solution:
    """Move an outbound truck away from the critical door to its best slot."""

    if not instance.outbound_trucks:
        return solution.copy()

    result = fast.evaluate(solution)
    sequence = solution.door_sequences.get(result.critical_door, [])
    truck = sequence[-1] if sequence else rng.choice(instance.outbound_trucks)

    best = solution.copy()
    best_objective = result.objective
    for candidate in _relocations(instance, solution, truck):
        candidate_objective = fast.evaluate(candidate).objective
        if candidate_objective < best_objective - EPSILON:
            best = candidate
            best_objective = candidate_objective
    return best


def _critical_destination_swap(
    instance: CrossDockInstance,
    solution: Solution,
    rng: random.Random,
    fast: FastEvaluator,
) -> Solution:
    """Swap the critical truck's destination with the best other carrier."""

    result = fast.evaluate(solution)
    critical = result.critical_truck

    best = solution.copy()
    best_objective = result.objective
    for other in instance.all_trucks:
        if other == critical:
            continue
        candidate = _destinations_swapped(solution, critical, other)
        candidate_objective = fast.evaluate(candidate).objective
        if candidate_objective < best_objective - EPSILON:
            best = candidate
            best_objective = candidate_objective
    return best


def _compound_door_moves(
    instance: CrossDockInstance,
    solution: Solution,
    compound: TruckId,
) -> Iterator[Solution]:
    """Door swaps with other compounds plus moves to free doors, for one compound."""

    for other in instance.compound_trucks:
        if other != compound:
            yield _compound_doors_swapped(solution, compound, other)

    occupied = {door for _, door in solution.compound_assignment.values()}
    destination, _ = solution.compound_assignment[compound]
    for door in instance.doors:
        if door in occupied:
            continue
        candidate = solution.copy()
        candidate.compound_assignment[compound] = (destination, door)
        yield candidate


def _tardy_truck_relocate(
    instance: CrossDockInstance,
    solution: Solution,
    rng: random.Random,
    fast: FastEvaluator,
) -> Solution:
    """Best relocation of the most tardy truck (time-window guided operator).

    Falls back to the critical-door relocate when no truck is late, so the
    operator stays meaningful on instances without time windows.
    """

    result = fast.evaluate(solution)
    truck = result.most_tardy_truck
    if truck is None:
        return _critical_outbound_relocate(instance, solution, rng, fast)

    if truck in solution.compound_assignment:
        moves = _compound_door_moves(instance, solution, truck)
    else:
        moves = _relocations(instance, solution, truck)

    best = solution.copy()
    best_objective = result.objective
    for candidate in moves:
        candidate_objective = fast.evaluate(candidate).objective
        if candidate_objective < best_objective - EPSILON:
            best = candidate
            best_objective = candidate_objective
    return best


def _tardy_destination_swap(
    instance: CrossDockInstance,
    solution: Solution,
    rng: random.Random,
    fast: FastEvaluator,
) -> Solution:
    """Best destination swap anchored on the most tardy truck.

    Falls back to the critical-truck destination swap when no truck is late.
    """

    result = fast.evaluate(solution)
    anchor = result.most_tardy_truck
    if anchor is None:
        return _critical_destination_swap(instance, solution, rng, fast)

    best = solution.copy()
    best_objective = result.objective
    for other in instance.all_trucks:
        if other == anchor:
            continue
        candidate = _destinations_swapped(solution, anchor, other)
        candidate_objective = fast.evaluate(candidate).objective
        if candidate_objective < best_objective - EPSILON:
            best = candidate
            best_objective = candidate_objective
    return best
