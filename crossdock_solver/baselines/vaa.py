from __future__ import annotations

from dataclasses import dataclass
import time

from crossdock_solver.baselines.random_baseline import BaselineRun
from crossdock_solver.core.evaluator import evaluate_solution
from crossdock_solver.core.feasibility import check_feasible
from crossdock_solver.core.solution import Solution
from crossdock_solver.data.instance import CrossDockInstance, DestinationId, DoorId, TruckId


@dataclass(frozen=True)
class RegretChoice:
    regret: float
    best_cost: float
    compound: TruckId
    destination: DestinationId


def vaa_solution(instance: CrossDockInstance) -> Solution:
    """Build the paper-style VAA constructive heuristic.

    The paper uses VAA only for compound-truck destination assignment
    with Eq. (23), then completes the heuristic with outbound-destination
    assignment, central-door assignment, FAT construction, and FT_m based
    outbound insertion.
    """

    # 1~2단계: 화물 처리비용으로 regret을 계산하여 각 컴파운드 트럭이
    # 유지할 목적지를 결정한다. 이 단계는 release/due time을 사용하지 않는다.
    compound_to_destination = _assign_compound_destinations_by_regret(instance)

    # 3단계: 컴파운드 트럭이 맡지 않은 목적지를 일반 outbound 트럭에 배정한다.
    outbound_destination_by_truck = _assign_outbound_destinations_by_paper_priority(
        instance,
        compound_to_destination,
    )

    # 이후 우선순위와 스케줄 계산에서 양방향으로 조회할 수 있도록
    # 목적지→carrier 및 트럭→목적지 매핑을 만든다.
    destination_to_truck = _destination_to_truck(
        compound_to_destination,
        outbound_destination_by_truck,
    )
    truck_to_destination = {
        **compound_to_destination,
        **outbound_destination_by_truck,
    }
    # 4단계: 각 목적지의 예상 준비·적재시간 T_d를 계산한다.
    # 이 우선순위 역시 실제 도착시각이 아닌 순수 처리시간만 사용한다.
    destination_priority = {
        destination: _destination_completion_priority(
            instance,
            destination,
            destination_to_truck,
            truck_to_destination,
        )
        for destination in instance.destinations
    }
    solution = Solution(
        compound_assignment={},
        outbound_assignment={},
        door_sequences={door: [] for door in instance.doors},
    )

    # 5~7단계: FAT를 구성하고 중심 도어에 배치한 뒤, 여기서부터 실제
    # release_time을 포함하여 각 도어의 첫 완료시각을 계산한다.
    door_finish = _assign_first_trucks_to_doors(
        instance,
        solution,
        compound_to_destination,
        outbound_destination_by_truck,
        destination_priority,
    )
    # 8단계: 남은 outbound를 T_d 내림차순으로 처리하면서 현재 FT_m이
    # 가장 작은 도어에 삽입하고 완료시각을 갱신한다.
    _assign_remaining_outbounds_by_ftm(
        instance,
        solution,
        outbound_destination_by_truck,
        destination_priority,
        door_finish,
    )
    # 모든 트럭·목적지가 정확히 한 번 배정되었는지 마지막으로 검증한다.
    check_feasible(instance, solution)
    return solution


def run_vaa(instance: CrossDockInstance) -> BaselineRun:
    start = time.perf_counter()
    solution = vaa_solution(instance)
    result = evaluate_solution(instance, solution)
    return BaselineRun(
        name="VAA",
        solution=solution,
        result=result,
        runtime_sec=time.perf_counter() - start,
        samples=1,
    )


def vva_solution(instance: CrossDockInstance) -> Solution:
    """Alias for the common VVA typo in experiment scripts."""

    return vaa_solution(instance)


def _assign_compound_destinations_by_regret(
    instance: CrossDockInstance,
) -> dict[TruckId, DestinationId]:
    unassigned_compounds = set(instance.compound_trucks)
    available_destinations = set(instance.destinations)
    assignment: dict[TruckId, DestinationId] = {}
    # 1단계: 모든 (컴파운드, 목적지) 조합의 Eq. (23) 비용을 캐시한다.
    # 비용은 하역량과 다른 트럭에서 받아야 할 적재량만 포함한다.
    cost_cache = {
        (compound, destination): _compound_destination_cost(instance, compound, destination)
        for compound in instance.compound_trucks
        for destination in instance.destinations
    }

    while unassigned_compounds:
        choices: list[RegretChoice] = []

        # 2-A: 트럭 관점 regret.
        # 한 트럭의 최선 목적지를 놓쳤을 때 증가하는 비용을 측정한다.
        for compound in sorted(unassigned_compounds):
            costs = sorted(
                (
                    (cost_cache[(compound, destination)], destination)
                    for destination in available_destinations
                ),
                key=lambda item: item[0],
            )
            best_cost, best_destination = costs[0]
            second_cost = costs[1][0] if len(costs) > 1 else best_cost
            choices.append(
                RegretChoice(
                    regret=second_cost - best_cost,
                    best_cost=best_cost,
                    compound=compound,
                    destination=best_destination,
                )
            )

        # 2-B: 목적지 관점 regret.
        # 한 목적지의 최선 carrier를 놓쳤을 때 증가하는 비용도 함께 측정한다.
        for destination in sorted(available_destinations):
            costs = sorted(
                (
                    (cost_cache[(compound, destination)], compound)
                    for compound in unassigned_compounds
                ),
                key=lambda item: item[0],
            )
            best_cost, best_compound = costs[0]
            second_cost = costs[1][0] if len(costs) > 1 else best_cost
            choices.append(
                RegretChoice(
                    regret=second_cost - best_cost,
                    best_cost=best_cost,
                    compound=best_compound,
                    destination=destination,
                )
            )

        # 가장 후회값이 큰 쌍을 먼저 고정한다. 동률이면 더 작은 best cost를
        # 우선하고, 마지막에는 ID를 사용해 결과를 결정적으로 만든다.
        choice = max(
            choices,
            key=lambda item: (item.regret, -item.best_cost, item.compound, item.destination),
        )
        assignment[choice.compound] = choice.destination
        unassigned_compounds.remove(choice.compound)
        available_destinations.remove(choice.destination)

    return assignment


def _compound_destination_cost(
    instance: CrossDockInstance,
    compound: TruckId,
    destination: DestinationId,
) -> float:
    # 트럭이 싣고 온 전체 화물의 처리시간.
    total_compound_handling = sum(
        instance.handling_time(compound, d)
        for d in instance.destinations
    )
    # 담당 목적지 화물은 트럭에 남기므로 하역하지 않는다.
    retained_handling = instance.handling_time(compound, destination)
    unload_time = total_compound_handling - retained_handling

    # 목적지 전체 화물 중 이 트럭이 이미 보유한 양을 제외한 값이
    # 다른 컴파운드 트럭에서 받아 다시 적재해야 할 처리시간이다.
    compound_loading_time = _destination_load(instance, destination) - retained_handling

    # Paper Eq. (23): 부분 하역시간 + 타 트럭에서 받을 화물의 적재시간.
    # 주의: release_time, enter_time, 도어 이동시간, due_time은 포함하지 않는다.
    return unload_time + compound_loading_time


def _assign_outbound_destinations_by_paper_priority(
    instance: CrossDockInstance,
    compound_to_destination: dict[TruckId, DestinationId],
) -> dict[TruckId, DestinationId]:
    """Paper Steps 3-5: assign remaining destinations to outbound trucks."""

    # 컴파운드 carrier가 이미 맡은 목적지를 제외한다.
    assigned_destinations = set(compound_to_destination.values())
    unassigned_destinations = {
        destination for destination in instance.destinations if destination not in assigned_destinations
    }
    outbound_destination_by_truck: dict[TruckId, DestinationId] = {}

    # 진입+이탈 소요시간이 짧은 outbound부터 목적지를 배정한다.
    # 실제 도착시각 release_time은 이 정렬에 포함되지 않는다.
    for outbound in sorted(
        instance.outbound_trucks,
        key=lambda truck: (instance.enter_time[truck] + instance.leave_time[truck], truck),
    ):
        # 아직 남은 목적지 중 예상 처리 우선순위가 가장 큰 목적지를 선택한다.
        destination = max(
            unassigned_destinations,
            key=lambda d: (_outbound_destination_priority(instance, d, compound_to_destination), d),
        )
        outbound_destination_by_truck[outbound] = destination
        unassigned_destinations.remove(destination)

    return outbound_destination_by_truck


def _assign_first_trucks_to_doors(
    instance: CrossDockInstance,
    solution: Solution,
    compound_to_destination: dict[TruckId, DestinationId],
    outbound_destination_by_truck: dict[TruckId, DestinationId],
    destination_priority: dict[DestinationId, float],
) -> dict[DoorId, float]:
    """Paper Steps 6-10: construct FAT and assign it to central doors."""

    # 6단계: 다른 모든 도어까지의 이동시간 합이 작은 도어를
    # 더 중심적인 도어로 정의한다.
    central_doors = sorted(
        instance.doors,
        key=lambda door: (
            sum(instance.travel(door, other) for other in instance.doors),
            door,
        ),
    )

    # 5단계: 각 도어에서 처음 처리할 FAT 트럭을 만든다.
    first_trucks = _first_assigned_trucks(
        instance,
        compound_to_destination,
        outbound_destination_by_truck,
        destination_priority,
    )

    # 높은 우선순위의 FAT 트럭과 중심성이 높은 도어를 순서대로 짝짓는다.
    for truck, door in zip(first_trucks, central_doors):
        if truck in instance.compound_index:
            solution.compound_assignment[truck] = (compound_to_destination[truck], door)
        else:
            solution.outbound_assignment[truck] = (outbound_destination_by_truck[truck], door)
            solution.door_sequences[door].append(truck)

    # All compound trucks are included in FAT by construction. The defensive
    # fallback keeps the solution feasible if an unusual instance has more
    # compound trucks than FAT slots.
    used_compound_doors = {door for _, door in solution.compound_assignment.values()}
    for compound in instance.compound_trucks:
        if compound in solution.compound_assignment:
            continue
        available_doors = [door for door in central_doors if door not in used_compound_doors]
        door = available_doors[0]
        solution.compound_assignment[compound] = (compound_to_destination[compound], door)
        used_compound_doors.add(door)

    # 7단계: 배정이 끝난 뒤 실제 release_time과 이송시간을 사용하여
    # 각 도어의 첫 작업 완료시각을 계산한다.
    return _door_finish_after_first_trucks(instance, solution)


def _assign_remaining_outbounds_by_ftm(
    instance: CrossDockInstance,
    solution: Solution,
    outbound_destination_by_truck: dict[TruckId, DestinationId],
    destination_priority: dict[DestinationId, float],
    door_finish: dict[DoorId, float],
) -> None:
    """Paper Step 11: insert remaining outbound trucks by highest T_d and lowest FT_m."""

    # FAT에 들어가지 않은 일반 outbound 트럭만 후속 삽입 대상으로 삼는다.
    remaining = [
        truck for truck in instance.outbound_trucks if truck not in solution.outbound_assignment
    ]
    # 예상 완료 우선순위 T_d가 큰 목적지의 outbound부터 배치한다.
    remaining.sort(
        key=lambda truck: (
            destination_priority[outbound_destination_by_truck[truck]],
            outbound_destination_by_truck[truck],
            truck,
        ),
        reverse=True,
    )

    for truck in remaining:
        destination = outbound_destination_by_truck[truck]
        # 현재 가장 빨리 비는 도어(최소 FT_m)를 선택한다.
        door = min(instance.doors, key=lambda m: (door_finish[m], m))
        solution.outbound_assignment[truck] = (destination, door)
        solution.door_sequences[door].append(truck)
        # 앞 트럭 완료, 화물 준비, outbound 도착 중 가장 늦은 시각부터
        # 작업을 시작하도록 해당 도어의 FT_m을 갱신한다.
        door_finish[door] = _outbound_finish_on_door(
            instance,
            solution,
            truck,
            door,
            previous_finish=door_finish[door],
        )


def _first_assigned_trucks(
    instance: CrossDockInstance,
    compound_to_destination: dict[TruckId, DestinationId],
    outbound_destination_by_truck: dict[TruckId, DestinationId],
    destination_priority: dict[DestinationId, float],
) -> list[TruckId]:
    """Paper Step 7: FAT contains all compounds plus longest outbound jobs if needed."""

    # 5단계: outbound 화물을 공급하는 모든 컴파운드 트럭을 FAT에 포함한다.
    first_trucks: list[TruckId] = list(instance.compound_trucks)
    remaining_slots = max(0, min(len(instance.doors), len(instance.all_trucks)) - len(first_trucks))
    # 도어가 남으면 담당 목적지의 총 적재량이 큰 outbound를 추가한다.
    if remaining_slots > 0:
        outbound_by_loading = sorted(
            instance.outbound_trucks,
            key=lambda truck: (
                _destination_load(instance, outbound_destination_by_truck[truck]),
                outbound_destination_by_truck[truck],
                truck,
            ),
            reverse=True,
        )
        first_trucks.extend(outbound_by_loading[:remaining_slots])

    destination_by_truck = {
        **compound_to_destination,
        **outbound_destination_by_truck,
    }
    # 담당 목적지의 T_d가 큰 트럭이 더 중심적인 도어를 받도록 정렬한다.
    # 이 순서는 release_time을 직접 고려하지 않는다.
    return sorted(
        first_trucks,
        key=lambda truck: (
            destination_priority[destination_by_truck[truck]],
            destination_by_truck[truck],
            truck,
        ),
        reverse=True,
    )


def _door_finish_after_first_trucks(
    instance: CrossDockInstance,
    solution: Solution,
) -> dict[DoorId, float]:
    # 7단계: FAT 배치 직후 도어별 FT_m을 실제 시간축에서 계산한다.
    door_finish = {door: 0.0 for door in instance.doors}

    for compound, (_, door) in solution.compound_assignment.items():
        door_finish[door] = _compound_finish(instance, solution, compound)

    for outbound, (_, door) in solution.outbound_assignment.items():
        door_finish[door] = _outbound_finish_on_door(
            instance,
            solution,
            outbound,
            door,
            previous_finish=door_finish[door],
        )

    return door_finish


def _destination_to_truck(
    compound_to_destination: dict[TruckId, DestinationId],
    outbound_destination_by_truck: dict[TruckId, DestinationId],
) -> dict[DestinationId, TruckId]:
    carriers: dict[DestinationId, TruckId] = {}
    for truck, destination in compound_to_destination.items():
        carriers[destination] = truck
    for truck, destination in outbound_destination_by_truck.items():
        carriers[destination] = truck
    return carriers


def _destination_completion_priority(
    instance: CrossDockInstance,
    destination: DestinationId,
    destination_to_truck: dict[DestinationId, TruckId],
    truck_to_destination: dict[TruckId, DestinationId],
) -> float:
    carrier = destination_to_truck[destination]
    if carrier in instance.compound_index:
        # 목적지 화물을 가진 source들의 하역 소요시간 중 최댓값을
        # 목적지의 근사 준비시간으로 사용한다. 실제 도착시각은 제외된다.
        ready = max(
            [_compound_unload_time(instance, carrier, truck_to_destination[carrier])]
            + [
                _compound_unload_time(instance, source, truck_to_destination[source])
                for source in instance.compound_trucks
                if source != carrier and instance.unit_amount(source, destination) > 0
            ],
            default=0.0,
        )
        # carrier 자신이 이미 보유한 화물을 제외한 추가 적재시간.
        load = sum(
            instance.handling_time(source, destination)
            for source in instance.compound_trucks
            if source != carrier
        )
        return ready + load

    compound_to_destination = {
        truck: truck_to_destination[truck]
        for truck in instance.compound_trucks
    }
    return _outbound_destination_priority(instance, destination, compound_to_destination)


def _outbound_destination_priority(
    instance: CrossDockInstance,
    destination: DestinationId,
    compound_to_destination: dict[TruckId, DestinationId],
) -> float:
    # 일반 outbound carrier는 모든 목적지 화물을 새로 적재해야 한다.
    # 가장 늦게 하역되는 source의 소요시간을 근사 준비시간으로 둔다.
    ready = max(
        (
            _compound_unload_time(instance, source, compound_to_destination[source])
            for source in instance.compound_trucks
            if instance.unit_amount(source, destination) > 0
        ),
        default=0.0,
    )
    return ready + _destination_load(instance, destination)


def _compound_finish(
    instance: CrossDockInstance,
    solution: Solution,
    compound: TruckId,
) -> float:
    destination, target_door = solution.compound_assignment[compound]
    # 7단계부터는 실제 도착시각을 반영한다.
    # 자체 하역 완료 = 도착 + 진입 + 유지 목적지를 제외한 하역시간.
    own_unload_finish = (
        instance.release_time[compound]
        + instance.enter_time[compound]
        + _compound_unload_time(instance, compound, destination)
    )
    # 다른 컴파운드 트럭에서 오는 화물이 이 도어에 모두 도착하는 시각.
    destination_ready = _destination_ready_at_door(instance, solution, destination, target_door, carrier=compound)
    load_time = sum(
        instance.handling_time(source, destination)
        for source in instance.compound_trucks
        if source != compound
    )
    # 자체 하역과 외부 화물 도착이 모두 끝난 후 추가 적재와 이탈을 수행한다.
    return max(own_unload_finish, destination_ready) + load_time + instance.leave_time[compound]


def _outbound_finish_on_door(
    instance: CrossDockInstance,
    solution: Solution,
    outbound: TruckId,
    door: DoorId,
    *,
    previous_finish: float,
) -> float:
    destination = solution.outbound_assignment[outbound][0]
    destination_ready = _destination_ready_at_door(instance, solution, destination, door, carrier=outbound)
    load_time = _destination_load(instance, destination)
    # 도어, 화물, 트럭이 모두 준비되어야 outbound 작업을 시작할 수 있다.
    start = max(previous_finish, destination_ready, instance.release_time[outbound])
    return start + instance.enter_time[outbound] + load_time + instance.leave_time[outbound]


def _destination_ready_at_door(
    instance: CrossDockInstance,
    solution: Solution,
    destination: DestinationId,
    target_door: DoorId,
    *,
    carrier: TruckId,
) -> float:
    ready = 0.0
    for source in instance.compound_trucks:
        if source == carrier:
            continue
        if instance.unit_amount(source, destination) <= 0:
            continue
        source_door = solution.compound_assignment[source][1]
        retained_destination = solution.compound_assignment[source][0]
        # 각 source의 실제 도착시각부터 진입과 부분 하역을 수행한다.
        source_unload_finish = (
            instance.release_time[source]
            + instance.enter_time[source]
            + _compound_unload_time(instance, source, retained_destination)
        )
        # source 하역 완료 후 target 도어까지의 내부 이송시간을 더한다.
        ready = max(ready, source_unload_finish + instance.travel(source_door, target_door))
    return ready


def _destination_load(instance: CrossDockInstance, destination: DestinationId) -> float:
    return sum(
        instance.handling_time(compound, destination)
        for compound in instance.compound_trucks
    )


def _compound_unload_time(
    instance: CrossDockInstance,
    compound: TruckId,
    retained_destination: DestinationId,
) -> float:
    return sum(
        instance.handling_time(compound, destination)
        for destination in instance.destinations
        if destination != retained_destination
    )
