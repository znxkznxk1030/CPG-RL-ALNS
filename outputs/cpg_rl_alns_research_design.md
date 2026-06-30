# CPG-RL-ALNS 연구 설계 및 MVP 구현 계획

대상 문제는 compound truck과 outbound truck이 공존하는 multi-door cross-docking truck scheduling이며, compound truck의 partial unloading을 명시적으로 고려한다. 목표는 destination assignment, dock-door assignment, outbound sequencing을 동시에 결정하여 makespan을 최소화하는 것이다.

제안 방법의 핵심은 기존 RL-SA처럼 작은 neighborhood를 무작위 또는 보상 기반으로 고르는 데서 멈추지 않고, 현재 schedule의 makespan을 실제로 만드는 critical bottleneck을 먼저 식별한 뒤 그 주변 region을 크게 destroy하고 강하게 repair하는 것이다.

짧은 이름은 다음과 같이 둔다.

```text
CPG-RL-ALNS
Critical-Path Guided Reinforcement Learning Adaptive Large Neighborhood Search
```

---

## 1. 알고리즘 설계

### 1.1 핵심 가설

기존 SA 기반 탐색의 한계는 solution 전체에 작은 move를 반복한다는 점이다. 이 문제에서는 compound truck의 destination assignment 하나가 다음 요소를 동시에 바꾼다.

- partial unloading volume
- destination별 load demand
- source door에서 carrier door까지의 transfer delay
- outbound truck의 loading start time
- door congestion
- 최종 makespan을 만드는 critical path

따라서 좋은 move는 단순히 임의 truck이나 door를 바꾸는 것이 아니라, 현재 makespan을 유발하는 door, truck, transfer edge, sequence segment를 찾아 그 region을 집중적으로 재최적화해야 한다.

### 1.2 Schedule 해석

구현에서 먼저 고정해야 할 schedule semantics는 다음과 같다.

1. Compound truck `i`는 하나의 destination `d_i`와 door `m_i`를 갖는다.
2. `i`가 `d_i`를 담당하면 `f[i][d_i][k]`는 unload하지 않고 truck 안에 유지된다.
3. `i`는 다른 destination 물량만 unload한다.
4. `d_i`가 다른 compound truck들로부터 받을 물량은 `i`가 carrier로서 load해야 한다.
5. Outbound truck `f`가 destination `d`를 담당하면 모든 compound truck에서 나온 `d` 물량을 load한다.
6. 같은 door의 outbound trucks는 `door_sequences[m]` 순서로 처리된다.
7. MVP evaluator에서는 product가 destination carrier door에 모두 도착한 뒤 loading을 시작한다고 가정한다.
8. full evaluator에서는 product type 또는 source-destination transfer 단위 node를 precedence graph에 넣어 더 세밀하게 계산한다.

MVP의 보수적 가정은 loading-start time을 늦게 잡을 수 있지만 feasibility와 구현 안정성을 보장한다. 논문 실험용 full evaluator는 이 가정을 완화할 수 있다.

### 1.3 전체 흐름

```python
def cpg_rl_alns(instance, config):
    pool = generate_initial_solutions(instance, config)
    current = min(pool, key=lambda s: evaluate_solution(instance, s).makespan)
    best = current.copy()

    controller = ContextualBanditController(actions=config.actions)
    temperature = initial_temperature(instance, current)

    while not stopping_condition():
        current_result = evaluate_solution(instance, current)

        context = extract_context(
            instance=instance,
            solution=current,
            result=current_result,
            best=best,
            temperature=temperature,
        )

        action = controller.select_action(context)

        destroyed = destroy(
            instance=instance,
            solution=current,
            result=current_result,
            action=action,
        )

        candidate = repair(
            instance=instance,
            partial_solution=destroyed,
            result=current_result,
            action=action,
        )

        candidate = local_search_polish(instance, candidate, budget=config.polish_budget)
        candidate_result = evaluate_solution(instance, candidate)

        accepted = accept_by_sa(
            old=current_result.makespan,
            new=candidate_result.makespan,
            temperature=temperature,
            rng=config.rng,
        )

        if accepted:
            current = candidate

        old_best = evaluate_solution(instance, best).makespan
        if candidate_result.makespan < old_best:
            best = candidate.copy()

        reward = compute_reward(
            old_result=current_result,
            new_result=candidate_result,
            old_best=old_best,
            accepted=accepted,
            runtime=action.runtime,
        )
        controller.update(context, action, reward)

        temperature = update_temperature(temperature, config)

    return best
```

---

## 2. 기존 Shahmardan & Sajadieh 방식과의 차이

아래 비교는 사용자가 요약한 기준 논문의 VAA + SA + RL/MAB neighborhood selection 구조를 기준으로 한다.

| 구분 | 기존 RL-SA 계열 | 제안 CPG-RL-ALNS |
|---|---|---|
| 초기해 | VAA 중심 단일 또는 제한적 초기해 | VAA, randomized VAA, matching, door-center greedy, random feasible multi-start |
| 탐색 단위 | 작은 neighborhood move | critical region 단위 large-neighborhood destroy/repair |
| 병목 인식 | objective 값 변화에 간접 반영 | precedence graph와 critical path로 makespan 원인 직접 추적 |
| RL action | neighborhood type 선택 | `(destroy operator, repair operator, destroy size)` 선택 |
| RL context | operator 성과 위주 | current gap, critical door utilization, transfer bottleneck, waiting, recent success 포함 |
| assignment 변화 | swap/insertion 중심 | bottleneck region 내 assignment, door, sequence를 동시에 재구성 |
| repair 방식 | heuristic neighborhood move | greedy, regret-k, matching, CP-SAT/MILP exact repair |
| exact optimization | 없음 또는 전체 문제에는 비현실적 | destroyed subproblem만 exact repair |
| partial unloading 처리 | evaluator 내부 반영 | destroy/repair cost와 critical transfer에도 명시 반영 |
| exploration | SA acceptance와 random move | SA acceptance + random destroy + contextual bandit uncertainty |
| 논문 contribution | RL로 neighborhood 선택 개선 | critical-path diagnosis + exact local repair + contextual RL operator selection |

가장 중요한 차이는 학습 대상이다. 기존 방식은 "어떤 neighborhood를 쓸 것인가"를 학습한다. 제안 방식은 "어떤 bottleneck region을 깨고 어떤 방식으로 복구할 것인가"를 학습한다.

---

## 3. Critical Path와 Critical Region

### 3.1 Precedence graph

Schedule을 DAG로 표현한다.

```text
G = (V, E)
```

대표 node는 다음과 같다.

```text
CT_i_enter
CT_i_unload_done
CT_i_load_start
CT_i_leave

OT_f_enter
OT_f_load_start
OT_f_leave

Transfer_i_to_c_d
Door_m_available_after_x
```

대표 edge는 다음 제약을 나타낸다.

| Edge | 의미 |
|---|---|
| `CT_i_enter -> CT_i_unload_done` | compound truck setup + unload processing |
| `CT_i_unload_done -> Carrier_c_load_start` | unloaded product availability + internal transfer |
| `Carrier_c_load_start -> Carrier_c_leave` | loading + departure |
| `Truck_a_leave -> Truck_b_enter` | 같은 door sequence |
| `Door_m_previous_finish -> Truck_enter` | door occupation |

각 node의 earliest time은 predecessor edge의 longest path로 계산된다.

```text
T(v) = max_{(u,v) in E} T(u) + w(u,v)
```

Critical path는 makespan node까지 도달하는 longest path이다.

```text
critical_path = argmax path length from source to sink
```

### 3.2 Critical region

Critical path에서 다음 정보를 추출한다.

```text
critical_door
critical_trucks
critical_destinations
critical_transfer_edges
critical_sequence_segment
```

이 정보를 ALNS destroy set으로 바꾼다.

| Region | Destroy 대상 |
|---|---|
| Critical door region | critical door의 trucks + 해당 door와 큰 flow를 주고받는 trucks |
| Critical truck region | latest truck + 같은 destination 또는 transfer 상호작용이 큰 trucks |
| High transfer edge region | `volume * travel_time` 상위 edge의 source/carrier trucks |
| Sequence bottleneck region | 같은 door에서 waiting time이 큰 연속 segment |
| Destination cluster region | loading demand가 큰 destination과 담당 carrier |
| Random region | 탐색 다양성 확보용 random trucks/destinations |

---

## 4. Destroy and Repair Operators

### 4.1 Destroy operators

| 이름 | 목적 | 기본 destroy set |
|---|---|---|
| D1 Critical Door Destroy | door congestion 완화 | `critical_door`에 배정된 trucks, high-flow adjacent trucks |
| D2 Critical Truck Destroy | latest truck 원인 완화 | critical path 내 delay contribution이 큰 truck과 관련 destination |
| D3 High Transfer Destroy | internal transfer time 감소 | `volume * travel` 상위 transfer edge의 양끝 truck |
| D4 Sequence Bottleneck Destroy | door sequence waiting 감소 | waiting time이 큰 outbound sequence segment |
| D5 Destination Regret Destroy | 비효율적 destination assignment 교정 | second-best 대비 regret이 큰 destination/truck |
| D6 Random Destroy | local optimum 탈출 | random truck/destination subset |

Destroy 결과는 다음 구조를 반환한다.

```python
@dataclass
class DestroyResult:
    partial_solution: Solution
    removed_compounds: set[str]
    removed_outbounds: set[str]
    removed_destinations: set[str]
    candidate_doors: set[int]
    metadata: dict
```

### 4.2 Repair operators

| 이름 | 역할 | 권장 사용 시점 |
|---|---|---|
| R1 Greedy Repair | 가장 작은 delta makespan insertion 반복 | large instances, 빠른 탐색 |
| R2 Regret-k Repair | 나중에 넣으면 손해가 큰 item 우선 삽입 | assignment coupling이 강할 때 |
| R3 Matching Repair | compound-destination 재배정 min-cost matching | destination assignment quality 개선 |
| R4 CP-SAT/MILP Exact Repair | small destroyed region을 부분 최적화 | critical region이 작고 중요할 때 |
| R5 Local Search Polish | repair 후 짧은 swap/relocation | 모든 repair 뒤 공통 polish |

Greedy insertion cost는 full reevaluation으로 시작한다. 이후 성능 최적화 단계에서 incremental evaluator로 바꾼다.

```python
delta = evaluate_solution(instance, solution_with_insert).makespan \
      - evaluate_solution(instance, current_solution).makespan
```

---

## 5. Mathematical Formulation

### 5.1 Sets and parameters

```text
I: compound trucks
F: outbound trucks
D: destinations
M: dock doors
K: product types
C = I union F: possible destination carriers

q_{i,d,k}: product amount initially loaded on compound i for destination d and type k
p_k: handling time per unit of product type k
tau_{m,n}: internal travel time from door m to door n
DE_c: entering/setup time of truck c
DL_c: leaving time of truck c
```

Cardinality 주의: "각 outbound truck은 정확히 하나의 destination"과 "각 destination은 정확히 하나의 truck"을 동시에 강제하려면 일반적으로 `|I| + |F| = |D|`가 필요하다. 데이터가 그렇지 않다면 dummy destination 또는 dummy outbound truck을 추가해야 한다. 구현에서는 이 검사를 명시적으로 둔다.

### 5.2 Assignment variables

```text
x_{i,d} = 1 if compound truck i carries destination d
z_{f,d} = 1 if outbound truck f carries destination d
a_{i,m} = 1 if compound truck i uses door m
b_{f,m} = 1 if outbound truck f uses door m
```

Carrier uniqueness:

```text
sum_{i in I} x_{i,d} + sum_{f in F} z_{f,d} = 1     for all d in D
```

Truck assignment:

```text
sum_{d in D} x_{i,d} = 1                             for all i in I
sum_{d in D} z_{f,d} = 1                             for all f in F
sum_{m in M} a_{i,m} = 1                             for all i in I
sum_{m in M} b_{f,m} = 1                             for all f in F
```

Compound first-stage door capacity:

```text
sum_{i in I} a_{i,m} <= 1                            for all m in M
```

### 5.3 Partial unloading and loading demand

If compound `i` carries destination `d`, its retained amount is not unloaded.

Unload processing time:

```text
U_i = sum_{d in D} sum_{k in K} q_{i,d,k} p_k (1 - x_{i,d})
```

If destination `d` is carried by compound `i`, loading demand onto `i` excludes its own retained load.

```text
L_i = sum_{d in D} x_{i,d} sum_{h in I, h != i} sum_{k in K} q_{h,d,k} p_k
```

If destination `d` is carried by outbound `f`, loading demand onto `f` includes all compound sources.

```text
L_f = sum_{d in D} z_{f,d} sum_{h in I} sum_{k in K} q_{h,d,k} p_k
```

### 5.4 Time variables

```text
S_i^U: unload start time of compound i
C_i^U: unload completion time of compound i
S_c^L: load start time of carrier c
C_c: final completion time of carrier c
Cmax: makespan
```

Compound unload completion:

```text
C_i^U >= S_i^U + DE_i + U_i
```

Product availability for carrier `c` at door `m_c`:

```text
S_c^L >= C_i^U + tau_{m_i,m_c}
```

for each source compound `i` that sends positive quantity to the destination carried by `c`, except the retained part when `c = i`.

Carrier completion:

```text
C_c >= S_c^L + L_c + DL_c
Cmax >= C_c
```

Door sequence constraints can be expressed with binary precedence variables.

```text
P_{u,v,m} = 1 if truck u precedes truck v on door m
```

For trucks assigned to the same door:

```text
S_v >= C_u - M_big * (1 - P_{u,v,m})
S_u >= C_v - M_big * P_{u,v,m}
```

The full formulation is a mixed-integer scheduling model. It is useful as a reference, but not as the main solver for large instances.

### 5.5 Objective

Primary objective:

```text
minimize Cmax
```

Research implementation may use a tie-breaking objective:

```text
minimize
    Cmax
  + lambda_1 * total_internal_transfer_time
  + lambda_2 * door_load_imbalance
  + lambda_3 * number_of_changed_assignments
```

---

## 6. Exact Repair Subproblem

### 6.1 Input

```python
@dataclass
class ExactRepairInput:
    instance: CrossDockInstance
    current_solution: Solution
    destroyed_compounds: set[str]
    destroyed_outbounds: set[str]
    destroyed_destinations: set[str]
    candidate_doors: set[int]
    fixed_assignments: dict
    time_limit_sec: float
```

### 6.2 Variables

```text
Y_{i,d,m} = 1 if destroyed compound i is assigned to destination d and door m
Z_{f,d,m} = 1 if destroyed outbound f is assigned to destination d and door m
P_{u,v,m} = 1 if u precedes v on door m
S_c, C_c, Cmax
```

Fixed trucks outside the destroyed region are constants. Only their induced availability, door occupation, and destination coverage constraints enter the subproblem.

### 6.3 Constraints

1. Each destroyed truck receives exactly one destination and one candidate door.
2. Each destroyed destination is covered exactly once by either a destroyed truck or a fixed feasible carrier allowed by the repair design.
3. Fixed assignments outside the region remain unchanged.
4. Door sequence is feasible for repaired outbound trucks and any fixed trucks on the same door.
5. Product availability includes fixed source compounds and destroyed source compounds.
6. Partial unloading is applied to destroyed compound assignments.
7. Completion times and local/global makespan are consistent.
8. Optional stability constraint limits excessive changes:

```text
number_of_changed_assignments <= Delta_max
```

### 6.4 CP-SAT vs MILP

Recommended default: CP-SAT.

Rationale:

- The repair subproblem is dominated by binary assignment, optional sequencing, and no-overlap style constraints.
- OR-Tools CP-SAT handles optional intervals and disjunctive scheduling more naturally than a big-M MILP.
- CP-SAT is robust for small destroyed regions with tight time limits.
- Integerized processing times are natural in truck scheduling benchmarks.

Use MILP when:

- processing times must remain continuous,
- LP relaxation bounds are important for analysis,
- the repair region is very small and a commercial MILP solver is available.

Practical recommendation:

```text
MVP: no exact repair
First research version: CP-SAT exact repair with integer-scaled times
Optional later version: MILP repair for comparison and lower-bound reporting
```

### 6.5 CP-SAT repair pseudocode

```python
def cp_sat_repair(input: ExactRepairInput) -> Solution:
    model = cp_model.CpModel()

    # 1. Create assignment variables for destroyed trucks.
    y = {
        (i, d, m): model.NewBoolVar(f"y_{i}_{d}_{m}")
        for i in input.destroyed_compounds
        for d in input.destroyed_destinations
        for m in input.candidate_doors
    }

    z = {
        (f, d, m): model.NewBoolVar(f"z_{f}_{d}_{m}")
        for f in input.destroyed_outbounds
        for d in input.destroyed_destinations
        for m in input.candidate_doors
    }

    # 2. Add one-assignment and one-cover constraints.
    add_assignment_constraints(model, y, z, input)
    add_fixed_coverage_constraints(model, y, z, input)

    # 3. Create start/completion variables and sequencing variables.
    starts, ends = add_time_variables(model, input)
    add_product_availability_constraints(model, starts, ends, y, z, input)
    add_door_sequence_constraints(model, starts, ends, y, z, input)

    # 4. Minimize global makespan plus regularization.
    cmax = model.NewIntVar(0, input.horizon, "cmax")
    for c in starts:
        model.Add(cmax >= ends[c])

    model.Minimize(
        cmax
        + input.lambda_transfer * total_transfer_expr(y, z, input)
        + input.lambda_changes * changed_assignment_expr(y, z, input)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = input.time_limit_sec
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return fallback_regret_repair(input)

    return decode_solution(input.current_solution, y, z, solver)
```

---

## 7. Contextual RL Operator Selection

### 7.1 Action

```python
@dataclass(frozen=True)
class ALNSAction:
    destroy_name: str
    repair_name: str
    destroy_size: str  # small, medium, large
```

Example action set:

```text
CriticalDoorDestroy + GreedyRepair + small
CriticalDoorDestroy + ExactRepair + medium
HighTransferDestroy + RegretRepair + medium
SequenceBottleneckDestroy + GreedyRepair + small
RandomDestroy + ExactRepair + large
```

### 7.2 Context features

```text
x_t = [
    current_makespan_norm,
    best_makespan_norm,
    relative_gap_to_best,
    iteration_ratio,
    temperature_norm,
    no_improvement_count_norm,
    critical_door_load_norm,
    critical_door_utilization,
    door_load_std_norm,
    total_transfer_time_norm,
    max_transfer_edge_time_norm,
    number_of_critical_trucks_norm,
    average_waiting_time_norm,
    recent_acceptance_rate,
    recent_best_update_rate,
    last_action_accepted
]
```

Use instance-level normalization, for example divide time features by current makespan or initial makespan.

### 7.3 LinUCB controller

LinUCB is the recommended first contextual bandit because it is stable, interpretable, and low overhead.

For each action `a`:

```text
A_a = I_d
b_a = 0_d
theta_a = A_a^{-1} b_a
score(a) = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)
```

Select:

```text
a_t = argmax_a score(a)
```

Update:

```text
A_a <- A_a + x x^T
b_a <- b_a + reward * x
```

### 7.4 Reward

Reward should separate accepted worse moves from true improvement.

```text
improvement_current = (old_makespan - new_makespan) / old_makespan
improvement_best = max(0, old_best_makespan - new_makespan) / old_best_makespan

reward =
    w1 * improvement_current
  + w2 * improvement_best
  + w3 * accepted_indicator
  - w4 * runtime_sec
  - w5 * infeasible_indicator
```

Suggested starting weights:

```text
w1 = 1.0
w2 = 3.0
w3 = 0.05
w4 = 0.01
w5 = 2.0
```

---

## 8. Python Architecture

### 8.1 Package layout

```text
crossdock_solver/
├── data/
│   ├── instance.py
│   ├── generator.py
│   └── loader.py
├── core/
│   ├── solution.py
│   ├── feasibility.py
│   ├── evaluator.py
│   ├── precedence_graph.py
│   └── critical_path.py
├── initial/
│   ├── random_init.py
│   ├── vaa.py
│   ├── randomized_vaa.py
│   ├── matching_init.py
│   └── greedy_init.py
├── alns/
│   ├── actions.py
│   ├── destroy.py
│   ├── repair.py
│   ├── local_search.py
│   ├── acceptance.py
│   └── loop.py
├── exact/
│   ├── cp_sat_repair.py
│   ├── milp_repair.py
│   └── subproblem_builder.py
├── rl/
│   ├── features.py
│   ├── reward.py
│   ├── contextual_bandit.py
│   └── linucb.py
├── baselines/
│   ├── vaa_only.py
│   ├── sa.py
│   ├── static_sa.py
│   ├── rl_sa.py
│   └── vanilla_alns.py
├── experiments/
│   ├── run_experiment.py
│   ├── ablation.py
│   ├── summarize.py
│   └── plot_results.py
└── main.py
```

### 8.2 Core dataclasses

```python
@dataclass(frozen=True)
class CrossDockInstance:
    compound_trucks: list[str]
    outbound_trucks: list[str]
    destinations: list[str]
    doors: list[int]
    product_types: list[str]
    flow: np.ndarray          # shape = [num_compound, num_dest, num_product]
    product_time: np.ndarray  # shape = [num_product]
    travel_time: np.ndarray   # shape = [num_doors, num_doors]
    enter_time: dict[str, float]
    leave_time: dict[str, float]
```

```python
@dataclass
class Solution:
    compound_assignment: dict[str, tuple[str, int]]
    outbound_assignment: dict[str, tuple[str, int]]
    door_sequences: dict[int, list[str]]

    def copy(self) -> "Solution":
        ...
```

```python
@dataclass
class ScheduleResult:
    makespan: float
    truck_start: dict[str, float]
    truck_finish: dict[str, float]
    door_finish: dict[int, float]
    door_load: dict[int, float]
    door_utilization: dict[int, float]
    total_transfer_time: float
    max_transfer_edge_time: float
    waiting_time: dict[str, float]
    critical_door: int
    critical_truck: str
    critical_path: list[str]
    precedence_graph: object | None
    metadata: dict
```

### 8.3 Feasibility checker

```python
def check_feasible(instance: CrossDockInstance, solution: Solution) -> None:
    assert set(solution.compound_assignment) == set(instance.compound_trucks)
    assert set(solution.outbound_assignment) == set(instance.outbound_trucks)

    covered = []
    for i, (d, _) in solution.compound_assignment.items():
        covered.append(d)
    for f, (d, _) in solution.outbound_assignment.items():
        covered.append(d)

    if sorted(covered) != sorted(instance.destinations):
        raise InfeasibleSolution("each destination must be covered exactly once")

    for i, (_, m) in solution.compound_assignment.items():
        if m not in instance.doors:
            raise InfeasibleSolution(f"invalid door for compound {i}")

    for f, (_, m) in solution.outbound_assignment.items():
        if m not in instance.doors:
            raise InfeasibleSolution(f"invalid door for outbound {f}")

    # Each outbound assigned to a door must appear exactly once in that door sequence.
    sequenced = [f for seq in solution.door_sequences.values() for f in seq]
    if sorted(sequenced) != sorted(instance.outbound_trucks):
        raise InfeasibleSolution("outbound trucks must appear exactly once in door sequences")

    # Compound first-stage door capacity.
    used = {}
    for i, (_, m) in solution.compound_assignment.items():
        used.setdefault(m, []).append(i)
    overloaded = {m: xs for m, xs in used.items() if len(xs) > 1}
    if overloaded:
        raise InfeasibleSolution(f"compound door capacity violated: {overloaded}")
```

### 8.4 MVP evaluator pseudocode

```python
def evaluate_solution(instance, solution, build_graph=False):
    check_feasible(instance, solution)

    # 1. Compound unload stage.
    ct_unload_finish = {}
    ct_finish = {}
    door_available = {m: 0.0 for m in instance.doors}
    transfer_edges = []

    for i in instance.compound_trucks:
        own_destination, door_i = solution.compound_assignment[i]
        unload_time = 0.0
        for d in instance.destinations:
            if d == own_destination:
                continue
            unload_time += handling_time(instance.flow[i, d, :], instance.product_time)

        start = door_available[door_i]
        ct_unload_finish[i] = start + instance.enter_time[i] + unload_time

    # 2. Carrier map for destinations.
    carrier = {}
    carrier_door = {}
    for i, (d, m) in solution.compound_assignment.items():
        carrier[d] = i
        carrier_door[d] = m
    for f, (d, m) in solution.outbound_assignment.items():
        carrier[d] = f
        carrier_door[d] = m

    # 3. Product availability at each carrier door.
    destination_ready = {}
    total_transfer_time = 0.0
    max_transfer_edge_time = 0.0

    for d in instance.destinations:
        c = carrier[d]
        target_door = carrier_door[d]
        ready = 0.0

        for i in instance.compound_trucks:
            source_door = solution.compound_assignment[i][1]
            if c == i:
                # retained load does not need unload or transfer
                continue

            volume_time = handling_time(instance.flow[i, d, :], instance.product_time)
            volume_amount = total_units(instance.flow[i, d, :])
            if volume_amount <= 0:
                continue

            travel = instance.travel_time[source_door, target_door]
            ready = max(ready, ct_unload_finish[i] + travel)

            edge_time = volume_amount * travel
            total_transfer_time += edge_time
            max_transfer_edge_time = max(max_transfer_edge_time, edge_time)
            transfer_edges.append((i, c, d, edge_time))

        destination_ready[d] = ready

    # 4. Compound carrier loading and leaving.
    # A compound occupies its assigned door until final leave in this MVP.
    for i in instance.compound_trucks:
        d, door_i = solution.compound_assignment[i]
        load_time = 0.0
        for h in instance.compound_trucks:
            if h == i:
                continue
            load_time += handling_time(instance.flow[h, d, :], instance.product_time)

        load_start = max(ct_unload_finish[i], destination_ready[d])
        finish = load_start + load_time + instance.leave_time[i]
        ct_finish[i] = finish
        door_available[door_i] = max(door_available[door_i], finish)

    # 5. Outbound trucks by door sequence.
    truck_start = {}
    truck_finish = dict(ct_finish)
    waiting_time = {}

    for m in instance.doors:
        prev_finish = door_available[m]
        for f in solution.door_sequences.get(m, []):
            d, door_f = solution.outbound_assignment[f]
            assert door_f == m
            load_time = 0.0
            for h in instance.compound_trucks:
                load_time += handling_time(instance.flow[h, d, :], instance.product_time)

            earliest = max(prev_finish, destination_ready[d])
            waiting_time[f] = max(0.0, earliest - destination_ready[d])
            truck_start[f] = earliest
            finish = earliest + instance.enter_time[f] + load_time + instance.leave_time[f]
            truck_finish[f] = finish
            prev_finish = finish

        door_available[m] = prev_finish

    makespan = max(truck_finish.values())
    door_load = compute_door_load(instance, solution)
    door_utilization = {m: door_load[m] / max(makespan, 1e-9) for m in instance.doors}
    critical_door = max(door_available, key=door_available.get)
    critical_truck = max(truck_finish, key=truck_finish.get)

    graph, critical_path = None, []
    if build_graph:
        graph = build_precedence_graph(...)
        critical_path = longest_path(graph)

    return ScheduleResult(
        makespan=makespan,
        truck_start=truck_start,
        truck_finish=truck_finish,
        door_finish=door_available,
        door_load=door_load,
        door_utilization=door_utilization,
        total_transfer_time=total_transfer_time,
        max_transfer_edge_time=max_transfer_edge_time,
        waiting_time=waiting_time,
        critical_door=critical_door,
        critical_truck=critical_truck,
        critical_path=critical_path,
        precedence_graph=graph,
        metadata={"transfer_edges": transfer_edges},
    )
```

---

## 9. MVP Version

처음 구현할 MVP는 full research algorithm이 아니라 correctness-first ALNS이다.

### 9.1 MVP에 반드시 포함

| 기능 | 이유 |
|---|---|
| `CrossDockInstance` | 데이터 구조 고정 |
| `Solution` | assignment와 sequence 표현 |
| `check_feasible` | repair 오류를 즉시 잡기 위함 |
| `evaluate_solution` | 모든 metaheuristic의 기반 |
| random feasible initialization | VAA 없이도 loop 검증 가능 |
| critical door detection | CPG의 최소 버전 |
| critical door destroy | bottleneck-guided destroy 시작점 |
| greedy repair | 빠른 feasible reconstruction |
| regret-k repair | greedy보다 강한 repair |
| SA acceptance | worse move 수용으로 local optimum 회피 |
| simple ALNS loop | 전체 integration 검증 |

### 9.2 MVP에서 제외하고 나중에 확장

| 제외 기능 | 확장 시점 |
|---|---|
| full precedence graph | evaluator 안정화 후 |
| critical path extraction | graph 검증 후 |
| high transfer / sequence / destination destroy | critical door destroy가 작동한 뒤 |
| CP-SAT exact repair | destroyed region API가 안정화된 뒤 |
| LinUCB contextual RL | action별 로그와 reward가 쌓인 뒤 |
| VAA/randomized VAA | random init baseline 후 |
| matching repair | assignment cost 정의 확정 후 |

### 9.3 MVP simple ALNS pseudocode

```python
def simple_alns(instance, config):
    current = random_feasible_solution(instance, config.rng)
    current_result = evaluate_solution(instance, current)
    best = current.copy()
    best_result = current_result

    temperature = config.initial_temperature

    for it in range(config.max_iter):
        result = evaluate_solution(instance, current)

        removed = critical_door_destroy(
            instance=instance,
            solution=current,
            result=result,
            size=config.destroy_size,
            rng=config.rng,
        )

        if config.repair == "greedy":
            candidate = greedy_repair(instance, removed, rng=config.rng)
        else:
            candidate = regret_k_repair(instance, removed, k=2, rng=config.rng)

        candidate_result = evaluate_solution(instance, candidate)

        if accept_by_sa(result.makespan, candidate_result.makespan, temperature, config.rng):
            current = candidate
            result = candidate_result

        if candidate_result.makespan < best_result.makespan:
            best = candidate.copy()
            best_result = candidate_result

        temperature *= config.cooling_rate

    return best, best_result
```

---

## 10. Baselines and Ablation Design

### 10.1 Baselines

| ID | Method | 목적 |
|---|---|---|
| B1 | VAA only | 논문 기반 초기해 품질 기준 |
| B2 | Random feasible + greedy polish | 매우 약한 constructive baseline |
| B3 | SA random neighborhood | 기존 SA형 탐색 기준 |
| B4 | SA static neighborhood probability | hand-tuned SA 기준 |
| B5 | RL-SA/MAB neighborhood selection | 기존 논문 방향과 직접 비교 |
| B6 | Vanilla ALNS without critical guidance | ALNS 효과만 분리 |
| B7 | Critical ALNS without exact repair | critical guidance 효과 측정 |
| B8 | Critical ALNS with exact repair, no RL | exact repair 효과 측정 |
| B9 | Full CPG-RL-ALNS | 제안 방법 |

### 10.2 Ablations

| ID | Variant | 제거 요소 | 확인하려는 주장 |
|---|---|---|---|
| A1 | no critical destroy | critical-path guided destroy | bottleneck targeting의 효과 |
| A2 | no exact repair | CP-SAT/MILP repair | exact local optimization의 효과 |
| A3 | no contextual RL | LinUCB operator controller | adaptive action selection의 효과 |
| A4 | no multi-start | diverse initialization | 초기 다양성의 효과 |
| A5 | random destroy only | all critical destroy operators | random ALNS 대비 우위 |
| A6 | greedy repair only | regret/matching/exact repair | strong repair의 효과 |
| A7 | exact repair only on small instances | exact repair scaling 제한 | instance size별 trade-off |
| A8 | no SA acceptance | worse move acceptance | escaping local optimum 효과 |

### 10.3 Metrics

```text
Best makespan
Average makespan
Standard deviation
Gap to optimal solution for small instances
Gap to best-known solution for large instances
Runtime
Time-to-best
Convergence curve
Number of accepted moves
Number of best updates
Operator selection frequency
Operator average reward
Critical door load reduction
Total internal transfer time
Door utilization standard deviation
```

### 10.4 Table formats

Main result table:

| Instance class | Method | Best | Avg | Std | Gap % | Runtime | Time-to-best |
|---|---:|---:|---:|---:|---:|---:|---:|

Ablation table:

| Instance class | Full | Variant | Delta makespan % | Delta runtime % | Interpretation |
|---|---:|---:|---:|---:|---|

Operator table:

| Operator pair | Frequency % | Acceptance % | Best-update count | Avg reward | Avg runtime |
|---|---:|---:|---:|---:|---:|

Critical bottleneck table:

| Method | Critical door finish | Door load std | Total transfer | Max transfer edge |
|---|---:|---:|---:|---:|

---

## 11. Instance Generation

Use four classes.

| Class | Compound | Outbound | Destinations | Doors |
|---|---:|---:|---:|---:|
| Small | 2-4 | 2-5 | 4-8 | 2-4 |
| Medium | 5-10 | 10-30 | 15-40 | 5-10 |
| Large | 20-50 | 50-200 | 50-300 | 10-50 |
| Very Large | 100+ | 500+ | 500+ | 100+ |

Control factors:

- flow density: sparse, medium, dense
- destination demand skew: uniform vs Zipf-like
- travel matrix structure: random, line layout, clustered doors
- compound coverage ratio: many vs few destinations naturally retained
- door congestion level: low, medium, high

For small instances, compute optimal or strong lower bounds with CP-SAT/MILP to report optimality gaps. For large instances, compare to best-known solutions across all methods and seeds.

---

## 12. Research Paper Positioning

### 12.1 Contributions

1. We propose a critical-path guided ALNS framework for compound-truck cross-docking scheduling with partial unloading.
2. We introduce an exact repair mechanism that optimizes only the destroyed bottleneck region while preserving the remaining schedule.
3. We develop a contextual reinforcement learning controller that adaptively selects destroy-repair operator pairs based on the current schedule state.
4. Computational experiments show that the proposed method improves upon VAA, SA, and RL-SA baselines in both solution quality and convergence speed.

### 12.2 Abstract draft

This study addresses a multi-door cross-docking truck scheduling problem with compound trucks and partial unloading, where destination assignment, dock-door assignment, and outbound sequencing must be jointly optimized to minimize makespan. Existing reinforcement learning based simulated annealing approaches primarily learn which local neighborhood structure to apply, but they do not explicitly identify the bottleneck components that determine the final completion time. We propose CPG-RL-ALNS, a critical-path guided adaptive large neighborhood search with exact repair and contextual reinforcement learning. The proposed method evaluates a schedule through a precedence representation, extracts critical doors, trucks, transfers, and sequence segments, and selectively destroys the bottleneck region. The destroyed subproblem is reconstructed using greedy, regret-based, matching-based, or CP-SAT exact repair operators. A contextual bandit controller adaptively selects destroy-repair-size actions from schedule-state features such as critical-door utilization, transfer intensity, waiting time, and recent operator performance. Computational experiments on small, medium, and large generated instances compare the proposed method against VAA, SA, RL-SA, vanilla ALNS, and ablated variants. The results are expected to demonstrate that targeting critical schedule regions improves both convergence speed and solution quality.

### 12.3 Method section outline

```text
1. Problem description
2. Schedule evaluation under partial unloading
3. Precedence graph and critical-path extraction
4. Critical-region destroy operators
5. Repair operators
   5.1 Greedy and regret-k repair
   5.2 Matching-based repair
   5.3 CP-SAT exact repair
6. Contextual RL operator selection
7. SA-style acceptance and algorithm summary
8. Complexity and implementation details
```

### 12.4 Experiment section outline

```text
1. Instance generation
2. Compared methods and parameter settings
3. Performance metrics
4. Main computational results
5. Convergence analysis
6. Operator behavior analysis
7. Ablation study
8. Sensitivity analysis
9. Discussion
```

### 12.5 Limitations and future work

- Exact repair can become expensive when destroyed regions are too large.
- MVP evaluator may overestimate loading start times if partial product arrivals could be loaded incrementally.
- LinUCB assumes a linear relation between context and reward; nonlinear policies may help later.
- The method depends on reliable schedule evaluation; inaccurate transfer modeling can mislead critical-path detection.
- Future work can add dynamic door capacity, stochastic arrival times, worker/forklift capacity, and online rescheduling.

---

## 13. 8-Minute Method Slide 구성

| Time | Slide | 핵심 메시지 |
|---:|---|---|
| 0:00-0:45 | 1. Problem | compound truck partial unloading 때문에 assignment-door-sequence가 강하게 결합됨 |
| 0:45-1:30 | 2. Limitation of RL-SA | 기존 RL은 neighborhood type만 고르고 bottleneck region을 직접 고치지 않음 |
| 1:30-2:20 | 3. Proposed Idea | makespan 원인을 critical path로 찾고 해당 region을 destroy/repair |
| 2:20-3:10 | 4. Schedule Evaluation | partial unloading, product availability, door sequence, transfer time 계산 |
| 3:10-4:05 | 5. Critical Region Destroy | critical door/truck/transfer/sequence/destination region 정의 |
| 4:05-5:00 | 6. Repair Layer | greedy, regret-k, matching, CP-SAT exact repair 비교 |
| 5:00-5:50 | 7. Contextual RL | action은 destroy-repair-size, context는 schedule bottleneck features |
| 5:50-6:40 | 8. Algorithm Flow | multi-start -> evaluate -> critical region -> repair -> SA accept -> RL update |
| 6:40-7:25 | 9. Experiment Design | baselines, ablations, metrics |
| 7:25-8:00 | 10. Expected Contribution | RL target을 neighborhood selection에서 bottleneck repair selection으로 확장 |

---

## 14. Implementation Roadmap

### Phase 1. Core correctness

1. `CrossDockInstance`
2. `Solution`
3. `check_feasible`
4. `evaluate_solution`
5. unit tests for partial unloading and door sequence

### Phase 2. MVP ALNS

1. random feasible initialization
2. critical door detection
3. critical door destroy
4. greedy repair
5. regret-k repair
6. SA acceptance
7. logging of makespan, accepted moves, best updates

### Phase 3. Research ALNS

1. full precedence graph
2. critical path extraction
3. high transfer destroy
4. sequence bottleneck destroy
5. destination regret destroy
6. local search polish

### Phase 4. Exact repair and RL

1. CP-SAT subproblem builder
2. exact repair fallback rules
3. reward function
4. LinUCB operator controller
5. operator analytics

### Phase 5. Experiments

1. instance generator
2. baselines
3. ablation runner
4. result tables
5. convergence plots

---

## 15. Recommended First Coding Request

다음 요청으로 구현을 시작하는 것이 가장 안정적이다.

```text
MVP 구현을 시작하자.

다음 파일 구조와 테스트 가능한 Python 코드를 작성해줘.

1. crossdock_solver/data/instance.py
2. crossdock_solver/core/solution.py
3. crossdock_solver/core/feasibility.py
4. crossdock_solver/core/evaluator.py
5. crossdock_solver/initial/random_init.py
6. crossdock_solver/alns/destroy.py
7. crossdock_solver/alns/repair.py
8. crossdock_solver/alns/acceptance.py
9. crossdock_solver/alns/loop.py
10. tests/test_evaluator.py
11. tests/test_simple_alns.py

아직 LinUCB, CP-SAT, full precedence graph는 넣지 말고,
나중에 확장 가능한 interface와 metadata field만 남겨줘.
```

