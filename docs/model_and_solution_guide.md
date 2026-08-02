# 논문 Mathematical Model 및 Solution Approach 쉬운 설명

대상 논문: *Truck scheduling in a multi-door cross-docking center with partial unloading*

## 1. 한 줄 요약

이 논문은 여러 dock door가 있는 cross-docking 센터에서, 물건을 일부만 내리고 다시 싣는 compound truck과 outbound truck을 어느 목적지와 dock door에 배정하고 어떤 순서로 처리할지 결정해서 전체 작업 완료시간, 즉 makespan을 최소화하는 문제를 다룬다.

논문의 핵심은 두 가지다.

- Section 2.2: 이 문제를 정확히 풀기 위한 mixed-integer mathematical model을 만든다.
- Section 3: 정확한 모델은 큰 문제에서 너무 느리므로, VAA 기반 초기해와 simulated annealing, 그리고 RL 기반 neighborhood 선택으로 빠르게 좋은 해를 찾는다.

## 2. 문제를 쉽게 이해하기

일반 outbound truck은 cross-dock에서 물건을 싣고 특정 목적지로 출발한다. compound truck은 조금 다르다. compound truck은 이미 여러 목적지의 물건을 싣고 들어오며, 특정 목적지를 맡게 되면 그 목적지 물건은 굳이 내리지 않고 그대로 가지고 갈 수 있다. 대신 나머지 목적지 물건은 내려야 한다. 또한 자신이 맡은 목적지의 부족한 물건은 다른 compound truck에서 내려온 물건을 받아서 싣는다.

간단한 예시는 다음과 같다.

| Truck | 처음 싣고 온 물건 |
|---|---|
| C1 | A 목적지 10개, B 목적지 5개 |
| C2 | A 목적지 4개, B 목적지 6개 |

C1이 A 목적지를 맡으면:

- C1은 A 목적지 10개를 내리지 않고 그대로 유지한다.
- C1은 B 목적지 5개만 내린다.
- C1은 C2가 내린 A 목적지 4개를 받아 싣는다.
- 따라서 C1의 작업시간은 "B를 내리는 시간 + C2에서 온 A를 싣는 시간 + 필요한 이동시간"에 영향을 받는다.

C1이 B 목적지를 맡으면 반대로 A 목적지 10개를 내려야 하고, C2에서 온 B 목적지 6개를 받아 싣는다. 논문은 이런 선택이 전체 makespan에 큰 영향을 준다고 보고, compound truck과 목적지의 배정을 먼저 중요하게 다룬다.

## 3. Mathematical Model

### 3.1 모델이 결정하는 것

수학 모델은 다음 결정을 동시에 한다.

- 각 compound truck이 어떤 목적지를 맡을지
- 각 outbound truck이 어떤 목적지를 맡을지
- 각 truck이 어느 dock door를 사용할지
- 같은 dock door를 쓰는 outbound truck들의 처리 순서
- compound truck과 outbound truck의 loading 시작 시각
- 모든 truck 작업이 끝나는 최종 완료시간, 즉 makespan

즉, 단순히 "목적지 배정"만 하는 모델이 아니라, 목적지 배정, dock door 배정, 순서, 시간 제약을 한 번에 고려하는 통합 스케줄링 모델이다.

### 3.2 주요 집합

| 기호 | 의미 |
|---|---|
| `I` | compound truck 집합 |
| `F` | outbound truck 집합 |
| `D` | 목적지 집합 |
| `M` | dock door 집합 |
| `K` | 제품 종류 집합 |

논문의 실험 설정에서는 목적지마다 정확히 하나의 배송 truck이 필요하므로, compound truck과 outbound truck의 수를 합치면 목적지 수와 맞도록 문제가 구성된다.

### 3.3 주요 파라미터

| 기호 | 의미 |
|---|---|
| `DE_i` | truck `i`가 dock door에 들어와 unloading/loading 준비를 마치는 데 걸리는 시간 |
| `DL_i` | truck `i`가 dock door를 떠나 yard로 이동하는 데 걸리는 시간 |
| `f_{i,d,k}` | compound truck `i`가 처음부터 싣고 온, 목적지 `d`로 가야 하는 제품 `k`의 수량 |
| `t_{m,n}` | dock door `m`에서 dock door `n`으로 제품 1단위를 옮기는 시간 |
| `t_k` | 제품 `k` 1단위를 싣거나 내리는 시간 |
| `b_{i,d}` | compound truck `i`가 목적지 `d` 물건을 가지고 있으면 1, 아니면 0 |

### 3.4 주요 변수

| 변수 | 의미 |
|---|---|
| `Cmax` | 전체 makespan |
| `a_i` | compound truck `i`가 loading을 시작하는 시각 |
| `d_f` | outbound truck `f`가 loading을 시작하는 시각 |
| `Y_{i,d,m}` | compound truck `i`가 목적지 `d`와 dock door `m`에 배정되면 1 |
| `Z_{f,d,n}` | outbound truck `f`가 목적지 `d`와 dock door `n`에 배정되면 1 |
| `P_{f,l}` | outbound truck `f`와 `l`이 같은 door에 있고 `f`가 `l`보다 먼저 처리되면 1 |
| `Q_f` | outbound truck `f`가 같은 door의 outbound sequence에서 첫 번째이면 1 |
| `L_f` | outbound truck `f`가 같은 door의 outbound sequence에서 마지막이면 1 |

### 3.5 목적함수

목적함수는 다음 하나다.

```text
minimize Cmax
```

`Cmax`는 모든 compound truck과 outbound truck의 완료시간 중 가장 큰 값이다. 따라서 이 값을 줄인다는 것은 센터 전체 작업이 가장 빨리 끝나도록 스케줄을 짠다는 의미다.

### 3.6 제약조건을 목적별로 풀어보기

#### A. Makespan 정의

제약 (1), (2)는 `Cmax`가 모든 truck의 완료시간보다 크거나 같아야 한다고 둔다.

- compound truck 완료시간 = loading 시작시각 `a_i` + 다른 truck에서 받아 싣는 시간 + dock을 떠나는 시간 `DL_i`
- outbound truck 완료시간 = loading 시작시각 `d_f` + 해당 목적지 물건을 싣는 시간 + dock을 떠나는 시간 `DL_f`

즉 어떤 truck 하나라도 늦게 끝나면 그 시간이 전체 makespan이 된다.

#### B. Compound truck의 partial unloading

제약 (3)은 compound truck이 자기 loading을 시작하려면, 자신이 맡지 않은 목적지의 물건을 모두 unload해야 한다고 둔다.

예를 들어 C1이 A 목적지를 맡으면 A 물건은 내릴 필요가 없다. B, C 목적지 물건만 unload한다. 이 제약이 논문의 partial unloading을 직접 반영하는 핵심 제약이다.

#### C. Compound truck이 loading하기 전 제품이 도착해야 함

제약 (4)는 compound truck이 어떤 목적지를 맡았을 때, 그 목적지의 물건 중 다른 compound truck에 있던 물건들이 unload되고, 필요한 dock door로 이동한 뒤에야 loading을 시작할 수 있다고 둔다.

예를 들어 C1이 A 목적지를 맡고, C2가 A 물건 4개를 가지고 있다면:

- C2가 A 물건을 unload해야 한다.
- 그 물건이 C2의 door에서 C1의 door로 이동해야 한다.
- 그 후에야 C1은 A 물건 4개를 추가로 load할 수 있다.

이 제약 때문에 논문 모델은 단순한 배정 문제가 아니라 material transfer timing까지 고려한다.

#### D. 같은 dock door를 쓰는 compound와 outbound의 충돌 방지

제약 (5)는 outbound truck이 compound truck과 같은 dock door를 쓰는 경우를 다룬다.

같은 door를 쓴다면 outbound truck은 compound truck이 loading을 끝내고 door를 떠난 뒤에야 들어올 수 있다. 따라서 outbound truck의 시작시각에는 compound truck의 완료, compound truck의 exit time, outbound truck의 entering time이 모두 반영된다.

이 제약은 "진행 중인 inbound 또는 compound 작업이 있는 dock door를 다른 truck이 동시에 사용할 수 없다"는 상황을 모델에 포함한다.

#### E. Outbound truck도 제품이 도착해야 loading 가능

제약 (6)은 outbound truck이 맡은 목적지의 물건이 모든 compound truck에서 unload되고, outbound truck의 dock door까지 이동한 뒤에야 loading할 수 있다고 둔다.

예를 들어 O1이 B 목적지를 맡고, B 물건이 C1과 C2에 나뉘어 있다면 O1은 C1과 C2의 B 물건이 모두 내려지고 O1의 door에 도착할 때까지 기다려야 한다.

#### F. 같은 door의 outbound truck 순서

제약 (7)은 같은 dock door에 여러 outbound truck이 배정된 경우의 순서를 다룬다.

예를 들어 같은 door에 O1 다음 O2가 처리된다면:

```text
O2 시작시각 >= O1 loading 완료 + O1 exit time + O2 entering time
```

따라서 하나의 door에서 outbound truck들이 겹쳐 처리되는 일이 없다.

#### G. Outbound sequence 변수의 일관성

제약 (8)-(12)는 `P`, `Q`, `L` 변수가 말이 되도록 만든다.

- 어떤 outbound truck은 같은 door에서 첫 번째이거나, 누군가의 뒤에 온다.
- 어떤 outbound truck은 같은 door에서 마지막이거나, 누군가의 앞에 온다.
- 같은 door에 배정된 truck 사이에서만 선후관계가 생긴다.

즉 sequence 변수가 실제 door 배정과 모순되지 않게 묶어주는 제약이다.

#### H. 배정 제약

제약 (13)-(16)은 배정의 기본 규칙이다.

- 각 compound truck은 정확히 하나의 목적지와 하나의 dock door에 배정된다.
- 하나의 dock door에는 최대 하나의 compound truck만 배정된다.
- 각 outbound truck은 정확히 하나의 목적지와 하나의 dock door에 배정된다.
- 각 목적지는 compound truck 또는 outbound truck 중 정확히 하나에게만 배정된다.

여기서 중요한 점은 목적지가 중복 배송되지도 않고 누락되지도 않는다는 것이다.

#### I. 변수 타입과 Big-M

제약 (17), (18)은 이진 변수와 연속 변수의 타입을 정의한다.

제약 (19)-(22)는 Big-M 값을 정의한다. Big-M은 특정 조건이 성립할 때만 시간 제약을 켜기 위한 장치다. 예를 들어 "두 truck이 같은 door에 있을 때만 순서 제약을 적용한다" 같은 조건부 제약을 mixed-integer model 안에서 표현하기 위해 사용한다.

### 3.7 모델의 핵심 의미

이 mathematical model은 다음 네 가지를 동시에 고려한다.

- partial unloading: compound truck이 맡은 목적지 물건은 내리지 않는다.
- product availability: 싣기 전에 물건이 unload되고 해당 door까지 도착해야 한다.
- dock door capacity: 같은 door에서 truck들이 동시에 처리되지 않는다.
- destination coverage: 모든 목적지는 정확히 한 truck에게 배정된다.

따라서 논문의 exact model은 현실 제약을 꽤 강하게 반영하지만, binary variable과 sequencing variable이 많아져서 큰 문제에서는 계산이 어렵다.

## 4. Solution Approach

### 4.1 왜 heuristic과 metaheuristic이 필요한가

논문은 destination assignment를 완화해도 truck scheduling 문제가 NP-hard라고 설명한다. 따라서 통합 모델 전체는 더 어렵다. 작은 인스턴스는 CPLEX/GAMS 같은 exact solver로 풀 수 있지만, 실제 규모에서는 합리적인 시간 안에 최적해를 보장하기 어렵다.

그래서 논문은 다음 순서로 접근한다.

1. 빠른 constructive heuristic `H`로 초기 feasible solution을 만든다.
2. 그 해를 simulated annealing의 초기해로 사용한다.
3. 여러 neighborhood search 중 어떤 것을 쓸지 RL 기반으로 학습한다.

### 4.2 Heuristic Algorithm H

Heuristic `H`는 "처음부터 괜찮은 스케줄을 빠르게 만드는" 절차다. 논문에서 매우 중요한 이유는 SA가 random solution에서 시작하지 않고 이 heuristic solution에서 시작하기 때문이다.

#### Step 1: compound truck과 목적지의 비용 `T_{i,d}` 계산

`T_{i,d}`는 compound truck `i`가 목적지 `d`를 맡는다고 가정했을 때의 대략적인 작업 부담이다.

구성은 두 부분이다.

- compound truck `i`가 목적지 `d` 외의 물건을 unload하는 시간
- 목적지 `d` 물건 중 다른 compound truck에 들어 있는 물건을 `i`가 load해야 하는 시간

쉽게 말하면:

```text
T_{i,d} = i가 d를 맡으면 내려야 하는 시간 + 다른 truck에서 받아 실어야 하는 시간
```

이 값이 작을수록 truck `i`가 목적지 `d`를 맡는 것이 자연스럽다.

#### Step 2: VAA로 compound truck 목적지 배정

논문에서 VAA는 전체 스케줄을 직접 만드는 알고리즘이 아니다. VAA는 compound truck을 목적지에 배정하는 데 사용된다.

VAA의 직관은 regret이다.

- 어떤 truck 또는 목적지에서 가장 좋은 선택과 두 번째로 좋은 선택의 차이를 본다.
- 차이가 크면 지금 좋은 선택을 놓쳤을 때 손해가 크다는 뜻이다.
- 손해가 큰 row 또는 column부터 먼저 확정한다.

예를 들어 C1이 A를 맡는 비용이 10, B를 맡는 비용이 30이면 regret은 20이다. C1은 A를 놓치면 손해가 크므로 우선적으로 A 배정을 고려한다.

이 단계가 끝나면 일부 목적지는 compound truck이 맡고, 남은 목적지는 outbound truck이 맡는다.

#### Step 3: 목적지와 truck의 우선순위 계산

논문은 다음 값을 계산한다.

| 값 | 의미 |
|---|---|
| `UT_i` | compound truck `i`가 실제로 unload해야 하는 총 시간 |
| `LT_d` | 목적지 `d`의 총 loading 시간 |
| `T_d` | 목적지 `d`가 얼마나 빨리 준비되고 끝날 수 있는지 보는 우선순위 값 |

`T_d`는 dock door 간 이동시간을 일단 무시하고, 목적지별 작업 부담과 대기 가능성을 보는 값이다. 큰 `T_d`를 가진 목적지는 오래 걸릴 가능성이 있으므로 먼저 좋은 door 또는 truck에 배정할 후보가 된다.

#### Step 4-5: outbound truck을 남은 목적지에 배정

outbound truck마다 다음 값을 계산한다.

```text
TEE_f = DE_f + DL_f
```

즉 outbound truck이 들어오고 나가는 데 걸리는 기본 changeover 부담이다.

논문은 `TEE_f`가 작은 outbound truck부터, 아직 배정되지 않은 목적지 중 `T_d`가 큰 목적지에 배정한다. 직관적으로는 빨리 들어오고 나갈 수 있는 outbound truck에게 부담이 큰 목적지를 먼저 맡기는 방식이다.

#### Step 6: dock door 중심성 `T_m` 계산

각 door `m`에 대해 다른 door들과의 이동시간 합을 계산한다.

```text
T_m = sum_n t_{m,n}
```

`T_m`이 작을수록 센터의 중앙에 가까운, 물건 이동에 유리한 door다. 논문은 작업 부담이 큰 truck을 더 좋은 door에 먼저 배정하려 한다.

#### Step 7-8: FAT 구성 및 첫 truck door 배정

`FAT`는 First Assigned Trucks의 약자다. 각 door에서 처음 처리할 truck 후보 목록이다.

논문의 규칙은 다음과 같다.

- compound truck은 모두 FAT에 넣는다.
- door 수가 compound truck 수보다 많으면, loading 부담이 큰 outbound truck 일부도 FAT에 넣는다.
- FAT의 truck들을 `T_d` 기준으로 정렬한다.
- 부담이 큰 truck을 `T_m`이 작은 좋은 door에 배정한다.

즉 "오래 걸릴 가능성이 큰 첫 작업을 이동 조건이 좋은 door에 둔다"는 전략이다.

#### Step 9-11: 시작시각과 door finish time 계산

compound truck에 대해 loading 시작시각과 완료시각을 계산하고, 각 door의 현재 완료시각 `FT_m`을 갱신한다.

그 다음 FAT에 포함된 outbound truck의 시작시각과 완료시각을 계산한다. 아직 배정되지 않은 outbound truck은 다음 규칙으로 순차 삽입한다.

- 남은 outbound truck 중 `T_d`가 가장 큰 목적지를 맡은 truck을 선택한다.
- 현재 가장 빨리 비는 door, 즉 `FT_m`이 가장 작은 door에 넣는다.
- 해당 door의 finish time을 갱신한다.

이 과정에서 door 경합이 반영된다. 처음 VAA 단계에서는 door 경합을 직접 보지 않지만, 이후 FAT와 `FT_m` 기반 삽입 단계에서 같은 door에서 겹치지 않도록 순서를 만든다.

#### Step 12: makespan 계산

모든 door와 truck의 완료시각을 계산한 뒤, 가장 늦은 완료시각을 `Cmax`로 둔다.

### 4.3 VAA의 역할을 정확히 구분하기

논문에서 VAA는 다음 역할만 한다.

```text
compound truck -> 목적지 배정
```

VAA 자체가 dock door 경합, outbound sequence, product transfer timing을 모두 직접 최적화하는 것은 아니다. 이 제약들은 그 뒤의 heuristic scheduling 단계와 mathematical model의 시간 제약에서 반영된다.

따라서 논문식 흐름은 다음처럼 이해하는 것이 정확하다.

```text
T_{i,d} 계산
-> VAA로 compound-destination 배정
-> 남은 목적지를 outbound에 배정
-> door 중심성과 truck 부담으로 door 배정
-> door별 sequence와 start/finish time 계산
-> makespan 평가
```

### 4.4 Simulated Annealing

논문의 SA는 heuristic `H`가 만든 해를 초기해로 사용한다. 일반적인 SA는 random solution에서 시작할 수 있지만, 이 논문은 H가 매우 빠르게 좋은 feasible solution을 만들기 때문에 H의 결과를 출발점으로 삼는다.

#### Solution representation

해는 `3 x D` 형태의 matrix로 표현된다.

| Row | 의미 |
|---|---|
| 1행 | 각 truck의 목적지 배정 |
| 2행 | 각 truck의 dock door 배정 |
| 3행 | door 내 처리 순서 |

열은 compound truck과 outbound truck을 나타낸다. 이 표현으로 목적지, door, sequence를 하나의 solution 안에서 동시에 바꿀 수 있다.

#### Neighborhood structures

SA는 현재 solution을 조금 바꾼 candidate solution을 만들고, 좋아지면 받아들이며, 나빠져도 확률적으로 받아들여 local optimum에서 빠져나오게 한다. 이때 solution을 바꾸는 방법이 neighborhood structure다.

논문에서 설명된 주요 neighborhood는 다음과 같다.

| 구분 | 설명 |
|---|---|
| `k=1` | 두 truck의 목적지를 swap |
| `k=2` | 두 compound truck의 dock door를 swap |
| `k=3` | 두 outbound truck의 dock door를 swap |
| `k=4` | outbound truck 하나를 다른 dock door로 insertion |
| `k=6` | compound truck을 `UT_i` 기반 roulette으로 고르고, `T_m` 기반 roulette으로 door를 골라 이동 |
| `k=7` | outbound truck을 loading 부담 기반 roulette으로 고르고, `T_m` 기반 roulette으로 door를 골라 이동 |
| `k=8` | compound truck과 목적지를 `T_{i,d}` 기반으로 골라 재배정 |

논문 본문에서는 일반적인 swap/insertion뿐 아니라 문제 특화 neighborhood를 추가한다. 특히 `UT_i`, `T_m`, `T_{i,d}` 같은 cross-docking 문제의 구조를 활용한다는 점이 중요하다.

### 4.5 RL 기반 neighborhood 선택

SA에서 중요한 질문은 "다음 candidate를 만들 때 어떤 neighborhood를 쓸 것인가"다. 무작위로 고르면 어떤 move가 현재 상황에 좋은지 학습하지 못한다.

논문은 이를 reinforcement learning 문제로 본다.

- action: neighborhood structure 선택
- environment: 현재 schedule과 평가 결과
- reward: candidate solution이 현재 solution보다 같거나 좋으면 1, 아니면 0
- goal: 개선 가능성이 높은 neighborhood를 더 자주 선택

#### MAB 방식

논문은 neighborhood를 slot machine처럼 본다. 어떤 neighborhood는 현재 문제에서 개선을 자주 만들고, 어떤 것은 거의 개선하지 못할 수 있다. MAB 방식은 각 neighborhood의 평균 보상을 추정해서 더 좋은 action을 선택하려 한다.

논문이 비교한 MAB 계열은 다음과 같다.

| 이름 | 의미 |
|---|---|
| SA-RL1 | incremental average로 action value 업데이트 |
| SA-RL2 | non-stationary 환경을 가정하고 고정 learning rate로 업데이트 |
| SA-RL3 | UCB Type 1, 덜 시도한 action도 exploration |
| SA-RL4 | UCB Type 2, UCB와 non-stationary 업데이트 결합 |

#### Q-learning 방식

Q-learning에서는 단순히 action 평균만 보지 않고 state도 본다. 논문에서 state는 현재 solution이 얼마나 오래 개선되지 않았는지, 즉 no-improvement count `NI`로 정의된다.

예를 들어 SA-RL5는 threshold를 다음처럼 둔다.

```text
(5, 10, 15, 20)
```

`NI`가 작으면 아직 현재 주변 탐색이 잘 되고 있는 상태이고, `NI`가 커지면 오래 정체된 상태다. 오래 정체되면 더 과감한 neighborhood가 필요할 수 있다.

논문은 두 가지 Q-learning 설정을 비교한다.

| 이름 | threshold |
|---|---|
| SA-RL5 | `(5, 10, 15, 20)` |
| SA-RL6 | `(10, 20, 50, 100)` |

각 반복에서 action 선택은 대략 다음 세 가지를 섞는다.

- 일정 확률로 random selection
- 일정 확률로 value 기반 roulette-wheel selection
- 나머지는 현재 가장 좋아 보이는 greedy action 선택

이 구조는 exploration과 exploitation을 동시에 하려는 장치다.

## 5. 논문 접근법의 전체 흐름

전체 알고리즘 흐름은 다음과 같이 정리할 수 있다.

```text
입력 데이터
  |
  v
compound truck과 목적지별 T_{i,d} 계산
  |
  v
VAA로 compound truck 목적지 배정
  |
  v
남은 목적지를 outbound truck에 배정
  |
  v
door 중심성 T_m 및 FAT 구성
  |
  v
door 배정, sequence 구성, start/finish time 계산
  |
  v
초기 feasible solution H 완성
  |
  v
SA 반복
  |
  +-- RL이 neighborhood 선택
  +-- candidate schedule 생성
  +-- makespan 평가
  +-- SA acceptance로 current solution 갱신
  +-- reward로 RL value 업데이트
  |
  v
best schedule 반환
```

## 6. 현재 코드와의 대응

현재 저장소의 구현과 논문 개념은 다음처럼 대응된다.

| 논문 개념 | 현재 코드 |
|---|---|
| VAA 기반 heuristic H | `crossdock_solver/baselines/vaa.py` |
| 논문식 `T_{i,d}` 계산 | `crossdock_solver/baselines/vaa.py`의 `_compound_destination_cost` |
| FAT 및 `FT_m` 기반 door 삽입 | `crossdock_solver/baselines/vaa.py`의 `_assign_first_trucks_to_doors`, `_assign_remaining_outbounds_by_ftm` |
| schedule 평가와 makespan 계산 | `crossdock_solver/core/evaluator.py` |
| 논문식 SA-RL5/SA-RL6 baseline | `crossdock_solver/baselines/paper_sa_rl.py` |
| 우리가 제안한 CPG-ALNS | `crossdock_solver/alns/` |

중요한 차이는 현재 MVP가 논문의 full exact MIP를 solver로 직접 푸는 구조는 아니라는 점이다. 현재 코드는 논문의 scheduling logic을 evaluator와 heuristic/metaheuristic으로 구현해 비교하는 방식이다. 논문의 exact mathematical model까지 완전히 추가하려면 별도의 MILP 모델링 계층, 예를 들어 OR-Tools CP-SAT, Pyomo, PuLP, Gurobi, CPLEX 연동이 필요하다.

## 7. 핵심 정리

Mathematical model의 핵심은 다음이다.

- 목적지 배정, dock door 배정, outbound 순서, 시작시각을 동시에 결정한다.
- partial unloading을 명시적으로 반영한다.
- 제품이 unload되고 door까지 이동해야 loading할 수 있다는 product availability 제약을 둔다.
- 같은 door에서 truck이 동시에 처리되지 않도록 sequence 제약을 둔다.
- 목적함수는 전체 작업 완료시간 `Cmax` 최소화다.

Solution approach의 핵심은 다음이다.

- exact model은 큰 문제에서 느리므로 heuristic과 metaheuristic을 사용한다.
- VAA는 compound truck과 목적지 배정을 위한 초기 단계다.
- door 경합과 순서는 VAA 이후 heuristic scheduling 단계에서 반영된다.
- SA는 H가 만든 좋은 초기해에서 출발한다.
- RL은 어떤 neighborhood를 선택할지 학습해서 SA의 탐색 성능을 높인다.

