# VAA-GILS for Compound-Truck Cross-Docking with Time Windows

컴파운드 트럭과 부분 하역을 갖는 multi-door cross-docking truck scheduling
문제를 다루는 연구 코드입니다. 기존 모델의 "모든 트럭이 시간 0에 도착"한다는
가정과 순수 makespan 목적을 완화하고, 트럭별 도착 시각과 소프트 마감을 포함한
Time Window 확장 문제를 구현합니다.

이 저장소의 중심 기여는 **VAA-GILS**입니다. VAA 구성 휴리스틱으로 좋은 초기해를
만든 뒤, 현재 스케줄의 병목을 직접 찾아 움직이는 guided iterated local search
엔진입니다. 학습 정책을 핵심으로 두지 않고, 문제 구조와 빠른 평가자, 강한
지역탐색으로 CP-SAT 기준해에 근접하는 것이 목표입니다.

## 핵심 기여

1. **Time Window 문제 확장**
   - 트럭별 도착 시각을 두어, 해당 시각 이전에는 작업을 시작할 수 없게 합니다.
   - 트럭별 소프트 마감을 두어, 완료 시각이 늦으면 그만큼 지연 비용을 계산합니다.
   - 목적은 전체 완료시간과 총 지연의 가중합을 최소화하는 것입니다.
   - MILP, CP-SAT, 조합적 하한, train/tuning/test seed 분리 벤치마크를 함께 제공합니다.

2. **VAA-GILS**
   - VAA 초기해를 출발점으로 사용합니다.
   - best-improvement descent로 초기해, 새 최고해, 최종해를 폴리시합니다.
   - critical door, critical truck, most-tardy truck을 찾아 병목 위치만 집중적으로 움직입니다.
   - simulated annealing 수락과 kick restart로 지역 최적점 탈출을 처리합니다.
   - FastEvaluator가 makespan과 총 지연을 동일하게 재현하면서 탐색 평가를 빠르게 만듭니다.

3. **학습 기반 연산자 선택의 한계 분석**
   - 같은 GILS 엔진 위에서 uniform, tabular Q-learning, transfer DQN 선택기를 비교합니다.
   - 실험에서는 learned selector가 uniform 선택을 안정적으로 이기지 못했습니다.
   - 결론은 "강한 deterministic local search 엔진에서는 학습 선택기의 기여가 작거나 사라질 수 있다"는 것입니다.

## 결과 요약

정확해 기준이 있는 셀에서만 near-optimal을 주장합니다. S 규모는 CP-SAT 300초,
M/L 일부는 CP-SAT 600초로 비교했고, CP-SAT가 가능해를 찾지 못한 큰 Time Window
셀에서는 best-known 기준으로 보고합니다.

| 관찰 | 결과 |
|---|---|
| CP-SAT 기준 셀 | GILS가 CP-SAT incumbent 대비 약 0.1~0.6% 이내 |
| S-none 셀 | CP-SAT가 모든 인스턴스 최적 증명, GILS는 증명된 최적해에 근접 |
| M-none 셀 | 최고 GILS 실행이 일부 CP-SAT incumbent를 개선 |
| 베이스라인 대비 | GILS-tabular가 VAA 대비 평균 2.39%, Paper-SA-RL5 대비 평균 1.16% 우수 |
| 실행 시간 | GILS 1,000 iteration 평균 S 0.11초, M 0.52초, L 2.3초 |
| CP-SAT 실행 시간 | 평균 S 76초, M 237초, L 376초 |
| 선택기 분석 | tabular Q와 uniform은 실질 차이가 작고, transfer DQN은 큰 예산에서 유의하게 나쁨 |

아래는 [paper/apiems2026_draft.md](paper/apiems2026_draft.md)의 Table 1 전체입니다.
test pool 기준이며, 셀당 5개 인스턴스와 5회 반복으로 계산했습니다. `Δbk`는
인스턴스별 best-known 해 대비 평균 gap입니다.

| Cell | Method | Mean obj ± std | Δbk (%) | vs CP-SAT (%) |
|---|---|---|---:|---:|
| S-none | VAA | 1443.3 ± 165.6 | 6.49 | +6.49 |
| | Paper-SA-RL5 | 1375.9 ± 185.3 | 1.21 | +1.21 |
| | GILS-uniform | 1362.5 ± 184.3 | 0.21 | +0.21 |
| | GILS-tabular | 1364.1 ± 184.7 | 0.33 | +0.33 |
| | GILS-DQN | 1370.0 ± 186.5 | 0.75 | +0.75 |
| S-medium | VAA | 6426.2 ± 1869.2 | 3.33 | +3.33 |
| | GILS-uniform | 6227.0 ± 1802.3 | 0.11 | +0.11 |
| | GILS-tabular | 6229.5 ± 1798.2 | 0.18 | +0.18 |
| | GILS-DQN | 6249.7 ± 1796.4 | 0.55 | +0.55 |
| S-tight | VAA | 9278.4 ± 1189.2 | 2.17 | +2.17 |
| | GILS-uniform | 9091.9 ± 1084.6 | 0.21 | +0.21 |
| | GILS-tabular | 9088.9 ± 1086.6 | 0.17 | +0.17 |
| | GILS-DQN | 9103.1 ± 1079.5 | 0.34 | +0.34 |
| M-none | VAA | 2567.5 ± 354.4 | 3.97 | +4.80 |
| | Paper-SA-RL5 | 2509.5 ± 344.4 | 1.64 | +0.99 |
| | GILS-uniform | 2479.0 ± 340.0 | 0.40 | +0.13 |
| | GILS-tabular | 2478.5 ± 340.7 | 0.38 | +0.11 |
| | GILS-DQN | 2478.8 ± 339.5 | 0.40 | +0.19 |
| M-medium | VAA | 21737.8 ± 3260.9 | 3.29 | — |
| | GILS-uniform | 21153.7 ± 3296.8 | 0.43 | — |
| | GILS-tabular | 21156.2 ± 3298.5 | 0.44 | — |
| | GILS-DQN | 21182.4 ± 3301.2 | 0.56 | — |
| M-tight | VAA | 26982.5 ± 3985.5 | 1.26 | — |
| | GILS-uniform | 26701.6 ± 3952.3 | 0.21 | — |
| | GILS-tabular | 26697.7 ± 3951.5 | 0.19 | — |
| | GILS-DQN | 26721.0 ± 3956.5 | 0.28 | — |
| L-none | VAA | 6050.8 ± 1701.2 | 1.67 | — |
| | Paper-SA-RL5 | 6046.3 ± 1705.0 | 1.57 | — |
| | GILS-uniform | 5972.2 ± 1687.8 | 0.28 | — |
| | GILS-tabular | 5967.8 ± 1682.6 | 0.22 | — |
| | GILS-DQN | 5968.6 ± 1681.0 | 0.24 | — |
| L-medium | VAA | 54288.8 ± 19106.9 | 0.88 | — |
| | GILS-uniform | 53955.7 ± 19029.2 | 0.19 | — |
| | GILS-tabular | 53935.2 ± 19034.1 | 0.14 | — |
| | GILS-DQN | 53962.1 ± 19028.8 | 0.20 | — |
| L-tight | VAA | 124638.8 ± 34762.6 | 0.59 | — |
| | GILS-uniform | 123961.1 ± 34581.4 | 0.05 | — |
| | GILS-tabular | 123949.1 ± 34589.2 | 0.04 | — |
| | GILS-DQN | 123963.1 ± 34580.9 | 0.06 | — |

자세한 실험 해석은 [paper/apiems2026_summary_ko.md](paper/apiems2026_summary_ko.md)를
참조하세요.

## 문제 설정

컴파운드 트럭은 입고와 출고 역할을 동시에 수행합니다. 한 컴파운드 트럭은 여러
목적지의 화물을 싣고 들어오며, 자신이 담당할 목적지의 화물만 남기고 나머지는
하역합니다. 다른 트럭에서 내려진 같은 목적지의 화물은 해당 carrier 트럭으로
이송되어 다시 적재됩니다.

해가 결정해야 하는 것은 세 가지입니다.

- 각 컴파운드 트럭이 담당할 목적지와 사용할 도어
- 각 outbound 트럭이 담당할 목적지와 사용할 도어
- 같은 도어를 공유하는 outbound 트럭들의 처리 순서

Time Window 확장은 다음처럼 문장 기준으로 구현되어 있습니다.

- **도착 시각.** 트럭마다 도착 시각이 있고, 그 전에는 작업을 시작할 수 없습니다.
- **소프트 마감.** 트럭마다 마감이 있고, 완료가 늦으면 늦은 만큼 지연으로 누적합니다.
- **목적.** 전체 완료시간과 총 지연의 가중합을 최소화합니다. 마감이 없으면 기존 makespan 최소화 문제로 돌아갑니다.

## VAA-GILS 동작 방식

코드상 엔진은 [crossdock_solver/baselines/vaa_qrl.py](crossdock_solver/baselines/vaa_qrl.py)에
있습니다. 초기 이름은 `VAA-QRL`이었지만, 논문과 README에서는 실제 기여에 맞춰
**VAA-GILS**로 부릅니다. Q-learning은 기본 선택기 중 하나일 뿐이고, 엔진의
성능을 만드는 핵심은 guided operators, descent, restart입니다.

### 1. Construction

VAA 휴리스틱으로 초기해를 만듭니다.

- 컴파운드 트럭의 유지 목적지를 regret 기준으로 배정합니다.
- 남은 목적지는 outbound 트럭에 배정합니다.
- 도어 완료 시각과 도착 시각을 고려해 트럭을 삽입합니다.

### 2. Descent

다음 이동들을 전수 평가하면서 best-improvement 방식으로 내려갑니다.

- outbound 트럭을 다른 도어와 위치로 재배치
- 컴파운드 트럭의 도어 swap 또는 빈 도어 이동
- 두 트럭의 담당 목적지 swap

초기해, 새 최고해, 최종해에 descent를 적용하므로, 좋은 후보가 나오면 같은
이웃 구조 안에서 가능한 바닥까지 바로 내려갑니다.

### 3. Guided Operators

일반 무작위 이웃만 쓰지 않고, 현재 평가 결과에서 병목을 찾아 직접 타격합니다.

| 연산자 | 역할 |
|---|---|
| `g1_critical_outbound_relocate` | critical door의 마지막 outbound를 최적 도어/위치로 재배치 |
| `g2_critical_destination_swap` | critical truck의 목적지를 다른 트럭과 swap |
| `g3_tardy_truck_relocate` | 가장 크게 지각한 트럭을 기준으로 재배치 |
| `g4_tardy_destination_swap` | 가장 크게 지각한 트럭을 기준으로 목적지 swap |

지각 트럭이 없으면 tardy 연산자는 critical 연산자로 fallback합니다.

### 4. Acceptance and Restart

후보가 현재해보다 좋으면 수락하고, 나쁘면 simulated annealing 확률로 일부
수락합니다. 일정 iteration 동안 최고해가 갱신되지 않으면 최고해 근처에서
무작위 kick을 적용하고 온도를 다시 올립니다. 이 restart가 없으면 같은 지역으로
반복 수렴하는 문제가 생깁니다.

### 5. Fast Evaluation

[crossdock_solver/core/fast_evaluator.py](crossdock_solver/core/fast_evaluator.py)는
기준 평가자와 같은 makespan 및 총 지연을 반환하면서 탐색용 부가 정보를 함께
냅니다.

- critical door
- critical truck
- 가장 크게 지각한 truck
- 목적함수 값

이 평가자가 guided operator와 descent를 싸게 만들어 GILS가 짧은 시간 안에
많은 후보를 비교할 수 있습니다.

## 선택기 실험

GILS의 operator selector는 plug-in입니다.

| 방법명 | 의미 |
|---|---|
| `GILS-uniform-1000` | 연산자를 균등 무작위 선택 |
| `GILS-1000` | tabular Q-learning 선택기 |
| `GILS-dqn-1000` | train pool에서 학습한 DQN을 test pool에 zero-shot 적용 |

이 비교는 "학습을 붙이면 좋아지는가"를 보기 위한 실험 장치입니다. 현재 결과는
GILS 엔진 자체가 충분히 강해서, 연산자 선택 학습의 추가 이득이 매우 작다는 쪽을
지지합니다.

## 코드 구조

| 경로 | 내용 |
|---|---|
| `crossdock_solver/data/` | 인스턴스 dataclass, benchmark generator, 인스턴스 특성값 |
| `crossdock_solver/core/` | feasibility, 기준 evaluator, FastEvaluator |
| `crossdock_solver/baselines/vaa.py` | VAA 구성 휴리스틱 |
| `crossdock_solver/baselines/vaa_qrl.py` | VAA-GILS 엔진과 guided operators |
| `crossdock_solver/baselines/paper_sa_rl.py` | 논문 스타일 SA-RL baseline |
| `crossdock_solver/rl/` | tabular selector, DQN selector, feature vector |
| `crossdock_solver/exact/` | MILP, CP-SAT, 조합적 lower bounds |
| `experiments/protocol.py` | train/tuning/test seed protocol |
| `experiments/methods.py` | 실험 method registry |
| `experiments/k1_run.py` | 메인 실험 batch 구성 |
| `experiments/k1_summary.py` | K1 결과 요약 |
| `experiments/k1_stats.py` | Wilcoxon 검정과 budget 분석 |
| `paper/` | 논문 초안과 한국어 요약 |
| `docs/` | 문제 정의, 벤치마크 설계, 연구 플랜 |

## 빠른 실행

테스트:

```bash
python -m pytest -q
```

MVP 예제:

```bash
python examples/run_mvp.py
```

예상 출력 형태:

```text
initial makespan: 2204.27
best makespan:    1456.27
critical door:    1
critical truck:   O2
iterations:       100
```

이미 생성된 K1 결과 요약:

```bash
python experiments/k1_summary.py
python experiments/k1_stats.py
```

K1 실험 재실행:

```bash
python experiments/k1_run.py search
python experiments/k1_run.py budget
python experiments/k1_run.py cpsat
```

`search`는 GILS/VAA/SA-RL 중심의 빠른 batch이고, `cpsat`은 오래 걸리는 정확해
batch입니다.

## 벤치마크 프로토콜

주요 K1 실험은 다음 격자를 사용합니다.

| 항목 | 설정 |
|---|---|
| 규모 | S, M, L |
| 흐름 패턴 | uniform |
| Time Window | none, medium, tight |
| 인스턴스 | 셀당 test 인스턴스 5개 |
| 반복 | stochastic method는 인스턴스당 5회 |
| 정확해 | S는 CP-SAT 300초, M/L 일부는 CP-SAT 600초 |
| seed | train, tuning, test pool 완전 분리 |

더 큰 설계 격자와 생성 규칙은 [docs/benchmark_design.md](docs/benchmark_design.md)와
[experiments/protocol.py](experiments/protocol.py)에 있습니다.

## 주요 method 이름

| method | 설명 |
|---|---|
| `VAA` | 구성 휴리스틱 baseline |
| `Paper-SA-RL5-1000` | 원 논문 스타일 Q-learning SA baseline |
| `GILS-uniform-1000` | uniform selector를 붙인 VAA-GILS |
| `GILS-1000` | tabular Q selector를 붙인 VAA-GILS |
| `GILS-dqn-1000` | DQN selector를 붙인 VAA-GILS |
| `CPSAT-300`, `CPSAT-600` | OR-Tools CP-SAT 정확해/하한 계산 |

전체 registry는 [experiments/methods.py](experiments/methods.py)에 있습니다.

## 정확해와 하한

- [crossdock_solver/exact/milp.py](crossdock_solver/exact/milp.py): PuLP/CBC 기반 MILP
- [crossdock_solver/exact/cpsat.py](crossdock_solver/exact/cpsat.py): OR-Tools CP-SAT 모델
- [crossdock_solver/exact/lower_bounds.py](crossdock_solver/exact/lower_bounds.py): critical-chain, door-area, structural tardiness lower bound

CP-SAT는 인컴번트와 증명된 하한을 함께 반환합니다. 다만 큰 Time Window 셀에서는
600초 안에 가능해를 못 내는 경우가 있어, GILS 결과는 best-known 관점에서도
함께 보고합니다.

## 문서

- [paper/apiems2026_summary_ko.md](paper/apiems2026_summary_ko.md): 한국어 실험 요약
- [paper/apiems2026_draft.md](paper/apiems2026_draft.md): APIEMS 2026 draft
- [paper/apiems2026_draft_ko.md](paper/apiems2026_draft_ko.md): 한국어 draft
- [docs/problem_definition.md](docs/problem_definition.md): 문제 정식화와 코드 대응
- [docs/benchmark_design.md](docs/benchmark_design.md): 벤치마크 설계 근거
- [docs/literature_scan_tw.md](docs/literature_scan_tw.md): Time Window 변형 문헌 스캔
- [docs/scie_research_plan.md](docs/scie_research_plan.md): 연구 단계와 gate 기록

## 환경 메모

이 저장소에는 고정된 requirements 파일이 없습니다. 현재 코드 경로는 NumPy, pytest,
PuLP, OR-Tools, PyTorch를 사용합니다. CP-SAT import 과정에서 pandas/pyarrow와
NumPy ABI가 충돌하는 환경을 위해 [crossdock_solver/exact/cpsat.py](crossdock_solver/exact/cpsat.py)에
최소 stub 우회가 들어 있습니다. pandas 기능 자체는 사용하지 않습니다.

## 한 줄 요약

이 저장소는 컴파운드 트럭 부분 하역 크로스도킹에 Time Window를 넣은 문제를
정식화하고, 정확해와 하한으로 검증 가능한 벤치마크 위에서 **학습보다 문제 구조를
직접 쓰는 VAA-GILS가 더 안정적인 주력 방법**임을 보이는 코드입니다.
