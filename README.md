# 컴파운드 트럭 크로스도킹 스케줄링 (Cross-Dock CPG-ALNS MVP)

컴파운드 트럭 크로스도킹 스케줄링 연구의 MVP 구현 저장소입니다.

문서:
- `docs/scie_research_plan.md` — SCIE 목표 연구 플랜(Phase 0~5, 게이트)
- `docs/problem_definition.md` — 문제 정식화(논문↔코드 대응, 시간창 확장 수식)
- `docs/benchmark_design.md` — 논문 실험 세팅 대조와 벤치마크 설계 결정
- `docs/literature_scan_tw.md` — 시간창 변형 문헌 신규성 스캔

구현 범위:

- MVP 인스턴스·해(Solution) 데이터클래스
- 가용성(feasibility) 검사기
- 부분하역을 반영한 스케줄 평가자
- 랜덤 가용해 초기화
- VAA/VVA 스타일 구성 휴리스틱 베이스라인
- 논문 스타일 Q-learning 시뮬레이티드 어닐링 베이스라인
- 목적지-에이전트 RL 구성 베이스라인
- 화물 행렬(cargo-matrix) 목적지-에이전트 RL 베이스라인
- 가변 크기 그래프 화물 행렬 RL 베이스라인
- VAA-QRL 모델: VAA 초기해 + Q-learning 유도 반복 지역탐색
- 랜덤 베이스라인
- 크리티컬 도어 destroy, greedy/regret-k repair, SA 수락, 단순 ALNS 루프
- 베이스라인 비교 실험
- 연구 인프라 (P0): 고속 탐색 평가자, S/M/L/XL 벤치마크 생성기(흐름 패턴 포함),
  train/tuning/test 시드 프로토콜, 재개 가능한 병렬 실험 러너
- 도착 시간창 (P1): 트럭별 release/due 시각과 타이트니스 3수준,
  지연(tardiness) 인지 평가자, VAA/VAA-QRL/MILP 지원
  (목적함수 = makespan + 가중치 × 총 지연)

요청서의 `VVA`는 `VAA`(Vogel's Approximation Algorithm)의 흔한 오타로 간주하며,
구현에는 `vva_solution()` 별칭도 제공됩니다.

## 정확해 솔버와 하한

- `exact/milp.py` (PuLP/CBC), `exact/cpsat.py` (OR-Tools CP-SAT): 시간창 포함 MVP
  모델을 정확해로 풀어 최적/하한을 제공. CP-SAT는 소규모(S)에서 최적을 증명하고,
  미증명 시에도 하한을 반환한다.
- `exact/lower_bounds.py`: 조합적 하한(임계 트럭 체인 + 도어 면적). CP-SAT가 무력한
  대규모(L/XL)에서 유효한(느슨한) 하한을 제공. 자세한 G1 게이트 판정은
  `docs/scie_research_plan.md` 참조.

### 환경 주의 (CP-SAT)

이 환경은 pandas/pyarrow가 NumPy 1.x로 빌드되어 NumPy 2.x와 ABI 충돌한다. ortools
`cp_model`이 pandas를 import할 때 깨진 pyarrow 네이티브가 로드되면 프로세스가
SIGBUS로 죽는다. `exact/cpsat.py`는 정상 pandas가 없을 때 pandas/pyarrow를 최소
스텁으로 대체해 이를 우회한다(우리 코드는 pandas 기능을 쓰지 않음). 근본 해결은
`pip install -U pandas pyarrow` 등으로 NumPy 2 호환 빌드를 설치하는 것이다.

## 테스트 실행

```bash
python -m pytest -q
```

최근 결과:

```text
61 passed, 27 warnings in 41s
```

## MVP 예제 실행

```bash
python examples/run_mvp.py
```

출력 예:

```text
initial makespan: 2204.27
best makespan:    1456.27
critical door:    1
critical truck:   O2
iterations:       100
```

## 실험 환경

실험 환경은 베이스 논문(Shahmardan & Sajadieh 2020) 체제에 정합하도록 설계했다.
상세 근거는 `docs/benchmark_design.md`, 문제 정식화는 `docs/problem_definition.md`,
연구 플랜은 `docs/scie_research_plan.md`를 참조.

### 규모 클래스 (`crossdock_solver/data/generator.py`의 `SIZE_CLASSES`)

논문 체제: 컴파운드 트럭 다수(비율 2/3), |컴파운드| = |도어|, |목적지| = 1.5·|컴파운드|.
크기 표기는 논문과 동일한 (I, D, M) = (컴파운드, 목적지, 도어).

| 항목 | S | M | L | XL |
|---|---:|---:|---:|---:|
| 컴파운드 트럭 (I) | 6 | 12 | 20 | 30 |
| 아웃바운드 트럭 | 3 | 6 | 10 | 15 |
| 목적지 (D) | 9 | 18 | 30 | 45 |
| 도어 (M) | 6 | 12 | 20 | 30 |
| 상품 종류 | 3 | 3 | 3 | 3 |
| 컴파운드 비율 | 0.67 | 0.67 | 0.67 | 0.67 |
| 시간 예산 (논문 공식) | 31s | 126s | 350s | 788s |
| 근거 | 논문 Table 2 정확해 최대 | 중간 | 논문 Table 4 대규모 최대 | 논문 확장 |

- 목적지 수 = 총 트럭 수(컴파운드+아웃바운드)는 MVP·논문 공통 제약(목적지당 캐리어 1대).
- |컴파운드| ≤ |도어| 제약을 |컴파운드| = |도어|로 채택(도어당 컴파운드 1대, 논문 대표 구성).
- 시간 예산 = `((|I|+|D|)/2)·|M|·0.7`초 (논문 Table 5로 검증, `characteristics.paper_time_budget`).

### 공통 생성 규칙

- 화물량(flow): (컴파운드 × 목적지 × 상품)별 정수 U[0, 20]을 기본으로 흐름 패턴 적용
  - `uniform`: 독립 균등
  - `skewed`: 파레토 가중치로 소수 목적지에 물량 집중
  - `clustered`: 컴파운드별 홈 클러스터 목적지에 물량 집중 (그 외 20% 스케일)
- 파라미터 프로파일 (`param_profile`):
  - `synthetic` (기본): DE/DL ~ U[1,5], t_k ~ U[1,5], 100×100 평면 유클리드/10 이송
  - `paper`: 논문 정합. DE/DL ~ U[0,20], t_k ~ U[3,10], I-형 선형 도어(인접 이송 = 1, |m−n|)
- 시간창 (4수준): `none` / `loose` / `medium` / `tight`.
  무제약 makespan 추정치 H(= 총 핸들링의 2배 ÷ 도어 수)를 기준으로
  release ~ U[0, ρ·H], due = release + δ·H.
  (ρ, δ) = loose (0.10, 1.00), medium (0.25, 0.60), tight (0.50, 0.35) —
  tight일수록 도착이 늦게 흩어지고 마감 여유가 짧아진다.

### 인스턴스 특성값 (`crossdock_solver/data/characteristics.py`)

- `paper_time_budget(instance)`: 논문 시간 예산 공식 (E6 동일 시간 예산 비교용).
- `dbpr(instance)` / `mean_dbpr`: 목적지 화물 집중도(논문 Eq. 32). 높을수록 부분하역 효과 큼.
- `compound_fraction(instance)`: 컴파운드 트럭 비율(논문 체제 0.67~0.80).

### 벤치마크 격자와 시드 프로토콜 (`experiments/protocol.py`)

- 셀 = 규모 4 × 흐름 패턴 3 × 시간창 4수준 = 48셀
- 인스턴스 시드는 (풀, 셀, 인덱스)에서 결정적으로 유도
- 시드 풀 3종 완전 분리 (테스트 셋 오버피팅 차단):
  - `train` (100,000대): RL 정책 학습 전용
  - `tuning` (200,000대): 하이퍼파라미터 튜닝 전용
  - `test` (300,000대): 최종 보고 전용 — 실험 확정 전 실행 금지
- 본 실험 계획: 셀당 test 인스턴스 10개 × 반복 30회 (논문 반복 20회보다 엄격)

### 실험 러너 (`experiments/runner.py`)

- 결과는 append-only JSONL로 축적, 중단 후 재실행 시 완료된 작업 자동 스킵
- 멀티프로세스 병렬(`workers`), wall-clock 예산(`budget_sec`) 지원
- 메서드는 `experiments/methods.py`의 `METHOD_REGISTRY`에 등록

```python
from pathlib import Path
from experiments.runner import Job, run_jobs

jobs = [
    Job(method="VAA-QRL-300", pool="tuning", size_class="M",
        flow_pattern="uniform", tw_tightness="medium", index=0, budget_sec=126.0)
]
run_jobs(jobs, Path("outputs/results.jsonl"), workers=4)
```

### 평가자 (`crossdock_solver/core/`)

- `evaluator.evaluate_solution`: 정확한 기준 평가자(가용성 검사·부가 지표 포함).
- `fast_evaluator.FastEvaluator`: 탐색용 고속 평가자. 정적 테이블 사전계산으로
  기준 대비 makespan·지연 동일, 속도 수십 배(등가성 fuzz 테스트로 보증).
  `objective = makespan + tardiness_weight · total_tardiness`.

## 베이스라인

베이스라인 비교에 포함된 방법:

| 방법 | 설명 |
|---|---|
| `Random-1` | 랜덤 가용해 1개 |
| `Random-30` | 랜덤 가용해 30개 중 최선 |
| `VAA` | 논문 스타일 VAA 구성 휴리스틱: Eq. (23), FAT, FT_m 삽입 |
| `Paper-SA-RL5-300` | 논문 스타일 VAA 초기화 + Q-learning 이웃 선택 SA |
| `DestAgent-RL-150` | 목적지-에이전트 RL: 목적지별 에이전트가 캐리어 트럭을 학습, 도어는 release 기반 greedy 스케줄러가 배정 |
| `CargoMatrix-RL-150` | VAA 순서 목적지-에이전트 RL, 상태에 9 컴파운드 × 3 목적지 화물량 행렬 포함 |
| `GraphCargoMatrix-RL-150` | 트럭/목적지/도어 노드와 화물·이송 엣지의 가변 크기 그래프 RL |
| `VAA-QRL-300` | 최강 모델: VAA 초기해 + Q-learning 유도 반복 지역탐색 (크리티컬 도어 연산자, descent 폴리시, 킥 재시작) |

VAA 휴리스틱은 MVP 표현이 허용하는 범위에서 논문 구성을 따릅니다:

1. 논문 Eq. (23)으로 `T_id` 계산: 부분하역 시간 + 다른 컴파운드로부터의 적재 시간.
2. VAA 후회(regret)로 컴파운드 트럭에 유지 목적지 할당.
3. 남은 목적지를 `TEE_f`와 목적지 우선순위 `T_d`로 아웃바운드 트럭에 할당.
4. 도어를 `T_m`(다른 모든 도어까지의 거리 합)으로 정렬.
5. 모든 컴파운드 트럭 + (도어가 남으면) 적재 시간이 긴 아웃바운드로 `FAT` 구성.
6. FAT 트럭을 중앙 도어에 배정하고 도어 완료 시각 `FT_m` 갱신.
7. 남은 아웃바운드를 `T_d` 내림차순으로 `FT_m`이 가장 낮은 도어에 삽입.

논문 스타일 SA-RL 베이스라인은 Shahmardan & Sajadieh의 주 모델을 따릅니다:

1. VAA 휴리스틱 해에서 시작.
2. 논문의 일반/맞춤 이웃 구조로 무브 생성.
3. SA 수락.
4. Q-learning으로 이웃 구조 선택.
5. 생성 해가 현재 해 이하이면 보상 `1`, 아니면 `0`.
6. 무개선 카운트 구간을 Q-learning 상태로 사용, SA-RL5 임계값 `(5, 10, 15, 20)`.

목적지-에이전트 RL 베이스라인은 별도의 실험 모델입니다:

1. 각 목적지를 하나의 에이전트로 간주.
2. 모든 목적지 에이전트가 작은 NumPy MLP 하나를 공유.
3. 각 에이전트는 아직 남은 캐리어 트럭 하나를 선택.
4. ε-greedy 탐험과 리플레이 버퍼 업데이트.
5. VAA makespan 대비 개선을 팀 보상으로 전 에이전트에 공유.
6. 학습된 캐리어 할당을 release 기반 greedy 스케줄러로 완성.

화물 행렬 RL 베이스라인은 상태를 더 명시화한 변형입니다:

1. VAA 해에서 시작해 그 목적지 순서를 사용.
2. 상태에 고정 화물 행렬 포함: 최대 9 컴파운드 × 현재 3-목적지 윈도우.
3. 행렬 값은 각 컴파운드가 윈도우 목적지로 운반하는 화물량.
4. 가용 컴파운드 슬롯·활성 윈도우 슬롯 마스크 추가.
5. 각 목적지 에이전트가 가용 캐리어 트럭 하나를 선택.
6. 동일한 release 기반 greedy 스케줄러로 완성.

그래프 화물 행렬 변형은 상태 표현만 교체합니다 (`CargoMatrix-RL-150`은 ablation용 유지):

1. 트럭/목적지/도어 노드의 가변 크기 그래프 상태 구성.
2. 컴파운드 → 남은 목적지 화물 엣지 추가.
3. 도어 간 이송 엣지 추가.
4. 현재 부분 할당에서 예상 도어 릴리스/부하/가동률/크리티컬 도어 피처 추가.
5. 노드·엣지 피처를 mean/max/min/std로 풀링.
6. 고정 크기 임베딩을 공유 NumPy MLP 정책에 입력.

VAA-QRL 모델은 논문의 "VAA 초기해 + Q-learning 연산자 선택" 프레임을 유지하며
탐색을 강화합니다:

1. VAA 해에서 시작해 최선개선 descent로 폴리시.
2. 무개선 상태 기반 Q-learning이 매 반복 연산자 선택.
3. 연산자 풀 = 논문 이웃 7종 + 크리티컬 도어 유도 연산자 2종
   (크리티컬 도어 아웃바운드 최적 재배치, 크리티컬 트럭 목적지 최적 스왑).
4. 보상 셰이핑: 전역 신기록 `2`, 현재 해 이하 `1`, 그 외 `0`.
5. 기하 냉각 SA 수락; 신기록 없이 30회 지나면 best 해 + 3-move 랜덤 킥으로
   재시작하고 재가열.
6. 신기록 해(및 최종 best)는 아웃바운드 재배치·컴파운드 도어 스왑/이동·목적지
   스왑 전수 descent로 폴리시.

## Guided ILS(VAA-GILS) 동작 원리

주 모델의 정확한 동작 설명입니다. G2 검증에서 Q-learning 선택기가 균등 랜덤과
성능 차이가 없음이 실측되어, 논문에서는 이 엔진을 학습 없는 결정적 유도
반복 지역탐색(**VAA-GILS**)으로 제시하고 선택기(uniform/tabular Q/DQN)는
ablation의 실험 장치로 다룹니다 (`crossdock_solver/baselines/vaa_qrl.py`).

### 해 표현과 목적

알고리즘이 만지는 결정은 세 가지: 컴파운드 트럭별 (유지 목적지, 도어),
아웃바운드 트럭별 (목적지, 도어), 도어별 아웃바운드 처리 순서. 무브의 좋고
나쁨은 FastEvaluator가 objective = makespan + λ·총지연으로 채점합니다
(λ = `tardiness_weight`, 시간창 없는 인스턴스는 λ=0).

### 전체 뼈대 (의사코드)

```text
current ← VAA(instance)          # 구성 휴리스틱으로 시작해
current ← Descent(current)       # 지역 최적점까지 내려간 뒤 출발
best ← current
T ← 0.05 × obj(current)          # SA 초기 온도

t = 1 .. max_iterations 반복:
    a ← 선택기가 연산자 하나 선택          # uniform / tabular Q / DQN
    cand ← 연산자_a(current)              # 이웃해 하나 생성
    if obj(cand) < obj(best) − ε:         # 전역 신기록이면
        cand ← Descent(cand)              #   그 지역의 바닥까지 착취
        best ← cand; since_best ← 0
    else: since_best += 1
    if SA수락(obj(current), obj(cand), T): current ← cand
    T ← 0.995 × T                         # 기하 냉각
    if since_best ≥ 30:                   # 정체 감지
        current ← best + 랜덤 무브 3회     # 킥 재시작
        T ← max(T, 0.02 × obj(best))      # 재가열
        since_best ← 0

best ← Descent(best)              # 마지막 폴리시 후 반환
```

한 반복 = "한 발짝 제안 → 수락 판정"의 싼 사이클이고, 비싼 착취(descent)는
신기록이 났을 때만 발동합니다.

### 연산자 — "한 발짝"의 두 종류

- **일반 연산자 7종** (원논문 이웃구조): 무작위 트럭 쌍의 목적지/도어 스왑,
  임의 도어·위치 삽입 등. 평가 1회짜리 싼 무브, 방향은 무작위.
- **유도(guided) 연산자 4종**: 현재 해를 먼저 평가해 병목을 식별한 뒤 그
  병목만 겨냥한 미니 탐색을 수행. 예: `g1`은 ① 크리티컬 도어 식별 → ② 그
  도어 대기열의 마지막 아웃바운드 선택 → ③ 모든 (도어, 위치) 재배치 후보를
  전부 평가 → ④ 최선 배치 반환(개선 없으면 원해 유지). `g2`는 크리티컬
  트럭의 목적지를 전 트럭과 스왑 시도, `g3`/`g4`는 같은 논리를 "가장 지각한
  트럭"에 적용한 시간창 버전(지각 없으면 크리티컬 버전으로 폴백). 수십 번
  평가하는 비싼 한 발이지만 항상 병목을 때립니다.

### Descent — 신기록의 바닥을 긁는 착취기

```text
반복:
    가능한 모든 무브 나열:
      · 모든 아웃바운드 × 모든 도어 × 모든 삽입 위치 (재배치)
      · 모든 컴파운드 쌍 도어 스왑 + 컴파운드 → 빈 도어 이동
      · 모든 트럭 쌍 목적지 스왑
    전부 평가 → 최선 무브가 개선이면 적용하고 반복, 아니면 종료
```

최선개선(best-improvement) 방식이라 결정적이며, 종료 시점엔 이 무브셋 기준
지역 최적점이 보장됩니다. 한 패스에 수백 회 평가가 들지만 FastEvaluator
(수십 µs/회) 덕에 밀리초 단위입니다.

### SA 수락과 킥 재시작

- 후보가 현재보다 좋으면 무조건 수락, 나쁘면 확률 `exp(−Δ/T)`로 수락.
  온도는 매 반복 ×0.995로 냉각.
- 30회 연속 신기록이 없으면 current를 best 복사본으로 되돌린 뒤 일반 연산자
  3개를 무작위 강제 적용(킥)해 best 근방의 다른 분지로 점프시키고, 온도를
  `0.02 × obj(best)`까지 재가열. 킥 없이 best로만 되돌리면 같은 골짜기로
  재수렴하는 것이 튜닝 실험에서 확인되었습니다.

### 역할 분담과 비용

| 부품 | 역할 | 비용 |
|---|---|---|
| 일반 연산자 + SA | 넓게 찔러보기 (다변화) | 평가 1회/반복 |
| 유도 연산자 | 병목 정밀 타격 | 평가 수십 회/발동 |
| descent | 유망 지역의 바닥 착취 (강화) | 평가 수백 회/신기록 |
| 킥 재시작 | 정체 탈출 | 평가 4회/발동 |

이 분담이 완결적이어서 — 발견은 SA+연산자가, 마무리는 descent가, 탈출은
킥이 — 어떤 연산자를 고르는지(선택기)의 한계 효용이 0에 수렴합니다. 나쁜
제안은 SA가 거르고, 좋은 제안의 마무리는 descent가 하기 때문입니다. G2
검증에서 uniform ≈ tabular ≈ DQN이 나온 구조적 이유이며, 이것이 논문의
"학습 무효성 실증" 기여의 근거입니다.

## 베이스라인 실험

실행:

```bash
python experiments/compare_baselines.py
```

스크립트가 생성하는 파일:

- `outputs/baseline_results.csv`
- `outputs/baseline_summary.md`

실험 설정:

- 인스턴스 클래스: `Tiny`, `Small`, `Medium-lite` (레거시 비교용 소형 클래스)
- 클래스당 시드: 3개
- 랜덤 베이스라인: 1회 및 best-of-30
- 논문 모델: VAA 초기화 + Q-learning SA, 300 반복
- 목적지-에이전트/화물 행렬/그래프 RL: 인스턴스당 150 에피소드
- VAA-QRL: VAA 초기화 + Q-learning 유도 반복 지역탐색, 300 반복
- 갭: 같은 인스턴스에서 관측된 최고 방법 대비 백분율

최신 결과:

| Instance | Method | N | Avg makespan | Avg gap % | Avg runtime sec | Wins |
|---|---:|---:|---:|---:|---:|---:|
| Tiny | Random-1 | 3 | 436.43 | 17.80 | 0.0002 | 0 |
| Tiny | Random-30 | 3 | 420.32 | 9.06 | 0.0023 | 0 |
| Tiny | VAA | 3 | 430.65 | 13.78 | 0.0002 | 0 |
| Tiny | Paper-SA-RL5-300 | 3 | 379.49 | 0.67 | 0.0275 | 1 |
| Tiny | DestAgent-RL-150 | 3 | 385.65 | 2.98 | 0.0916 | 1 |
| Tiny | CargoMatrix-RL-150 | 3 | 385.65 | 3.02 | 0.0797 | 1 |
| Tiny | GraphCargoMatrix-RL-150 | 3 | 388.99 | 3.70 | 0.8653 | 0 |
| Tiny | VAA-QRL-300 | 3 | 377.65 | 0.00 | 0.0141 | 3 |
| Small | Random-1 | 3 | 1624.05 | 38.89 | 0.0001 | 0 |
| Small | Random-30 | 3 | 1370.93 | 15.36 | 0.0020 | 0 |
| Small | VAA | 3 | 1275.57 | 7.18 | 0.0003 | 0 |
| Small | Paper-SA-RL5-300 | 3 | 1195.21 | 0.53 | 0.0324 | 1 |
| Small | DestAgent-RL-150 | 3 | 1239.93 | 4.05 | 0.1375 | 0 |
| Small | CargoMatrix-RL-150 | 3 | 1229.26 | 3.29 | 0.1110 | 0 |
| Small | GraphCargoMatrix-RL-150 | 3 | 1236.44 | 3.73 | 1.0288 | 0 |
| Small | VAA-QRL-300 | 3 | 1190.34 | 0.05 | 0.0257 | 2 |
| Medium-lite | Random-1 | 3 | 2169.29 | 35.73 | 0.0002 | 0 |
| Medium-lite | Random-30 | 3 | 1810.84 | 11.90 | 0.0038 | 0 |
| Medium-lite | VAA | 3 | 1751.43 | 8.74 | 0.0009 | 0 |
| Medium-lite | Paper-SA-RL5-300 | 3 | 1643.75 | 1.84 | 0.0515 | 0 |
| Medium-lite | DestAgent-RL-150 | 3 | 1634.55 | 1.36 | 0.1922 | 0 |
| Medium-lite | CargoMatrix-RL-150 | 3 | 1656.78 | 2.80 | 0.1304 | 0 |
| Medium-lite | GraphCargoMatrix-RL-150 | 3 | 1639.07 | 1.60 | 1.7166 | 0 |
| Medium-lite | VAA-QRL-300 | 3 | 1613.51 | 0.00 | 0.0461 | 3 |

## 해석

논문 스타일 VAA는 논문 구성에 더 충실하게 재구현되어, 이전의 greedy 평가자
변형보다 덜 공격적입니다. 강한 지역 최적화기가 아니라 빠른 초기해 생성기입니다.

`VAA-QRL-300`이 모든 인스턴스 클래스에서 가장 강합니다: Tiny/Small/Medium-lite
전부에서 평균 makespan 최저, 평균 갭 0.00~0.05%, 9개 인스턴스 중 8승. 논문의
VAA + Q-learning 프레임을 유지하면서 크리티컬 도어 유도 연산자, 신기록 descent
폴리시, 재가열 킥 재시작을 더했고, 고속 탐색 평가자 덕분에 표에서 가장 빠른
학습 기반 방법이기도 합니다.

기존 베이스라인 중에서는 `Paper-SA-RL5-300`이 Tiny/Small에서,
`DestAgent-RL-150`이 Medium-lite에서 가장 강합니다. 예상 도어 릴리스/부하
피처를 쓰는 `GraphCargoMatrix-RL-150`은 Medium-lite에서 고정
`CargoMatrix-RL-150`보다 낫지만 Tiny/Small에서는 못하고, 결정 단계마다 예상
도어 프로파일을 구축·평가하느라 상당히 느립니다.
