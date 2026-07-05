# 문제 정의: 도착 시간창이 있는 컴파운드 트럭 크로스도킹 스케줄링

이 문서는 (1) 베이스 논문(Shahmardan & Sajadieh 2020, C&IE 139:106134)의 모델을
정식으로 옮기고, (2) 본 연구가 추가한 도착 시간창(release)과 출발 마감(due) 확장을
명시하며, (3) 저장소 코드와의 대응을 정리한다.

## 1. 집합 · 인덱스

| 기호 | 의미 | 코드 |
|---|---|---|
| I | 컴파운드 트럭 (하역 후 아웃바운드로 재사용) | `compound_trucks` |
| F | 아웃바운드 트럭 (적재·배송 전용) | `outbound_trucks` |
| D | 목적지 | `destinations` |
| K | 상품 종류 | `product_types` |
| M | 도크 도어 | `doors` |

관계: 각 트럭(컴파운드·아웃바운드)은 목적지 하나에 배정되고, 각 목적지는 정확히
한 트럭이 담당한다(캐리어). 따라서 |I| + |F| = |D|, 그리고 |I| ≤ |M|.
(논문 Section 2.1 가정, 제약 (13)–(16). 우리 `instance.validate()`와 동일.)

## 2. 파라미터

| 기호 | 의미 | 코드 |
|---|---|---|
| DE_i | 트럭 i의 진입 시간 (도어 도착·하역 준비까지) | `enter_time[i]` |
| DL_i | 트럭 i의 출차 시간 (도어 이탈·yard 이동까지) | `leave_time[i]` |
| f_idk | 컴파운드 i에 초기 적재된, 목적지 d행 상품 k의 개수 | `flow[i,d,k]` |
| t_k | 상품 종류 k 단위당 하역/적재 시간 | `product_time[k]` |
| t_mn | 도어 m→n 단위 이송 시간 | `travel_time[m,n]` |
| b_id | 컴파운드 i가 목적지 d행 화물을 실었으면 1 (Σ_k f_idk > 0) | `unit_amount(i,d) > 0` |

파생량:
- 핸들링 시간 h_id = Σ_k f_idk · t_k = `handling_time(i,d)`
- 목적지 부하 L_d = Σ_{i∈I} h_id (목적지 d 전체 적재 시간)

## 3. 결정 변수 (논문)

| 기호 | 의미 |
|---|---|
| τ | makespan (최대 완료 시각) |
| a_i | 컴파운드 i의 적재 시작 시각 |
| δ_f | 아웃바운드 f의 적재 시작 시각 (논문 표기 d_f — 아래 마감 d̄와 구분 위해 δ 사용) |
| Y_idm | 컴파운드 i가 도어 m·목적지 d에 배정되면 1 |
| Z_fdn | 아웃바운드 f가 도어 n·목적지 d에 배정되면 1 |
| Q_f, L_f | f가 같은 도어의 아웃바운드 중 첫 번째 / 마지막이면 1 |
| P_fl | f, l이 같은 도어이고 f가 l보다 앞서면 1 |

해(Solution) 표현은 코드에서 `compound_assignment: truck→(dest,door)`,
`outbound_assignment: truck→(dest,door)`, `door_sequences: door→[trucks]`로 대응.
(논문 Fig. 4의 3×D 행렬 표현과 동치.)

## 4. 목적함수와 핵심 제약 (논문)

목적: **min τ** (makespan).

- (1),(2) τ ≥ 각 컴파운드/아웃바운드 트럭의 완료 시각.
- (3) 컴파운드는 자신이 담당하지 않는 목적지 화물의 하역이 끝난 뒤 적재 시작.
- (5) 같은 도어라면 아웃바운드는 그 도어의 컴파운드가 떠난 뒤 적재 시작.
- (6) 아웃바운드는 담당 화물이 배정 도어에 도착한 뒤 적재 시작.
- (8)–(12) 같은 도어 아웃바운드들의 순서.
- (13) 각 컴파운드는 도어 1·목적지 1. (14) 도어당 컴파운드 최대 1대.
- (15) 각 아웃바운드는 도어 1·목적지 1.
- (16) 각 목적지는 전체 트럭 중 정확히 하나에 배정 (캐리어 유일).
- big-M M_1..M_4는 Eq. (19)–(22).

## 5. 타이밍 모델 (평가자 기준)

해가 주어지면 makespan은 다음으로 결정된다 (`core/evaluator.py`,
`core/fast_evaluator.py` 동일 규칙):

컴파운드 i (담당 목적지 d, 도어 m):
```
unload_finish_i = r_i + DE_i + (Σ_{d'≠d} h_{i,d'})           # 부분하역: 담당 d 제외
dest_ready_d    = max_{s∈I, s≠carrier, b_{s,d}>0} ( unload_finish_s + t_{door(s)→m} )
load_start_i    = max(unload_finish_i, dest_ready_d)
finish_i        = load_start_i + (Σ_{s∈I, s≠i} h_{s,d}) + DL_i
```

아웃바운드 f (담당 목적지 d, 도어 m, 도어 내 직전 완료 prev):
```
start_f  = max(prev, dest_ready_d, r_f)
finish_f = start_f + DE_f + L_d + DL_f
```

makespan = max_f finish_f. (논문에서 r_i = r_f = 0.)

## 6. 본 연구의 확장

논문 Section 2.1의 두 가정을 완화한다.

### 6.1 도착 시간창 (release) — "모든 트럭 시각 0 도착" 가정 완화

- 파라미터 r_f ≥ 0 (트럭 f의 도착/가용 시각) 추가. `release_time[f]`.
- 5절 타이밍의 시작 시각에 반영: 컴파운드는 `unload_finish`에 r_i 가산,
  아웃바운드는 `start_f = max(prev, dest_ready, r_f)`.
- r_f = 0 이면 논문 모델과 완전히 동일 (회귀 동일성 테스트로 보장).

### 6.2 출발 마감 (due) + 지연 — 순수 makespan 목적 확장

- 파라미터 d̄_f (트럭 f의 마감; 코드 `due_time[f]`, 미지정 시 +∞) 추가.
- 소프트 제약: 마감 초과분을 지연으로 벌점.
  ```
  tardiness_f  = max(0, finish_f - d̄_f)
  total_tardiness = Σ_f tardiness_f
  objective = τ + λ · total_tardiness          # λ = tardiness_weight
  ```
- 하드 제약이 아닌 소프트로 둔 이유: 타이트한 인스턴스에서 실행 가능해 유지.
- d̄_f = +∞ (또는 λ = 0) 이면 objective = makespan 으로 논문과 동일.

### 6.3 MILP 확장 (`exact/milp.py`)

- 트럭별 지연 변수 tard_f ≥ 0, tard_f ≥ finish_f − d̄_f (마감 있는 트럭만).
- 목적: min τ + λ · Σ_f tard_f.
- release는 a_i, δ_f 시작 제약의 하한으로 추가.
- Tiny급에서 MILP 최적해를 평가자로 재평가해 목적값 일치 검증 완료.

## 7. 시간창 인스턴스 생성 (`data/generator.py`)

무제약 makespan 추정치 H = 2 · (Σ h_id) / |M| 를 기준으로:
- r_f ~ U[0, ρ·H], d̄_f = r_f + δ·H
- 타이트니스 (ρ, δ): loose (0.10, 1.00) / medium (0.25, 0.60) / tight (0.50, 0.35)
- `tw_tightness=None`: r_f=0, d̄_f=∞ (논문 원본 조건)

## 8. 표기 주의

- 논문의 d_f (아웃바운드 적재 시작 시각)와 본 연구의 마감 d̄_f는 다른 양이다.
  본 문서는 적재 시작을 δ_f, 마감을 d̄_f로 구분한다.
- 논문 DE/DL = 코드 enter_time/leave_time. 논문 τ = 코드 makespan.

## 9. 복잡도

목적지 배정을 완화하면 트럭 스케줄링(NP-hard, Kuo 2013)으로 환원되므로 본 문제도
NP-hard. 소규모만 정확해가 실용적이며, 대규모는 휴리스틱(VAA)+메타휴리스틱으로
근사한다. 시간창 추가는 복잡도 등급을 낮추지 않는다.
