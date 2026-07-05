# SCIE 목표 연구 플랜: 학습 전이 기반 크로스도킹 스케줄링

목표 논문: "도착 시간창이 있는 컴파운드 트럭 크로스도킹 스케줄링을 위한 일반화 가능한
학습 기반 반복 지역탐색" (가제)

- 주 기여 1 (축 A): 인스턴스 분포에서 오프라인 학습한 정책을 처음 보는 인스턴스에
  재학습 없이 적용하는 학습 전이(learning-to-optimize) 프레임
- 주 기여 2: 도착 시간창(arrival time window) 확장 모델과 그에 맞춘 VAA/ILS/MILP
- 주 기여 3: 대규모·통계적으로 엄밀한 실험 (최적 갭, ablation, 동일 시간 예산)
- 타깃 저널: 1순위 Computers & Operations Research, 2순위 Computers & Industrial
  Engineering / International Journal of Production Research
- 전체 기간 추정: 약 5~6개월 (집필 포함)

## 현재 자산과 갭

| 자산 | 상태 | 갭 |
|---|---|---|
| 평가자 `core/evaluator.py` | 정확하지만 순수 파이썬 전체 재평가 | 대규모 반복 실험의 병목. 증분(delta) 평가 필요 |
| VAA `baselines/vaa.py` | 논문 충실 구현 | 시간창 인지 없음 |
| VAA-QRL `baselines/vaa_qrl.py` | 전 클래스 1위, 단 인스턴스별 재학습 | 전이 학습 구조로 확장 필요 |
| 그래프 상태 `baselines/graph_cargo_rl.py` | 가변 크기 그래프 인코딩 프로토타입 | 전이 정책의 상태 표현 출발점 |
| ALNS `alns/` | destroy/repair/loop 보유 | 튜닝 안 됨. 강한 베이스라인으로 재정비 필요 |
| MILP `exact/milp.py` | PuLP/CBC, MVP 모델 | 시간창 확장 + CP-SAT 도입 필요 (CBC는 중형 이상 무력) |
| 생성기 `data/generator.py` | 소형 랜덤 생성 | 규모/시간창/흐름 패턴 파라미터화 필요 |
| 환경 | torch, pulp 사용 가능 | ortools(CP-SAT), irace 또는 SMAC 설치 필요 |

> 진행 현황 (2026-07-03): P0 완료 (FastEvaluator L급 45µs로 G0 통과, 생성기/시드
> 프로토콜/러너 완성). P1 완료 (release/due 필드, 평가자·VAA·VAA-QRL·MILP 시간창
> 반영, MILP-평가자 목적값 일치 검증, 무제약 회귀 동일성 확인). ortools 설치 완료,
> CP-SAT 모델 구현이 다음 작업. 벤치마크 셀은 무제약 레벨 포함 48셀로 확정.

## Phase 0 — 실험 인프라 (2주)

대규모 실험이 가능한 토대를 먼저 만든다. 이후 모든 Phase가 여기 의존한다.

1. 증분 평가자
   - 무브 종류별 delta evaluation: 아웃바운드 재배치는 영향받는 도어 2개만,
     목적지 스왑은 관련 캐리어/이송 체인만 재계산
   - 전체 재평가와의 등가성 테스트 (기존 인스턴스 + 랜덤 무브 fuzz 테스트)
   - 목표: 트럭 60대 인스턴스에서 무브 평가 100µs 이하 (현재 전체 재평가 대비 ~50배)
   - 부족하면 numpy 벡터화 또는 핵심 루프 Cython/Rust 포팅 (예비 옵션)
2. 인스턴스 생성기 확장 (`data/generator.py`)
   - 규모 클래스: S(트럭 10/도어 5), M(30/10), L(60/20), XL(100/30)
   - 흐름 패턴: uniform / skewed(파레토) / clustered(목적지 그룹) 3종
   - 시간창 타이트니스: loose / medium / tight 3수준 (Phase 1 정의 사용)
3. 시드 프로토콜 (오버피팅 차단 — 최우선 원칙)
   - train pool (RL 학습 전용) / tuning pool (하이퍼파라미터·irace 전용) /
     test pool (보고 전용, 실험 확정 전까지 실행 금지)의 시드 대역 분리
   - `experiments/protocol.py`에 상수로 고정하고 문서화
4. 실험 러너
   - 결과를 append-only JSONL/CSV로 축적, 실패 재시작 가능, 병렬 실행(멀티프로세스)
   - wall-clock 예산 기반 종료 조건 지원 (iteration 기반과 병행)

산출물: 증분 평가자 + 등가성 테스트, 확장 생성기, 시드 프로토콜 문서, 러너.

## Phase 1 — 문제 확장: 도착 시간창 (2~3주)

1. 모델 정의
   - 트럭별 도착 시각 r_f: 도어 작업은 max(도어 가용, 목적지 준비, r_f) 이후 시작
   - 출발 마감 d_f: 소프트 제약, 목적함수 = makespan + λ·총 지연(tardiness)
     (하드 시간창은 가용성 붕괴 위험이 커서 소프트로 시작, λ 감도는 Phase 4에서)
   - 타이트니스 정의: 시간창 폭을 무제약 makespan 추정치의 비율로 매개변수화
2. 코드 반영 (의존 순서대로)
   - `data/instance.py`: r_f, d_f 필드 추가 (기본값으로 기존 인스턴스 호환 유지)
   - `core/evaluator.py` + 증분 평가자: 시작 시각 규칙과 지연 계산
   - `core/feasibility.py`: 변경 없음 예상 (소프트 제약이므로)
   - `baselines/vaa.py`: FT_m 삽입 시 r_f 반영, 목적지 우선순위에 마감 여유 반영
   - 기존 베이스라인 전부 시간창 인스턴스에서 동작 확인
   - `exact/milp.py`: 시간창 항 추가. CBC로 S급 검증 후 OR-Tools CP-SAT 병행 구현
     (CP-SAT가 스케줄링 하한·인컴번트 모두 CBC보다 강함)
3. 검증
   - Tiny급에서 MILP/CP-SAT 최적해와 평가자 makespan 일치 테스트
   - 시간창 도입 전후 회귀 테스트 (r_f=0, d_f=∞이면 기존 결과와 동일해야 함)

산출물: 확장 문제 정의 문서(수식 포함), 확장 평가자/MILP/CP-SAT, 회귀 테스트.

게이트 G1: CP-SAT가 M급(트럭 30대)에서 1시간 내 유의미한 하한을 못 주면
하한 전략을 LP 완화 + 조합 하한(도어 부하 하한)으로 교체.

## Phase 2 — 축 A: 학습 전이 모델 (5~6주, 핵심)

2a. 상태 표현 (1주)
- 탐색 상태 피처: 진행률, 온도, best 대비 갭, 정체 길이, 연산자별 최근 성공률,
  도어 부하 분산, 크리티컬 도어 집중도, 시간창 여유 통계
- 인스턴스 피처: 규모(트럭/도어 비), 흐름 밀도·왜도, 시간창 타이트니스
- 모든 피처는 규모 불변(비율·정규화)으로 설계 — 크기 일반화의 전제

2b. 정책 구조 (2주)
- 주 모델 GVAA-QRL: VAA-QRL의 연산자 선택을 tabular Q → 컨텍스트 DQN(torch,
  2층 MLP)으로 교체. 행동 = 연산자 9종(+시간창 유도 연산자 추가분)
- 시간창 유도 연산자 추가: 최대 지연 트럭의 최적 재배치, 마감 임박 목적지 스왑
- 보조 variant(옵션): `graph_cargo_rl`의 그래프 상태를 attention/GNN 인코더로
  교체한 학습 구성(construction) 정책 — 리스크 크므로 옵션으로 분리, 게이트 G2 이후 결정

2c. 학습 파이프라인 (2주)
- train pool S/M 인스턴스 수천 개에서 에피소드 = ILS 1회 실행으로 오프라인 학습
- replay buffer, target network, ε 스케줄. 체크포인트 + validation pool 성능 곡선
- RL 하이퍼파라미터는 tuning pool에서만 조정

2d. 내부 검증 (1주) — 게이트 G2
- validation pool에서 zero-shot GVAA-QRL vs (i) 균등 랜덤 연산자 선택,
  (ii) 인스턴스별 tabular VAA-QRL, 동일 wall-clock 비교
- 통과 기준: (i)를 통계적으로 유의하게 이기고 (ii)와 동등 이상
- 실패 시 피벗: 정책을 UCB/Thompson 컨텍스트 밴딧으로 단순화하고 논문 기여를
  "시간창 확장 + 엄밀 비교 연구"로 재포지셔닝 (Phase 3~4는 그대로 유효)

산출물: GVAA-QRL 구현, 학습 스크립트, 사전학습 체크포인트, G2 검증 리포트.

## Phase 3 — 비교군 구축 (3주, Phase 2와 병렬 가능)

1. 정확해법: CP-SAT(시간제한 1h) — S급 최적 갭, M/L급 하한 대비 갭
2. 튜닝된 ALNS: 기존 `alns/` 모듈에 Phase 1 연산자 반영, irace(또는 SMAC)로
   tuning pool에서 튜닝. "강한 전통 베이스라인" 역할
3. Paper-SA-RL: 시간창 대응 버전 (공정성: 동일 연산자 풀 접근권 부여 변형도 포함)
4. Q-ALNS 스타일 베이스라인 (Li et al. 2025 방법의 본 문제 이식): Q-learning이
   destroy/repair 연산자를 적응 선택하는 ALNS. 기존 `alns/` 모듈 + Q-learning
   선택 코드 재사용. 직접 재현이 아니라 방법 적응임을 논문에 명시
   (그들의 문제는 storage/AGV/도어 모드 구조라 인스턴스 이식 불가, 데이터 비공개)
5. 연산자 선택 대안 (ablation 겸 베이스라인): 균등 랜덤 / ALNS 룰렛 가중치 /
   UCB / Thompson / tabular Q / DQN — 동일 ILS 골격에서 선택기만 교체
6. (옵션) L2D류 GNN 디스패칭 재현 — 기간 초과 시 관련연구 비교표로 대체

산출물: 베이스라인 스위트, irace 튜닝 로그(재현성 자료).

## Phase 4 — 본 실험 (4주, 계산 시간 포함)

벤치마크 격자: 규모 4 (S/M/L/XL) × 흐름 패턴 3 × 시간창 3수준 = 36셀,
셀당 test 인스턴스 10개 × 반복 30회. 예산: 규모별 wall-clock 10s/60s/180s/600s.

| 실험 | 내용 | 논문 내 역할 |
|---|---|---|
| E1 주 비교 | 전 방법 × 전 셀, 평균±표준편차, 최적/하한 갭 | 메인 테이블 |
| E2 통계 검정 | Friedman + Holm posthoc, 쌍별 Wilcoxon, 효과크기 | 주장의 근거 |
| E3 크기 일반화 | S/M 학습 → L/XL zero-shot vs fine-tune vs from-scratch | 축 A 핵심 |
| E4 분포 이탈 | 학습에 없는 흐름 패턴·타이트니스에서 zero-shot | 축 A 핵심 |
| E5 ablation | -RL, -유도연산자, -킥재시작, -descent, 선택기 6종 교체 | 각 부품 기여 증명 |
| E6 시간-품질 | time-to-target, performance profile 곡선 | 동일 예산 공정성 |
| E7 감도 분석 | λ(지연 가중), restart_after, kick 크기 | 강건성 |
| E8 케이스 스터디 | 실제/준실제 물류센터 시나리오 1건 | 실무 어필 (데이터 확보 실패 시 문헌 기반 시나리오로 대체) |

원칙: test pool은 모든 구현·튜닝 확정 후 단 1회 실행. 모든 결과 원본은 저장소에 보존.

## Phase 5 — 집필·공개 (3~4주)

- 재현성 패키지: 코드, 인스턴스 생성 스크립트+시드, 사전학습 체크포인트,
  실험 러너 원커맨드화, README
- 논문 구성: 문제 정의(시간창 MILP 수식) → 방법(GVAA-QRL) → 실험(E1~E8) →
  일반화 분석 → 한계(단일 상품군 가정 등 잔여 단순화 명시)
- 원 논문(Shahmardan & Sajadieh) 인스턴스 확보 시도 (저자 컨택) — 성공 시 직접 비교 절 추가

## 게이트와 리스크

| 게이트 | 시점 | 기준 | 실패 시 |
|---|---|---|---|
| G0 | Phase 0 말 | 증분 평가자 등가성 + 목표 속도 | Cython/Rust 포팅 1주 추가 |
| G1 | Phase 1 말 | CP-SAT M급 하한 품질 | LP완화+조합 하한으로 교체 |
| G2 | Phase 2d | zero-shot이 랜덤 선택을 유의하게 이김 | 밴딧으로 단순화 + 논문 재포지셔닝 |
| G3 | Phase 4 중간 | E3에서 zero-shot이 from-scratch 대비 열세 아님 | fine-tune을 주 결과로 승격 |

주요 리스크: (1) 평가자 속도 — 전체 실험 시간을 지배, Phase 0에 배치한 이유.
(2) RL 전이 이득 부재 — G2 피벗 경로 확보. (3) 실데이터 미확보 — E8 대체안 준비.
(4) 계산 자원 — XL급 30반복은 병렬 머신 필요, 필요 시 클라우드 배치 검토.

## 타임라인 요약

| 주차 | Phase |
|---|---|
| 1~2 | P0 인프라 |
| 3~5 | P1 시간창 확장 (P3 일부 병렬 착수) |
| 6~11 | P2 학습 전이 + P3 비교군 병렬 |
| 12~15 | P4 본 실험 |
| 16~19 | P5 집필·공개 |
