# 시간창 변형 문헌 신규성 스캔 (2026-07-03)

질문: "컴파운드 트럭 + 부분하역 크로스도킹 스케줄링에 도착 시간창(release)과
출발 마감(due)을 결합한 변형"이 기존 연구에 존재하는가?

## 스캔 방법

1. 베이스 논문 확인: Shahmardan & Sajadieh (2020), "Truck scheduling in a
   multi-door cross-docking center with partial unloading — RL-based SA",
   Computers & Industrial Engineering 139:106134.
2. Semantic Scholar 인용 그래프: 베이스 논문을 인용한 약 52편 전수 스캔
   (제목/초록 기준: 컴파운드/부분하역 여부 x 시간창 여부 x RL 여부).
3. 키워드 검색: "compound truck" x TW, "partial unloading" x TW,
   crossdock TW survey, DRL crossdock generalization.

## 핵심 결과

### 1. 컴파운드 트럭 x 시간창 조합은 발견되지 않음

베이스 논문 인용 ~52편 중 컴파운드 트럭/부분하역을 시간창(도착/마감)과 결합한
논문 없음. 컴파운드 트럭 라인의 후속 자체가 희소하며, 가장 가까운 것은
"repeated loading" 신선식품 스케줄링 (Pan et al. 2021, 시간창 없음).

### 2. 시간창 크로스도킹 자체는 성숙 분야 (신규성 주장 불가 영역)

일반(순수 인바운드/아웃바운드) 트럭 스케줄링에서 시간창·도착시각·마감은 다수:
- Crossdock truck scheduling with time windows: earliness/tardiness/storage
  (J. Intelligent Manufacturing 2014)
- Scheduling trucks in a multi-door cross-docking system with time windows (2019)
- Truck scheduling with fixed due dates and shipment sorting (C&IE 2018)
- Cross-dock scheduling with TW + departure deadline (Scientia Iranica 2023)
- Arrival times + learning effect (Production & Manufacturing Research 2014)
- Scheduling under full/partial/no information on inbound arrivals (C&OR 2011)

따라서 주장 문구는 반드시 "시간창이 있는 크로스도킹 최초"가 아니라
**"컴파운드 트럭·부분하역 모델의 시간창 확장 최초"** 로 한정해야 한다.

### 3. 최근접 경쟁 논문: Li et al., EJOR 2025 (Q-ALNS)

"Integrated trucks assignment and scheduling problem with mixed service mode
docks: A Q-learning based ALNS" (EJOR 2025; arXiv:2412.09090). 본문 확인 결과:

| 축 | Li et al. 2025 | 본 연구 |
|---|---|---|
| 시간창 | 있음: (r_i, d_i), 마감 연장 허용 | 있음: release + soft due |
| 유연성 위치 | 도어 수준 (mixed service mode dock) | 트럭 수준 (compound truck + 부분하역) |
| 트럭 역할 | 순수 인바운드/아웃바운드로 고정 | 컴파운드 (하역+적재 겸용) |
| 목적함수 | tardiness + makespan + 이송거리 | makespan + weighted tardiness |
| RL | Q-learning, 인스턴스별 온라인 | 오프라인 사전학습 -> zero-shot 전이 (축 A) |
| 최대 규모 | 트럭 200대 / 도어 10개 | XL: 트럭 100대 / 도어 30개 |

시사점:
- 주제(크로스도킹 + 시간창 + Q-learning 메타휴리스틱)가 2025년 EJOR에 실릴 만큼
  시의성 있음이 확인됨. 관련연구 절에서 반드시 비교 포지셔닝 필요.
- 차별점 두 축이 선명해짐: (1) 트럭 수준 유연성(컴파운드+부분하역) vs 도어 수준,
  (2) 전이 가능한 사전학습 정책 vs 인스턴스별 온라인 학습.
- 규모 기준 상향 압박: 이들이 200대까지 실험 -> XL(100대)로는 부족해 보일 수 있음.
  XXL(200대급) 셀 추가 검토.

### 4. 크로스도킹 특화 "학습 전이/일반화"는 미개척으로 보임

DRL 일반화(크기/분포) 연구는 VRP·JSSP에 집중. 크로스도킹 트럭 스케줄링에서
오프라인 사전학습 정책의 미지 인스턴스 zero-shot 적용을 다룬 논문은 이번
스캔에서 발견되지 않음. 단, 이 주장은 집필 전 별도 정밀 스캔 1회 더 필요.

## 신규성 판정

- 주장 1 "도착 시간창·출발 마감이 있는 컴파운드 트럭(부분하역) 크로스도킹
  스케줄링의 최초 정식화": **방어 가능** (이번 스캔 기준).
- 주장 2 "크로스도킹 트럭 스케줄링에서 최초의 오프라인 학습-전이 정책":
  **잠정 방어 가능**, 집필 시점 재검증 필요.

## 주의사항 (스캔의 한계)

1. Semantic Scholar 인용 목록이 완전하지 않을 수 있음 (특히 최신·비영어 문헌).
2. 중국어권 문헌 미스캔.

## 베이스 논문 모델 정합성 (원문 확인, 2026-07-05)

원문 Section 2.1 정독 결과, 앞서 "MVP가 논문보다 좁다"고 본 것은 오류였음.
논문도 우리 MVP와 동일한 핵심 가정을 사용:

- "Each truck ... should be assigned to just one destination" → 목적지당 캐리어
  1대(분할 배송 없음) + |컴파운드| + |아웃바운드| = |목적지| (우리 validate와 동일)
- "The number of compound trucks is less than or equal to that of doors" →
  |컴파운드| ≤ |도어| (우리 validate와 동일)
- Fig. 3 예시: 목적지 4, 컴파운드 3, 아웃바운드 1 → 3+1=4, 목적지당 트럭 1대

따라서 목적지·캐리어·트럭수 축에서 우리 MVP는 논문의 축소판이 아니라 정합.
모델 확장(옵션 1)은 논문 충실성을 위해 불필요 (넣으면 논문을 넘어서는 별개 기여).

우리가 실제로 추가한 것 = 논문의 두 가정 완화:
- "All compound and outbound trucks are available at the beginning of horizon
  time" → 도착 시각 r_f 도입으로 완화
- 목적함수가 순수 makespan → 마감 d_f + 지연(tardiness) 항 추가

결론: 신규성 주장 1("논문 모델의 시간창 확장")은 축소 우려 없이 깨끗하게 성립.

## 플랜 반영 사항

1. P3 비교군에 Li et al. Q-ALNS의 "방법 이식" 베이스라인 추가 (플랜 P3-4 반영).
   환경 직접 재현은 불성립: 그들의 문제는 storage/AGV 2단 구조 + 도어 모드
   결정이라 우리의 컴파운드 트럭 구조와 다르고, 인스턴스도 비공개(린이 실측
   데이터 스케일링). VAA 등 우리 핵심 부품은 그들의 문제에 정의되지 않음.
2. 벤치마크에 XXL(트럭 200대급) 셀 추가 검토 — 경쟁 논문 규모 대응.
3. P5 집필 직전 주장 2 재스캔 일정 추가.
