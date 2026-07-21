# 석사논문 연구 계획 (지도교수 피드백 반영, 2026-07-21)

권상진 교수님 피드백을 반영한 실행 계획. 핵심 지침 3가지를 방향으로 고정하고,
"새 아이디어 추가 대신 다듬기 + 실험 보강"으로 완성도 높은 석사논문을 목표.

## 0. 피드백 → 방향 고정

1. **기여 재배치.** 헤드라인은 RL이 아니라 **(a) 현실적 운영 조건으로의 문제
   확장(Time Windows, soft due date) + (b) 병목 유도 GILS 휴리스틱**. RL은
   성능 기여가 없으므로 **ablation / negative finding**으로 정리, 시간 투자 최소화.
2. **용어 규율.** `optimal`/`최적`은 CP-SAT로 최적성이 *증명된* small-size
   인스턴스에만. medium/large는 `high-quality` / `near-optimal`(CP-SAT 비교
   가능 셀 한정) / `best-known`.
3. **논문 스토리(교수 제안 4단):** 문제 확장 → 새로운 탐색 전략(GILS) →
   정확해(CP-SAT) 기반 검증 → RL selector 효과 분석(ablation).

## 1. 우선순위 작업

### Phase A — 서술/프레이밍 정리 (즉시, 저비용)

- **A1. 제목·기여 재작성.** 제목에서 RL을 내리고 문제 확장 + GILS를 전면에.
  기여 문장을 4단 스토리에 맞춰 재배열. (paper/apiems2026_draft.md,
  apiems2026_draft_ko.md, docs/apiems_paper_outline.md)
- **A2. 용어 전수 스윕.** `grep -rniE "optimal|최적화|최적해"` 로 원고 전체
  점검 → 증명 셀 외에는 high-quality/near-optimal/best-known으로 교체.
  특히 초록·결과·결론.
- **A3. 명명 정리.** 레포명 `CPG-RL-ALNS` 가 주 방법(VAA-GILS)·메시지와
  어긋남 → GILS 중심으로 정리하거나 README에 "주 방법=VAA-GILS, RL은 ablation"
  명시.

### Phase B — GILS를 핵심 기여로 "입증" (최우선 신규 실험)

현재 selector만 ablate하고 있어, "성능이 guided 구조에서 나온다"는 주장을
직접 뒷받침하는 실험이 없다. 이게 논문 논지의 척추다.

- **B1. Guided operator ablation (필수).** 연산자 풀을 바꿔가며 비교:
  - `generic-only`(NEIGHBORHOODS 7종) vs `+critical`(g1,g2) vs
    `+tardy`(full ACTIONS_TW). 동일 예산·동일 selector(uniform)로.
  - 기대: guided를 빼면 gap이 유의하게 악화 / selector를 바꿔도 불변 →
    "성능=구조, selector 아님" 대비가 성립.
  - 구현: `run_vaa_qrl`에 연산자 풀(action set)을 주입 가능하게 소폭 수정
    (지금은 내부에서 ACTIONS/ACTIONS_TW 자동 선택). selector에 넘기는
    action 튜플과 엔진 내부 풀을 일치시키는 파라미터 하나 추가.
- **B2. 컴포넌트 ablation.** VAA init vs random init(이미 init_sensitivity
  일부 존재), descent on/off, SA acceptance on/off, kick-restart on/off.
  각 컴포넌트의 gap 기여도 표 1장. → "attractor를 만드는 것이 무엇인지" 규명.
- **B3. (선택) 병목 유도 근거 시각화.** guided가 실제로 critical/tardy
  트럭을 얼마나 자주 개선하는지 카운트 → 방법 정당화 그림 1장.

### Phase C — 문제 확장의 비자명성 강화

- **C1. TW 난이도 논증.** compound 부분 하역·이송·carrier 구조에서 release/
  due가 목적함수와 왜 비자명하게 얽히는지 1문단(리뷰어 예상 질문 방어).
- **C2. 모델 명료화.** MILP(원 논문 방식) ↔ CP-SAT 재정식화(reified) 등가성
  명시, 결과 일치 확인. soft due 모델링·λ 선택 근거 서술.
- **C3. λ 민감도.** tardiness_weight(λ)를 바꿔가며 makespan-지연 trade-off
  곡선 1장. λ=1 선택의 정당화.

### Phase D — 실험 보강 (통계·범위)

- **D1. 셀당 인스턴스 확장 5→20.** 통계력 강화(백그라운드 실행). test pool은
  방법·하이퍼파라미터 동결 후에만 touch(현재 동결 상태 유지).
- **D2. TW 셀 공정 baseline.** Paper-SA의 TW 대응(가중치 반영, λ=0이면 기존
  동작 보존)으로 TW 셀에서도 공정 비교. (진행 중이던 항목)
- **D3. 하한/커버리지.** 대규모 TW 셀 하한이 느슨함(171–273%)은 정직하게
  한계로 서술하되, 가능하면 M/L에서 CP-SAT 시간을 늘려 near-optimal 비교
  가능 셀을 넓힘. best-known 기준 서술 원칙 유지.
- **D4. Selector ablation 마무리(저비용).** UCB/Thompson을 ablation 폭에만
  포함(구현 완료). **DQN 다중 시드 학습은 후순위/생략** — RL 고도화에 시간
  쓰지 말라는 지침에 따름.

### Phase E — 석사논문 패키징

- 목표 매체 확정(교수와 짧게 통화: 국내 학회지 vs SCIE). novelty bar가 여기서
  갈림. APIEMS는 이 결과의 중간 발표/피드백 창구로 활용 가능.
- 논문 구조를 4단 스토리로 정리 → 석사논문 목차로 확장.

## 2. 하지 말 것 (지침)

- RL(DQN 등) 성능 개선에 추가 투자 X. selector 비교는 negative finding으로 봉인.
- 새 문제축(동적 도착 등) 지금 추가 X. → **future work**로만 언급(동적은
  학습이 구조적으로 유리해지는 자연스러운 후속).

## 3. 다음 2주 액션

1. (A1–A2) 원고 리프레이밍 + 용어 스윕 — 1~2일.
2. (B1) 엔진에 operator-pool 주입 파라미터 추가 → guided ablation 실행 —
   3~4일. **이번 사이클의 핵심 산출물.**
3. (B2) 컴포넌트 ablation 실행 — 2일.
4. (D1) 20-인스턴스 그리드 백그라운드 실행 시작.
5. 교수님께 회신: 4단 스토리로 재구성하겠다 + guided ablation을 핵심 증거로
   추가하겠다 + 목표 매체 문의.

## 4. 성공 기준

- guided ablation에서 "guided 제거 시 유의 악화, selector 변경 시 불변"이
  나오면 논문 논지 확정.
- 용어·프레이밍이 교수 지침과 일치.
- 20-인스턴스로 Table 1/2의 통계적 결론 유지.
