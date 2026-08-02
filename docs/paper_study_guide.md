# 논문 학습 플랜 & 필독 레퍼런스 (2026-07-30)

본 논문(`paper/apiems2026_draft.md`)은 **6개 분야**의 교차점에 있다: 크로스도킹
스케줄링(도메인) · 지역탐색 메타휴리스틱 · 적응형/학습형 연산자 선택 · 제약
프로그래밍/MILP · 조합최적화를 위한 기계학습 · 실험 통계. "논문을 100% 이해"
하려면 각 절이 어느 지식을 전제하는지 알고, 그 지식의 원천을 읽어야 한다.

★ = 반드시 읽을 것. ☆ = 여유 있으면.

---

## 0. 논문 → 필요 지식 매핑

| 논문 절 | 요구 지식 | 우선 레퍼런스 |
|---|---|---|
| §1–2 서론·관련연구 | 크로스도킹·컴파운드 트럭·시간창 스케줄링 | [5]★ [4] [1][2][3] |
| §3 문제·수리모형 | MILP(big-M, disjunctive), CP-SAT, 하한, release/due 스케줄링 | [5]★ [11] Pinedo, CP Handbook |
| §4 VAA-GILS | ILS, descent, SA, VAA, 유도 지역탐색 | [14]★ [13] Hoos&Stützle★ [12] |
| §4.1·§6.2·§7 선택 정책·ablation | 적응형 연산자 선택, 하이퍼휴리스틱, Q/DQN | Burke2013★ Fialho2010★ [9][10] |
| §2·§6·§7 학습형 메타휴리스틱 비판 | L2O의 방법론적 함정 | Bengio2021★ [6][7][8] |
| §5–6 실험·통계 | 시드 프로토콜, Wilcoxon, 짝비교 | Derrac2011★ |

---

## 1. 학습 플랜 (5단계, 권장 4~6주)

### Phase 0 — 자기 논문 + 베이스 논문 (1주, 최우선)
> "내 논문의 모든 문장을 원 논문 대조로 재구성한다."

1. **본 논문을 §3까지 손으로 재유도.** 타이밍 규칙(컴파운드 하역 완료
   `r_i + DE_i + Σh`, 목적지 준비, 적재 시작 max 조건)을 작은 예제(S 규모)로
   직접 계산. `crossdock_solver/core/`의 evaluator와 대조.
2. **[5] Shahmardan & Sajadieh 2020을 처음부터 끝까지 정독.** ★ 이게 논문의
   토대다. 부분하역 의미, MILP 정식화, VAA 구성, SA-RL(밴딧/Q) 선택을 원문
   수준으로 이해. 본 논문이 "무엇을 확장(시간창)하고 무엇을 재평가(RL 무용)"
   하는지가 여기서 갈린다.
3. **자가 점검**(아래 §3) S-none 문항을 통과할 것.

### Phase 1 — 지역탐색·메타휴리스틱 골격 (1주)
> §4가 왜 그렇게 설계됐는지 이해.

- **[14] Lourenço, Martin, Stützle "Iterated Local Search"** ★ (Handbook of
  Metaheuristics, 3rd ed.). ILS의 4요소(초기해·지역탐색·perturbation·수용)를
  본 논문의 (VAA·descent·kick-restart·SA수용)에 1:1 매핑.
- **Hoos & Stützle, *Stochastic Local Search*** ★ (2004). 1~2장 + best/first
  improvement, plateau, restart. "왜 best-improvement descent가 최대 기여자
  (B2)인지"의 언어를 여기서 얻는다.
- **[13] Kirkpatrick 1983 (SA)** + **[12] Korukoğlu&Ballı 2011 (VAM)**: 수용
  기준과 Vogel 후회의 원리.

### Phase 2 — 논문의 척추: 적응형/학습형 연산자 선택 (1~1.5주)
> §4.1·§6.2·§7의 negative finding을 "제대로" 이해·방어.

- **Burke et al. 2013 "Hyper-heuristics: a survey of the state of the art"** ★
  (JORS). "연산자를 고르는 상위 수준 탐색"이라는 프레임. 본 논문이 ablate하는
  대상의 학계 지형.
- **Fialho et al. 2010 "Adaptive Operator Selection"(밴딧 기반)** ★. AOS가
  이론적으로 언제 이득인지 → §7의 4조건과 대조. 본 논문의 반례가 이 문헌의
  어떤 가정을 깨는지 파악.
- **[9] Watkins&Dayan 1992 (Q-learning)**, **[10] Mnih 2015 (DQN)**: SA-RL5의
  tabular Q와 본 논문 transfer DQN의 원리. 상태·보상·ε-greedy·오프라인 학습.
- **Bengio, Lodi, Prouvost 2021 "ML for CO: a methodological tour d'horizon"** ★
  (EJOR). 학습형 조합최적화의 함정(약한 baseline, 불공정 예산)을 정면으로 다룸
  → §7 방법론적 기여의 학술적 근거. **본 논문의 메시지를 가장 잘 뒷받침하는
  외부 문헌.**

### Phase 3 — 정확해·수리모형 (0.5~1주)
> §3.3–3.5의 CP-SAT·하한을 이해.

- **[11] OR-Tools CP-SAT** ★ (공식 문서 + Laurent Perron 튜토리얼 영상).
  reified 제약, interval/NoOverlap, 하한·인컴번트 반환 개념.
- **Rossi, van Beek, Walsh, *Handbook of Constraint Programming*** ☆ (해당 장만).
  big-M disjunction ↔ reified 등가성의 배경.
- **Pinedo, *Scheduling*** ☆: release date `r_j`, due date, tardiness `T_j`,
  `Σw_jT_j` 목적의 표준 이론 — 본 논문 목적함수의 스케줄링적 뿌리.

### Phase 4 — 도메인 (0.5주, 병행 가능)
- **[1] Boysen&Fliedner 2010** ★, **[2] Van Belle 2012**, **[3] Ladier&Alpan
  2016** (Omega 3부작): 크로스도킹 스케줄링 분류·서베이·산업 간극. §2를 원전
  수준으로.
- **[4] Joo&Kim 2013**: 컴파운드 트럭의 최초 정의.
- **[6] Li et al. 2025 (Q-ALNS)** ★: 최근접 경쟁 문제. 유연성의 위치(도어 vs
  트럭) 차이를 정확히 이해 → T2-1 경쟁 baseline 이식의 근거.

### Phase 5 — 실험 방법론·통계 (0.5주)
- **Derrac et al. 2011 "A practical tutorial on nonparametric tests..."** ★
  (Swarm & Evol. Comput.). Wilcoxon signed-rank, 짝비교, 다중비교 보정 —
  §5·§6 통계의 교과서. 셀당 n, p값 해석, D1(5→20)의 통계력 논리를 여기서.
- ☆ Kerschke et al. 2019 "Automated Algorithm Selection: Survey": 선택 정책을
  더 넓은 알고리즘 선택 맥락에 배치.

---

## 2. 우선순위 압축 (시간이 없다면 이 6편만)

1. **[5] Shahmardan & Sajadieh 2020** — 토대. 없으면 논문 절반이 안 보임.
2. **[14] ILS (Lourenço/Martin/Stützle)** — §4 방법의 문법.
3. **Burke et al. 2013 하이퍼휴리스틱 서베이** — ablate 대상의 지형.
4. **Bengio/Lodi/Prouvost 2021** — §7 메시지의 학술적 정당성.
5. **[11] CP-SAT 문서** — §3 정확해 검증.
6. **Derrac et al. 2011** — §5–6 통계.

---

## 3. 자가 점검 질문 (100% 이해 체크리스트)

절별로 "막힘 없이 답할 수 있는가"로 이해도를 진단한다.

**§3 문제**
- 컴파운드 트럭이 유지 목적지를 바꾸면 makespan과 지연이 각각 어떻게 변하나?
- big-M MILP의 하한이 규모가 커질 때 왜 무력해지고, CP-SAT reified가 왜 이를
  개선하나?
- 조합 하한 두 개(임계 체인·도어 면적)가 각각 무엇을 하방 바운드하나?
  왜 그 최댓값이 유효 하한인가?

**§4 방법**
- ILS의 4요소를 VAA-GILS 컴포넌트에 매핑하라. B2에서 왜 descent가 최대
  기여이고 SA수용은 동력이 아닌가(직관)?
- 유도 연산자 g1~g4가 §3.6의 "두 병목(임계·지연)"과 어떻게 대응하나?
- g3/g4의 "표집 가중 효과" nuance(none 셀)를 설명하라.

**§6.2·§7 핵심 메시지**
- "성능은 selector가 아니라 유도 구조에서 온다"를 B1·B2·K1 수치로 3문장 논증.
- 학습 selector를 무효화한 4조건을 말하고, 각 조건을 부정하면 왜 학습이
  되살아나는지 예를 들라. (→ T3-1 실험 설계로 이어짐)
- transfer DQN이 tabular보다도 나쁜 이유의 가설은?

**§C3(λ)**
- λ=0에서 tardiness_norm이 0.99인데 makespan_norm이 0(최소)이 아닌 0.126인
  이유는? (탐색 노이즈 vs 진짜 트레이드오프 구분)
- 가장 급한 knee는 λ≈0.25인데 왜 λ=1을 택했나?

**§5–6 통계**
- Wilcoxon signed-rank를 왜 t-검정 대신 쓰나? 짝(pair)은 무엇으로 맞추나?
- test 풀을 방법 동결 전에 만지면 안 되는 이유를 한 문장으로.

> 이 질문들에 자료 없이 답할 수 있으면 사실상 100% 이해다. 막히는 절이 곧
> 위 학습 플랜에서 먼저 볼 Phase다.

---

## 4. 실물 확보 팁
- 서베이/핸드북 장(Burke2013, Bengio2021, Derrac2011, ILS 챕터)은 Google
  Scholar에서 저자 PDF 다수 공개.
- [5]·[6]은 학교 도서관 프록시(C&IE/EJOR, Elsevier)로.
- CP-SAT는 유료 논문이 아니라 공식 문서·YouTube 튜토리얼이 가장 빠르다.
