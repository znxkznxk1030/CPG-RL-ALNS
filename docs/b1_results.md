# Phase B1 결과: Guided operator ablation (2026-07-21)

**질문:** 성능이 학습된 selector에서 나오는가, 아니면 병목 유도 연산자
구조에서 나오는가? 이를 직접 가르기 위해, **선택 정책을 uniform으로 고정**하고
**연산자 풀만** generic(7) / critical(+g1,g2=9) / full(+g3,g4=11)로 바꿔 비교.

- 실행: `experiments/b1_run.py` → `outputs/b1_guided_ablation.jsonl`
  (test pool, 3 규모 × 3 TW × 5 인스턴스 × 5 반복 × 3 풀 = 675 run)
- 요약: `experiments/b1_summary.py` → `outputs/b1_summary.txt`
- 재현성 확인: `full` arm은 기존 `GILS-uniform-1000`과 소수점까지 동일.

## 셀별 best-known 대비 평균 gap (%)

| cell | generic | critical | full |
|---|---:|---:|---:|
| S-none | 0.45 | 0.17 | 0.10 |
| S-medium | 0.41 | 0.17 | 0.08 |
| S-tight | 0.35 | 0.17 | 0.16 |
| M-none | 0.39 | 0.28 | 0.24 |
| M-medium | 0.45 | 0.37 | 0.28 |
| M-tight | 0.25 | 0.19 | 0.11 |
| L-none | 0.22 | 0.17 | 0.12 |
| L-medium | 0.09 | 0.07 | 0.06 |
| L-tight | 0.01 | 0.01 | 0.00 |

**모든 셀에서 generic ≥ critical ≥ full** (단조). 유도 연산자를 더할수록
목적함수가 낮아진다.

## Paired Wilcoxon (양측, 상대 목적함수 차)

| 전환 | 범위 | n | 개선 mean% | p | 판정 |
|---|---|---:|---:|---:|---|
| generic → critical (g1,g2 추가) | 전체 | 136 | +0.114 | <10⁻⁴ | 유의 |
| critical → full (g3,g4 추가) | 전체 | 172 | +0.048 | <10⁻⁴ | 유의 |
| critical → full | TW 셀만 | 116 | +0.045 | 0.0004 | 유의 |
| critical → full | none 셀만 | 56 | +0.053 | 0.0014 | 유의 |
| generic → full (전체 유도) | 전체 | 166 | +0.161 | <10⁻⁴ | 유의 |

(+ 값 = 더 풍부한 풀이 더 낮은 목적함수)

## 해석 (논문 논지)

1. **유도 연산자는 일관되게·유의하게 성능을 높인다.** 9개 셀 전부에서 단조
   개선이고 모든 Wilcoxon이 p<0.002로 유의. 방향이 절대 뒤집히지 않는다.
2. **selector 학습과의 대비가 핵심.** K1의 selector ablation에서는 효과가
   ≤0.17%p이고 **예산에 따라 방향이 뒤집혔으며 학습 정책이 이기는 구간이
   없었다.** 반면 유도 연산자는 단조·항상 유의·무방향전환 →
   **"성능은 selector가 아니라 유도 구조에서 나온다"**는 주장을 직접 뒷받침.
3. **정직한 분해.** 성능의 큰 덩어리는 그 앞 단계에서 온다:
   S-none 기준 VAA 6.49% → generic 풀 GILS 0.45%(descent+SA+restart 엔진) →
   full 0.10%(유도 연산자 정련). 즉 엔진 골격이 대부분을 만들고, 유도
   연산자가 그 위에서 유의하게 더 조인다. 엔진 골격 각 요소의 기여는
   Phase B2(컴포넌트 ablation)에서 분해.

## 정직하게 밝힐 nuance

- `none` 셀에서도 full>critical(+0.053%)이 유의한데, g3/g4는 지각 트럭이
  없으면 g1/g2로 fallback한다. 따라서 이 개선의 일부는 **새 연산자 능력이
  아니라, 유도 이동이 uniform 분포에서 더 큰 선택 확률(4/11 vs 2/9)을 받는
  표집 가중 효과**다. 새 *능력*을 보이는 깨끗한 결과는 (a) generic→critical
  전체, (b) full→critical의 **TW 셀** 부분이다.
- 후속 정련(선택): 유도 이동 선택 확률을 고정한 채 g3/g4 유무만 비교하면
  이 nuance를 제거할 수 있다(B2 후보).

## 논문 반영

- 새 표: "Operator-pool ablation" (위 gap 표 + Wilcoxon). selector ablation
  표(Table 2) 바로 옆에 배치해 "구조 vs 학습" 대비를 시각적으로 보여줌.
- 문장: "유도 연산자 풀은 목적함수를 일관되게 유의하게 낮추는 반면(단조,
  p<0.002, 무방향전환), 학습된 선택 정책은 그렇지 못했다(≤0.17%p, 예산에 따라
  방향 전환, 학습 정책 무승)."
