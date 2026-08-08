# Phase D1 결과: 셀당 인스턴스 5→20 확장 (2026-07-30)

**목적:** Table 2(주 해품질 비교)의 통계력 강화. 방법·하이퍼파라미터 동결
상태에서 test 풀 인스턴스를 5→20으로 확장.

- 실행: `experiments/k1_run.py search` (SEARCH_INDICES=range(20)로 확장;
  budget·cpsat 배치는 5 유지). append-only 러너가 기존 0–4는 skip, 5–19만
  실행(2385 job 추가). 시작 전 `outputs/k1_results.pre_d1.jsonl` 백업.
- 재집계: `experiments/k1_stats.py`, `k1_summary.py`.

## 결론: 주장 전부 유지·강화

### Wilcoxon (20-inst)
| 비교 | 5-inst | 20-inst | 판정 |
|---|---|---|---|
| GILS vs VAA | +2.39 (n=45) | **+2.64 (n=180)** | 유의 유지·강화 |
| GILS vs Paper-SA-RL5 | +1.16 (n=75) | **+1.33 (n=297)** | 유의 유지·강화 |
| GILS vs DQN | +0.143 (n=203) | +0.096 (n=789) | DQN 열세 유지 |
| uniform vs DQN | +0.144 (n=185) | +0.106 (n=693) | DQN 열세 유지 |
| tabular vs uniform | 0.00, p=0.28 | −0.01, p=0.038 | 학습 무익(오히려 근소 열세) |

- **주 서열 GILS < Paper-SA-RL5 < VAA: 9셀 전부 유지.** VAA·Paper-SA 대비
  효과크기가 오히려 커짐.
- **RL negative finding 견고**: DQN 전 예산 열세, tabular는 20-inst에서
  uniform보다 근소 열세(−0.01%p)로 "학습 무익" 메시지가 오히려 강화.

## 범위(정직성)

- D1은 **주 결과(표 6.1)와 그 방법-비교 Wilcoxon만** 20-inst로 확장.
- 통제된 ablation(표 6.2~6.4=B1/B2/selector, 그림 6.3 budget)은 **원 5-inst
  설계 유지** — 동일예산·동일구조 분해이며 일반화 주장이 아님. budget 배치의
  50/200/3000-iter는 5-inst만 있어 혼합 방지.
- **vs CP-SAT 열**: CP-SAT는 idx 0–4에만 있어 불변(정확해 정박 부분집합).
  표에서 평균목적값·Δbk(20-inst)와 vs CP-SAT(5-inst 부분집합)의 기준 차이를
  캡션·본문에 명시.

## 논문 반영

- 영문: Table 2 전면 교체(20-inst obj·Δbk), §6.1 관찰 (i)(ii) 갱신,
  §5 Protocol에 20/5 범위 명시.
- 국문: 표 6.1·관찰·§5.2 동일 반영.
- 하네스: `k1_run.py`에 `SEARCH_INDICES`, docstring 갱신.
