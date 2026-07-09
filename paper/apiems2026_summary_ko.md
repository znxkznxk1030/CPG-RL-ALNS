# VAA-GILS 실험 결과 요약

## 1. 연구 개요

본 연구는 컴파운드 트럭 부분 하역 크로스도킹 문제에 트럭별 도착 시각과
소프트 마감을 추가한 Time Window 확장 문제를 다룬다. 기존 모델은 모든 트럭이
시간 0에 가용하다고 가정하고 makespan만 최소화하지만, 실제 출고 운영에서는
트럭 도착 시각과 마감 지연을 함께 고려해야 한다.

제안 방법은 **VAA-GILS**이다. VAA 구성 휴리스틱으로 초기해를 만든 뒤, 현재
스케줄의 병목을 찾아 집중적으로 개선하는 guided iterated local search 방식이다.

## 2. 핵심 기여

- **문제 확장:** 컴파운드 트럭 부분 하역 모델에 도착 시각과 소프트 마감을 반영했다.
- **방법 제안:** 병목 기반 guided operator, descent, restart를 결합한 VAA-GILS를 제안했다.
- **실험 분석:** uniform, tabular Q-learning, DQN 선택기를 비교하여 학습 기반 연산자
  선택의 추가 효과를 검증했다.

## 3. 실험 질문

1. VAA-GILS가 기존 VAA 및 Paper-SA-RL5보다 좋은가?
2. CP-SAT가 해를 주는 작은 셀에서 VAA-GILS가 정확해 기준에 얼마나 가까운가?
3. 연산자 선택을 학습시키는 것이 uniform 선택보다 실질적으로 나은가?

## 4. 실험 설정

- 규모: S, M, L
- Time Window: none, medium, tight
- 인스턴스: 각 셀당 test 인스턴스 5개
- 반복: stochastic method는 인스턴스당 5회
- 비교 방법: VAA, Paper-SA-RL5, GILS-uniform, GILS-tabular, GILS-DQN
- 정확해 기준: S는 CP-SAT 300초, M/L 일부는 CP-SAT 600초

CP-SAT가 해를 낸 셀에서는 CP-SAT 대비 gap을 함께 보고한다. CP-SAT가 600초 안에
가능해를 못 낸 큰 Time Window 셀에서는 best-known 해 대비 gap을 본다.

## 5. 전체 실험 테이블

`Δbk`는 인스턴스별 best-known 해 대비 평균 gap이다.

| 셀 | 방법 | Mean obj ± std | Δbk (%) | CP-SAT 대비 (%) |
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

## 6. 결과 해석

- **S-none:** VAA의 gap은 6.49%, Paper-SA-RL5는 1.21%인 반면, GILS 계열은
  0.21~0.75% 수준이다. 기본 문제에서도 GILS의 개선 효과가 뚜렷하다.
- **S-medium/S-tight:** Time Window가 있는 소규모 셀에서도 GILS는 CP-SAT 대비
  0.11~0.55% 수준으로 근접한다.
- **M-none:** GILS는 CP-SAT incumbent 대비 0.11~0.19% 수준이다. 일부 실행에서는
  CP-SAT incumbent보다 더 좋은 해를 찾는다.
- **M/L Time Window 셀:** CP-SAT가 제한 시간 안에 가능해를 못 낸 경우가 있어
  CP-SAT 대비 값은 없다. 이 경우 GILS 결과가 현재 best-known 기준이다.
- **선택기 비교:** GILS-uniform, GILS-tabular, GILS-DQN의 차이는 작다. 따라서
  성능의 핵심은 학습 선택기보다는 GILS의 병목 기반 탐색 구조로 해석된다.

## 7. 요약

VAA-GILS는 정확해와 비교 가능한 셀에서 CP-SAT incumbent에 매우 근접하고,
기존 VAA 및 Paper-SA-RL5보다 일관되게 낮은 objective를 보인다. 특히 큰
Time Window 셀에서는 CP-SAT가 제한 시간 내 가능해를 못 내는 경우에도 안정적으로
best-known 해를 제공한다.

또한 학습 기반 selector가 uniform selector를 안정적으로 이기지 못했다. 따라서
본 연구의 핵심 주장은 “RL selector가 성능을 만든다”가 아니라, **문제 구조를
활용한 guided ILS 엔진이 성능을 만든다**는 방향이 더 적절하다.

## 8. 논의 포인트

- 방법 이름은 VAA-GILS로 정리하는 것이 결과와 잘 맞는다.
- near-optimal 주장은 CP-SAT 기준이 있는 셀로 제한하는 것이 보수적이다.
- 큰 Time Window 셀에서는 CP-SAT 대비가 아니라 best-known 기준으로 해석해야 한다.
- DQN 결과는 주 방법이 아니라 selector ablation 또는 negative finding으로 배치하는 것이 자연스럽다.
