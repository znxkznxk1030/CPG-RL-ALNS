# Cross-Dock CPG-ALNS MVP

This repository contains the MVP implementation for the compound-truck cross-docking scheduling study.

Implemented scope:

- MVP instance and solution dataclasses
- Feasibility checker
- Schedule evaluator with partial unloading
- Random feasible initialization
- VAA/VVA-style constructive baseline
- Paper-style Q-learning simulated annealing baseline
- Random baseline
- Critical-door destroy
- Greedy and regret-k repair
- SA acceptance
- Simple critical-door ALNS loop
- Baseline comparison experiment

`VVA` in the request is treated as the common typo for `VAA` (`Vogel's Approximation Algorithm`). The implementation also provides a `vva_solution()` alias.

## Run Tests

```bash
python -m pytest -q
```

Latest result:

```text
12 passed in 0.08s
```

## Run MVP Example

```bash
python examples/run_mvp.py
```

Example output:

```text
initial makespan: 2204.27
best makespan:    1456.27
critical door:    1
critical truck:   O2
iterations:       100
```

## Baselines

The baseline comparison currently includes:

| Method | Description |
|---|---|
| `Random-1` | One random feasible solution |
| `Random-30` | Best solution among 30 random feasible samples |
| `VAA` | Regret-based VAA-style constructive heuristic |
| `Paper-SA-RL5-300` | Paper-style VAA-initialized simulated annealing with Q-learning neighborhood selection |
| `CPG-ALNS-300` | Critical-door ALNS initialized from VAA, 300 iterations, regret-2 repair |

The MVP VAA heuristic works in three stages:

1. Assign compound trucks to retained destinations using row/column regret over compound-destination costs.
2. Assign compound doors by workload and door centrality.
3. Assign remaining destinations to outbound trucks and greedily insert outbound trucks into door sequences.

The paper-style SA-RL baseline follows the main model proposed by Shahmardan and Sajadieh:

1. Start from the VAA heuristic solution.
2. Generate moves with the paper's generic and tailor-made neighborhood structures.
3. Use SA acceptance.
4. Use Q-learning to select neighborhood structures.
5. Use reward `1` when the generated solution is no worse than the current solution, otherwise `0`.
6. Use no-improvement count bins as the Q-learning state, with the SA-RL5 thresholds `(5, 10, 15, 20)`.

## Baseline Experiment

Run:

```bash
python experiments/compare_baselines.py
```

The script writes:

- `outputs/baseline_results.csv`
- `outputs/baseline_summary.md`

Experiment settings:

- Instance classes: `Tiny`, `Small`, `Medium-lite`
- Seeds per class: 3
- Random baseline: one-shot and best-of-30
- Paper model baseline: VAA initialization + Q-learning SA, 300 iterations
- Proposed MVP: critical-door ALNS with VAA initialization, 300 iterations, regret-2 repair
- Gap: percentage gap against the best method observed on the same generated instance

Latest results:

| Instance | Method | N | Avg makespan | Avg gap % | Avg runtime sec | Wins |
|---|---:|---:|---:|---:|---:|---:|
| Tiny | Random-1 | 3 | 436.43 | 16.67 | 0.0001 | 0 |
| Tiny | Random-30 | 3 | 420.32 | 8.13 | 0.0022 | 0 |
| Tiny | VAA | 3 | 408.82 | 5.10 | 0.0015 | 1 |
| Tiny | Paper-SA-RL5-300 | 3 | 379.98 | 0.00 | 0.0525 | 3 |
| Tiny | CPG-ALNS-300 | 3 | 383.49 | 0.77 | 0.4659 | 1 |
| Small | Random-1 | 3 | 1624.05 | 37.96 | 0.0001 | 0 |
| Small | Random-30 | 3 | 1370.93 | 14.60 | 0.0033 | 0 |
| Small | VAA | 3 | 1213.98 | 1.38 | 0.0042 | 0 |
| Small | Paper-SA-RL5-300 | 3 | 1197.47 | 0.00 | 0.0378 | 3 |
| Small | CPG-ALNS-300 | 3 | 1213.98 | 1.38 | 0.8848 | 0 |
| Medium-lite | Random-1 | 3 | 2169.29 | 35.47 | 0.0002 | 0 |
| Medium-lite | Random-30 | 3 | 1810.84 | 11.69 | 0.0049 | 0 |
| Medium-lite | VAA | 3 | 1676.07 | 3.84 | 0.0097 | 1 |
| Medium-lite | Paper-SA-RL5-300 | 3 | 1616.47 | 0.00 | 0.0714 | 3 |
| Medium-lite | CPG-ALNS-300 | 3 | 1676.07 | 3.84 | 1.4608 | 1 |

## Interpretation

VAA is much stronger than random construction in this MVP benchmark. `Random-30` improves over `Random-1`, but it remains clearly worse than VAA on average.

The paper-style `Paper-SA-RL5-300` is currently the strongest method in this smoke benchmark. Increasing the proposed MVP from `CPG-ALNS-60` to `CPG-ALNS-300` did not materially improve solution quality, while runtime increased. This is a useful diagnostic result: the bottleneck is operator diversity and repair strength, not simply iteration count. The paper model already has diverse neighborhood structures and Q-learning operator selection, while the MVP CPG-ALNS still only has critical-door destroy and regret repair. The next useful improvements for the proposed method are high-transfer destroy, sequence-bottleneck destroy, local-search polish, exact repair, and contextual RL operator selection.
