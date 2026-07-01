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
13 passed in 0.16s
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
| `VAA` | Paper-style VAA constructive heuristic: Eq. (23), FAT, FT_m insertion |
| `Paper-SA-RL5-300` | Paper-style VAA-initialized simulated annealing with Q-learning neighborhood selection |
| `CPG-ALNS-300` | Critical-door ALNS initialized from VAA, 300 iterations, regret-2 repair |

The VAA heuristic now follows the paper's construction as closely as the MVP representation allows:

1. Calculate `T_id` using paper Eq. (23): partial unloading time plus loading time from other compounds.
2. Use VAA regret to assign compound trucks to retained destinations.
3. Assign remaining destinations to outbound trucks by `TEE_f` and destination priority `T_d`.
4. Sort doors by `T_m`, the total distance from each door to all other doors.
5. Build `FAT` from all compound trucks plus longest-loading outbound trucks when doors remain.
6. Assign FAT trucks to central doors and update door finish times `FT_m`.
7. Insert remaining outbound trucks by highest `T_d` into the door with lowest current `FT_m`.

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
| Tiny | Random-1 | 3 | 436.43 | 17.00 | 0.0003 | 0 |
| Tiny | Random-30 | 3 | 420.32 | 8.37 | 0.0022 | 0 |
| Tiny | VAA | 3 | 430.65 | 13.03 | 0.0006 | 0 |
| Tiny | Paper-SA-RL5-300 | 3 | 379.49 | 0.00 | 0.0418 | 3 |
| Tiny | CPG-ALNS-300 | 3 | 411.62 | 6.41 | 0.5402 | 0 |
| Small | Random-1 | 3 | 1624.05 | 38.08 | 0.0002 | 0 |
| Small | Random-30 | 3 | 1370.93 | 14.75 | 0.0036 | 0 |
| Small | VAA | 3 | 1275.57 | 6.62 | 0.0015 | 0 |
| Small | Paper-SA-RL5-300 | 3 | 1195.21 | 0.00 | 0.0571 | 3 |
| Small | CPG-ALNS-300 | 3 | 1275.57 | 6.62 | 1.5416 | 0 |
| Medium-lite | Random-1 | 3 | 2169.29 | 33.24 | 0.0003 | 0 |
| Medium-lite | Random-30 | 3 | 1810.84 | 9.84 | 0.0082 | 0 |
| Medium-lite | VAA | 3 | 1751.43 | 6.78 | 0.0045 | 0 |
| Medium-lite | Paper-SA-RL5-300 | 3 | 1643.75 | 0.00 | 0.1246 | 3 |
| Medium-lite | CPG-ALNS-300 | 3 | 1724.76 | 4.95 | 1.8588 | 0 |

## Interpretation

The paper-style VAA is now intentionally more faithful to the paper's construction, so it is less aggressive than the previous greedy-evaluator variant. It is a fast initial solution generator rather than a strong local optimizer.

The paper-style `Paper-SA-RL5-300` is currently the strongest method in this smoke benchmark. The proposed MVP `CPG-ALNS-300` improves over VAA on Medium-lite but is still weaker than the paper RL-SA model. This remains a useful diagnostic result: the bottleneck is operator diversity and repair strength. The paper model already has diverse neighborhood structures and Q-learning operator selection, while the MVP CPG-ALNS still only has critical-door destroy and regret repair. The next useful improvements for the proposed method are high-transfer destroy, sequence-bottleneck destroy, local-search polish, exact repair, and contextual RL operator selection.
