# Cross-Dock CPG-ALNS MVP

This repository contains the MVP implementation for the compound-truck cross-docking scheduling study.

Implemented scope:

- MVP instance and solution dataclasses
- Feasibility checker
- Schedule evaluator with partial unloading
- Random feasible initialization
- VAA/VVA-style constructive baseline
- Paper-style Q-learning simulated annealing baseline
- Destination-agent RL constructive baseline
- Cargo-matrix destination-agent RL baseline
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
17 passed in 0.16s
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
| `DestAgent-RL-150` | Destination-agent RL: each destination agent learns a carrier-truck choice, then a release-time greedy scheduler assigns doors |
| `CargoMatrix-RL-150` | VAA-ordered destination-agent RL with a 9 compound x 3 destination cargo-count matrix in the state |
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

The destination-agent RL baseline is a separate experimental model inspired by the attached `lcl_gym/rl` code:

1. Treat each destination as one agent.
2. Share one small NumPy MLP across all destination agents.
3. Each destination agent chooses one still-available carrier truck.
4. Use epsilon-greedy exploration and replay-buffer updates.
5. Give all destination agents the same final reward based on improvement over the VAA makespan.
6. Complete the learned carrier assignment with a deterministic door-release greedy scheduler.

The cargo-matrix RL baseline is a more explicit state variant:

1. Start from the VAA solution and use its destination order.
2. Each state includes a fixed cargo matrix: up to 9 compound trucks x current 3-destination window.
3. Matrix values are the cargo counts each compound truck carries for each window destination.
4. Add masks for available compound slots and active destination-window slots.
5. Each destination agent still chooses one available carrier truck.
6. The final carrier assignment is completed with the same release-time greedy scheduler.

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
- Destination-agent RL baseline: 150 training episodes per generated instance
- Cargo-matrix RL baseline: 150 training episodes per generated instance
- Proposed MVP: critical-door ALNS with VAA initialization, 300 iterations, regret-2 repair
- Gap: percentage gap against the best method observed on the same generated instance

Latest results:

| Instance | Method | N | Avg makespan | Avg gap % | Avg runtime sec | Wins |
|---|---:|---:|---:|---:|---:|---:|
| Tiny | Random-1 | 3 | 436.43 | 17.00 | 0.0001 | 0 |
| Tiny | Random-30 | 3 | 420.32 | 8.37 | 0.0033 | 0 |
| Tiny | VAA | 3 | 430.65 | 13.03 | 0.0009 | 0 |
| Tiny | Paper-SA-RL5-300 | 3 | 379.49 | 0.00 | 0.0341 | 3 |
| Tiny | DestAgent-RL-150 | 3 | 385.65 | 2.29 | 0.1214 | 1 |
| Tiny | CargoMatrix-RL-150 | 3 | 385.65 | 2.32 | 0.1133 | 1 |
| Tiny | CPG-ALNS-300 | 3 | 411.62 | 6.41 | 0.4568 | 0 |
| Small | Random-1 | 3 | 1624.05 | 38.08 | 0.0001 | 0 |
| Small | Random-30 | 3 | 1370.93 | 14.75 | 0.0032 | 0 |
| Small | VAA | 3 | 1275.57 | 6.62 | 0.0014 | 0 |
| Small | Paper-SA-RL5-300 | 3 | 1195.21 | 0.00 | 0.0555 | 3 |
| Small | DestAgent-RL-150 | 3 | 1239.93 | 3.52 | 0.2256 | 0 |
| Small | CargoMatrix-RL-150 | 3 | 1229.26 | 2.75 | 0.2533 | 0 |
| Small | CPG-ALNS-300 | 3 | 1275.57 | 6.62 | 1.3168 | 0 |
| Medium-lite | Random-1 | 3 | 2169.29 | 34.15 | 0.0002 | 0 |
| Medium-lite | Random-30 | 3 | 1810.84 | 10.68 | 0.0049 | 0 |
| Medium-lite | VAA | 3 | 1751.43 | 7.56 | 0.0031 | 0 |
| Medium-lite | Paper-SA-RL5-300 | 3 | 1643.75 | 0.75 | 0.1037 | 1 |
| Medium-lite | DestAgent-RL-150 | 3 | 1634.55 | 0.27 | 0.4398 | 2 |
| Medium-lite | CargoMatrix-RL-150 | 3 | 1656.78 | 1.69 | 0.4562 | 0 |
| Medium-lite | CPG-ALNS-300 | 3 | 1724.76 | 5.74 | 1.6612 | 0 |

## Interpretation

The paper-style VAA is now intentionally more faithful to the paper's construction, so it is less aggressive than the previous greedy-evaluator variant. It is a fast initial solution generator rather than a strong local optimizer.

The paper-style `Paper-SA-RL5-300` is still strongest on Tiny and Small. `CargoMatrix-RL-150` improves over `DestAgent-RL-150` on Small, which supports the idea that exposing compound-truck cargo contents helps. It is weaker on Medium-lite, where `DestAgent-RL-150` still has the best average makespan and 2 of 3 wins. This suggests that the cargo matrix is useful but not sufficient by itself; the model still needs stronger credit assignment and door-aware actions to be reliable.

The proposed MVP `CPG-ALNS-300` is still weaker than both RL baselines in this smoke benchmark. The next useful improvements are high-transfer destroy, sequence-bottleneck destroy, local-search polish, exact repair, and contextual RL operator selection.
