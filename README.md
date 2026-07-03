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
- Variable-size graph cargo-matrix RL baseline
- VAA-QRL model: VAA initialization + Q-learning guided iterated local search
- Random baseline
- Critical-door destroy
- Greedy and regret-k repair
- SA acceptance
- Simple critical-door ALNS loop
- Baseline comparison experiment
- Research infrastructure (P0, see `docs/scie_research_plan.md`): fast search
  evaluator, S/M/L/XL benchmark generator with flow patterns, train/tuning/test
  seed protocol, resumable parallel experiment runner

`VVA` in the request is treated as the common typo for `VAA` (`Vogel's Approximation Algorithm`). The implementation also provides a `vva_solution()` alias.

## Run Tests

```bash
python -m pytest -q
```

Latest result:

```text
44 passed, 8 warnings in 1.32s
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
| `GraphCargoMatrix-RL-150` | Variable-size graph RL using truck, destination, and door nodes plus cargo and travel edges |
| `VAA-QRL-300` | Best model: VAA initialization + Q-learning guided iterated local search with critical-door operators, descent polish, and kick-restarts |

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

The graph cargo-matrix variant keeps `CargoMatrix-RL-150` available for ablation and replaces only the state representation:

1. Build variable-size graph states with truck, destination, and dock-door nodes.
2. Add cargo edges from compound trucks to remaining destinations.
3. Add door-travel edges between dock doors.
4. Add projected door-release, door-workload, door-utilization, and critical-door features from the current partial assignment.
5. Pool node and edge features with mean, max, min, and standard deviation.
6. Feed the fixed pooled embedding to the same shared NumPy MLP policy.

The VAA-QRL model keeps the paper's "VAA initial solution + Q-learning operator selection" frame and strengthens the search:

1. Start from the VAA heuristic solution and polish it with best-improvement descent.
2. Q-learning over no-improvement states selects a move operator each iteration.
3. The operator pool is the paper's seven neighborhoods plus two critical-door guided operators: best relocation of an outbound truck away from the critical door, and best destination swap for the critical truck.
4. Reward is shaped: `2` for a new global best, `1` for a solution no worse than the current one, `0` otherwise.
5. SA acceptance with geometric cooling; after 30 iterations without a new best, restart from the best solution with a 3-move random kick and reheating.
6. Every new best solution (and the final best) is polished by a deterministic descent over outbound relocations, compound door swaps/moves, and destination swaps.

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
- Graph cargo-matrix RL baseline: 150 training episodes per generated instance
- VAA-QRL model: VAA initialization + Q-learning guided iterated local search, 300 iterations
- Gap: percentage gap against the best method observed on the same generated instance

Latest results:

| Instance | Method | N | Avg makespan | Avg gap % | Avg runtime sec | Wins |
|---|---:|---:|---:|---:|---:|---:|
| Tiny | Random-1 | 3 | 436.43 | 17.80 | 0.0002 | 0 |
| Tiny | Random-30 | 3 | 420.32 | 9.06 | 0.0023 | 0 |
| Tiny | VAA | 3 | 430.65 | 13.78 | 0.0002 | 0 |
| Tiny | Paper-SA-RL5-300 | 3 | 379.49 | 0.67 | 0.0275 | 1 |
| Tiny | DestAgent-RL-150 | 3 | 385.65 | 2.98 | 0.0916 | 1 |
| Tiny | CargoMatrix-RL-150 | 3 | 385.65 | 3.02 | 0.0797 | 1 |
| Tiny | GraphCargoMatrix-RL-150 | 3 | 388.99 | 3.70 | 0.8653 | 0 |
| Tiny | VAA-QRL-300 | 3 | 377.65 | 0.00 | 0.0141 | 3 |
| Small | Random-1 | 3 | 1624.05 | 38.89 | 0.0001 | 0 |
| Small | Random-30 | 3 | 1370.93 | 15.36 | 0.0020 | 0 |
| Small | VAA | 3 | 1275.57 | 7.18 | 0.0003 | 0 |
| Small | Paper-SA-RL5-300 | 3 | 1195.21 | 0.53 | 0.0324 | 1 |
| Small | DestAgent-RL-150 | 3 | 1239.93 | 4.05 | 0.1375 | 0 |
| Small | CargoMatrix-RL-150 | 3 | 1229.26 | 3.29 | 0.1110 | 0 |
| Small | GraphCargoMatrix-RL-150 | 3 | 1236.44 | 3.73 | 1.0288 | 0 |
| Small | VAA-QRL-300 | 3 | 1190.34 | 0.05 | 0.0257 | 2 |
| Medium-lite | Random-1 | 3 | 2169.29 | 35.73 | 0.0002 | 0 |
| Medium-lite | Random-30 | 3 | 1810.84 | 11.90 | 0.0038 | 0 |
| Medium-lite | VAA | 3 | 1751.43 | 8.74 | 0.0009 | 0 |
| Medium-lite | Paper-SA-RL5-300 | 3 | 1643.75 | 1.84 | 0.0515 | 0 |
| Medium-lite | DestAgent-RL-150 | 3 | 1634.55 | 1.36 | 0.1922 | 0 |
| Medium-lite | CargoMatrix-RL-150 | 3 | 1656.78 | 2.80 | 0.1304 | 0 |
| Medium-lite | GraphCargoMatrix-RL-150 | 3 | 1639.07 | 1.60 | 1.7166 | 0 |
| Medium-lite | VAA-QRL-300 | 3 | 1613.51 | 0.00 | 0.0461 | 3 |

## Interpretation

The paper-style VAA is now intentionally more faithful to the paper's construction, so it is less aggressive than the previous greedy-evaluator variant. It is a fast initial solution generator rather than a strong local optimizer.

`VAA-QRL-300` is the strongest method on every instance class: it has the lowest average makespan and a 0.00-0.05 % average gap on Tiny, Small, and Medium-lite, winning 8 of 9 generated instances. It keeps the paper's VAA + Q-learning frame but adds critical-door guided operators, a new-best descent polish, and kick-restarts with reheating. With the fast search evaluator it is now also the fastest learning-based method in the table.

Among the previous baselines, `Paper-SA-RL5-300` remains strongest on Tiny and Small, and `DestAgent-RL-150` on Medium-lite. `GraphCargoMatrix-RL-150` with projected door release/workload features is better than fixed `CargoMatrix-RL-150` on Medium-lite, but worse on Tiny and Small and substantially slower because it builds and evaluates a projected door profile at each decision step.
