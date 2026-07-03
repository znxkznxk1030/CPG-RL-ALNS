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
- Top-load cargo-matrix destination-agent RL baseline
- Variable-size graph cargo-matrix RL baseline
- Top-load cargo-matrix RL initialized CPG-ALNS baseline
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
24 passed, 8 warnings in 0.98s
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
| `TopLoad-CargoMatrix-RL-150` | Cargo-matrix RL whose active 3-destination window is sorted by remaining destination load, with VAA order as tie-breaker |
| `GraphCargoMatrix-RL-150` | Variable-size graph RL using truck, destination, and door nodes plus cargo and travel edges |
| `CPG-ALNS-300` | Critical-door ALNS initialized from VAA, 300 iterations, regret-2 repair |
| `TopLoad-CargoMatrix-RL-150+CPG-ALNS-300` | CPG-ALNS initialized from the TopLoad-CargoMatrix-RL solution |

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

The top-load cargo-matrix variant keeps the same fixed state size but changes the 3-destination window:

1. Rank remaining destinations by total inbound cargo load.
2. Use the top 3 remaining destinations as the cargo matrix columns.
3. Break equal-load ties by the VAA destination order.
4. Use the learned TopLoad-CargoMatrix-RL solution as an optional initial solution for CPG-ALNS.

The graph cargo-matrix variant keeps `CargoMatrix-RL-150` available for ablation and replaces only the state representation:

1. Build variable-size graph states with truck, destination, and dock-door nodes.
2. Add cargo edges from compound trucks to remaining destinations.
3. Add door-travel edges between dock doors.
4. Pool node and edge features with mean, max, min, and standard deviation.
5. Feed the fixed pooled embedding to the same shared NumPy MLP policy.

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
- Top-load cargo-matrix RL baseline: 150 training episodes per generated instance
- Graph cargo-matrix RL baseline: 150 training episodes per generated instance
- Proposed MVP: critical-door ALNS with VAA initialization, 300 iterations, regret-2 repair
- Top-load RL initialized CPG-ALNS: 300 ALNS iterations after TopLoad-CargoMatrix-RL
- Gap: percentage gap against the best method observed on the same generated instance

Latest results:

| Instance | Method | N | Avg makespan | Avg gap % | Avg runtime sec | Wins |
|---|---:|---:|---:|---:|---:|---:|
| Tiny | Random-1 | 3 | 436.43 | 17.00 | 0.0003 | 0 |
| Tiny | Random-30 | 3 | 420.32 | 8.37 | 0.0047 | 0 |
| Tiny | VAA | 3 | 430.65 | 13.03 | 0.0006 | 0 |
| Tiny | Paper-SA-RL5-300 | 3 | 379.49 | 0.00 | 0.0366 | 3 |
| Tiny | DestAgent-RL-150 | 3 | 385.65 | 2.29 | 0.1082 | 1 |
| Tiny | CargoMatrix-RL-150 | 3 | 385.65 | 2.32 | 0.1155 | 1 |
| Tiny | TopLoad-CargoMatrix-RL-150 | 3 | 385.65 | 2.32 | 0.1153 | 1 |
| Tiny | GraphCargoMatrix-RL-150 | 3 | 386.32 | 2.55 | 0.5166 | 1 |
| Tiny | CPG-ALNS-300 | 3 | 411.62 | 6.41 | 0.5732 | 0 |
| Tiny | TopLoad-CargoMatrix-RL-150+CPG-ALNS-300 | 3 | 384.32 | 1.80 | 0.7741 | 1 |
| Small | Random-1 | 3 | 1624.05 | 38.08 | 0.0001 | 0 |
| Small | Random-30 | 3 | 1370.93 | 14.75 | 0.0033 | 0 |
| Small | VAA | 3 | 1275.57 | 6.62 | 0.0014 | 0 |
| Small | Paper-SA-RL5-300 | 3 | 1195.21 | 0.00 | 0.0562 | 3 |
| Small | DestAgent-RL-150 | 3 | 1239.93 | 3.52 | 0.2284 | 0 |
| Small | CargoMatrix-RL-150 | 3 | 1229.26 | 2.75 | 0.2435 | 0 |
| Small | TopLoad-CargoMatrix-RL-150 | 3 | 1248.57 | 4.62 | 0.2427 | 0 |
| Small | GraphCargoMatrix-RL-150 | 3 | 1226.10 | 2.45 | 0.9431 | 0 |
| Small | CPG-ALNS-300 | 3 | 1275.57 | 6.62 | 1.4566 | 0 |
| Small | TopLoad-CargoMatrix-RL-150+CPG-ALNS-300 | 3 | 1248.57 | 4.62 | 1.5798 | 0 |
| Medium-lite | Random-1 | 3 | 2169.29 | 34.20 | 0.0002 | 0 |
| Medium-lite | Random-30 | 3 | 1810.84 | 10.71 | 0.0049 | 0 |
| Medium-lite | VAA | 3 | 1751.43 | 7.60 | 0.0031 | 0 |
| Medium-lite | Paper-SA-RL5-300 | 3 | 1643.75 | 0.78 | 0.1058 | 0 |
| Medium-lite | DestAgent-RL-150 | 3 | 1634.55 | 0.30 | 0.4431 | 2 |
| Medium-lite | CargoMatrix-RL-150 | 3 | 1656.78 | 1.72 | 0.4646 | 0 |
| Medium-lite | TopLoad-CargoMatrix-RL-150 | 3 | 1647.74 | 1.11 | 0.4643 | 0 |
| Medium-lite | GraphCargoMatrix-RL-150 | 3 | 1632.04 | 0.11 | 1.6259 | 1 |
| Medium-lite | CPG-ALNS-300 | 3 | 1724.76 | 5.77 | 1.7768 | 0 |
| Medium-lite | TopLoad-CargoMatrix-RL-150+CPG-ALNS-300 | 3 | 1647.74 | 1.11 | 2.4147 | 0 |

## Interpretation

The paper-style VAA is now intentionally more faithful to the paper's construction, so it is less aggressive than the previous greedy-evaluator variant. It is a fast initial solution generator rather than a strong local optimizer.

The paper-style `Paper-SA-RL5-300` is still strongest on Tiny and Small. `GraphCargoMatrix-RL-150` improves over the original `CargoMatrix-RL-150` on Small and Medium-lite, and it gives the best average makespan on Medium-lite. It is slower than the fixed cargo matrix because graph construction pools all truck, destination, door, cargo-edge, and travel-edge features each decision step.

`TopLoad-CargoMatrix-RL-150+CPG-ALNS-300` improves over plain `CPG-ALNS-300` on Tiny and is much stronger than VAA-initialized ALNS on Medium-lite, but it does not beat `Paper-SA-RL5-300` on Tiny/Small or `GraphCargoMatrix-RL-150` on Medium-lite. The main ALNS bottleneck is runtime: regret repair enumerates many insertion candidates and calls `evaluate_solution()` on a completed schedule for each candidate.

## Scaled Door Experiment

Run:

```bash
python experiments/compare_scaled_doors.py
```

The script writes:

- `outputs/scaled_door_results.csv`
- `outputs/scaled_door_summary.md`

Scaled settings:

| Instance | Compound trucks | Outbound trucks | Destinations | Dock doors | Product types | Seeds |
|---|---:|---:|---:|---:|---:|---|
| `Medium` | 6 | 9 | 15 | 8 | 4 | 401, 402 |
| `Big` | 8 | 12 | 20 | 10 | 4 | 501, 502 |

Latest scaled-door results:

| Instance | Method | N | Avg makespan | Avg gap % | Avg runtime sec | Wins |
|---|---:|---:|---:|---:|---:|---:|
| Medium | Random-1 | 2 | 3906.32 | 62.13 | 0.0004 | 0 |
| Medium | Random-30 | 2 | 2946.62 | 21.17 | 0.0128 | 0 |
| Medium | VAA | 2 | 2532.86 | 4.57 | 0.0165 | 0 |
| Medium | Paper-SA-RL5-300 | 2 | 2475.16 | 2.06 | 0.2196 | 0 |
| Medium | DestAgent-RL-150 | 2 | 2467.44 | 1.71 | 1.7087 | 0 |
| Medium | CargoMatrix-RL-150 | 2 | 2455.24 | 1.21 | 1.6264 | 0 |
| Medium | TopLoad-CargoMatrix-RL-150 | 2 | 2472.70 | 1.96 | 1.6025 | 0 |
| Medium | GraphCargoMatrix-RL-150 | 2 | 2452.70 | 1.02 | 5.1951 | 1 |
| Medium | CPG-ALNS-300 | 2 | 2447.61 | 0.95 | 14.0938 | 0 |
| Medium | TopLoad-CargoMatrix-RL-150+CPG-ALNS-300 | 2 | 2443.53 | 0.87 | 14.8762 | 1 |
| Big | Random-1 | 2 | 5263.53 | 31.17 | 0.0006 | 0 |
| Big | Random-30 | 2 | 4919.70 | 22.49 | 0.0193 | 0 |
| Big | VAA | 2 | 4144.19 | 3.19 | 0.0400 | 0 |
| Big | Paper-SA-RL5-300 | 2 | 4102.77 | 2.14 | 0.2383 | 0 |
| Big | DestAgent-RL-150 | 2 | 4061.38 | 1.11 | 4.2261 | 0 |
| Big | CargoMatrix-RL-150 | 2 | 4050.11 | 0.84 | 4.3942 | 1 |
| Big | TopLoad-CargoMatrix-RL-150 | 2 | 4046.90 | 0.75 | 4.3350 | 0 |
| Big | GraphCargoMatrix-RL-150 | 2 | 4054.38 | 0.95 | 16.9459 | 0 |
| Big | CPG-ALNS-300 | 2 | 4053.05 | 0.89 | 71.1340 | 1 |
| Big | TopLoad-CargoMatrix-RL-150+CPG-ALNS-300 | 2 | 4042.78 | 0.65 | 78.3342 | 0 |

In the scaled-door setting, `GraphCargoMatrix-RL-150` improves over fixed `CargoMatrix-RL-150` on Medium but not on Big. `TopLoad-CargoMatrix-RL-150+CPG-ALNS-300` still gives the best average makespan on both Medium and Big. The tradeoff is speed: Big averages 78.33 seconds because each ALNS regret-repair iteration evaluates many truck-destination-door-position insertion candidates by rebuilding and re-evaluating a full schedule.
