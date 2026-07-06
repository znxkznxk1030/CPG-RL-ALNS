# APIEMS 2026 Full Paper Outline (8 pages)

Working title: **"Compound-Truck Cross-Docking Scheduling with Arrival Time
Windows: A Near-Optimal Guided Iterated Local Search and the Limits of Learned
Operator Selection"**

Deadline: full paper (8 pages) by 2026-07-31, via https://www.apiems2026.org/.
Full-paper track is required for Best Paper / journal recommendation (IJPR,
Engineering Optimization, IEMS 등).

Theme fit: "Beyond Intelligence: Toward Optimized Decision-Making" — our story
is literally "beyond (machine) intelligence: well-designed deterministic search
matches an exact solver where learning adds nothing."

## Section plan (page budget)

1. **Introduction (1.0p)**
   - Cross-docking, compound trucks with partial unloading (Shahmardan &
     Sajadieh 2020) — trucks serve inbound and outbound roles.
   - Gap: all-trucks-at-time-zero assumption; no due dates. Real docks have
     staggered arrivals and departure deadlines.
   - Contributions (3): (i) first time-window extension of the compound-truck
     partial-unloading model + MILP/CP-SAT + bounds + open benchmark;
     (ii) a deterministic guided ILS that matches CP-SAT incumbents in
     sub-second time; (iii) a systematic ablation showing learned operator
     selection (tabular Q, transfer DQN) and learned initialization add
     nothing in this regime, with an analysis of when learning can help.

2. **Related work (0.75p)**
   - Compound trucks / partial unloading line (base paper + sparse follow-ups).
   - TW crossdock scheduling for pure in/outbound trucks (mature; cite 4-6).
   - Closest: Li et al. 2025 EJOR Q-ALNS — dock-level mixed service vs our
     truck-level flexibility; their RL also per-instance online.
   - Learned metaheuristics & critiques of weak-baseline comparisons.

3. **Problem definition (1.25p)** — from docs/problem_definition.md
   - Sets, parameters, decisions (assignment + door sequencing).
   - Timing semantics (partial unloading, transfer, door queues).
   - Extension: release r_f (start rule) + soft due d_f;
     objective = makespan + lambda * total tardiness.
   - MILP sketch (constraints referenced, not fully listed — cite base model);
     CP-SAT reification note; combinatorial bounds (critical-chain, door-area,
     structural tardiness).

4. **Guided iterated local search (1.5p)**
   - VAA construction (brief, from base paper) as initialization.
   - Engine: best-improvement descent (relocations, door swaps/moves,
     destination swaps), SA acceptance, kick-restart with reheating.
   - Guided operators: critical-door relocate, critical destination swap,
     tardy-truck relocate, tardy destination swap.
   - Operator-selection policies as a plug-in: uniform / tabular Q (paper
     style) / DQN with scale-invariant features (27-dim) trained offline on a
     train pool — described honestly as the apparatus for the ablation.

5. **Experimental design (1.0p)**
   - Paper-aligned sizes S(6,9,6)/M(12,18,12)/L(20,30,20), compound ratio 2/3.
   - TW tightness levels (none/medium/tight), (rho, delta) parameterization.
   - Train/tuning/test seed-pool separation; test pool touched once.
   - 5 instances/cell x 5 reps; 1000 iterations; CP-SAT 300s (S) / 600s (M,L
     subset); gaps vs max(CP-SAT bound, combinatorial objective bound).

6. **Results (1.75p)**
   - Table 1 (main): mean+-std objective per method per cell; gap to bound;
     vs CP-SAT incumbent. Key rows from K1: S-none GILS within 0.2-0.8% of
     proven optima; beats Paper-SA (1.2%) and VAA (6.5%). TW cells: GILS
     improves VAA ~3%, engine dominates.
   - Table/Fig 2 (ablation): uniform vs tabular vs DQN — statistically
     indistinguishable everywhere (+ init-sensitivity: random start converges
     to same final within +-0.33%). Wilcoxon p-values.
   - Fig 3: budget sensitivity or runtime-vs-quality (GILS sub-second vs
     CP-SAT minutes).
   - Honest note: L-scale bound looseness; near-optimality argued via S proof
     + CP-SAT incumbent ties.

7. **Discussion: when can learning help? (0.5p)**
   - Conditions that voided learning here: cheap evaluation x static
     deterministic x saturating budget x strong local search.
   - Inverted conditions (dynamic arrivals, expensive evaluation, real-time
     latency, extreme scale) as future work — sets up the dynamic follow-up.

8. **Conclusion (0.25p)**

## Assets already available

- Problem definition: docs/problem_definition.md (translate/condense)
- Benchmark rationale: docs/benchmark_design.md
- Novelty positioning: docs/literature_scan_tw.md
- Numbers: outputs/k1_results.jsonl (+ CP-SAT batch), experiments/k1_summary.py
- Ablation numbers: experiments/validate_g2.py, experiments/init_sensitivity.py

## To do before submission

- [ ] CP-SAT batch complete -> final Table 1 numbers
- [ ] Wilcoxon tests for ablation table
- [ ] Budget sensitivity mini-experiment (iterations 50/200/1000)
- [ ] English draft (LaTeX/Word per APIEMS template — check template on site)
- [ ] Naming: present engine as "VAA-GILS"; tabular-Q variant is one ablation arm
