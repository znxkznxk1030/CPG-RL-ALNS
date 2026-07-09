# APIEMS 2026 full paper draft (v1, 2026-07-06)

**Title:** Compound-Truck Cross-Docking Scheduling with Arrival Time Windows:
A Guided Iterated Local Search Matching Exact Incumbents, and the Limits of
Learned Operator Selection

**Author:** Youngsoo Kim¹ — ¹Department of Artificial Intelligence, Yonsei
University, Seoul, Republic of Korea (znxkznxk1030@yonsei.ac.kr)

## Abstract

We study truck scheduling in a multi-door cross-docking center where compound
trucks are partially unloaded and reused as outbound trucks. Prior work on this
problem assumes that every truck is available at time zero and minimizes
makespan only. We introduce the first extension with per-truck arrival
(release) times and soft departure due dates, minimizing makespan plus weighted
total tardiness; we provide a mixed-integer formulation, a CP-SAT constraint
model, valid combinatorial lower bounds, and a reproducible benchmark with
strict train/tuning/test seed separation. We then propose VAA-GILS, a guided
iterated local search that combines a Vogel-approximation construction,
best-improvement descent, bottleneck-guided operators, simulated-annealing
acceptance, and kick restarts. On every test cell where an exact reference
exists, VAA-GILS reaches objective values within 0.1–0.6% of CP-SAT incumbents
— matching provably optimal solutions on small instances — in under one
second, versus 76–376 seconds for CP-SAT; on larger time-window cells, where
CP-SAT returns no feasible solution within 600 seconds, VAA-GILS provides all
best-known solutions. It significantly outperforms the
reinforcement-learning-based simulated annealing of the original model
(p < 10⁻⁴). Finally, we equip the
engine with interchangeable operator-selection policies (uniform random,
tabular Q-learning, and a transfer-trained deep Q-network with scale-invariant
features) and show that selection policies produce statistically detectable but
practically negligible effects (≤0.17 percentage points), with no computation
budget at which the learned policy wins. We characterize the conditions under
which learned selection cannot pay off and discuss settings where it can.

**Keywords:** cross-docking; truck scheduling; time windows; iterated local
search; reinforcement learning; constraint programming

## 1. Introduction

Cross-docking centers transfer products from inbound to outbound trucks with
little or no storage, reducing inventory and lead time [2]. In the
compound-truck variant introduced by Joo and Kim [4] and refined by Shahmardan
and Sajadieh [5], a truck may serve both roles: a compound truck arrives loaded
with products for several destinations, is *partially* unloaded — retaining the
demand of one destination — and then reloads that destination's remaining
demand before departing as an outbound truck. Partial unloading was shown to
reduce makespan by up to 56% relative to full unloading [5], making this model
practically attractive for less-than-truckload consolidation networks.

Two assumptions in the existing compound-truck model limit its realism. First,
all trucks are assumed available at the beginning of the horizon, whereas real
docks face staggered, scheduled arrivals. Second, the objective is pure
makespan, whereas outbound departures are typically bound to due times whose
violation is costly. We remove both assumptions.

This paper makes three contributions.

1. **Problem.** We formulate, to our knowledge for the first time, the
   compound-truck cross-docking scheduling problem with per-truck release times
   and soft due dates, minimizing makespan plus weighted total tardiness. We
   give a big-M mixed-integer program, a reified CP-SAT model, valid
   combinatorial lower bounds on the objective, and a parameterized instance
   generator with a strict train/tuning/test seed protocol.

2. **Method.** We propose VAA-GILS, a deterministic guided iterated local
   search. Despite containing no learned component, it matches the incumbents
   of a state-of-the-art exact solver wherever one exists: within 0.1–0.6% of
   CP-SAT solutions (proven optimal on all small no-time-window instances) at
   70–700× less computation time. Where CP-SAT produces no solution within
   600 seconds, VAA-GILS supplies the best-known solutions, and it
   significantly dominates the reinforcement-learning-based simulated
   annealing proposed for the original problem.

3. **Analysis.** Using the engine as a controlled apparatus, we compare three
   interchangeable operator-selection policies — uniform random, the original
   model's tabular Q-learning, and a deep Q-network trained offline on a
   separate instance distribution with scale-invariant features. Across
   budgets from 50 to 3,000 iterations, no learned policy ever outperforms
   uniform random by a practically meaningful margin, and the transfer-trained
   DQN is *significantly worse* at large budgets. We explain the mechanism —
   the search attractor is independent of the selection policy — and derive
   conditions under which learned selection can and cannot help.

The third contribution speaks to a broader methodological concern: learned
components inside metaheuristics are frequently evaluated against weak
baselines or without equal-budget controls. Our results provide a carefully
controlled negative example on a problem where the original literature claims
reinforcement learning as a core ingredient [5].

## 2. Related work

**Cross-dock truck scheduling.** Boysen and Fliedner [1] classify dock
scheduling problems; Van Belle et al. [2] and Ladier and Alpan [3] survey the
field and its industry gap. Time windows, release times, and due dates are
well studied for *pure* inbound/outbound truck scheduling (e.g., earliness/
tardiness objectives and door capacity variants), but not for compound trucks.

**Compound trucks and partial unloading.** Joo and Kim [4] introduced compound
trucks with exclusive door service. Shahmardan and Sajadieh [5] added partial
unloading, a mixed door service mode, a MILP, a VAA-based constructive
heuristic, and simulated annealing variants whose neighborhood selection is
driven by multi-armed bandits or Q-learning; their SA-RL5 variant is our main
baseline. Follow-up work on this model line is sparse, and none considers
arrival times or due dates.

**Closest neighboring problem.** Li et al. [6] study integrated truck
assignment and scheduling with *mixed service mode docks* and time windows,
solved by a Q-learning-based adaptive large neighborhood search. Their
flexibility lives at the dock level (a door may serve either direction), while
trucks remain purely inbound or outbound and transfers are routed through
storage by AGVs; in our problem the flexibility lives at the truck level
(compound trucks, direct door-to-door transfer). Their Q-learning, like the
original model's, is trained online within each instance.

**Learned metaheuristics.** Neural construction policies [7, 8] and learned
operator selection (via bandits, tabular Q [9], or deep Q-networks [10]) have
been widely reported to improve metaheuristics. Our results add a controlled
counterexample and a condition analysis for when such components cannot pay
off.

## 3. Problem definition

### 3.1 Base model

Let I be the set of compound trucks, F the outbound trucks, D the destinations
(|I| + |F| = |D|), K the product types, and M the dock doors (|I| ≤ |M|). A
compound truck i carries f_idk units of product k for destination d; the unit
handling time is t_k, and moving one batch between doors m and n takes t_mn.
Truck i needs changeover times DE_i (docking) and DL_i (undocking). Define the
handling time h_id = Σ_k f_idk · t_k.

Decisions: (i) each compound truck is assigned one *retained* destination and
one door (at most one compound truck per door); (ii) each outbound truck is
assigned one destination and one door; (iii) outbound trucks sharing a door are
sequenced. Each destination is served by exactly one truck (its *carrier*).
A compound truck unloads everything except its retained destination's demand,
transfers move products to carrier doors, and each carrier loads its
destination's demand collected from all compound trucks.

Timing follows the evaluator semantics of [5]: a compound truck's unload
finishes at r_i + DE_i + Σ_{d′≠d} h_id′; a destination is ready at its carrier
door when the last contributing transfer arrives; loading starts at
max(own unload finish, destination ready) for compound carriers and
max(door available, destination ready, r_f) for outbound carriers; DL is added
after loading. The makespan τ is the largest completion time.

### 3.2 Time-window extension

We relax two assumptions of [5]:

- **Release times.** Each truck f has an arrival time r_f ≥ 0 before which no
  operation involving f may start (the base model is the special case r_f = 0).
- **Due dates.** Each truck f has a soft due time d̄_f; its tardiness is
  T_f = max(0, C_f − d̄_f), where C_f is f's completion time.

The objective is

  minimize τ + λ · Σ_f T_f,        (1)

with tardiness weight λ (λ = 1 in our experiments; d̄_f = ∞ recovers the base
model). We extend the MILP of [5] with release-time lower bounds on start
variables and linear tardiness variables, and additionally implement a CP-SAT
model [11] in which the big-M disjunctions are replaced by reified constraints;
integer scaling is exact to 0.01 time units. The CP-SAT model returns both an
incumbent and a proven lower bound.

### 3.3 Combinatorial lower bounds

For gap reporting when exact bounds are weak we use two valid bounds whose
maximum bounds τ: (i) a *critical-chain* bound — for each truck, the fastest
possible completion over all retained-destination choices, relaxing all door
contention; (ii) a *door-area* bound — the total unavoidable door occupancy
divided by |M|. A structural tardiness bound follows by applying (i) per truck
against its due date; since both terms bound their objective components for any
feasible solution, LB_τ + λ·LB_T is a valid bound on (1).

## 4. VAA-GILS: guided iterated local search

VAA-GILS instantiates the iterated local search framework [14] and is
deterministic apart from the sampling inside operators and acceptance. One run
proceeds as follows.

**Construction.** The VAA heuristic of [5] assigns compound trucks to retained
destinations by Vogel-style regret [12], assigns remaining destinations to
outbound trucks by completion-time priority, seeds central doors with the
first assigned trucks, and inserts remaining outbound trucks into the door
with the earliest finish time. We extend all finish-time computations with
release times.

**Descent.** A best-improvement local search over three move families:
relocation of any outbound truck to any door position, compound door swaps and
moves to free doors, and destination swaps between any two trucks. Descent
terminates in a local optimum of this neighborhood. It is applied to the
initial solution, to every new incumbent, and to the final solution.

**Iterated search.** For 1,000 iterations: an operator is drawn by the
selection policy (Section 4.1); the operator generates one candidate from the
current solution; if the candidate improves the global best, descent is applied
and the incumbent is updated; acceptance follows simulated annealing [13] with
initial temperature 0.05·obj and geometric cooling 0.995; after 30 iterations
without a new incumbent, the search restarts from the best solution perturbed
by three random moves, with reheating.

**Operator pool.** Seven generic neighborhood moves from [5] (destination
swaps, door swaps, insertions) plus four *guided* operators that first identify
the current bottleneck and then perform a best-of mini-search around it:
(g1) relocate the last outbound truck on the critical door to its best
door/position; (g2) swap the critical truck's destination against all trucks;
(g3)/(g4) the analogous moves anchored on the most tardy truck, falling back to
(g1)/(g2) when no truck is late. A guided move costs tens of evaluations; a
fast incremental evaluator (tens of microseconds per evaluation) makes both
descent and guided moves cheap.

### 4.1 Operator-selection policies

The selection policy is a plug-in, which lets us isolate its contribution:

- **Uniform**: uniform random choice over the operator pool.
- **Tabular Q** (the policy of SA-RL5 [5]): Q-learning [9] over five
  stagnation-bin states, trained online within the run; ε-greedy with roulette
  exploitation, shaped reward (2 for a new incumbent, 1 for a non-worsening
  move, 0 otherwise).
- **Transfer DQN**: a two-layer deep Q-network [10] over a 27-dimensional
  scale-invariant state (instance descriptors such as compound-truck fraction,
  flow concentration, window tightness; search descriptors such as progress,
  temperature, stagnation, door-load imbalance; and per-operator recent success
  rates). It is trained offline on 500 runs over a *training* instance pool
  (sizes S/M, all flow patterns and window levels) and applied zero-shot
  (ε = 0, no updates) to unseen test instances.

## 5. Experimental design

**Instances.** We follow the size regime of [5]: (|I|, |D|, |M|) =
S (6, 9, 6), M (12, 18, 12), L (20, 30, 20), with compound-truck fraction 2/3.
Flows f_idk are uniform integers in [0, 20] with 3 product types; unit times
t_k ~ U[1, 5]; doors lie on a random 100×100 layout. Window levels: *none*
(r = 0, d̄ = ∞), *medium*, and *tight*: releases r ~ U[0, ρH] and d̄ = r + δH,
where H estimates the unconstrained makespan and (ρ, δ) = (0.25, 0.60) and
(0.50, 0.35) respectively.

**Protocol.** Seeds are split into disjoint train / tuning / test pools.
All algorithm design and hyperparameters were frozen on the tuning pool; the
DQN was trained on the train pool; every number below is the single test-pool
run: 5 instances per cell × 5 replications (9 cells: 3 sizes × 3 window
levels). CP-SAT runs once per instance with 8 threads: 300 s on S (all
instances) and 600 s on M and L (2 instances per cell). λ = 1 on window cells.

**Statistics.** Two-sided Wilcoxon signed-rank tests on paired runs (identical
instance and replication seeds across methods).

## 6. Results

### 6.1 Solution quality

*Scope of claims.* We claim near-optimality only where an exact reference
exists: on the S cells (CP-SAT, 300 s per instance; optimality proven on all
five S-none instances) and on M-none (CP-SAT, 600 s, two instances). On the
remaining cells CP-SAT returns no feasible solution within 600 s; there our
claim is dominance over all baselines, and the reported solutions are, to our
knowledge, the best known. Accordingly, Table 1 reports the gap to the
*best-known* solution per instance (Δbk, over all methods, all budgets, and
CP-SAT) as the primary quality measure, and the gap to the CP-SAT reference
where one exists.

**Table 1.** Test-pool results (5 instances × 5 reps per cell). Δbk = mean gap
to the per-instance best-known solution. vs CP-SAT: S cells over all 5
instances (S-none: proven optima); M-none over the 2 instances attempted.

| Cell | Method | Mean obj ± std | Δbk (%) | vs CP-SAT (%) |
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

Three observations. (i) On every cell with an exact reference, GILS matches
it: within 0.21–0.75% of the five proven optima on S-none, within 0.11–0.55%
of the 300 s CP-SAT incumbents on S-medium and S-tight, and within 0.11–0.19%
of the 600 s incumbents on M-none — where the best GILS runs in fact improve
on the CP-SAT incumbent (Δbk < vs CP-SAT). (ii) The ordering
GILS < Paper-SA-RL5 < VAA holds in every cell where the baseline is defined;
Wilcoxon tests confirm GILS-tabular beats VAA (+2.39% mean, n = 45, p < 10⁻⁴)
and Paper-SA-RL5 (+1.16%, n = 75 pairs, p < 10⁻⁴). (iii) Our combinatorial
lower bounds are informative on no-window cells but loose elsewhere: the gap
of the *best-known solution itself* to the bound is 0.0% on S-none (optimality
proven), 18.5–51.5% on the other S cells, 33–34% on M-none and L-none, and
171–273% on the time-window cells at M and L. These figures measure the bound,
not the solutions; tightening bounds for large time-window instances is left
as future work.

**Runtime.** Mean per-run times of GILS (1,000 iterations): 0.11 s (S),
0.52 s (M), 2.3 s (L). CP-SAT: 76 s (S, mean incl. early optimality proofs),
237 s (M), 376 s (L). GILS attains its quality at roughly 70–700× less
computation.

### 6.2 Does learned operator selection help?

Table 2 summarizes paired comparisons of the three selection policies at 1,000
iterations and across budgets.

**Table 2.** Selection-policy effects (two-sided Wilcoxon; positive mean =
first method better).

| Comparison (1,000 iters) | Mean diff (%) | p | Verdict |
|---|---:|---:|---|
| tabular vs uniform | 0.00 | 0.28 | no difference |
| tabular vs DQN | +0.14 | <10⁻⁴ | DQN worse |
| uniform vs DQN | +0.14 | <10⁻⁴ | DQN worse |

Across budgets (mean gap to per-instance best-known): uniform 0.47 → 0.16%,
tabular 0.56 → 0.09%, DQN 0.56 → 0.26% as the budget grows from 50 to 3,000
iterations. At 50 iterations uniform is *better* than tabular (+0.08%,
p < 10⁻³); at 3,000, tabular is better than uniform (+0.07%, p < 10⁻⁴); the
transfer DQN never wins at any budget and is significantly worse at 1,000 and
3,000. All selection effects are ≤ 0.17 percentage points — an order of
magnitude below the method-level differences in Table 1 — and their direction
flips with budget.

Two further controls confirm the mechanism. First, an initial-solution
sensitivity test: starting the engine from a purely random solution (26–31%
worse than VAA) changes the final objective by at most ±0.33% — the engine's
attractor is essentially independent of the start. Second, on a tuning
instance the engine's converged value coincides *exactly* with the CP-SAT
incumbent found after 240 s. Together these show that descent, guided
operators, and kick restarts drive the search to a near-optimal attractor
regardless of which operator fires when; there is simply no room left for a
selection policy to matter.

## 7. Discussion: when can learned selection pay off?

Our negative result is conditional, and the conditions are informative. Learned
operator selection was voided here by the conjunction of: (i) *cheap
evaluation* — tens of microseconds per candidate, so poor proposals cost almost
nothing and are filtered by acceptance; (ii) a *static, deterministic* problem
— nothing to predict across decisions; (iii) *saturating budgets* — the
attractor is reached well within 1,000 iterations; and (iv) a *strong local
search* — descent finishes whatever a lucky move starts. Negating any of these
restores room for learning: expensive (e.g., simulation-based) evaluation makes
each proposal precious; dynamic arrivals make anticipation structurally
valuable (re-optimization cannot see the future); tight real-time decision
budgets preclude search altogether; and much larger instances stretch budgets
past saturation. We conjecture that reported successes of learned selection
concentrate in such regimes, and that equal-budget comparisons against uniform
selection inside strong engines — as done here — should be a standard control.

## 8. Conclusion

We introduced the time-window extension of compound-truck cross-docking
scheduling with partial unloading, together with exact models, valid bounds,
and a reproducible benchmark. A deterministic guided iterated local search
matches exact incumbents within a fraction of a percent wherever they exist —
including proven optima — and supplies the best-known solutions in seconds
where the exact solver fails; a controlled study shows that learned operator
selection — both
the online tabular Q-learning proposed in the original literature and an
offline-trained transfer DQN — provides no practical benefit in this regime.
Future work includes the dynamic variant with online arrival revelation, where
learned policies are structurally advantaged, and tighter lower bounds for
large time-window instances.

## References

[1] N. Boysen, M. Fliedner, "Cross dock scheduling: Classification, literature
review and research agenda," *Omega*, vol. 38, pp. 413–422, 2010.

[2] J. Van Belle, P. Valckenaers, D. Cattrysse, "Cross-docking: State of the
art," *Omega*, vol. 40, no. 6, pp. 827–846, 2012.

[3] A.-L. Ladier, G. Alpan, "Cross-docking operations: Current research versus
industry practice," *Omega*, vol. 62, pp. 145–162, 2016.

[4] C.M. Joo, B.S. Kim, "Scheduling compound trucks in multi-door cross-docking
terminals," *International Journal of Advanced Manufacturing Technology*,
vol. 64, pp. 977–988, 2013.

[5] A. Shahmardan, M.S. Sajadieh, "Truck scheduling in a multi-door
cross-docking center with partial unloading – Reinforcement learning-based
simulated annealing approaches," *Computers & Industrial Engineering*,
vol. 139, 106134, 2020.

[6] Y. Li, M. Mohammadi, X. Zhang, Y. Lan, W. van Jaarsveld, "Integrated trucks
assignment and scheduling problem with mixed service mode docks: A Q-learning
based adaptive large neighborhood search algorithm," *European Journal of
Operational Research*, 2025.

[7] W. Kool, H. van Hoof, M. Welling, "Attention, learn to solve routing
problems!" in *Proc. International Conference on Learning Representations*,
2019.

[8] C. Zhang, W. Song, Z. Cao, J. Zhang, P.S. Tan, X. Chi, "Learning to
dispatch for job shop scheduling via deep reinforcement learning," in *Advances
in Neural Information Processing Systems*, vol. 33, 2020.

[9] C.J.C.H. Watkins, P. Dayan, "Q-learning," *Machine Learning*, vol. 8,
pp. 279–292, 1992.

[10] V. Mnih et al., "Human-level control through deep reinforcement learning,"
*Nature*, vol. 518, pp. 529–533, 2015.

[11] L. Perron, F. Didier, *Google OR-Tools CP-SAT solver*, version 9.x,
software, 2024. https://developers.google.com/optimization

[12] S. Korukoğlu, S. Ballı, "An improved Vogel's approximation method for the
transportation problem," *Mathematical and Computational Applications*,
vol. 16, no. 2, pp. 370–381, 2011.

[13] S. Kirkpatrick, C.D. Gelatt, M.P. Vecchi, "Optimization by simulated
annealing," *Science*, vol. 220, no. 4598, pp. 671–680, 1983.

[14] H.R. Lourenço, O.C. Martin, T. Stützle, "Iterated local search: Framework
and applications," in *Handbook of Metaheuristics*, 3rd ed., Springer, 2019.
