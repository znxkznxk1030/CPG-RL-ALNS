"""Method registry for the experiment runner.

All methods must be defined at module import time so multiprocessing workers
(spawn start method on macOS) can resolve them after re-import. A method is a
callable `(instance, seed, budget_sec) -> dict` returning at least `makespan`
and `runtime_sec`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from crossdock_solver.baselines.paper_sa_rl import (
    NEIGHBORHOODS,
    PaperSARLConfig,
    run_paper_sa_rl,
)
from crossdock_solver.baselines.vaa import run_vaa
from crossdock_solver.baselines.vaa_qrl import ACTIONS, ACTIONS_TW, VaaQRLConfig, run_vaa_qrl
from crossdock_solver.data.instance import CrossDockInstance


# Operator pools for the guided-operator ablation (Phase B1). The selection
# policy is held fixed (uniform random) so that only the pool changes:
#   generic  = the 7 paper neighborhoods, no guidance;
#   critical = generic + g1/g2 (bottleneck-guided moves on the makespan-critical
#              door/truck);
#   full     = critical + g3/g4 (tardiness-guided moves), == ACTIONS_TW.
GENERIC_ACTIONS: tuple[str, ...] = tuple(NEIGHBORHOODS)
OPERATOR_POOLS: dict[str, tuple[str, ...]] = {
    "generic": GENERIC_ACTIONS,
    "critical": ACTIONS,
    "full": ACTIONS_TW,
}


MethodFn = Callable[[CrossDockInstance, int, float | None], dict]

ROOT = Path(__file__).resolve().parents[1]
DQN_CHECKPOINT = ROOT / "outputs" / "models" / "gvaa_dqn.pt"

_dqn_agent_cache = None


def _auto_weight(instance: CrossDockInstance) -> float:
    """Tardiness weight 1.0 on time-window instances, 0.0 otherwise."""

    return 1.0 if any(d != float("inf") for d in instance.due_time.values()) else 0.0


def _dqn_agent():
    global _dqn_agent_cache
    if _dqn_agent_cache is None:
        from crossdock_solver.rl.dqn import DQNAgent

        _dqn_agent_cache = DQNAgent.load(DQN_CHECKPOINT)
    return _dqn_agent_cache


def _vaa(instance: CrossDockInstance, seed: int, budget_sec: float | None) -> dict:
    run = run_vaa(instance)
    weight = _auto_weight(instance)
    return {
        "makespan": run.result.makespan,
        "total_tardiness": run.result.total_tardiness,
        "objective": run.result.makespan + weight * run.result.total_tardiness,
        "runtime_sec": run.runtime_sec,
    }


def _paper_sa_rl(iterations: int) -> MethodFn:
    def method(instance: CrossDockInstance, seed: int, budget_sec: float | None) -> dict:
        run = run_paper_sa_rl(
            instance,
            PaperSARLConfig(max_iterations=iterations, seed=seed),
        )
        return {"makespan": run.result.makespan, "runtime_sec": run.runtime_sec}

    return method


def _vaa_qrl(iterations: int, tardiness_weight: float = 0.0) -> MethodFn:
    def method(instance: CrossDockInstance, seed: int, budget_sec: float | None) -> dict:
        run = run_vaa_qrl(
            instance,
            VaaQRLConfig(
                max_iterations=iterations,
                time_budget_sec=budget_sec,
                tardiness_weight=tardiness_weight,
                seed=seed,
            ),
        )
        return {
            "makespan": run.result.makespan,
            "total_tardiness": run.result.total_tardiness,
            "objective": run.result.makespan + tardiness_weight * run.result.total_tardiness,
            "runtime_sec": run.runtime_sec,
        }

    return method


def _gils(iterations: int, selector_name: str = "tabular") -> MethodFn:
    """Guided ILS with automatic tardiness weight and pluggable selector."""

    def method(instance: CrossDockInstance, seed: int, budget_sec: float | None) -> dict:
        weight = _auto_weight(instance)
        selector = None
        if selector_name == "uniform":
            from crossdock_solver.rl.selectors import UniformSelector

            selector = UniformSelector(ACTIONS_TW)
        elif selector_name == "dqn":
            from crossdock_solver.rl.selectors import DQNSelector

            selector = DQNSelector(_dqn_agent(), ACTIONS_TW, epsilon=0.0, train=False)

        run = run_vaa_qrl(
            instance,
            VaaQRLConfig(
                max_iterations=iterations,
                time_budget_sec=budget_sec,
                tardiness_weight=weight,
                seed=seed,
            ),
            selector=selector,
        )
        return {
            "makespan": run.result.makespan,
            "total_tardiness": run.result.total_tardiness,
            "objective": run.result.makespan + weight * run.result.total_tardiness,
            "runtime_sec": run.runtime_sec,
        }

    return method


def _gils_pool(iterations: int, pool: str) -> MethodFn:
    """GILS with uniform selection over a restricted operator pool.

    Guided-operator ablation (Phase B1): selection policy is fixed to uniform
    random, and only the operator pool varies (generic / critical / full). This
    isolates the contribution of the bottleneck- and tardiness-guided operators,
    which the selector ablation cannot show.
    """

    actions = OPERATOR_POOLS[pool]

    def method(instance: CrossDockInstance, seed: int, budget_sec: float | None) -> dict:
        from crossdock_solver.rl.selectors import UniformSelector

        weight = _auto_weight(instance)
        run = run_vaa_qrl(
            instance,
            VaaQRLConfig(
                max_iterations=iterations,
                time_budget_sec=budget_sec,
                tardiness_weight=weight,
                seed=seed,
            ),
            selector=UniformSelector(actions),
        )
        return {
            "makespan": run.result.makespan,
            "total_tardiness": run.result.total_tardiness,
            "objective": run.result.makespan + weight * run.result.total_tardiness,
            "runtime_sec": run.runtime_sec,
        }

    return method


def _gils_ablate(iterations: int, drop: str) -> MethodFn:
    """GILS (uniform selector, full pool) with one engine component removed.

    Leave-one-out component ablation (Phase B2). `drop` in:
      none    -> full engine (reference, == GILS-uniform);
      init    -> VAA construction replaced by a random feasible start;
      descent -> best-improvement descent removed;
      sa      -> SA acceptance replaced by greedy (accept only improvements);
      restart -> kick-restart on stagnation removed.
    """

    def method(instance: CrossDockInstance, seed: int, budget_sec: float | None) -> dict:
        import random

        from crossdock_solver.rl.selectors import UniformSelector

        weight = _auto_weight(instance)
        initial = None
        if drop == "init":
            from crossdock_solver.initial.random_init import random_feasible_solution

            initial = random_feasible_solution(instance, random.Random(seed))

        run = run_vaa_qrl(
            instance,
            VaaQRLConfig(
                max_iterations=iterations,
                time_budget_sec=budget_sec,
                tardiness_weight=weight,
                seed=seed,
                use_descent=(drop != "descent"),
                use_sa_acceptance=(drop != "sa"),
                use_restart=(drop != "restart"),
            ),
            initial_solution=initial,
            selector=UniformSelector(ACTIONS_TW),
        )
        return {
            "makespan": run.result.makespan,
            "total_tardiness": run.result.total_tardiness,
            "objective": run.result.makespan + weight * run.result.total_tardiness,
            "runtime_sec": run.runtime_sec,
        }

    return method


def _cpsat(time_limit_sec: float) -> MethodFn:
    def method(instance: CrossDockInstance, seed: int, budget_sec: float | None) -> dict:
        from crossdock_solver.exact.cpsat import ExactCPSATConfig, solve_exact_cpsat

        weight = _auto_weight(instance)
        result = solve_exact_cpsat(
            instance,
            ExactCPSATConfig(
                time_limit_sec=time_limit_sec,
                workers=8,
                tardiness_weight=weight,
            ),
        )
        record = {
            "status": result.status,
            "proven_optimal": result.proven_optimal,
            "lower_bound": result.lower_bound,
            "runtime_sec": result.runtime_sec,
            "objective": result.objective_value,
        }
        if result.run is not None:
            record["makespan"] = result.run.result.makespan
            record["total_tardiness"] = result.run.result.total_tardiness
        return record

    return method


METHOD_REGISTRY: dict[str, MethodFn] = {
    "VAA": _vaa,
    "Paper-SA-RL5-300": _paper_sa_rl(300),
    "Paper-SA-RL5-1000": _paper_sa_rl(1000),
    "VAA-QRL-50": _vaa_qrl(50),
    "VAA-QRL-300": _vaa_qrl(300),
    "VAA-QRL-1000": _vaa_qrl(1000),
    "VAA-QRL-300-tw1": _vaa_qrl(300, tardiness_weight=1.0),
    "VAA-QRL-1000-tw1": _vaa_qrl(1000, tardiness_weight=1.0),
    "GILS-1000": _gils(1000, "tabular"),
    "GILS-uniform-1000": _gils(1000, "uniform"),
    "GILS-dqn-1000": _gils(1000, "dqn"),
    "GILS-50": _gils(50, "tabular"),
    "GILS-uniform-50": _gils(50, "uniform"),
    "GILS-dqn-50": _gils(50, "dqn"),
    "GILS-200": _gils(200, "tabular"),
    "GILS-uniform-200": _gils(200, "uniform"),
    "GILS-dqn-200": _gils(200, "dqn"),
    "GILS-3000": _gils(3000, "tabular"),
    "GILS-uniform-3000": _gils(3000, "uniform"),
    "GILS-dqn-3000": _gils(3000, "dqn"),
    "GILS-generic-1000": _gils_pool(1000, "generic"),
    "GILS-critical-1000": _gils_pool(1000, "critical"),
    "GILS-full-1000": _gils_pool(1000, "full"),
    "GILS-ablate-none-1000": _gils_ablate(1000, "none"),
    "GILS-ablate-init-1000": _gils_ablate(1000, "init"),
    "GILS-ablate-descent-1000": _gils_ablate(1000, "descent"),
    "GILS-ablate-sa-1000": _gils_ablate(1000, "sa"),
    "GILS-ablate-restart-1000": _gils_ablate(1000, "restart"),
    "CPSAT-300": _cpsat(300.0),
    "CPSAT-600": _cpsat(600.0),
}
