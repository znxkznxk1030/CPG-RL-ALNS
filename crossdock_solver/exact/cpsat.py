from __future__ import annotations

from dataclasses import dataclass
import sys
import time
import types


_STUB_MARK = "_crossdock_stub"


def _ensure_pandas_importable() -> None:
    """Let `ortools.sat.python.cp_model` import in a broken-NumPy environment.

    cp_model imports pandas at module load only for optional Series/Index typing
    helpers this module never uses. In this environment pandas pulls a
    NumPy-1.x-built pyarrow native library that ABI-conflicts with NumPy 2 and
    SIGBUS-crashes the process, and merely attempting the import already loads
    that native library. So, unless a working pandas is already imported, install
    minimal stubs for pandas and pyarrow to keep the broken native from loading.
    If a real pandas is already present it is used unchanged.

    This is a workaround for a broken environment (pandas/pyarrow built against
    NumPy 1.x while NumPy 2.x is installed); the proper fix is to repair those
    packages. See the environment note in README.
    """

    existing = sys.modules.get("pandas")
    if existing is not None and not getattr(existing, _STUB_MARK, False):
        return

    class _Any:
        pass

    for name in ("pandas", "pyarrow"):
        if getattr(sys.modules.get(name), _STUB_MARK, False):
            continue
        stub = types.ModuleType(name)
        setattr(stub, _STUB_MARK, True)
        stub.Index = _Any
        stub.Series = _Any
        stub.DataFrame = _Any
        stub.__getattr__ = lambda _name: _Any  # type: ignore[attr-defined]
        sys.modules[name] = stub


_ensure_pandas_importable()

from ortools.sat.python import cp_model  # noqa: E402

from crossdock_solver.baselines.random_baseline import BaselineRun
from crossdock_solver.core.evaluator import evaluate_solution
from crossdock_solver.core.solution import Solution
from crossdock_solver.data.instance import CrossDockInstance
from crossdock_solver.exact.milp import _big_m


TIME_SCALE = 100
WEIGHT_SCALE = 1000


@dataclass(frozen=True)
class ExactCPSATConfig:
    time_limit_sec: float | None = 120.0
    workers: int = 8
    msg: bool = False
    tardiness_weight: float = 0.0
    name: str = "Exact-CPSAT"


@dataclass(frozen=True)
class ExactCPSATResult:
    """Result of an OR-Tools CP-SAT solve.

    `objective_value` is the incumbent objective (equals the evaluator objective
    on the extracted solution). `lower_bound` is CP-SAT's best proven lower
    bound. When `proven_optimal` is True the two coincide. `lower_bound` is the
    quantity used to report an optimality gap for the G1 gate.
    """

    run: BaselineRun | None
    status: str
    proven_optimal: bool
    objective_value: float | None
    lower_bound: float | None
    runtime_sec: float


def solve_exact_cpsat(
    instance: CrossDockInstance,
    config: ExactCPSATConfig | None = None,
) -> ExactCPSATResult:
    """Solve the MVP scheduling model exactly with OR-Tools CP-SAT.

    Mirrors the PuLP/CBC formulation in `exact/milp.py` (same decisions and
    timing semantics as the evaluator) but replaces big-M disjunctions with
    reified constraints, which propagate tighter lower bounds. All time
    quantities are scaled to integers by `TIME_SCALE`.
    """

    config = config or ExactCPSATConfig()
    start_time = time.perf_counter()

    compounds = tuple(instance.compound_trucks)
    outbounds = tuple(instance.outbound_trucks)
    destinations = tuple(instance.destinations)
    doors = tuple(instance.doors)

    def scale(value: float) -> int:
        return int(round(value * TIME_SCALE))

    handling = {
        (c, d): scale(instance.handling_time(c, d))
        for c in compounds
        for d in destinations
    }
    total_handling = {
        c: sum(handling[(c, d)] for d in destinations) for c in compounds
    }
    compound_load = {
        (c, d): sum(handling[(s, d)] for s in compounds if s != c)
        for c in compounds
        for d in destinations
    }
    outbound_load = {
        d: sum(handling[(s, d)] for s in compounds) for d in destinations
    }
    enter = {t: scale(instance.enter_time[t]) for t in instance.all_trucks}
    leave = {t: scale(instance.leave_time[t]) for t in instance.all_trucks}
    release = {t: scale(instance.release_time[t]) for t in instance.all_trucks}
    travel = {
        (a, b): scale(instance.travel(a, b)) for a in doors for b in doors
    }
    horizon = scale(_big_m(instance))

    model = cp_model.CpModel()

    x = {
        (c, d, m): model.NewBoolVar(f"x_{c}_{d}_{m}")
        for c in compounds
        for d in destinations
        for m in doors
    }
    z = {
        (o, d, m): model.NewBoolVar(f"z_{o}_{d}_{m}")
        for o in outbounds
        for d in destinations
        for m in doors
    }
    cdest = {(c, d): model.NewBoolVar(f"cdest_{c}_{d}") for c in compounds for d in destinations}
    cdoor = {(c, m): model.NewBoolVar(f"cdoor_{c}_{m}") for c in compounds for m in doors}
    odest = {(o, d): model.NewBoolVar(f"odest_{o}_{d}") for o in outbounds for d in destinations}
    odoor = {(o, m): model.NewBoolVar(f"odoor_{o}_{m}") for o in outbounds for m in doors}

    uf = {c: model.NewIntVar(0, horizon, f"uf_{c}") for c in compounds}
    cf = {c: model.NewIntVar(0, horizon, f"cf_{c}") for c in compounds}
    os = {o: model.NewIntVar(0, horizon, f"os_{o}") for o in outbounds}
    of = {o: model.NewIntVar(0, horizon, f"of_{o}") for o in outbounds}
    cmax = model.NewIntVar(0, horizon, "cmax")

    # Assignment structure -----------------------------------------------------
    for c in compounds:
        model.Add(sum(x[(c, d, m)] for d in destinations for m in doors) == 1)
        for d in destinations:
            model.Add(cdest[(c, d)] == sum(x[(c, d, m)] for m in doors))
        for m in doors:
            model.Add(cdoor[(c, m)] == sum(x[(c, d, m)] for d in destinations))

    for m in doors:
        model.Add(sum(x[(c, d, m)] for c in compounds for d in destinations) <= 1)

    for o in outbounds:
        model.Add(sum(z[(o, d, m)] for d in destinations for m in doors) == 1)
        for d in destinations:
            model.Add(odest[(o, d)] == sum(z[(o, d, m)] for m in doors))
        for m in doors:
            model.Add(odoor[(o, m)] == sum(z[(o, d, m)] for d in destinations))
        model.Add(os[o] >= release[o])

    for d in destinations:
        model.Add(
            sum(x[(c, d, m)] for c in compounds for m in doors)
            + sum(z[(o, d, m)] for o in outbounds for m in doors)
            == 1
        )

    # Compound timing ----------------------------------------------------------
    for c in compounds:
        model.Add(
            uf[c]
            == release[c]
            + enter[c]
            + total_handling[c]
            - sum(handling[(c, d)] * x[(c, d, m)] for d in destinations for m in doors)
        )
        model.Add(cmax >= cf[c])
        for d in destinations:
            model.Add(
                cf[c] >= uf[c] + compound_load[(c, d)] + leave[c]
            ).OnlyEnforceIf(cdest[(c, d)])
            for s in compounds:
                if s == c or instance.unit_amount(s, d) <= 0:
                    continue
                for m in doors:
                    for sm in doors:
                        model.Add(
                            cf[c]
                            >= uf[s] + travel[(sm, m)] + compound_load[(c, d)] + leave[c]
                        ).OnlyEnforceIf([x[(c, d, m)], cdoor[(s, sm)]])

    # Outbound timing ----------------------------------------------------------
    for o in outbounds:
        model.Add(cmax >= of[o])
        for d in destinations:
            model.Add(
                of[o] >= os[o] + enter[o] + outbound_load[d] + leave[o]
            ).OnlyEnforceIf(odest[(o, d)])
            for m in doors:
                for s in compounds:
                    if instance.unit_amount(s, d) <= 0:
                        continue
                    for sm in doors:
                        model.Add(
                            os[o] >= uf[s] + travel[(sm, m)]
                        ).OnlyEnforceIf([z[(o, d, m)], cdoor[(s, sm)]])
        for m in doors:
            for c in compounds:
                model.Add(os[o] >= cf[c]).OnlyEnforceIf([odoor[(o, m)], cdoor[(c, m)]])

    # Outbound sequencing on shared doors -------------------------------------
    before = {}
    for first in outbounds:
        for second in outbounds:
            if first == second:
                continue
            for m in doors:
                b = model.NewBoolVar(f"before_{first}_{second}_{m}")
                before[(first, second, m)] = b
                model.AddImplication(b, odoor[(first, m)])
                model.AddImplication(b, odoor[(second, m)])
                model.Add(os[second] >= of[first]).OnlyEnforceIf(b)
    for i, first in enumerate(outbounds):
        for second in outbounds[i + 1:]:
            for m in doors:
                model.Add(
                    before[(first, second, m)] + before[(second, first, m)]
                    >= odoor[(first, m)] + odoor[(second, m)] - 1
                )

    # Objective ----------------------------------------------------------------
    due_trucks = [t for t in instance.all_trucks if instance.due_time[t] != float("inf")]
    compound_set = set(compounds)
    tard = {t: model.NewIntVar(0, horizon, f"tard_{t}") for t in due_trucks}
    for t in due_trucks:
        finish_var = cf[t] if t in compound_set else of[t]
        model.Add(tard[t] >= finish_var - scale(instance.due_time[t]))

    weight_int = int(round(config.tardiness_weight * WEIGHT_SCALE))
    objective = model.NewIntVar(0, WEIGHT_SCALE * horizon * (1 + len(due_trucks)), "objective")
    model.Add(
        objective == WEIGHT_SCALE * cmax + weight_int * sum(tard[t] for t in due_trucks)
    )
    model.Minimize(objective)

    # Solve --------------------------------------------------------------------
    solver = cp_model.CpSolver()
    if config.time_limit_sec is not None:
        solver.parameters.max_time_in_seconds = float(config.time_limit_sec)
    solver.parameters.num_search_workers = config.workers
    solver.parameters.log_search_progress = config.msg
    status_code = solver.Solve(model)
    runtime_sec = time.perf_counter() - start_time

    status = solver.StatusName(status_code)
    divisor = TIME_SCALE * WEIGHT_SCALE
    proven_optimal = status_code == cp_model.OPTIMAL

    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ExactCPSATResult(
            run=None,
            status=status,
            proven_optimal=False,
            objective_value=None,
            lower_bound=solver.BestObjectiveBound() / divisor,
            runtime_sec=runtime_sec,
        )

    solution = _extract_solution(instance, solver, x, z, os)
    result = evaluate_solution(instance, solution)
    return ExactCPSATResult(
        run=BaselineRun(
            name=config.name,
            solution=solution,
            result=result,
            runtime_sec=runtime_sec,
            samples=1,
        ),
        status=status,
        proven_optimal=proven_optimal,
        objective_value=solver.ObjectiveValue() / divisor,
        lower_bound=solver.BestObjectiveBound() / divisor,
        runtime_sec=runtime_sec,
    )


def _extract_solution(instance, solver, x, z, os) -> Solution:
    compound_assignment = {}
    outbound_assignment = {}
    door_sequences = {door: [] for door in instance.doors}

    for compound in instance.compound_trucks:
        for destination in instance.destinations:
            for door in instance.doors:
                if solver.Value(x[(compound, destination, door)]) == 1:
                    compound_assignment[compound] = (destination, door)

    for outbound in instance.outbound_trucks:
        for destination in instance.destinations:
            for door in instance.doors:
                if solver.Value(z[(outbound, destination, door)]) == 1:
                    outbound_assignment[outbound] = (destination, door)
                    door_sequences[door].append(outbound)

    for sequence in door_sequences.values():
        sequence.sort(key=lambda o: (solver.Value(os[o]), o))

    return Solution(
        compound_assignment=compound_assignment,
        outbound_assignment=outbound_assignment,
        door_sequences=door_sequences,
    )
