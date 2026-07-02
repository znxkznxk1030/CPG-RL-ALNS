from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossdock_solver.alns.loop import ALNSConfig, simple_alns
from crossdock_solver.baselines.cargo_matrix_rl import (
    CargoMatrixRLConfig,
    run_cargo_matrix_rl,
    run_topload_cargo_matrix_rl,
)
from crossdock_solver.baselines.destination_agent_rl import (
    DestinationAgentRLConfig,
    run_destination_agent_rl,
)
from crossdock_solver.baselines.paper_sa_rl import PaperSARLConfig, run_paper_sa_rl
from crossdock_solver.baselines.random_baseline import random_best_of, random_one_solution
from crossdock_solver.baselines.vaa import run_vaa
from crossdock_solver.core.evaluator import evaluate_solution
from crossdock_solver.data.generator import generate_random_instance
from crossdock_solver.exact.milp import ExactMILPConfig, solve_exact_milp


@dataclass(frozen=True)
class InstanceSpec:
    name: str
    num_compounds: int
    num_outbounds: int
    num_doors: int
    num_products: int
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class RawObservation:
    instance_class: str
    seed: int
    method: str
    makespan: float
    runtime_sec: float
    best_for_instance: float
    gap_pct: float


DEFAULT_SPECS = (
    InstanceSpec("Tiny", 2, 4, 3, 2, (101, 102, 103)),
    InstanceSpec("Small", 3, 5, 4, 3, (201, 202, 203)),
    InstanceSpec("Medium-lite", 4, 6, 5, 3, (301, 302, 303)),
)


def run_suite(
    *,
    specs: tuple[InstanceSpec, ...] = DEFAULT_SPECS,
    random_samples: int = 30,
    paper_iterations: int = 300,
    destination_agent_episodes: int = 150,
    cargo_matrix_episodes: int = 150,
    alns_iterations: int = 300,
    milp_time_limit_sec: int | None = None,
    milp_instance_names: tuple[str, ...] = (),
    output_dir: Path = ROOT / "outputs",
    raw_filename: str = "baseline_results.csv",
    summary_filename: str = "baseline_summary.md",
    summary_title: str = "Baseline Experiment Summary",
) -> list[RawObservation]:
    output_dir.mkdir(parents=True, exist_ok=True)
    observations: list[RawObservation] = []

    for spec in specs:
        for seed in spec.seeds:
            instance = generate_random_instance(
                num_compounds=spec.num_compounds,
                num_outbounds=spec.num_outbounds,
                num_doors=spec.num_doors,
                num_products=spec.num_products,
                seed=seed,
            )

            method_rows: list[tuple[str, float, float]] = []

            random_one = random_one_solution(instance, seed=seed + 10_000)
            method_rows.append(
                (random_one.name, random_one.result.makespan, random_one.runtime_sec)
            )

            random_many = random_best_of(
                instance,
                samples=random_samples,
                seed=seed + 20_000,
            )
            method_rows.append(
                (random_many.name, random_many.result.makespan, random_many.runtime_sec)
            )

            vaa = run_vaa(instance)
            method_rows.append((vaa.name, vaa.result.makespan, vaa.runtime_sec))

            if milp_time_limit_sec is not None and spec.name in milp_instance_names:
                milp = solve_exact_milp(
                    instance,
                    ExactMILPConfig(
                        time_limit_sec=milp_time_limit_sec,
                        msg=False,
                        name=f"MILP-CBC-{milp_time_limit_sec}s",
                    ),
                )
                if milp.run is not None:
                    method_rows.append(
                        (
                            milp.run.name,
                            milp.run.result.makespan,
                            milp.run.runtime_sec,
                        )
                    )

            paper_sa_rl = run_paper_sa_rl(
                instance,
                PaperSARLConfig(
                    max_iterations=paper_iterations,
                    seed=seed + 25_000,
                ),
                initial_solution=vaa.solution,
            )
            method_rows.append(
                (
                    paper_sa_rl.name,
                    paper_sa_rl.result.makespan,
                    paper_sa_rl.runtime_sec,
                )
            )

            destination_agent_rl = run_destination_agent_rl(
                instance,
                DestinationAgentRLConfig(
                    episodes=destination_agent_episodes,
                    seed=seed + 27_000,
                ),
                initial_solution=vaa.solution,
            )
            method_rows.append(
                (
                    destination_agent_rl.run.name,
                    destination_agent_rl.run.result.makespan,
                    destination_agent_rl.run.runtime_sec,
                )
            )

            cargo_matrix_rl = run_cargo_matrix_rl(
                instance,
                CargoMatrixRLConfig(
                    episodes=cargo_matrix_episodes,
                    seed=seed + 28_000,
                ),
                initial_solution=vaa.solution,
            )
            method_rows.append(
                (
                    cargo_matrix_rl.run.name,
                    cargo_matrix_rl.run.result.makespan,
                    cargo_matrix_rl.run.runtime_sec,
                )
            )

            topload_cargo_matrix_rl = run_topload_cargo_matrix_rl(
                instance,
                CargoMatrixRLConfig(
                    episodes=cargo_matrix_episodes,
                    seed=seed + 29_000,
                ),
                initial_solution=vaa.solution,
            )
            method_rows.append(
                (
                    topload_cargo_matrix_rl.run.name,
                    topload_cargo_matrix_rl.run.result.makespan,
                    topload_cargo_matrix_rl.run.runtime_sec,
                )
            )

            alns_start = time.perf_counter()
            initial = vaa.solution
            run = simple_alns(
                instance,
                ALNSConfig(
                    max_iterations=alns_iterations,
                    destroy_size="small",
                    repair_name="regret",
                    regret_k=2,
                    seed=seed + 30_000,
                ),
                initial_solution=initial,
            )
            alns_runtime = time.perf_counter() - alns_start
            method_rows.append(
                (
                    f"CPG-ALNS-{alns_iterations}",
                    run.best_result.makespan,
                    alns_runtime,
                )
            )

            topload_alns_start = time.perf_counter()
            topload_alns = simple_alns(
                instance,
                ALNSConfig(
                    max_iterations=alns_iterations,
                    destroy_size="small",
                    repair_name="regret",
                    regret_k=2,
                    seed=seed + 31_000,
                ),
                initial_solution=topload_cargo_matrix_rl.run.solution,
            )
            topload_alns_runtime = (
                topload_cargo_matrix_rl.run.runtime_sec
                + time.perf_counter()
                - topload_alns_start
            )
            method_rows.append(
                (
                    f"TopLoad-CargoMatrix-RL-{cargo_matrix_episodes}"
                    f"+CPG-ALNS-{alns_iterations}",
                    topload_alns.best_result.makespan,
                    topload_alns_runtime,
                )
            )

            best_for_instance = min(row[1] for row in method_rows)
            for method, makespan, runtime in method_rows:
                gap_pct = 100.0 * (makespan - best_for_instance) / best_for_instance
                observations.append(
                    RawObservation(
                        instance_class=spec.name,
                        seed=seed,
                        method=method,
                        makespan=makespan,
                        runtime_sec=runtime,
                        best_for_instance=best_for_instance,
                        gap_pct=gap_pct,
                    )
                )

    _write_raw_csv(observations, output_dir / raw_filename)
    _write_summary_markdown(
        observations,
        output_dir / summary_filename,
        specs=specs,
        random_samples=random_samples,
        paper_iterations=paper_iterations,
        destination_agent_episodes=destination_agent_episodes,
        cargo_matrix_episodes=cargo_matrix_episodes,
        alns_iterations=alns_iterations,
        milp_time_limit_sec=milp_time_limit_sec,
        milp_instance_names=milp_instance_names,
        raw_filename=raw_filename,
        summary_title=summary_title,
    )
    return observations


def _write_raw_csv(observations: list[RawObservation], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "instance_class",
                "seed",
                "method",
                "makespan",
                "runtime_sec",
                "best_for_instance",
                "gap_pct",
            ]
        )
        for obs in observations:
            writer.writerow(
                [
                    obs.instance_class,
                    obs.seed,
                    obs.method,
                    f"{obs.makespan:.6f}",
                    f"{obs.runtime_sec:.6f}",
                    f"{obs.best_for_instance:.6f}",
                    f"{obs.gap_pct:.6f}",
                ]
            )


def _write_summary_markdown(
    observations: list[RawObservation],
    path: Path,
    *,
    specs: tuple[InstanceSpec, ...],
    random_samples: int,
    paper_iterations: int,
    destination_agent_episodes: int,
    cargo_matrix_episodes: int,
    alns_iterations: int,
    milp_time_limit_sec: int | None,
    milp_instance_names: tuple[str, ...],
    raw_filename: str,
    summary_title: str,
) -> None:
    lines = [
        f"# {summary_title}",
        "",
        "This summary is generated by `python experiments/compare_baselines.py`.",
        "",
        "Settings:",
        f"- Random baseline: one-shot and best-of-{random_samples}.",
        f"- Paper model baseline: VAA initialization + Q-learning SA, {paper_iterations} iterations.",
        "- Destination-agent RL baseline: each destination agent learns a carrier-truck choice, "
        f"{destination_agent_episodes} training episodes per instance.",
        "- Cargo-matrix RL baseline: VAA-ordered destination agents observe a "
        f"9 compound x 3 destination cargo matrix, {cargo_matrix_episodes} training episodes per instance.",
        "- TopLoad-CargoMatrix-RL baseline: the active 3-destination cargo window is sorted by "
        "remaining destination load with VAA as tie-breaker.",
        "- TopLoad-CargoMatrix-RL + CPG-ALNS: top-load cargo RL initial solution followed by "
        f"critical-door ALNS, {alns_iterations} iterations, regret-2 repair.",
        f"- Proposed MVP: critical-door ALNS with VAA initialization, {alns_iterations} iterations, regret-2 repair.",
        "- Gap is measured against the best method observed on the same generated instance.",
        "",
        "| Instance | Method | N | Avg makespan | Avg gap % | Avg runtime sec | Wins |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if milp_time_limit_sec is not None and milp_instance_names:
        lines.insert(
            10,
            "- MILP-CBC row is a time-limited incumbent; it is not a certified optimum "
            "unless CBC proves optimality within the limit. "
            f"limit={milp_time_limit_sec}s, instances={', '.join(milp_instance_names)}.",
        )

    for row in _aggregate(observations, specs):
        lines.append(
            "| {instance_class} | {method} | {n} | {avg_makespan:.2f} | "
            "{avg_gap_pct:.2f} | {avg_runtime_sec:.4f} | {wins} |".format(**row)
        )

    lines.extend(
        [
            "",
            f"Raw observations are saved in `outputs/{raw_filename}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _aggregate(
    observations: list[RawObservation],
    specs: tuple[InstanceSpec, ...],
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, str], list[RawObservation]] = defaultdict(list)
    for obs in observations:
        grouped[(obs.instance_class, obs.method)].append(obs)

    instance_order = {spec.name: idx for idx, spec in enumerate(specs)}
    method_order = {
        "Random-1": 0,
        "Random-30": 1,
        "VAA": 2,
        "Exact-MILP": 3,
        "MILP-CBC-60s": 4,
        "MILP-CBC-120s": 4,
        "Paper-SA-RL5-300": 5,
        "DestAgent-RL-150": 6,
        "CargoMatrix-RL-150": 7,
        "TopLoad-CargoMatrix-RL-150": 8,
        "CPG-ALNS-300": 9,
        "TopLoad-CargoMatrix-RL-150+CPG-ALNS-300": 10,
    }

    rows = []
    for (instance_class, method), items in grouped.items():
        rows.append(
            {
                "instance_class": instance_class,
                "method": method,
                "n": len(items),
                "avg_makespan": statistics.mean(item.makespan for item in items),
                "avg_gap_pct": statistics.mean(item.gap_pct for item in items),
                "avg_runtime_sec": statistics.mean(item.runtime_sec for item in items),
                "wins": sum(1 for item in items if abs(item.gap_pct) < 1e-9),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            instance_order.get(str(row["instance_class"]), len(instance_order)),
            method_order.get(str(row["method"]), len(method_order)),
            str(row["method"]),
        ),
    )


def main() -> None:
    observations = run_suite()
    print((ROOT / "outputs" / "baseline_summary.md").read_text(encoding="utf-8"))
    print(f"Wrote {len(observations)} raw observations to outputs/baseline_results.csv")


if __name__ == "__main__":
    main()
