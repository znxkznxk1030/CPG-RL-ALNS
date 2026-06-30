from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossdock_solver.alns.loop import ALNSConfig, simple_alns
from crossdock_solver.core.evaluator import evaluate_solution
from crossdock_solver.data.generator import generate_random_instance
from crossdock_solver.initial.random_init import random_feasible_solution


def main() -> None:
    instance = generate_random_instance(
        num_compounds=4,
        num_outbounds=6,
        num_doors=5,
        num_products=3,
        seed=42,
    )
    initial = random_feasible_solution(instance)
    initial_result = evaluate_solution(instance, initial)

    run = simple_alns(
        instance,
        ALNSConfig(max_iterations=100, repair_name="regret", destroy_size="small", seed=42),
        initial_solution=initial,
    )

    print(f"initial makespan: {initial_result.makespan:.2f}")
    print(f"best makespan:    {run.best_result.makespan:.2f}")
    print(f"critical door:    {run.best_result.critical_door}")
    print(f"critical truck:   {run.best_result.critical_truck}")
    print(f"iterations:       {len(run.logs)}")


if __name__ == "__main__":
    main()
