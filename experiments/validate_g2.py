"""G2 gate validation: zero-shot policy vs uniform-random vs tabular selectors.

Plan phase 2d. Compares three operator-selection policies inside the identical
guided-ILS engine on tuning-pool instances (never seen in training):

- uniform: uniform random operator choice (the floor the policy must beat)
- tabular: per-instance online tabular Q-learning (the incumbent VAA-QRL)
- dqn: pretrained DQN applied zero-shot (epsilon = 0, no learning)

Pass criterion (full gate, run at scale): dqn beats uniform significantly and
is not worse than tabular at equal budget.

Run:
    python experiments/validate_g2.py [--checkpoint outputs/models/gvaa_dqn.pt]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossdock_solver.baselines.vaa_qrl import ACTIONS_TW, VaaQRLConfig, run_vaa_qrl
from crossdock_solver.rl.dqn import DQNAgent
from crossdock_solver.rl.selectors import DQNSelector, UniformSelector
from experiments.protocol import BenchmarkCell, cell_instance
from experiments.train_gvaa import DEFAULT_CHECKPOINT


def validate(
    checkpoint: Path,
    *,
    size_classes: tuple[str, ...] = ("S", "M", "L"),
    tw_levels: tuple[str | None, ...] = (None, "medium"),
    indices: tuple[int, ...] = (0, 1, 2),
    reps: tuple[int, ...] = (0, 1, 2),
    iterations: int = 300,
) -> dict:
    agent = DQNAgent.load(checkpoint)

    def make_selector(name: str):
        if name == "uniform":
            return UniformSelector(ACTIONS_TW)
        if name == "dqn":
            return DQNSelector(agent, ACTIONS_TW, epsilon=0.0, train=False)
        return None  # tabular default

    rows = defaultdict(lambda: defaultdict(list))
    for size in size_classes:
        for tw in tw_levels:
            cell = BenchmarkCell(size, "uniform", tw)
            weight = 1.0 if tw is not None else 0.0
            for index in indices:
                instance = cell_instance("tuning", cell, index)
                for rep in reps:
                    for method in ("uniform", "tabular", "dqn"):
                        run = run_vaa_qrl(
                            instance,
                            VaaQRLConfig(
                                max_iterations=iterations,
                                tardiness_weight=weight,
                                seed=90_000 + index * 100 + rep,
                            ),
                            selector=make_selector(method),
                        )
                        objective = (
                            run.result.makespan
                            + weight * run.result.total_tardiness
                        )
                        rows[cell.name][method].append(objective)

    print(f"{'cell':<14}{'uniform':>12}{'tabular':>12}{'dqn':>12}  dqn vs uniform / tabular")
    summary = {}
    for cell_name, methods in rows.items():
        means = {m: statistics.mean(v) for m, v in methods.items()}
        vs_uniform = 100.0 * (means["uniform"] - means["dqn"]) / means["uniform"]
        vs_tabular = 100.0 * (means["tabular"] - means["dqn"]) / means["tabular"]
        summary[cell_name] = {**means, "vs_uniform_pct": vs_uniform, "vs_tabular_pct": vs_tabular}
        print(
            f"{cell_name:<14}{means['uniform']:>12.1f}{means['tabular']:>12.1f}"
            f"{means['dqn']:>12.1f}  {vs_uniform:+6.2f}% / {vs_tabular:+6.2f}%"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    validate(args.checkpoint, iterations=args.iterations)


if __name__ == "__main__":
    main()
