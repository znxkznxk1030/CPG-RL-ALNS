"""Offline training of the transferable operator-selection policy (GVAA-QRL).

Plan phase 2c: episodes are full guided-ILS runs on train-pool instances drawn
from the S/M cells (all flow patterns and time-window levels). The DQN learns
across instances; the checkpoint is later applied zero-shot to unseen tuning-
and test-pool instances.

Run:
    python experiments/train_gvaa.py [--episodes 300] [--iterations 300]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossdock_solver.baselines.vaa import vaa_solution
from crossdock_solver.baselines.vaa_qrl import ACTIONS_TW, VaaQRLConfig, run_vaa_qrl
from crossdock_solver.core.fast_evaluator import FastEvaluator
from crossdock_solver.rl.dqn import DQNAgent
from crossdock_solver.rl.features import feature_dim
from crossdock_solver.rl.selectors import DQNSelector
from experiments.protocol import BenchmarkCell, benchmark_cells, cell_instance


DEFAULT_CHECKPOINT = ROOT / "outputs" / "models" / "gvaa_dqn.pt"


@dataclass(frozen=True)
class GVAATrainConfig:
    episodes: int = 300
    iterations_per_episode: int = 300
    size_classes: tuple[str, ...] = ("S", "M")
    instances_per_cell: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.6
    tw_tardiness_weight: float = 1.0
    seed: int = 0
    checkpoint_path: Path = DEFAULT_CHECKPOINT


def train(config: GVAATrainConfig) -> tuple[DQNAgent, list[dict]]:
    rng = random.Random(config.seed)
    cells = [
        cell for cell in benchmark_cells() if cell.size_class in config.size_classes
    ]
    agent = DQNAgent(
        input_dim=feature_dim(len(ACTIONS_TW)),
        num_actions=len(ACTIONS_TW),
        seed=config.seed,
    )

    decay_episodes = max(1, int(config.episodes * config.epsilon_decay_fraction))
    history: list[dict] = []
    start = time.perf_counter()

    for episode in range(config.episodes):
        cell = rng.choice(cells)
        index = rng.randrange(config.instances_per_cell)
        instance = cell_instance("train", cell, index)
        tardiness_weight = (
            config.tw_tardiness_weight if cell.tw_tightness is not None else 0.0
        )

        fraction = min(1.0, episode / decay_episodes)
        epsilon = config.epsilon_start + fraction * (
            config.epsilon_end - config.epsilon_start
        )

        selector = DQNSelector(agent, ACTIONS_TW, epsilon=epsilon, train=True)
        run = run_vaa_qrl(
            instance,
            VaaQRLConfig(
                max_iterations=config.iterations_per_episode,
                tardiness_weight=tardiness_weight,
                seed=config.seed * 100_000 + episode,
            ),
            selector=selector,
        )

        fast = FastEvaluator(instance, tardiness_weight=tardiness_weight)
        vaa_objective = fast.evaluate(vaa_solution(instance)).objective
        final_objective = (
            run.result.makespan + tardiness_weight * run.result.total_tardiness
        )
        improvement = (vaa_objective - final_objective) / max(1e-9, vaa_objective)

        history.append(
            {
                "episode": episode,
                "cell": cell.name,
                "index": index,
                "epsilon": round(epsilon, 4),
                "vaa_objective": vaa_objective,
                "final_objective": final_objective,
                "improvement": improvement,
            }
        )

    agent.save(config.checkpoint_path)
    history_path = config.checkpoint_path.with_suffix(".history.jsonl")
    with history_path.open("w", encoding="utf-8") as sink:
        for record in history:
            sink.write(json.dumps(record) + "\n")

    elapsed = time.perf_counter() - start
    window = max(1, len(history) // 5)
    early = sum(r["improvement"] for r in history[:window]) / window
    late = sum(r["improvement"] for r in history[-window:]) / window
    print(
        f"trained {config.episodes} episodes in {elapsed:.1f}s | "
        f"improvement over VAA: first-{window} avg {early:.4f} -> last-{window} avg {late:.4f} | "
        f"checkpoint: {config.checkpoint_path}"
    )
    return agent, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    train(
        GVAATrainConfig(
            episodes=args.episodes,
            iterations_per_episode=args.iterations,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
        )
    )


if __name__ == "__main__":
    main()
