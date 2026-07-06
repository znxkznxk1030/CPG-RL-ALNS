"""Operator-selection policies for the guided ILS engine.

The engine (`baselines/vaa_qrl.run_vaa_qrl`) delegates "which operator now" to
a selector. Three families are provided:

- `UniformSelector`: uniform random choice (G2 ablation floor).
- `TabularQSelector`: the paper-style tabular Q-learning over stagnation bins,
  learned online per instance. This replicates the original in-loop logic
  exactly (same RNG call pattern), so the default engine behavior is unchanged.
- `DQNSelector`: a feature-based DQN policy. In training mode it learns across
  instances; with `epsilon=0` and a pretrained agent it runs zero-shot on
  unseen instances (plan axis A).
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from crossdock_solver.baselines.paper_sa_rl import _select_action


@dataclass
class SelectionContext:
    """State snapshot handed to the selector at each iteration."""

    state_bin: int
    features: np.ndarray | None = None


class UniformSelector:
    needs_features = False

    def __init__(self, actions: tuple[str, ...]) -> None:
        self.actions = actions

    def select(self, context: SelectionContext, rng: random.Random) -> str:
        return rng.choice(self.actions)

    def observe(
        self,
        context: SelectionContext,
        action: str,
        reward: float,
        next_context: SelectionContext,
    ) -> None:
        return None


class TabularQSelector:
    """Paper-style Q-learning over no-improvement bins (per-instance, online)."""

    needs_features = False

    def __init__(
        self,
        actions: tuple[str, ...],
        *,
        learning_rate: float,
        discount_factor: float,
        random_prob: float,
        roulette_prob: float,
    ) -> None:
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.random_prob = random_prob
        self.roulette_prob = roulette_prob
        self.q_values = {
            (state, action): 0.0
            for state in range(1, 6)
            for action in actions
        }

    def select(self, context: SelectionContext, rng: random.Random) -> str:
        return _select_action(
            context.state_bin,
            self.actions,
            self.q_values,
            rng,
            random_prob=self.random_prob,
            roulette_prob=self.roulette_prob,
        )

    def observe(
        self,
        context: SelectionContext,
        action: str,
        reward: float,
        next_context: SelectionContext,
    ) -> None:
        state = context.state_bin
        next_state = next_context.state_bin
        self.q_values[(state, action)] += self.learning_rate * (
            reward
            + self.discount_factor
            * max(self.q_values[(next_state, a)] for a in self.actions)
            - self.q_values[(state, action)]
        )


class DQNSelector:
    """Feature-based DQN policy; zero-shot when pretrained with epsilon=0."""

    needs_features = True

    def __init__(
        self,
        agent,
        actions: tuple[str, ...],
        *,
        epsilon: float = 0.0,
        train: bool = False,
    ) -> None:
        self.agent = agent
        self.actions = actions
        self.epsilon = epsilon
        self.train = train

    def select(self, context: SelectionContext, rng: random.Random) -> str:
        if self.epsilon > 0.0 and rng.random() < self.epsilon:
            return rng.choice(self.actions)
        q = self.agent.q_values(context.features)
        return self.actions[int(np.argmax(q))]

    def observe(
        self,
        context: SelectionContext,
        action: str,
        reward: float,
        next_context: SelectionContext,
    ) -> None:
        if not self.train:
            return
        self.agent.observe(
            context.features,
            self.actions.index(action),
            reward,
            next_context.features,
        )
