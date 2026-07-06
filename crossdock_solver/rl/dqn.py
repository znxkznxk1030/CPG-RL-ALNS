"""Small torch DQN for transferable operator selection (plan phase 2b/2c).

The agent is intentionally compact: operator selection is a low-dimensional
problem (feature vector ~27, actions ~9). Training uses a replay buffer and a
target network; all transitions are treated as non-terminal (continuing task,
bounded by the discount factor).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, input_dim: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, input_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, input_dim), dtype=np.float32)
        self.size = 0
        self.cursor = 0
        self._rng = np.random.default_rng(seed)

    def push(self, state, action, reward, next_state) -> None:
        i = self.cursor
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self._rng.integers(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
        )

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        *,
        hidden: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.9,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        warmup: int = 500,
        train_every: int = 1,
        target_sync: int = 500,
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden = hidden
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup = warmup
        self.train_every = train_every
        self.target_sync = target_sync

        self.online = QNetwork(input_dim, num_actions, hidden)
        self.target = QNetwork(input_dim, num_actions, hidden)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity, input_dim, seed=seed)
        self.steps = 0
        self.updates = 0

    @torch.no_grad()
    def q_values(self, features: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0)
        return self.online(tensor).squeeze(0).numpy()

    def observe(self, state, action: int, reward: float, next_state) -> None:
        self.buffer.push(state, action, reward, next_state)
        self.steps += 1
        if len(self.buffer) >= self.warmup and self.steps % self.train_every == 0:
            self._train_step()

    def _train_step(self) -> None:
        states, actions, rewards, next_states = self.buffer.sample(self.batch_size)
        states_t = torch.from_numpy(states)
        actions_t = torch.from_numpy(actions)
        rewards_t = torch.from_numpy(rewards)
        next_t = torch.from_numpy(next_states)

        q = self.online(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q = rewards_t + self.gamma * self.target(next_t).max(dim=1).values
        loss = nn.functional.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.online.state_dict(),
                "input_dim": self.input_dim,
                "num_actions": self.num_actions,
                "hidden": self.hidden,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "DQNAgent":
        payload = torch.load(Path(path), weights_only=True)
        agent = cls(
            payload["input_dim"],
            payload["num_actions"],
            hidden=payload["hidden"],
        )
        agent.online.load_state_dict(payload["state_dict"])
        agent.target.load_state_dict(payload["state_dict"])
        return agent
