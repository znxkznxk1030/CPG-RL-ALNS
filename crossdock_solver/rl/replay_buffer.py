from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Fixed-size circular replay buffer for shared destination agents."""

    def __init__(self, capacity: int, obs_size: int, *, seed: int = 0) -> None:
        self.capacity = capacity
        self.obs_size = obs_size
        self.rng = np.random.default_rng(seed)
        self._obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def push(self, obs: np.ndarray, action: int, reward: float) -> None:
        idx = self._ptr
        self._obs[idx] = obs
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._ptr = (idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, self._size, size=batch_size)
        return self._obs[idx], self._actions[idx], self._rewards[idx]

    def __len__(self) -> int:
        return self._size

