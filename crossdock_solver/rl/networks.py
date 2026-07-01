from __future__ import annotations

import numpy as np


class NumpyMLP:
    """Small NumPy MLP used by the destination-agent RL baseline.

    This follows the attached RL folder's dependency-light DQN style:
    a shared two-layer network, manual backpropagation, and Adam updates.
    """

    def __init__(
        self,
        obs_size: int,
        hidden: int,
        n_actions: int,
        *,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.obs_size = obs_size
        self.hidden = hidden
        self.n_actions = n_actions
        self.lr = lr

        rng = np.random.default_rng(seed)
        self.w1 = rng.standard_normal((obs_size, hidden)) * np.sqrt(2.0 / obs_size)
        self.b1 = np.zeros(hidden)
        self.w2 = rng.standard_normal((hidden, n_actions)) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(n_actions)
        self.t = 0
        self._init_adam()

    def _init_adam(self) -> None:
        self.mw1 = np.zeros_like(self.w1)
        self.vw1 = np.zeros_like(self.w1)
        self.mb1 = np.zeros_like(self.b1)
        self.vb1 = np.zeros_like(self.b1)
        self.mw2 = np.zeros_like(self.w2)
        self.vw2 = np.zeros_like(self.w2)
        self.mb2 = np.zeros_like(self.b2)
        self.vb2 = np.zeros_like(self.b2)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        single = obs.ndim == 1
        if single:
            obs = obs[np.newaxis, :]

        hidden = np.maximum(0.0, obs @ self.w1 + self.b1)
        q_values = hidden @ self.w2 + self.b2
        return q_values[0] if single else q_values

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> float:
        batch = obs.shape[0]

        z1 = obs @ self.w1 + self.b1
        a1 = np.maximum(0.0, z1)
        q_values = a1 @ self.w2 + self.b2

        predicted = q_values[np.arange(batch), actions]
        delta = targets - predicted
        loss = float(np.mean(delta**2))

        dq = np.zeros_like(q_values)
        dq[np.arange(batch), actions] = -2.0 * delta / batch

        dw2 = a1.T @ dq
        db2 = dq.sum(axis=0)
        da1 = dq @ self.w2.T
        dz1 = da1 * (z1 > 0)
        dw1 = obs.T @ dz1
        db1 = dz1.sum(axis=0)

        self.t += 1
        step = self.t

        def adam(param: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray) -> None:
            m[:] = beta1 * m + (1.0 - beta1) * grad
            v[:] = beta2 * v + (1.0 - beta2) * (grad**2)
            m_hat = m / (1.0 - beta1**step)
            v_hat = v / (1.0 - beta2**step)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

        adam(self.w1, dw1, self.mw1, self.vw1)
        adam(self.b1, db1, self.mb1, self.vb1)
        adam(self.w2, dw2, self.mw2, self.vw2)
        adam(self.b2, db2, self.mb2, self.vb2)
        return loss

