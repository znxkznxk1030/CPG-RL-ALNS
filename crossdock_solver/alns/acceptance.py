from __future__ import annotations

import math
import random


def accept_by_sa(
    old_makespan: float,
    new_makespan: float,
    temperature: float,
    rng: random.Random | None = None,
) -> bool:
    rng = rng or random.Random()
    delta = new_makespan - old_makespan
    if delta <= 0:
        return True
    if temperature <= 0:
        return False
    probability = math.exp(-delta / temperature)
    return rng.random() < probability

