"""Seed protocol and benchmark cell definitions.

The three pools are strictly separated to prevent test-set overfitting:

- train pool: RL policy training only.
- tuning pool: hyperparameter selection (irace/SMAC, manual tuning) only.
- test pool: final reporting only. Never run experiments on test seeds until
  all methods and hyperparameters are frozen.

Seeds are derived deterministically from (pool, cell, index), so an instance
is fully identified by its cell name and pool index.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from crossdock_solver.data.generator import (
    FLOW_PATTERNS,
    SIZE_CLASSES,
    generate_benchmark_instance,
)
from crossdock_solver.data.instance import CrossDockInstance


POOL_BASES: dict[str, int] = {
    "train": 100_000,
    "tuning": 200_000,
    "test": 300_000,
}
POOL_SIZE = 100_000
_CELL_STRIDE = 1_000


@dataclass(frozen=True)
class BenchmarkCell:
    size_class: str
    flow_pattern: str

    @property
    def name(self) -> str:
        return f"{self.size_class}-{self.flow_pattern}"


def benchmark_cells() -> tuple[BenchmarkCell, ...]:
    return tuple(
        BenchmarkCell(size_class=size, flow_pattern=pattern)
        for size, pattern in product(SIZE_CLASSES, FLOW_PATTERNS)
    )


def cell_seed(pool: str, cell: BenchmarkCell, index: int) -> int:
    """Deterministic seed for the index-th instance of a cell in a pool."""

    if pool not in POOL_BASES:
        raise ValueError(f"pool must be one of {tuple(POOL_BASES)}, got {pool!r}")
    if not 0 <= index < _CELL_STRIDE:
        raise ValueError(f"index must be in [0, {_CELL_STRIDE}), got {index}")

    cells = benchmark_cells()
    try:
        cell_index = cells.index(cell)
    except ValueError:
        raise ValueError(f"unknown benchmark cell {cell!r}") from None
    return POOL_BASES[pool] + cell_index * _CELL_STRIDE + index


def cell_instance(pool: str, cell: BenchmarkCell, index: int) -> CrossDockInstance:
    return generate_benchmark_instance(
        cell.size_class,
        cell.flow_pattern,
        seed=cell_seed(pool, cell, index),
    )


def pool_of_seed(seed: int) -> str | None:
    for pool, base in POOL_BASES.items():
        if base <= seed < base + POOL_SIZE:
            return pool
    return None
