from __future__ import annotations

import pytest

from experiments.protocol import (
    POOL_BASES,
    BenchmarkCell,
    benchmark_cells,
    cell_instance,
    cell_seed,
    pool_of_seed,
)


def test_pools_are_disjoint_across_all_cells() -> None:
    seeds: set[int] = set()
    for pool in POOL_BASES:
        for cell in benchmark_cells():
            for index in range(3):
                seed = cell_seed(pool, cell, index)
                assert seed not in seeds
                seeds.add(seed)
                assert pool_of_seed(seed) == pool


def test_cells_cover_size_pattern_and_tightness_grid() -> None:
    cells = benchmark_cells()
    assert len(cells) == 48
    assert BenchmarkCell("S", "uniform") in cells
    assert BenchmarkCell("S", "uniform", "tight") in cells
    assert BenchmarkCell("XL", "clustered", "loose") in cells


def test_cell_instance_is_deterministic() -> None:
    cell = BenchmarkCell("S", "skewed")
    first = cell_instance("tuning", cell, 0)
    second = cell_instance("tuning", cell, 0)
    assert (first.flow == second.flow).all()
    assert (first.travel_time == second.travel_time).all()
    assert first.enter_time == second.enter_time
    assert first.leave_time == second.leave_time


def test_invalid_pool_and_index_raise() -> None:
    cell = BenchmarkCell("S", "uniform")
    with pytest.raises(ValueError):
        cell_seed("validation", cell, 0)
    with pytest.raises(ValueError):
        cell_seed("train", cell, 1_000)
    with pytest.raises(ValueError):
        cell_seed("train", BenchmarkCell("S", "zipf"), 0)
