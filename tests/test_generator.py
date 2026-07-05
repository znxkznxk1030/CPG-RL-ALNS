from __future__ import annotations

import numpy as np
import pytest

from crossdock_solver.data.generator import (
    FLOW_PATTERNS,
    SIZE_CLASSES,
    generate_benchmark_instance,
    generate_random_instance,
)


def test_all_size_classes_and_patterns_generate_valid_instances() -> None:
    for size_class, shape in SIZE_CLASSES.items():
        for pattern in FLOW_PATTERNS:
            instance = generate_benchmark_instance(size_class, pattern, seed=7)
            assert len(instance.compound_trucks) == shape.num_compounds
            assert len(instance.outbound_trucks) == shape.num_outbounds
            assert len(instance.doors) == shape.num_doors
            assert np.all(instance.flow >= 0)


def test_generation_is_deterministic_per_seed() -> None:
    first = generate_benchmark_instance("M", "skewed", seed=42)
    second = generate_benchmark_instance("M", "skewed", seed=42)
    other = generate_benchmark_instance("M", "skewed", seed=43)

    assert np.array_equal(first.flow, second.flow)
    assert not np.array_equal(first.flow, other.flow)


def test_skewed_pattern_concentrates_destination_load() -> None:
    uniform = generate_benchmark_instance("M", "uniform", seed=5)
    skewed = generate_benchmark_instance("M", "skewed", seed=5)

    def load_cv(instance) -> float:
        loads = np.array([
            sum(instance.handling_time(c, d) for c in instance.compound_trucks)
            for d in instance.destinations
        ])
        return float(loads.std() / (loads.mean() + 1e-9))

    assert load_cv(skewed) > load_cv(uniform)


def test_clustered_pattern_favors_home_cluster() -> None:
    instance = generate_benchmark_instance("M", "clustered", seed=5)
    flow_by_compound_dest = instance.flow.sum(axis=2)

    per_compound_max = flow_by_compound_dest.max(axis=1)
    per_compound_mean = flow_by_compound_dest.mean(axis=1)
    assert np.all(per_compound_max >= 2 * per_compound_mean)


def test_time_windows_follow_tightness_levels() -> None:
    unconstrained = generate_benchmark_instance("S", "uniform", seed=9)
    assert all(value == 0.0 for value in unconstrained.release_time.values())
    assert all(value == float("inf") for value in unconstrained.due_time.values())

    loose = generate_benchmark_instance("S", "uniform", seed=9, tw_tightness="loose")
    tight = generate_benchmark_instance("S", "uniform", seed=9, tw_tightness="tight")

    for instance in (loose, tight):
        for truck in instance.all_trucks:
            assert instance.release_time[truck] >= 0.0
            assert instance.due_time[truck] >= instance.release_time[truck]
            assert instance.due_time[truck] < float("inf")

    def mean_width(instance) -> float:
        widths = [
            instance.due_time[truck] - instance.release_time[truck]
            for truck in instance.all_trucks
        ]
        return sum(widths) / len(widths)

    assert mean_width(tight) < mean_width(loose)
    assert max(tight.release_time.values()) > max(loose.release_time.values())


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError):
        generate_benchmark_instance("XXL", "uniform", seed=1)
    with pytest.raises(ValueError):
        generate_random_instance(
            num_compounds=2,
            num_outbounds=3,
            num_doors=3,
            flow_pattern="zipf",
            seed=1,
        )
