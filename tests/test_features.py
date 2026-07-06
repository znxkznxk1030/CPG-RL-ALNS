from __future__ import annotations

import numpy as np

from crossdock_solver.core.fast_evaluator import FastEvaluator
from crossdock_solver.baselines.vaa import vaa_solution
from crossdock_solver.data.generator import generate_benchmark_instance
from crossdock_solver.rl.features import (
    INSTANCE_FEATURE_DIM,
    SEARCH_FEATURE_DIM,
    OperatorSuccessTracker,
    SearchState,
    build_feature_vector,
    feature_dim,
    instance_features,
    search_features,
)


def _search_state(instance, result):
    return SearchState(
        iteration=10,
        max_iterations=300,
        temperature=50.0,
        initial_temperature=100.0,
        current=result,
        best=result,
        no_improvement=3,
        no_improvement_cap=20,
        since_best=5,
        restart_after=30,
        num_trucks=len(instance.all_trucks),
    )


def test_feature_vector_has_fixed_dim_across_sizes() -> None:
    actions = ("a", "b", "c")
    tracker = OperatorSuccessTracker(actions)

    dims = []
    for size in ("S", "L"):
        instance = generate_benchmark_instance(size, "uniform", seed=1, tw_tightness="medium")
        fast = FastEvaluator(instance, tardiness_weight=1.0)
        result = fast.evaluate(vaa_solution(instance))
        vec = build_feature_vector(
            instance_features(instance), _search_state(instance, result), tracker
        )
        dims.append(vec.shape[0])
        assert np.all(np.isfinite(vec))

    assert dims[0] == dims[1] == feature_dim(len(actions))


def test_instance_features_are_bounded_and_scale_invariant_shape() -> None:
    for size in ("S", "M", "L", "XL"):
        for tightness in (None, "tight"):
            instance = generate_benchmark_instance(
                size, "skewed", seed=3, tw_tightness=tightness
            )
            feats = instance_features(instance)
            assert feats.shape == (INSTANCE_FEATURE_DIM,)
            assert np.all(feats >= 0.0)
            assert np.all(feats <= 4.0)


def test_search_features_bounded() -> None:
    instance = generate_benchmark_instance("S", "uniform", seed=2, tw_tightness="tight")
    fast = FastEvaluator(instance, tardiness_weight=1.0)
    result = fast.evaluate(vaa_solution(instance))
    feats = search_features(_search_state(instance, result))

    assert feats.shape == (SEARCH_FEATURE_DIM,)
    assert np.all(feats >= 0.0)
    assert np.all(feats <= 4.0)


def test_tracker_moves_toward_outcomes() -> None:
    tracker = OperatorSuccessTracker(("op1", "op2"))
    for _ in range(20):
        tracker.update("op1", True)
        tracker.update("op2", False)
    vec = tracker.vector()
    assert vec[0] > 0.9
    assert vec[1] < 0.1
