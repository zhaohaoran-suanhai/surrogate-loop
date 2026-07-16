from dataclasses import replace

import numpy as np
import pytest

from surrogate_loop.operator.heat1d.evaluation import (
    FieldMetrics,
    compute_field_metrics,
    deeponet_is_acceptable,
    solver_is_acceptable,
)


def test_exact_prediction_has_zero_metrics() -> None:
    reference = np.ones((3, 5, 7))
    reference[:, :, [0, -1]] = 0.0

    metrics = compute_field_metrics(reference, reference.copy(), target_std=2.0)

    assert metrics.median_relative_l2 == 0.0
    assert metrics.p95_relative_l2 == 0.0
    assert metrics.worst_relative_l2 == 0.0
    assert metrics.normalized_rmse == 0.0
    assert metrics.initial_relative_l2 == 0.0
    assert metrics.boundary_max_absolute_error == 0.0


def test_field_metrics_match_manual_uniform_error() -> None:
    reference = np.ones((2, 3, 4))
    prediction = reference + 0.1

    metrics = compute_field_metrics(reference, prediction, target_std=0.5)

    np.testing.assert_allclose(metrics.median_relative_l2, 0.1)
    np.testing.assert_allclose(metrics.p95_relative_l2, 0.1)
    np.testing.assert_allclose(metrics.worst_relative_l2, 0.1)
    np.testing.assert_allclose(metrics.normalized_rmse, 0.2)
    np.testing.assert_allclose(metrics.initial_relative_l2, 0.1)
    np.testing.assert_allclose(metrics.boundary_max_absolute_error, 0.1)


def test_one_failed_threshold_rejects_deeponet(smoke_operator_spec) -> None:
    passing = FieldMetrics(
        median_relative_l2=0.01,
        p95_relative_l2=0.04,
        worst_relative_l2=0.09,
        normalized_rmse=0.01,
        initial_relative_l2=0.02,
        boundary_max_absolute_error=0.001,
    )

    assert deeponet_is_acceptable(passing, smoke_operator_spec.acceptance) is True
    assert (
        deeponet_is_acceptable(
            replace(passing, worst_relative_l2=0.11),
            smoke_operator_spec.acceptance,
        )
        is False
    )


def test_solver_gate_uses_p95_and_boundary(smoke_operator_spec, small_heat_split) -> None:
    assert (
        solver_is_acceptable(
            small_heat_split.train,
            smoke_operator_spec.solver_acceptance,
        )
        is True
    )


def test_metrics_reject_shape_mismatch_and_nonfinite_values() -> None:
    reference = np.ones((2, 3, 4))

    with pytest.raises(ValueError, match="形状"):
        compute_field_metrics(reference, np.ones((2, 3, 5)), target_std=1.0)
    prediction = reference.copy()
    prediction[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="有限"):
        compute_field_metrics(reference, prediction, target_std=1.0)
    with pytest.raises(ValueError, match="target_std"):
        compute_field_metrics(reference, reference, target_std=0.0)
