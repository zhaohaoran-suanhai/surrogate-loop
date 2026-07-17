from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from surrogate_loop.operator.elasticity2d.config import load_elasticity_spec
from surrogate_loop.operator.elasticity2d.evaluation import (
    compute_elasticity_metrics,
    elasticity_is_acceptable,
)
from surrogate_loop.operator.elasticity2d.problem import (
    elasticity_basis_features,
    elasticity_features,
    traction_density,
)

ROOT = Path(__file__).resolve().parents[4]


def make_field_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.linspace(0.0, 1.0, 101)
    coordinates = np.concatenate(
        (
            np.column_stack((np.zeros_like(y), y)),
            np.column_stack((np.full_like(y, 4.0), y)),
        )
    )
    parameters = np.array(
        [
            [2.0, 0.3, 0.005, 0.0, 0.5, 0.1],
            [3.0, 0.35, 0.008, np.pi / 2, 0.4, 0.12],
        ]
    )
    fields = np.zeros((2, coordinates.shape[0], 2), dtype=np.float64)
    right = coordinates[:, 0] == 4.0
    fields[0, right, 0] = 0.01 * (1.0 + coordinates[right, 1])
    fields[0, right, 1] = 0.002 * coordinates[right, 1]
    fields[1, right, 0] = -0.001 * coordinates[right, 1]
    fields[1, right, 1] = 0.02 * (1.0 + coordinates[right, 1])
    return parameters, coordinates, fields


def test_angle_features_are_periodic() -> None:
    left = np.array([[1.0, 0.3, 0.01, -np.pi, 0.5, 0.1]])
    right = np.array([[1.0, 0.3, 0.01, np.pi, 0.5, 0.1]])

    np.testing.assert_allclose(
        elasticity_features(left), elasticity_features(right), atol=1e-15
    )
    assert elasticity_features(left).shape == (1, 5)


def test_elasticity_basis_features_exclude_load_direction_and_scale() -> None:
    first = np.array([[2.0, 0.31, 0.004, 0.2, 0.45, 0.11]])
    second = np.array([[5.0, 0.31, 0.009, -2.4, 0.45, 0.11]])

    np.testing.assert_allclose(
        elasticity_basis_features(first),
        elasticity_basis_features(second),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        elasticity_basis_features(first),
        np.array([[0.31, 0.45, 0.11]]),
    )


@pytest.mark.parametrize(("center", "width"), [(0.2, 0.08), (0.5, 0.12), (0.8, 0.2)])
def test_traction_density_integrates_to_one(center: float, width: float) -> None:
    y = np.linspace(0.0, 1.0, 10001)

    density = traction_density(y, y0=center, width=width)

    np.testing.assert_allclose(np.trapezoid(density, y), 1.0, rtol=2e-8)
    assert np.all(density > 0.0)


def test_exact_vector_field_has_zero_errors() -> None:
    parameters, coordinates, fields = make_field_inputs()

    metrics = compute_elasticity_metrics(fields, fields.copy(), parameters, coordinates)

    assert metrics.median_relative_l2 == 0.0
    assert metrics.p95_relative_l2 == 0.0
    assert metrics.worst_relative_l2 == 0.0
    assert metrics.p95_tip_error == 0.0
    assert metrics.p95_compliance_error == 0.0
    assert metrics.clamp_max_absolute_error == 0.0


def test_uniform_field_scaling_has_matching_relative_metrics() -> None:
    parameters, coordinates, fields = make_field_inputs()
    prediction = fields * 1.1

    metrics = compute_elasticity_metrics(fields, prediction, parameters, coordinates)

    np.testing.assert_allclose(metrics.median_relative_l2, 0.1)
    np.testing.assert_allclose(metrics.p95_relative_l2, 0.1)
    np.testing.assert_allclose(metrics.worst_relative_l2, 0.1)
    np.testing.assert_allclose(metrics.p95_tip_error, 0.1)
    np.testing.assert_allclose(metrics.p95_compliance_error, 0.1)
    assert metrics.clamp_max_absolute_error == 0.0


def test_acceptance_requires_every_metric_and_speedup() -> None:
    spec = load_elasticity_spec(ROOT / "examples/elasticity_2d_cantilever/full.json")
    parameters, coordinates, fields = make_field_inputs()
    passing = compute_elasticity_metrics(fields, fields.copy(), parameters, coordinates)

    assert elasticity_is_acceptable(passing, spec.acceptance, speedup=100.0) is True
    assert elasticity_is_acceptable(passing, spec.acceptance, speedup=99.9) is False
    assert (
        elasticity_is_acceptable(
            replace(passing, worst_relative_l2=0.151),
            spec.acceptance,
            speedup=100.0,
        )
        is False
    )


def test_metrics_reject_nonfinite_prediction() -> None:
    parameters, coordinates, fields = make_field_inputs()
    prediction = fields.copy()
    prediction[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="有限"):
        compute_elasticity_metrics(fields, prediction, parameters, coordinates)
