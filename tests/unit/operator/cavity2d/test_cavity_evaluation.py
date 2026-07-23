import numpy as np

from surrogate_loop.operator.cavity2d.evaluation import (
    _vortex_centers,
    cavity_is_acceptable,
    compute_cavity_metrics,
)


def sample_fields() -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    fields = np.array(
        [
            [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, -1.0, 1.0]],
            [[0.0, 0.0, -2.0], [2.0, 0.0, 0.0], [0.0, -2.0, 0.0], [2.0, -2.0, 2.0]],
        ]
    )
    return coordinates, fields


def test_pressure_metric_ignores_constant_offset() -> None:
    coordinates, reference = sample_fields()
    prediction = reference.copy()
    prediction[:, :, 2] += 50.0

    metrics = compute_cavity_metrics(coordinates, reference, prediction)

    assert metrics.pressure_median_relative_l2 == 0.0
    assert metrics.velocity_median_relative_l2 == 0.0
    assert metrics.predictions_finite is True
    assert metrics.vortex_center_p95_error == 0.0


def test_full_acceptance_uses_all_fixed_hard_gates() -> None:
    coordinates, reference = sample_fields()
    accepted = compute_cavity_metrics(coordinates, reference, reference.copy())
    prediction = reference.copy()
    prediction[0, :, :2] *= 2.0
    rejected = compute_cavity_metrics(coordinates, reference, prediction)

    assert cavity_is_acceptable(accepted, cpu_speedup=100.0) is True
    assert cavity_is_acceptable(accepted, cpu_speedup=99.9) is False
    assert cavity_is_acceptable(rejected, cpu_speedup=1000.0) is False


def test_nonfinite_prediction_is_rejected() -> None:
    coordinates, reference = sample_fields()
    prediction = reference.copy()
    prediction[0, 0, 0] = np.nan

    metrics = compute_cavity_metrics(coordinates, reference, prediction)

    assert metrics.predictions_finite is False
    assert cavity_is_acceptable(metrics, cpu_speedup=1000.0) is False


def test_primary_vortex_center_is_recovered_on_scattered_cell_centers() -> None:
    axis = np.linspace(0.02, 0.98, 21)
    x, y = np.meshgrid(axis, axis)
    coordinates = np.column_stack((x.ravel(), y.ravel()))
    rng = np.random.default_rng(20260723)
    coordinates += rng.uniform(-0.005, 0.005, size=coordinates.shape)
    x_coord = coordinates[:, 0]
    y_coord = coordinates[:, 1]
    velocity = np.column_stack(
        (
            -np.pi * np.sin(np.pi * x_coord) * np.cos(np.pi * y_coord),
            np.pi * np.cos(np.pi * x_coord) * np.sin(np.pi * y_coord),
        )
    )[None, ...]
    velocity[0, 0] = 0.0

    center = _vortex_centers(coordinates, velocity)[0]

    assert np.linalg.norm(center - np.array([0.5, 0.5])) <= 0.05
