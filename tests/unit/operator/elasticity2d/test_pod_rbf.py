from __future__ import annotations

import numpy as np
import pytest

from surrogate_loop.operator.elasticity2d.pod_rbf import PodRbfBaseline


def _synthetic_training_data() -> tuple[np.ndarray, np.ndarray]:
    sample_count = 12
    index = np.arange(sample_count, dtype=np.float64)
    young_modulus = 100.0 + 7.0 * index
    poisson_ratio = 0.20 + 0.01 * (index % 8)
    load_ratio = 5.0e-4 + 2.0e-5 * index
    load_magnitude = young_modulus * load_ratio
    angle = -0.45 + 0.08 * index
    load_center = 0.15 + 0.06 * (index % 10)
    load_width = 0.08 + 0.01 * ((3.0 * index) % 7)
    parameters = np.column_stack(
        (young_modulus, poisson_ratio, load_magnitude, angle, load_center, load_width)
    )

    point_index = np.arange(9, dtype=np.float64)
    basis_x = np.column_stack(
        (
            1.0 + 0.1 * point_index,
            np.sin(0.3 * point_index),
            np.cos(0.2 * point_index),
        )
    )
    basis_y = np.column_stack(
        (
            0.5 + 0.08 * point_index,
            np.cos(0.25 * point_index),
            np.sin(0.35 * point_index),
        )
    )
    shape_coefficients = np.column_stack(
        (
            1.0 + poisson_ratio + 0.2 * load_center,
            np.cos(angle) + load_width,
            np.sin(angle) + poisson_ratio * load_center,
        )
    )
    shape_fields = np.stack(
        (shape_coefficients @ basis_x.T, shape_coefficients @ basis_y.T), axis=-1
    )
    fields = shape_fields * load_ratio[:, None, None]
    return parameters, fields


def test_pod_rbf_reconstructs_training_fields_and_reports_summary() -> None:
    parameters, fields = _synthetic_training_data()

    baseline = PodRbfBaseline(energy_threshold=1.0, max_components=12).fit(
        parameters, fields
    )

    prediction = baseline.predict(parameters)
    assert prediction.shape == fields.shape
    np.testing.assert_allclose(prediction, fields, rtol=1e-8, atol=1e-11)
    assert baseline.summary() == {
        "components": 3,
        "explained_energy": pytest.approx(1.0),
        "energy_threshold": 1.0,
        "max_components": 12,
        "training_samples": 12,
        "field_shape": [9, 2],
    }


def test_pod_rbf_preserves_exact_load_modulus_scaling() -> None:
    parameters, fields = _synthetic_training_data()
    baseline = PodRbfBaseline(energy_threshold=0.999, max_components=8).fit(
        parameters, fields
    )
    query = parameters[:1].copy()
    doubled_load = query.copy()
    doubled_load[:, 2] *= 2.0
    doubled_modulus = query.copy()
    doubled_modulus[:, 0] *= 2.0

    original = baseline.predict(query)

    np.testing.assert_allclose(baseline.predict(doubled_load), 2.0 * original, rtol=1e-10)
    np.testing.assert_allclose(
        baseline.predict(doubled_modulus), 0.5 * original, rtol=1e-10
    )


def test_pod_rbf_rejects_invalid_usage() -> None:
    parameters, fields = _synthetic_training_data()
    baseline = PodRbfBaseline(energy_threshold=0.999, max_components=8)

    with pytest.raises(RuntimeError, match="尚未拟合"):
        baseline.predict(parameters[:1])
    with pytest.raises(ValueError, match="字段形状"):
        baseline.fit(parameters, fields[..., 0])
    with pytest.raises(ValueError, match="互异"):
        baseline.fit(np.repeat(parameters[:1], 2, axis=0), np.repeat(fields[:1], 2, axis=0))


@pytest.mark.parametrize(
    ("energy_threshold", "max_components"),
    [(0.0, 8), (1.01, 8), (0.999, 0)],
)
def test_pod_rbf_rejects_invalid_configuration(
    energy_threshold: float, max_components: int
) -> None:
    with pytest.raises(ValueError):
        PodRbfBaseline(energy_threshold=energy_threshold, max_components=max_components)
