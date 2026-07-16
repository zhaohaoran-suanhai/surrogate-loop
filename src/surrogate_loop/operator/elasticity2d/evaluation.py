from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from surrogate_loop.operator.elasticity2d.config import ElasticityAcceptanceSpec
from surrogate_loop.operator.elasticity2d.problem import (
    HEIGHT,
    LENGTH,
    traction_density,
    validate_parameter_array,
)


@dataclass(frozen=True)
class ElasticityMetrics:
    median_relative_l2: float
    p95_relative_l2: float
    worst_relative_l2: float
    p95_tip_error: float
    p95_compliance_error: float
    clamp_max_absolute_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_elasticity_metrics(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    parameters: NDArray[np.float64],
    coordinates: NDArray[np.float64],
) -> ElasticityMetrics:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    parameters = validate_parameter_array(parameters)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    expected_shape = (parameters.shape[0], coordinates.shape[0], 2)
    if reference.shape != expected_shape or prediction.shape != expected_shape:
        raise ValueError(f"位移场形状必须为 {expected_shape}")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("观测坐标形状必须为 (n_points, 2)")
    if not (
        np.isfinite(reference).all()
        and np.isfinite(prediction).all()
        and np.isfinite(coordinates).all()
    ):
        raise ValueError("参考场、预测场和坐标必须全部有限")
    difference = prediction - reference
    flat_reference = reference.reshape(reference.shape[0], -1)
    flat_difference = difference.reshape(difference.shape[0], -1)
    reference_norm = np.maximum(np.linalg.norm(flat_reference, axis=1), 1e-12)
    relative_l2 = np.linalg.norm(flat_difference, axis=1) / reference_norm

    right_indices, right_y = _boundary_indices(coordinates, LENGTH)
    reference_right = reference[:, right_indices, :]
    prediction_right = prediction[:, right_indices, :]
    reference_tip = np.trapezoid(reference_right, right_y, axis=1) / HEIGHT
    prediction_tip = np.trapezoid(prediction_right, right_y, axis=1) / HEIGHT
    tip_denominator = np.maximum(np.linalg.norm(reference_tip, axis=1), 1e-12)
    tip_error = np.linalg.norm(prediction_tip - reference_tip, axis=1) / tip_denominator

    reference_compliance = _compliance(reference_right, right_y, parameters)
    prediction_compliance = _compliance(prediction_right, right_y, parameters)
    compliance_error = np.abs(prediction_compliance - reference_compliance) / np.maximum(
        np.abs(reference_compliance), 1e-12
    )

    left_indices, _ = _boundary_indices(coordinates, 0.0)
    clamp_error = float(np.max(np.abs(prediction[:, left_indices, :])))
    return ElasticityMetrics(
        median_relative_l2=float(np.median(relative_l2)),
        p95_relative_l2=float(np.quantile(relative_l2, 0.95)),
        worst_relative_l2=float(np.max(relative_l2)),
        p95_tip_error=float(np.quantile(tip_error, 0.95)),
        p95_compliance_error=float(np.quantile(compliance_error, 0.95)),
        clamp_max_absolute_error=clamp_error,
    )


def elasticity_is_acceptable(
    metrics: ElasticityMetrics,
    acceptance: ElasticityAcceptanceSpec,
    speedup: float,
) -> bool:
    values = np.fromiter(metrics.to_dict().values(), dtype=np.float64)
    return bool(
        np.isfinite(values).all()
        and np.isfinite(speedup)
        and metrics.median_relative_l2 <= acceptance.max_median_relative_l2
        and metrics.p95_relative_l2 <= acceptance.max_p95_relative_l2
        and metrics.worst_relative_l2 <= acceptance.max_worst_relative_l2
        and metrics.p95_tip_error <= acceptance.max_p95_tip_error
        and metrics.p95_compliance_error <= acceptance.max_p95_compliance_error
        and metrics.clamp_max_absolute_error <= acceptance.max_clamp_absolute_error
        and speedup >= acceptance.min_cpu_speedup
    )


def _boundary_indices(
    coordinates: NDArray[np.float64], x_value: float
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    indices = np.flatnonzero(
        np.isclose(coordinates[:, 0], x_value, rtol=0.0, atol=1e-12)
    )
    if indices.size < 2:
        raise ValueError(f"x={x_value:g} 边界至少需要两个观测点")
    order = np.argsort(coordinates[indices, 1])
    indices = indices[order]
    y = coordinates[indices, 1]
    if np.any(np.diff(y) <= 0.0) or not np.isclose(y[0], 0.0) or not np.isclose(
        y[-1], HEIGHT
    ):
        raise ValueError(f"x={x_value:g} 边界观测点必须严格覆盖 [0, 1]")
    return indices, y


def _compliance(
    right_displacement: NDArray[np.float64],
    y: NDArray[np.float64],
    parameters: NDArray[np.float64],
) -> NDArray[np.float64]:
    values = np.empty(parameters.shape[0], dtype=np.float64)
    for index, row in enumerate(parameters):
        magnitude = row[2]
        angle = row[3]
        density = traction_density(y, y0=float(row[4]), width=float(row[5]))
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        traction = magnitude * density[:, None] * direction[None, :]
        integrand = np.sum(traction * right_displacement[index], axis=1)
        values[index] = np.trapezoid(integrand, y)
    return values
