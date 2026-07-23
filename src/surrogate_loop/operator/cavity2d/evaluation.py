from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata

REGULAR_GRID_SIZE = 101


@dataclass(frozen=True)
class CavityMetrics:
    velocity_median_relative_l2: float
    velocity_p95_relative_l2: float
    velocity_worst_relative_l2: float
    pressure_median_relative_l2: float
    pressure_p95_relative_l2: float
    pressure_worst_relative_l2: float
    vortex_center_p95_error: float
    centerline_velocity_relative_l2: float
    horizontal_centerline_velocity_relative_l2: float
    vertical_centerline_velocity_relative_l2: float
    predictions_finite: bool
    divergence_median_rms: float | None
    momentum_median_rms: float | None


def _relative_l2(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    error = np.linalg.norm(
        (prediction - reference).reshape(reference.shape[0], -1),
        axis=1,
    )
    denominator = np.linalg.norm(reference.reshape(reference.shape[0], -1), axis=1)
    return error / np.maximum(denominator, 1e-12)


def _summary(values: np.ndarray) -> tuple[float, float, float]:
    if not np.isfinite(values).all():
        return float("inf"), float("inf"), float("inf")
    return (
        float(np.median(values)),
        float(np.percentile(values, 95)),
        float(np.max(values)),
    )


def _grid_indices(
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    x_values = np.unique(coordinates[:, 0])
    y_values = np.unique(coordinates[:, 1])
    if x_values.size * y_values.size != coordinates.shape[0]:
        return None
    lookup = {
        (float(x), float(y)): index for index, (x, y) in enumerate(coordinates)
    }
    try:
        indices = np.asarray(
            [
                [lookup[(float(x), float(y))] for x in x_values]
                for y in y_values
            ],
            dtype=np.int64,
        )
    except KeyError:
        return None
    return x_values, y_values, indices


def _regular_fields(
    coordinates: np.ndarray,
    fields: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = _grid_indices(coordinates)
    if grid is not None:
        x_values, y_values, indices = grid
        return x_values, y_values, fields[:, indices, :]
    x_values = np.linspace(
        float(coordinates[:, 0].min()),
        float(coordinates[:, 0].max()),
        REGULAR_GRID_SIZE,
    )
    y_values = np.linspace(
        float(coordinates[:, 1].min()),
        float(coordinates[:, 1].max()),
        REGULAR_GRID_SIZE,
    )
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    regular = np.empty(
        (fields.shape[0], REGULAR_GRID_SIZE, REGULAR_GRID_SIZE, fields.shape[2]),
        dtype=np.float64,
    )
    for sample_index in range(fields.shape[0]):
        for component_index in range(fields.shape[2]):
            source = fields[sample_index, :, component_index]
            interpolated = griddata(
                coordinates,
                source,
                (x_grid, y_grid),
                method="linear",
            )
            missing = ~np.isfinite(interpolated)
            if missing.any():
                interpolated[missing] = griddata(
                    coordinates,
                    source,
                    (x_grid[missing], y_grid[missing]),
                    method="nearest",
                )
            regular[sample_index, :, :, component_index] = interpolated
    return x_values, y_values, regular


def _vortex_centers(
    coordinates: np.ndarray,
    velocity: np.ndarray,
) -> np.ndarray:
    if not np.isfinite(velocity).all():
        return np.full((velocity.shape[0], 2), np.nan, dtype=np.float64)
    x_values, y_values, regular = _regular_fields(coordinates, velocity)
    u = regular[:, :, :, 0]
    psi = np.zeros_like(u)
    for y_index in range(1, y_values.size):
        delta_y = y_values[y_index] - y_values[y_index - 1]
        psi[:, y_index, :] = (
            psi[:, y_index - 1, :]
            + 0.5 * (u[:, y_index - 1, :] + u[:, y_index, :]) * delta_y
        )
    if x_values.size > 2 and y_values.size > 2:
        interior = psi[:, 1:-1, 1:-1]
        flat = np.argmin(interior.reshape(interior.shape[0], -1), axis=1)
        y_index, x_index = np.unravel_index(
            flat,
            (y_values.size - 2, x_values.size - 2),
        )
        x_index = x_index + 1
        y_index = y_index + 1
    else:
        flat = np.argmin(psi.reshape(psi.shape[0], -1), axis=1)
        y_index, x_index = np.unravel_index(
            flat,
            (y_values.size, x_values.size),
        )
    return np.column_stack((x_values[x_index], y_values[y_index]))


def _physics_diagnostics(
    coordinates: np.ndarray,
    fields: np.ndarray,
    reynolds: np.ndarray | None,
) -> tuple[float | None, float | None]:
    divergence_rms, momentum_rms = _physics_diagnostic_values(
        coordinates,
        fields,
        reynolds,
    )
    return (
        float(np.median(divergence_rms)),
        float(np.median(momentum_rms)) if momentum_rms is not None else None,
    )


def _physics_diagnostic_values(
    coordinates: np.ndarray,
    fields: np.ndarray,
    reynolds: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    x_values, y_values, regular = _regular_fields(coordinates, fields)
    if x_values.size < 2 or y_values.size < 2:
        return np.full(fields.shape[0], np.nan), None
    u = regular[:, :, :, 0]
    v = regular[:, :, :, 1]
    pressure = regular[:, :, :, 2]
    du_dx = np.gradient(u, x_values, axis=2)
    dv_dy = np.gradient(v, y_values, axis=1)
    divergence = du_dx + dv_dy
    divergence_rms = np.sqrt(np.mean(np.square(divergence), axis=(1, 2)))
    momentum_values: np.ndarray | None = None
    if reynolds is not None:
        reynolds = np.asarray(reynolds, dtype=np.float64)
        if reynolds.shape == (fields.shape[0],):
            du_dy = np.gradient(u, y_values, axis=1)
            dv_dx = np.gradient(v, x_values, axis=2)
            dp_dx = np.gradient(pressure, x_values, axis=2)
            dp_dy = np.gradient(pressure, y_values, axis=1)
            laplace_u = np.gradient(du_dx, x_values, axis=2) + np.gradient(
                du_dy,
                y_values,
                axis=1,
            )
            laplace_v = np.gradient(dv_dx, x_values, axis=2) + np.gradient(
                dv_dy,
                y_values,
                axis=1,
            )
            inverse_re = 1.0 / reynolds[:, None, None]
            residual_u = u * du_dx + v * du_dy + dp_dx - inverse_re * laplace_u
            residual_v = u * dv_dx + v * dv_dy + dp_dy - inverse_re * laplace_v
            residual = np.sqrt(
                np.mean(np.square(residual_u) + np.square(residual_v), axis=(1, 2))
            )
            momentum_values = residual
    return divergence_rms, momentum_values


def _centerline_fields(
    coordinates: np.ndarray,
    fields: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_values, y_values, regular = _regular_fields(coordinates, fields[:, :, :2])
    x_index = int(np.argmin(np.abs(x_values - 0.5)))
    y_index = int(np.argmin(np.abs(y_values - 0.5)))
    vertical = regular[:, :, x_index, :]
    horizontal = regular[:, y_index, :, :]
    return horizontal, vertical


def compute_cavity_metrics(
    coordinates: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    reynolds: np.ndarray | None = None,
) -> CavityMetrics:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or reference.ndim != 3
        or reference.shape != prediction.shape
        or reference.shape[1:] != (coordinates.shape[0], 3)
    ):
        raise ValueError("cavity metric arrays have incompatible shapes")
    finite = bool(np.isfinite(prediction).all())
    velocity_errors = _relative_l2(reference[:, :, :2], prediction[:, :, :2])
    reference_pressure = reference[:, :, 2] - reference[:, :, 2].mean(
        axis=1,
        keepdims=True,
    )
    predicted_pressure = prediction[:, :, 2] - prediction[:, :, 2].mean(
        axis=1,
        keepdims=True,
    )
    pressure_errors = _relative_l2(
        reference_pressure[:, :, None],
        predicted_pressure[:, :, None],
    )
    velocity_summary = _summary(velocity_errors)
    pressure_summary = _summary(pressure_errors)
    reference_centers = _vortex_centers(coordinates, reference[:, :, :2])
    predicted_centers = _vortex_centers(coordinates, prediction[:, :, :2])
    vortex_errors = np.linalg.norm(predicted_centers - reference_centers, axis=1)
    vortex_p95 = (
        float(np.percentile(vortex_errors, 95))
        if np.isfinite(vortex_errors).all()
        else float("inf")
    )
    reference_horizontal, reference_vertical = _centerline_fields(
        coordinates,
        reference,
    )
    predicted_horizontal, predicted_vertical = _centerline_fields(
        coordinates,
        prediction,
    )
    horizontal_centerline = _relative_l2(
        reference_horizontal,
        predicted_horizontal,
    )
    vertical_centerline = _relative_l2(
        reference_vertical,
        predicted_vertical,
    )
    centerline = _relative_l2(
        np.concatenate((reference_horizontal, reference_vertical), axis=1),
        np.concatenate((predicted_horizontal, predicted_vertical), axis=1),
    )
    centerline_value = (
        float(np.median(centerline))
        if np.isfinite(centerline).all()
        else float("inf")
    )
    divergence, momentum = (
        _physics_diagnostics(coordinates, prediction, reynolds)
        if finite
        else (None, None)
    )
    return CavityMetrics(
        velocity_median_relative_l2=velocity_summary[0],
        velocity_p95_relative_l2=velocity_summary[1],
        velocity_worst_relative_l2=velocity_summary[2],
        pressure_median_relative_l2=pressure_summary[0],
        pressure_p95_relative_l2=pressure_summary[1],
        pressure_worst_relative_l2=pressure_summary[2],
        vortex_center_p95_error=vortex_p95,
        centerline_velocity_relative_l2=centerline_value,
        horizontal_centerline_velocity_relative_l2=(
            float(np.median(horizontal_centerline))
            if np.isfinite(horizontal_centerline).all()
            else float("inf")
        ),
        vertical_centerline_velocity_relative_l2=(
            float(np.median(vertical_centerline))
            if np.isfinite(vertical_centerline).all()
            else float("inf")
        ),
        predictions_finite=finite,
        divergence_median_rms=divergence,
        momentum_median_rms=momentum,
    )


def cavity_is_acceptable(metrics: CavityMetrics, cpu_speedup: float) -> bool:
    return all(
        (
            metrics.velocity_median_relative_l2 <= 0.02,
            metrics.velocity_p95_relative_l2 <= 0.05,
            metrics.velocity_worst_relative_l2 <= 0.10,
            metrics.pressure_median_relative_l2 <= 0.05,
            metrics.pressure_p95_relative_l2 <= 0.10,
            metrics.pressure_worst_relative_l2 <= 0.20,
            metrics.vortex_center_p95_error <= 0.05,
            metrics.predictions_finite,
            cpu_speedup >= 100.0,
        )
    )


__all__ = ["CavityMetrics", "cavity_is_acceptable", "compute_cavity_metrics"]
