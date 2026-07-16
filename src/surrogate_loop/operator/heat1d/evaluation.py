from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from surrogate_loop.operator.config import (
    OperatorAcceptanceSpec,
    SolverAcceptanceSpec,
)
from surrogate_loop.operator.heat1d.dataset import HeatDataset


@dataclass(frozen=True)
class FieldMetrics:
    median_relative_l2: float
    p95_relative_l2: float
    worst_relative_l2: float
    normalized_rmse: float
    initial_relative_l2: float
    boundary_max_absolute_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_field_metrics(
    reference: NDArray[np.float64],
    prediction: NDArray[np.float64],
    target_std: float = 1.0,
) -> FieldMetrics:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if reference.shape != prediction.shape or reference.ndim != 3:
        raise ValueError("参考场与预测场形状必须一致且为 (n_cases, nt, nx)")
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError("参考场与预测场必须全部有限")
    if not np.isfinite(target_std) or target_std <= 0.0:
        raise ValueError("target_std 必须是有限正数")
    difference = prediction - reference
    flat_difference = difference.reshape(reference.shape[0], -1)
    flat_reference = reference.reshape(reference.shape[0], -1)
    denominators = np.maximum(np.linalg.norm(flat_reference, axis=1), 1e-12)
    relative_l2 = np.linalg.norm(flat_difference, axis=1) / denominators
    initial_denominator = max(float(np.linalg.norm(reference[:, 0, :])), 1e-12)
    initial_relative_l2 = float(
        np.linalg.norm(difference[:, 0, :]) / initial_denominator
    )
    return FieldMetrics(
        median_relative_l2=float(np.median(relative_l2)),
        p95_relative_l2=float(np.quantile(relative_l2, 0.95)),
        worst_relative_l2=float(np.max(relative_l2)),
        normalized_rmse=float(np.sqrt(np.mean(difference**2)) / target_std),
        initial_relative_l2=initial_relative_l2,
        boundary_max_absolute_error=float(
            np.max(np.abs(difference[:, :, [0, -1]]))
        ),
    )


def solver_is_acceptable(
    dataset: HeatDataset, acceptance: SolverAcceptanceSpec
) -> bool:
    boundary_error = float(np.max(np.abs(dataset.fields[:, :, [0, -1]])))
    p95_relative_l2 = float(np.quantile(dataset.solver_relative_l2, 0.95))
    return (
        np.isfinite(boundary_error)
        and np.isfinite(p95_relative_l2)
        and boundary_error <= acceptance.max_boundary_error
        and p95_relative_l2 <= acceptance.max_p95_relative_l2
    )


def deeponet_is_acceptable(
    metrics: FieldMetrics, acceptance: OperatorAcceptanceSpec
) -> bool:
    values = np.fromiter(metrics.to_dict().values(), dtype=np.float64)
    return bool(
        np.isfinite(values).all()
        and metrics.median_relative_l2 <= acceptance.max_median_relative_l2
        and metrics.p95_relative_l2 <= acceptance.max_p95_relative_l2
        and metrics.worst_relative_l2 <= acceptance.max_worst_relative_l2
        and metrics.initial_relative_l2 <= acceptance.max_initial_relative_l2
        and metrics.boundary_max_absolute_error
        <= acceptance.max_boundary_absolute_error
    )
