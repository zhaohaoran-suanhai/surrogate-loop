from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning

from surrogate_loop.config import AcceptanceSpec
from surrogate_loop.split import DatasetSplit


@dataclass(frozen=True)
class Metrics:
    rmse: float
    nrmse: float
    mae: float
    max_absolute_error: float

    def to_dict(self) -> dict[str, float]:
        return {
            "rmse": self.rmse,
            "nrmse": self.nrmse,
            "mae": self.mae,
            "max_absolute_error": self.max_absolute_error,
        }


@dataclass(frozen=True)
class SelectionResult:
    selected_name: str
    selected_model: Any
    validation_metrics: dict[str, Metrics]
    test_metrics: Metrics
    accepted: bool
    failures: dict[str, str]


def compute_metrics(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> Metrics:
    error = np.asarray(y_pred) - np.asarray(y_true)
    rmse = float(np.sqrt(np.mean(error**2)))
    target_range = float(np.max(y_true) - np.min(y_true))
    if target_range <= 0:
        raise ValueError("NRMSE 的目标范围必须大于零")
    return Metrics(
        rmse=rmse,
        nrmse=rmse / target_range,
        mae=float(np.mean(np.abs(error))),
        max_absolute_error=float(np.max(np.abs(error))),
    )


def train_select_and_test(
    split: DatasetSplit,
    candidates: dict[str, Any],
    acceptance: AcceptanceSpec,
) -> SelectionResult:
    validation_metrics: dict[str, Metrics] = {}
    failures: dict[str, str] = {}
    fitted: dict[str, Any] = {}
    for name, model in candidates.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(split.train_x, split.train_y)
            prediction = model.predict(split.validation_x)
            validation_metrics[name] = compute_metrics(split.validation_y, prediction)
            fitted[name] = model
        except Exception as error:  # noqa: BLE001 - 一个候选失败不能中止其余候选
            failures[name] = f"{type(error).__name__}: {error}"
    if not fitted:
        raise RuntimeError(f"所有候选模型训练失败：{failures}")

    order = {name: index for index, name in enumerate(candidates)}
    selected_name = min(
        fitted,
        key=lambda name: (validation_metrics[name].nrmse, order[name]),
    )
    selected_model = fitted[selected_name]
    test_metrics = compute_metrics(split.test_y, selected_model.predict(split.test_x))
    accepted = (
        test_metrics.nrmse <= acceptance.max_nrmse
        and test_metrics.max_absolute_error <= acceptance.max_absolute_error
    )
    return SelectionResult(
        selected_name=selected_name,
        selected_model=selected_model,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        accepted=accepted,
        failures=failures,
    )
