from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HeatParameters:
    alpha: float
    amplitude_1: float
    amplitude_2: float


def make_grid(nx: int, nt: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if nx < 3:
        raise ValueError("空间网格至少需要三个节点")
    if nt < 2:
        raise ValueError("时间网格至少需要两个节点")
    return np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, nt)


def validate_parameters(parameters: HeatParameters) -> None:
    values = (parameters.alpha, parameters.amplitude_1, parameters.amplitude_2)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("热传导参数必须有限")
    if parameters.alpha <= 0.0:
        raise ValueError("alpha 必须大于零")


def validate_grid(
    x: NDArray[np.float64], t: NDArray[np.float64]
) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("空间网格必须是一维且至少包含三个节点")
    if t.ndim != 1 or t.size < 2:
        raise ValueError("时间网格必须是一维且至少包含两个节点")
    if not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError("空间和时间网格必须有限")
    dx_values = np.diff(x)
    dt_values = np.diff(t)
    if np.any(dx_values <= 0.0) or np.any(dt_values <= 0.0):
        raise ValueError("空间和时间网格必须严格递增")
    if not np.allclose(dx_values, dx_values[0], rtol=1e-12, atol=1e-14):
        raise ValueError("空间网格必须均匀")
    if not np.allclose(dt_values, dt_values[0], rtol=1e-12, atol=1e-14):
        raise ValueError("时间网格必须均匀")
    if not np.isclose(x[0], 0.0) or not np.isclose(x[-1], 1.0):
        raise ValueError("空间网格必须覆盖 [0, 1]")
    if not np.isclose(t[0], 0.0) or not np.isclose(t[-1], 1.0):
        raise ValueError("时间网格必须覆盖 [0, 1]")
    return float(dx_values[0]), float(dt_values[0])
