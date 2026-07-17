from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

LENGTH = 4.0
HEIGHT = 1.0
PARAMETER_DIM = 6


def elasticity_basis_features(
    parameters: NDArray[np.float64],
) -> NDArray[np.float64]:
    values = validate_parameter_array(parameters)
    return values[:, [1, 4, 5]].copy()


def elasticity_features(
    parameters: NDArray[np.float64],
) -> NDArray[np.float64]:
    values = validate_parameter_array(parameters)
    return np.column_stack(
        (
            values[:, 1],
            np.cos(values[:, 3]),
            np.sin(values[:, 3]),
            values[:, 4],
            values[:, 5],
        )
    )


def validate_parameter_array(
    parameters: NDArray[np.float64],
) -> NDArray[np.float64]:
    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != PARAMETER_DIM or values.shape[0] == 0:
        raise ValueError("二维弹性参数形状必须为 (n_samples, 6)")
    if not np.isfinite(values).all():
        raise ValueError("二维弹性参数必须全部有限")
    young_modulus = values[:, 0]
    poisson_ratio = values[:, 1]
    load_magnitude = values[:, 2]
    load_center = values[:, 4]
    load_width = values[:, 5]
    if np.any(young_modulus <= 0.0):
        raise ValueError("弹性模量必须为正数")
    if np.any((poisson_ratio <= 0.0) | (poisson_ratio >= 0.5)):
        raise ValueError("泊松比必须位于 (0, 0.5)")
    if np.any(load_magnitude <= 0.0):
        raise ValueError("载荷大小必须为正数")
    if np.any((load_center < 0.0) | (load_center > HEIGHT)):
        raise ValueError("载荷中心必须位于右边界范围内")
    if np.any(load_width <= 0.0):
        raise ValueError("载荷宽度必须为正数")
    if np.any(load_magnitude / young_modulus > 1e-2 + 1e-15):
        raise ValueError("载荷与弹性模量之比超过小变形合同")
    return values


def traction_density(
    y: NDArray[np.float64],
    *,
    y0: float,
    width: float,
    height: float = HEIGHT,
) -> NDArray[np.float64]:
    coordinates = np.asarray(y, dtype=np.float64)
    scalars = (y0, width, height)
    if not np.isfinite(coordinates).all() or not all(math.isfinite(v) for v in scalars):
        raise ValueError("载荷坐标和参数必须有限")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("载荷宽度和边界高度必须为正数")
    if y0 < 0.0 or y0 > height:
        raise ValueError("载荷中心必须位于右边界范围内")
    root_two = math.sqrt(2.0)
    normalizer = width * math.sqrt(math.pi / 2.0) * (
        math.erf((height - y0) / (root_two * width))
        - math.erf(-y0 / (root_two * width))
    )
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("局部载荷归一化常数无效")
    return np.exp(-0.5 * np.square((coordinates - y0) / width)) / normalizer
