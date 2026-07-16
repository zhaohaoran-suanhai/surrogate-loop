from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp


def _validate_finite(gamma: float, time: float) -> None:
    if not math.isfinite(gamma) or not math.isfinite(time):
        raise ValueError("gamma 和 time 必须是有限数值")


def analytical_solution(gamma: float, time: float) -> float:
    _validate_finite(gamma, time)
    x = gamma * time
    if abs(x) < 1e-4:
        return float(
            time**2 / 4
            + gamma * time**3 / 12
            + gamma**2 * time**4 / 48
            + gamma**3 * time**5 / 240
        )
    return float((np.expm1(x) - x) / (2 * gamma**2))


def numerical_endpoint(gamma: float, time_end: float = 1.0) -> float:
    _validate_finite(gamma, time_end)

    def right_hand_side(time: float, state: np.ndarray) -> np.ndarray:
        return np.array([gamma * state[0] + 0.5 * time])

    result = solve_ivp(
        right_hand_side,
        (0.0, time_end),
        np.array([0.0]),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    if not result.success or result.y.size == 0:
        raise RuntimeError(f"ODE 数值求解失败：{result.message}")
    value = float(result.y[0, -1])
    if not math.isfinite(value):
        raise RuntimeError("ODE 数值求解产生非有限结果")
    return value
