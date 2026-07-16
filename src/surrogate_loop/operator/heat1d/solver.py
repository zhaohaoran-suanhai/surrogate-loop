from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded

from surrogate_loop.operator.heat1d.problem import (
    HeatParameters,
    validate_grid,
    validate_parameters,
)


def solve_case(
    parameters: HeatParameters,
    x: NDArray[np.float64],
    t: NDArray[np.float64],
) -> NDArray[np.float64]:
    validate_parameters(parameters)
    dx, dt = validate_grid(x, t)
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    field = np.zeros((t.size, x.size), dtype=np.float64)
    field[0, 1:-1] = (
        parameters.amplitude_1 * np.sin(np.pi * x[1:-1])
        + parameters.amplitude_2 * np.sin(2.0 * np.pi * x[1:-1])
    )

    ratio = parameters.alpha * dt / dx**2
    interior_size = x.size - 2
    left_matrix = np.zeros((3, interior_size), dtype=np.float64)
    left_matrix[0, 1:] = -0.5 * ratio
    left_matrix[1, :] = 1.0 + ratio
    left_matrix[2, :-1] = -0.5 * ratio

    for step in range(t.size - 1):
        current = field[step]
        right_hand_side = (
            (1.0 - ratio) * current[1:-1]
            + 0.5 * ratio * (current[:-2] + current[2:])
        )
        field[step + 1, 1:-1] = solve_banded(
            (1, 1),
            left_matrix,
            right_hand_side,
            overwrite_ab=False,
            overwrite_b=False,
            check_finite=False,
        )

    if not np.isfinite(field).all():
        raise FloatingPointError("Crank–Nicolson 求解产生非有限值")
    return field
