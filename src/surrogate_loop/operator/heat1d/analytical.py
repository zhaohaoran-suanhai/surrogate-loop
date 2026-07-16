from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from surrogate_loop.operator.heat1d.problem import (
    HeatParameters,
    validate_grid,
    validate_parameters,
)


def analytical_solution(
    parameters: HeatParameters,
    x: NDArray[np.float64],
    t: NDArray[np.float64],
) -> NDArray[np.float64]:
    validate_parameters(parameters)
    validate_grid(x, t)
    xx = np.asarray(x, dtype=np.float64)[None, :]
    tt = np.asarray(t, dtype=np.float64)[:, None]
    first_mode = (
        parameters.amplitude_1
        * np.exp(-parameters.alpha * np.pi**2 * tt)
        * np.sin(np.pi * xx)
    )
    second_mode = (
        parameters.amplitude_2
        * np.exp(-4.0 * parameters.alpha * np.pi**2 * tt)
        * np.sin(2.0 * np.pi * xx)
    )
    return first_mode + second_mode
