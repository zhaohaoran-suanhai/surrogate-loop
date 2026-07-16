import numpy as np
import pytest

from surrogate_loop.operator.heat1d.analytical import analytical_solution
from surrogate_loop.operator.heat1d.problem import HeatParameters, make_grid
from surrogate_loop.operator.heat1d.solver import solve_case


def _relative_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def test_analytical_solution_satisfies_initial_and_boundary_conditions() -> None:
    parameters = HeatParameters(alpha=0.1, amplitude_1=1.1, amplitude_2=-0.2)
    x, t = make_grid(nx=65, nt=51)

    field = analytical_solution(parameters, x, t)
    expected_initial = 1.1 * np.sin(np.pi * x) - 0.2 * np.sin(2.0 * np.pi * x)

    assert field.shape == (51, 65)
    np.testing.assert_allclose(field[0], expected_initial, atol=1e-14)
    np.testing.assert_allclose(field[:, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(field[:, -1], 0.0, atol=1e-14)


def test_crank_nicolson_converges_toward_analytical_solution() -> None:
    parameters = HeatParameters(alpha=0.2, amplitude_1=0.8, amplitude_2=0.3)
    coarse_x, coarse_t = make_grid(nx=65, nt=51)
    fine_x, fine_t = make_grid(nx=129, nt=101)

    coarse_error = _relative_l2(
        analytical_solution(parameters, coarse_x, coarse_t),
        solve_case(parameters, coarse_x, coarse_t),
    )
    fine_error = _relative_l2(
        analytical_solution(parameters, fine_x, fine_t),
        solve_case(parameters, fine_x, fine_t),
    )

    assert fine_error < coarse_error
    assert fine_error < 1e-3


def test_solver_keeps_zero_boundaries_exactly() -> None:
    parameters = HeatParameters(alpha=0.05, amplitude_1=1.2, amplitude_2=-0.3)
    x, t = make_grid(nx=65, nt=51)

    field = solve_case(parameters, x, t)

    np.testing.assert_array_equal(field[:, 0], 0.0)
    np.testing.assert_array_equal(field[:, -1], 0.0)
    assert np.isfinite(field).all()


@pytest.mark.parametrize(
    ("parameters", "x", "t", "message"),
    [
        (
            HeatParameters(alpha=0.0, amplitude_1=1.0, amplitude_2=0.0),
            np.linspace(0.0, 1.0, 5),
            np.linspace(0.0, 1.0, 5),
            "alpha",
        ),
        (
            HeatParameters(alpha=0.1, amplitude_1=1.0, amplitude_2=0.0),
            np.array([0.0, 0.2, 0.6, 1.0]),
            np.linspace(0.0, 1.0, 5),
            "均匀",
        ),
        (
            HeatParameters(alpha=0.1, amplitude_1=1.0, amplitude_2=0.0),
            np.linspace(0.0, 1.0, 5),
            np.array([0.0, 0.2, 0.6, 1.0]),
            "均匀",
        ),
    ],
)
def test_solver_rejects_invalid_parameters_or_grids(parameters, x, t, message) -> None:
    with pytest.raises(ValueError, match=message):
        solve_case(parameters, x, t)
