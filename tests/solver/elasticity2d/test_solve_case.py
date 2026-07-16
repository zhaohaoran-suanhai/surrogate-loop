from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytestmark = [
    pytest.mark.fenicsx,
    pytest.mark.filterwarnings("error::scipy.sparse.SparseEfficiencyWarning"),
]

from solvers.fenicsx.elasticity2d.interpolate import (  # noqa: E402
    interpolate_displacement,
    observation_coordinates,
)
from solvers.fenicsx.elasticity2d.solve import solve_case  # noqa: E402


def test_tiny_case_is_finite_clamped_and_balanced() -> None:
    case = solve_case(
        np.array([2.0, 0.3, 0.005, -np.pi / 2.0, 0.5, 0.12]),
        mesh_shape=(16, 4),
        backend="pyamg",
        tolerance=1e-10,
    )

    assert case.relative_residual <= 1e-8
    np.testing.assert_allclose(
        case.reaction + case.applied_force,
        0.0,
        rtol=1e-5,
        atol=1e-10,
    )
    assert case.iterations > 0
    assert case.solve_seconds > 0.0
    assert np.isfinite(case.stress_summary["von_mises_max"])

    coordinates = observation_coordinates(17, 5)
    values = interpolate_displacement(case.solution, coordinates)
    assert values.shape == (85, 2)
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values[np.isclose(coordinates[:, 0], 0.0)], 0.0, atol=1e-12)


@pytest.mark.parametrize("backend", ["unknown", "petsc"])
def test_solver_rejects_unapproved_backends(backend: str) -> None:
    with pytest.raises(ValueError, match="pyamg|scipy"):
        solve_case(
            np.array([2.0, 0.3, 0.005, 0.0, 0.5, 0.12]),
            mesh_shape=(4, 1),
            backend=backend,
            tolerance=1e-8,
        )


def test_interpolation_rejects_points_outside_domain() -> None:
    case = solve_case(
        np.array([2.0, 0.3, 0.005, 0.0, 0.5, 0.12]),
        mesh_shape=(4, 1),
        backend="scipy",
        tolerance=1e-10,
    )
    with pytest.raises(ValueError, match="域外"):
        interpolate_displacement(case.solution, np.array([[4.1, 0.5]]))
