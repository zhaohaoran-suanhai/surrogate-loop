import math

import pytest

from surrogate_loop.domain import analytical_solution, numerical_endpoint


@pytest.mark.parametrize("gamma", [-1.0, 0.0, 1e-8, 1.0])
def test_numerical_endpoint_matches_analytical_solution(gamma: float) -> None:
    expected = analytical_solution(gamma, 1.0)
    actual = numerical_endpoint(gamma)

    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)


def test_non_finite_gamma_is_rejected() -> None:
    with pytest.raises(ValueError, match="有限"):
        analytical_solution(float("nan"), 1.0)
