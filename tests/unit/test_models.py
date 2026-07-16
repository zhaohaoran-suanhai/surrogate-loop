import numpy as np
import pytest

from surrogate_loop.models import build_candidates

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def test_candidate_registry_builds_trainable_models() -> None:
    names = ("prs_1", "prs_2", "prs_3", "gpr", "mlp")
    candidates = build_candidates(20260716, names)
    x = np.linspace(-1, 1, 24).reshape(-1, 1)
    y = 0.25 + 0.08 * x.ravel()

    assert tuple(candidates) == names
    for model in candidates.values():
        prediction = model.fit(x, y).predict(np.array([[0.2]]))
        assert np.isfinite(prediction).all()
