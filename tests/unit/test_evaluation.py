import math

import numpy as np

from surrogate_loop.config import load_spec
from surrogate_loop.data import generate_dataset
from surrogate_loop.evaluation import compute_metrics, train_select_and_test
from surrogate_loop.models import build_candidates
from surrogate_loop.split import split_dataset


def test_metric_formula_is_explicit() -> None:
    metrics = compute_metrics(np.array([0.0, 2.0]), np.array([0.0, 1.0]))

    assert math.isclose(metrics.rmse, math.sqrt(0.5))
    assert math.isclose(metrics.nrmse, math.sqrt(0.5) / 2)
    assert metrics.mae == 0.5
    assert metrics.max_absolute_error == 1.0


def test_validation_selection_produces_an_accepted_test_result(smoke_spec_path) -> None:
    spec = load_spec(smoke_spec_path)
    split = split_dataset(generate_dataset(spec), spec.sampling)
    candidates = build_candidates(spec.sampling.seed, spec.models.candidates)

    result = train_select_and_test(split, candidates, spec.acceptance)

    assert result.selected_name in spec.models.candidates
    assert set(result.validation_metrics) == set(spec.models.candidates)
    assert result.test_metrics.nrmse <= spec.acceptance.max_nrmse
    assert result.accepted is True
