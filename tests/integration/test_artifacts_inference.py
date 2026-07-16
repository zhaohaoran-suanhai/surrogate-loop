import math

import pytest

from surrogate_loop.artifacts import create_run_directory, save_successful_run
from surrogate_loop.config import load_spec
from surrogate_loop.data import generate_dataset
from surrogate_loop.evaluation import train_select_and_test
from surrogate_loop.inference import predict_endpoint
from surrogate_loop.models import build_candidates
from surrogate_loop.split import split_dataset


def build_saved_run(tmp_path, smoke_spec_path):
    spec = load_spec(smoke_spec_path)
    dataset = generate_dataset(spec)
    split = split_dataset(dataset, spec.sampling)
    selection = train_select_and_test(
        split,
        build_candidates(spec.sampling.seed, spec.models.candidates),
        spec.acceptance,
    )
    run_dir = create_run_directory(tmp_path)
    save_successful_run(run_dir, spec, "测试自然语言请求", dataset, split, selection)
    return run_dir, selection


def test_saved_run_reloads_and_rejects_invalid_inputs(tmp_path, smoke_spec_path) -> None:
    run_dir, selection = build_saved_run(tmp_path, smoke_spec_path)

    prediction = predict_endpoint(run_dir, 0.35)
    expected = float(selection.selected_model.predict([[0.35]])[0])
    assert math.isclose(prediction, expected, rel_tol=0, abs_tol=1e-12)
    for invalid in (1.2, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            predict_endpoint(run_dir, invalid)


def test_manifest_detects_tampering(tmp_path, smoke_spec_path) -> None:
    run_dir, _ = build_saved_run(tmp_path, smoke_spec_path)
    spec_path = run_dir / "spec.json"
    spec_path.write_text(spec_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="哈希"):
        predict_endpoint(run_dir, 0.35)
