from pathlib import Path

import numpy as np
import pytest

from surrogate_loop.operator.cavity2d.config import load_cavity_spec
from surrogate_loop.operator.cavity2d.sampling import (
    build_cavity_sample_plan,
    write_solver_request,
)


def test_smoke_contract_is_exact() -> None:
    spec = load_cavity_spec(Path("examples/cavity_2d_fluent/smoke.json"))

    assert spec.mode == "smoke"
    assert spec.problem.problem_id == "fluent_lid_driven_cavity_steady_v1"
    assert (
        spec.sampling.train_cases,
        spec.sampling.validation_cases,
        spec.sampling.test_cases,
    ) == (16, 4, 4)


def test_sampling_is_reproducible_and_disjoint() -> None:
    spec = load_cavity_spec(Path("examples/cavity_2d_fluent/smoke.json"))

    first = build_cavity_sample_plan(spec)
    second = build_cavity_sample_plan(spec)

    assert np.array_equal(first.reynolds, second.reynolds)
    assert len(set(first.reynolds.tolist())) == 24
    assert first.split.tolist().count("train") == 16
    assert first.split.tolist().count("validation") == 4
    assert first.split.tolist().count("development_test") == 4
    assert np.all((first.reynolds >= 10.0) & (first.reynolds <= 400.0))


def test_vertical_and_calibration_contracts_are_exact() -> None:
    vertical = build_cavity_sample_plan(
        load_cavity_spec(Path("examples/cavity_2d_fluent/vertical.json"))
    )
    calibration = build_cavity_sample_plan(
        load_cavity_spec(Path("examples/cavity_2d_fluent/calibration.json"))
    )

    assert vertical.reynolds.tolist() == [100.0]
    assert vertical.split.tolist() == ["protocol"]
    assert calibration.reynolds.tolist() == [10.0, 100.0, 400.0]
    assert calibration.split.tolist() == ["calibration"] * 3


def test_write_solver_request_uses_shared_strict_schema(tmp_path: Path) -> None:
    spec = load_cavity_spec(Path("examples/cavity_2d_fluent/vertical.json"))
    plan = build_cavity_sample_plan(spec)

    path = write_solver_request(tmp_path / "request", spec, plan)

    assert path.name == "solver-request.json"
    text = path.read_text(encoding="utf-8")
    assert '"sample_id": "vertical-000"' in text
    assert '"split": "protocol"' in text


def test_unknown_config_field_is_rejected(tmp_path: Path) -> None:
    source = Path("examples/cavity_2d_fluent/vertical.json").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "bad.json"
    path.write_text(source.replace('"mode":', '"unexpected": 1, "mode":'), encoding="utf-8")

    with pytest.raises(ValueError):
        load_cavity_spec(path)
