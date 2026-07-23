import json
from pathlib import Path

import numpy as np
import pytest

from surrogate_loop.cli import build_parser, main
from surrogate_loop.operator.cavity2d.inference import (
    predict_accepted_cavity,
)
from surrogate_loop.operator.cavity2d.pipeline import run_cavity_pipeline
from tests.integration.cavity2d.test_artifacts_pipeline import (
    write_synthetic_fluent_pipeline,
)


def test_full_accepted_run_supports_protected_inference(tmp_path: Path) -> None:
    config = Path("examples/cavity_2d_fluent/full.json")
    fluent = write_synthetic_fluent_pipeline(tmp_path, config)
    payload = json.loads(fluent.read_text(encoding="utf-8"))
    for batch in payload["batches"]:
        for sample in batch["samples"]:
            sample["wall_time_seconds"] = 10.0
    fluent.write_text(json.dumps(payload), encoding="utf-8")
    result = run_cavity_pipeline(config, fluent, tmp_path / "runs", "Full test")
    output = tmp_path / "prediction.npz"

    response = predict_accepted_cavity(result.run_dir, 50.0, output)

    assert result.status == "accepted"
    assert response["shape"] == [9, 3]
    with np.load(output, allow_pickle=False) as archive:
        assert archive["velocity"].shape == (9, 2)
        assert archive["pressure"].shape == (9,)
    with pytest.raises(ValueError, match="\\[10,400\\]"):
        predict_accepted_cavity(result.run_dir, 500.0, tmp_path / "bad.npz")
    with pytest.raises(ValueError, match="protected"):
        predict_accepted_cavity(
            result.run_dir,
            50.0,
            result.run_dir / "prediction.npz",
        )


def test_cavity_cli_exposes_fixed_commands_and_plan(tmp_path: Path, capsys) -> None:
    parser = build_parser()
    help_text = parser.format_help()

    assert "cavity2d" in help_text
    code = main(
        [
            "cavity2d",
            "plan",
            "--config",
            "examples/cavity_2d_fluent/vertical.json",
            "--output-dir",
            str(tmp_path / "plan"),
        ]
    )

    assert code == 0
    assert (tmp_path / "plan" / "solver-request.json").is_file()
    assert '"status": "planned"' in capsys.readouterr().out


def test_solver_verification_rejects_pipeline_for_a_different_sample_plan(
    tmp_path: Path,
) -> None:
    fluent = write_synthetic_fluent_pipeline(
        tmp_path,
        Path("examples/cavity_2d_fluent/vertical.json"),
    )

    code = main(
        [
            "cavity2d",
            "verify-solver",
            "--config",
            "examples/cavity_2d_fluent/calibration.json",
            "--fluent-pipeline",
            str(fluent),
            "--output-dir",
            str(tmp_path / "verified"),
        ]
    )

    assert code == 2
