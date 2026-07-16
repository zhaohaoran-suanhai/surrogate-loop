from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytestmark = pytest.mark.fenicsx

from solvers.fenicsx.elasticity2d.cli import main  # noqa: E402
from solvers.fenicsx.elasticity2d.quality import (  # noqa: E402
    generate_datasets,
    run_calibration,
    run_manufactured_convergence,
)


def _write_tiny_job(path: Path) -> Path:
    payload = {
        "protocol_version": "elasticity-job-v1",
        "problem_id": "elasticity_2d_cantilever_v1",
        "solver": {
            "mesh_shape": [8, 2],
            "observation_shape": [9, 3],
            "backend": "scipy",
            "tolerance": 1e-10,
        },
        "quality": {
            "residual_max": 1e-8,
            "force_balance_max": 1e-5,
            "clamp_max": 1e-12,
            "mesh_relative_l2_max": 0.5,
            "linearity_relative_max": 1e-6,
        },
        "samples": [
            {
                "sample_id": "train-00000-aaaaaaaaaaaa",
                "role": "train",
                "parameters": [2.0, 0.30, 0.004, -1.5707963267948966, 0.5, 0.12],
            },
            {
                "sample_id": "validation-00001-bbbbbbbbbbbb",
                "role": "validation",
                "parameters": [3.0, 0.25, 0.005, 0.0, 0.4, 0.10],
            },
            {
                "sample_id": "development_test-00002-cccccccccccc",
                "role": "development_test",
                "parameters": [4.0, 0.35, 0.006, 1.5707963267948966, 0.6, 0.15],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_manufactured_solution_has_p2_convergence() -> None:
    report = run_manufactured_convergence([(8, 2), (16, 4), (32, 8)])

    assert report.minimum_l2_rate >= 2.5
    assert report.minimum_h1_rate >= 1.5
    assert len(report.levels) == 3


def test_generate_writes_physically_separate_datasets(tmp_path: Path) -> None:
    manifest = generate_datasets(_write_tiny_job(tmp_path / "job.json"), tmp_path)

    development = tmp_path / "datasets/development.npz"
    sealed = tmp_path / "datasets/sealed_test.npz"
    manifest_path = tmp_path / "datasets/dataset_manifest.json"
    diagnostics_path = tmp_path / "diagnostics/solver_quality.json"
    assert development.is_file()
    assert sealed.is_file()
    assert manifest_path.is_file()
    assert diagnostics_path.is_file()
    assert manifest.development_sha256 != manifest.sealed_test_sha256

    with np.load(development, allow_pickle=False) as arrays:
        assert arrays["sample_ids"].tolist() == [
            "train-00000-aaaaaaaaaaaa",
            "validation-00001-bbbbbbbbbbbb",
        ]
        assert arrays["fields"].shape == (2, 27, 2)
        assert arrays["coordinates"].shape == (27, 2)
    with np.load(sealed, allow_pickle=False) as arrays:
        assert arrays["sample_ids"].tolist() == [
            "development_test-00002-cccccccccccc"
        ]
        assert arrays["fields"].shape == (1, 27, 2)


def test_generate_rejects_duplicate_sample_identity(tmp_path: Path) -> None:
    job_path = _write_tiny_job(tmp_path / "job.json")
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    payload["samples"][1]["sample_id"] = payload["samples"][0]["sample_id"]
    job_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="样本.*重复"):
        generate_datasets(job_path, tmp_path / "output")


def test_calibration_records_mesh_and_load_linearity(tmp_path: Path) -> None:
    job_path = _write_tiny_job(tmp_path / "job.json")
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    payload["samples"] = payload["samples"][:1]
    job_path.write_text(json.dumps(payload), encoding="utf-8")

    output = run_calibration(job_path, tmp_path / "output")
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["manufactured"]["minimum_l2_rate"] >= 2.5
    assert report["manufactured"]["minimum_h1_rate"] >= 1.5
    assert report["samples"][0]["mesh_relative_l2"] <= 0.5
    assert report["samples"][0]["linearity_relative_error"] <= 1e-6


def test_failed_quality_gate_writes_no_success_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job_path = _write_tiny_job(tmp_path / "job.json")
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    payload["quality"]["residual_max"] = 1e-30
    job_path.write_text(json.dumps(payload), encoding="utf-8")
    output_dir = tmp_path / "output"

    code = main(
        ["generate", "--job", str(job_path), "--output-dir", str(output_dir)]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "relative_residual" in captured.err
    assert not (output_dir / "datasets/dataset_manifest.json").exists()
