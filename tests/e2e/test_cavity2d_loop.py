from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.integration.cavity2d.test_artifacts_pipeline import (
    write_synthetic_fluent_pipeline,
)

ROOT = Path(__file__).resolve().parents[2]


def test_synthetic_fluent_pipeline_exercises_cavity_cli_subprocess(
    tmp_path: Path,
) -> None:
    config = ROOT / "examples/cavity_2d_fluent/smoke.json"
    fluent = write_synthetic_fluent_pipeline(tmp_path, config)
    runs_dir = tmp_path / "runs"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "surrogate_loop",
            "cavity2d",
            "run",
            "--config",
            str(config),
            "--fluent-pipeline",
            str(fluent),
            "--runs-dir",
            str(runs_dir),
            "--request",
            "合成 Fluent 数据，仅验证跨进程闭环编排",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    response = json.loads(completed.stdout)
    assert response["status"] == "development_complete"
    run_dir = Path(response["run_dir"])
    assert run_dir.parent == runs_dir.resolve()
    assert (run_dir / "artifact_manifest.json").is_file()
    assert (run_dir / "model_card.md").is_file()
