import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "tests/fixtures/heat_operator_tiny.json"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "surrogate_loop", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        encoding="utf-8",
        env=environment,
    )


def test_cli_executes_complete_heat_operator_loop(tmp_path) -> None:
    validation = run_cli("operator", "validate", "--config", str(CONFIG))
    assert validation.returncode == 0, validation.stderr
    assert json.loads(validation.stdout)["status"] == "valid"

    execution = run_cli(
        "operator",
        "run",
        "--config",
        str(CONFIG),
        "--runs-dir",
        str(tmp_path),
        "--request",
        "训练测试规模的一维热传导 DeepONet",
    )
    assert execution.returncode == 0, execution.stderr
    payload = json.loads(execution.stdout)
    run_dir = Path(payload["run_dir"])
    assert payload["status"] == "accepted"

    report = run_cli("operator", "report", "--run-dir", str(run_dir))
    assert report.returncode == 0, report.stderr
    assert json.loads(report.stdout)["status"] == "accepted"

    point = run_cli(
        "operator",
        "predict",
        "--run-dir",
        str(run_dir),
        "--alpha",
        "0.1",
        "--a",
        "1.0",
        "--b",
        "0.0",
        "--x",
        "0.5",
        "--t",
        "0.25",
    )
    assert point.returncode == 0, point.stderr
    assert np.isfinite(json.loads(point.stdout)["u"])

    output = tmp_path / "predicted_field.npz"
    field = run_cli(
        "operator",
        "predict",
        "--run-dir",
        str(run_dir),
        "--alpha",
        "0.1",
        "--a",
        "1.0",
        "--b",
        "0.0",
        "--nx",
        "9",
        "--nt",
        "7",
        "--output",
        str(output),
    )
    assert field.returncode == 0, field.stderr
    assert json.loads(field.stdout)["shape"] == [7, 9]
    assert output.exists()

    rejected = run_cli(
        "operator",
        "predict",
        "--run-dir",
        str(run_dir),
        "--alpha",
        "0.21",
        "--a",
        "1.0",
        "--b",
        "0.0",
        "--x",
        "0.5",
        "--t",
        "0.25",
    )
    assert rejected.returncode == 2
    assert "训练参数域" in rejected.stderr
