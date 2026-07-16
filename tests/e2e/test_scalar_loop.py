import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples/forced_reaction_scalar/smoke.json"


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


def test_cli_executes_the_complete_scalar_loop(tmp_path) -> None:
    validation = run_cli("validate", "--config", str(CONFIG))
    assert validation.returncode == 0
    assert json.loads(validation.stdout)["status"] == "valid"

    execution = run_cli(
        "run",
        "--config",
        str(CONFIG),
        "--smoke",
        "--runs-dir",
        str(tmp_path),
        "--request",
        "gamma 在 -1 到 1 内，训练并选择最佳代理模型",
    )
    assert execution.returncode == 0, execution.stderr
    payload = json.loads(execution.stdout)
    run_dir = Path(payload["run_dir"])
    assert payload["status"] == "accepted"

    report = run_cli("report", "--run-dir", str(run_dir))
    assert json.loads(report.stdout)["status"] == "accepted"

    prediction = run_cli("predict", "--run-dir", str(run_dir), "--gamma", "0.35")
    assert prediction.returncode == 0
    assert 0.0 < json.loads(prediction.stdout)["u_at_1"] < 1.0

    rejected = run_cli("predict", "--run-dir", str(run_dir), "--gamma", "1.2")
    assert rejected.returncode == 2
    assert "超出训练参数域" in rejected.stderr
