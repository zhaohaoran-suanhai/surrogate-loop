import json
import shutil
import subprocess
from pathlib import Path

import pytest

from surrogate_loop.operator.external_solver import (
    CONDA_ENV_NAME,
    build_solver_command,
    doctor_solver_environment,
    run_solver_process,
)

ROOT = Path(__file__).resolve().parents[4]


def test_solver_command_is_fixed_and_never_uses_shell(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: "C:/Miniforge3/Scripts/conda.exe" if name == "conda" else None,
    )

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, '{"status":"ok"}', "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    completed = run_solver_process("doctor", (), tmp_path, 30.0)

    assert completed.returncode == 0
    assert captured["command"] == [
        "C:/Miniforge3/Scripts/conda.exe",
        "run",
        "-n",
        CONDA_ENV_NAME,
        "python",
        "-m",
        "solvers.fenicsx.elasticity2d.cli",
        "doctor",
    ]
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["shell"] is False
    assert kwargs["cwd"] == tmp_path
    assert kwargs["timeout"] == 30.0
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert environment["PYTHONIOENCODING"] == "utf-8"
    assert environment["PYTHONUTF8"] == "1"


def test_unknown_solver_action_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "conda.exe")

    with pytest.raises(ValueError, match="固定动作"):
        build_solver_command("powershell")


def test_missing_conda_has_actionable_error(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(
        "surrogate_loop.operator.external_solver._default_conda_candidates",
        lambda: (),
    )

    with pytest.raises(RuntimeError, match="Miniforge"):
        build_solver_command("doctor")


def test_solver_command_finds_default_miniforge_when_path_is_stale(
    monkeypatch, tmp_path
) -> None:
    conda = tmp_path / "miniforge3/Scripts/conda.exe"
    conda.parent.mkdir(parents=True)
    conda.touch()
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(
        "surrogate_loop.operator.external_solver._default_conda_candidates",
        lambda: (conda,),
    )

    command = build_solver_command("doctor")

    assert command[0] == str(conda)


def test_doctor_validates_required_versions(monkeypatch, tmp_path) -> None:
    payload = {
        "status": "ok",
        "python": "3.12.11",
        "dolfinx": "0.11.0",
        "pyamg": "5.3.0",
        "scipy": "1.16.0",
        "petsc4py_available": False,
    }
    monkeypatch.setattr(
        "surrogate_loop.operator.external_solver.run_solver_process",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, json.dumps(payload), ""
        ),
    )

    assert doctor_solver_environment(tmp_path) == payload


def test_doctor_ignores_conda_wrapper_output_before_final_json(
    monkeypatch, tmp_path
) -> None:
    payload = {
        "status": "ok",
        "python": "3.12.13",
        "dolfinx": "0.11.0",
        "pyamg": "5.3.0",
        "scipy": "1.18.0",
        "petsc4py_available": False,
    }
    wrapped_stdout = (
        "(surrogate-loop-fenicsx-0.11)>SET DISTUTILS_USE_SDK=1\n"
        + json.dumps(payload)
        + "\n"
    )
    monkeypatch.setattr(
        "surrogate_loop.operator.external_solver.run_solver_process",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, wrapped_stdout, ""
        ),
    )

    assert doctor_solver_environment(tmp_path) == payload


def test_doctor_rejects_petsc_windows_environment(monkeypatch, tmp_path) -> None:
    payload = {
        "status": "ok",
        "python": "3.12.11",
        "dolfinx": "0.11.0",
        "pyamg": "5.3.0",
        "scipy": "1.16.0",
        "petsc4py_available": True,
    }
    monkeypatch.setattr(
        "surrogate_loop.operator.external_solver.run_solver_process",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, json.dumps(payload), ""
        ),
    )

    with pytest.raises(RuntimeError, match="petsc4py"):
        doctor_solver_environment(tmp_path)


def test_fenicsx_environment_uses_windows_pyamg_contract() -> None:
    environment = (ROOT / "environments/fenicsx-0.11.yml").read_text(encoding="utf-8")

    assert "name: surrogate-loop-fenicsx-0.11" in environment
    assert "fenics-dolfinx=0.11.*" in environment
    assert "pyamg>=5,<6" in environment
    assert "psutil>=5,<7" in environment
    assert "nodefaults" in environment
    assert "petsc" not in environment.lower()
