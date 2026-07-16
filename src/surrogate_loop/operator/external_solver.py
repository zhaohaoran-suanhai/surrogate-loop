from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

CONDA_ENV_NAME = "surrogate-loop-fenicsx-0.11"
SOLVER_MODULE = "solvers.fenicsx.elasticity2d.cli"
ALLOWED_ACTIONS = frozenset({"doctor", "calibrate", "generate"})


def _default_conda_candidates() -> tuple[Path, ...]:
    home = Path.home()
    return (
        home / "miniforge3" / "Scripts" / "conda.exe",
        home / "Miniforge3" / "Scripts" / "conda.exe",
        home / "AppData" / "Local" / "miniforge3" / "Scripts" / "conda.exe",
    )


def _find_conda() -> str | None:
    located = shutil.which("conda")
    if located is not None:
        return located
    return next(
        (str(candidate) for candidate in _default_conda_candidates() if candidate.is_file()),
        None,
    )


def build_solver_command(action: str, *arguments: str) -> list[str]:
    if action not in ALLOWED_ACTIONS:
        raise ValueError("外部求解器只允许 doctor、calibrate、generate 三种固定动作")
    conda = _find_conda()
    if conda is None:
        raise RuntimeError("未找到 Conda；请先安装 Miniforge 并重新打开终端")
    return [
        conda,
        "run",
        "-n",
        CONDA_ENV_NAME,
        "python",
        "-m",
        SOLVER_MODULE,
        action,
        *arguments,
    ]


def run_solver_process(
    action: str,
    arguments: Sequence[str],
    repo_root: Path,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("外部求解器超时必须是有限正数")
    command = build_solver_command(action, *tuple(arguments))
    return subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        shell=False,
    )


def doctor_solver_environment(repo_root: Path) -> dict[str, object]:
    completed = run_solver_process("doctor", (), repo_root, 30.0)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "未知错误"
        raise RuntimeError(f"FEniCSx 环境诊断失败：{detail}")
    payload = parse_solver_json(completed.stdout, "doctor")
    return _validate_doctor_payload(payload)


def parse_solver_json(stdout: str, action: str) -> dict[str, object]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise RuntimeError(f"FEniCSx {action} 未返回有效 JSON") from error
    except IndexError as error:
        raise RuntimeError(f"FEniCSx {action} 未返回有效 JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"FEniCSx {action} 返回值必须是 JSON 对象")
    return payload


def _validate_doctor_payload(payload: dict[str, object]) -> dict[str, object]:
    required = {
        "status",
        "python",
        "dolfinx",
        "pyamg",
        "scipy",
        "petsc4py_available",
    }
    if not required <= set(payload):
        raise RuntimeError("FEniCSx doctor 返回字段不完整")
    if payload["status"] != "ok":
        raise RuntimeError("FEniCSx doctor 未报告 ok 状态")
    if not str(payload["dolfinx"]).startswith("0.11."):
        raise RuntimeError("FEniCSx/DOLFINx 版本必须为 0.11.x")
    if payload["petsc4py_available"] is not False:
        raise RuntimeError("Windows 原生求解环境不得依赖 petsc4py")
    for name in ("python", "pyamg", "scipy"):
        if not isinstance(payload[name], str) or not payload[name]:
            raise RuntimeError(f"FEniCSx doctor 的 {name} 版本无效")
    return payload
