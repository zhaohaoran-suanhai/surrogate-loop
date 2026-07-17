import os
import subprocess
import sys


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "surrogate_loop", *args],
        check=False,
        capture_output=True,
        encoding="utf-8",
        env=environment,
    )


def test_module_entrypoint_displays_help() -> None:
    result = run_cli("--help")

    assert result.returncode == 0
    assert "标量代理模型最小闭环命令行入口" in result.stdout


def test_module_entrypoint_displays_version() -> None:
    result = run_cli("--version")

    assert result.returncode == 0
    assert result.stdout.strip() == "surrogate-loop 0.1.0"


def test_module_entrypoint_displays_elasticity_help() -> None:
    result = run_cli("elasticity2d", "--help")

    assert result.returncode == 0
    assert "二维线弹性神经算子闭环" in result.stdout
