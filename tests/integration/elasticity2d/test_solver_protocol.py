from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("dolfinx")
pytestmark = pytest.mark.fenicsx

ROOT = Path(__file__).resolve().parents[3]


def test_doctor_process_emits_one_json_document() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvers.fenicsx.elasticity2d.cli",
            "doctor",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30.0,
        shell=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["dolfinx"].startswith("0.11.")
    assert payload["petsc4py_available"] is False
    assert completed.stdout.count("{") == 1


def test_generate_requires_explicit_paths() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "solvers.fenicsx.elasticity2d.cli",
            "generate",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30.0,
        shell=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
