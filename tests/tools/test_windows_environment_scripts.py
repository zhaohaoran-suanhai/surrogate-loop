from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.tools.powershell import (
    POWERSHELL,
)
from tests.tools.powershell import (
    ps_quote as _ps_quote,
)
from tests.tools.powershell import (
    run_powershell as _run_powershell,
)

pytestmark = pytest.mark.skipif(
    POWERSHELL is None,
    reason="Windows PowerShell 5.1 is required for Windows migration tool tests",
)

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "tools/windows-migration/SurrogateLoopMigration.psm1"
TOOLS = ROOT / "tools/windows-migration"


def _environment_plan(exists: bool) -> list[dict[str, object]]:
    literal = "$true" if exists else "$false"
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Get-EnvironmentPlan -CondaEnvironmentExists {literal} | "
        "ConvertTo-Json -Depth 10 -Compress"
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    return payload if isinstance(payload, list) else [payload]


def test_environment_plan_creates_or_updates_without_prune() -> None:
    create = _environment_plan(False)
    update = _environment_plan(True)
    assert [item["name"] for item in create[:2]] == ["uv-python", "uv-sync"]
    assert create[2]["arguments"][:2] == ["env", "create"]
    assert update[2]["arguments"][:4] == [
        "env",
        "update",
        "-n",
        "surrogate-loop-fenicsx-0.11",
    ]
    assert "--prune" not in update[2]["arguments"]


def test_environment_plan_uses_fixed_paths_and_commands() -> None:
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Get-EnvironmentPlan -CondaEnvironmentExists $false -UvPath 'fixed-uv' "
        f"-CondaPath 'fixed-conda' -RepositoryRoot {_ps_quote(ROOT)} | "
        "ConvertTo-Json -Depth 10 -Compress"
    )
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert [item["file_path"] for item in plan] == [
        "fixed-uv",
        "fixed-uv",
        "fixed-conda",
        "fixed-uv",
        "fixed-uv",
    ]
    assert all(Path(item["working_directory"]).resolve() == ROOT for item in plan)


def test_prerequisite_json_always_has_actionable_schema() -> None:
    script = TOOLS / "Test-Prerequisites.ps1"
    completed = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-File", script, "-Json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode in {0, 2}
    payload = json.loads(completed.stdout)
    assert payload["stage"] == "prerequisites"
    assert payload["exit_code"] == completed.returncode
    assert isinstance(payload["evidence"]["checks"], list)
    for check in payload["evidence"]["checks"]:
        assert {"name", "status", "evidence", "guidance"} <= set(check)


def test_environment_scripts_do_not_install_or_escalate() -> None:
    forbidden = (
        "Invoke-Expression",
        "winget ",
        "-Verb RunAs",
        "Set-ExecutionPolicy",
        "conda env remove",
        "Remove-Item",
    )
    content = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            TOOLS / "Test-Prerequisites.ps1",
            TOOLS / "Initialize-Environments.ps1",
        )
    )
    assert all(token not in content for token in forbidden)
    assert "SupportsShouldProcess" in content


def test_fixed_command_restores_python_environment_under_whatif() -> None:
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        "$oldWhatIf=$WhatIfPreference;$WhatIfPreference=$true;"
        "Remove-Item Env:PYTHONUTF8,Env:PYTHONIOENCODING -ErrorAction SilentlyContinue "
        "-WhatIf:$false;"
        "try {"
        f"Invoke-FixedCommand -FilePath {_ps_quote(sys.executable)} "
        f"-Arguments @('-c','print(1)') -WorkingDirectory {_ps_quote(ROOT)} | Out-Null;"
        "$state=[ordered]@{"
        "python_utf8=(Test-Path Env:PYTHONUTF8);"
        "python_io_encoding=(Test-Path Env:PYTHONIOENCODING)"
        "};$state | ConvertTo-Json -Compress"
        "} finally {$WhatIfPreference=$oldWhatIf}"
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "python_utf8": False,
        "python_io_encoding": False,
    }


def test_initialize_whatif_json_is_single_planned_object() -> None:
    prerequisites = subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            TOOLS / "Test-Prerequisites.ps1",
            "-Json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if prerequisites.returncode != 0:
        pytest.skip("current machine does not satisfy the guarded prerequisite path")

    completed = subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            TOOLS / "Initialize-Environments.ps1",
            "-WhatIf",
            "-Json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "planned"
    assert payload["exit_code"] == 0
    assert [item["name"] for item in payload["evidence"]["plan"]] == [
        "uv-python",
        "uv-sync",
        "conda-environment",
        "python-imports",
        "fenicsx-doctor",
    ]
    assert payload["evidence"]["executed"] == []
