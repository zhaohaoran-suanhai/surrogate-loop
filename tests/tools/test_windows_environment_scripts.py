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


@pytest.mark.parametrize(
    ("kind", "report_tokens", "predict_tokens"),
    (
        ("scalar", ["report"], ["predict", "--gamma", "0.35"]),
        ("heat1d", ["operator", "report"], ["operator", "predict", "--alpha", "0.1"]),
        (
            "elasticity2d",
            ["elasticity2d", "report"],
            ["elasticity2d", "predict", "--e", "3"],
        ),
    ),
)
def test_model_verification_plan_is_allowlisted(
    tmp_path: Path,
    kind: str,
    report_tokens: list[str],
    predict_tokens: list[str],
) -> None:
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Get-ModelVerificationPlan -ModelKind '{kind}' "
        f"-RunDir {_ps_quote(tmp_path)} | ConvertTo-Json -Depth 10 -Compress"
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert all(token in payload[0]["arguments"] for token in report_tokens)
    assert all(token in payload[1]["arguments"] for token in predict_tokens)


def test_unknown_model_kind_and_incomplete_full_chain_are_rejected(
    tmp_path: Path,
) -> None:
    unknown = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Get-ModelVerificationPlan -ModelKind 'shell' -RunDir {_ps_quote(tmp_path)}"
    )
    assert unknown.returncode != 0

    incomplete = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        "Get-InstallationPlan -Level 'FullChain'"
    )
    assert incomplete.returncode != 0


def test_installation_plans_never_run_formal_training(tmp_path: Path) -> None:
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Get-InstallationPlan -Level 'FullChain' -ModelKind 'elasticity2d' "
        f"-AcceptedRunDir {_ps_quote(tmp_path)} | ConvertTo-Json -Depth 10 -Compress"
    )
    assert completed.returncode == 0, completed.stderr
    text = completed.stdout
    assert "test_elasticity2d_fenicsx_loop.py" in text
    for forbidden in ("calibrate", "full.json", "elasticity2d run", "sealed-test"):
        assert forbidden not in text


def test_installation_levels_are_cumulative(tmp_path: Path) -> None:
    expected_names = {
        "Prerequisites": [],
        "Python": ["cli-help", "cli-version", "cuda-backward", "ruff", "pytest"],
        "Fenicsx": [
            "cli-help",
            "cli-version",
            "cuda-backward",
            "ruff",
            "pytest",
            "fenicsx-doctor",
            "solver-tests",
        ],
        "FullChain": [
            "cli-help",
            "cli-version",
            "cuda-backward",
            "ruff",
            "pytest",
            "fenicsx-doctor",
            "solver-tests",
            "real-fenicsx-e2e",
        ],
    }
    for level, names in expected_names.items():
        extra = (
            f" -ModelKind 'elasticity2d' -AcceptedRunDir {_ps_quote(tmp_path)}"
            if level == "FullChain"
            else ""
        )
        completed = _run_powershell(
            f"Import-Module {_ps_quote(MODULE)} -Force;"
            f"Get-InstallationPlan -Level '{level}'{extra} | "
            "ConvertTo-Json -Depth 10 -Compress"
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout or "[]")
        plan = payload if isinstance(payload, list) else [payload]
        assert [item["name"] for item in plan] == names


def test_installation_script_restores_e2e_environment_and_refuses_report_overwrite() -> (
    None
):
    content = (TOOLS / "Test-Installation.ps1").read_text(encoding="utf-8")
    assert "try" in content and "finally" in content
    assert "SURROGATE_LOOP_RUN_FENICSX_E2E" in content
    assert "Test-Path -LiteralPath $ReportPath" in content
    assert "[IO.FileMode]::CreateNew" in content
    assert "exit 5" in content
    assert "[IO.Path]::IsPathRooted($ReportPath)" in content
    assert "Join-Path $PWD.Path $ReportPath" in content
    conditional = content.index("if ($Level -eq 'FullChain')")
    model_binding = content.index("$planParameters['ModelKind'] = $ModelKind")
    plan_call = content.index("Get-InstallationPlan @planParameters")
    assert conditional < model_binding < plan_call
    assert "-ModelKind $ModelKind -UvPath" not in content
