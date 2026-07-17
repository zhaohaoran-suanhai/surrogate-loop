from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

from tests.tools.powershell import POWERSHELL, ps_quote, run_powershell

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "tools/windows-migration/SurrogateLoopMigration.psm1"

pytestmark = pytest.mark.skipif(
    POWERSHELL is None,
    reason="Windows PowerShell 5.1 is required for Windows migration tool tests",
)


def test_module_resolves_repository_root_independent_of_current_directory(tmp_path: Path) -> None:
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        "(Get-SurrogateRepositoryRoot).Path",
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()).resolve() == ROOT


def test_migration_result_has_stable_json_schema() -> None:
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        "$r=New-MigrationResult -Status 'ok' -Stage 'unit' -Message 'done' "
        "-Evidence @{value=3} -ExitCode 0 -ElapsedSeconds 1.25;"
        "$r | ConvertTo-MigrationJson"
    )
    payload = json.loads(completed.stdout)
    assert payload == {
        "elapsed_seconds": 1.25,
        "evidence": {"value": 3},
        "exit_code": 0,
        "message": "done",
        "stage": "unit",
        "status": "ok",
    }


def test_fixed_command_preserves_argument_boundaries() -> None:
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        "$r=Invoke-FixedCommand "
        f"-FilePath {ps_quote(sys.executable)} "
        "-Arguments @('-c','import json,sys; print(json.dumps(sys.argv[1:]))',"
        "'space value','semi;colon','quote\"inside') "
        f"-WorkingDirectory {ps_quote(ROOT)};"
        "$r | ConvertTo-Json -Depth 10 -Compress"
    )
    assert completed.returncode == 0, completed.stderr
    process = json.loads(completed.stdout)
    assert process["exit_code"] == 0
    assert json.loads(process["stdout"]) == [
        "space value",
        "semi;colon",
        'quote"inside',
    ]


def test_file_manifest_is_sorted_and_detects_tampering(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "b.txt").write_text("b", encoding="utf-8")
    (run_dir / "a.txt").write_text("a", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        f"$m=Get-FileManifest -Root {ps_quote(run_dir)};"
        f"[IO.File]::WriteAllText({ps_quote(manifest_path)},"
        "($m | ConvertTo-Json -Depth 10),"
        "(New-Object Text.UTF8Encoding($false)));"
        f"Test-FileManifest -Root {ps_quote(run_dir)} -Files $m"
    )
    assert completed.returncode == 0, completed.stderr
    files = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [item["path"] for item in files] == ["a.txt", "b.txt"]

    (run_dir / "a.txt").write_text("changed", encoding="utf-8")
    failed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        f"$m=Get-Content {ps_quote(manifest_path)} -Raw | ConvertFrom-Json;"
        f"Test-FileManifest -Root {ps_quote(run_dir)} -Files $m"
    )
    assert failed.returncode != 0
    assert "SHA-256" in failed.stderr


@pytest.mark.parametrize(
    "entry",
    ("../escape.txt", "/absolute.txt", "C:/drive.txt", "run/item.txt:ads", "run/../escape"),
)
def test_zip_entry_validation_rejects_unsafe_paths(tmp_path: Path, entry: str) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(entry, "bad")
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        f"Test-SafeZipEntries -ArchivePath {ps_quote(archive)} "
        f"-DestinationRoot {ps_quote(tmp_path / 'destination')}"
    )
    assert completed.returncode != 0
    assert "ZIP" in completed.stderr
