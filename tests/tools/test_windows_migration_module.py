from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

from tests.tools.powershell import (
    POWERSHELL,
    POWERSHELL_7,
    POWERSHELL_EXECUTABLES,
    ps_quote,
    run_powershell,
)

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "tools/windows-migration/SurrogateLoopMigration.psm1"

pytestmark = pytest.mark.skipif(
    POWERSHELL is None,
    reason="Windows PowerShell 5.1 is required for Windows migration tool tests",
)


@pytest.fixture(params=POWERSHELL_EXECUTABLES, ids=lambda executable: Path(executable).name)
def powershell_executable(request: pytest.FixtureRequest) -> str:
    return request.param


def test_module_resolves_repository_root_independent_of_current_directory(
    tmp_path: Path, powershell_executable: str
) -> None:
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        "(Get-SurrogateRepositoryRoot).Path",
        cwd=tmp_path,
        executable=powershell_executable,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()).resolve() == ROOT


def test_migration_result_has_stable_json_schema(powershell_executable: str) -> None:
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        "$r=New-MigrationResult -Status 'ok' -Stage 'unit' -Message 'done' "
        "-Evidence @{value=3} -ExitCode 0 -ElapsedSeconds 1.25;"
        "$r | ConvertTo-MigrationJson",
        executable=powershell_executable,
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


def test_fixed_command_preserves_argument_boundaries(powershell_executable: str) -> None:
    arguments = [
        "-c",
        "import json,sys; print(json.dumps(sys.argv[1:]))",
        "",
        "space value",
        "semi;colon",
        'quote"inside',
        'backslash\\"before-quote',
        "trailing\\",
        "two-trailing\\\\",
    ]
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        "$r=Invoke-FixedCommand "
        f"-FilePath {ps_quote(sys.executable)} "
        f"-Arguments @({','.join(ps_quote(argument) for argument in arguments)}) "
        f"-WorkingDirectory {ps_quote(ROOT)};"
        "$r | ConvertTo-Json -Depth 10 -Compress",
        executable=powershell_executable,
    )
    assert completed.returncode == 0, completed.stderr
    process = json.loads(completed.stdout)
    assert process["exit_code"] == 0
    assert json.loads(process["stdout"]) == arguments[2:]


def test_fixed_command_sets_utf8_environment_when_parent_keys_are_absent(
    powershell_executable: str,
) -> None:
    arguments = [
        "-c",
        "import json,os; print(json.dumps([os.environ.get('PYTHONUTF8'), "
        "os.environ.get('PYTHONIOENCODING')]))",
    ]
    completed = run_powershell(
        "$env:PYTHONUTF8=$null;$env:PYTHONIOENCODING=$null;"
        f"Import-Module {ps_quote(MODULE)} -Force;"
        "$result=Invoke-FixedCommand "
        f"-FilePath {ps_quote(sys.executable)} "
        f"-Arguments @({','.join(ps_quote(argument) for argument in arguments)}) "
        f"-WorkingDirectory {ps_quote(ROOT)};"
        "$result | ConvertTo-Json -Depth 10 -Compress",
        executable=powershell_executable,
    )
    assert completed.returncode == 0, completed.stderr
    process = json.loads(completed.stdout)
    assert process["exit_code"] == 0
    assert json.loads(process["stdout"]) == ["1", "utf-8"]


def test_json_migration_output_writes_one_compact_object(powershell_executable: str) -> None:
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        "$result=New-MigrationResult -Status 'ok' -Stage 'unit' -Message 'done' "
        "-Evidence @{value=3} -ExitCode 0 -ElapsedSeconds 1.25;"
        "Write-MigrationOutput -Result $result -Json",
        executable=powershell_executable,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert completed.stdout == json.dumps(payload, separators=(",", ":")) + "\n"
    assert payload["status"] == "ok"


def test_file_manifest_is_sorted_and_detects_tampering(
    tmp_path: Path, powershell_executable: str
) -> None:
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
        f"Test-FileManifest -Root {ps_quote(run_dir)} -Files $m",
        executable=powershell_executable,
    )
    assert completed.returncode == 0, completed.stderr
    files = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [item["path"] for item in files] == ["a.txt", "b.txt"]

    (run_dir / "a.txt").write_text("changed", encoding="utf-8")
    failed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        f"$m=Get-Content {ps_quote(manifest_path)} -Raw | ConvertFrom-Json;"
        f"Test-FileManifest -Root {ps_quote(run_dir)} -Files $m",
        executable=powershell_executable,
    )
    assert failed.returncode != 0
    assert "SHA-256" in failed.stderr


def test_file_manifest_paths_use_ordinal_sorting(
    tmp_path: Path, powershell_executable: str
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paths = ["z.txt", "\u00e4.txt", "a.txt", "\u00e9.txt"]
    for path in paths:
        (run_dir / path).write_text(path, encoding="utf-8")
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        f"Get-FileManifest -Root {ps_quote(run_dir)} | ConvertTo-Json -Depth 10",
        executable=powershell_executable,
    )
    assert completed.returncode == 0, completed.stderr
    files = json.loads(completed.stdout)
    assert [item["path"] for item in files] == ["a.txt", "z.txt", "\u00e4.txt", "\u00e9.txt"]


@pytest.mark.parametrize(
    "entry",
    ("../escape.txt", "/absolute.txt", "C:/drive.txt", "run/item.txt:ads", "run/../escape"),
)
def test_zip_entry_validation_rejects_unsafe_paths(
    tmp_path: Path, entry: str, powershell_executable: str
) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(entry, "bad")
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        f"Test-SafeZipEntries -ArchivePath {ps_quote(archive)} "
        f"-DestinationRoot {ps_quote(tmp_path / 'destination')}",
        executable=powershell_executable,
    )
    assert completed.returncode != 0
    assert "ZIP" in completed.stderr


@pytest.mark.parametrize(
    "entries",
    (
        ("run/file.txt", "run\\file.txt"),
        ("run/file.txt", "run/./file.txt"),
        ("run/file.txt", "RUN/FILE.TXT"),
    ),
)
def test_zip_entry_validation_rejects_normalized_path_collisions(
    tmp_path: Path, entries: tuple[str, str], powershell_executable: str
) -> None:
    archive = tmp_path / "collision.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for entry in entries:
            info = zipfile.ZipInfo("placeholder")
            info.filename = entry
            bundle.writestr(info, "data")
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        f"Test-SafeZipEntries -ArchivePath {ps_quote(archive)} "
        f"-DestinationRoot {ps_quote(tmp_path / 'destination')}",
        executable=powershell_executable,
    )
    assert completed.returncode != 0
    assert "ZIP" in completed.stderr


def test_zip_entry_validation_rejects_reparse_destination_root(
    tmp_path: Path, powershell_executable: str
) -> None:
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("run/file.txt", "data")
    target = tmp_path / "target"
    target.mkdir()
    destination = tmp_path / "destination"
    junction = run_powershell(
        f"New-Item -ItemType Junction -Path {ps_quote(destination)} "
        f"-Target {ps_quote(target)} | Out-Null",
        executable=powershell_executable,
    )
    if junction.returncode != 0:
        pytest.skip("cannot create a temporary junction to test ZIP reparse protection")
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        f"Test-SafeZipEntries -ArchivePath {ps_quote(archive)} "
        f"-DestinationRoot {ps_quote(destination)}",
        executable=powershell_executable,
    )
    assert completed.returncode != 0
    assert "reparse" in completed.stderr.lower()


def test_powershell_7_availability_is_recorded() -> None:
    if POWERSHELL_7 is None:
        pytest.skip("pwsh.exe is unavailable; PowerShell 7 execution cannot be verified")
    assert POWERSHELL_7 in POWERSHELL_EXECUTABLES
