from __future__ import annotations

import hashlib
import json
import zipfile
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

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "tools/windows-migration/SurrogateLoopMigration.psm1"
TOOLS = ROOT / "tools/windows-migration"

pytestmark = pytest.mark.skipif(
    POWERSHELL is None,
    reason="Windows PowerShell 5.1 is required for Windows migration tool tests",
)


def test_bundle_archive_contains_manifest_run_tree_and_checksum(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "accepted-run"
    output = tmp_path / "output"
    run_dir.mkdir()
    output.mkdir()
    (run_dir / "status.json").write_text(
        '{"status":"accepted"}', encoding="utf-8"
    )
    (run_dir / "weights.bin").write_bytes(b"weights")
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        "$verification=[pscustomobject]@{status='accepted'};"
        f"New-RunBundleArchive -RunDir {_ps_quote(run_dir)} "
        "-ModelKind 'elasticity2d' "
        f"-OutputDirectory {_ps_quote(output)} -RepositoryRoot {_ps_quote(ROOT)} "
        "-Verification $verification | ConvertTo-Json -Depth 20 -Compress"
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    archive = Path(result["archive_path"])
    checksum = Path(result["checksum_path"])
    assert archive.name == "accepted-run.surrogate-run.zip"
    assert checksum.name == "accepted-run.surrogate-run.sha256.json"
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        manifest = json.loads(bundle.read("bundle.json"))
    assert "run/accepted-run/status.json" in names
    assert "run/accepted-run/weights.bin" in names
    assert manifest["schema_version"] == 1
    assert manifest["model_kind"] == "elasticity2d"
    assert manifest["run_id"] == "accepted-run"
    assert manifest["export_repo_commit"]
    assert isinstance(manifest["export_repo_dirty"], bool)
    assert [item["path"] for item in manifest["files"]] == [
        "status.json",
        "weights.bin",
    ]
    sidecar = json.loads(checksum.read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == 1
    assert sidecar["archive_name"] == archive.name
    assert sidecar["archive_bytes"] == archive.stat().st_size
    assert sidecar["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_bundle_export_rejects_nonaccepted_and_existing_output(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    output = tmp_path / "output"
    run_dir.mkdir()
    output.mkdir()
    rejected = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        "$verification=[pscustomobject]@{status='rejected'};"
        f"New-RunBundleArchive -RunDir {_ps_quote(run_dir)} -ModelKind 'scalar' "
        f"-OutputDirectory {_ps_quote(output)} -RepositoryRoot {_ps_quote(ROOT)} "
        "-Verification $verification"
    )
    assert rejected.returncode != 0
    assert "accepted" in rejected.stderr

    (output / "run.surrogate-run.zip").write_bytes(b"existing")
    conflict = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        "$verification=[pscustomobject]@{status='accepted'};"
        f"New-RunBundleArchive -RunDir {_ps_quote(run_dir)} -ModelKind 'scalar' "
        f"-OutputDirectory {_ps_quote(output)} -RepositoryRoot {_ps_quote(ROOT)} "
        "-Verification $verification"
    )
    assert conflict.returncode != 0
    assert "已存在" in conflict.stderr


def test_bundle_export_rejects_unsafe_run_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "unsafe run"
    output = tmp_path / "output"
    run_dir.mkdir()
    output.mkdir()
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        "$verification=[pscustomobject]@{status='accepted'};"
        f"New-RunBundleArchive -RunDir {_ps_quote(run_dir)} -ModelKind 'scalar' "
        f"-OutputDirectory {_ps_quote(output)} -RepositoryRoot {_ps_quote(ROOT)} "
        "-Verification $verification"
    )
    assert completed.returncode != 0
    assert "run_id" in completed.stderr
    assert list(output.iterdir()) == []


def test_export_script_validates_before_packaging_and_has_no_arbitrary_command() -> (
    None
):
    content = (TOOLS / "Export-AcceptedRun.ps1").read_text(encoding="utf-8")
    assert content.index("Invoke-ModelVerification") < content.index(
        "New-RunBundleArchive"
    )
    assert "ValidateSet('scalar', 'heat1d', 'elasticity2d')" in content
    assert "Invoke-Expression" not in content
