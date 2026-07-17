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


def _make_bundle(base: Path) -> tuple[Path, Path]:
    run_dir = base / "accepted-run"
    output = base / "output"
    run_dir.mkdir(parents=True)
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
    payload = json.loads(completed.stdout)
    return Path(payload["archive_path"]), Path(payload["checksum_path"])


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


def test_verified_bundle_expands_only_to_owned_staging(tmp_path: Path) -> None:
    archive, checksum = _make_bundle(tmp_path)
    runs = tmp_path / "target-runs"
    runs.mkdir()
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Expand-VerifiedRunBundle -ArchivePath {_ps_quote(archive)} "
        f"-ChecksumPath {_ps_quote(checksum)} -RunsDirectory {_ps_quote(runs)} "
        f"-TargetRepositoryRoot {_ps_quote(ROOT)} | "
        "ConvertTo-Json -Depth 20 -Compress"
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    staging = Path(payload["staging_root"])
    assert staging.parent == runs
    assert staging.name.startswith(".migration-staging-")
    assert (Path(payload["run_dir"]) / "weights.bin").read_bytes() == b"weights"


def test_verified_bundle_can_publish_then_remove_empty_staging(tmp_path: Path) -> None:
    archive, checksum = _make_bundle(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"$expanded=Expand-VerifiedRunBundle -ArchivePath {_ps_quote(archive)} "
        f"-ChecksumPath {_ps_quote(checksum)} -RunsDirectory {_ps_quote(runs)} "
        f"-TargetRepositoryRoot {_ps_quote(ROOT)};"
        "$published=Publish-ImportedRun -ExpandedBundle $expanded;"
        f"Remove-OwnedStagingDirectory -Path $expanded.staging_root "
        f"-RunsDirectory {_ps_quote(runs)};"
        "$published | ConvertTo-Json -Depth 10 -Compress"
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    final = Path(payload["run_dir"])
    assert final == runs / "accepted-run"
    assert (final / "weights.bin").read_bytes() == b"weights"
    assert not any(path.name.startswith(".migration-staging-") for path in runs.iterdir())


def test_bundle_import_rejects_tampered_archive_and_target_conflict(
    tmp_path: Path,
) -> None:
    archive, checksum = _make_bundle(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"tamper")
    runs = tmp_path / "runs"
    runs.mkdir()
    tampered = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Expand-VerifiedRunBundle -ArchivePath {_ps_quote(archive)} "
        f"-ChecksumPath {_ps_quote(checksum)} -RunsDirectory {_ps_quote(runs)} "
        f"-TargetRepositoryRoot {_ps_quote(ROOT)}"
    )
    assert tampered.returncode != 0
    assert "SHA-256" in tampered.stderr

    archive, checksum = _make_bundle(tmp_path / "fresh")
    (runs / "accepted-run").mkdir()
    conflict = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Expand-VerifiedRunBundle -ArchivePath {_ps_quote(archive)} "
        f"-ChecksumPath {_ps_quote(checksum)} -RunsDirectory {_ps_quote(runs)} "
        f"-TargetRepositoryRoot {_ps_quote(ROOT)}"
    )
    assert conflict.returncode != 0
    assert "已存在" in conflict.stderr
    assert not any(path.name.startswith(".migration-staging-") for path in runs.iterdir())


def test_owned_staging_cleanup_refuses_unowned_paths(tmp_path: Path) -> None:
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Remove-OwnedStagingDirectory -Path {_ps_quote(ordinary)} "
        f"-RunsDirectory {_ps_quote(tmp_path)}"
    )
    assert completed.returncode != 0
    assert ordinary.exists()


def test_import_script_verifies_before_publish_and_cleans_in_finally() -> None:
    content = (TOOLS / "Import-AcceptedRun.ps1").read_text(encoding="utf-8")
    assert content.index("Expand-VerifiedRunBundle") < content.index(
        "Invoke-ModelVerification"
    )
    assert content.index("Invoke-ModelVerification") < content.index(
        "Publish-ImportedRun"
    )
    assert "finally" in content
    assert "Remove-OwnedStagingDirectory" in content
    assert "Invoke-Expression" not in content


@pytest.mark.parametrize(
    "entries",
    (
        {"bundle.json": "{}", "../escape.txt": "bad"},
        {
            "bundle.json": "{}",
            "run/id/A.txt": "one",
            "run/id/a.txt": "two",
        },
    ),
)
def test_bundle_import_rejects_unsafe_or_case_duplicate_entries(
    tmp_path: Path,
    entries: dict[str, str],
) -> None:
    archive = tmp_path / "malicious.surrogate-run.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, value in entries.items():
            bundle.writestr(name, value)
    checksum = tmp_path / "malicious.surrogate-run.sha256.json"
    checksum.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "archive_name": archive.name,
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    runs.mkdir()
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Expand-VerifiedRunBundle -ArchivePath {_ps_quote(archive)} "
        f"-ChecksumPath {_ps_quote(checksum)} -RunsDirectory {_ps_quote(runs)} "
        f"-TargetRepositoryRoot {_ps_quote(ROOT)}"
    )
    assert completed.returncode != 0
    assert "ZIP" in completed.stderr
