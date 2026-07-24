import json
from pathlib import Path

import pytest

from surrogate_loop.operator.cavity2d.artifacts import (
    CavityRunState,
    consume_sealed_test_once,
    freeze_cavity_run,
    read_cavity_state,
    verify_artifact_manifest,
    write_artifact_manifest,
    write_cavity_state,
)


def test_freeze_hashes_files_and_sealed_test_is_one_time(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "model.json").write_text("{}", encoding="utf-8")
    (tmp_path / "spec.json").write_text("{}", encoding="utf-8")
    write_cavity_state(tmp_path, CavityRunState.MODEL_SELECTED)

    freeze_cavity_run(
        tmp_path,
        ["spec.json", "model/model.json"],
        mode="full",
    )
    consume_sealed_test_once(tmp_path)

    assert read_cavity_state(tmp_path) == CavityRunState.FROZEN
    with pytest.raises(RuntimeError, match="already"):
        consume_sealed_test_once(tmp_path)


def test_smoke_freeze_finishes_as_development_only(tmp_path: Path) -> None:
    (tmp_path / "spec.json").write_text("{}", encoding="utf-8")
    write_cavity_state(tmp_path, CavityRunState.MODEL_SELECTED)

    freeze_cavity_run(tmp_path, ["spec.json"], mode="smoke")

    assert read_cavity_state(tmp_path) == CavityRunState.DEVELOPMENT_COMPLETE


def test_final_manifest_detects_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "model.json"
    artifact.write_text(json.dumps({"model": 1}), encoding="utf-8")
    write_artifact_manifest(tmp_path, ["model.json"])
    verify_artifact_manifest(tmp_path)
    artifact.write_text(json.dumps({"model": 2}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256"):
        verify_artifact_manifest(tmp_path)
