from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path

from surrogate_loop.operator.field_data import sha256_file


class CavityRunState(StrEnum):
    PLANNED = "planned"
    DATA_VERIFIED = "data_verified"
    MODEL_SELECTED = "model_selected"
    FROZEN = "frozen"
    DEVELOPMENT_COMPLETE = "development_complete"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_cavity_state(run_dir: Path, state: CavityRunState) -> None:
    _write_json_atomic(run_dir / "status.json", {"status": state.value})


def read_cavity_state(run_dir: Path) -> CavityRunState:
    payload = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    return CavityRunState(payload["status"])


def _hash_files(run_dir: Path, files: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in sorted(set(files)):
        path = run_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"required cavity artifact is missing: {name}")
        hashes[name] = sha256_file(path)
    return hashes


def freeze_cavity_run(run_dir: Path, files: list[str], *, mode: str) -> None:
    if read_cavity_state(run_dir) != CavityRunState.MODEL_SELECTED:
        raise RuntimeError("cavity run must be model_selected before freeze")
    if mode not in {"smoke", "full"}:
        raise ValueError("only smoke or full runs can be frozen")
    _write_json_atomic(
        run_dir / "freeze_manifest.json",
        {
            "schema_version": 1,
            "mode": mode,
            "files": _hash_files(run_dir, files),
        },
    )
    write_cavity_state(
        run_dir,
        CavityRunState.DEVELOPMENT_COMPLETE
        if mode == "smoke"
        else CavityRunState.FROZEN,
    )


def consume_sealed_test_once(run_dir: Path) -> None:
    if read_cavity_state(run_dir) != CavityRunState.FROZEN:
        raise RuntimeError("sealed test requires a frozen Full run")
    marker = run_dir / "sealed-test-consumed.json"
    if marker.exists():
        raise RuntimeError("sealed test was already consumed")
    _write_json_atomic(marker, {"consumed": True})


def write_artifact_manifest(run_dir: Path, files: list[str]) -> Path:
    path = run_dir / "artifact_manifest.json"
    _write_json_atomic(
        path,
        {"schema_version": 1, "files": _hash_files(run_dir, files)},
    )
    return path


def verify_artifact_manifest(run_dir: Path) -> dict[str, str]:
    payload = json.loads(
        (run_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("files"), dict)
    ):
        raise RuntimeError("cavity artifact manifest schema is invalid")
    hashes = payload["files"]
    for name, expected in hashes.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise RuntimeError("cavity artifact manifest entry is invalid")
        if sha256_file(run_dir / name) != expected:
            raise RuntimeError(f"cavity artifact SHA-256 mismatch: {name}")
    return dict(hashes)


__all__ = [
    "CavityRunState",
    "consume_sealed_test_once",
    "freeze_cavity_run",
    "read_cavity_state",
    "verify_artifact_manifest",
    "write_artifact_manifest",
    "write_cavity_state",
]
