from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from surrogate_loop.operator.field_data import (
    FieldDataset,
    save_field_dataset,
    sha256_file,
)

PROBLEM_ID = "fluent_lid_driven_cavity_steady_v1"
REQUIRED_PIPELINE_KEYS = {
    "status",
    "problem_id",
    "solver_request",
    "solver_request_sha256",
    "mesh_sha256",
    "batches",
}
OPTIONAL_PIPELINE_KEYS = {
    "schema_version",
    "mesh",
    "failed_attempts",
    "reload_audit",
    "reload_audits",
}
REQUEST_KEYS = {
    "schema_version",
    "problem_id",
    "request_id",
    "mesh_sha256",
    "samples",
}
REQUEST_SAMPLE_KEYS = {"sample_id", "reynolds", "split"}
BATCH_KEYS = {
    "stage",
    "ok",
    "batch_index",
    "run_root",
    "fields",
    "fields_sha256",
    "samples",
    "artifact_sha256",
}
BATCH_V2_KEYS = {
    "fluent_version",
    "pyfluent_version",
    "runner_evidence_sha256",
}
BATCH_SAMPLE_KEYS = {
    "sample_id",
    "reynolds",
    "split",
    "wall_time_seconds",
    "case",
    "data",
    "setup_audit",
    "transcript",
    "acceptance",
}
SAMPLE_ACCEPTANCE_KEYS = {
    "ok",
    "ready_for_surrogate_dataset",
    "setup_exact",
    "initialized",
    "iterations",
    "residuals",
    "residuals_ok",
    "fields_finite",
    "artifacts_nonempty",
    "fatal_markers",
}
REQUIRED_ARCHIVE_KEYS = {
    "sample_ids",
    "parameters",
    "cell_ids",
    "coordinates",
    "fields",
}


def _require_exact_keys(
    payload: dict[str, object],
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
    *,
    label: str,
) -> None:
    actual = set(payload)
    if not required <= actual or not actual <= required | optional:
        raise RuntimeError(f"{label} schema mismatch")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _digest_equal(actual: str, expected: str) -> bool:
    digest = re.compile(r"^[0-9a-fA-F]{64}$")
    return (
        bool(digest.fullmatch(actual))
        and bool(digest.fullmatch(expected))
        and actual.lower() == expected.lower()
    )


@dataclass(frozen=True)
class CavityDatasetFiles:
    development: Path
    sealed_test: Path | None
    manifest: Path
    coordinates_sha256: str


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read verified JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _coordinates_sha256(coordinates: np.ndarray) -> str:
    values = np.ascontiguousarray(coordinates, dtype=np.float64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _load_batch(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if not _digest_equal(sha256_file(path), expected_sha256):
        raise RuntimeError("batch field SHA-256 mismatch")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != REQUIRED_ARCHIVE_KEYS:
                raise RuntimeError("batch field archive schema mismatch")
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in sorted(REQUIRED_ARCHIVE_KEYS)
            }
    except RuntimeError:
        raise
    except (OSError, ValueError) as exc:
        raise RuntimeError("cannot load safe batch field archive") from exc
    sample_ids = arrays["sample_ids"]
    parameters = np.asarray(arrays["parameters"], dtype=np.float64)
    raw_cell_ids = arrays["cell_ids"]
    if raw_cell_ids.dtype.kind not in {"i", "u"}:
        raise RuntimeError("batch cell IDs must be integer values")
    cell_ids = np.asarray(raw_cell_ids, dtype=np.int64)
    coordinates = np.asarray(arrays["coordinates"], dtype=np.float64)
    fields = np.asarray(arrays["fields"], dtype=np.float64)
    sample_count = sample_ids.shape[0] if sample_ids.ndim == 1 else -1
    if (
        sample_count <= 0
        or parameters.shape != (sample_count, 1)
        or cell_ids.ndim != 1
        or coordinates.shape != (cell_ids.size, 2)
        or fields.shape != (sample_count, cell_ids.size, 3)
    ):
        raise RuntimeError("batch field array shapes are invalid")
    if sample_ids.dtype.kind not in {"U", "S"}:
        raise RuntimeError("batch sample IDs must not require pickle")
    if (
        len(set(np.asarray(sample_ids, dtype=np.str_).tolist())) != sample_count
        or not np.array_equal(cell_ids, np.arange(cell_ids.size, dtype=np.int64))
        or not np.array_equal(
            np.lexsort((coordinates[:, 0], coordinates[:, 1])),
            np.arange(coordinates.shape[0], dtype=np.int64),
        )
        or np.any(coordinates < 0.0)
        or np.any(coordinates > 1.0)
        or np.any(parameters[:, 0] < 10.0)
        or np.any(parameters[:, 0] > 400.0)
    ):
        raise RuntimeError("batch fixed mesh or sample identity is invalid")
    if not all(
        np.isfinite(values).all() for values in (parameters, coordinates, fields)
    ):
        raise RuntimeError("batch field arrays must be finite")
    if not np.allclose(fields[:, :, 2].mean(axis=1), 0.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("batch pressure fields are not mean-free")
    return {
        "sample_ids": np.asarray(sample_ids, dtype=np.str_),
        "parameters": parameters,
        "cell_ids": cell_ids,
        "coordinates": coordinates,
        "fields": fields,
    }


def _validate_sample_acceptance(
    acceptance: object,
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(acceptance, dict):
        raise RuntimeError(f"{label} acceptance is invalid")
    _require_exact_keys(
        acceptance,
        SAMPLE_ACCEPTANCE_KEYS,
        label=f"{label} acceptance",
    )
    residuals = acceptance.get("residuals")
    fatal_markers = acceptance.get("fatal_markers")
    if (
        acceptance.get("ok") is not True
        or acceptance.get("ready_for_surrogate_dataset") is not True
        or acceptance.get("setup_exact") is not True
        or acceptance.get("initialized") is not True
        or acceptance.get("residuals_ok") is not True
        or acceptance.get("fields_finite") is not True
        or acceptance.get("artifacts_nonempty") is not True
        or not isinstance(acceptance.get("iterations"), int)
        or int(acceptance["iterations"]) <= 0
        or int(acceptance["iterations"]) > 5000
        or not isinstance(residuals, dict)
        or set(residuals) != {"continuity", "x-velocity", "y-velocity"}
        or not all(
            isinstance(value, (int, float))
            and np.isfinite(value)
            and float(value) <= 1e-6
            for value in residuals.values()
        )
        or fatal_markers != []
    ):
        raise RuntimeError(f"{label} acceptance is not complete")
    return acceptance


def _artifact_paths(
    batch: dict[str, object],
    rows: list[dict[str, object]],
) -> dict[str, Path]:
    paths = {"fields": Path(str(batch["fields"]))}
    run_root = Path(str(batch["run_root"]))
    for index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        paths.update(
            {
                f"sample_{index:03d}_case": Path(str(row["case"])),
                f"sample_{index:03d}_data": Path(str(row["data"])),
                f"sample_{index:03d}_setup_audit": Path(
                    str(row["setup_audit"])
                ),
                f"sample_{index:03d}_transcript": Path(str(row["transcript"])),
                f"sample_{index:03d}_acceptance": (
                    run_root / "evidence" / sample_id / "acceptance.json"
                ),
            }
        )
    return paths


def _validate_batch_runner_evidence(
    batch: dict[str, object],
    *,
    request_path: Path,
    request_sha256: str,
    mesh_sha256: str,
    rows: list[dict[str, object]],
    schema_version: int,
) -> None:
    run_root = Path(str(batch["run_root"]))
    evidence_paths = {
        "job": run_root / "job.json",
        "runner_acceptance": run_root / "acceptance.json",
        "runner_checkpoint": run_root / "checkpoint.json",
        "process_audit": run_root / "process_audit.json",
        "launch_report": run_root / "evidence" / "launch_report.json",
        "workflow_checkpoint": run_root / "evidence" / "checkpoint.json",
        "workflow_acceptance": run_root / "evidence" / "acceptance.json",
    }
    if not all(
        _inside(path, run_root) and path.is_file() and path.stat().st_size > 0
        for path in evidence_paths.values()
    ):
        raise RuntimeError("Fluent batch Runner evidence is missing")
    if schema_version == 2:
        hashes = batch.get("runner_evidence_sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(evidence_paths):
            raise RuntimeError("Fluent batch Runner manifest schema mismatch")
        for name, path in evidence_paths.items():
            expected = hashes.get(name)
            if not isinstance(expected, str) or not _digest_equal(
                sha256_file(path),
                expected,
            ):
                raise RuntimeError(f"Fluent batch Runner artifact mismatch: {name}")

    job = _read_json(evidence_paths["job"])
    inputs = job.get("inputs")
    parameters = job.get("parameters")
    if (
        job.get("schema_version") != 1
        or job.get("workflow") != "lid_driven_cavity.solve_batch"
        or not isinstance(inputs, dict)
        or not isinstance(inputs.get("solver_request"), str)
        or Path(str(inputs["solver_request"])).resolve() != request_path.resolve()
        or not isinstance(parameters, dict)
        or parameters.get("batch_index") != batch["batch_index"]
        or job.get("output_root") != str(run_root)
    ):
        raise RuntimeError("Fluent batch Runner job identity mismatch")

    runner_acceptance = _read_json(evidence_paths["runner_acceptance"])
    workflow_acceptance = _read_json(evidence_paths["workflow_acceptance"])
    if (
        runner_acceptance.get("ok") is not True
        or runner_acceptance.get("ready_for_surrogate_dataset") is not True
        or runner_acceptance.get("batch_index") != batch["batch_index"]
        or runner_acceptance.get("samples") != rows
        or runner_acceptance.get("artifact_sha256")
        != batch["artifact_sha256"]
        or workflow_acceptance != runner_acceptance
    ):
        raise RuntimeError("Fluent batch Runner acceptance mismatch")

    checkpoint = _read_json(evidence_paths["runner_checkpoint"])
    artifacts = checkpoint.get("artifacts")
    process = _read_json(evidence_paths["process_audit"])
    ownership = process.get("ownership")
    if (
        checkpoint.get("ok") is not True
        or checkpoint.get("workflow") != "lid_driven_cavity.solve_batch"
        or checkpoint.get("stage") != "complete"
        or not isinstance(artifacts, dict)
        or artifacts.get("fields") != batch["fields"]
        or artifacts.get("launch_report") != str(evidence_paths["launch_report"])
        or not isinstance(ownership, dict)
        or ownership.get("remaining_pids") != []
    ):
        raise RuntimeError("Fluent batch Runner completion evidence is invalid")

    launch = _read_json(evidence_paths["launch_report"])
    if (
        launch.get("process_started") is not True
        or launch.get("connected") is not True
        or launch.get("exit_confirmed") is not True
        or (
            schema_version == 2
            and launch.get("fluent_version_reported")
            != batch.get("fluent_version")
        )
    ):
        raise RuntimeError("Fluent batch launch evidence is incomplete")

    workflow_checkpoint = _read_json(evidence_paths["workflow_checkpoint"])
    if (
        workflow_checkpoint.get("ok") is not True
        or workflow_checkpoint.get("request_sha256") != request_sha256
        or not isinstance(workflow_checkpoint.get("mesh_sha256"), str)
        or not _digest_equal(
            str(workflow_checkpoint["mesh_sha256"]),
            mesh_sha256,
        )
        or workflow_checkpoint.get("batch_index") != batch["batch_index"]
        or (
            schema_version == 2
            and workflow_checkpoint.get("pyfluent_version")
            != batch.get("pyfluent_version")
        )
    ):
        raise RuntimeError("Fluent batch workflow identity evidence is invalid")


def _validate_reload_evidence(
    row: dict[str, object],
    *,
    batch_index: int,
    representative: dict[str, object],
    legacy: bool,
) -> None:
    required = {"ok", "run_root", "reload_audit", "stage"}
    if not legacy:
        required |= {"batch_index", "reload_audit_sha256"}
    _require_exact_keys(row, required, label="reload audit summary")
    expected_stage = "reload_audit" if legacy else f"reload-audit-{batch_index:03d}"
    if row.get("ok") is not True or row.get("stage") != expected_stage:
        raise RuntimeError("Fluent pipeline reload audit is not accepted")
    if not legacy and row.get("batch_index") != batch_index:
        raise RuntimeError("reload audit batch identity mismatch")
    run_root_value = row.get("run_root")
    audit_value = row.get("reload_audit")
    if not isinstance(run_root_value, str) or not isinstance(audit_value, str):
        raise RuntimeError("reload audit paths are invalid")
    run_root = Path(run_root_value)
    audit_path = Path(audit_value)
    if not _inside(audit_path, run_root) or not audit_path.is_file():
        raise RuntimeError("reload audit artifact is invalid")
    audit_hash = row.get("reload_audit_sha256")
    if audit_hash is not None and (
        not isinstance(audit_hash, str)
        or not _digest_equal(sha256_file(audit_path), audit_hash)
    ):
        raise RuntimeError("reload audit SHA-256 mismatch")

    audit = _read_json(audit_path)
    audit_acceptance = audit.get("acceptance")
    if (
        audit.get("ok") is not True
        or not isinstance(audit_acceptance, dict)
        or audit_acceptance.get("ok") is not True
        or audit_acceptance.get("ready_for_reload") is not True
        or audit_acceptance.get("setup_exact") is not True
        or audit_acceptance.get("iteration_position_matches") is not True
        or audit_acceptance.get("data_valid") is not True
        or audit_acceptance.get("fields_readable") is not True
        or audit.get("case") != representative.get("case")
        or audit.get("data") != representative.get("data")
        or audit.get("expected_reynolds") != representative.get("reynolds")
        or audit_acceptance.get("expected_iterations")
        != representative["acceptance"]["iterations"]
    ):
        raise RuntimeError("reload audit scientific evidence is incomplete")

    runner_acceptance = _read_json(run_root / "acceptance.json")
    checkpoint = _read_json(run_root / "checkpoint.json")
    process = _read_json(run_root / "process_audit.json")
    artifacts = checkpoint.get("artifacts")
    ownership = process.get("ownership")
    launch_value = artifacts.get("launch_report") if isinstance(artifacts, dict) else None
    if (
        runner_acceptance.get("ok") is not True
        or runner_acceptance.get("ready_for_reload") is not True
        or checkpoint.get("ok") is not True
        or checkpoint.get("workflow") != "lid_driven_cavity.reload_audit"
        or checkpoint.get("stage") != "complete"
        or not isinstance(artifacts, dict)
        or artifacts.get("reload_audit") != str(audit_path)
        or not isinstance(launch_value, str)
        or not isinstance(ownership, dict)
        or ownership.get("remaining_pids") != []
    ):
        raise RuntimeError("reload audit Runner evidence is incomplete")
    launch_path = Path(launch_value)
    if not _inside(launch_path, run_root):
        raise RuntimeError("reload audit launch evidence is invalid")
    launch = _read_json(launch_path)
    if (
        launch.get("process_started") is not True
        or launch.get("connected") is not True
        or launch.get("exit_confirmed") is not True
    ):
        raise RuntimeError("reload audit launch evidence is incomplete")


def import_verified_cavity_dataset(
    pipeline_complete: Path,
    run_dir: Path,
    *,
    expected_sample_ids: tuple[str, ...] | None = None,
    expected_reynolds: np.ndarray | None = None,
    expected_splits: np.ndarray | None = None,
    expected_mesh_sha256: str | None = None,
) -> CavityDatasetFiles:
    pipeline_complete = pipeline_complete.resolve()
    payload = _read_json(pipeline_complete)
    _require_exact_keys(
        payload,
        REQUIRED_PIPELINE_KEYS,
        OPTIONAL_PIPELINE_KEYS,
        label="pipeline-complete.json",
    )
    has_legacy_reload = "reload_audit" in payload
    has_reload_list = "reload_audits" in payload
    schema_version = payload.get("schema_version", 1)
    if (
        schema_version not in {1, 2}
        or has_legacy_reload == has_reload_list
        or (schema_version == 1 and not has_legacy_reload)
        or (schema_version == 2 and not has_reload_list)
    ):
        raise RuntimeError("pipeline-complete.json reload evidence is ambiguous")
    if payload.get("status") != "complete":
        raise RuntimeError("Fluent pipeline is not complete")
    if payload.get("problem_id") != PROBLEM_ID:
        raise RuntimeError("Fluent pipeline problem ID mismatch")
    failed_attempts = payload.get("failed_attempts", [])
    if not isinstance(failed_attempts, list):
        raise RuntimeError("Fluent failed-attempt evidence is invalid")
    for attempt in failed_attempts:
        if (
            not isinstance(attempt, dict)
            or set(attempt) != {"stage", "attempt", "run_root", "error"}
            or not isinstance(attempt.get("stage"), str)
            or not isinstance(attempt.get("attempt"), int)
            or int(attempt["attempt"]) <= 0
            or not isinstance(attempt.get("run_root"), str)
            or not isinstance(attempt.get("error"), str)
        ):
            raise RuntimeError("Fluent failed-attempt evidence is invalid")
    request_value = payload.get("solver_request")
    request_hash = payload.get("solver_request_sha256")
    if not isinstance(request_value, str) or not isinstance(request_hash, str):
        raise RuntimeError("Fluent solver request evidence is invalid")
    request_path = Path(request_value)
    if not _digest_equal(sha256_file(request_path), request_hash):
        raise RuntimeError("Fluent solver request SHA-256 mismatch")
    request = _read_json(request_path)
    _require_exact_keys(request, REQUEST_KEYS, label="Fluent solver request")
    if request.get("schema_version") != 1:
        raise RuntimeError("Fluent solver request schema version is unsupported")
    request_mesh_sha256 = request.get("mesh_sha256")
    pipeline_mesh_sha256 = payload.get("mesh_sha256")
    if (
        request.get("problem_id") != PROBLEM_ID
        or not isinstance(request_mesh_sha256, str)
        or not isinstance(pipeline_mesh_sha256, str)
        or not _digest_equal(request_mesh_sha256, pipeline_mesh_sha256)
    ):
        raise RuntimeError("Fluent request identity does not match pipeline")
    if expected_mesh_sha256 is not None and not _digest_equal(
        request_mesh_sha256,
        expected_mesh_sha256,
    ):
        raise RuntimeError("Fluent request mesh does not match configured mesh")
    request_samples = request.get("samples")
    if not isinstance(request_samples, list) or not request_samples:
        raise RuntimeError("Fluent request samples are invalid")
    request_rows: list[dict[str, object]] = []
    for row in request_samples:
        if not isinstance(row, dict):
            raise RuntimeError("Fluent request sample schema mismatch")
        _require_exact_keys(
            row,
            REQUEST_SAMPLE_KEYS,
            label="Fluent request sample",
        )
        if (
            not isinstance(row.get("sample_id"), str)
            or not isinstance(row.get("reynolds"), (int, float))
            or not np.isfinite(row["reynolds"])
            or not isinstance(row.get("split"), str)
        ):
            raise RuntimeError("Fluent request sample values are invalid")
        request_rows.append(row)
    request_ids = [str(row["sample_id"]) for row in request_rows]
    request_reynolds = np.asarray(
        [float(row["reynolds"]) for row in request_rows],
        dtype=np.float64,
    )
    request_splits = np.asarray([str(row["split"]) for row in request_rows])
    expected_values = (expected_sample_ids, expected_reynolds, expected_splits)
    if any(value is not None for value in expected_values):
        if not all(value is not None for value in expected_values):
            raise ValueError("expected cavity sample plan must be provided completely")
        assert expected_sample_ids is not None
        assert expected_reynolds is not None
        assert expected_splits is not None
        if (
            request_ids != list(expected_sample_ids)
            or not np.array_equal(request_reynolds, expected_reynolds)
            or not np.array_equal(request_splits, expected_splits)
        ):
            raise RuntimeError("Fluent request does not match configured sample plan")

    batches = payload.get("batches")
    if not isinstance(batches, list) or not batches:
        raise RuntimeError("Fluent pipeline batches are missing")
    batch_rows: list[dict[str, object]] = []
    for batch in batches:
        if not isinstance(batch, dict):
            raise RuntimeError("Fluent batch evidence is invalid")
        _require_exact_keys(
            batch,
            BATCH_KEYS,
            BATCH_V2_KEYS,
            label="Fluent batch evidence",
        )
        if schema_version == 2 and (
            not BATCH_V2_KEYS <= set(batch)
            or not all(
                isinstance(batch.get(name), str) and bool(batch[name])
                for name in {"fluent_version", "pyfluent_version"}
            )
            or not isinstance(batch.get("runner_evidence_sha256"), dict)
        ):
            raise RuntimeError("Fluent batch version evidence is invalid")
        batch_rows.append(batch)
    batch_arrays: list[dict[str, np.ndarray]] = []
    coordinate_reference: np.ndarray | None = None
    cell_reference: np.ndarray | None = None
    imported_sources: list[dict[str, object]] = []
    imported_root = run_dir / "imported"
    request_cursor = 0
    for expected_index, batch in enumerate(
        sorted(batch_rows, key=lambda row: int(row["batch_index"]))
    ):
        if (
            batch.get("ok") is not True
            or batch.get("batch_index") != expected_index
            or batch.get("stage") != f"batch-{expected_index:03d}"
            or not isinstance(batch.get("run_root"), str)
        ):
            raise RuntimeError("Fluent batch is not accepted or ordered")
        sample_values = batch.get("samples")
        if (
            not isinstance(sample_values, list)
            or not 1 <= len(sample_values) <= 8
            or request_cursor + len(sample_values) > len(request_rows)
        ):
            raise RuntimeError("Fluent batch samples are invalid")
        rows: list[dict[str, object]] = []
        for offset, row in enumerate(sample_values):
            if not isinstance(row, dict):
                raise RuntimeError("Fluent batch sample evidence is invalid")
            _require_exact_keys(
                row,
                BATCH_SAMPLE_KEYS,
                label="Fluent batch sample evidence",
            )
            expected_row = request_rows[request_cursor + offset]
            if (
                row.get("sample_id") != expected_row["sample_id"]
                or row.get("reynolds") != expected_row["reynolds"]
                or row.get("split") != expected_row["split"]
                or not isinstance(row.get("wall_time_seconds"), (int, float))
                or not np.isfinite(row["wall_time_seconds"])
                or float(row["wall_time_seconds"]) <= 0.0
            ):
                raise RuntimeError("Fluent batch sample identity mismatch")
            _validate_sample_acceptance(
                row.get("acceptance"),
                label=str(row["sample_id"]),
            )
            rows.append(row)
        request_cursor += len(rows)
        fields_value = batch.get("fields")
        fields_hash = batch.get("fields_sha256")
        if not isinstance(fields_value, str) or not isinstance(fields_hash, str):
            raise RuntimeError("Fluent batch field evidence is invalid")
        source_path = Path(fields_value)
        arrays = _load_batch(source_path, fields_hash)
        if arrays["sample_ids"].tolist() != [
            str(row["sample_id"]) for row in rows
        ]:
            raise RuntimeError("Fluent batch archive sample identity mismatch")
        run_root = Path(str(batch["run_root"]))
        artifact_hashes = batch.get("artifact_sha256")
        if not isinstance(artifact_hashes, dict):
            raise RuntimeError("Fluent batch artifact manifest is invalid")
        artifact_paths = _artifact_paths(batch, rows)
        if set(artifact_hashes) != set(artifact_paths):
            raise RuntimeError("Fluent batch artifact manifest schema mismatch")
        for name, artifact_path in artifact_paths.items():
            expected_hash = artifact_hashes.get(name)
            if (
                not isinstance(expected_hash, str)
                or not _inside(artifact_path, run_root)
                or not artifact_path.is_file()
                or artifact_path.stat().st_size <= 0
                or not _digest_equal(sha256_file(artifact_path), expected_hash)
            ):
                raise RuntimeError(f"Fluent batch artifact mismatch: {name}")
        if not _digest_equal(str(artifact_hashes["fields"]), fields_hash):
            raise RuntimeError("Fluent batch field manifest identity mismatch")
        for row in rows:
            acceptance_path = (
                run_root
                / "evidence"
                / str(row["sample_id"])
                / "acceptance.json"
            )
            if _read_json(acceptance_path) != row["acceptance"]:
                raise RuntimeError("Fluent sample acceptance file mismatch")
        _validate_batch_runner_evidence(
            batch,
            request_path=request_path,
            request_sha256=request_hash,
            mesh_sha256=pipeline_mesh_sha256,
            rows=rows,
            schema_version=schema_version,
        )
        if coordinate_reference is None:
            coordinate_reference = arrays["coordinates"]
            cell_reference = arrays["cell_ids"]
        elif not (
            np.array_equal(arrays["coordinates"], coordinate_reference)
            and np.array_equal(arrays["cell_ids"], cell_reference)
        ):
            raise RuntimeError("Fluent batch coordinate or cell identity mismatch")
        imported_sources.append(
            {
                "batch_index": expected_index,
                "source": str(source_path),
                "sha256": fields_hash,
            }
        )
        batch_arrays.append(arrays)
    if request_cursor != len(request_rows):
        raise RuntimeError("Fluent batches do not cover the complete sample plan")

    if has_legacy_reload:
        legacy_reload = payload.get("reload_audit")
        if len(batch_rows) != 1 or not isinstance(legacy_reload, dict):
            raise RuntimeError("legacy reload evidence requires exactly one batch")
        reload_rows = [legacy_reload]
        legacy_reload_mode = True
    else:
        reload_values = payload.get("reload_audits")
        if (
            not isinstance(reload_values, list)
            or len(reload_values) != len(batch_rows)
            or not all(isinstance(row, dict) for row in reload_values)
        ):
            raise RuntimeError("reload audit count does not match Fluent batches")
        reload_rows = list(reload_values)
        legacy_reload_mode = False
    for batch_index, reload_row in enumerate(reload_rows):
        _validate_reload_evidence(
            reload_row,
            batch_index=batch_index,
            representative=batch_rows[batch_index]["samples"][0],
            legacy=legacy_reload_mode,
        )

    run_dir.mkdir(parents=True, exist_ok=False)
    imported_root.mkdir()
    for source in imported_sources:
        batch_index = int(source["batch_index"])
        source_path = Path(str(source["source"]))
        copied = imported_root / f"batch-{batch_index:03d}.npz"
        shutil.copy2(source_path, copied)
        if not _digest_equal(
            sha256_file(copied),
            str(source["sha256"]),
        ):
            raise RuntimeError("copied Fluent batch SHA-256 mismatch")
        source["copied"] = str(copied)

    sample_ids = np.concatenate([row["sample_ids"] for row in batch_arrays])
    parameters = np.concatenate([row["parameters"] for row in batch_arrays])
    fields = np.concatenate([row["fields"] for row in batch_arrays])
    if sample_ids.tolist() != request_ids or not np.array_equal(
        parameters[:, 0],
        request_reynolds,
    ):
        raise RuntimeError("Fluent sample plan identity mismatch")
    assert coordinate_reference is not None
    coordinate_hash = _coordinates_sha256(coordinate_reference)
    sealed_mask = request_splits == "sealed_test"
    development_mask = ~sealed_mask
    if not development_mask.any():
        raise RuntimeError("verified dataset has no development samples")

    def dataset(mask: np.ndarray) -> FieldDataset:
        return FieldDataset(
            sample_ids=sample_ids[mask],
            parameters=parameters[mask],
            coordinates=coordinate_reference,
            fields=fields[mask],
            diagnostics={},
        )

    development_path = run_dir / "development.npz"
    development_sha256 = save_field_dataset(
        development_path,
        dataset(development_mask),
    )
    sealed_path: Path | None = None
    sealed_sha256: str | None = None
    if sealed_mask.any():
        sealed_path = run_dir / "sealed_test.npz"
        sealed_sha256 = save_field_dataset(sealed_path, dataset(sealed_mask))
    manifest_path = run_dir / "solver_provenance.json"
    manifest = {
        "problem_id": PROBLEM_ID,
        "mesh_sha256": payload["mesh_sha256"],
        "pipeline_complete": str(pipeline_complete),
        "solver_request": str(request_path),
        "solver_request_sha256": request_hash,
        "coordinates_sha256": coordinate_hash,
        "development": str(development_path),
        "development_sha256": development_sha256,
        "sealed_test": str(sealed_path) if sealed_path is not None else None,
        "sealed_test_sha256": sealed_sha256,
        "split_by_sample_id": dict(
            zip(request_ids, request_splits.tolist(), strict=True)
        ),
        "batches": imported_sources,
    }
    _write_json_atomic(manifest_path, manifest)
    return CavityDatasetFiles(
        development=development_path,
        sealed_test=sealed_path,
        manifest=manifest_path,
        coordinates_sha256=coordinate_hash,
    )


__all__ = ["CavityDatasetFiles", "import_verified_cavity_dataset"]
