from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from surrogate_loop.operator.cavity2d.protocol import (
    import_verified_cavity_dataset,
)
from surrogate_loop.operator.field_data import load_field_dataset


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def verified_cavity_pipeline(tmp_path: Path) -> Path:
    root = tmp_path / "fluent-pipeline"
    root.mkdir()
    request = root / "solver-request.json"
    samples = [
        {"sample_id": "s0", "reynolds": 10.0, "split": "train"},
        {"sample_id": "s1", "reynolds": 100.0, "split": "validation"},
        {"sample_id": "s2", "reynolds": 400.0, "split": "development_test"},
    ]
    request.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "problem_id": "fluent_lid_driven_cavity_steady_v1",
                "request_id": "synthetic",
                "mesh_sha256": (
                    "9B09F1287DB71978E10C67A528616C1C95118CFCF4F763ABAD047DF565E6A6DD"
                ),
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    batch_root = root / "batch-000"
    (batch_root / "fields").mkdir(parents=True)
    archive = batch_root / "fields" / "batch-fields.npz"
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    fields = np.zeros((3, 3, 3), dtype=np.float64)
    for index, reynolds in enumerate([10.0, 100.0, 400.0]):
        fields[index, :, 0] = reynolds * coordinates[:, 0]
        fields[index, :, 1] = reynolds * coordinates[:, 1]
        fields[index, :, 2] = np.array([-1.0, 0.0, 1.0])
    np.savez_compressed(
        archive,
        sample_ids=np.array(["s0", "s1", "s2"]),
        parameters=np.array([[10.0], [100.0], [400.0]]),
        cell_ids=np.array([0, 1, 2]),
        coordinates=coordinates,
        fields=fields,
    )
    sample_rows = []
    artifact_paths = {"fields": archive}
    for index, sample in enumerate(samples):
        sample_id = sample["sample_id"]
        solver_root = batch_root / "solver" / sample_id
        evidence_root = batch_root / "evidence" / sample_id
        solver_root.mkdir(parents=True)
        evidence_root.mkdir(parents=True)
        case = solver_root / f"{sample_id}.cas.h5"
        data = solver_root / f"{sample_id}.dat.h5"
        setup = evidence_root / "setup-audit.json"
        transcript = evidence_root / "solver-transcript.trn"
        acceptance_path = evidence_root / "acceptance.json"
        case.write_bytes(b"case")
        data.write_bytes(b"data")
        setup.write_text('{"setup_exact":true}', encoding="utf-8")
        transcript.write_text("20 1e-7 1e-8 1e-8\n", encoding="utf-8")
        acceptance = {
            "ok": True,
            "ready_for_surrogate_dataset": True,
            "setup_exact": True,
            "initialized": True,
            "iterations": 20,
            "residuals": {
                "continuity": 1e-7,
                "x-velocity": 1e-8,
                "y-velocity": 1e-8,
            },
            "residuals_ok": True,
            "fields_finite": True,
            "artifacts_nonempty": True,
            "fatal_markers": [],
        }
        acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")
        sample_rows.append(
            {
                **sample,
                "wall_time_seconds": 1.0,
                "case": str(case),
                "data": str(data),
                "setup_audit": str(setup),
                "transcript": str(transcript),
                "acceptance": acceptance,
            }
        )
        artifact_paths.update(
            {
                f"sample_{index:03d}_case": case,
                f"sample_{index:03d}_data": data,
                f"sample_{index:03d}_setup_audit": setup,
                f"sample_{index:03d}_transcript": transcript,
                f"sample_{index:03d}_acceptance": acceptance_path,
            }
        )
    artifact_sha256 = {
        name: sha256(path) for name, path in artifact_paths.items()
    }
    launch_report = batch_root / "evidence" / "launch_report.json"
    launch_report.write_text(
        json.dumps(
            {
                "process_started": True,
                "connected": True,
                "exit_confirmed": True,
                "fluent_version_reported": "Ansys Fluent 2024 R1",
            }
        ),
        encoding="utf-8",
    )
    runner_acceptance = {
        "ok": True,
        "ready_for_surrogate_dataset": True,
        "batch_index": 0,
        "samples": sample_rows,
        "artifact_sha256": artifact_sha256,
    }
    (batch_root / "acceptance.json").write_text(
        json.dumps(runner_acceptance),
        encoding="utf-8",
    )
    (batch_root / "evidence" / "acceptance.json").write_text(
        json.dumps(runner_acceptance),
        encoding="utf-8",
    )
    (batch_root / "checkpoint.json").write_text(
        json.dumps(
            {
                "ok": True,
                "workflow": "lid_driven_cavity.solve_batch",
                "stage": "complete",
                "artifacts": {
                    "fields": str(archive),
                    "launch_report": str(launch_report),
                },
            }
        ),
        encoding="utf-8",
    )
    (batch_root / "process_audit.json").write_text(
        json.dumps({"ownership": {"remaining_pids": []}}),
        encoding="utf-8",
    )
    (batch_root / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": "lid_driven_cavity.solve_batch",
                "inputs": {"solver_request": str(request)},
                "parameters": {"batch_index": 0},
                "output_root": str(batch_root),
            }
        ),
        encoding="utf-8",
    )
    (batch_root / "evidence" / "checkpoint.json").write_text(
        json.dumps(
            {
                "ok": True,
                "request_sha256": sha256(request),
                "mesh_sha256": (
                    "9B09F1287DB71978E10C67A528616C1C95118CFCF4F763ABAD047DF565E6A6DD"
                ),
                "batch_index": 0,
                "pyfluent_version": "0.37.0",
            }
        ),
        encoding="utf-8",
    )
    runner_evidence_paths = {
        "job": batch_root / "job.json",
        "runner_acceptance": batch_root / "acceptance.json",
        "runner_checkpoint": batch_root / "checkpoint.json",
        "process_audit": batch_root / "process_audit.json",
        "launch_report": launch_report,
        "workflow_checkpoint": batch_root / "evidence" / "checkpoint.json",
        "workflow_acceptance": batch_root / "evidence" / "acceptance.json",
    }
    reload_root = root / "reload-000"
    reload_evidence = reload_root / "evidence"
    reload_evidence.mkdir(parents=True)
    reload_acceptance = {
        "ok": True,
        "ready_for_reload": True,
        "setup_exact": True,
        "iteration_position_matches": True,
        "expected_iterations": 20,
        "actual_iterations": 20,
        "data_valid": True,
        "fields_readable": True,
    }
    reload_audit = reload_evidence / "reload-audit.json"
    reload_audit.write_text(
        json.dumps(
            {
                "ok": True,
                "case": sample_rows[0]["case"],
                "data": sample_rows[0]["data"],
                "expected_reynolds": sample_rows[0]["reynolds"],
                "acceptance": reload_acceptance,
            }
        ),
        encoding="utf-8",
    )
    reload_launch_report = reload_evidence / "launch_report.json"
    reload_launch_report.write_text(
        json.dumps(
            {
                "process_started": True,
                "connected": True,
                "exit_confirmed": True,
            }
        ),
        encoding="utf-8",
    )
    (reload_root / "acceptance.json").write_text(
        json.dumps(reload_acceptance),
        encoding="utf-8",
    )
    (reload_root / "checkpoint.json").write_text(
        json.dumps(
            {
                "ok": True,
                "workflow": "lid_driven_cavity.reload_audit",
                "stage": "complete",
                "artifacts": {
                    "reload_audit": str(reload_audit),
                    "launch_report": str(reload_launch_report),
                },
            }
        ),
        encoding="utf-8",
    )
    (reload_root / "process_audit.json").write_text(
        json.dumps({"ownership": {"remaining_pids": []}}),
        encoding="utf-8",
    )
    complete = {
        "schema_version": 2,
        "status": "complete",
        "problem_id": "fluent_lid_driven_cavity_steady_v1",
        "solver_request": str(request),
        "solver_request_sha256": sha256(request),
        "mesh_sha256": (
            "9B09F1287DB71978E10C67A528616C1C95118CFCF4F763ABAD047DF565E6A6DD"
        ),
        "batches": [
            {
                "stage": "batch-000",
                "ok": True,
                "batch_index": 0,
                "run_root": str(batch_root),
                "fields": str(archive),
                "fields_sha256": sha256(archive),
                "fluent_version": "Ansys Fluent 2024 R1",
                "pyfluent_version": "0.37.0",
                "samples": sample_rows,
                "artifact_sha256": artifact_sha256,
                "runner_evidence_sha256": {
                    name: sha256(path)
                    for name, path in runner_evidence_paths.items()
                },
            }
        ],
        "reload_audits": [
            {
                "stage": "reload-audit-000",
                "batch_index": 0,
                "ok": True,
                "run_root": str(reload_root),
                "reload_audit": str(reload_audit),
                "reload_audit_sha256": sha256(reload_audit),
            }
        ],
    }
    (root / "pipeline-complete.json").write_text(
        json.dumps(complete),
        encoding="utf-8",
    )
    return root


def test_import_rejects_incomplete_pipeline(tmp_path: Path) -> None:
    complete = tmp_path / "pipeline-complete.json"
    complete.write_text('{"status":"pending"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="complete"):
        import_verified_cavity_dataset(complete, tmp_path / "run")


def test_imports_verified_batches_to_safe_field_dataset(
    verified_cavity_pipeline: Path,
    tmp_path: Path,
) -> None:
    files = import_verified_cavity_dataset(
        verified_cavity_pipeline / "pipeline-complete.json",
        tmp_path / "run",
    )

    dataset = load_field_dataset(files.development)
    assert dataset.sample_ids.tolist() == ["s0", "s1", "s2"]
    assert dataset.fields.shape == (3, 3, 3)
    assert files.sealed_test is None
    assert files.manifest.is_file()


def test_import_accepts_equivalent_hex_digest_case(
    verified_cavity_pipeline: Path,
    tmp_path: Path,
) -> None:
    complete = verified_cavity_pipeline / "pipeline-complete.json"
    payload = json.loads(complete.read_text(encoding="utf-8"))
    payload["mesh_sha256"] = str(payload["mesh_sha256"]).lower()
    complete.write_text(json.dumps(payload), encoding="utf-8")

    files = import_verified_cavity_dataset(complete, tmp_path / "run")

    assert files.development.is_file()


def test_import_rejects_coordinate_or_hash_mismatch(
    verified_cavity_pipeline: Path,
    tmp_path: Path,
) -> None:
    archive = next(verified_cavity_pipeline.rglob("batch-fields.npz"))
    with np.load(archive, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    arrays["coordinates"][0, 0] += 0.1
    np.savez_compressed(archive, **arrays)

    with pytest.raises(RuntimeError, match="SHA-256|coordinate"):
        import_verified_cavity_dataset(
            verified_cavity_pipeline / "pipeline-complete.json",
            tmp_path / "run",
        )


def test_import_rejects_unknown_pipeline_fields(
    verified_cavity_pipeline: Path,
    tmp_path: Path,
) -> None:
    complete = verified_cavity_pipeline / "pipeline-complete.json"
    payload = json.loads(complete.read_text(encoding="utf-8"))
    payload["unreviewed"] = True
    complete.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="schema mismatch"):
        import_verified_cavity_dataset(complete, tmp_path / "run")


@pytest.mark.parametrize("schema_version", [1, 3])
def test_import_rejects_pipeline_version_that_does_not_match_v2_reload_schema(
    verified_cavity_pipeline: Path,
    tmp_path: Path,
    schema_version: int,
) -> None:
    complete = verified_cavity_pipeline / "pipeline-complete.json"
    payload = json.loads(complete.read_text(encoding="utf-8"))
    payload["schema_version"] = schema_version
    complete.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="reload evidence"):
        import_verified_cavity_dataset(complete, tmp_path / "run")


def test_import_rejects_tampered_case_even_when_fields_are_unchanged(
    verified_cavity_pipeline: Path,
    tmp_path: Path,
) -> None:
    case = next(verified_cavity_pipeline.rglob("*.cas.h5"))
    case.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="artifact mismatch"):
        import_verified_cavity_dataset(
            verified_cavity_pipeline / "pipeline-complete.json",
            tmp_path / "run",
        )
