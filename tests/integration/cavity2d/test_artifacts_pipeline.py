from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from surrogate_loop.operator.cavity2d.config import load_cavity_spec
from surrogate_loop.operator.cavity2d.pipeline import run_cavity_pipeline
from surrogate_loop.operator.cavity2d.sampling import build_cavity_sample_plan


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_synthetic_fluent_pipeline(tmp_path: Path, config: Path) -> Path:
    root = tmp_path / "fluent"
    root.mkdir()
    spec = load_cavity_spec(config)
    plan = build_cavity_sample_plan(spec)
    samples = [
        {
            "sample_id": sample_id,
            "reynolds": float(reynolds),
            "split": str(split),
        }
        for sample_id, reynolds, split in zip(
            plan.sample_ids,
            plan.reynolds,
            plan.split,
            strict=True,
        )
    ]
    request = root / "solver-request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "problem_id": spec.problem.problem_id,
                "request_id": f"synthetic-{spec.mode}",
                "mesh_sha256": spec.mesh_sha256,
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    axis = np.linspace(0.0, 1.0, 3)
    x, y = np.meshgrid(axis, axis)
    coordinates = np.column_stack((x.ravel(), y.ravel()))
    batches = []
    reload_audits = []
    for batch_index, start in enumerate(range(0, len(samples), 8)):
        selected = samples[start : start + 8]
        batch_root = root / f"batch-{batch_index:03d}"
        (batch_root / "fields").mkdir(parents=True)
        archive = batch_root / "fields" / "batch-fields.npz"
        fields = []
        for sample in selected:
            scale = np.log10(sample["reynolds"])
            u = scale * coordinates[:, 0] * (1.0 - coordinates[:, 1])
            v = -scale * coordinates[:, 1] * (1.0 - coordinates[:, 0])
            pressure = scale * (
                coordinates[:, 0] - coordinates[:, 0].mean()
            )
            fields.append(np.column_stack((u, v, pressure)))
        np.savez_compressed(
            archive,
            sample_ids=np.asarray([row["sample_id"] for row in selected]),
            parameters=np.asarray([[row["reynolds"]] for row in selected]),
            cell_ids=np.arange(coordinates.shape[0]),
            coordinates=coordinates,
            fields=np.stack(fields),
        )
        sample_rows = []
        artifact_paths = {"fields": archive}
        for sample_index, sample in enumerate(selected):
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
                    "wall_time_seconds": 10.0,
                    "case": str(case),
                    "data": str(data),
                    "setup_audit": str(setup),
                    "transcript": str(transcript),
                    "acceptance": acceptance,
                }
            )
            artifact_paths.update(
                {
                    f"sample_{sample_index:03d}_case": case,
                    f"sample_{sample_index:03d}_data": data,
                    f"sample_{sample_index:03d}_setup_audit": setup,
                    f"sample_{sample_index:03d}_transcript": transcript,
                    f"sample_{sample_index:03d}_acceptance": acceptance_path,
                }
            )
        artifact_sha256 = {
            name: sha256(path) for name, path in artifact_paths.items()
        }
        batch_launch = batch_root / "evidence" / "launch_report.json"
        batch_launch.write_text(
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
            "batch_index": batch_index,
            "samples": sample_rows,
            "artifact_sha256": artifact_sha256,
        }
        for path in (
            batch_root / "acceptance.json",
            batch_root / "evidence" / "acceptance.json",
        ):
            path.write_text(json.dumps(runner_acceptance), encoding="utf-8")
        (batch_root / "checkpoint.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "workflow": "lid_driven_cavity.solve_batch",
                    "stage": "complete",
                    "artifacts": {
                        "fields": str(archive),
                        "launch_report": str(batch_launch),
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
                    "parameters": {"batch_index": batch_index},
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
                    "mesh_sha256": spec.mesh_sha256,
                    "batch_index": batch_index,
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
            "launch_report": batch_launch,
            "workflow_checkpoint": batch_root / "evidence" / "checkpoint.json",
            "workflow_acceptance": batch_root / "evidence" / "acceptance.json",
        }
        batches.append(
            {
                "stage": f"batch-{batch_index:03d}",
                "ok": True,
                "batch_index": batch_index,
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
        )
        representative = sample_rows[0]
        reload_root = root / f"reload-{batch_index:03d}"
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
        reload_path = reload_evidence / "reload-audit.json"
        reload_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "case": representative["case"],
                    "data": representative["data"],
                    "expected_reynolds": representative["reynolds"],
                    "acceptance": reload_acceptance,
                }
            ),
            encoding="utf-8",
        )
        launch_report = reload_evidence / "launch_report.json"
        launch_report.write_text(
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
                        "reload_audit": str(reload_path),
                        "launch_report": str(launch_report),
                    },
                }
            ),
            encoding="utf-8",
        )
        (reload_root / "process_audit.json").write_text(
            json.dumps({"ownership": {"remaining_pids": []}}),
            encoding="utf-8",
        )
        reload_audits.append(
            {
                "stage": f"reload-audit-{batch_index:03d}",
                "batch_index": batch_index,
                "ok": True,
                "run_root": str(reload_root),
                "reload_audit": str(reload_path),
                "reload_audit_sha256": sha256(reload_path),
            }
        )
    complete = root / "pipeline-complete.json"
    complete.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "complete",
                "problem_id": spec.problem.problem_id,
                "solver_request": str(request),
                "solver_request_sha256": sha256(request),
                "mesh_sha256": spec.mesh_sha256,
                "batches": batches,
                "reload_audits": reload_audits,
            }
        ),
        encoding="utf-8",
    )
    return complete


def test_smoke_pipeline_finishes_as_development_complete(tmp_path: Path) -> None:
    config = Path("examples/cavity_2d_fluent/smoke.json")
    fluent = write_synthetic_fluent_pipeline(tmp_path, config)

    result = run_cavity_pipeline(
        config,
        fluent,
        tmp_path / "runs",
        request_text="训练二维方腔 POD-RBF Smoke",
    )

    assert result.status == "development_complete"
    assert (result.run_dir / "validation_metrics.json").is_file()
    assert (result.run_dir / "development_test_metrics.json").is_file()
    assert (result.run_dir / "model_card.md").is_file()
    for name in (
        "evaluation_details.json",
        "evaluation_arrays.npz",
        "field_comparison.png",
        "streamlines.png",
        "centerlines.png",
        "README.md",
    ):
        assert (result.run_dir / "report" / name).is_file()
    assert result.test_metrics is not None
    selection = json.loads(
        (result.run_dir / "model_selection.json").read_text(encoding="utf-8")
    )
    assert len(selection["candidates"]) == 18
