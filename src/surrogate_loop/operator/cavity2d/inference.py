from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from surrogate_loop.operator.cavity2d.artifacts import (
    CavityRunState,
    read_cavity_state,
    verify_artifact_manifest,
)
from surrogate_loop.operator.cavity2d.config import load_cavity_spec
from surrogate_loop.operator.cavity2d.model import load_cavity_model
from surrogate_loop.operator.cavity2d.protocol import (
    import_verified_cavity_dataset,
)
from surrogate_loop.operator.cavity2d.sampling import build_cavity_sample_plan
from surrogate_loop.operator.field_data import load_field_dataset


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def predict_accepted_cavity(
    run_dir: Path,
    reynolds: float,
    output: Path,
) -> dict[str, object]:
    run_dir = run_dir.resolve()
    verify_artifact_manifest(run_dir)
    if read_cavity_state(run_dir) != CavityRunState.ACCEPTED:
        raise RuntimeError("cavity inference requires an accepted Full run")
    value = float(reynolds)
    if not np.isfinite(value) or not 10.0 <= value <= 400.0:
        raise ValueError("Reynolds must be finite and in [10,400]")
    output = output.resolve()
    if _inside(output, run_dir):
        raise ValueError("output cannot overwrite protected cavity run artifacts")
    dataset = load_field_dataset(run_dir / "data" / "development.npz")
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (run_dir / "data" / "solver_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    coordinates_sha256 = hashlib.sha256(
        np.ascontiguousarray(dataset.coordinates, dtype=np.float64).tobytes()
    ).hexdigest()
    problem = spec.get("problem")
    problem_id = problem.get("problem_id") if isinstance(problem, dict) else None
    mesh_sha256 = provenance.get("mesh_sha256")
    if (
        not isinstance(problem_id, str)
        or not isinstance(mesh_sha256, str)
        or provenance.get("problem_id") != problem_id
        or provenance.get("coordinates_sha256") != coordinates_sha256
    ):
        raise RuntimeError("cavity run problem, mesh, or coordinates identity mismatch")
    model = load_cavity_model(
        run_dir / "model",
        problem_id=problem_id,
        mesh_sha256=mesh_sha256,
        coordinates_sha256=coordinates_sha256,
    )
    prediction = model.predict(np.asarray([value], dtype=np.float64))[0]
    if (
        prediction.shape != (dataset.coordinates.shape[0], 3)
        or not np.isfinite(prediction).all()
    ):
        raise RuntimeError("cavity prediction is non-finite or has invalid shape")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        reynolds=np.asarray([value], dtype=np.float64),
        coordinates=dataset.coordinates,
        velocity=prediction[:, :2],
        pressure=prediction[:, 2],
    )
    return {
        "status": "predicted",
        "run_id": run_dir.name,
        "reynolds": value,
        "output": str(output),
        "shape": list(prediction.shape),
    }


def verify_solver_pipeline(
    config_path: Path,
    fluent_pipeline: Path,
    output_dir: Path,
) -> dict[str, object]:
    spec = load_cavity_spec(config_path)
    if spec.mode not in {"vertical", "calibration"}:
        raise ValueError("verify-solver accepts only vertical or calibration")
    plan = build_cavity_sample_plan(spec)
    files = import_verified_cavity_dataset(
        fluent_pipeline,
        output_dir,
        expected_sample_ids=plan.sample_ids,
        expected_reynolds=plan.reynolds,
        expected_splits=plan.split,
        expected_mesh_sha256=spec.mesh_sha256,
    )
    report = {
        "status": "protocol_verified",
        "mode": spec.mode,
        "development": str(files.development),
        "coordinates_sha256": files.coordinates_sha256,
    }
    (output_dir / "protocol-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def read_cavity_report(run_dir: Path) -> dict[str, object]:
    verify_artifact_manifest(run_dir)
    payload: dict[str, object] = {
        "status": read_cavity_state(run_dir).value,
        "validation_metrics": json.loads(
            (run_dir / "validation_metrics.json").read_text(encoding="utf-8")
        ),
    }
    for name in ("development_test_metrics", "test_metrics"):
        path = run_dir / f"{name}.json"
        if path.is_file():
            payload[name] = json.loads(path.read_text(encoding="utf-8"))
    return payload


__all__ = [
    "predict_accepted_cavity",
    "read_cavity_report",
    "verify_solver_pipeline",
]
