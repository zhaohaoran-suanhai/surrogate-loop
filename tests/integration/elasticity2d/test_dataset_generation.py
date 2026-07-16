from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from surrogate_loop.operator import external_solver
from surrogate_loop.operator.elasticity2d.config import load_elasticity_spec
from surrogate_loop.operator.elasticity2d.dataset import (
    DatasetFiles,
    generate_or_reuse_dataset,
    load_development_partitions,
    write_solver_job,
)
from surrogate_loop.operator.elasticity2d.sampling import build_sample_plan
from surrogate_loop.operator.field_data import sha256_file

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples/elasticity_2d_cantilever"


def test_solver_job_preserves_canonical_sample_identity(tmp_path: Path) -> None:
    spec = load_elasticity_spec(EXAMPLES / "smoke.json")
    plan = build_sample_plan(spec)

    job_path = write_solver_job(spec, plan, tmp_path)
    payload = json.loads(job_path.read_text(encoding="utf-8"))

    assert payload["protocol_version"] == "elasticity-job-v1"
    assert [item["sample_id"] for item in payload["samples"]] == plan.sample_ids.tolist()
    assert [item["role"] for item in payload["samples"]] == plan.roles.tolist()
    assert payload["solver"]["mesh_shape"] == [128, 32]
    assert payload["solver"]["observation_shape"] == [65, 17]


def test_solver_failure_prevents_dataset_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = load_elasticity_spec(EXAMPLES / "smoke.json")
    plan = build_sample_plan(spec)
    monkeypatch.setattr(
        external_solver,
        "run_solver_process",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 2, "", "force balance failed"
        ),
    )

    with pytest.raises(RuntimeError, match="force balance failed"):
        generate_or_reuse_dataset(spec, plan, tmp_path, ROOT)


def test_training_partitions_do_not_expose_sealed_file(tmp_path: Path) -> None:
    spec = load_elasticity_spec(EXAMPLES / "full.json")
    plan = build_sample_plan(spec)
    development_indices = np.flatnonzero(
        np.isin(plan.roles, np.array(["train", "validation"]))
    )
    sealed_indices = np.flatnonzero(plan.roles == "sealed_test")
    coordinates = np.array([[0.0, 0.0], [4.0, 1.0]], dtype=np.float64)
    development_path = tmp_path / "development.npz"
    sealed_path = tmp_path / "sealed_test.npz"
    np.savez_compressed(
        development_path,
        sample_ids=plan.sample_ids[development_indices],
        roles=plan.roles[development_indices],
        parameters=plan.parameters[development_indices],
        coordinates=coordinates,
        fields=np.zeros((development_indices.size, 2, 2), dtype=np.float64),
    )
    np.savez_compressed(
        sealed_path,
        sample_ids=plan.sample_ids[sealed_indices],
        roles=plan.roles[sealed_indices],
        parameters=plan.parameters[sealed_indices],
        coordinates=coordinates,
        fields=np.zeros((sealed_indices.size, 2, 2), dtype=np.float64),
    )
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    files = DatasetFiles(
        development_path=development_path,
        sealed_test_path=sealed_path,
        manifest_path=manifest_path,
        development_sha256=sha256_file(development_path),
        sealed_test_sha256=sha256_file(sealed_path),
    )

    partitions = load_development_partitions(files, plan)

    assert partitions.train.sample_ids.size == 512
    assert partitions.validation.sample_ids.size == 96
    assert not hasattr(partitions, "test")
    assert not hasattr(partitions, "sealed_test")


def test_validated_dataset_is_reused_only_with_matching_solver_versions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = load_elasticity_spec(EXAMPLES / "smoke.json")
    plan = build_sample_plan(spec)
    calls = 0

    def fake_run(action, arguments, repo_root, timeout_seconds):
        nonlocal calls
        calls += 1
        output_dir = Path(arguments[3])
        manifest = _write_fake_solver_output(output_dir, spec, plan)
        response = json.dumps({"status": "ok", "manifest": str(manifest.resolve())})
        return subprocess.CompletedProcess([], 0, f"conda wrapper\n{response}\n", "")

    monkeypatch.setattr(external_solver, "run_solver_process", fake_run)
    monkeypatch.setattr(
        external_solver,
        "doctor_solver_environment",
        lambda repo_root: {
            "python": "3.12.13",
            "dolfinx": "0.11.0",
            "pyamg": "5.3.0",
            "scipy": "1.18.0",
        },
    )

    first = generate_or_reuse_dataset(spec, plan, tmp_path, ROOT)
    second = generate_or_reuse_dataset(spec, plan, tmp_path, ROOT)

    assert calls == 1
    assert second == first
    assert load_development_partitions(second, plan).train.sample_ids.size == 96

    second.sealed_test_path.write_bytes(
        second.sealed_test_path.read_bytes() + b"tampered"
    )
    repaired = generate_or_reuse_dataset(spec, plan, tmp_path, ROOT)
    assert calls == 2
    assert sha256_file(repaired.sealed_test_path) == repaired.sealed_test_sha256


def _write_fake_solver_output(output_dir, spec, plan) -> Path:
    datasets = output_dir / "datasets"
    diagnostics = output_dir / "diagnostics"
    datasets.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)
    x, y = np.meshgrid(
        np.linspace(0.0, 4.0, spec.observation.nx),
        np.linspace(0.0, 1.0, spec.observation.ny),
        indexing="xy",
    )
    coordinates = np.column_stack((x.ravel(), y.ravel()))
    development = np.flatnonzero(np.isin(plan.roles, ["train", "validation"]))
    sealed = np.flatnonzero(plan.roles == "development_test")
    paths = {
        "development": datasets / "development.npz",
        "sealed_test": datasets / "sealed_test.npz",
    }
    for name, indices in (("development", development), ("sealed_test", sealed)):
        np.savez_compressed(
            paths[name],
            sample_ids=plan.sample_ids[indices],
            roles=plan.roles[indices],
            parameters=plan.parameters[indices],
            coordinates=coordinates,
            fields=np.zeros((indices.size, coordinates.shape[0], 2), dtype=np.float64),
        )
    quality_path = diagnostics / "solver_quality.json"
    quality_path.write_text("{}", encoding="utf-8")
    software = {
        "python": "3.12.13",
        "fenicsx": "0.11.0",
        "dolfinx": "0.11.0",
        "ufl": "2026.1.0",
        "numpy": "2.5.1",
        "scipy": "1.18.0",
        "pyamg": "5.3.0",
        "mpi4py": "4.1.2",
        "mpi": "Intel MPI",
        "platform": "Windows",
        "petsc4py_available": False,
    }
    records = [
        {
            "sample_id": str(sample_id),
            "role": str(role),
            "parameters": parameters.tolist(),
            "diagnostics": {
                "relative_residual": 1e-12,
                "force_balance_error": 1e-12,
                "clamp_error": 0.0,
                "solve_seconds": 0.01,
                "iterations": 1,
                "observed_peak_rss_mb": 100.0,
            },
            "stress_summary": {"von_mises_max": 1.0},
        }
        for sample_id, role, parameters in zip(
            plan.sample_ids, plan.roles, plan.parameters, strict=True
        )
    ]
    manifest = {
        "protocol_version": "elasticity-field-v1",
        "status": "complete",
        "problem_id": "elasticity_2d_cantilever_v1",
        "software": software,
        "solver": {
            "mesh_shape": [spec.mesh.nx, spec.mesh.ny],
            "observation_shape": [spec.observation.nx, spec.observation.ny],
            "backend": "pyamg",
            "tolerance": 1e-10,
            "element": "Lagrange-P2-triangle",
            "timing_scope": "assembly_solve_interpolation",
        },
        "coordinates": coordinates.tolist(),
        "samples": records,
        "files": {
            "development": {
                "path": "datasets/development.npz",
                "sha256": sha256_file(paths["development"]),
                "samples": int(development.size),
                "arrays": {},
            },
            "sealed_test": {
                "path": "datasets/sealed_test.npz",
                "sha256": sha256_file(paths["sealed_test"]),
                "samples": int(sealed.size),
                "arrays": {},
            },
            "solver_quality": {
                "path": "diagnostics/solver_quality.json",
                "sha256": sha256_file(quality_path),
            },
        },
    }
    manifest_path = datasets / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path
