from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from surrogate_loop.operator import external_solver
from surrogate_loop.operator.elasticity2d.config import ElasticityRunSpec
from surrogate_loop.operator.elasticity2d.sampling import SamplePlan
from surrogate_loop.operator.field_data import FieldDataset, sha256_file

JOB_PROTOCOL = "elasticity-job-v1"
FIELD_PROTOCOL = "elasticity-field-v1"
_TEST_ROLES = frozenset({"development_test", "sealed_test"})


@dataclass(frozen=True)
class DatasetFiles:
    development_path: Path
    sealed_test_path: Path
    manifest_path: Path
    development_sha256: str
    sealed_test_sha256: str


@dataclass(frozen=True)
class DevelopmentPartitions:
    train: FieldDataset
    validation: FieldDataset


def write_solver_job(
    spec: ElasticityRunSpec,
    sample_plan: SamplePlan,
    run_dir: Path,
) -> Path:
    _validate_plan_against_spec(spec, sample_plan)
    payload = {
        "protocol_version": JOB_PROTOCOL,
        "problem_id": spec.problem.template,
        "solver": {
            "mesh_shape": [spec.mesh.nx, spec.mesh.ny],
            "observation_shape": [spec.observation.nx, spec.observation.ny],
            "backend": spec.solver.backend,
            "tolerance": min(1e-10, spec.solver.max_relative_residual / 100.0),
        },
        "quality": {
            "residual_max": spec.solver.max_relative_residual,
            "force_balance_max": spec.solver.max_force_balance_error,
            "clamp_max": spec.acceptance.max_clamp_absolute_error,
            "mesh_relative_l2_max": spec.solver.max_mesh_convergence_error,
            "linearity_relative_max": spec.solver.max_load_linearity_error,
        },
        "samples": [
            {
                "sample_id": str(sample_id),
                "role": str(role),
                "parameters": parameters.tolist(),
            }
            for sample_id, role, parameters in zip(
                sample_plan.sample_ids,
                sample_plan.roles,
                sample_plan.parameters,
                strict=True,
            )
        ],
    }
    path = run_dir.resolve() / "solver_job.json"
    _atomic_json(path, payload)
    return path


def generate_or_reuse_dataset(
    spec: ElasticityRunSpec,
    sample_plan: SamplePlan,
    run_dir: Path,
    repo_root: Path,
) -> DatasetFiles:
    if spec.mode == "calibration":
        raise ValueError("校准模式不生成训练数据集")
    job_path = write_solver_job(spec, sample_plan, run_dir)
    output_dir = run_dir.resolve() / "solver_output"
    cache_path = run_dir.resolve() / "solver_dataset_request.json"
    reused = _try_reuse(
        spec, sample_plan, job_path, output_dir, cache_path, repo_root
    )
    if reused is not None:
        return reused

    completed = external_solver.run_solver_process(
        "generate",
        ("--job", str(job_path), "--output-dir", str(output_dir)),
        repo_root,
        timeout_seconds=max(600.0, float(sample_plan.sample_ids.size) * 120.0),
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "未知错误"
        raise RuntimeError(f"FEniCSx 数据生成失败：{detail}")
    response = external_solver.parse_solver_json(completed.stdout, "generate")
    if response.get("status") != "ok" or not isinstance(response.get("manifest"), str):
        raise RuntimeError("FEniCSx generate 返回状态或清单路径无效")
    manifest_path = Path(response["manifest"]).resolve()
    expected_manifest = output_dir / "datasets" / "dataset_manifest.json"
    if manifest_path != expected_manifest.resolve():
        raise RuntimeError("FEniCSx generate 返回了作业目录外的清单")
    try:
        files, software = _validate_manifest(spec, sample_plan, manifest_path)
    except AssertionError as error:
        raise RuntimeError("FEniCSx 数据清单数值身份无效") from error
    _atomic_json(
        cache_path,
        {
            "job_sha256": sha256_file(job_path),
            "manifest_sha256": sha256_file(manifest_path),
            "software": software,
        },
    )
    return files


def load_development_partitions(
    files: DatasetFiles,
    sample_plan: SamplePlan,
) -> DevelopmentPartitions:
    arrays = _load_npz(files.development_path, files.development_sha256)
    expected_keys = {"sample_ids", "roles", "parameters", "coordinates", "fields"}
    if set(arrays) != expected_keys:
        raise RuntimeError("开发数据集数组字段无效")
    sample_ids = np.asarray(arrays["sample_ids"], dtype=np.str_)
    roles = np.asarray(arrays["roles"], dtype=np.str_)
    parameters = np.asarray(arrays["parameters"], dtype=np.float64)
    expected_indices = np.flatnonzero(np.isin(sample_plan.roles, ["train", "validation"]))
    np.testing.assert_array_equal(sample_ids, sample_plan.sample_ids[expected_indices])
    np.testing.assert_array_equal(roles, sample_plan.roles[expected_indices])
    np.testing.assert_allclose(
        parameters, sample_plan.parameters[expected_indices], rtol=0.0, atol=0.0
    )
    dataset = FieldDataset(
        sample_ids=sample_ids,
        parameters=parameters,
        coordinates=np.asarray(arrays["coordinates"], dtype=np.float64),
        fields=np.asarray(arrays["fields"], dtype=np.float64),
        diagnostics={},
    )
    train = np.flatnonzero(roles == "train").astype(np.int64)
    validation = np.flatnonzero(roles == "validation").astype(np.int64)
    if train.size == 0 or validation.size == 0 or np.any(np.isin(roles, list(_TEST_ROLES))):
        raise RuntimeError("开发数据集划分无效或混入测试样本")
    return DevelopmentPartitions(
        train=dataset.subset(train),
        validation=dataset.subset(validation),
    )


def _try_reuse(
    spec: ElasticityRunSpec,
    sample_plan: SamplePlan,
    job_path: Path,
    output_dir: Path,
    cache_path: Path,
    repo_root: Path,
) -> DatasetFiles | None:
    manifest_path = output_dir / "datasets" / "dataset_manifest.json"
    if not cache_path.is_file() or not manifest_path.is_file():
        return None
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache.get("job_sha256") != sha256_file(job_path):
            return None
        if cache.get("manifest_sha256") != sha256_file(manifest_path):
            return None
        files, software = _validate_manifest(spec, sample_plan, manifest_path)
        doctor = external_solver.doctor_solver_environment(repo_root)
        version_keys = ("python", "dolfinx", "pyamg", "scipy")
        if any(doctor.get(key) != software.get(key) for key in version_keys):
            return None
        return files
    except (AssertionError, OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return None


def _validate_manifest(
    spec: ElasticityRunSpec,
    sample_plan: SamplePlan,
    manifest_path: Path,
) -> tuple[DatasetFiles, dict[str, Any]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("无法读取 FEniCSx 数据清单") from error
    expected_top = {
        "protocol_version",
        "status",
        "problem_id",
        "software",
        "solver",
        "coordinates",
        "samples",
        "files",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_top:
        raise RuntimeError("FEniCSx 数据清单字段无效")
    if (
        manifest["protocol_version"] != FIELD_PROTOCOL
        or manifest["status"] != "complete"
        or manifest["problem_id"] != spec.problem.template
    ):
        raise RuntimeError("FEniCSx 数据清单身份无效")
    software = manifest["software"]
    required_software = {
        "python",
        "fenicsx",
        "dolfinx",
        "ufl",
        "numpy",
        "scipy",
        "pyamg",
        "mpi4py",
        "mpi",
        "platform",
        "petsc4py_available",
    }
    if (
        not isinstance(software, dict)
        or set(software) != required_software
        or software["petsc4py_available"] is not False
        or not str(software["dolfinx"]).startswith("0.11.")
    ):
        raise RuntimeError("FEniCSx 数据清单软件版本无效")
    solver = manifest["solver"]
    if (
        not isinstance(solver, dict)
        or solver.get("mesh_shape") != [spec.mesh.nx, spec.mesh.ny]
        or solver.get("observation_shape") != [spec.observation.nx, spec.observation.ny]
        or solver.get("backend") != spec.solver.backend
        or solver.get("element") != "Lagrange-P2-triangle"
        or solver.get("timing_scope") != "assembly_solve_interpolation"
    ):
        raise RuntimeError("FEniCSx 数据清单求解器配置无效")
    coordinates = _expected_coordinates(spec)
    np.testing.assert_allclose(
        np.asarray(manifest["coordinates"], dtype=np.float64),
        coordinates,
        rtol=0.0,
        atol=0.0,
    )
    _validate_sample_records(spec, sample_plan, manifest["samples"])

    files_payload = manifest["files"]
    if not isinstance(files_payload, dict) or set(files_payload) != {
        "development",
        "sealed_test",
        "solver_quality",
    }:
        raise RuntimeError("FEniCSx 数据清单文件记录无效")
    root = manifest_path.parents[1]
    development_path = root / "datasets" / "development.npz"
    sealed_path = root / "datasets" / "sealed_test.npz"
    development_hash = _validated_file_hash(
        files_payload["development"], development_path, "datasets/development.npz"
    )
    sealed_hash = _validated_file_hash(
        files_payload["sealed_test"], sealed_path, "datasets/sealed_test.npz"
    )
    quality_path = root / "diagnostics" / "solver_quality.json"
    _validated_file_hash(
        files_payload["solver_quality"],
        quality_path,
        "diagnostics/solver_quality.json",
    )
    _validate_partition_file(
        development_path,
        development_hash,
        sample_plan,
        coordinates,
        roles={"train", "validation"},
    )
    _validate_partition_file(
        sealed_path,
        sealed_hash,
        sample_plan,
        coordinates,
        roles=set(_TEST_ROLES),
    )
    return (
        DatasetFiles(
            development_path=development_path,
            sealed_test_path=sealed_path,
            manifest_path=manifest_path,
            development_sha256=development_hash,
            sealed_test_sha256=sealed_hash,
        ),
        software,
    )


def _validate_sample_records(
    spec: ElasticityRunSpec,
    sample_plan: SamplePlan,
    records: Any,
) -> None:
    if not isinstance(records, list) or len(records) != sample_plan.sample_ids.size:
        raise RuntimeError("FEniCSx 数据清单样本数无效")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "sample_id",
            "role",
            "parameters",
            "diagnostics",
            "stress_summary",
        }:
            raise RuntimeError("FEniCSx 数据清单样本字段无效")
        if (
            record["sample_id"] != sample_plan.sample_ids[index]
            or record["role"] != sample_plan.roles[index]
        ):
            raise RuntimeError("FEniCSx 数据清单样本身份无效")
        np.testing.assert_allclose(
            np.asarray(record["parameters"], dtype=np.float64),
            sample_plan.parameters[index],
            rtol=0.0,
            atol=0.0,
        )
        diagnostics = record["diagnostics"]
        required_diagnostics = {
            "relative_residual",
            "force_balance_error",
            "clamp_error",
            "solve_seconds",
            "iterations",
            "observed_peak_rss_mb",
        }
        limits = {
            "relative_residual": spec.solver.max_relative_residual,
            "force_balance_error": spec.solver.max_force_balance_error,
            "clamp_error": spec.acceptance.max_clamp_absolute_error,
        }
        if (
            not isinstance(diagnostics, dict)
            or set(diagnostics) != required_diagnostics
            or any(
                not math.isfinite(float(diagnostics.get(name, math.inf)))
                or float(diagnostics[name]) > limit
                for name, limit in limits.items()
            )
            or float(diagnostics["solve_seconds"]) <= 0.0
            or int(diagnostics["iterations"]) <= 0
            or float(diagnostics["observed_peak_rss_mb"]) <= 0.0
        ):
            raise RuntimeError("FEniCSx 数据清单样本诊断未通过门禁")
        stress = record["stress_summary"]
        if (
            not isinstance(stress, dict)
            or "von_mises_max" not in stress
            or not stress
            or any(not math.isfinite(float(value)) for value in stress.values())
        ):
            raise RuntimeError("FEniCSx 数据清单应力诊断无效")


def _validate_partition_file(
    path: Path,
    digest: str,
    sample_plan: SamplePlan,
    coordinates: np.ndarray,
    roles: set[str],
) -> None:
    arrays = _load_npz(path, digest)
    if set(arrays) != {"sample_ids", "roles", "parameters", "coordinates", "fields"}:
        raise RuntimeError("FEniCSx NPZ 数组字段无效")
    expected = np.flatnonzero(np.isin(sample_plan.roles, list(roles)))
    np.testing.assert_array_equal(arrays["sample_ids"], sample_plan.sample_ids[expected])
    np.testing.assert_array_equal(arrays["roles"], sample_plan.roles[expected])
    np.testing.assert_allclose(
        arrays["parameters"], sample_plan.parameters[expected], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(arrays["coordinates"], coordinates, rtol=0.0, atol=0.0)
    fields = np.asarray(arrays["fields"])
    if (
        np.asarray(arrays["sample_ids"]).dtype.kind != "U"
        or np.asarray(arrays["roles"]).dtype.kind != "U"
        or np.asarray(arrays["parameters"]).dtype != np.dtype(np.float64)
        or np.asarray(arrays["coordinates"]).dtype != np.dtype(np.float64)
        or fields.dtype != np.dtype(np.float64)
        or fields.shape != (expected.size, coordinates.shape[0], 2)
        or not np.isfinite(fields).all()
    ):
        raise RuntimeError("FEniCSx NPZ 字段形状或数值无效")


def _validated_file_hash(payload: Any, path: Path, relative: str) -> str:
    if not isinstance(payload, dict) or payload.get("path", "").replace("\\", "/") != relative:
        raise RuntimeError("FEniCSx 数据文件路径记录无效")
    digest = payload.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or sha256_file(path) != digest:
        raise RuntimeError("FEniCSx 数据文件 SHA-256 校验失败")
    return digest


def _load_npz(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("FEniCSx NPZ SHA-256 校验失败")
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError) as error:
        raise RuntimeError("无法安全读取 FEniCSx NPZ") from error


def _validate_plan_against_spec(
    spec: ElasticityRunSpec,
    sample_plan: SamplePlan,
) -> None:
    expected = (
        spec.sampling.train_cases
        + spec.sampling.validation_cases
        + spec.sampling.test_cases
    )
    if sample_plan.sample_ids.size != expected:
        raise ValueError("样本计划数量与规格不一致")
    allowed = (
        {"calibration"}
        if spec.mode == "calibration"
        else {"train", "validation", "development_test", "sealed_test"}
    )
    if not set(sample_plan.roles.tolist()) <= allowed:
        raise ValueError("样本计划角色与运行模式不一致")


def _expected_coordinates(spec: ElasticityRunSpec) -> np.ndarray:
    x, y = np.meshgrid(
        np.linspace(0.0, spec.problem.length, spec.observation.nx),
        np.linspace(0.0, spec.problem.height, spec.observation.ny),
        indexing="xy",
    )
    return np.column_stack((x.ravel(), y.ravel()))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
