from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import dolfinx
import mpi4py
import numpy as np
import psutil
import pyamg
import scipy
import scipy.sparse.linalg
import ufl
from dolfinx import fem, la, mesh
from mpi4py import MPI

from solvers.fenicsx.elasticity2d.interpolate import (
    interpolate_displacement,
    observation_coordinates,
)
from solvers.fenicsx.elasticity2d.problem import HEIGHT, LENGTH
from solvers.fenicsx.elasticity2d.solve import SolvedCase, solve_case

JOB_PROTOCOL = "elasticity-job-v1"
FIELD_PROTOCOL = "elasticity-field-v1"
_SAMPLE_ID = re.compile(r"^(train|validation|test)-[0-9]{6}$")


@dataclass(frozen=True)
class ManufacturedLevel:
    mesh_shape: tuple[int, int]
    l2_error: float
    h1_error: float


@dataclass(frozen=True)
class ManufacturedReport:
    levels: tuple[ManufacturedLevel, ...]
    l2_rates: tuple[float, ...]
    h1_rates: tuple[float, ...]
    minimum_l2_rate: float
    minimum_h1_rate: float


@dataclass(frozen=True)
class DatasetManifest:
    manifest_path: Path
    development_path: Path
    sealed_test_path: Path
    development_sha256: str
    sealed_test_sha256: str
    development_samples: int
    sealed_test_samples: int


@dataclass(frozen=True)
class _SolverSettings:
    mesh_shape: tuple[int, int]
    observation_shape: tuple[int, int]
    backend: Literal["pyamg", "scipy"]
    tolerance: float


@dataclass(frozen=True)
class _QualityThresholds:
    residual_max: float
    force_balance_max: float
    clamp_max: float
    mesh_relative_l2_max: float
    linearity_relative_max: float


@dataclass(frozen=True)
class _Sample:
    sample_id: str
    role: Literal["train", "validation", "test"]
    parameters: np.ndarray


@dataclass(frozen=True)
class _Job:
    problem_id: str
    solver: _SolverSettings
    quality: _QualityThresholds
    samples: tuple[_Sample, ...]


def run_manufactured_convergence(
    meshes: list[tuple[int, int]],
) -> ManufacturedReport:
    if len(meshes) < 3:
        raise ValueError("制造解收敛至少需要三层网格")
    levels = tuple(_solve_manufactured_level(shape) for shape in meshes)
    l2_rates = _convergence_rates([level.l2_error for level in levels], meshes)
    h1_rates = _convergence_rates([level.h1_error for level in levels], meshes)
    return ManufacturedReport(
        levels=levels,
        l2_rates=l2_rates,
        h1_rates=h1_rates,
        minimum_l2_rate=min(l2_rates),
        minimum_h1_rate=min(h1_rates),
    )


def generate_datasets(job_path: Path, output_dir: Path) -> DatasetManifest:
    job = _load_job(job_path)
    coordinates = observation_coordinates(*job.solver.observation_shape)
    records: list[dict[str, Any]] = []
    fields: list[np.ndarray] = []
    for sample in job.samples:
        case = solve_case(
            sample.parameters,
            job.solver.mesh_shape,
            job.solver.backend,
            job.solver.tolerance,
        )
        values = interpolate_displacement(case.solution, coordinates)
        diagnostics = _case_diagnostics(case, values, coordinates)
        _enforce_case_quality(sample.sample_id, diagnostics, job.quality)
        fields.append(values)
        records.append(
            {
                "sample_id": sample.sample_id,
                "role": sample.role,
                "parameters": sample.parameters.tolist(),
                "diagnostics": diagnostics,
                "stress_summary": case.stress_summary,
            }
        )

    field_array = np.stack(fields).astype(np.float64, copy=False)
    development_indices = np.array(
        [index for index, sample in enumerate(job.samples) if sample.role != "test"],
        dtype=np.int64,
    )
    test_indices = np.array(
        [index for index, sample in enumerate(job.samples) if sample.role == "test"],
        dtype=np.int64,
    )
    if development_indices.size == 0 or test_indices.size == 0:
        raise ValueError("作业必须同时包含开发样本和封存测试样本")

    datasets_dir = output_dir.resolve() / "datasets"
    diagnostics_dir = output_dir.resolve() / "diagnostics"
    development_path = datasets_dir / "development.npz"
    sealed_path = datasets_dir / "sealed_test.npz"
    _write_partition(
        development_path, job, development_indices, coordinates, field_array
    )
    _write_partition(sealed_path, job, test_indices, coordinates, field_array)
    development_hash = _sha256(development_path)
    sealed_hash = _sha256(sealed_path)

    quality_path = diagnostics_dir / "solver_quality.json"
    _atomic_json(
        quality_path,
        {
            "protocol_version": FIELD_PROTOCOL,
            "status": "passed",
            "thresholds": asdict(job.quality),
            "samples": records,
        },
    )
    manifest_path = datasets_dir / "dataset_manifest.json"
    manifest_payload = {
        "protocol_version": FIELD_PROTOCOL,
        "status": "complete",
        "problem_id": job.problem_id,
        "software": software_versions(),
        "solver": {
            **asdict(job.solver),
            "mesh_shape": list(job.solver.mesh_shape),
            "observation_shape": list(job.solver.observation_shape),
            "element": "Lagrange-P1-triangle",
        },
        "coordinates": coordinates.tolist(),
        "samples": records,
        "files": {
            "development": _file_manifest(
                development_path, development_hash, development_indices.size
            ),
            "sealed_test": _file_manifest(sealed_path, sealed_hash, test_indices.size),
            "solver_quality": {
                "path": str(quality_path.relative_to(output_dir.resolve())),
                "sha256": _sha256(quality_path),
            },
        },
    }
    _atomic_json(manifest_path, manifest_payload)
    return DatasetManifest(
        manifest_path=manifest_path,
        development_path=development_path,
        sealed_test_path=sealed_path,
        development_sha256=development_hash,
        sealed_test_sha256=sealed_hash,
        development_samples=int(development_indices.size),
        sealed_test_samples=int(test_indices.size),
    )


def run_calibration(job_path: Path, output_dir: Path) -> Path:
    job = _load_job(job_path)
    manufactured = run_manufactured_convergence([(8, 2), (16, 4), (32, 8)])
    if manufactured.minimum_l2_rate < 2.5 or manufactured.minimum_h1_rate < 1.5:
        raise RuntimeError("P2 制造解收敛阶未通过质量门禁")
    coordinates = observation_coordinates(*job.solver.observation_shape)
    records: list[dict[str, Any]] = []
    refined_shape = tuple(2 * value for value in job.solver.mesh_shape)
    for sample in job.samples:
        coarse = solve_case(
            sample.parameters,
            job.solver.mesh_shape,
            job.solver.backend,
            job.solver.tolerance,
        )
        refined = solve_case(
            sample.parameters,
            refined_shape,
            job.solver.backend,
            job.solver.tolerance,
        )
        half_load = sample.parameters.copy()
        half_load[2] *= 0.5
        half = solve_case(
            half_load,
            job.solver.mesh_shape,
            job.solver.backend,
            job.solver.tolerance,
        )
        coarse_values = interpolate_displacement(coarse.solution, coordinates)
        refined_values = interpolate_displacement(refined.solution, coordinates)
        half_values = interpolate_displacement(half.solution, coordinates)
        diagnostics = _case_diagnostics(coarse, coarse_values, coordinates)
        diagnostics["mesh_relative_l2"] = _relative_l2(
            refined_values, coarse_values
        )
        diagnostics["linearity_relative_error"] = _relative_l2(
            coarse_values, 2.0 * half_values
        )
        _enforce_case_quality(sample.sample_id, diagnostics, job.quality, calibration=True)
        records.append({"sample_id": sample.sample_id, **diagnostics})
    output_path = output_dir.resolve() / "diagnostics" / "calibration.json"
    _atomic_json(
        output_path,
        {
            "protocol_version": FIELD_PROTOCOL,
            "status": "passed",
            "base_mesh_shape": list(job.solver.mesh_shape),
            "refined_mesh_shape": list(refined_shape),
            "manufactured": asdict(manufactured),
            "samples": records,
        },
    )
    return output_path


def software_versions() -> dict[str, str | bool]:
    return {
        "python": platform.python_version(),
        "fenicsx": dolfinx.__version__,
        "dolfinx": dolfinx.__version__,
        "ufl": ufl.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pyamg": pyamg.__version__,
        "mpi4py": mpi4py.__version__,
        "mpi": MPI.Get_library_version().strip(),
        "platform": platform.platform(),
        "petsc4py_available": bool(dolfinx.has_petsc4py),
    }


def _solve_manufactured_level(mesh_shape: tuple[int, int]) -> ManufacturedLevel:
    nx, ny = _positive_pair(mesh_shape, "制造解网格")
    domain = mesh.create_rectangle(
        MPI.COMM_SELF,
        [np.array([0.0, 0.0]), np.array([LENGTH, HEIGHT])],
        [nx, ny],
        cell_type=mesh.CellType.triangle,
    )
    space = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    coordinate = ufl.SpatialCoordinate(domain)
    exact = ufl.as_vector(
        (
            ufl.sin(math.pi * coordinate[0]) * ufl.sin(math.pi * coordinate[1]),
            ufl.cos(math.pi * coordinate[0]) * ufl.sin(math.pi * coordinate[1]),
        )
    )
    young_modulus, poisson_ratio = 2.0, 0.3
    shear = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lambda_ps = young_modulus * poisson_ratio / (1.0 - poisson_ratio**2)

    def epsilon(value: Any) -> Any:
        return ufl.sym(ufl.grad(value))

    def sigma(value: Any) -> Any:
        strain = epsilon(value)
        return 2.0 * shear * strain + lambda_ps * ufl.tr(strain) * ufl.Identity(2)

    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 8})
    body_force = -ufl.div(sigma(exact))
    bilinear = fem.form(ufl.inner(sigma(trial), epsilon(test)) * dx)
    linear = fem.form(ufl.dot(body_force, test) * dx)
    facet_dim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain, facet_dim, lambda x: np.full(x.shape[1], True)
    )
    dofs = fem.locate_dofs_topological(space, facet_dim, facets)
    boundary = fem.Function(space)
    boundary.interpolate(fem.Expression(exact, space.element.interpolation_points))
    bc = fem.dirichletbc(boundary, dofs)
    matrix = fem.assemble_matrix(bilinear, bcs=[bc]).to_scipy().tocsr()
    right_hand_side = fem.assemble_vector(linear)
    fem.apply_lifting(right_hand_side.array, [bilinear], bcs=[[bc]])
    right_hand_side.scatter_reverse(la.InsertMode.add)
    bc.set(right_hand_side.array)
    solution = fem.Function(space)
    solution.x.array[:] = scipy.sparse.linalg.spsolve(matrix, right_hand_side.array)
    solution.x.scatter_forward()
    difference = solution - exact
    l2_error = math.sqrt(
        float(fem.assemble_scalar(fem.form(ufl.inner(difference, difference) * dx)))
    )
    h1_error = math.sqrt(
        float(
            fem.assemble_scalar(
                fem.form(ufl.inner(ufl.grad(difference), ufl.grad(difference)) * dx)
            )
        )
    )
    return ManufacturedLevel((nx, ny), l2_error, h1_error)


def _convergence_rates(
    errors: list[float], meshes: list[tuple[int, int]]
) -> tuple[float, ...]:
    rates: list[float] = []
    for index in range(len(errors) - 1):
        refinement = meshes[index + 1][0] / meshes[index][0]
        if refinement <= 1.0 or meshes[index + 1][1] / meshes[index][1] != refinement:
            raise ValueError("制造解网格必须按相同比例均匀加密")
        rates.append(math.log(errors[index] / errors[index + 1]) / math.log(refinement))
    return tuple(rates)


def _load_job(path: Path) -> _Job:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取求解作业：{path}") from error
    _exact_keys(payload, {"protocol_version", "problem_id", "solver", "quality", "samples"}, "作业")
    if payload["protocol_version"] != JOB_PROTOCOL:
        raise ValueError("求解作业协议版本不受支持")
    if payload["problem_id"] != "elasticity_2d_cantilever_v1":
        raise ValueError("求解作业问题标识不受支持")
    solver_payload = payload["solver"]
    _exact_keys(
        solver_payload,
        {"mesh_shape", "observation_shape", "backend", "tolerance"},
        "求解器配置",
    )
    backend = solver_payload["backend"]
    if backend not in {"pyamg", "scipy"}:
        raise ValueError("求解后端只能是 pyamg 或 scipy")
    tolerance = _positive_float(solver_payload["tolerance"], "求解容差")
    quality_payload = payload["quality"]
    quality_keys = {
        "residual_max",
        "force_balance_max",
        "clamp_max",
        "mesh_relative_l2_max",
        "linearity_relative_max",
    }
    _exact_keys(quality_payload, quality_keys, "质量门槛")
    quality = _QualityThresholds(
        **{key: _positive_float(quality_payload[key], key) for key in quality_keys}
    )
    sample_payloads = payload["samples"]
    if not isinstance(sample_payloads, list) or not sample_payloads:
        raise ValueError("求解作业必须包含非空样本列表")
    samples: list[_Sample] = []
    identities: set[str] = set()
    for item in sample_payloads:
        _exact_keys(item, {"sample_id", "role", "parameters"}, "样本")
        sample_id = item["sample_id"]
        role = item["role"]
        if not isinstance(sample_id, str) or _SAMPLE_ID.fullmatch(sample_id) is None:
            raise ValueError("样本标识格式无效")
        if sample_id in identities:
            raise ValueError("样本标识重复")
        if role not in {"train", "validation", "test"} or not sample_id.startswith(f"{role}-"):
            raise ValueError("样本角色与标识不一致")
        parameters = np.asarray(item["parameters"], dtype=np.float64)
        if parameters.shape != (6,) or not np.isfinite(parameters).all():
            raise ValueError("样本参数必须包含六个有限值")
        identities.add(sample_id)
        samples.append(_Sample(sample_id, role, parameters))
    return _Job(
        problem_id=payload["problem_id"],
        solver=_SolverSettings(
            mesh_shape=_positive_pair(solver_payload["mesh_shape"], "有限元网格"),
            observation_shape=_positive_pair(
                solver_payload["observation_shape"], "观测网格", minimum=2
            ),
            backend=backend,
            tolerance=tolerance,
        ),
        quality=quality,
        samples=tuple(samples),
    )


def _case_diagnostics(
    case: SolvedCase, fields: np.ndarray, coordinates: np.ndarray
) -> dict[str, float | int]:
    clamp_mask = np.isclose(coordinates[:, 0], 0.0)
    return {
        "relative_residual": case.relative_residual,
        "force_balance_error": float(
            np.linalg.norm(case.reaction + case.applied_force)
            / max(np.linalg.norm(case.applied_force), 1e-30)
        ),
        "clamp_error": float(np.max(np.abs(fields[clamp_mask]))),
        "solve_seconds": case.solve_seconds,
        "iterations": case.iterations,
        "observed_peak_rss_mb": max(
            case.observed_peak_rss_mb,
            psutil.Process().memory_info().rss / (1024.0 * 1024.0),
        ),
    }


def _enforce_case_quality(
    sample_id: str,
    diagnostics: dict[str, float | int],
    thresholds: _QualityThresholds,
    *,
    calibration: bool = False,
) -> None:
    checks = {
        "relative_residual": thresholds.residual_max,
        "force_balance_error": thresholds.force_balance_max,
        "clamp_error": thresholds.clamp_max,
    }
    if calibration:
        checks.update(
            {
                "mesh_relative_l2": thresholds.mesh_relative_l2_max,
                "linearity_relative_error": thresholds.linearity_relative_max,
            }
        )
    failed = [name for name, limit in checks.items() if float(diagnostics[name]) > limit]
    if failed:
        raise RuntimeError(f"样本 {sample_id} 未通过质量门禁：{', '.join(failed)}")


def _write_partition(
    path: Path,
    job: _Job,
    indices: np.ndarray,
    coordinates: np.ndarray,
    fields: np.ndarray,
) -> None:
    samples = [job.samples[int(index)] for index in indices]
    _atomic_npz(
        path,
        {
            "sample_ids": np.asarray([sample.sample_id for sample in samples]),
            "roles": np.asarray([sample.role for sample in samples]),
            "parameters": np.stack([sample.parameters for sample in samples]),
            "coordinates": coordinates.astype(np.float64, copy=False),
            "fields": fields[indices],
        },
    )


def _file_manifest(path: Path, digest: str, samples: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as arrays:
        return {
            "path": str(Path("datasets") / path.name),
            "sha256": digest,
            "samples": int(samples),
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()
            },
        }


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_l2(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.linalg.norm(reference - prediction)
        / max(np.linalg.norm(reference), 1e-30)
    )


def _positive_pair(value: Any, name: str, minimum: int = 1) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or any(item < minimum for item in value)
    ):
        raise ValueError(f"{name}必须是两个不小于 {minimum} 的整数")
    return int(value[0]), int(value[1])


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}必须是有限正数")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name}必须是有限正数")
    return result


def _exact_keys(payload: Any, expected: set[str], name: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{name}字段不完整或包含未知字段")
