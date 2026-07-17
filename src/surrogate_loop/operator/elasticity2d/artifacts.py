from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import joblib
import numpy as np
import torch

from surrogate_loop.operator import external_solver
from surrogate_loop.operator.elasticity2d.config import ElasticityRunSpec
from surrogate_loop.operator.elasticity2d.dataset import DatasetFiles
from surrogate_loop.operator.elasticity2d.deeponet import build_elasticity_deeponet
from surrogate_loop.operator.elasticity2d.evaluation import (
    compute_elasticity_metrics,
    elasticity_is_acceptable,
)
from surrogate_loop.operator.elasticity2d.pod_rbf import PodRbfBaseline
from surrogate_loop.operator.elasticity2d.sampling import SamplePlan
from surrogate_loop.operator.elasticity2d.training import (
    SelectedTraining,
    predict_dataset,
)
from surrogate_loop.operator.field_data import (
    FieldDataset,
    FieldNormalization,
    sha256_file,
)


class ElasticityRunState(StrEnum):
    CREATED = "created"
    SOLVER_ACCEPTED = "solver_accepted"
    TRAINED = "trained"
    FROZEN = "frozen"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


_ALLOWED_TRANSITIONS = frozenset(
    {
        (ElasticityRunState.CREATED, ElasticityRunState.SOLVER_ACCEPTED),
        (ElasticityRunState.SOLVER_ACCEPTED, ElasticityRunState.TRAINED),
        (ElasticityRunState.TRAINED, ElasticityRunState.FROZEN),
        (ElasticityRunState.FROZEN, ElasticityRunState.ACCEPTED),
        (ElasticityRunState.FROZEN, ElasticityRunState.REJECTED),
        (ElasticityRunState.CREATED, ElasticityRunState.FAILED),
        (ElasticityRunState.SOLVER_ACCEPTED, ElasticityRunState.FAILED),
        (ElasticityRunState.TRAINED, ElasticityRunState.FAILED),
        (ElasticityRunState.FROZEN, ElasticityRunState.FAILED),
    }
)

_FREEZE_FILES = (
    "spec.json",
    "sample_plan.json",
    "dataset_identity.json",
    "normalization.json",
    "pod_rbf.joblib",
    "network.json",
    "training_candidates.json",
    "deeponet_state.pt",
)


@dataclass(frozen=True)
class FreezeManifest:
    version: int
    problem: str
    mode: str
    selected_seed: int
    development_sha256: str
    sealed_test_sha256: str
    files: dict[str, str]


@dataclass(frozen=True)
class AcceptanceResult:
    status: str
    deeponet_metrics: dict[str, float]
    pod_rbf_metrics: dict[str, float]
    neural_median_seconds: float
    fenicsx_median_seconds: float
    speedup: float


def read_run_state(run_dir: Path) -> ElasticityRunState:
    path = run_dir.resolve() / "status.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("无法读取二维弹性运行状态") from error
    if not isinstance(payload, dict) or set(payload) != {"state"}:
        raise RuntimeError("二维弹性运行状态文件格式无效")
    try:
        return ElasticityRunState(payload["state"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("二维弹性运行状态值无效") from error


def transition_run(
    run_dir: Path,
    expected: ElasticityRunState | None,
    target: ElasticityRunState,
) -> None:
    directory = run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with _state_lock(directory):
        _transition_unlocked(directory, expected, target)


def freeze_run(
    run_dir: Path,
    spec: ElasticityRunSpec,
    sample_plan: SamplePlan,
    dataset_files: DatasetFiles,
    normalization: FieldNormalization,
    baseline: PodRbfBaseline,
    selected_training: SelectedTraining,
) -> FreezeManifest:
    directory = run_dir.resolve()
    _validate_freeze_inputs(spec, sample_plan, dataset_files, selected_training)
    with _state_lock(directory):
        if read_run_state(directory) is not ElasticityRunState.TRAINED:
            raise RuntimeError("二维弹性运行只有 trained 状态可以冻结")
        _write_json_atomic(directory / "spec.json", spec.model_dump(mode="json"))
        _write_json_atomic(
            directory / "sample_plan.json",
            {
                "sample_ids": sample_plan.sample_ids.tolist(),
                "roles": sample_plan.roles.tolist(),
                "parameters": sample_plan.parameters.tolist(),
            },
        )
        _write_json_atomic(
            directory / "dataset_identity.json",
            {
                "development_path": str(dataset_files.development_path.resolve()),
                "development_sha256": dataset_files.development_sha256,
                "sealed_test_path": str(dataset_files.sealed_test_path.resolve()),
                "sealed_test_sha256": dataset_files.sealed_test_sha256,
                "solver_manifest_path": str(dataset_files.manifest_path.resolve()),
                "solver_manifest_sha256": sha256_file(dataset_files.manifest_path),
            },
        )
        _write_json_atomic(
            directory / "normalization.json",
            {
                "feature_mean": normalization.feature_mean.tolist(),
                "feature_std": normalization.feature_std.tolist(),
                "coordinate_mean": normalization.coordinate_mean.tolist(),
                "coordinate_std": normalization.coordinate_std.tolist(),
                "target_rms": normalization.target_rms.tolist(),
            },
        )
        _joblib_dump_atomic(directory / "pod_rbf.joblib", baseline)
        _write_json_atomic(
            directory / "network.json",
            {
                "branch_input_dim": 5,
                "trunk_input_dim": 2,
                "output_dim": 2,
                "hidden_width": spec.model.hidden_width,
                "hidden_layers": spec.model.hidden_layers,
                "latent_dim": spec.model.latent_dim,
            },
        )
        _write_json_atomic(
            directory / "training_candidates.json",
            {
                "selected_seed": selected_training.selected_seed,
                "candidates": [
                    {
                        "seed": candidate.seed,
                        "history": [asdict(record) for record in candidate.history],
                        "best_epoch": candidate.best_epoch,
                        "validation_loss": candidate.validation_loss,
                        "stop_reason": candidate.stop_reason,
                        "device": candidate.device,
                        "elapsed_seconds": candidate.elapsed_seconds,
                        "peak_cuda_memory_mb": candidate.peak_cuda_memory_mb,
                    }
                    for candidate in selected_training.candidates
                ],
            },
        )
        _torch_save_atomic(
            directory / "deeponet_state.pt", selected_training.selected.state_dict
        )
        manifest = FreezeManifest(
            version=1,
            problem=spec.problem.template,
            mode=spec.mode,
            selected_seed=selected_training.selected_seed,
            development_sha256=dataset_files.development_sha256,
            sealed_test_sha256=dataset_files.sealed_test_sha256,
            files={name: sha256_file(directory / name) for name in _FREEZE_FILES},
        )
        _write_json_atomic(directory / "freeze_manifest.json", asdict(manifest))
        _transition_unlocked(
            directory, ElasticityRunState.TRAINED, ElasticityRunState.FROZEN
        )
    return manifest


def verify_freeze_manifest(run_dir: Path) -> FreezeManifest:
    directory = run_dir.resolve()
    try:
        payload = json.loads(
            (directory / "freeze_manifest.json").read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict) or set(payload) != {
            "version",
            "problem",
            "mode",
            "selected_seed",
            "development_sha256",
            "sealed_test_sha256",
            "files",
        }:
            raise ValueError("字段集合无效")
        manifest = FreezeManifest(**payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise RuntimeError("二维弹性冻结清单格式无效") from error
    if manifest.version != 1 or set(manifest.files) != set(_FREEZE_FILES):
        raise RuntimeError("二维弹性冻结清单版本或文件集合无效")
    for name, expected in manifest.files.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError("二维弹性冻结清单 SHA-256 记录无效")
        if sha256_file(directory / name) != expected:
            raise RuntimeError(f"二维弹性冻结文件 SHA-256 校验失败：{name}")
    return manifest


def evaluate_sealed_once(
    run_dir: Path,
    dataset_files: DatasetFiles,
) -> AcceptanceResult:
    directory = run_dir.resolve()
    with _state_lock(directory):
        state = read_run_state(directory)
        if state in {ElasticityRunState.ACCEPTED, ElasticityRunState.REJECTED}:
            raise RuntimeError("二维弹性封存测试已经消费")
        if state is not ElasticityRunState.FROZEN:
            raise RuntimeError("二维弹性封存测试只能在 frozen 状态消费")
        try:
            manifest = verify_freeze_manifest(directory)
            spec = ElasticityRunSpec.model_validate(
                _read_json_object(directory / "spec.json", "二维弹性规格")
            )
            dataset = _load_sealed_dataset(directory, dataset_files, manifest)
            normalization = _load_normalization(directory / "normalization.json")
            model = build_elasticity_deeponet(spec.model).cpu()
            state_dict = torch.load(
                directory / "deeponet_state.pt",
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            deeponet_prediction = predict_dataset(
                model,
                dataset,
                normalization,
                torch.device("cpu"),
                spec.training.query_batch_size,
            )
            deeponet_metrics = compute_elasticity_metrics(
                dataset.fields,
                deeponet_prediction,
                dataset.parameters,
                dataset.coordinates,
            )
            baseline = joblib.load(directory / "pod_rbf.joblib")
            if not isinstance(baseline, PodRbfBaseline):
                raise RuntimeError("冻结 POD-RBF 产物类型无效")
            pod_prediction = baseline.predict(dataset.parameters)
            pod_metrics = compute_elasticity_metrics(
                dataset.fields,
                pod_prediction,
                dataset.parameters,
                dataset.coordinates,
            )
            neural_seconds = _benchmark_neural(
                model,
                dataset,
                normalization,
                spec.training.query_batch_size,
            )
            fenicsx_seconds = _benchmark_fenicsx(directory, spec, dataset)
            speedup = fenicsx_seconds / neural_seconds
            status = (
                ElasticityRunState.ACCEPTED
                if elasticity_is_acceptable(
                    deeponet_metrics, spec.acceptance, speedup
                )
                else ElasticityRunState.REJECTED
            )
            result = AcceptanceResult(
                status=status.value,
                deeponet_metrics=deeponet_metrics.to_dict(),
                pod_rbf_metrics=pod_metrics.to_dict(),
                neural_median_seconds=neural_seconds,
                fenicsx_median_seconds=fenicsx_seconds,
                speedup=speedup,
            )
            _write_json_atomic(directory / "acceptance.json", asdict(result))
            _write_json_atomic(
                directory / "sealed_test_summary.json",
                {
                    "samples": int(dataset.parameters.shape[0]),
                    "sample_ids": dataset.sample_ids.tolist(),
                    "sha256": dataset_files.sealed_test_sha256,
                },
            )
            _write_json_atomic(
                directory / "acceptance_stage.json",
                {
                    "status": "complete",
                    "acceptance_sha256": sha256_file(directory / "acceptance.json"),
                    "sealed_summary_sha256": sha256_file(
                        directory / "sealed_test_summary.json"
                    ),
                    "freeze_manifest_sha256": sha256_file(
                        directory / "freeze_manifest.json"
                    ),
                    "fenicsx_benchmark_manifest_sha256": _optional_sha256(
                        directory
                        / "fenicsx_benchmark"
                        / "datasets"
                        / "dataset_manifest.json"
                    ),
                },
            )
            _transition_unlocked(directory, ElasticityRunState.FROZEN, status)
            return result
        except Exception as error:
            _write_json_atomic(
                directory / "acceptance_error.json",
                {"type": type(error).__name__, "message": str(error)},
            )
            _transition_unlocked(
                directory,
                ElasticityRunState.FROZEN,
                ElasticityRunState.FAILED,
            )
            raise


@contextmanager
def _state_lock(run_dir: Path) -> Iterator[None]:
    lock_path = run_dir / ".state.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("二维弹性运行状态正在被其他流程修改") from error
    os.close(descriptor)
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _transition_unlocked(
    directory: Path,
    expected: ElasticityRunState | None,
    target: ElasticityRunState,
) -> None:
    state_path = directory / "status.json"
    if expected is None:
        if state_path.exists():
            raise RuntimeError("二维弹性运行当前状态已经存在")
        if target is not ElasticityRunState.CREATED:
            raise RuntimeError("二维弹性运行初始状态只允许为 created")
    else:
        current = read_run_state(directory)
        if current is not expected:
            raise RuntimeError(
                f"二维弹性运行当前状态为 {current.value}，预期为 {expected.value}"
            )
        if (current, target) not in _ALLOWED_TRANSITIONS:
            raise RuntimeError(f"二维弹性运行不允许 {current.value} → {target.value}")
    _write_json_atomic(state_path, {"state": target.value})


def _validate_freeze_inputs(
    spec: ElasticityRunSpec,
    sample_plan: SamplePlan,
    dataset_files: DatasetFiles,
    selected_training: SelectedTraining,
) -> None:
    expected_samples = (
        spec.sampling.train_cases
        + spec.sampling.validation_cases
        + spec.sampling.test_cases
    )
    if sample_plan.sample_ids.size != expected_samples:
        raise ValueError("冻结样本计划数量与规格不一致")
    if sha256_file(dataset_files.development_path) != dataset_files.development_sha256:
        raise RuntimeError("冻结前开发数据 SHA-256 校验失败")
    seeds = tuple(candidate.seed for candidate in selected_training.candidates)
    if seeds != spec.training.seeds:
        raise ValueError("冻结候选训练记录与规格随机种子不一致")
    if (
        selected_training.selected_seed not in seeds
        or selected_training.selected.seed != selected_training.selected_seed
        or selected_training.selected is not selected_training.candidates[seeds.index(
            selected_training.selected_seed
        )]
    ):
        raise ValueError("冻结选中训练记录身份无效")


def _load_sealed_dataset(
    run_dir: Path,
    dataset_files: DatasetFiles,
    manifest: FreezeManifest,
) -> FieldDataset:
    identity = _read_json_object(run_dir / "dataset_identity.json", "数据身份")
    expected_identity = {
        "development_path": str(dataset_files.development_path.resolve()),
        "development_sha256": dataset_files.development_sha256,
        "sealed_test_path": str(dataset_files.sealed_test_path.resolve()),
        "sealed_test_sha256": dataset_files.sealed_test_sha256,
        "solver_manifest_path": str(dataset_files.manifest_path.resolve()),
        "solver_manifest_sha256": sha256_file(dataset_files.manifest_path),
    }
    if identity != expected_identity:
        raise RuntimeError("封存测试数据身份与冻结清单不一致")
    if (
        manifest.development_sha256 != dataset_files.development_sha256
        or manifest.sealed_test_sha256 != dataset_files.sealed_test_sha256
    ):
        raise RuntimeError("封存测试数据 SHA-256 身份与冻结清单不一致")
    if sha256_file(dataset_files.sealed_test_path) != dataset_files.sealed_test_sha256:
        raise RuntimeError("封存测试数据 SHA-256 校验失败")
    try:
        with np.load(dataset_files.sealed_test_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "sample_ids",
                "roles",
                "parameters",
                "coordinates",
                "fields",
            }:
                raise RuntimeError("封存测试 NPZ 数组字段无效")
            sample_ids = np.asarray(archive["sample_ids"], dtype=np.str_).copy()
            roles = np.asarray(archive["roles"], dtype=np.str_).copy()
            parameters = np.asarray(archive["parameters"], dtype=np.float64).copy()
            coordinates = np.asarray(archive["coordinates"], dtype=np.float64).copy()
            fields = np.asarray(archive["fields"], dtype=np.float64).copy()
    except RuntimeError:
        raise
    except (OSError, ValueError) as error:
        raise RuntimeError("无法安全读取封存测试 NPZ") from error
    if roles.shape != sample_ids.shape or not np.all(roles == "sealed_test"):
        raise RuntimeError("封存测试 NPZ 包含非 sealed_test 样本")
    plan = _read_json_object(run_dir / "sample_plan.json", "样本计划")
    plan_roles = np.asarray(plan.get("roles"), dtype=np.str_)
    sealed_indices = np.flatnonzero(plan_roles == "sealed_test")
    try:
        np.testing.assert_array_equal(
            sample_ids,
            np.asarray(plan["sample_ids"], dtype=np.str_)[sealed_indices],
        )
        np.testing.assert_allclose(
            parameters,
            np.asarray(plan["parameters"], dtype=np.float64)[sealed_indices],
            rtol=0.0,
            atol=0.0,
        )
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError("封存测试样本身份与冻结样本计划不一致") from error
    return FieldDataset(
        sample_ids=sample_ids,
        parameters=parameters,
        coordinates=coordinates,
        fields=fields,
        diagnostics={},
    )


def _load_normalization(path: Path) -> FieldNormalization:
    payload = _read_json_object(path, "归一化统计")
    if set(payload) != {
        "feature_mean",
        "feature_std",
        "coordinate_mean",
        "coordinate_std",
        "target_rms",
    }:
        raise RuntimeError("二维弹性归一化统计字段无效")
    try:
        return FieldNormalization(
            **{name: np.asarray(value, dtype=np.float64) for name, value in payload.items()}
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("二维弹性归一化统计数值无效") from error


def _benchmark_neural(
    model: torch.nn.Module,
    dataset: FieldDataset,
    normalization: FieldNormalization,
    query_batch_size: int,
) -> float:
    benchmark_dataset = dataset.subset(np.array([0], dtype=np.int64))
    for _ in range(10):
        predict_dataset(
            model,
            benchmark_dataset,
            normalization,
            torch.device("cpu"),
            query_batch_size,
        )
    timings = np.empty(100, dtype=np.float64)
    for index in range(timings.size):
        started = time.perf_counter()
        predict_dataset(
            model,
            benchmark_dataset,
            normalization,
            torch.device("cpu"),
            query_batch_size,
        )
        timings[index] = time.perf_counter() - started
    median = float(np.median(timings))
    if not np.isfinite(median) or median <= 0.0:
        raise RuntimeError("神经网络 CPU 速度基准无效")
    return median


def _benchmark_fenicsx(
    run_dir: Path,
    spec: ElasticityRunSpec,
    dataset: FieldDataset,
) -> float:
    count = min(5, dataset.parameters.shape[0])
    if count == 0:
        raise RuntimeError("FEniCSx 速度基准缺少封存样本")
    benchmark_dir = run_dir / "fenicsx_benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=False)
    first_parameters = dataset.parameters[0].tolist()
    samples = [
        {
            "sample_id": "train-99999-000000000000",
            "role": "train",
            "parameters": first_parameters,
        }
    ]
    samples.extend(
        {
            "sample_id": str(dataset.sample_ids[index]),
            "role": "sealed_test",
            "parameters": dataset.parameters[index].tolist(),
        }
        for index in range(count)
    )
    job = {
        "protocol_version": "elasticity-job-v1",
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
        "samples": samples,
    }
    job_path = benchmark_dir / "benchmark_job.json"
    _write_json_atomic(job_path, job)
    repo_root = Path(__file__).resolve().parents[4]
    completed = external_solver.run_solver_process(
        "generate",
        ("--job", str(job_path), "--output-dir", str(benchmark_dir)),
        repo_root,
        timeout_seconds=max(600.0, count * 120.0),
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "未知错误"
        raise RuntimeError(f"FEniCSx 速度基准失败：{detail}")
    response = external_solver.parse_solver_json(completed.stdout, "generate")
    manifest_path = Path(str(response.get("manifest", ""))).resolve()
    expected_path = (benchmark_dir / "datasets" / "dataset_manifest.json").resolve()
    if response.get("status") != "ok" or manifest_path != expected_path:
        raise RuntimeError("FEniCSx 速度基准返回身份无效")
    payload = _read_json_object(manifest_path, "FEniCSx 速度基准清单")
    solver_payload = payload.get("solver")
    if (
        not isinstance(solver_payload, dict)
        or solver_payload.get("timing_scope") != "assembly_solve_interpolation"
    ):
        raise RuntimeError("FEniCSx 速度基准计时范围无效")
    records = payload.get("samples")
    if not isinstance(records, list):
        raise RuntimeError("FEniCSx 速度基准样本记录无效")
    selected_ids = set(dataset.sample_ids[:count].tolist())
    timings = [
        float(record["diagnostics"]["solve_seconds"])
        for record in records
        if isinstance(record, dict) and record.get("sample_id") in selected_ids
    ]
    if len(timings) != count or not np.isfinite(timings).all() or min(timings) <= 0.0:
        raise RuntimeError("FEniCSx 速度基准计时无效")
    return float(np.median(timings))


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取{label}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label}必须是 JSON 对象")
    return payload


def _optional_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _joblib_dump_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _torch_save_atomic(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state_dict, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
