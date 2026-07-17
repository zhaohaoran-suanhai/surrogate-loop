from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from surrogate_loop.operator.elasticity2d.artifacts import (
    AcceptanceResult,
    ElasticityRunState,
    _write_json_atomic,
    evaluate_sealed_once,
    freeze_run,
    read_run_state,
    transition_run,
)
from surrogate_loop.operator.elasticity2d.config import (
    ElasticityRunSpec,
    load_elasticity_spec,
)
from surrogate_loop.operator.elasticity2d.dataset import (
    DatasetFiles,
    generate_or_reuse_dataset,
    load_development_partitions,
)
from surrogate_loop.operator.elasticity2d.deeponet import build_elasticity_deeponet
from surrogate_loop.operator.elasticity2d.evaluation import compute_elasticity_metrics
from surrogate_loop.operator.elasticity2d.pod_rbf import PodRbfBaseline
from surrogate_loop.operator.elasticity2d.problem import elasticity_features
from surrogate_loop.operator.elasticity2d.reporting import write_smoke_diagnostics
from surrogate_loop.operator.elasticity2d.sampling import build_sample_plan
from surrogate_loop.operator.elasticity2d.training import (
    predict_dataset,
    train_and_select,
)
from surrogate_loop.operator.field_data import FieldDataset, FieldNormalization, sha256_file
from surrogate_loop.operator.runtime import resolve_device, seed_everything


@dataclass(frozen=True)
class ElasticityRunResult:
    run_dir: Path
    status: str
    deeponet_metrics: dict[str, float]
    pod_rbf_metrics: dict[str, float]


def run_elasticity_pipeline(
    config_path: Path,
    runs_dir: Path,
    request: str,
) -> ElasticityRunResult:
    if not request.strip():
        raise ValueError("二维弹性运行请求不能为空")
    spec = load_elasticity_spec(config_path)
    if spec.mode == "calibration":
        raise ValueError("calibration 配置必须通过专用校准入口运行")
    run_dir = _resolve_run_directory(runs_dir, spec, request)
    resumed = _completed_result(run_dir, spec)
    if resumed is not None:
        return resumed

    try:
        device = resolve_device(spec.runtime.device)
        seed_everything(spec.sampling.seed)
        sample_plan = build_sample_plan(spec)
        repo_root = Path(__file__).resolve().parents[4]
        dataset_files = generate_or_reuse_dataset(
            spec, sample_plan, run_dir, repo_root
        )
        state = read_run_state(run_dir)
        if state is ElasticityRunState.CREATED:
            transition_run(
                run_dir,
                ElasticityRunState.CREATED,
                ElasticityRunState.SOLVER_ACCEPTED,
            )
            state = ElasticityRunState.SOLVER_ACCEPTED

        if state in {ElasticityRunState.SOLVER_ACCEPTED, ElasticityRunState.TRAINED}:
            partitions = load_development_partitions(dataset_files, sample_plan)
            normalization = FieldNormalization.fit(
                elasticity_features(partitions.train.parameters),
                partitions.train.coordinates,
                partitions.train.fields,
            )
            baseline = PodRbfBaseline(
                energy_threshold=spec.pod.energy_threshold,
                max_components=spec.pod.max_components,
            ).fit(partitions.train.parameters, partitions.train.fields)
            selected = train_and_select(
                spec, partitions, normalization, device
            )
            if state is ElasticityRunState.SOLVER_ACCEPTED:
                transition_run(
                    run_dir,
                    ElasticityRunState.SOLVER_ACCEPTED,
                    ElasticityRunState.TRAINED,
                )
            if spec.mode == "smoke":
                return _evaluate_development(
                    run_dir,
                    spec,
                    dataset_files,
                    normalization,
                    baseline,
                    selected.selected.state_dict,
                )
            freeze_run(
                run_dir,
                spec,
                sample_plan,
                dataset_files,
                normalization,
                baseline,
                selected,
            )
            state = ElasticityRunState.FROZEN

        if state is ElasticityRunState.FROZEN:
            acceptance = evaluate_sealed_once(run_dir, dataset_files)
            return _result_from_acceptance(run_dir, acceptance)
        raise RuntimeError(f"二维弹性流水线无法从状态 {state.value} 恢复")
    except Exception as error:
        _write_json_atomic(
            run_dir / "pipeline_error.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        try:
            state = read_run_state(run_dir)
            if state in {
                ElasticityRunState.CREATED,
                ElasticityRunState.SOLVER_ACCEPTED,
                ElasticityRunState.TRAINED,
                ElasticityRunState.FROZEN,
            }:
                transition_run(run_dir, state, ElasticityRunState.FAILED)
        except RuntimeError:
            pass
        raise


def _resolve_run_directory(
    runs_dir: Path,
    spec: ElasticityRunSpec,
    request: str,
) -> Path:
    identity = {
        "request": request,
        "spec": spec.model_dump(mode="json"),
    }
    canonical = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    run_dir = runs_dir.resolve() / f"elasticity-{spec.mode}-{digest[:12]}"
    identity_payload = {**identity, "identity_sha256": digest}
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        _write_json_atomic(run_dir / "request.json", identity_payload)
        transition_run(run_dir, None, ElasticityRunState.CREATED)
        return run_dir
    try:
        existing = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("二维弹性恢复请求身份无法读取") from error
    if existing != identity_payload:
        raise RuntimeError("二维弹性恢复请求身份不一致")
    if read_run_state(run_dir) is ElasticityRunState.FAILED:
        raise RuntimeError("failed 运行不得以相同身份恢复")
    return run_dir


def _completed_result(
    run_dir: Path,
    spec: ElasticityRunSpec,
) -> ElasticityRunResult | None:
    state = read_run_state(run_dir)
    if spec.mode == "smoke" and state is ElasticityRunState.TRAINED:
        path = run_dir / "development_evaluation.json"
        stage_path = run_dir / "development_stage.json"
        if path.is_file() and stage_path.is_file() and _stage_hash_matches(
            stage_path, path
        ):
            return _read_result(path, run_dir)
    if state in {ElasticityRunState.ACCEPTED, ElasticityRunState.REJECTED}:
        if _acceptance_stage_matches(run_dir):
            return _read_result(run_dir / "acceptance.json", run_dir)
        raise RuntimeError("二维弹性验收阶段摘要完整性校验失败")
    return None


def _evaluate_development(
    run_dir: Path,
    spec: ElasticityRunSpec,
    dataset_files: DatasetFiles,
    normalization: FieldNormalization,
    baseline: PodRbfBaseline,
    state_dict: dict[str, torch.Tensor],
) -> ElasticityRunResult:
    dataset = _load_development_test(dataset_files)
    model = build_elasticity_deeponet(spec.model).to("cpu")
    model.load_state_dict(state_dict)
    prediction = predict_dataset(
        model,
        dataset,
        normalization,
        torch.device("cpu"),
        spec.training.query_batch_size,
    )
    deeponet_metrics = compute_elasticity_metrics(
        dataset.fields, prediction, dataset.parameters, dataset.coordinates
    ).to_dict()
    pod_prediction = baseline.predict(dataset.parameters)
    pod_metrics = compute_elasticity_metrics(
        dataset.fields, pod_prediction, dataset.parameters, dataset.coordinates
    ).to_dict()
    diagnostic_hashes = write_smoke_diagnostics(
        run_dir,
        dataset,
        prediction,
        dataset_files.manifest_path,
    )
    result = ElasticityRunResult(
        run_dir=run_dir,
        status="development_complete",
        deeponet_metrics=deeponet_metrics,
        pod_rbf_metrics=pod_metrics,
    )
    _write_json_atomic(
        run_dir / "development_evaluation.json",
        {
            "status": result.status,
            "deeponet_metrics": deeponet_metrics,
            "pod_rbf_metrics": pod_metrics,
        },
    )
    _write_json_atomic(
        run_dir / "development_stage.json",
        {
            "status": "complete",
            "result_sha256": sha256_file(run_dir / "development_evaluation.json"),
            "diagnostic_sha256": diagnostic_hashes,
        },
    )
    return result


def _load_development_test(files: DatasetFiles) -> FieldDataset:
    if sha256_file(files.sealed_test_path) != files.sealed_test_sha256:
        raise RuntimeError("Smoke 开发测试数据 SHA-256 校验失败")
    try:
        with np.load(files.sealed_test_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "sample_ids",
                "roles",
                "parameters",
                "coordinates",
                "fields",
            }:
                raise RuntimeError("Smoke 开发测试数据字段无效")
            roles = np.asarray(archive["roles"], dtype=np.str_)
            if not np.all(roles == "development_test"):
                raise RuntimeError("Smoke 测试数据角色必须为 development_test")
            return FieldDataset(
                sample_ids=np.asarray(archive["sample_ids"], dtype=np.str_).copy(),
                parameters=np.asarray(archive["parameters"], dtype=np.float64).copy(),
                coordinates=np.asarray(archive["coordinates"], dtype=np.float64).copy(),
                fields=np.asarray(archive["fields"], dtype=np.float64).copy(),
                diagnostics={},
            )
    except RuntimeError:
        raise
    except (OSError, ValueError) as error:
        raise RuntimeError("无法安全读取 Smoke 开发测试数据") from error


def _result_from_acceptance(
    run_dir: Path,
    acceptance: AcceptanceResult,
) -> ElasticityRunResult:
    return ElasticityRunResult(
        run_dir=run_dir,
        status=acceptance.status,
        deeponet_metrics=acceptance.deeponet_metrics,
        pod_rbf_metrics=acceptance.pod_rbf_metrics,
    )


def _read_result(path: Path, run_dir: Path) -> ElasticityRunResult:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ElasticityRunResult(
            run_dir=run_dir,
            status=str(payload["status"]),
            deeponet_metrics={
                name: float(value)
                for name, value in payload["deeponet_metrics"].items()
            },
            pod_rbf_metrics={
                name: float(value)
                for name, value in payload["pod_rbf_metrics"].items()
            },
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("二维弹性阶段结果无法恢复") from error


def _stage_hash_matches(stage_path: Path, result_path: Path) -> bool:
    try:
        payload = json.loads(stage_path.read_text(encoding="utf-8"))
        return bool(
            isinstance(payload, dict)
            and set(payload) == {"status", "result_sha256", "diagnostic_sha256"}
            and payload["status"] == "complete"
            and payload["result_sha256"] == sha256_file(result_path)
            and _diagnostic_hashes_match(stage_path.parent, payload["diagnostic_sha256"])
        )
    except (OSError, json.JSONDecodeError, RuntimeError):
        return False


def _diagnostic_hashes_match(run_dir: Path, payload: object) -> bool:
    if not isinstance(payload, dict) or set(payload) != {
        "diagnostics/displacement_comparison.png",
        "diagnostics/fenicsx_stress_summary.png",
    }:
        return False
    return all(
        isinstance(digest, str)
        and len(digest) == 64
        and sha256_file(run_dir / relative) == digest
        for relative, digest in payload.items()
    )


def _acceptance_stage_matches(run_dir: Path) -> bool:
    try:
        payload = json.loads(
            (run_dir / "acceptance_stage.json").read_text(encoding="utf-8")
        )
        return bool(
            isinstance(payload, dict)
            and set(payload)
            == {
                "status",
                "acceptance_sha256",
                "sealed_summary_sha256",
                "freeze_manifest_sha256",
                "fenicsx_benchmark_manifest_sha256",
            }
            and payload["status"] == "complete"
            and payload["acceptance_sha256"]
            == sha256_file(run_dir / "acceptance.json")
            and payload["sealed_summary_sha256"]
            == sha256_file(run_dir / "sealed_test_summary.json")
            and payload["freeze_manifest_sha256"]
            == sha256_file(run_dir / "freeze_manifest.json")
            and _optional_stage_hash_matches(
                payload["fenicsx_benchmark_manifest_sha256"],
                run_dir
                / "fenicsx_benchmark"
                / "datasets"
                / "dataset_manifest.json",
            )
        )
    except (OSError, json.JSONDecodeError, RuntimeError):
        return False


def _optional_stage_hash_matches(expected: object, path: Path) -> bool:
    if expected is None:
        return not path.exists()
    return isinstance(expected, str) and expected == sha256_file(path)
