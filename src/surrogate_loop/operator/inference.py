from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from surrogate_loop.operator.artifacts import (
    OPERATOR_MANIFEST_VERSION,
    REQUIRED_HASHED_FILES,
    sha256_file,
)
from surrogate_loop.operator.config import OperatorRunSpec
from surrogate_loop.operator.heat1d.dataset import NormalizationStats
from surrogate_loop.operator.heat1d.deeponet import (
    DeepONet,
    apply_heat_constraints,
    build_deeponet,
)
from surrogate_loop.operator.heat1d.evaluation import (
    FieldMetrics,
    deeponet_is_acceptable,
)
from surrogate_loop.operator.runtime import resolve_device

MAX_FIELD_QUERY_POINTS = 1_000_000


@dataclass(frozen=True)
class OperatorBundle:
    spec: OperatorRunSpec
    model: DeepONet
    normalization: NormalizationStats
    manifest: dict[str, object]
    device: torch.device


@dataclass(frozen=True)
class VerifiedOperatorRun:
    run_dir: Path
    spec: OperatorRunSpec
    manifest: dict[str, object]
    test_metrics: FieldMetrics
    pod_metrics: dict[str, object]
    training: dict[str, object]


def load_operator_bundle(
    run_dir: Path,
    device_request: str = "auto",
) -> OperatorBundle:
    run_dir = run_dir.resolve()
    verified = verify_operator_run(run_dir, require_accepted=True)
    if verified.manifest["status"] != "accepted":
        raise RuntimeError("该 DeepONet 运行未通过验收，禁止加载推理")
    spec = verified.spec
    normalization = _load_normalization(run_dir / "normalization.json")
    network = json.loads((run_dir / "network.json").read_text(encoding="utf-8"))
    expected_network = {
        "branch_input_dim": 3,
        "trunk_input_dim": 2,
        "hidden_width": spec.model.hidden_width,
        "hidden_layers": spec.model.hidden_layers,
        "latent_dim": spec.model.latent_dim,
    }
    if network != expected_network:
        raise RuntimeError("网络结构配置与运行规格不一致")
    device = resolve_device(device_request)
    model = build_deeponet(spec.model).to(device)
    state_dict = torch.load(
        run_dir / "deeponet_state.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return OperatorBundle(spec, model, normalization, verified.manifest, device)


def load_operator_spec_metadata(run_dir: Path) -> OperatorRunSpec:
    run_dir = run_dir.resolve()
    manifest = _load_manifest(run_dir)
    _verify_one_hash(run_dir, manifest, "spec.json")
    return OperatorRunSpec.model_validate_json(
        (run_dir / "spec.json").read_text(encoding="utf-8")
    )


def verify_operator_run(
    run_dir: Path, *, require_accepted: bool = False
) -> VerifiedOperatorRun:
    run_dir = run_dir.resolve()
    manifest = _load_manifest(run_dir)
    for name in sorted(REQUIRED_HASHED_FILES):
        _verify_one_hash(run_dir, manifest, name)
    spec = OperatorRunSpec.model_validate_json(
        (run_dir / "spec.json").read_text(encoding="utf-8")
    )
    expected_role = (
        "confirmatory_holdout" if spec.mode == "full" else "development_holdout"
    )
    if manifest["evaluation_role"] != expected_role:
        raise RuntimeError("运行清单评价角色与规格不一致")
    test_metrics = _load_field_metrics(run_dir / "test_metrics.json")
    expected_status = (
        "accepted"
        if deeponet_is_acceptable(test_metrics, spec.acceptance)
        else "rejected"
    )
    if manifest["status"] != expected_status:
        raise RuntimeError("运行清单状态与已验证指标不一致")
    _verify_solver_metrics(run_dir / "solver_metrics.json", spec)
    network = _load_json_object(run_dir / "network.json")
    expected_network = {
        "branch_input_dim": 3,
        "trunk_input_dim": 2,
        "hidden_width": spec.model.hidden_width,
        "hidden_layers": spec.model.hidden_layers,
        "latent_dim": spec.model.latent_dim,
    }
    if network != expected_network:
        raise RuntimeError("网络结构配置与运行规格不一致")
    if require_accepted and expected_status != "accepted":
        raise RuntimeError("该 DeepONet 运行未通过验收，禁止加载推理")
    return VerifiedOperatorRun(
        run_dir=run_dir,
        spec=spec,
        manifest=manifest,
        test_metrics=test_metrics,
        pod_metrics=_load_json_object(run_dir / "pod_metrics.json"),
        training=_load_json_object(run_dir / "training_history.json"),
    )


def validate_prediction_request(
    spec: OperatorRunSpec,
    alpha: float,
    amplitude_1: float,
    amplitude_2: float,
    *,
    x: float | None = None,
    t: float | None = None,
    nx: int | None = None,
    nt: int | None = None,
) -> None:
    _validate_parameters(
        spec, np.array([alpha, amplitude_1, amplitude_2], dtype=np.float64)
    )
    if x is not None or t is not None:
        if x is None or t is None:
            raise ValueError("点预测必须同时提供 --x 和 --t")
        _validate_coordinates(np.array([x]), "x")
        _validate_coordinates(np.array([t]), "t")
        return
    resolved_nx = spec.grid.nx if nx is None else nx
    resolved_nt = spec.grid.nt if nt is None else nt
    validate_field_grid(resolved_nx, resolved_nt)


def validate_field_grid(nx: int, nt: int) -> None:
    if nx < 2 or nt < 2:
        raise ValueError("场预测的 nx 和 nt 必须至少为 2")
    _validate_query_count(nx, nt)


def predict_point(
    bundle: OperatorBundle,
    alpha: float,
    amplitude_1: float,
    amplitude_2: float,
    *,
    x: float,
    t: float,
) -> float:
    field = predict_field(
        bundle,
        alpha,
        amplitude_1,
        amplitude_2,
        x=np.array([x], dtype=np.float64),
        t=np.array([t], dtype=np.float64),
    )
    return float(field[0, 0])


def predict_field(
    bundle: OperatorBundle,
    alpha: float,
    amplitude_1: float,
    amplitude_2: float,
    *,
    x: NDArray[np.float64],
    t: NDArray[np.float64],
) -> NDArray[np.float64]:
    parameters = np.array([[alpha, amplitude_1, amplitude_2]], dtype=np.float64)
    _validate_parameters(bundle.spec, parameters[0])
    x = _validate_coordinates(x, "x")
    t = _validate_coordinates(t, "t")
    _validate_query_count(int(x.size), int(t.size))
    coordinates = np.stack(np.meshgrid(x, t, indexing="xy"), axis=-1).reshape(-1, 2)
    normalized_parameters = bundle.normalization.normalize_parameters(parameters).astype(
        np.float32
    )
    normalized_coordinates = bundle.normalization.normalize_coordinates(coordinates).astype(
        np.float32
    )
    branch = torch.as_tensor(normalized_parameters, device=bundle.device)
    physical_branch = torch.as_tensor(parameters.astype(np.float32), device=bundle.device)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, normalized_coordinates.shape[0], 4096):
            trunk = torch.as_tensor(
                normalized_coordinates[start : start + 4096], device=bundle.device
            )
            physical_trunk = torch.as_tensor(
                coordinates[start : start + 4096].astype(np.float32),
                device=bundle.device,
            )
            constrained = apply_heat_constraints(
                bundle.model(branch, trunk),
                physical_branch,
                physical_trunk,
                bundle.normalization.target_mean,
                bundle.normalization.target_std,
            )
            predictions.append(constrained.cpu().numpy())
    normalized_field = np.concatenate(predictions, axis=1)
    field = bundle.normalization.denormalize_targets(normalized_field)
    return field.reshape(t.size, x.size)


def _validate_parameters(spec: OperatorRunSpec, parameters: np.ndarray) -> None:
    ranges = (
        spec.problem.alpha,
        spec.problem.amplitude_1,
        spec.problem.amplitude_2,
    )
    if not all(
        math.isfinite(float(value)) and bounds.low <= value <= bounds.high
        for value, bounds in zip(parameters, ranges, strict=True)
    ):
        raise ValueError("输入参数超出训练参数域或不是有限数")


def _validate_coordinates(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or array.size == 0
        or not np.isfinite(array).all()
        or np.any(array < 0.0)
        or np.any(array > 1.0)
        or np.any(np.diff(array) <= 0.0)
    ):
        raise ValueError(f"{name} 坐标必须是查询域 [0,1] 内的有限严格递增一维数组")
    return array


def _load_manifest(run_dir: Path) -> dict[str, object]:
    manifest = _load_json_object(run_dir / "manifest.json")
    required_keys = {
        "version",
        "problem",
        "evaluation_role",
        "status",
        "runtime",
        "sha256",
    }
    if set(manifest) != required_keys:
        raise RuntimeError("运行清单字段集合无效")
    if manifest["version"] != OPERATOR_MANIFEST_VERSION:
        raise RuntimeError("运行清单版本不受支持")
    if manifest["problem"] != "heat_1d_operator_v1":
        raise RuntimeError("运行清单问题类型无效")
    if manifest["evaluation_role"] not in {
        "development_holdout",
        "confirmatory_holdout",
    }:
        raise RuntimeError("运行清单评价角色无效")
    if manifest["status"] not in {"accepted", "rejected"}:
        raise RuntimeError("运行清单状态无效")
    hashes = manifest["sha256"]
    if not isinstance(hashes, dict) or set(hashes) != REQUIRED_HASHED_FILES:
        raise RuntimeError("运行清单必需文件集合无效")
    if not all(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        for value in hashes.values()
    ):
        raise RuntimeError("运行清单 SHA-256 格式无效")
    return manifest


def _validate_query_count(nx: int, nt: int) -> None:
    if nx * nt > MAX_FIELD_QUERY_POINTS:
        raise ValueError(f"场预测查询点总数不能超过 {MAX_FIELD_QUERY_POINTS}")


def _verify_one_hash(
    run_dir: Path, manifest: dict[str, object], name: str
) -> None:
    hashes = manifest["sha256"]
    if not isinstance(hashes, dict):
        raise RuntimeError("运行清单哈希字段无效")
    path = run_dir / name
    expected = hashes[name]
    if not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f"运行产物哈希校验失败：{name}")


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取可信 JSON 产物：{path.name}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON 产物必须是对象：{path.name}")
    return payload


def _load_field_metrics(path: Path) -> FieldMetrics:
    payload = _load_json_object(path)
    expected = {
        "median_relative_l2",
        "p95_relative_l2",
        "worst_relative_l2",
        "normalized_rmse",
        "initial_relative_l2",
        "boundary_max_absolute_error",
    }
    if set(payload) != expected:
        raise RuntimeError("测试指标字段集合无效")
    try:
        metrics = FieldMetrics(**{name: float(payload[name]) for name in expected})
    except (TypeError, ValueError) as error:
        raise RuntimeError("测试指标不是有效数值") from error
    if not np.isfinite(np.fromiter(metrics.to_dict().values(), dtype=np.float64)).all():
        raise RuntimeError("测试指标包含 NaN 或 Inf")
    return metrics


def _verify_solver_metrics(path: Path, spec: OperatorRunSpec) -> None:
    payload = _load_json_object(path)
    try:
        boundary = float(payload["boundary_max_absolute_error"])
        p95 = float(payload["p95_relative_l2"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("求解器指标格式无效") from error
    if not (
        math.isfinite(boundary)
        and math.isfinite(p95)
        and boundary <= spec.solver_acceptance.max_boundary_error
        and p95 <= spec.solver_acceptance.max_p95_relative_l2
    ):
        raise RuntimeError("已保存的求解器指标未通过规格验收")


def _load_normalization(path: Path) -> NormalizationStats:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return NormalizationStats(
        parameter_mean=np.asarray(payload["parameter_mean"], dtype=np.float64),
        parameter_std=np.asarray(payload["parameter_std"], dtype=np.float64),
        coordinate_mean=np.asarray(payload["coordinate_mean"], dtype=np.float64),
        coordinate_std=np.asarray(payload["coordinate_std"], dtype=np.float64),
        target_mean=float(payload["target_mean"]),
        target_std=float(payload["target_std"]),
    )
