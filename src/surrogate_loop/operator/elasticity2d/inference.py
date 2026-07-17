from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from surrogate_loop.operator.elasticity2d.artifacts import (
    ElasticityRunState,
    FreezeManifest,
    read_run_state,
    verify_freeze_manifest,
)
from surrogate_loop.operator.elasticity2d.config import ElasticityRunSpec
from surrogate_loop.operator.elasticity2d.deeponet import (
    apply_elasticity_constraints,
    build_elasticity_deeponet,
)
from surrogate_loop.operator.elasticity2d.development_report import (
    read_verified_development_report,
)
from surrogate_loop.operator.elasticity2d.problem import elasticity_basis_features
from surrogate_loop.operator.field_data import FieldNormalization, sha256_file
from surrogate_loop.operator.runtime import resolve_device
from surrogate_loop.operator.vector_deeponet import VectorDeepONet

MAX_QUERY_POINTS = 1_000_000


@dataclass(frozen=True)
class ElasticityBundle:
    spec: ElasticityRunSpec
    model: VectorDeepONet
    normalization: FieldNormalization
    manifest: FreezeManifest
    device: torch.device


def load_elasticity_spec_metadata(run_dir: Path) -> ElasticityRunSpec:
    directory = run_dir.resolve()
    manifest = _read_json(directory / "freeze_manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or "spec.json" not in files:
        raise RuntimeError("冻结清单缺少规格哈希")
    if sha256_file(directory / "spec.json") != files["spec.json"]:
        raise RuntimeError("二维弹性规格 SHA-256 校验失败")
    return ElasticityRunSpec.model_validate_json(
        (directory / "spec.json").read_text(encoding="utf-8")
    )


def validate_elasticity_request(
    spec: ElasticityRunSpec,
    parameters: NDArray[np.float64],
    coordinates: NDArray[np.float64] | None = None,
    *,
    nx: int | None = None,
    ny: int | None = None,
) -> None:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (1, 6) or not np.isfinite(values).all():
        raise ValueError("预测参数必须是六个有限值")
    ranges = (
        spec.problem.young_modulus,
        spec.problem.poisson_ratio,
        spec.problem.load_magnitude,
        spec.problem.load_angle,
        spec.problem.load_center,
        spec.problem.load_width,
    )
    if any(
        value < bounds.low or value > bounds.high
        for value, bounds in zip(values[0], ranges, strict=True)
    ):
        raise ValueError("预测参数超出训练参数域")
    if values[0, 2] / values[0, 0] > 1e-2:
        raise ValueError("载荷与弹性模量之比超过小变形合同")
    if coordinates is not None:
        _validate_coordinates(coordinates)
        if nx is not None or ny is not None:
            raise ValueError("点预测和规则网格参数不能同时提供")
        return
    resolved_nx = spec.observation.nx if nx is None else nx
    resolved_ny = spec.observation.ny if ny is None else ny
    if resolved_nx < 2 or resolved_ny < 2:
        raise ValueError("规则网格 nx 和 ny 必须至少为 2")
    if resolved_nx * resolved_ny > MAX_QUERY_POINTS:
        raise ValueError(f"查询点总数不能超过 {MAX_QUERY_POINTS}")


def load_elasticity_bundle(
    run_dir: Path,
    device_request: str = "auto",
) -> ElasticityBundle:
    directory = run_dir.resolve()
    manifest, state, _ = verify_elasticity_acceptance(directory)
    if state is not ElasticityRunState.ACCEPTED:
        raise RuntimeError("二维弹性运行未通过验收，禁止加载推理")
    spec = ElasticityRunSpec.model_validate_json(
        (directory / "spec.json").read_text(encoding="utf-8")
    )
    normalization = _load_normalization(directory / "normalization.json")
    expected_network = {
        "architecture": spec.model.architecture,
        "branch_input_dim": 3,
        "trunk_input_dim": 2,
        "output_dim": 4,
        "hidden_width": spec.model.hidden_width,
        "hidden_layers": spec.model.hidden_layers,
        "latent_dim": spec.model.latent_dim,
    }
    if _read_json(directory / "network.json") != expected_network:
        raise RuntimeError("二维弹性网络结构与规格不一致")
    device = resolve_device(device_request)
    model = build_elasticity_deeponet(spec.model).to(device)
    state_dict = torch.load(
        directory / "deeponet_state.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()
    return ElasticityBundle(spec, model, normalization, manifest, device)


def verify_elasticity_acceptance(
    run_dir: Path,
) -> tuple[FreezeManifest, ElasticityRunState, dict[str, object]]:
    directory = run_dir.resolve()
    manifest = verify_freeze_manifest(directory)
    state = read_run_state(directory)
    if state not in {ElasticityRunState.ACCEPTED, ElasticityRunState.REJECTED}:
        raise RuntimeError("二维弹性运行尚未完成验收")
    _verify_acceptance_stage(directory)
    acceptance = _read_json(directory / "acceptance.json")
    if acceptance.get("status") != state.value:
        raise RuntimeError("二维弹性验收状态与指标文件不一致")
    return manifest, state, acceptance


def read_elasticity_report(
    run_dir: Path,
) -> tuple[ElasticityRunState, dict[str, object]]:
    directory = run_dir.resolve()
    state = read_run_state(directory)
    if state in {ElasticityRunState.ACCEPTED, ElasticityRunState.REJECTED}:
        _, verified_state, payload = verify_elasticity_acceptance(directory)
        return verified_state, payload
    if state is not ElasticityRunState.TRAINED:
        raise RuntimeError("二维弹性运行尚无可读报告")
    return state, read_verified_development_report(directory)


def predict_elasticity_points(
    bundle: ElasticityBundle,
    parameters: NDArray[np.float64],
    coordinates: NDArray[np.float64],
) -> NDArray[np.float64]:
    values = np.asarray(parameters, dtype=np.float64)
    points = _validate_coordinates(coordinates)
    validate_elasticity_request(bundle.spec, values, points)
    features = bundle.normalization.normalize_features(
        elasticity_basis_features(values)
    ).astype(np.float32)
    normalized_points = bundle.normalization.normalize_coordinates(points).astype(
        np.float32
    )
    with torch.no_grad():
        raw = bundle.model(
            torch.as_tensor(features, device=bundle.device),
            torch.as_tensor(normalized_points, device=bundle.device),
        )
        prediction = apply_elasticity_constraints(
            raw,
            torch.as_tensor(values.astype(np.float32), device=bundle.device),
            torch.as_tensor(points.astype(np.float32), device=bundle.device),
        )
    return prediction[0].cpu().numpy().astype(np.float64)


def _validate_coordinates(coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
    points = np.asarray(coordinates, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or points.shape[0] == 0
        or points.shape[0] > MAX_QUERY_POINTS
        or not np.isfinite(points).all()
        or np.any(points[:, 0] < 0.0)
        or np.any(points[:, 0] > 4.0)
        or np.any(points[:, 1] < 0.0)
        or np.any(points[:, 1] > 1.0)
    ):
        raise ValueError("预测坐标必须是区域 [0,4]×[0,1] 内的有限 (n,2) 数组")
    return points


def _verify_acceptance_stage(run_dir: Path) -> None:
    stage = _read_json(run_dir / "acceptance_stage.json")
    if set(stage) != {
        "status",
        "acceptance_sha256",
        "sealed_summary_sha256",
        "freeze_manifest_sha256",
        "fenicsx_benchmark_manifest_sha256",
    }:
        raise RuntimeError("二维弹性验收阶段字段无效")
    hashes = {
        "acceptance_sha256": run_dir / "acceptance.json",
        "sealed_summary_sha256": run_dir / "sealed_test_summary.json",
        "freeze_manifest_sha256": run_dir / "freeze_manifest.json",
    }
    if stage.get("status") != "complete" or any(
        stage.get(name) != sha256_file(path) for name, path in hashes.items()
    ):
        raise RuntimeError("二维弹性验收阶段完整性校验失败")
    benchmark = stage.get("fenicsx_benchmark_manifest_sha256")
    benchmark_path = run_dir / "fenicsx_benchmark/datasets/dataset_manifest.json"
    if benchmark is None and benchmark_path.exists():
        raise RuntimeError("FEniCSx 基准证据未进入验收摘要")
    if benchmark is not None and benchmark != sha256_file(benchmark_path):
        raise RuntimeError("FEniCSx 基准证据完整性校验失败")


def _load_normalization(path: Path) -> FieldNormalization:
    payload = _read_json(path)
    try:
        return FieldNormalization(
            **{name: np.asarray(value, dtype=np.float64) for name, value in payload.items()}
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("二维弹性归一化统计无效") from error


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取可信 JSON：{path.name}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"可信 JSON 必须为对象：{path.name}")
    return value
