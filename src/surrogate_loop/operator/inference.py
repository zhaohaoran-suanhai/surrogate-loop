from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from surrogate_loop.operator.artifacts import sha256_file
from surrogate_loop.operator.config import OperatorRunSpec
from surrogate_loop.operator.heat1d.dataset import NormalizationStats
from surrogate_loop.operator.heat1d.deeponet import (
    DeepONet,
    apply_heat_constraints,
    build_deeponet,
)
from surrogate_loop.operator.runtime import resolve_device


@dataclass(frozen=True)
class OperatorBundle:
    spec: OperatorRunSpec
    model: DeepONet
    normalization: NormalizationStats
    manifest: dict[str, object]
    device: torch.device


def load_operator_bundle(
    run_dir: Path, device_request: str = "auto"
) -> OperatorBundle:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        if sha256_file(run_dir / name) != expected:
            raise RuntimeError(f"运行产物哈希校验失败：{name}")
    if manifest.get("status") != "accepted":
        raise RuntimeError("该 DeepONet 运行未通过验收，禁止加载推理")
    spec = OperatorRunSpec.model_validate_json(
        (run_dir / "spec.json").read_text(encoding="utf-8")
    )
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
    return OperatorBundle(spec, model, normalization, manifest, device)


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
        or np.any(np.diff(array) < 0.0)
    ):
        raise ValueError(f"{name} 坐标必须是查询域 [0,1] 内的有限递增一维数组")
    return array


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
