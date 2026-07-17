from __future__ import annotations

import torch
from torch import Tensor

from surrogate_loop.operator.elasticity2d.config import VectorDeepONetSpec
from surrogate_loop.operator.elasticity2d.problem import LENGTH
from surrogate_loop.operator.vector_deeponet import VectorDeepONet


def build_elasticity_deeponet(spec: VectorDeepONetSpec) -> VectorDeepONet:
    if spec.architecture != "directional_linear_v2":
        raise ValueError(f"不支持的二维弹性网络架构：{spec.architecture}")
    return VectorDeepONet(
        branch_input_dim=3,
        trunk_input_dim=2,
        output_dim=4,
        hidden_width=spec.hidden_width,
        hidden_layers=spec.hidden_layers,
        latent_dim=spec.latent_dim,
    )


def apply_elasticity_constraints(
    raw: Tensor,
    physical_parameters: Tensor,
    physical_coordinates: Tensor,
) -> Tensor:
    """施加悬臂梁固支边界和线弹性载荷/模量尺度关系。"""
    if raw.ndim != 3:
        raise ValueError("约束前的 Vector DeepONet 输出必须为三维张量")
    if raw.shape[2] != 4:
        raise ValueError("约束前的 Vector DeepONet 输出必须包含 4 个方向响应基分量")
    if physical_parameters.ndim != 2 or physical_parameters.shape[1] != 6:
        raise ValueError("物理参数形状必须为 (batch, 6)")
    if physical_coordinates.ndim != 2 or physical_coordinates.shape[1] != 2:
        raise ValueError("物理坐标形状必须为 (queries, 2)")
    expected_shape = (
        physical_parameters.shape[0],
        physical_coordinates.shape[0],
        4,
    )
    if raw.shape != expected_shape:
        raise ValueError(f"Vector DeepONet 输出形状必须为 {expected_shape}")
    if not torch.isfinite(physical_parameters).all():
        raise ValueError("物理参数必须全部有限")
    if not torch.isfinite(physical_coordinates).all():
        raise ValueError("物理坐标必须全部有限")
    if torch.any(physical_parameters[:, 0] <= 0.0):
        raise ValueError("弹性模量必须为正数")

    scale = (
        physical_parameters[:, 2] / physical_parameters[:, 0]
    )[:, None, None]
    clamp = (physical_coordinates[:, 0] / LENGTH)[None, :, None]
    angle = physical_parameters[:, 3][:, None, None]
    horizontal = raw[..., :2]
    vertical = raw[..., 2:]
    directional_response = torch.cos(angle) * horizontal + torch.sin(angle) * vertical
    return scale * clamp * directional_response
