from __future__ import annotations

import torch
from torch import Tensor, nn

from surrogate_loop.operator.config import DeepONetSpec


class DeepONet(nn.Module):
    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        hidden_width: int,
        hidden_layers: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        dimensions = (
            branch_input_dim,
            trunk_input_dim,
            hidden_width,
            hidden_layers,
            latent_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("DeepONet 的输入维度、宽度、层数和潜在维度必须为正数")
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        self.hidden_width = hidden_width
        self.hidden_layers = hidden_layers
        self.latent_dim = latent_dim
        self.branch_net = _make_mlp(
            branch_input_dim, hidden_width, hidden_layers, latent_dim
        )
        self.trunk_net = _make_mlp(trunk_input_dim, hidden_width, hidden_layers, latent_dim)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, branch: Tensor, trunk: Tensor) -> Tensor:
        if branch.ndim != 2 or branch.shape[1] != self.branch_input_dim:
            raise ValueError(
                f"Branch 输入形状必须为 (batch, {self.branch_input_dim})"
            )
        if trunk.ndim != 2 or trunk.shape[1] != self.trunk_input_dim:
            raise ValueError(f"Trunk 输入形状必须为 (queries, {self.trunk_input_dim})")
        branch_features = self.branch_net(branch)
        trunk_features = self.trunk_net(trunk)
        return torch.einsum("bp,qp->bq", branch_features, trunk_features) + self.bias


def _make_mlp(input_dim: int, width: int, layers: int, output_dim: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    current_width = input_dim
    for _ in range(layers):
        modules.extend((nn.Linear(current_width, width), nn.Tanh()))
        current_width = width
    modules.append(nn.Linear(current_width, output_dim))
    network = nn.Sequential(*modules)
    for module in network.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return network


def build_deeponet(spec: DeepONetSpec) -> DeepONet:
    return DeepONet(
        branch_input_dim=3,
        trunk_input_dim=2,
        hidden_width=spec.hidden_width,
        hidden_layers=spec.hidden_layers,
        latent_dim=spec.latent_dim,
    )


def apply_heat_constraints(
    raw_normalized: Tensor,
    physical_parameters: Tensor,
    physical_coordinates: Tensor,
    target_mean: float,
    target_std: float,
) -> Tensor:
    if raw_normalized.ndim != 2:
        raise ValueError("约束前的 DeepONet 输出必须为二维张量")
    if physical_parameters.ndim != 2 or physical_parameters.shape[1] != 3:
        raise ValueError("物理参数形状必须为 (batch, 3)")
    if physical_coordinates.ndim != 2 or physical_coordinates.shape[1] != 2:
        raise ValueError("物理坐标形状必须为 (queries, 2)")
    expected_shape = (physical_parameters.shape[0], physical_coordinates.shape[0])
    if raw_normalized.shape != expected_shape:
        raise ValueError(f"DeepONet 输出形状必须为 {expected_shape}")
    if target_std <= 0.0:
        raise ValueError("target_std 必须为正数")
    x = physical_coordinates[:, 0]
    t = physical_coordinates[:, 1]
    amplitude_1 = physical_parameters[:, 1:2]
    amplitude_2 = physical_parameters[:, 2:3]
    initial_field = amplitude_1 * torch.sin(torch.pi * x)[None, :] + amplitude_2 * torch.sin(
        2.0 * torch.pi * x
    )[None, :]
    raw_physical = raw_normalized * target_std + target_mean
    correction_scale = (4.0 * x * (1.0 - x) * t)[None, :]
    constrained_physical = initial_field + correction_scale * raw_physical
    return (constrained_physical - target_mean) / target_std
