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
