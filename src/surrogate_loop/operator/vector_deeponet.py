from __future__ import annotations

from torch import Tensor, nn


class VectorDeepONet(nn.Module):
    """共享 Branch/Trunk 编码、输出笛卡尔积向量场的 DeepONet。"""

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        output_dim: int,
        hidden_width: int,
        hidden_layers: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        dimensions = (
            branch_input_dim,
            trunk_input_dim,
            output_dim,
            hidden_width,
            hidden_layers,
            latent_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("Vector DeepONet 的所有维度、宽度和层数必须为正数")
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        self.output_dim = output_dim
        self.hidden_width = hidden_width
        self.hidden_layers = hidden_layers
        self.latent_dim = latent_dim
        self.branch_net = _make_mlp(
            branch_input_dim, hidden_width, hidden_layers, latent_dim
        )
        self.trunk_net = _make_mlp(
            trunk_input_dim, hidden_width, hidden_layers, latent_dim
        )
        self.output_head = nn.Linear(latent_dim, output_dim)
        _initialize_linear(self.output_head)

    def forward(self, branch: Tensor, trunk: Tensor) -> Tensor:
        if branch.ndim != 2 or branch.shape[1] != self.branch_input_dim:
            raise ValueError(
                f"Branch 输入形状必须为 (batch, {self.branch_input_dim})"
            )
        if trunk.ndim != 2 or trunk.shape[1] != self.trunk_input_dim:
            raise ValueError(f"Trunk 输入形状必须为 (queries, {self.trunk_input_dim})")
        branch_features = self.branch_net(branch)
        trunk_features = self.trunk_net(trunk)
        joint = branch_features[:, None, :] * trunk_features[None, :, :]
        return self.output_head(joint)


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
            _initialize_linear(module)
    return network


def _initialize_linear(module: nn.Linear) -> None:
    nn.init.xavier_uniform_(module.weight)
    nn.init.zeros_(module.bias)
