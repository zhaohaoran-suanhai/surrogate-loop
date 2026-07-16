from __future__ import annotations

import pytest
import torch
from torch import nn

from surrogate_loop.operator.elasticity2d.config import VectorDeepONetSpec
from surrogate_loop.operator.elasticity2d.deeponet import (
    apply_elasticity_constraints,
    build_elasticity_deeponet,
)
from surrogate_loop.operator.vector_deeponet import VectorDeepONet


def test_vector_deeponet_cartesian_output_and_gradients() -> None:
    model = VectorDeepONet(5, 2, 2, hidden_width=16, hidden_layers=2, latent_dim=8)

    output = model(torch.randn(4, 5), torch.randn(11, 2))
    output.square().mean().backward()

    assert output.shape == (4, 11, 2)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_vector_deeponet_uses_xavier_initialized_linear_layers() -> None:
    torch.manual_seed(17)
    model = VectorDeepONet(5, 2, 2, hidden_width=16, hidden_layers=2, latent_dim=8)

    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]

    assert linear_layers
    assert all(torch.count_nonzero(layer.weight) > 0 for layer in linear_layers)
    assert all(torch.count_nonzero(layer.bias) == 0 for layer in linear_layers)


@pytest.mark.parametrize(
    "dimensions",
    [
        (0, 2, 2, 16, 2, 8),
        (5, 0, 2, 16, 2, 8),
        (5, 2, 0, 16, 2, 8),
        (5, 2, 2, 0, 2, 8),
        (5, 2, 2, 16, 0, 8),
        (5, 2, 2, 16, 2, 0),
    ],
)
def test_vector_deeponet_rejects_nonpositive_dimensions(
    dimensions: tuple[int, int, int, int, int, int],
) -> None:
    with pytest.raises(ValueError, match="必须为正数"):
        VectorDeepONet(*dimensions)


def test_elasticity_builder_uses_structured_spec() -> None:
    spec = VectorDeepONetSpec(hidden_width=24, hidden_layers=3, latent_dim=10)

    model = build_elasticity_deeponet(spec)

    assert model.branch_input_dim == 5
    assert model.trunk_input_dim == 2
    assert model.output_dim == 2
    assert model.hidden_width == 24
    assert model.hidden_layers == 3
    assert model.latent_dim == 10


def test_elasticity_constraints_enforce_clamp_and_p_over_e() -> None:
    raw = torch.ones(1, 2, 2)
    coordinates = torch.tensor([[0.0, 0.5], [4.0, 0.5]])
    base = torch.tensor([[2.0, 0.3, 0.005, 0.0, 0.5, 0.1]])
    doubled_load = base.clone()
    doubled_load[:, 2] *= 2.0
    doubled_modulus = base.clone()
    doubled_modulus[:, 0] *= 2.0

    base_output = apply_elasticity_constraints(raw, base, coordinates)

    torch.testing.assert_close(base_output[:, 0], torch.zeros(1, 2))
    torch.testing.assert_close(
        apply_elasticity_constraints(raw, doubled_load, coordinates)[:, 1],
        2.0 * base_output[:, 1],
    )
    torch.testing.assert_close(
        apply_elasticity_constraints(raw, doubled_modulus, coordinates)[:, 1],
        0.5 * base_output[:, 1],
    )


@pytest.mark.parametrize(
    ("branch", "trunk", "message"),
    [
        (torch.randn(5), torch.randn(3, 2), "Branch"),
        (torch.randn(3, 4), torch.randn(3, 2), "Branch"),
        (torch.randn(3, 5), torch.randn(2), "Trunk"),
        (torch.randn(3, 5), torch.randn(3, 3), "Trunk"),
    ],
)
def test_vector_deeponet_rejects_invalid_inputs(branch, trunk, message: str) -> None:
    model = VectorDeepONet(5, 2, 2, hidden_width=8, hidden_layers=1, latent_dim=4)

    with pytest.raises(ValueError, match=message):
        model(branch, trunk)


@pytest.mark.parametrize(
    ("raw", "parameters", "coordinates", "message"),
    [
        (torch.ones(2, 2), torch.ones(1, 6), torch.ones(2, 2), "输出"),
        (torch.ones(1, 2, 2), torch.ones(1, 5), torch.ones(2, 2), "参数"),
        (torch.ones(1, 2, 2), torch.ones(1, 6), torch.ones(2, 3), "坐标"),
        (torch.ones(2, 2, 2), torch.ones(1, 6), torch.ones(2, 2), "形状"),
    ],
)
def test_elasticity_constraints_reject_invalid_shapes(
    raw: torch.Tensor,
    parameters: torch.Tensor,
    coordinates: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        apply_elasticity_constraints(raw, parameters, coordinates)
