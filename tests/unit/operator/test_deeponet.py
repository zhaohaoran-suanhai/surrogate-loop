import pytest
import torch

from surrogate_loop.operator.config import DeepONetSpec
from surrogate_loop.operator.heat1d.deeponet import DeepONet, build_deeponet


def test_deeponet_cartesian_forward_and_gradients() -> None:
    model = DeepONet(3, 2, hidden_width=16, hidden_layers=2, latent_dim=8)
    branch = torch.randn(4, 3)
    trunk = torch.randn(11, 2)

    output = model(branch, trunk)
    output.square().mean().backward()

    assert output.shape == (4, 11)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_state_dict_round_trip_preserves_predictions() -> None:
    torch.manual_seed(7)
    original = DeepONet(3, 2, 16, 2, 8)
    branch = torch.randn(2, 3)
    trunk = torch.randn(5, 2)
    expected = original(branch, trunk).detach()
    restored = DeepONet(3, 2, 16, 2, 8)

    restored.load_state_dict(original.state_dict())

    torch.testing.assert_close(restored(branch, trunk), expected)


def test_builder_uses_structured_model_spec() -> None:
    spec = DeepONetSpec(hidden_width=32, hidden_layers=3, latent_dim=12)

    model = build_deeponet(spec)

    assert model.branch_input_dim == 3
    assert model.trunk_input_dim == 2
    assert model.latent_dim == 12


@pytest.mark.parametrize(
    ("branch", "trunk", "message"),
    [
        (torch.randn(3), torch.randn(5, 2), "Branch"),
        (torch.randn(4, 2), torch.randn(5, 2), "Branch"),
        (torch.randn(4, 3), torch.randn(2), "Trunk"),
        (torch.randn(4, 3), torch.randn(5, 3), "Trunk"),
    ],
)
def test_deeponet_rejects_invalid_input_shapes(branch, trunk, message) -> None:
    model = DeepONet(3, 2, 16, 2, 8)

    with pytest.raises(ValueError, match=message):
        model(branch, trunk)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_deeponet_forward_and_backward_on_cuda() -> None:
    model = DeepONet(3, 2, 16, 2, 8).to("cuda")
    branch = torch.randn(8, 3, device="cuda")
    trunk = torch.randn(32, 2, device="cuda")

    loss = model(branch, trunk).square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
