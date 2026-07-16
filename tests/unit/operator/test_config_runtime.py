import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from surrogate_loop.operator.config import OperatorRunSpec, load_operator_spec
from surrogate_loop.operator.runtime import resolve_device, runtime_summary, seed_everything

ROOT = Path(__file__).resolve().parents[3]


def test_canonical_operator_specs_load() -> None:
    smoke = load_operator_spec(ROOT / "examples/heat_1d_operator/smoke.json")
    full = load_operator_spec(ROOT / "examples/heat_1d_operator/full.json")

    assert smoke.problem.template == "heat_1d_operator_v1"
    assert smoke.grid.nx == 65
    assert smoke.sampling.train_cases == 64
    assert smoke.training.max_epochs == 1500
    assert smoke.training.patience == 120
    assert full.sampling.train_cases == 512
    assert full.training.max_epochs == 600
    assert full.runtime.device == "auto"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"unknown": True}),
        lambda payload: payload["problem"]["alpha"].update({"high": 0.21}),
        lambda payload: payload["grid"].update({"nx": 66}),
        lambda payload: payload["sampling"].update({"train_cases": 65}),
        lambda payload: payload["training"].update({"max_epochs": 1501}),
        lambda payload: payload["runtime"].update({"dtype": "float64"}),
    ],
)
def test_invalid_smoke_specs_are_rejected(mutation) -> None:
    payload = json.loads(
        (ROOT / "examples/heat_1d_operator/smoke.json").read_text(encoding="utf-8")
    )
    mutation(payload)

    with pytest.raises(ValidationError):
        OperatorRunSpec.model_validate(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["solver_acceptance"].update(
            {"max_p95_relative_l2": 0.01}
        ),
        lambda payload: payload["pod"].update({"energy_threshold": 0.9}),
        lambda payload: payload["model"].update({"hidden_width": 64}),
        lambda payload: payload["training"].update({"learning_rate": 0.01}),
        lambda payload: payload["acceptance"].update(
            {
                "max_median_relative_l2": 0.2,
                "max_p95_relative_l2": 0.3,
                "max_worst_relative_l2": 0.4,
                "max_initial_relative_l2": 0.2,
                "max_boundary_absolute_error": 0.1,
            }
        ),
    ],
)
def test_full_scientific_contract_cannot_be_relaxed(mutation) -> None:
    payload = json.loads(
        (ROOT / "examples/heat_1d_operator/full.json").read_text(encoding="utf-8")
    )
    mutation(payload)

    with pytest.raises(ValidationError, match="full"):
        OperatorRunSpec.model_validate(payload)


def test_explicit_cpu_device_and_seed_are_reproducible() -> None:
    torch = pytest.importorskip("torch")

    assert resolve_device("cpu").type == "cpu"
    seed_everything(17)
    first = torch.rand(4)
    seed_everything(17)
    second = torch.rand(4)

    torch.testing.assert_close(first, second)


def test_explicit_cuda_requires_available_cuda(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="配置要求 CUDA"):
        resolve_device("cuda")


def test_auto_device_records_cuda_initialization_fallback(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("driver init")),
    )

    with pytest.warns(RuntimeWarning, match="回退 CPU"):
        device = resolve_device("auto")

    assert device.type == "cpu"
    assert "driver init" in str(runtime_summary(device)["device_fallback_reason"])
