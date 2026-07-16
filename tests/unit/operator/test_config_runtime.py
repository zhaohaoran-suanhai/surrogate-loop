import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from surrogate_loop.operator.config import OperatorRunSpec, load_operator_spec
from surrogate_loop.operator.runtime import resolve_device, seed_everything

ROOT = Path(__file__).resolve().parents[3]


def test_canonical_operator_specs_load() -> None:
    smoke = load_operator_spec(ROOT / "examples/heat_1d_operator/smoke.json")
    full = load_operator_spec(ROOT / "examples/heat_1d_operator/full.json")

    assert smoke.problem.template == "heat_1d_operator_v1"
    assert smoke.grid.nx == 65
    assert smoke.sampling.train_cases == 64
    assert full.sampling.train_cases == 512
    assert full.runtime.device == "auto"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"unknown": True}),
        lambda payload: payload["problem"]["alpha"].update({"high": 0.21}),
        lambda payload: payload["grid"].update({"nx": 66}),
        lambda payload: payload["sampling"].update({"train_cases": 65}),
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
