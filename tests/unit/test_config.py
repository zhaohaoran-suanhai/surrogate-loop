import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from surrogate_loop.config import RunSpec, load_spec

ROOT = Path(__file__).resolve().parents[2]


def test_canonical_specs_load() -> None:
    full = load_spec(ROOT / "examples/forced_reaction_scalar/full.json")
    smoke = load_spec(ROOT / "examples/forced_reaction_scalar/smoke.json")

    assert full.sampling.train_cases == 120
    assert smoke.sampling.train_cases == 24
    assert smoke.mode == "smoke"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update({"unknown": True}),
        lambda data: data["problem"]["gamma"].update({"high": 1.2}),
        lambda data: data["models"].update({"candidates": ["gpr", "gpr"]}),
        lambda data: data["sampling"].update({"train_cases": 25}),
    ],
)
def test_invalid_smoke_specs_are_rejected(mutation) -> None:
    data = json.loads(
        (ROOT / "examples/forced_reaction_scalar/smoke.json").read_text(encoding="utf-8")
    )
    mutation(data)

    with pytest.raises(ValidationError):
        RunSpec.model_validate(data)
