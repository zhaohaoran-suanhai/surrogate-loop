from pathlib import Path

import pytest


@pytest.fixture
def smoke_spec_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples/forced_reaction_scalar/smoke.json"
