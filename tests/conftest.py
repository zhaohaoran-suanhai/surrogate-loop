from pathlib import Path

import pytest


@pytest.fixture
def smoke_spec_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples/forced_reaction_scalar/smoke.json"


@pytest.fixture(scope="session")
def smoke_operator_spec():
    from surrogate_loop.operator.config import load_operator_spec

    root = Path(__file__).resolve().parents[1]
    return load_operator_spec(root / "examples/heat_1d_operator/smoke.json")


@pytest.fixture(scope="session")
def small_heat_split(smoke_operator_spec):
    from surrogate_loop.operator.heat1d.dataset import generate_dataset, split_dataset

    dataset = generate_dataset(smoke_operator_spec)
    return split_dataset(dataset, smoke_operator_spec.sampling)
