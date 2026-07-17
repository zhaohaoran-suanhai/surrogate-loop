import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from surrogate_loop.operator.elasticity2d.config import (
    ElasticityRunSpec,
    load_elasticity_spec,
)
from surrogate_loop.operator.elasticity2d.sampling import SamplePlan, build_sample_plan

ROOT = Path(__file__).resolve().parents[4]
EXAMPLES = ROOT / "examples/elasticity_2d_cantilever"


def test_canonical_elasticity_specs_load() -> None:
    calibration = load_elasticity_spec(EXAMPLES / "calibration.json")
    smoke = load_elasticity_spec(EXAMPLES / "smoke.json")
    full = load_elasticity_spec(EXAMPLES / "full.json")

    assert calibration.mode == "calibration"
    assert calibration.sampling.train_cases == 16
    assert (smoke.mesh.nx, smoke.mesh.ny, smoke.observation.nx) == (128, 32, 65)
    assert (full.mesh.nx, full.mesh.ny, full.mesh.degree) == (256, 64, 2)
    assert (full.observation.nx, full.observation.ny) == (129, 33)
    assert (
        full.sampling.train_cases,
        full.sampling.validation_cases,
        full.sampling.test_cases,
    ) == (512, 96, 128)
    assert full.training.seeds == (20260716, 20260717, 20260718)
    assert full.acceptance.max_p95_relative_l2 == 0.08


def test_canonical_specs_lock_directional_linear_v2() -> None:
    for name in ("calibration.json", "smoke.json", "full.json"):
        spec = load_elasticity_spec(EXAMPLES / name)
        assert spec.model.architecture == "directional_linear_v2"


def test_full_contract_rejects_legacy_vector_architecture() -> None:
    payload = json.loads((EXAMPLES / "full.json").read_text(encoding="utf-8"))
    payload["model"]["architecture"] = "legacy_vector_v1"

    with pytest.raises(ValidationError, match="architecture"):
        ElasticityRunSpec.model_validate(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"unknown": True}),
        lambda payload: payload["problem"]["young_modulus"].update({"high": 5.1}),
        lambda payload: payload["mesh"].update({"nx": 255}),
        lambda payload: payload["observation"].update({"ny": 32}),
        lambda payload: payload["sampling"].update({"test_cases": 127}),
        lambda payload: payload["training"].update({"seeds": [20260716]}),
        lambda payload: payload["acceptance"].update(
            {"max_p95_relative_l2": 0.081}
        ),
    ],
)
def test_full_contract_cannot_be_changed(mutation) -> None:
    payload = json.loads((EXAMPLES / "full.json").read_text(encoding="utf-8"))
    mutation(payload)

    with pytest.raises(ValidationError):
        ElasticityRunSpec.model_validate(payload)


def test_sample_plan_is_deterministic_and_roles_are_disjoint() -> None:
    spec = load_elasticity_spec(EXAMPLES / "smoke.json")

    first = build_sample_plan(spec)
    second = build_sample_plan(spec)

    np.testing.assert_array_equal(first.parameters, second.parameters)
    np.testing.assert_array_equal(first.sample_ids, second.sample_ids)
    np.testing.assert_array_equal(first.roles, second.roles)
    assert first.parameters.shape == (144, 6)
    assert len(set(first.sample_ids.tolist())) == 144
    assert np.sum(first.roles == "train") == 96
    assert np.sum(first.roles == "validation") == 24
    assert np.sum(first.roles == "development_test") == 24


def test_calibration_sample_plan_preserves_full_role_name() -> None:
    spec = load_elasticity_spec(EXAMPLES / "calibration.json")

    plan = build_sample_plan(spec)

    assert plan.roles.dtype.itemsize >= len("calibration") * 4
    assert np.all(plan.roles == "calibration")
    assert all(sample_id.startswith("calibration-") for sample_id in plan.sample_ids)


def test_sample_plan_respects_parameter_domain() -> None:
    spec = load_elasticity_spec(EXAMPLES / "full.json")
    plan = build_sample_plan(spec)
    lower = np.array([1.0, 0.2, 0.002, -np.pi, 0.2, 0.08])
    upper = np.array([5.0, 0.45, 0.01, np.pi, 0.8, 0.2])

    assert np.all(plan.parameters >= lower)
    assert np.all(plan.parameters <= upper)
    assert np.all(plan.parameters[:, 2] / plan.parameters[:, 0] <= 1e-2)
    assert np.sum(plan.roles == "sealed_test") == 128


def test_sample_plan_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="样本 ID"):
        SamplePlan(
            sample_ids=np.array(["duplicate", "duplicate"]),
            parameters=np.ones((2, 6)),
            roles=np.array(["train", "validation"]),
        )
