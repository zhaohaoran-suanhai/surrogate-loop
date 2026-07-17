from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from surrogate_loop.operator.elasticity2d.config import load_elasticity_spec
from surrogate_loop.operator.elasticity2d.dataset import (
    generate_or_reuse_dataset,
    load_development_partitions,
)
from surrogate_loop.operator.elasticity2d.deeponet import (
    apply_elasticity_constraints,
    build_elasticity_deeponet,
)
from surrogate_loop.operator.elasticity2d.problem import elasticity_basis_features
from surrogate_loop.operator.elasticity2d.sampling import build_sample_plan
from surrogate_loop.operator.field_data import FieldNormalization

pytestmark = pytest.mark.skipif(
    os.environ.get("SURROGATE_LOOP_RUN_FENICSX_E2E") != "1",
    reason="set SURROGATE_LOOP_RUN_FENICSX_E2E=1 explicitly",
)

ROOT = Path(__file__).resolve().parents[2]


def test_real_fenicsx_protocol_reaches_one_training_step(tmp_path) -> None:
    canonical = load_elasticity_spec(
        ROOT / "examples/elasticity_2d_cantilever/smoke.json"
    )
    spec = canonical.model_copy(
        update={
            "mesh": canonical.mesh.model_copy(update={"nx": 8, "ny": 2}),
            "observation": canonical.observation.model_copy(update={"nx": 9, "ny": 3}),
            "sampling": canonical.sampling.model_copy(
                update={"train_cases": 2, "validation_cases": 1, "test_cases": 1}
            ),
            "model": canonical.model.model_copy(
                update={"hidden_width": 8, "hidden_layers": 1, "latent_dim": 4}
            ),
        }
    )
    plan = build_sample_plan(spec)

    files = generate_or_reuse_dataset(spec, plan, tmp_path / "run", ROOT)

    manifest = json.loads(files.manifest_path.read_text(encoding="utf-8"))
    assert manifest["software"]["dolfinx"].startswith("0.11.")
    assert manifest["solver"]["timing_scope"] == "assembly_solve_interpolation"
    assert all(
        sample["diagnostics"]["relative_residual"] <= spec.solver.max_relative_residual
        and sample["diagnostics"]["force_balance_error"]
        <= spec.solver.max_force_balance_error
        for sample in manifest["samples"]
    )
    partitions = load_development_partitions(files, plan)
    normalization = FieldNormalization.fit(
        elasticity_basis_features(partitions.train.parameters),
        partitions.train.coordinates,
        partitions.train.fields,
    )
    model = build_elasticity_deeponet(spec.model)
    features = torch.as_tensor(
        normalization.normalize_features(
            elasticity_basis_features(partitions.train.parameters)
        ).astype(np.float32)
    )
    coordinates = torch.as_tensor(
        normalization.normalize_coordinates(partitions.train.coordinates).astype(
            np.float32
        )
    )
    physical_parameters = torch.as_tensor(
        partitions.train.parameters.astype(np.float32)
    )
    physical_coordinates = torch.as_tensor(
        partitions.train.coordinates.astype(np.float32)
    )
    targets = torch.as_tensor(partitions.train.fields.astype(np.float32))

    prediction = apply_elasticity_constraints(
        model(features, coordinates), physical_parameters, physical_coordinates
    )
    loss = torch.mean((prediction - targets).square())
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
