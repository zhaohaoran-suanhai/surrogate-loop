from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from surrogate_loop.cli import main
from surrogate_loop.operator.elasticity2d.config import load_elasticity_spec
from surrogate_loop.operator.elasticity2d.deeponet import build_elasticity_deeponet
from surrogate_loop.operator.elasticity2d.inference import (
    ElasticityBundle,
    predict_elasticity_points,
    validate_elasticity_request,
)
from surrogate_loop.operator.elasticity2d.problem import elasticity_features
from surrogate_loop.operator.field_data import FieldNormalization

ROOT = Path(__file__).resolve().parents[3]


def test_invalid_parameter_is_rejected_before_torch_load(tmp_path, monkeypatch) -> None:
    run_dir = _minimal_frozen_metadata(tmp_path)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: pytest.fail("域外请求不得加载权重"),
    )

    code = main(
        [
            "elasticity2d",
            "predict",
            "--run-dir",
            str(run_dir),
            "--e",
            "6",
            "--nu",
            ".3",
            "--p",
            ".005",
            "--theta",
            "0",
            "--y0",
            ".5",
            "--w",
            ".1",
            "--x",
            "2",
            "--y",
            ".5",
        ]
    )

    assert code == 2


def test_elasticity_cli_help_lists_fixed_commands(capsys) -> None:
    with pytest.raises(SystemExit) as captured:
        main(["elasticity2d", "--help"])

    assert captured.value.code == 0
    output = capsys.readouterr().out
    for command in ("doctor", "validate", "calibrate", "run", "report", "predict"):
        assert command in output


def test_point_inference_enforces_domain_and_vector_shape() -> None:
    spec = load_elasticity_spec(ROOT / "examples/elasticity_2d_cantilever/full.json")
    parameters = np.array([[2.0, 0.3, 0.004, 0.0, 0.5, 0.1]])
    coordinates = np.array([[0.0, 0.5], [4.0, 0.5]])
    features = elasticity_features(
        np.array(
            [
                [2.0, 0.25, 0.003, -0.2, 0.4, 0.09],
                [3.0, 0.35, 0.005, 0.3, 0.6, 0.15],
            ]
        )
    )
    normalization = FieldNormalization.fit(
        features, coordinates, np.ones((2, 2, 2))
    )
    model = build_elasticity_deeponet(spec.model)
    for model_parameter in model.parameters():
        model_parameter.data.zero_()
    model.output_head.bias.data.fill_(1.0)
    bundle = ElasticityBundle(
        spec, model, normalization, object(), torch.device("cpu")
    )

    prediction = predict_elasticity_points(bundle, parameters, coordinates)

    assert prediction.shape == (2, 2)
    np.testing.assert_allclose(prediction[0], 0.0, atol=0.0)
    np.testing.assert_allclose(prediction[1], 0.002, rtol=1e-6)
    with pytest.raises(ValueError, match="区域"):
        validate_elasticity_request(spec, parameters, np.array([[4.01, 0.5]]))
    with pytest.raises(ValueError, match="1000000"):
        validate_elasticity_request(spec, parameters, nx=1001, ny=1000)


def _minimal_frozen_metadata(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec_bytes = (ROOT / "examples/elasticity_2d_cantilever/full.json").read_bytes()
    (run_dir / "spec.json").write_bytes(spec_bytes)
    digest = hashlib.sha256(spec_bytes).hexdigest()
    (run_dir / "freeze_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "problem": "elasticity_2d_cantilever_v1",
                "mode": "full",
                "selected_seed": 20260716,
                "development_sha256": "0" * 64,
                "sealed_test_sha256": "1" * 64,
                "files": {"spec.json": digest},
            }
        ),
        encoding="utf-8",
    )
    return run_dir
