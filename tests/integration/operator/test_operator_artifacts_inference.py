import json

import numpy as np
import pytest
import torch

from surrogate_loop.operator.artifacts import save_operator_run
from surrogate_loop.operator.heat1d.dataset import HeatDatasetSplit, NormalizationStats
from surrogate_loop.operator.heat1d.deeponet import build_deeponet
from surrogate_loop.operator.heat1d.evaluation import compute_field_metrics
from surrogate_loop.operator.heat1d.pod_gpr import PodGprBaseline
from surrogate_loop.operator.heat1d.training import TrainingRecord, TrainingResult
from surrogate_loop.operator.inference import (
    load_operator_bundle,
    predict_field,
    predict_point,
)


@pytest.fixture
def operator_run_dir(tmp_path, smoke_operator_spec, small_heat_split):
    spec = smoke_operator_spec.model_copy(
        update={
            "model": smoke_operator_spec.model.model_copy(
                update={"hidden_width": 16, "hidden_layers": 2, "latent_dim": 8}
            )
        }
    )
    split = HeatDatasetSplit(
        train=small_heat_split.train.subset(np.arange(8)),
        validation=small_heat_split.validation.subset(np.arange(2)),
        test=small_heat_split.test.subset(np.arange(2)),
    )
    normalization = NormalizationStats.fit(split.train)
    model = build_deeponet(spec.model)
    training = TrainingResult(
        state_dict={
            name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
        },
        history=(TrainingRecord(0, 1.0, 0.9, 0.001),),
        best_epoch=0,
        stop_reason="max_epochs",
        device="cpu",
        elapsed_seconds=0.1,
        peak_cuda_memory_mb=0.0,
    )
    baseline = PodGprBaseline(0.999, 4, spec.sampling.seed).fit(
        split.train.parameters, split.train.fields
    )
    pod_prediction = baseline.predict(split.test.parameters)
    pod_metrics = compute_field_metrics(
        split.test.fields,
        pod_prediction,
        normalization.target_std,
    )
    test_prediction = split.test.fields.copy()
    test_metrics = compute_field_metrics(
        split.test.fields,
        test_prediction,
        normalization.target_std,
    )
    run_dir = tmp_path / "operator-run"
    run_dir.mkdir()
    save_operator_run(
        run_dir=run_dir,
        spec=spec,
        request_text="测试可信 DeepONet 产物",
        dataset=split.train,
        split=split,
        normalization=normalization,
        baseline=baseline,
        pod_metrics=pod_metrics,
        training=training,
        test_metrics=test_metrics,
        test_prediction=test_prediction,
        status="accepted",
        runtime={"device": "cpu", "torch": torch.__version__},
    )
    return run_dir


def test_saved_bundle_reloads_and_predicts(operator_run_dir) -> None:
    bundle = load_operator_bundle(operator_run_dir, "cpu")

    point = predict_point(bundle, 0.1, 1.0, 0.0, x=0.5, t=0.25)
    field = predict_field(
        bundle,
        0.1,
        1.0,
        0.0,
        x=np.linspace(0.0, 1.0, 9),
        t=np.linspace(0.0, 1.0, 7),
    )

    assert np.isfinite(point)
    assert field.shape == (7, 9)
    assert np.isfinite(field).all()


def test_checkpoint_hash_damage_is_rejected(operator_run_dir) -> None:
    checkpoint = operator_run_dir / "deeponet_state.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"damage")

    with pytest.raises(RuntimeError, match="哈希校验失败"):
        load_operator_bundle(operator_run_dir, "cpu")


@pytest.mark.parametrize(
    ("alpha", "amplitude_1", "amplitude_2"),
    [
        (0.21, 1.0, 0.0),
        (0.1, 1.21, 0.0),
        (0.1, 1.0, -0.31),
        (np.nan, 1.0, 0.0),
    ],
)
def test_out_of_domain_parameters_are_rejected(
    operator_run_dir, alpha, amplitude_1, amplitude_2
) -> None:
    bundle = load_operator_bundle(operator_run_dir, "cpu")

    with pytest.raises(ValueError, match="训练参数域"):
        predict_point(bundle, alpha, amplitude_1, amplitude_2, x=0.5, t=0.25)


def test_out_of_domain_coordinates_are_rejected(operator_run_dir) -> None:
    bundle = load_operator_bundle(operator_run_dir, "cpu")

    with pytest.raises(ValueError, match="查询域"):
        predict_point(bundle, 0.1, 1.0, 0.0, x=-0.1, t=0.25)


def test_rejected_manifest_cannot_be_loaded(operator_run_dir) -> None:
    manifest_path = operator_run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "rejected"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="未通过验收"):
        load_operator_bundle(operator_run_dir, "cpu")
