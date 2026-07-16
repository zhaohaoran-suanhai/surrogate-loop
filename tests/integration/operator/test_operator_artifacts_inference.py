import json

import numpy as np
import pytest
import torch

from surrogate_loop.cli import main
from surrogate_loop.operator.artifacts import save_operator_run, write_failed_run
from surrogate_loop.operator.heat1d.dataset import (
    HeatDataset,
    HeatDatasetSplit,
    NormalizationStats,
)
from surrogate_loop.operator.heat1d.deeponet import build_deeponet
from surrogate_loop.operator.heat1d.evaluation import compute_field_metrics
from surrogate_loop.operator.heat1d.pod_gpr import PodGprBaseline
from surrogate_loop.operator.heat1d.training import (
    TrainingFailure,
    TrainingRecord,
    TrainingResult,
    predict_dataset,
)
from surrogate_loop.operator.inference import (
    load_operator_bundle,
    predict_field,
    predict_point,
    verify_operator_run,
)


@pytest.fixture
def operator_run_inputs(smoke_operator_spec, small_heat_split):
    spec = smoke_operator_spec.model_copy(
        update={
            "model": smoke_operator_spec.model.model_copy(
                update={"hidden_width": 16, "hidden_layers": 2, "latent_dim": 8}
            ),
            "acceptance": smoke_operator_spec.acceptance.model_copy(
                update={
                    "max_median_relative_l2": 100.0,
                    "max_p95_relative_l2": 100.0,
                    "max_worst_relative_l2": 100.0,
                    "max_initial_relative_l2": 100.0,
                    "max_boundary_absolute_error": 100.0,
                }
            ),
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
    test_prediction = predict_dataset(
        model,
        split.test,
        normalization,
        torch.device("cpu"),
        query_batch_size=32,
    )
    test_metrics = compute_field_metrics(
        split.test.fields,
        test_prediction,
        normalization.target_std,
    )
    parts = (split.train, split.validation, split.test)
    dataset = HeatDataset(
        parameters=np.concatenate([part.parameters for part in parts], axis=0),
        x=split.train.x,
        t=split.train.t,
        fields=np.concatenate([part.fields for part in parts], axis=0),
        solver_relative_l2=np.concatenate(
            [part.solver_relative_l2 for part in parts], axis=0
        ),
    )
    return {
        "spec": spec,
        "request_text": "测试可信 DeepONet 产物",
        "dataset": dataset,
        "split": split,
        "normalization": normalization,
        "baseline": baseline,
        "pod_metrics": pod_metrics,
        "training": training,
        "test_metrics": test_metrics,
        "test_prediction": test_prediction,
        "status": "accepted",
        "runtime": {"device": "cpu", "torch": torch.__version__},
    }


@pytest.fixture
def operator_run_dir(tmp_path, operator_run_inputs):
    run_dir = tmp_path / "operator-run"
    run_dir.mkdir()
    save_operator_run(run_dir=run_dir, **operator_run_inputs)
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


def test_save_rejects_metrics_not_produced_by_checkpoint(
    tmp_path, operator_run_inputs
) -> None:
    inputs = dict(operator_run_inputs)
    split = inputs["split"]
    normalization = inputs["normalization"]
    mismatched_prediction = split.test.fields.copy()
    inputs["test_prediction"] = mismatched_prediction
    inputs["test_metrics"] = compute_field_metrics(
        split.test.fields,
        mismatched_prediction,
        normalization.target_std,
    )

    with pytest.raises(ValueError, match="检查点"):
        save_operator_run(run_dir=tmp_path, **inputs)


def test_save_rejects_dataset_split_identity_mismatch(
    tmp_path, operator_run_inputs
) -> None:
    inputs = dict(operator_run_inputs)
    inputs["dataset"] = inputs["split"].train

    with pytest.raises(ValueError, match="完整数据集"):
        save_operator_run(run_dir=tmp_path, **inputs)


def test_checkpoint_hash_damage_is_rejected(operator_run_dir) -> None:
    checkpoint = operator_run_dir / "deeponet_state.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"damage")

    with pytest.raises(RuntimeError, match="哈希校验失败"):
        load_operator_bundle(operator_run_dir, "cpu")


def test_manifest_requires_the_exact_hashed_file_set(operator_run_dir) -> None:
    manifest_path = operator_run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"].pop("dataset.npz")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="必需文件集合"):
        verify_operator_run(operator_run_dir)


def test_manifest_status_is_recomputed_from_hashed_metrics(operator_run_dir) -> None:
    manifest_path = operator_run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "rejected"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="状态与已验证指标不一致"):
        verify_operator_run(operator_run_dir)


def test_report_input_damage_is_rejected(operator_run_dir) -> None:
    metrics = operator_run_dir / "pod_metrics.json"
    metrics.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="哈希校验失败"):
        verify_operator_run(operator_run_dir)


def test_failed_training_writes_diagnostic_state(tmp_path) -> None:
    failure = TrainingFailure(
        "injected failure",
        reason="non_finite_train_loss",
        failure_epoch=3,
        state_dict={"weight": torch.ones(1)},
        history=(TrainingRecord(2, 0.2, 0.3, 0.001),),
    )

    write_failed_run(tmp_path, tmp_path / "spec.json", failure)

    assert (tmp_path / "failed_deeponet_state.pt").is_file()
    payload = json.loads(
        (tmp_path / "training_failure.json").read_text(encoding="utf-8")
    )
    assert payload["reason"] == "non_finite_train_loss"
    assert payload["failure_epoch"] == 3


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


def test_cli_rejects_invalid_parameters_before_loading_weights(
    operator_run_dir, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: pytest.fail("无效请求不应加载权重"),
    )

    exit_code = main(
        [
            "operator",
            "predict",
            "--run-dir",
            str(operator_run_dir),
            "--alpha",
            "0.21",
            "--a",
            "1.0",
            "--b",
            "0.0",
            "--x",
            "0.5",
            "--t",
            "0.25",
        ]
    )

    assert exit_code == 2
    assert "训练参数域" in capsys.readouterr().err


def test_out_of_domain_coordinates_are_rejected(operator_run_dir) -> None:
    bundle = load_operator_bundle(operator_run_dir, "cpu")

    with pytest.raises(ValueError, match="查询域"):
        predict_point(bundle, 0.1, 1.0, 0.0, x=-0.1, t=0.25)


def test_duplicate_coordinates_are_rejected(operator_run_dir) -> None:
    bundle = load_operator_bundle(operator_run_dir, "cpu")

    with pytest.raises(ValueError, match="严格递增"):
        predict_field(
            bundle,
            0.1,
            1.0,
            0.0,
            x=np.array([0.0, 0.5, 0.5, 1.0]),
            t=np.array([0.0, 1.0]),
        )


def test_rejected_manifest_cannot_be_loaded(operator_run_dir) -> None:
    manifest_path = operator_run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "rejected"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="状态与已验证指标不一致"):
        load_operator_bundle(operator_run_dir, "cpu")
