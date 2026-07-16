from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

import surrogate_loop.operator.elasticity2d.training as training_module
from surrogate_loop.operator.elasticity2d.config import (
    ElasticityRunSpec,
    load_elasticity_spec,
)
from surrogate_loop.operator.elasticity2d.deeponet import build_elasticity_deeponet
from surrogate_loop.operator.elasticity2d.problem import elasticity_features
from surrogate_loop.operator.elasticity2d.training import (
    TrainingFailure,
    predict_dataset,
    train_and_select,
    train_one_seed,
)
from surrogate_loop.operator.field_data import FieldDataset, FieldNormalization

ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class TrainingOnlyPartitions:
    train: FieldDataset
    validation: FieldDataset

    @property
    def sealed_test(self):
        raise AssertionError("训练器禁止访问封存测试集")


def _tiny_spec(*, seeds: tuple[int, ...] = (23, 11)) -> ElasticityRunSpec:
    smoke = load_elasticity_spec(ROOT / "examples/elasticity_2d_cantilever/smoke.json")
    return smoke.model_copy(
        update={
            "model": smoke.model.model_copy(
                update={"hidden_width": 12, "hidden_layers": 1, "latent_dim": 6}
            ),
            "training": smoke.training.model_copy(
                update={
                    "max_epochs": 2,
                    "patience": 2,
                    "case_batch_size": 3,
                    "query_batch_size": 5,
                    "min_delta": 0.0,
                    "max_minutes": 1.0,
                    "seeds": seeds,
                }
            ),
        }
    )


def _partitions() -> TrainingOnlyPartitions:
    x, y = np.meshgrid(np.linspace(0.0, 4.0, 4), np.linspace(0.0, 1.0, 3))
    coordinates = np.column_stack((x.ravel(), y.ravel()))

    def build_dataset(offset: int, count: int, role: str) -> FieldDataset:
        index = np.arange(offset, offset + count, dtype=np.float64)
        young_modulus = 1.2 + 0.15 * index
        poisson_ratio = 0.22 + 0.015 * (index % 6)
        load = 0.0025 + 0.0003 * (index % 8)
        angle = -0.7 + 0.18 * index
        center = 0.25 + 0.08 * (index % 6)
        width = 0.09 + 0.015 * (index % 5)
        parameters = np.column_stack(
            (young_modulus, poisson_ratio, load, angle, center, width)
        )
        clamp = coordinates[:, 0] / 4.0
        shape_x = (
            0.5
            + poisson_ratio[:, None]
            + 0.2 * np.cos(angle)[:, None] * coordinates[:, 1]
        )
        shape_y = (
            np.sin(angle)[:, None]
            + 0.3 * (coordinates[:, 1][None, :] - center[:, None])
            + width[:, None]
        )
        fields = np.stack((shape_x, shape_y), axis=-1)
        fields *= (load / young_modulus)[:, None, None] * clamp[None, :, None]
        return FieldDataset(
            sample_ids=np.array(
                [f"{role}-{int(value):05d}-000000000000" for value in index]
            ),
            parameters=parameters,
            coordinates=coordinates,
            fields=fields,
            diagnostics={},
        )

    return TrainingOnlyPartitions(
        train=build_dataset(0, 6, "train"),
        validation=build_dataset(10, 3, "validation"),
    )


def _normalization(partitions: TrainingOnlyPartitions) -> FieldNormalization:
    return FieldNormalization.fit(
        elasticity_features(partitions.train.parameters),
        partitions.train.coordinates,
        partitions.train.fields,
    )


def test_training_selects_by_validation_without_test_access() -> None:
    spec = _tiny_spec()
    partitions = _partitions()

    selected = train_and_select(
        spec,
        partitions,
        _normalization(partitions),
        torch.device("cpu"),
    )

    expected = min(
        selected.candidates, key=lambda candidate: (candidate.validation_loss, candidate.seed)
    )
    assert selected.selected_seed == expected.seed
    assert selected.selected is expected
    assert len(selected.candidates) == len(spec.training.seeds)
    assert all(candidate.seed in spec.training.seeds for candidate in selected.candidates)


def test_cpu_training_is_reproducible_and_keeps_best_state_on_cpu() -> None:
    spec = _tiny_spec(seeds=(17,))
    partitions = _partitions()
    normalization = _normalization(partitions)

    first = train_one_seed(spec, partitions, normalization, torch.device("cpu"), 17)
    second = train_one_seed(spec, partitions, normalization, torch.device("cpu"), 17)

    assert first.best_epoch == second.best_epoch
    assert first.validation_loss == pytest.approx(second.validation_loss, abs=1e-12)
    np.testing.assert_allclose(
        [record.train_loss for record in first.history],
        [record.train_loss for record in second.history],
        rtol=0.0,
        atol=1e-12,
    )
    assert all(tensor.device.type == "cpu" for tensor in first.state_dict.values())


def test_predict_dataset_restores_vector_field_shape_and_clamp() -> None:
    spec = _tiny_spec(seeds=(17,))
    partitions = _partitions()
    normalization = _normalization(partitions)
    result = train_one_seed(spec, partitions, normalization, torch.device("cpu"), 17)
    model = build_elasticity_deeponet(spec.model)
    model.load_state_dict(result.state_dict)

    prediction = predict_dataset(
        model,
        partitions.validation,
        normalization,
        torch.device("cpu"),
        query_batch_size=5,
    )

    assert prediction.shape == partitions.validation.fields.shape
    assert np.isfinite(prediction).all()
    clamp = partitions.validation.coordinates[:, 0] == 0.0
    np.testing.assert_allclose(prediction[:, clamp], 0.0, rtol=0.0, atol=0.0)


def test_non_finite_loss_preserves_diagnostic_checkpoint(monkeypatch) -> None:
    spec = _tiny_spec(seeds=(17,))
    partitions = _partitions()
    monkeypatch.setattr(
        training_module,
        "apply_elasticity_constraints",
        lambda raw, *_: torch.full_like(raw, float("nan")),
    )

    with pytest.raises(TrainingFailure) as captured:
        train_one_seed(
            spec,
            partitions,
            _normalization(partitions),
            torch.device("cpu"),
            17,
        )

    assert captured.value.reason == "non_finite_train_loss"
    assert captured.value.failure_epoch == 0
    assert captured.value.state_dict


def test_oom_preserves_diagnostic_checkpoint(monkeypatch) -> None:
    spec = _tiny_spec(seeds=(17,))
    partitions = _partitions()
    monkeypatch.setattr(
        training_module,
        "apply_elasticity_constraints",
        lambda *_: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("injected OOM")),
    )

    with pytest.raises(TrainingFailure) as captured:
        train_one_seed(
            spec,
            partitions,
            _normalization(partitions),
            torch.device("cpu"),
            17,
        )

    assert captured.value.reason == "cuda_oom"
    assert captured.value.state_dict
