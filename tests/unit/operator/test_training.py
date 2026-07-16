import numpy as np
import torch

from surrogate_loop.operator.heat1d.dataset import NormalizationStats
from surrogate_loop.operator.heat1d.deeponet import build_deeponet
from surrogate_loop.operator.heat1d.training import predict_dataset, train_deeponet


def _tiny_training_spec(smoke_operator_spec):
    return smoke_operator_spec.model_copy(
        update={
            "model": smoke_operator_spec.model.model_copy(
                update={"hidden_width": 16, "hidden_layers": 2, "latent_dim": 8}
            ),
            "training": smoke_operator_spec.training.model_copy(
                update={
                    "max_epochs": 3,
                    "patience": 2,
                    "case_batch_size": 4,
                    "query_batch_size": 32,
                    "max_minutes": 1.0,
                }
            ),
        }
    )


class _TrainingOnlySplit:
    def __init__(self, train, validation) -> None:
        self.train = train
        self.validation = validation

    @property
    def test(self):
        raise AssertionError("训练器禁止访问测试集")


def test_training_returns_best_state_without_consuming_test_fields(
    smoke_operator_spec, small_heat_split
) -> None:
    spec = _tiny_training_spec(smoke_operator_spec)
    train = small_heat_split.train.subset(np.arange(8))
    validation = small_heat_split.validation.subset(np.arange(2))
    split = _TrainingOnlySplit(train, validation)
    normalization = NormalizationStats.fit(train)

    result = train_deeponet(spec, split, normalization, torch.device("cpu"))

    assert result.best_epoch >= 0
    assert 1 <= len(result.history) <= 3
    assert result.stop_reason in {"max_epochs", "early_stopping", "time_budget"}
    assert np.isfinite([record.validation_loss for record in result.history]).all()
    assert result.device == "cpu"
    assert all(tensor.device.type == "cpu" for tensor in result.state_dict.values())


def test_cpu_training_is_reproducible(smoke_operator_spec, small_heat_split) -> None:
    spec = _tiny_training_spec(smoke_operator_spec)
    train = small_heat_split.train.subset(np.arange(8))
    validation = small_heat_split.validation.subset(np.arange(2))
    split = _TrainingOnlySplit(train, validation)
    normalization = NormalizationStats.fit(train)

    first = train_deeponet(spec, split, normalization, torch.device("cpu"))
    second = train_deeponet(spec, split, normalization, torch.device("cpu"))

    assert first.best_epoch == second.best_epoch
    np.testing.assert_allclose(
        [record.train_loss for record in first.history],
        [record.train_loss for record in second.history],
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        [record.validation_loss for record in first.history],
        [record.validation_loss for record in second.history],
        rtol=0.0,
        atol=1e-10,
    )


def test_predict_dataset_restores_physical_field_shape(
    smoke_operator_spec, small_heat_split
) -> None:
    spec = _tiny_training_spec(smoke_operator_spec)
    train = small_heat_split.train.subset(np.arange(8))
    validation = small_heat_split.validation.subset(np.arange(2))
    normalization = NormalizationStats.fit(train)
    result = train_deeponet(
        spec,
        _TrainingOnlySplit(train, validation),
        normalization,
        torch.device("cpu"),
    )
    model = build_deeponet(spec.model)
    model.load_state_dict(result.state_dict)

    prediction = predict_dataset(
        model,
        validation,
        normalization,
        torch.device("cpu"),
        query_batch_size=17,
    )

    assert prediction.shape == validation.fields.shape
    assert np.isfinite(prediction).all()
