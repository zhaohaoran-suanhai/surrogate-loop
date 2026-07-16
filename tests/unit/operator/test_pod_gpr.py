import numpy as np
import pytest

from surrogate_loop.operator.heat1d.pod_gpr import PodGprBaseline


def test_pod_gpr_fits_and_reconstructs_field_shape(small_heat_split) -> None:
    train = small_heat_split.train.subset(np.arange(16))
    validation = small_heat_split.validation.subset(np.arange(4))
    model = PodGprBaseline(energy_threshold=0.999, max_components=8, seed=7)

    fitted = model.fit(train.parameters, train.fields)
    prediction = fitted.predict(validation.parameters)

    assert fitted is model
    assert prediction.shape == validation.fields.shape
    assert np.isfinite(prediction).all()
    assert 1 <= model.summary()["components"] <= 8
    assert 0.999 <= model.summary()["explained_energy"] <= 1.0


def test_pod_gpr_nearly_reconstructs_its_training_fields(small_heat_split) -> None:
    train = small_heat_split.train.subset(np.arange(16))
    model = PodGprBaseline(energy_threshold=0.999, max_components=8, seed=7)

    prediction = model.fit(train.parameters, train.fields).predict(train.parameters)
    relative_l2 = np.linalg.norm(prediction - train.fields) / np.linalg.norm(train.fields)

    assert relative_l2 < 0.05


def test_pod_gpr_rejects_prediction_before_fit() -> None:
    model = PodGprBaseline(energy_threshold=0.999, max_components=8, seed=7)

    with pytest.raises(RuntimeError, match="尚未拟合"):
        model.predict(np.zeros((1, 3)))


def test_pod_gpr_rejects_invalid_parameter_shape(small_heat_split) -> None:
    train = small_heat_split.train.subset(np.arange(8))
    model = PodGprBaseline(energy_threshold=0.999, max_components=4, seed=7)
    model.fit(train.parameters, train.fields)

    with pytest.raises(ValueError, match="参数形状"):
        model.predict(np.zeros((2, 2)))
