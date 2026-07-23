from pathlib import Path

import numpy as np
import pytest

from surrogate_loop.operator.cavity2d.model import (
    fit_candidate,
    load_cavity_model,
    save_cavity_model,
)


def synthetic_fields(reynolds: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    values = []
    for re_value in reynolds:
        scale = np.log10(re_value)
        u = scale * x * (1.0 - y)
        v = scale * y * (1.0 - x)
        pressure = scale * (x - x.mean())
        values.append(np.column_stack((u, v, pressure)))
    return np.stack(values)


def test_model_predicts_and_round_trips_without_pickle(tmp_path: Path) -> None:
    reynolds = np.array([10.0, 30.0, 100.0, 200.0, 400.0])
    coordinates = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    fields = synthetic_fields(reynolds, coordinates)
    model = fit_candidate(
        reynolds,
        fields,
        energy_threshold=0.9999,
        kernel="cubic",
        smoothing=0.0,
    )

    expected = model.predict(np.array([50.0]))
    identity = {
        "problem_id": "fluent_lid_driven_cavity_steady_v1",
        "mesh_sha256": "a" * 64,
        "coordinates_sha256": "b" * 64,
    }
    save_cavity_model(tmp_path, model, **identity)
    restored = load_cavity_model(tmp_path, **identity)

    assert np.allclose(restored.predict(np.array([50.0])), expected)
    assert np.allclose(expected[:, :, 2].mean(axis=1), 0.0, atol=1e-14)
    assert not list(tmp_path.glob("*.pkl"))
    assert not list(tmp_path.glob("*.joblib"))


def test_model_load_rejects_wrong_problem_or_mesh_identity(tmp_path: Path) -> None:
    reynolds = np.array([10.0, 100.0, 400.0])
    coordinates = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    model = fit_candidate(
        reynolds,
        synthetic_fields(reynolds, coordinates),
        energy_threshold=0.999,
        kernel="cubic",
        smoothing=0.0,
    )
    save_cavity_model(
        tmp_path,
        model,
        problem_id="fluent_lid_driven_cavity_steady_v1",
        mesh_sha256="a" * 64,
        coordinates_sha256="b" * 64,
    )

    with pytest.raises(RuntimeError, match="identity"):
        load_cavity_model(
            tmp_path,
            problem_id="different",
            mesh_sha256="a" * 64,
            coordinates_sha256="b" * 64,
        )


def test_pod_components_are_separate_and_capped_by_training_count() -> None:
    reynolds = np.geomspace(10.0, 400.0, 5)
    coordinates = np.column_stack(
        (np.linspace(0.0, 1.0, 10), np.linspace(1.0, 0.0, 10))
    )
    model = fit_candidate(
        reynolds,
        synthetic_fields(reynolds, coordinates),
        energy_threshold=0.9999,
        kernel="thin_plate_spline",
        smoothing=1e-10,
    )

    assert model.velocity.components.shape[0] <= 4
    assert model.pressure.components.shape[0] <= 4
    assert model.velocity.components.shape[1] == 20
    assert model.pressure.components.shape[1] == 10


@pytest.mark.parametrize("kernel", ["cubic", "thin_plate_spline", "multiquadric"])
def test_all_fixed_kernels_predict_finite_values(kernel: str) -> None:
    reynolds = np.geomspace(10.0, 400.0, 6)
    coordinates = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    model = fit_candidate(
        reynolds,
        synthetic_fields(reynolds, coordinates),
        energy_threshold=0.999,
        kernel=kernel,
        smoothing=1e-8,
    )

    assert np.isfinite(model.predict(np.array([25.0, 250.0]))).all()
