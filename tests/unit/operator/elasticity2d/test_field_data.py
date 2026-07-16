from pathlib import Path

import numpy as np
import pytest

from surrogate_loop.operator.field_data import (
    FieldDataset,
    FieldNormalization,
    load_field_dataset,
    save_field_dataset,
)


def make_dataset() -> FieldDataset:
    return FieldDataset(
        sample_ids=np.array(["case-a", "case-b"]),
        parameters=np.array(
            [
                [1.0, 0.3, 0.005, 0.0, 0.5, 0.1],
                [2.0, 0.4, 0.008, 1.0, 0.6, 0.2],
            ]
        ),
        coordinates=np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 1.0]]),
        fields=np.arange(12, dtype=np.float64).reshape(2, 3, 2) / 1000.0,
        diagnostics={
            "relative_residual": np.array([1e-10, 2e-10]),
            "reaction": np.array([[-0.005, 0.0], [-0.004, -0.006]]),
        },
    )


def test_vector_field_round_trip_preserves_arrays_and_diagnostics(tmp_path: Path) -> None:
    dataset = make_dataset()

    digest = save_field_dataset(tmp_path / "field.npz", dataset)
    loaded = load_field_dataset(tmp_path / "field.npz", digest)

    assert len(digest) == 64
    np.testing.assert_array_equal(loaded.sample_ids, dataset.sample_ids)
    np.testing.assert_array_equal(loaded.parameters, dataset.parameters)
    np.testing.assert_array_equal(loaded.coordinates, dataset.coordinates)
    np.testing.assert_array_equal(loaded.fields, dataset.fields)
    np.testing.assert_array_equal(
        loaded.diagnostics["relative_residual"],
        dataset.diagnostics["relative_residual"],
    )
    np.testing.assert_array_equal(
        loaded.diagnostics["reaction"], dataset.diagnostics["reaction"]
    )


def test_hash_damage_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "field.npz"
    digest = save_field_dataset(path, make_dataset())
    with path.open("ab") as stream:
        stream.write(b"damage")

    with pytest.raises(RuntimeError, match="SHA-256"):
        load_field_dataset(path, digest)


def test_field_dataset_rejects_inconsistent_shape() -> None:
    with pytest.raises(ValueError, match="字段形状"):
        FieldDataset(
            sample_ids=np.array(["a", "b"]),
            parameters=np.ones((2, 6)),
            coordinates=np.ones((3, 2)),
            fields=np.ones((2, 4, 2)),
            diagnostics={"residual": np.ones(2)},
        )


def test_field_dataset_rejects_invalid_diagnostic_name() -> None:
    with pytest.raises(ValueError, match="诊断名称"):
        FieldDataset(
            sample_ids=np.array(["a"]),
            parameters=np.ones((1, 2)),
            coordinates=np.ones((3, 2)),
            fields=np.ones((1, 3, 1)),
            diagnostics={"bad/name": np.ones(1)},
        )


def test_subset_selects_complete_cases() -> None:
    dataset = make_dataset()

    selected = dataset.subset(np.array([1]))

    assert selected.sample_ids.tolist() == ["case-b"]
    assert selected.fields.shape == (1, 3, 2)
    np.testing.assert_array_equal(selected.coordinates, dataset.coordinates)
    np.testing.assert_array_equal(selected.diagnostics["reaction"], [[-0.004, -0.006]])


def test_normalization_uses_supplied_training_arrays_and_round_trips() -> None:
    dataset = make_dataset()
    features = np.array([[0.3, 1.0], [0.4, -1.0]])

    stats = FieldNormalization.fit(features, dataset.coordinates, dataset.fields)

    np.testing.assert_allclose(stats.feature_mean, features.mean(axis=0))
    normalized_features = stats.normalize_features(features)
    normalized_coordinates = stats.normalize_coordinates(dataset.coordinates)
    scaled_targets = stats.scale_targets(dataset.fields)
    np.testing.assert_allclose(normalized_features.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(normalized_coordinates.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(stats.unscale_targets(scaled_targets), dataset.fields)
    assert stats.target_rms.shape == (2,)
    assert np.all(stats.target_rms > 0.0)
