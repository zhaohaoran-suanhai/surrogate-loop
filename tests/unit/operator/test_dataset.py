import numpy as np

from surrogate_loop.operator.heat1d.dataset import (
    HeatDataset,
    NormalizationStats,
    generate_dataset,
    split_dataset,
)


def test_sampling_is_deterministic_and_dataset_shapes_are_consistent(
    smoke_operator_spec,
) -> None:
    first = generate_dataset(smoke_operator_spec)
    second = generate_dataset(smoke_operator_spec)

    np.testing.assert_array_equal(first.parameters, second.parameters)
    np.testing.assert_array_equal(first.fields, second.fields)
    assert first.parameters.shape == (96, 3)
    assert first.fields.shape == (96, 51, 65)
    assert first.solver_relative_l2.shape == (96,)
    assert np.quantile(first.solver_relative_l2, 0.95) <= 0.005


def test_split_is_by_complete_case_without_leakage(smoke_operator_spec) -> None:
    split = split_dataset(generate_dataset(smoke_operator_spec), smoke_operator_spec.sampling)

    train = {tuple(row) for row in split.train.parameters}
    validation = {tuple(row) for row in split.validation.parameters}
    test = {tuple(row) for row in split.test.parameters}

    assert len(train) == 64
    assert len(validation) == 16
    assert len(test) == 16
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_normalization_uses_training_data_and_round_trips_targets(
    smoke_operator_spec,
) -> None:
    split = split_dataset(generate_dataset(smoke_operator_spec), smoke_operator_spec.sampling)
    stats = NormalizationStats.fit(split.train)

    np.testing.assert_allclose(stats.parameter_mean, split.train.parameters.mean(axis=0))
    assert not np.allclose(stats.parameter_mean, split.validation.parameters.mean(axis=0))
    normalized_parameters = stats.normalize_parameters(split.train.parameters)
    np.testing.assert_allclose(normalized_parameters.mean(axis=0), 0.0, atol=1e-12)
    normalized_targets = stats.normalize_targets(split.train.fields)
    np.testing.assert_allclose(normalized_targets.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(normalized_targets.std(), 1.0, atol=1e-12)
    np.testing.assert_allclose(stats.denormalize_targets(normalized_targets), split.train.fields)


def test_coordinate_normalization_uses_fixed_training_grid(smoke_operator_spec) -> None:
    split = split_dataset(generate_dataset(smoke_operator_spec), smoke_operator_spec.sampling)
    stats = NormalizationStats.fit(split.train)
    flat_coordinates = np.stack(
        np.meshgrid(split.train.x, split.train.t, indexing="xy"), axis=-1
    ).reshape(-1, 2)

    normalized = stats.normalize_coordinates(flat_coordinates)

    assert flat_coordinates.shape == (51 * 65, 2)
    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-12)


def test_dataset_rejects_inconsistent_shapes() -> None:
    with np.testing.assert_raises_regex(ValueError, "字段形状"):
        HeatDataset(
            parameters=np.zeros((2, 3)),
            x=np.linspace(0.0, 1.0, 5),
            t=np.linspace(0.0, 1.0, 3),
            fields=np.zeros((2, 3, 4)),
            solver_relative_l2=np.zeros(2),
        )
