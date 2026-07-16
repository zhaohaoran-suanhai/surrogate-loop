import numpy as np

from surrogate_loop.config import load_spec
from surrogate_loop.data import generate_dataset
from surrogate_loop.split import split_dataset


def test_case_split_has_expected_shapes_and_no_overlap(smoke_spec_path) -> None:
    spec = load_spec(smoke_spec_path)
    split = split_dataset(generate_dataset(spec), spec.sampling)

    assert split.train_x.shape == (24, 1)
    assert split.validation_x.shape == (8, 1)
    assert split.test_x.shape == (8, 1)
    train = set(np.ravel(split.train_x))
    validation = set(np.ravel(split.validation_x))
    test = set(np.ravel(split.test_x))
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
