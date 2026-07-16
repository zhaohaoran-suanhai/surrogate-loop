import numpy as np

from surrogate_loop.config import load_spec
from surrogate_loop.data import generate_dataset


def test_dataset_generation_is_reproducible(smoke_spec_path) -> None:
    spec = load_spec(smoke_spec_path)

    first = generate_dataset(spec)
    second = generate_dataset(spec)

    np.testing.assert_array_equal(first.gamma, second.gamma)
    np.testing.assert_allclose(first.target, second.target, rtol=0, atol=0)
    assert first.gamma.shape == (40,)
    assert first.target.shape == (40,)
