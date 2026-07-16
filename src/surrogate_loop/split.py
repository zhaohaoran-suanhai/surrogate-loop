from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from surrogate_loop.config import SamplingSpec
from surrogate_loop.data import CaseDataset


@dataclass(frozen=True)
class DatasetSplit:
    train_x: NDArray[np.float64]
    train_y: NDArray[np.float64]
    validation_x: NDArray[np.float64]
    validation_y: NDArray[np.float64]
    test_x: NDArray[np.float64]
    test_y: NDArray[np.float64]


def split_dataset(dataset: CaseDataset, sampling: SamplingSpec) -> DatasetSplit:
    train_end = sampling.train_cases
    validation_end = train_end + sampling.validation_cases
    expected = validation_end + sampling.test_cases
    if dataset.gamma.size != expected or dataset.target.size != expected:
        raise ValueError("数据集工况数与配置不一致")

    x = dataset.gamma.reshape(-1, 1)
    return DatasetSplit(
        train_x=x[:train_end],
        train_y=dataset.target[:train_end],
        validation_x=x[train_end:validation_end],
        validation_y=dataset.target[train_end:validation_end],
        test_x=x[validation_end:],
        test_y=dataset.target[validation_end:],
    )
