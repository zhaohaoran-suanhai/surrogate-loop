from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from surrogate_loop.config import RunSpec
from surrogate_loop.domain import analytical_solution, numerical_endpoint


@dataclass(frozen=True)
class CaseDataset:
    gamma: NDArray[np.float64]
    target: NDArray[np.float64]


def _sample_unique(spec: RunSpec, count: int) -> NDArray[np.float64]:
    generator = np.random.default_rng(spec.sampling.seed)
    for _ in range(10):
        values = generator.uniform(spec.problem.gamma.low, spec.problem.gamma.high, count)
        if np.unique(values).size == count:
            return values.astype(np.float64)
    raise RuntimeError("无法生成互不重复的 gamma 工况")


def generate_dataset(spec: RunSpec) -> CaseDataset:
    sampling = spec.sampling
    count = sampling.train_cases + sampling.validation_cases + sampling.test_cases
    gamma = _sample_unique(spec, count)
    target = np.empty(count, dtype=np.float64)
    for index, value in enumerate(gamma):
        numerical = numerical_endpoint(float(value), spec.problem.time_end)
        analytical = analytical_solution(float(value), spec.problem.time_end)
        if not np.isclose(numerical, analytical, rtol=1e-9, atol=1e-9):
            raise RuntimeError(f"gamma={value} 的数值标签未通过解析解校验")
        target[index] = numerical
    return CaseDataset(gamma=gamma, target=target)
