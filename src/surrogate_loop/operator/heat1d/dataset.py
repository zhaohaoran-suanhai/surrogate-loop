from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from surrogate_loop.operator.config import OperatorRunSpec, SamplingSpec
from surrogate_loop.operator.heat1d.analytical import analytical_solution
from surrogate_loop.operator.heat1d.problem import HeatParameters, make_grid
from surrogate_loop.operator.heat1d.solver import solve_case


@dataclass(frozen=True)
class HeatDataset:
    parameters: NDArray[np.float64]
    x: NDArray[np.float64]
    t: NDArray[np.float64]
    fields: NDArray[np.float64]
    solver_relative_l2: NDArray[np.float64]

    def __post_init__(self) -> None:
        parameters = np.asarray(self.parameters, dtype=np.float64)
        x = np.asarray(self.x, dtype=np.float64)
        t = np.asarray(self.t, dtype=np.float64)
        fields = np.asarray(self.fields, dtype=np.float64)
        solver_relative_l2 = np.asarray(self.solver_relative_l2, dtype=np.float64)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "solver_relative_l2", solver_relative_l2)
        if parameters.ndim != 2 or parameters.shape[1] != 3:
            raise ValueError("参数形状必须为 (n_cases, 3)")
        expected_field_shape = (parameters.shape[0], t.size, x.size)
        if fields.shape != expected_field_shape:
            raise ValueError(f"字段形状必须为 {expected_field_shape}")
        if solver_relative_l2.shape != (parameters.shape[0],):
            raise ValueError("求解器误差形状必须为 (n_cases,)")
        arrays = (parameters, x, t, fields, solver_relative_l2)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("数据集数组必须全部有限")

    def subset(self, indices: NDArray[np.int64]) -> HeatDataset:
        selected = np.asarray(indices, dtype=np.int64)
        return HeatDataset(
            parameters=self.parameters[selected].copy(),
            x=self.x.copy(),
            t=self.t.copy(),
            fields=self.fields[selected].copy(),
            solver_relative_l2=self.solver_relative_l2[selected].copy(),
        )


@dataclass(frozen=True)
class HeatDatasetSplit:
    train: HeatDataset
    validation: HeatDataset
    test: HeatDataset


@dataclass(frozen=True)
class NormalizationStats:
    parameter_mean: NDArray[np.float64]
    parameter_std: NDArray[np.float64]
    coordinate_mean: NDArray[np.float64]
    coordinate_std: NDArray[np.float64]
    target_mean: float
    target_std: float

    @classmethod
    def fit(cls, train: HeatDataset) -> NormalizationStats:
        coordinates = _coordinate_grid(train.x, train.t)
        return cls(
            parameter_mean=train.parameters.mean(axis=0),
            parameter_std=_safe_std(train.parameters, axis=0),
            coordinate_mean=coordinates.mean(axis=0),
            coordinate_std=_safe_std(coordinates, axis=0),
            target_mean=float(train.fields.mean()),
            target_std=float(_safe_std(train.fields, axis=None)),
        )

    def normalize_parameters(
        self, parameters: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return (np.asarray(parameters, dtype=np.float64) - self.parameter_mean) / self.parameter_std

    def normalize_coordinates(
        self, coordinates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        normalized = np.asarray(coordinates, dtype=np.float64) - self.coordinate_mean
        return normalized / self.coordinate_std

    def normalize_targets(self, targets: NDArray[np.float64]) -> NDArray[np.float64]:
        return (np.asarray(targets, dtype=np.float64) - self.target_mean) / self.target_std

    def denormalize_targets(self, targets: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(targets, dtype=np.float64) * self.target_std + self.target_mean


def _safe_std(array: NDArray[np.float64], axis: int | None) -> NDArray[np.float64]:
    value = np.asarray(array, dtype=np.float64).std(axis=axis)
    return np.where(value < 1e-12, 1.0, value)


def _coordinate_grid(
    x: NDArray[np.float64], t: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.stack(np.meshgrid(x, t, indexing="xy"), axis=-1).reshape(-1, 2)


def generate_dataset(spec: OperatorRunSpec) -> HeatDataset:
    total_cases = (
        spec.sampling.train_cases
        + spec.sampling.validation_cases
        + spec.sampling.test_cases
    )
    unit_samples = qmc.LatinHypercube(d=3, seed=spec.sampling.seed).random(total_cases)
    lower = np.array(
        [
            spec.problem.alpha.low,
            spec.problem.amplitude_1.low,
            spec.problem.amplitude_2.low,
        ]
    )
    upper = np.array(
        [
            spec.problem.alpha.high,
            spec.problem.amplitude_1.high,
            spec.problem.amplitude_2.high,
        ]
    )
    parameter_samples = qmc.scale(unit_samples, lower, upper)
    x, t = make_grid(spec.grid.nx, spec.grid.nt)
    fields = np.empty((total_cases, t.size, x.size), dtype=np.float64)
    relative_l2 = np.empty(total_cases, dtype=np.float64)
    for index, values in enumerate(parameter_samples):
        parameters = HeatParameters(
            alpha=float(values[0]),
            amplitude_1=float(values[1]),
            amplitude_2=float(values[2]),
        )
        numerical = solve_case(parameters, x, t)
        reference = analytical_solution(parameters, x, t)
        fields[index] = numerical
        relative_l2[index] = np.linalg.norm(numerical - reference) / np.linalg.norm(reference)

    boundary_error = float(np.max(np.abs(fields[:, :, [0, -1]])))
    p95_relative_l2 = float(np.quantile(relative_l2, 0.95))
    if boundary_error > spec.solver_acceptance.max_boundary_error:
        raise RuntimeError(
            f"数值求解器边界误差 {boundary_error:.6g} 超过阈值 "
            f"{spec.solver_acceptance.max_boundary_error:.6g}"
        )
    if p95_relative_l2 > spec.solver_acceptance.max_p95_relative_l2:
        raise RuntimeError(
            f"数值求解器解析解 p95 相对 L2 误差 {p95_relative_l2:.6g} 超过阈值 "
            f"{spec.solver_acceptance.max_p95_relative_l2:.6g}"
        )
    return HeatDataset(parameter_samples, x, t, fields, relative_l2)


def split_dataset(dataset: HeatDataset, sampling: SamplingSpec) -> HeatDatasetSplit:
    expected_cases = sampling.train_cases + sampling.validation_cases + sampling.test_cases
    if dataset.parameters.shape[0] != expected_cases:
        raise ValueError(f"数据集必须包含 {expected_cases} 个算例")
    indices = np.random.default_rng(sampling.seed).permutation(expected_cases)
    train_end = sampling.train_cases
    validation_end = train_end + sampling.validation_cases
    return HeatDatasetSplit(
        train=dataset.subset(indices[:train_end]),
        validation=dataset.subset(indices[train_end:validation_end]),
        test=dataset.subset(indices[validation_end:]),
    )
