from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ParameterRange(StrictModel):
    low: float
    high: float

    @model_validator(mode="after")
    def validate_bounds(self) -> ParameterRange:
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError("参数范围必须有限")
        if self.low >= self.high:
            raise ValueError("参数下界必须小于上界")
        return self


class HeatProblemSpec(StrictModel):
    template: Literal["heat_1d_operator_v1"]
    alpha: ParameterRange
    amplitude_1: ParameterRange
    amplitude_2: ParameterRange
    x_start: Literal[0.0]
    x_end: Literal[1.0]
    t_start: Literal[0.0]
    t_end: Literal[1.0]

    @model_validator(mode="after")
    def validate_canonical_problem(self) -> HeatProblemSpec:
        expected = {
            "alpha": (0.05, 0.2),
            "amplitude_1": (0.8, 1.2),
            "amplitude_2": (-0.3, 0.3),
        }
        for name, bounds in expected.items():
            value = getattr(self, name)
            if (value.low, value.high) != bounds:
                raise ValueError(f"{name} 范围必须为 {bounds}")
        return self


class GridSpec(StrictModel):
    nx: int = Field(ge=3)
    nt: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_odd_grid(self) -> GridSpec:
        if self.nx % 2 == 0 or self.nt % 2 == 0:
            raise ValueError("空间和时间网格点数必须为奇数")
        return self


class SamplingSpec(StrictModel):
    seed: Literal[20260716]
    train_cases: int = Field(gt=0)
    validation_cases: int = Field(gt=0)
    test_cases: int = Field(gt=0)


class SolverAcceptanceSpec(StrictModel):
    max_boundary_error: float = Field(gt=0)
    max_p95_relative_l2: float = Field(gt=0)


class PodSpec(StrictModel):
    energy_threshold: float = Field(gt=0, le=1)
    max_components: int = Field(gt=0, le=32)


class DeepONetSpec(StrictModel):
    hidden_width: int = Field(gt=0)
    hidden_layers: int = Field(gt=0)
    latent_dim: int = Field(gt=0)


class TrainingSpec(StrictModel):
    max_epochs: int = Field(gt=0)
    patience: int = Field(gt=0)
    case_batch_size: int = Field(gt=0)
    query_batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    lr_factor: float = Field(gt=0, lt=1)
    min_learning_rate: float = Field(gt=0)
    min_delta: float = Field(ge=0)
    max_minutes: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_training_relationships(self) -> TrainingSpec:
        if self.patience > self.max_epochs:
            raise ValueError("patience 不能超过 max_epochs")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate 不能超过 learning_rate")
        return self


class OperatorAcceptanceSpec(StrictModel):
    max_median_relative_l2: float = Field(gt=0)
    max_p95_relative_l2: float = Field(gt=0)
    max_worst_relative_l2: float = Field(gt=0)
    max_initial_relative_l2: float = Field(gt=0)
    max_boundary_absolute_error: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_quantile_order(self) -> OperatorAcceptanceSpec:
        if not (
            self.max_median_relative_l2
            <= self.max_p95_relative_l2
            <= self.max_worst_relative_l2
        ):
            raise ValueError("median、p95、worst 验收阈值必须依次非递减")
        return self


class RuntimeSpec(StrictModel):
    device: Literal["auto", "cpu", "cuda"]
    dtype: Literal["float32"]


class OperatorRunSpec(StrictModel):
    mode: Literal["smoke", "full"]
    problem: HeatProblemSpec
    grid: GridSpec
    sampling: SamplingSpec
    solver_acceptance: SolverAcceptanceSpec
    pod: PodSpec
    model: DeepONetSpec
    training: TrainingSpec
    acceptance: OperatorAcceptanceSpec
    runtime: RuntimeSpec

    @model_validator(mode="after")
    def validate_scale(self) -> OperatorRunSpec:
        cases = (
            self.sampling.train_cases,
            self.sampling.validation_cases,
            self.sampling.test_cases,
        )
        if self.mode == "full":
            if (self.grid.nx, self.grid.nt) != (129, 101):
                raise ValueError("full 网格必须为 (129, 101)")
            if cases != (512, 96, 128):
                raise ValueError("full 工况数必须为 (512, 96, 128)")
            expected_training = (600, 60, 32, 512, 60.0)
            actual_training = (
                self.training.max_epochs,
                self.training.patience,
                self.training.case_batch_size,
                self.training.query_batch_size,
                self.training.max_minutes,
            )
            if actual_training != expected_training:
                raise ValueError(f"full 训练预算必须为 {expected_training}")
        else:
            if self.grid.nx > 65 or self.grid.nt > 51:
                raise ValueError("smoke 网格不能超过 (65, 51)")
            limits = (64, 16, 16)
            if any(actual > limit for actual, limit in zip(cases, limits, strict=True)):
                raise ValueError(f"smoke 工况数不能超过 {limits}")
            if self.training.max_epochs > 1500 or self.training.max_minutes > 10:
                raise ValueError("smoke 训练预算不能超过 1500 epoch 和 10 分钟")
        if self.training.case_batch_size > self.sampling.train_cases:
            raise ValueError("case_batch_size 不能超过训练工况数")
        if self.training.query_batch_size > self.grid.nx * self.grid.nt:
            raise ValueError("query_batch_size 不能超过时空网格点数")
        return self


def load_operator_spec(path: Path) -> OperatorRunSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return OperatorRunSpec.model_validate(payload)
